from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

LOCAL_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Local-only application configuration."""

    data_dir: Path
    allowed_origins: tuple[str, ...]
    dependency_update_cache: Path
    dependency_update_check_enabled: bool
    dependency_update_interval_hours: int

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        default_data_dir: Path,
    ) -> AppSettings:
        values = environment if environment is not None else os.environ
        raw_origins = values.get("VERIFYVISION_ALLOWED_ORIGINS", "")
        origins = tuple(origin.strip().rstrip("/") for origin in raw_origins.split(",") if origin.strip())
        if any(origin == "*" for origin in origins):
            raise ValueError("VERIFYVISION_ALLOWED_ORIGINS must list explicit origins, not '*'")
        if not origins:
            origins = LOCAL_ORIGINS

        data_dir = Path(values.get("VERIFYVISION_DATA_DIR", str(default_data_dir))).expanduser()
        update_cache = Path(
            values.get(
                "VERIFYVISION_DEPENDENCY_UPDATE_CACHE",
                str(default_data_dir.parents[2] / ".cache" / "dependency-update-status.json"),
            )
        ).expanduser()
        return cls(
            data_dir=data_dir,
            allowed_origins=origins,
            dependency_update_cache=update_cache,
            dependency_update_check_enabled=_boolean(values, "VERIFYVISION_DEPENDENCY_UPDATE_CHECK", default=True),
            dependency_update_interval_hours=_bounded_int(
                values,
                "VERIFYVISION_DEPENDENCY_UPDATE_INTERVAL_HOURS",
                default=24,
                minimum=1,
                maximum=168,
            ),
        )


def _boolean(values: Mapping[str, str], key: str, *, default: bool) -> bool:
    raw = values.get(key)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be true or false")


def _bounded_int(
    values: Mapping[str, str],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = values.get(key, str(default))
    try:
        parsed = int(raw)
    except ValueError as error:
        raise ValueError(f"{key} must be an integer") from error
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return parsed
