from pathlib import Path

import pytest

from app.config import AppSettings


def test_local_configuration_has_explicit_safe_defaults(tmp_path: Path) -> None:
    default_data = tmp_path / "backend" / "data" / "projects"
    settings = AppSettings.from_environment({}, default_data_dir=default_data)

    assert settings.data_dir == default_data
    assert settings.allowed_origins == ("http://localhost:5173", "http://127.0.0.1:5173")
    assert settings.dependency_update_cache == tmp_path / ".cache" / "dependency-update-status.json"
    assert settings.dependency_update_check_enabled is True
    assert settings.dependency_update_interval_hours == 24


def test_configuration_accepts_local_data_and_update_overrides(tmp_path: Path) -> None:
    settings = AppSettings.from_environment(
        {
            "VERIFYVISION_DATA_DIR": str(tmp_path / "projects"),
            "VERIFYVISION_DEPENDENCY_UPDATE_CACHE": str(tmp_path / "updates.json"),
            "VERIFYVISION_DEPENDENCY_UPDATE_CHECK": "false",
            "VERIFYVISION_DEPENDENCY_UPDATE_INTERVAL_HOURS": "48",
        },
        default_data_dir=tmp_path / "default",
    )

    assert settings.data_dir == tmp_path / "projects"
    assert settings.dependency_update_cache == tmp_path / "updates.json"
    assert settings.dependency_update_check_enabled is False
    assert settings.dependency_update_interval_hours == 48


def test_configuration_rejects_wildcard_cors_and_invalid_check_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit origins"):
        AppSettings.from_environment({"VERIFYVISION_ALLOWED_ORIGINS": "*"}, default_data_dir=tmp_path)
    with pytest.raises(ValueError, match="true or false"):
        AppSettings.from_environment(
            {"VERIFYVISION_DEPENDENCY_UPDATE_CHECK": "sometimes"},
            default_data_dir=tmp_path,
        )
