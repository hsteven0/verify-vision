from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock, Thread
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from packaging.version import InvalidVersion, Version

from app.dependencies.models import DependencyUpdate, DependencyUpdateStatus, UpdateRisk

logger = logging.getLogger("uvicorn.error")

PYTHON_PACKAGES: tuple[tuple[str, UpdateRisk], ...] = (
    ("torch", "critical_runtime"),
    ("torchvision", "critical_runtime"),
    ("transformers", "critical_runtime"),
    ("ultralytics", "critical_runtime"),
    ("peft", "critical_runtime"),
    ("numpy", "critical_runtime"),
    ("opencv-python-headless", "critical_runtime"),
    ("lmdb", "critical_runtime"),
    ("fastapi", "application"),
    ("pillow", "application"),
    ("pydantic", "application"),
    ("python-multipart", "application"),
    ("uvicorn", "application"),
)


class DependencyUpdateService:
    """Checks registries at set intervals without installing packages."""

    def __init__(self, cache_path: Path, *, interval_hours: int = 24) -> None:
        self.cache_path = cache_path
        self.interval = timedelta(hours=interval_hours)
        self._lock = Lock()
        self._checking = False
        self._repository_root = Path(__file__).resolve().parents[3]

    def status(self) -> DependencyUpdateStatus:
        with self._lock:
            if self._checking:
                cached = self._read_cache()
                return cached.model_copy(update={"state": "checking", "detail": None})
            return self._read_cache()

    def check_if_due_in_background(self) -> bool:
        if not self.is_due():
            return False
        with self._lock:
            if self._checking:
                return False
            self._checking = True
        Thread(target=self._background_check, name="dependency-update-check", daemon=True).start()
        return True

    def check_now(self) -> DependencyUpdateStatus:
        with self._lock:
            if self._checking:
                return self._read_cache().model_copy(update={"state": "checking", "detail": None})
            self._checking = True
        try:
            return self._perform_check()
        finally:
            with self._lock:
                self._checking = False

    def is_due(self, *, now: datetime | None = None) -> bool:
        current = now or datetime.now(UTC)
        cached = self._read_cache()
        return cached.last_checked_at is None or current >= (
            cached.next_check_at or cached.last_checked_at + self.interval
        )

    def _background_check(self) -> None:
        try:
            self._perform_check()
        except Exception:  # pragma: no cover - defensive background boundary
            logger.debug("Dependency update check failed", exc_info=True)
        finally:
            with self._lock:
                self._checking = False

    def _perform_check(self) -> DependencyUpdateStatus:
        installed = self._installed_dependencies()
        updates: list[DependencyUpdate] = []
        successful = 0
        with ThreadPoolExecutor(max_workers=6, thread_name_prefix="dependency-registry") as pool:
            futures = {
                pool.submit(self._latest_version, ecosystem, package): (
                    ecosystem,
                    package,
                    current,
                    risk,
                )
                for ecosystem, package, current, risk in installed
            }
            for future in as_completed(futures):
                ecosystem, package, current, risk = futures[future]
                try:
                    latest, release_url = future.result()
                except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError):
                    continue
                successful += 1
                if _is_newer(latest, current):
                    updates.append(
                        DependencyUpdate(
                            package=package,
                            ecosystem=ecosystem,
                            risk=risk,
                            current_version=current,
                            latest_version=latest,
                            release_url=release_url,
                        )
                    )

        checked_at = datetime.now(UTC)
        if successful == 0:
            result = DependencyUpdateStatus(
                state="unavailable",
                last_checked_at=checked_at,
                next_check_at=checked_at + self.interval,
                detail="Package registries are unavailable. VerifyVision still works offline.",
            )
        else:
            updates.sort(key=lambda item: (item.risk, item.ecosystem, item.package.casefold()))
            result = DependencyUpdateStatus(
                state="updates_available" if updates else "up_to_date",
                last_checked_at=checked_at,
                next_check_at=checked_at + self.interval,
                packages_checked=successful,
                updates=updates,
                detail=("Updates are available. Nothing was installed." if updates else None),
            )
        self._write_cache(result)
        return result

    def _installed_dependencies(self) -> list[tuple[str, str, str, UpdateRisk]]:
        installed: list[tuple[str, str, str, UpdateRisk]] = []
        for package, risk in PYTHON_PACKAGES:
            try:
                current = version(package)
            except PackageNotFoundError:
                continue
            installed.append(("python", package, current, risk))

        lock_path = self._repository_root / "frontend" / "package-lock.json"
        package_path = self._repository_root / "frontend" / "package.json"
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            manifest = json.loads(package_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return installed
        requested = set(manifest.get("dependencies", {})) | set(manifest.get("devDependencies", {}))
        packages = lock.get("packages", {})
        for package in sorted(requested):
            node = packages.get(f"node_modules/{package}", {})
            current = node.get("version")
            if isinstance(current, str):
                installed.append(("frontend", package, current, "application"))
        return installed

    @staticmethod
    def _latest_version(ecosystem: str, package: str) -> tuple[str, str]:
        if ecosystem == "python":
            url = f"https://pypi.org/pypi/{quote(package, safe='')}/json"
            payload = _read_json(url)
            return str(payload["info"]["version"]), str(payload["info"]["package_url"])
        encoded = quote(package, safe="")
        url = f"https://registry.npmjs.org/{encoded}/latest"
        payload = _read_json(url)
        return str(payload["version"]), f"https://www.npmjs.com/package/{package}"

    def _read_cache(self) -> DependencyUpdateStatus:
        try:
            return DependencyUpdateStatus.model_validate_json(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return DependencyUpdateStatus()

    def _write_cache(self, status: DependencyUpdateStatus) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            temporary.write_text(status.model_dump_json(indent=2), encoding="utf-8")
            temporary.replace(self.cache_path)
        except OSError:
            logger.debug("Could not persist dependency update status", exc_info=True)


def _read_json(url: str) -> dict[str, object]:
    request = Request(url, headers={"User-Agent": "VerifyVision dependency checker/0.1"})
    with urlopen(request, timeout=3) as response:
        return json.loads(response.read(1_000_000).decode("utf-8"))


def _is_newer(candidate: str, current: str) -> bool:
    try:
        return Version(candidate) > Version(current)
    except InvalidVersion:
        return candidate != current
