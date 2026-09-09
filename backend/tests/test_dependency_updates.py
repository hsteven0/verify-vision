from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import URLError

from app.dependencies.models import DependencyUpdateStatus
from app.dependencies.service import DependencyUpdateService


def test_dependency_check_reports_updates_without_installing(tmp_path: Path, monkeypatch) -> None:
    service = DependencyUpdateService(tmp_path / "updates.json")
    monkeypatch.setattr(
        service,
        "_installed_dependencies",
        lambda: [
            ("python", "torch", "2.12.1+cu126", "critical_runtime"),
            ("frontend", "react", "19.2.8", "application"),
        ],
    )

    def latest(ecosystem: str, package: str) -> tuple[str, str]:
        if package == "torch":
            return "2.13.0", "https://pypi.org/project/torch/"
        return "19.2.8", "https://www.npmjs.com/package/react"

    monkeypatch.setattr(service, "_latest_version", latest)

    result = service.check_now()

    assert result.state == "updates_available"
    assert result.packages_checked == 2
    assert [(item.package, item.risk) for item in result.updates] == [("torch", "critical_runtime")]
    assert service.cache_path.is_file()


def test_dependency_check_is_nonfatal_when_registries_are_offline(tmp_path: Path, monkeypatch) -> None:
    service = DependencyUpdateService(tmp_path / "updates.json")
    monkeypatch.setattr(
        service,
        "_installed_dependencies",
        lambda: [("python", "fastapi", "0.141.1", "application")],
    )
    monkeypatch.setattr(
        service,
        "_latest_version",
        lambda ecosystem, package: (_ for _ in ()).throw(URLError("offline")),
    )

    result = service.check_now()

    assert result.state == "unavailable"
    assert "works offline" in (result.detail or "")


def test_cached_check_observes_interval(tmp_path: Path) -> None:
    checked = datetime.now(UTC)
    cache = tmp_path / "updates.json"
    cache.write_text(
        DependencyUpdateStatus(
            state="up_to_date",
            last_checked_at=checked,
            next_check_at=checked + timedelta(hours=24),
            packages_checked=4,
        ).model_dump_json(),
        encoding="utf-8",
    )
    service = DependencyUpdateService(cache)

    assert service.is_due(now=checked + timedelta(hours=1)) is False
    assert service.is_due(now=checked + timedelta(hours=25)) is True


def test_corrupt_update_cache_degrades_to_not_checked(tmp_path: Path) -> None:
    cache = tmp_path / "updates.json"
    cache.write_text("not-json", encoding="utf-8")

    assert DependencyUpdateService(cache).status().state == "not_checked"
