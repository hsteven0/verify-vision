import json
import tomllib
from pathlib import Path

from app.main import create_app
from app.version import APP_VERSION


def test_release_version_is_synchronized_across_application_manifests() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    backend_manifest = tomllib.loads((repository_root / "backend" / "pyproject.toml").read_text(encoding="utf-8"))
    frontend_manifest = json.loads((repository_root / "frontend" / "package.json").read_text(encoding="utf-8"))
    frontend_lock = json.loads((repository_root / "frontend" / "package-lock.json").read_text(encoding="utf-8"))

    assert APP_VERSION == "1.0.0"
    assert backend_manifest["project"]["version"] == APP_VERSION
    assert frontend_manifest["version"] == APP_VERSION
    assert frontend_lock["version"] == APP_VERSION
    assert frontend_lock["packages"][""]["version"] == APP_VERSION
    assert create_app().version == APP_VERSION
