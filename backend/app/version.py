from pathlib import Path


def _read_application_version() -> str:
    version = Path(__file__).with_name("VERSION").read_text(encoding="utf-8").strip()
    if not version:
        raise RuntimeError("VerifyVision application version is empty")
    return version


APP_VERSION = _read_application_version()
