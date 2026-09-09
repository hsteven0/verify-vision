from pathlib import Path

import pytest

from app.inference.factory import (
    ProviderConfigurationError,
    create_localization_provider,
    validate_runtime_provider,
)
from app.inference.locateanything import LocateAnythingVisionLocalizationProvider
from app.main import create_app
from tests.fakes import MockVisionLocalizationProvider


def test_provider_factory_is_always_lazy_locateanything() -> None:
    provider = create_localization_provider({})

    assert isinstance(provider, LocateAnythingVisionLocalizationProvider)
    assert provider.model_name == "nvidia/LocateAnything-3B"
    assert provider.model_revision == "c32291ca5e996f5a7a485845b4f57a233936bba0"
    assert provider.runtime_status == "not_loaded"


def test_provider_factory_applies_locateanything_runtime_configuration() -> None:
    provider = create_localization_provider(
        {
            "VERIFYVISION_LOCATEANYTHING_DEVICE": "cuda:1",
            "VERIFYVISION_LOCATEANYTHING_DTYPE": "fp16",
        }
    )

    assert isinstance(provider, LocateAnythingVisionLocalizationProvider)
    assert provider.device == "cuda:1"
    assert provider.runtime_diagnostics.requested_dtype == "float16"


def test_retired_provider_flags_cannot_select_fake_inference() -> None:
    provider = create_localization_provider({"VERIFYVISION_MODE": "demo", "VERIFYVISION_MODEL_PROVIDER": "mock"})

    assert isinstance(provider, LocateAnythingVisionLocalizationProvider)
    assert provider.name == "locateanything"


def test_provider_factory_rejects_invalid_locateanything_configuration() -> None:
    with pytest.raises(ProviderConfigurationError, match="integer"):
        create_localization_provider({"VERIFYVISION_LOCATEANYTHING_MAX_NEW_TOKENS": "many"})


def test_runtime_rejects_non_locateanything_dependency_injection(tmp_path: Path) -> None:
    mock = MockVisionLocalizationProvider()
    with pytest.raises(ProviderConfigurationError, match="requires the locateanything"):
        validate_runtime_provider(mock)
    with pytest.raises(ProviderConfigurationError, match="requires the locateanything"):
        create_app(
            data_dir=tmp_path,
            frontend_dist=tmp_path / "missing",
            localization_provider=mock,
        )


def test_local_scripts_expose_one_runtime_and_project_drive_caches() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    runtime = (repository_root / "scripts" / "manage-local.ps1").read_text(encoding="utf-8")
    setup = (repository_root / "scripts" / "setup-local.ps1").read_text(encoding="utf-8")

    assert "VERIFYVISION_MODE" not in runtime
    assert "VERIFYVISION_MODEL_PROVIDER" not in runtime
    assert "$env:HF_HOME" in setup
    assert "$env:TORCH_HOME" in setup
    assert "$npmCommand --prefix" in setup
    assert " ci" in setup
    assert "diagnose_locateanything.py" in setup
