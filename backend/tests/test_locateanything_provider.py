import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app.inference import decord_compat
from app.inference.locateanything import (
    LocateAnythingVisionLocalizationProvider,
    normalize_locateanything_boxes,
    select_locateanything_dtype,
)
from app.inference.provider import (
    LocalizationRequest,
    ProviderResponseError,
    ProviderUnavailableError,
)


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "NVIDIA GeForce RTX Test"

    @staticmethod
    def is_bf16_supported() -> bool:
        return True

    @staticmethod
    def get_device_capability(index: int) -> tuple[int, int]:
        assert index == 0
        return (8, 6)


FAKE_RUNTIME = SimpleNamespace(
    __version__="2.12.1+cu126",
    version=SimpleNamespace(cuda="12.6", hip=None),
    cuda=FakeCuda(),
    bfloat16=object(),
    float16=object(),
    float32=object(),
)


class UnavailableCuda:
    @staticmethod
    def is_available() -> bool:
        return False


CPU_ONLY_RUNTIME = SimpleNamespace(
    __version__="2.12.1+cu126",
    version=SimpleNamespace(cuda="12.6", hip=None),
    cuda=UnavailableCuda(),
    bfloat16=object(),
    float16=object(),
    float32=object(),
)


def write_test_image(path: Path, width: int = 400, height: int = 200) -> None:
    Image.new("RGB", (width, height), color=(20, 40, 60)).save(path)


def test_normalizes_multiple_official_coordinate_blocks() -> None:
    boxes = normalize_locateanything_boxes(
        "objects <box><100><200><600><800></box> and <box><0><0><1000><1000></box>",
        image_width=400,
        image_height=200,
    )

    assert boxes[0].model_dump() == {"x": 40.0, "y": 40.0, "width": 200.0, "height": 120.0}
    assert boxes[1].model_dump() == {"x": 0.0, "y": 0.0, "width": 400.0, "height": 200.0}


def test_normalization_supports_a_valid_zero_result_response() -> None:
    assert normalize_locateanything_boxes("No matching instances were found.", 640, 480) == []
    assert normalize_locateanything_boxes("<ref>person</ref><box>none</box>", 640, 480) == []


@pytest.mark.parametrize(
    "answer",
    [
        "<box><10><20><30></box>",
        "<box><20><20><10><40></box>",
        "<box><10><20><1001><40></box>",
        "<box><10><20></box>",
        "<box><10><20><30><40>",
        "<box><10><20><30><40></box><box>none</box>",
        "   ",
    ],
)
def test_normalization_rejects_malformed_or_unsafe_boxes(answer: str) -> None:
    with pytest.raises(ProviderResponseError):
        normalize_locateanything_boxes(answer, 640, 480)


def test_provider_reuses_worker_and_normalizes_boxes(
    tmp_path: Path,
) -> None:
    class FakeWorker:
        load_count = 0
        prompts: list[str] = []

        def __init__(self, model_name: str, *, device: str, dtype: object) -> None:
            assert model_name == "nvidia/LocateAnything-3B"
            assert device == "cuda:0"
            assert dtype is FAKE_RUNTIME.bfloat16
            FakeWorker.load_count += 1

        def ground_multi(self, image: Image.Image, phrase: str, **kwargs: object) -> dict:
            assert image.mode == "RGB"
            assert image.size == (400, 200)
            assert image.getpixel((0, 0)) == (20, 40, 60)
            assert kwargs["generation_mode"] == "hybrid"
            assert kwargs["max_new_tokens"] == 8192
            FakeWorker.prompts.append(phrase)
            return {"answer": "<box><100><100><300><500></box> <box><500><200><900><800></box>"}

    image_path = tmp_path / "scene.png"
    write_test_image(image_path)
    provider = LocateAnythingVisionLocalizationProvider(
        worker_class_loader=lambda: FakeWorker,
        runtime_loader=lambda: FAKE_RUNTIME,  # type: ignore[arg-type]
        platform_name="linux",
        python_version=(3, 12, 10),
    )
    request = LocalizationRequest(
        image_key="image-1",
        image_path=image_path,
        image_width=400,
        image_height=200,
        prompt="person  wearing a helmet",
        label_name="Person",
    )

    first = asyncio.run(provider.localize(request))
    first_timing = provider.last_timing
    second = asyncio.run(provider.localize(request))
    second_timing = provider.last_timing

    assert FakeWorker.load_count == 1
    assert FakeWorker.prompts == [request.prompt, request.prompt]
    assert first == second
    assert len(first) == 2
    assert len({prediction.prediction_id for prediction in first}) == 2
    assert all(prediction.confidence is None for prediction in first)
    assert all(prediction.label_hint == "Person" for prediction in first)
    assert all(prediction.box.fits_within(400, 200) for prediction in first)
    assert provider.runtime_status == "ready"
    assert provider.last_diagnostics is not None
    assert provider.last_diagnostics.image_key == "image-1"
    assert provider.last_diagnostics.prompt == request.prompt
    assert provider.last_diagnostics.raw_boxes == (
        (100, 100, 300, 500),
        (500, 200, 900, 800),
    )
    assert provider.last_diagnostics.normalized_boxes == tuple(item.box for item in first)
    assert first_timing is not None
    assert first_timing.model_loaded_this_request is True
    assert first_timing.image_decode_ms >= 0
    assert first_timing.gpu_inference_ms >= 0
    assert second_timing is not None
    assert second_timing.model_loaded_this_request is False


def test_provider_accepts_a_zero_detection_worker_response(tmp_path: Path) -> None:
    class EmptyWorker:
        def __init__(self, model_name: str, *, device: str, dtype: object) -> None:
            pass

        def ground_multi(self, image: Image.Image, phrase: str, **kwargs: object) -> dict:
            return {"answer": "No matching instances were found."}

    image_path = tmp_path / "scene.png"
    write_test_image(image_path)
    provider = LocateAnythingVisionLocalizationProvider(
        worker_class_loader=lambda: EmptyWorker,
        runtime_loader=lambda: FAKE_RUNTIME,  # type: ignore[arg-type]
        platform_name="linux",
        python_version=(3, 12, 10),
    )

    predictions = asyncio.run(
        provider.localize(
            LocalizationRequest(
                image_key="image-1",
                image_path=image_path,
                image_width=400,
                image_height=200,
                prompt="traffic light",
                label_name="Traffic light",
            )
        )
    )

    assert predictions == []
    assert provider.runtime_status == "ready"


def test_auto_device_reports_missing_cuda_without_loading_a_worker(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    write_test_image(image_path)
    provider = LocateAnythingVisionLocalizationProvider(
        worker_class_loader=lambda: pytest.fail("worker must not load without CUDA"),
        runtime_loader=lambda: CPU_ONLY_RUNTIME,  # type: ignore[arg-type]
        platform_name="linux",
        python_version=(3, 12, 10),
    )

    with pytest.raises(ProviderUnavailableError, match="CUDA is not available"):
        asyncio.run(
            provider.localize(
                LocalizationRequest(
                    image_key="image-1",
                    image_path=image_path,
                    image_width=400,
                    image_height=200,
                    prompt="person",
                    label_name="Person",
                )
            )
        )


def test_explicit_cpu_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="auto, cuda"):
        LocateAnythingVisionLocalizationProvider(device="cpu")


def test_auto_dtype_uses_native_float16_on_turing_cuda(tmp_path: Path) -> None:
    class TuringCuda(FakeCuda):
        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            assert index == 0
            return (7, 5)

        @staticmethod
        def is_bf16_supported() -> bool:
            return False

    turing_runtime = SimpleNamespace(
        __version__="2.12.1+cu126",
        version=SimpleNamespace(cuda="12.6", hip=None),
        cuda=TuringCuda(),
        bfloat16=object(),
        float16=object(),
        float32=object(),
    )

    class DtypeWorker:
        def __init__(self, model_name: str, *, device: str, dtype: object) -> None:
            assert dtype is turing_runtime.float16

        def ground_multi(self, image: Image.Image, phrase: str, **kwargs: object) -> dict:
            return {"answer": "<box><100><100><900><900></box>"}

    image_path = tmp_path / "scene.png"
    write_test_image(image_path)
    provider = LocateAnythingVisionLocalizationProvider(
        worker_class_loader=lambda: DtypeWorker,
        runtime_loader=lambda: turing_runtime,  # type: ignore[arg-type]
        platform_name="win32",
        python_version=(3, 12, 10),
    )

    predictions = asyncio.run(
        provider.localize(
            LocalizationRequest(
                image_key="image-1",
                image_path=image_path,
                image_width=400,
                image_height=200,
                prompt="person",
                label_name="Person",
            )
        )
    )

    assert len(predictions) == 1
    assert provider.runtime_diagnostics.compute_capability == "7.5"
    assert provider.runtime_diagnostics.requested_dtype == "auto"
    assert provider.runtime_diagnostics.selected_dtype == "float16"
    assert "pre-Ampere" in (provider.runtime_diagnostics.dtype_reason or "")


def test_auto_dtype_uses_bfloat16_on_bf16_capable_ampere_cuda() -> None:
    selection = select_locateanything_dtype(FakeCuda(), 0, "auto")

    assert selection.compute_capability == (8, 6)
    assert selection.selected == "bfloat16"
    assert "Ampere-or-newer" in selection.reason


@pytest.mark.parametrize("requested", ["float16", "float32"])
def test_explicit_dtype_override_is_preserved(requested: str) -> None:
    selection = select_locateanything_dtype(FakeCuda(), 0, requested)

    assert selection.requested == requested
    assert selection.selected == requested
    assert "explicit" in selection.reason


def test_explicit_bfloat16_override_is_preserved_when_supported() -> None:
    selection = select_locateanything_dtype(FakeCuda(), 0, "bfloat16")

    assert selection.selected == "bfloat16"
    assert "explicit BF16" in selection.reason


def test_explicit_bfloat16_override_requires_reported_support() -> None:
    class NoBfloatCuda(FakeCuda):
        @staticmethod
        def is_bf16_supported() -> bool:
            return False

    with pytest.raises(ProviderUnavailableError, match="does not report BF16 support"):
        select_locateanything_dtype(NoBfloatCuda(), 0, "bfloat16")


def test_unknown_capability_uses_conservative_float16_auto_policy() -> None:
    class UnknownCapabilityCuda(FakeCuda):
        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            raise RuntimeError("capability query unavailable")

    selection = select_locateanything_dtype(UnknownCapabilityCuda(), 0, "auto")

    assert selection.selected == "float16"
    assert selection.compute_capability is None
    assert "conservatively" in selection.reason


def test_invalid_dtype_override_fails_at_configuration() -> None:
    with pytest.raises(ValueError, match="auto, bfloat16, float16, or float32"):
        LocateAnythingVisionLocalizationProvider(dtype="float64")


def test_provider_rejects_macos_before_importing_torch(
    tmp_path: Path,
) -> None:
    runtime_was_loaded = False

    def load_runtime() -> object:
        nonlocal runtime_was_loaded
        runtime_was_loaded = True
        return FAKE_RUNTIME

    image_path = tmp_path / "scene.png"
    write_test_image(image_path)
    provider = LocateAnythingVisionLocalizationProvider(
        runtime_loader=load_runtime,  # type: ignore[arg-type]
        platform_name="darwin",
        python_version=(3, 12, 10),
    )
    request = LocalizationRequest(
        image_key="image-1",
        image_path=image_path,
        image_width=400,
        image_height=200,
        prompt="person",
        label_name="Person",
    )

    with pytest.raises(ProviderUnavailableError, match="Windows or Linux"):
        asyncio.run(provider.localize(request))

    assert runtime_was_loaded is False
    assert provider.runtime_status == "unavailable"


@pytest.mark.parametrize("platform_name", ["win32", "linux"])
def test_windows_and_linux_cuda_load_the_same_real_worker(tmp_path: Path, platform_name: str) -> None:
    class PlatformWorker:
        def __init__(self, model_name: str, *, device: str, dtype: object) -> None:
            assert model_name == "nvidia/LocateAnything-3B"
            assert device == "cuda:0"
            assert dtype is FAKE_RUNTIME.bfloat16

        def ground_multi(self, image: Image.Image, phrase: str, **kwargs: object) -> dict:
            return {"answer": "<box><100><200><900><800></box>"}

    image_path = tmp_path / "wide.png"
    write_test_image(image_path, width=900, height=300)
    provider = LocateAnythingVisionLocalizationProvider(
        worker_class_loader=lambda: PlatformWorker,
        runtime_loader=lambda: FAKE_RUNTIME,  # type: ignore[arg-type]
        platform_name=platform_name,
        python_version=(3, 12, 10),
    )

    predictions = asyncio.run(
        provider.localize(
            LocalizationRequest(
                image_key="wide-image",
                image_path=image_path,
                image_width=900,
                image_height=300,
                prompt="guitar",
                label_name="Guitar",
            )
        )
    )

    assert predictions[0].box.model_dump() == {
        "x": 90.0,
        "y": 60.0,
        "width": 720.0,
        "height": 180.0,
    }
    assert provider.runtime_diagnostics.available is True
    assert provider.runtime_diagnostics.gpu_name == "NVIDIA GeForce RTX Test"


def test_rocm_runtime_is_rejected_as_unsupported_amd() -> None:
    rocm_runtime = SimpleNamespace(
        __version__="2.12.1+rocm",
        version=SimpleNamespace(cuda=None, hip="7.2"),
        cuda=FakeCuda(),
        bfloat16=object(),
        float16=object(),
        float32=object(),
    )
    provider = LocateAnythingVisionLocalizationProvider(
        runtime_loader=lambda: rocm_runtime,  # type: ignore[arg-type]
        platform_name="win32",
        python_version=(3, 12, 10),
    )

    assert provider.runtime_diagnostics.available is False
    assert "ROCm" in (provider.runtime_diagnostics.detail or "")
    assert provider.runtime_status == "unavailable"


def test_python_314_is_reported_unsupported_without_importing_torch() -> None:
    runtime_was_loaded = False

    def load_runtime() -> object:
        nonlocal runtime_was_loaded
        runtime_was_loaded = True
        return FAKE_RUNTIME

    provider = LocateAnythingVisionLocalizationProvider(
        runtime_loader=load_runtime,  # type: ignore[arg-type]
        platform_name="win32",
        python_version=(3, 14, 1),
    )

    assert provider.runtime_diagnostics.available is False
    assert "Python 3.14.1" in (provider.runtime_diagnostics.detail or "")
    assert runtime_was_loaded is False


def test_cpu_only_pytorch_build_is_unavailable() -> None:
    cpu_build = SimpleNamespace(
        __version__="2.12.1+cpu",
        version=SimpleNamespace(cuda=None, hip=None),
        cuda=UnavailableCuda(),
        bfloat16=object(),
        float16=object(),
        float32=object(),
    )
    provider = LocateAnythingVisionLocalizationProvider(
        runtime_loader=lambda: cpu_build,  # type: ignore[arg-type]
        platform_name="win32",
        python_version=(3, 12, 10),
    )

    assert provider.runtime_diagnostics.available is False
    assert "no CUDA support" in (provider.runtime_diagnostics.detail or "")


def test_invalid_cuda_index_is_reported_before_model_loading() -> None:
    provider = LocateAnythingVisionLocalizationProvider(
        device="cuda:2",
        worker_class_loader=lambda: pytest.fail("worker must not load for invalid CUDA index"),
        runtime_loader=lambda: FAKE_RUNTIME,  # type: ignore[arg-type]
        platform_name="win32",
        python_version=(3, 12, 10),
    )

    assert provider.runtime_diagnostics.available is False
    assert "index 2" in (provider.runtime_diagnostics.detail or "")


def test_worker_failure_has_no_fake_fallback(
    tmp_path: Path,
) -> None:
    class BrokenWorker:
        def __init__(self, model_name: str, *, device: str, dtype: object) -> None:
            raise RuntimeError("simulated model load failure")

    image_path = tmp_path / "scene.png"
    write_test_image(image_path)
    provider = LocateAnythingVisionLocalizationProvider(
        worker_class_loader=lambda: BrokenWorker,
        runtime_loader=lambda: FAKE_RUNTIME,  # type: ignore[arg-type]
        platform_name="win32",
        python_version=(3, 12, 10),
    )

    with pytest.raises(ProviderUnavailableError, match="could not start"):
        asyncio.run(
            provider.localize(
                LocalizationRequest(
                    image_key="actual-upload",
                    image_path=image_path,
                    image_width=400,
                    image_height=200,
                    prompt="people",
                    label_name="Person",
                )
            )
        )

    assert provider.runtime_status == "error"


def test_windows_decord_shim_rejects_video(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = sys.modules.pop("decord", None)
    monkeypatch.setattr(decord_compat.importlib.util, "find_spec", lambda _name: None)
    try:
        decord_compat.install_image_only_decord_stub()
        stub = sys.modules["decord"]
        assert stub.__spec__ is not None
        with pytest.raises(RuntimeError, match="Video decoding is not available"):
            stub.VideoReader("video.mp4")
    finally:
        sys.modules.pop("decord", None)
        if existing is not None:
            sys.modules["decord"] = existing
