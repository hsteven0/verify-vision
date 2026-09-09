from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "model-cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "model-cache" / "torch"))
sys.path.insert(0, str(ROOT / "backend"))

from app.inference.factory import DEFAULT_LOCATEANYTHING_REVISION
from app.inference.locateanything import (
    LocateAnythingVisionLocalizationProvider,
)
from app.inference.provider import LocalizationRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile repeated real LocateAnything-3B prompts against one image."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "prompts", nargs="+", help="Prompts to run in order on one resident model"
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Provider dtype policy to profile (default: auto)",
    )
    return parser.parse_args()


async def profile(
    image_path: Path, prompts: list[str], dtype: str = "auto"
) -> list[dict[str, object]]:
    with Image.open(image_path) as image:
        width, height = image.size

    provider = LocateAnythingVisionLocalizationProvider(
        worker_path=ROOT
        / ".vendor"
        / "Eagle"
        / "Embodied"
        / "locateanything_worker.py",
        model_revision=os.environ.get(
            "VERIFYVISION_LOCATEANYTHING_REVISION", DEFAULT_LOCATEANYTHING_REVISION
        ),
        dtype=dtype,
    )
    diagnostics = provider.runtime_diagnostics
    print(
        json.dumps(
            {
                "gpu": diagnostics.gpu_name,
                "cuda_capability": diagnostics.compute_capability,
                "requested_dtype": diagnostics.requested_dtype,
                "selected_dtype": diagnostics.selected_dtype,
                "dtype_reason": diagnostics.dtype_reason,
            }
        ),
        flush=True,
    )
    results: list[dict[str, object]] = []
    for index, prompt in enumerate(prompts):
        started = perf_counter()
        predictions = await provider.localize(
            LocalizationRequest(
                image_key=image_path.name,
                image_path=image_path,
                image_width=width,
                image_height=height,
                prompt=prompt,
                label_name=prompt.title(),
            )
        )
        result: dict[str, object] = {
            "request": index + 1,
            "prompt": prompt,
            "detections": len(predictions),
            "wall_ms": round((perf_counter() - started) * 1000, 2),
            "runtime_status": provider.runtime_status,
            "device": provider.device,
            "requested_dtype": dtype,
            "selected_dtype": diagnostics.selected_dtype,
            "boxes": [prediction.box.model_dump() for prediction in predictions],
        }
        timing = getattr(provider, "last_timing", None)
        if timing is not None:
            result["stages_ms"] = timing.as_dict()
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise SystemExit(f"Image does not exist: {image_path}")
    asyncio.run(profile(image_path, args.prompts, args.dtype))


if __name__ == "__main__":
    main()
