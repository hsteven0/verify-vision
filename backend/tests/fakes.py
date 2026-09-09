from __future__ import annotations

import hashlib
from uuid import NAMESPACE_URL, uuid5

from app.domain.models import BoundingBox
from app.inference.provider import LocalizationPrediction, LocalizationRequest


class MockVisionLocalizationProvider:
    """Deterministic test double used only in tests."""

    name = "mock"
    model_name = "deterministic-layout-v1"
    device = "cpu"
    runtime_status = "ready"

    async def localize(self, request: LocalizationRequest) -> list[LocalizationPrediction]:
        identity = (
            f"{self.name}:{self.model_name}:{request.image_key}:"
            f"{request.prompt.strip().casefold()}:{request.label_name.casefold()}"
        )
        digest = hashlib.sha256(identity.encode("utf-8")).digest()
        predictions: list[LocalizationPrediction] = []
        for index, (width_fraction, height_fraction) in enumerate(((0.32, 0.30), (0.24, 0.26), (0.19, 0.21))):
            width = max(1.0, request.image_width * width_fraction)
            height = max(1.0, request.image_height * height_fraction)
            available_x = max(0.0, request.image_width - width)
            available_y = max(0.0, request.image_height - height)
            predictions.append(
                LocalizationPrediction(
                    prediction_id=uuid5(NAMESPACE_URL, f"{identity}:{index}"),
                    box=BoundingBox(
                        x=available_x * (digest[index * 2] / 255),
                        y=available_y * (digest[index * 2 + 1] / 255),
                        width=width,
                        height=height,
                    ),
                    confidence=round(0.72 + (digest[12 + index] / 255) * 0.23, 3),
                    label_hint=request.label_name,
                )
            )
        return predictions
