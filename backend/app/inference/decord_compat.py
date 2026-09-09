from __future__ import annotations

import importlib.util
import sys
from types import ModuleType


def install_image_only_decord_stub() -> None:
    """Install the decord imports used for image inference on Windows.

    This shim cannot decode video.
    """

    if "decord" in sys.modules or importlib.util.find_spec("decord") is not None:
        return

    module = ModuleType("decord")
    module.__spec__ = importlib.util.spec_from_loader("decord", loader=None)

    class VideoReader:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("Video decoding is not available on Windows.")

    class Context:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    class Bridge:
        @staticmethod
        def set_bridge(_name: str) -> None:
            return None

    module.VideoReader = VideoReader
    module.cpu = Context
    module.gpu = Context
    module.bridge = Bridge
    sys.modules["decord"] = module
