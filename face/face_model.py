"""Wrapper around InsightFace buffalo_l. Loads detection + recognition only.

Public:
    FaceModel()                    # constructor loads model into GPU
    model.detect_and_embed(img)    # img: BGR ndarray -> list[Face]
    FaceModel.decode_image(bytes)  # raw JPEG/PNG bytes -> BGR ndarray
"""

import site
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np


class Face(TypedDict):
    bbox: list[float]              # [x1, y1, x2, y2]
    kps: list[list[float]] | None  # 5 keypoints
    embedding: np.ndarray          # (512,) float32, L2-normalized
    det_score: float


def _preload_nvidia_libs() -> None:
    """onnxruntime-gpu can't locate the pip-installed CUDA 12 libs on its own;
    dlopen them with RTLD_GLOBAL before importing onnxruntime."""
    import ctypes
    import glob

    for sp in site.getsitepackages():
        root = Path(sp) / "nvidia"
        if not root.exists():
            continue
        for so in glob.glob(str(root / "*/lib/*.so*")):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


class FaceModel:
    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0,
                 det_size: tuple[int, int] = (640, 640)):
        _preload_nvidia_libs()
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        print(f"[face_model] {model_name} ready (ctx={ctx_id}, det_size={det_size})")

    def detect_and_embed(self, img_bgr: np.ndarray) -> list[Face]:
        faces = self.app.get(img_bgr)
        return [
            Face(
                bbox=f.bbox.tolist(),
                kps=f.kps.tolist() if f.kps is not None else None,
                embedding=f.normed_embedding.astype(np.float32),
                det_score=float(f.det_score),
            )
            for f in faces
        ]

    @staticmethod
    def decode_image(image_bytes: bytes) -> np.ndarray | None:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
