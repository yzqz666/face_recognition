"""Register a single owner from a photo.

Public:
    register(photo, name, room=None, phone=None, *, model=None, storage=None) -> dict

Raises:
    DecodeError, NoFaceError, MultipleFacesError, FaceTooSmallError, DuplicateOwnerError
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .face_model import FaceModel
from .storage import Storage


MIN_FACE_SIZE = 80         # px, shorter side of bbox
DUPLICATE_SIMILARITY = 0.55  # reject if max cosine >= this against existing gallery


# ---------- exceptions ----------
class RegisterError(Exception):
    pass


class DecodeError(RegisterError):
    pass


class NoFaceError(RegisterError):
    pass


class MultipleFacesError(RegisterError):
    def __init__(self, n: int):
        super().__init__(f"multiple faces detected ({n})")
        self.n = n


class FaceTooSmallError(RegisterError):
    def __init__(self, short_side: int):
        super().__init__(f"face too small ({short_side}px < {MIN_FACE_SIZE}px)")
        self.short_side = short_side


class DuplicateOwnerError(RegisterError):
    def __init__(self, owner_id: int, similarity: float):
        super().__init__(
            f"possible duplicate of owner_id={owner_id} (sim={similarity:.3f})"
        )
        self.owner_id = owner_id
        self.similarity = similarity


# ---------- main API ----------
def register(
    photo: Union[str, Path, bytes],
    name: str,
    room: Optional[str] = None,
    phone: Optional[str] = None,
    *,
    model: Optional[FaceModel] = None,
    storage: Optional[Storage] = None,
) -> dict:
    """Run the full registration pipeline.

    `photo` accepts a file path (str/Path) or raw bytes (already-loaded JPEG/PNG).
    `model` and `storage` are optional; if omitted, fresh instances are created
    (slow on every call — pass them in if you call register() repeatedly).
    """
    photo_bytes = _read_photo(photo)

    if model is None:
        model = FaceModel()
    if storage is None:
        storage = Storage()

    img = FaceModel.decode_image(photo_bytes)
    if img is None:
        raise DecodeError("cannot decode image")

    faces = model.detect_and_embed(img)
    if len(faces) == 0:
        raise NoFaceError("no face detected")
    if len(faces) > 1:
        raise MultipleFacesError(len(faces))

    face = faces[0]
    x1, y1, x2, y2 = face["bbox"]
    short_side = int(min(x2 - x1, y2 - y1))
    if short_side < MIN_FACE_SIZE:
        raise FaceTooSmallError(short_side)

    embedding = face["embedding"]
    if len(storage) > 0:
        dup_id, dup_sim = storage.search_top1(embedding)
        if dup_sim is not None and dup_sim >= DUPLICATE_SIMILARITY:
            raise DuplicateOwnerError(dup_id, dup_sim)

    face_photo_bytes = _crop_face_jpeg(img, face["bbox"])
    meta = storage.save_owner(
        name,
        embedding,
        photo_bytes,
        face_photo_bytes=face_photo_bytes,
        room=room,
        phone=phone,
    )
    return {
        **meta,
        "face_bbox": face["bbox"],
        "det_score": face["det_score"],
        "gallery_size": len(storage),
    }


def _read_photo(photo: Union[str, Path, bytes]) -> bytes:
    if isinstance(photo, (str, Path)):
        return Path(photo).read_bytes()
    if isinstance(photo, (bytes, bytearray)):
        return bytes(photo)
    raise TypeError(f"photo must be path or bytes, got {type(photo).__name__}")


def _crop_face_jpeg(img_bgr: np.ndarray, bbox: list[float]) -> bytes:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(np.floor(x1))))
    y1 = max(0, min(h - 1, int(np.floor(y1))))
    x2 = max(x1 + 1, min(w, int(np.ceil(x2))))
    y2 = max(y1 + 1, min(h, int(np.ceil(y2))))

    crop = img_bgr[y1:y2, x1:x2]
    ok, encoded = cv2.imencode(".jpg", crop)
    if not ok:
        raise RegisterError("cannot encode face crop")
    return encoded.tobytes()
