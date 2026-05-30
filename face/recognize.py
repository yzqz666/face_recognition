"""1:N face recognition against the registered gallery.

Public:
    recognize(photo, *, threshold=DEFAULT_THRESHOLD, model=None, storage=None) -> dict

Result dict always has a "decision" field. Possible values:
    "matched"       — top-1 similarity >= threshold; "owner" populated
    "stranger"      — top-1 similarity <  threshold; "owner" is null
    "no_face"       — detector found no face
    "empty_gallery" — gallery has no registered owners
    "decode_error"  — image bytes could not be decoded
"""

from pathlib import Path
from typing import Optional, Union

from .face_model import FaceModel
from .storage import Storage


DEFAULT_THRESHOLD = 0.30  # calibrated to FAR ~ 1e-4 on FaceScrub eval; adjust per env


def recognize(
    photo: Union[str, Path, bytes],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    model: Optional[FaceModel] = None,
    storage: Optional[Storage] = None,
) -> dict:
    """Run the full recognition pipeline. Returns a result dict; never raises
    for "expected" failures (no face, empty gallery, decode error) — those are
    reported via the `decision` field instead."""
    photo_bytes = _read_photo(photo)

    if model is None:
        model = FaceModel()
    if storage is None:
        storage = Storage()

    img = FaceModel.decode_image(photo_bytes)
    if img is None:
        return {"decision": "decode_error"}

    faces = model.detect_and_embed(img)
    if len(faces) == 0:
        return {"decision": "no_face"}

    # If multiple faces, pick the one with the largest bbox area (closest to camera).
    face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) *
                                    (f["bbox"][3] - f["bbox"][1]))
    n_faces = len(faces)

    if len(storage) == 0:
        return {"decision": "empty_gallery", "n_faces": n_faces,
                "face_bbox": face["bbox"]}

    owner_id, sim = storage.search_top1(face["embedding"])
    matched = sim is not None and sim >= threshold
    owner = storage.get_owner(owner_id) if matched else None

    return {
        "decision": "matched" if matched else "stranger",
        "owner_id": owner_id if matched else None,
        "owner": owner,
        "similarity": sim,
        "threshold": threshold,
        "face_bbox": face["bbox"],
        "n_faces_detected": n_faces,
    }


def _read_photo(photo: Union[str, Path, bytes]) -> bytes:
    if isinstance(photo, (str, Path)):
        return Path(photo).read_bytes()
    if isinstance(photo, (bytes, bytearray)):
        return bytes(photo)
    raise TypeError(f"photo must be path or bytes, got {type(photo).__name__}")
