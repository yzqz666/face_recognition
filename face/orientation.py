import cv2
import numpy as np

from .face_model import FaceModel


ROTATION_CANDIDATES = (0, 180, 90, 270)


def rotate_image(img_bgr: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return img_bgr
    if degrees == 90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported rotation: {degrees}")


def detect_faces_with_rotation(model: FaceModel, img_bgr: np.ndarray) -> tuple[np.ndarray, list, int]:
    """Detect faces, trying common right-angle rotations if needed."""
    for degrees in ROTATION_CANDIDATES:
        candidate = rotate_image(img_bgr, degrees)
        faces = model.detect_and_embed(candidate)
        if faces:
            return candidate, faces, degrees
    return img_bgr, [], 0
