from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face import FaceModel
from face.register import _crop_face_jpeg
from face.storage import DEFAULT_DATA_DIR


def main() -> None:
    owners_dir = DEFAULT_DATA_DIR / "owners"
    model = FaceModel()

    for owner_dir in sorted(owners_dir.iterdir()):
        if not owner_dir.is_dir():
            continue

        photo_path = owner_dir / "photo.jpg"
        face_path = owner_dir / "face.jpg"
        if not photo_path.exists():
            print(f"{owner_dir.name}: skip, photo.jpg missing")
            continue

        image = cv2.imdecode(
            np.frombuffer(photo_path.read_bytes(), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            print(f"{owner_dir.name}: skip, cannot decode photo.jpg")
            continue

        faces = model.detect_and_embed(image)
        if not faces:
            print(f"{owner_dir.name}: skip, no face detected")
            continue

        face = max(
            faces,
            key=lambda f: (f["bbox"][2] - f["bbox"][0]) *
            (f["bbox"][3] - f["bbox"][1]),
        )
        face_path.write_bytes(_crop_face_jpeg(image, face["bbox"]))
        print(f"{owner_dir.name}: wrote face.jpg ({len(faces)} face(s) detected)")


if __name__ == "__main__":
    main()
