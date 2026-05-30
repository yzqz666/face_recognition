"""Production face recognition app.

Public entry points:
    from face_recoginition import FaceModel, Face
    from face_recoginition import Storage
    from face_recoginition import register, recognize, capture
"""

from .capture import capture
from .face_model import Face, FaceModel
from .manage import delete_owner, list_owners, update_owner
from .recognize import recognize
from .register import register
from .storage import Storage

__all__ = [
    "FaceModel", "Face", "Storage", "register", "recognize", "capture",
    "list_owners", "update_owner", "delete_owner",
]
