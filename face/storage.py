"""File-based storage for owners. Each owner is one folder under owners/:

    business_data/
    ├── state.json                  {"next_owner_id": <int>}
    └── owners/
	    └── 000001/
	        ├── meta.json           {"owner_id":1, "name":..., "room":..., ...}
	        ├── embedding.npy       (512,) float32, L2-normalized
	        ├── photo.jpg
	        └── face.jpg

In addition to the on-disk form, Storage keeps an in-memory (N, 512) matrix
and parallel ids/meta arrays so 1:N search is just a numpy matmul.
"""

import json
import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "business_data"
EMBEDDING_DIM = 512
_UNSET = object()


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        Path(tmp).replace(path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp.", suffix=".npy")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
        Path(tmp).replace(path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


class Storage:
    def __init__(self, data_dir: Path | str | None = None):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.owners_dir = self.data_dir / "owners"
        self.state_path = self.data_dir / "state.json"
        self.owners_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self.feats = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        self.ids = np.empty((0,), dtype=np.int64)
        self._meta: dict[int, dict] = {}
        self._scan_owners()

    # ---------- public ----------
    def save_owner(self, name: str, embedding: np.ndarray,
                   photo_bytes: bytes, *, face_photo_bytes: bytes | None = None,
                   room: Optional[str] = None, phone: Optional[str] = None) -> dict:
        """Allocate next owner_id, write owner files, update in-memory state.
        Returns the meta dict (including owner_id)."""
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        assert embedding.shape == (EMBEDDING_DIM,), \
            f"embedding must be ({EMBEDDING_DIM},), got {embedding.shape}"

        with self._lock:
            owner_id = self._allocate_id()
            owner_dir = self.owners_dir / f"{owner_id:06d}"
            owner_dir.mkdir(parents=True, exist_ok=True)

            meta = {
                "owner_id": owner_id,
                "name": name,
                "room": room,
                "phone": phone,
                "registered_at": datetime.now().isoformat(timespec="seconds"),
            }
            try:
                _write_json_atomic(owner_dir / "meta.json", meta)
                _save_npy_atomic(owner_dir / "embedding.npy", embedding)
                (owner_dir / "photo.jpg").write_bytes(photo_bytes)
                if face_photo_bytes is not None:
                    (owner_dir / "face.jpg").write_bytes(face_photo_bytes)
            except Exception:
                # best-effort cleanup of partial files
                for n in ("meta.json", "embedding.npy", "photo.jpg", "face.jpg"):
                    (owner_dir / n).unlink(missing_ok=True)
                try:
                    owner_dir.rmdir()
                except OSError:
                    pass
                raise

            # update memory
            self.feats = np.vstack([self.feats, embedding.reshape(1, -1)])
            self.ids = np.append(self.ids, np.int64(owner_id))
            self._meta[owner_id] = meta
            return meta

    def list_owners(self) -> list[dict]:
        with self._lock:
            return [self._meta[i] for i in sorted(self._meta)]

    def get_owner(self, owner_id: int) -> dict | None:
        with self._lock:
            return self._meta.get(owner_id)

    def update_owner(self, owner_id: int, *, name=_UNSET,
                     room=_UNSET, phone=_UNSET) -> dict | None:
        """Update owner metadata. Returns updated meta, or None if not found.

        Omitted fields are left unchanged. Pass room=None or phone=None to clear
        those fields.
        """
        owner_id = int(owner_id)
        with self._lock:
            current = self._meta.get(owner_id)
            if current is None:
                return None

            updated = dict(current)
            if name is not _UNSET:
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("name must be a non-empty string")
                updated["name"] = name
            if room is not _UNSET:
                updated["room"] = room
            if phone is not _UNSET:
                updated["phone"] = phone

            owner_dir = self.owners_dir / f"{owner_id:06d}"
            if not owner_dir.exists():
                return None

            _write_json_atomic(owner_dir / "meta.json", updated)
            self._meta[owner_id] = updated
            return dict(updated)

    def delete_owner(self, owner_id: int) -> dict | None:
        """Delete an owner from disk and the in-memory search index.

        Returns the deleted meta, or None if the owner does not exist.
        """
        owner_id = int(owner_id)
        with self._lock:
            current = self._meta.get(owner_id)
            if current is None:
                return None

            owner_dir = self.owners_dir / f"{owner_id:06d}"
            if owner_dir.exists():
                shutil.rmtree(owner_dir)

            keep = self.ids != owner_id
            self.ids = self.ids[keep]
            self.feats = self.feats[keep]
            deleted = self._meta.pop(owner_id)
            return dict(deleted)

    def search_top1(self, embedding: np.ndarray) -> tuple[int | None, float | None]:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        with self._lock:
            if len(self.ids) == 0:
                return None, None
            sims = self.feats @ emb
            idx = int(np.argmax(sims))
            return int(self.ids[idx]), float(sims[idx])

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    # ---------- internal ----------
    def _allocate_id(self) -> int:
        state = self._load_state()
        owner_id = state["next_owner_id"]
        state["next_owner_id"] = owner_id + 1
        _write_json_atomic(self.state_path, state)
        return owner_id

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"next_owner_id": 1}

    def _scan_owners(self) -> None:
        feats, ids_list, meta = [], [], {}
        for d in sorted(self.owners_dir.iterdir()) if self.owners_dir.exists() else []:
            if not d.is_dir():
                continue
            meta_path = d / "meta.json"
            emb_path = d / "embedding.npy"
            if not (meta_path.exists() and emb_path.exists()):
                print(f"[storage] skip incomplete owner dir: {d.name}")
                continue
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            e = np.load(emb_path).astype(np.float32).reshape(-1)
            if e.shape != (EMBEDDING_DIM,):
                print(f"[storage] skip {d.name}: bad embedding shape {e.shape}")
                continue
            feats.append(e)
            ids_list.append(int(m["owner_id"]))
            meta[int(m["owner_id"])] = m
        if feats:
            self.feats = np.vstack(feats).astype(np.float32)
            self.ids = np.array(ids_list, dtype=np.int64)
        self._meta = meta
        print(f"[storage] loaded {len(self._meta)} owners from {self.data_dir}")
