from .storage import Storage, _UNSET


def list_owners(*, storage: Storage | None = None) -> list[dict]:
    """List all registered owners."""
    if storage is None:
        storage = Storage()
    return storage.list_owners()


def update_owner(
    owner_id: int,
    *,
    name=_UNSET,
    room=_UNSET,
    phone=_UNSET,
    storage: Storage | None = None,
) -> dict | None:
    """Update owner metadata by owner_id."""
    if storage is None:
        storage = Storage()
    return storage.update_owner(owner_id, name=name, room=room, phone=phone)


def delete_owner(owner_id: int, *, storage: Storage | None = None) -> dict | None:
    """Delete an owner by owner_id."""
    if storage is None:
        storage = Storage()
    return storage.delete_owner(owner_id)
