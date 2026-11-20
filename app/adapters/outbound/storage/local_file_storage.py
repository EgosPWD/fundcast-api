"""FileStorage adapter on the local filesystem (UPLOADS_DIR)."""

import re
import uuid
from pathlib import Path

_SAFE_SUFFIX = re.compile(r"^\.[a-z0-9]{1,10}$")


class LocalFileStorage:
    """Stores uploads under server-generated names: the client filename never becomes a
    path, so it cannot traverse directories or overwrite another file."""

    def __init__(self, uploads_dir: str) -> None:
        self._root = Path(uploads_dir).resolve()

    def save(self, filename: str, content: bytes) -> str:
        self._root.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename.replace("\\", "/")).suffix.lower()
        target = self._root / f"{uuid.uuid4().hex}{suffix if _SAFE_SUFFIX.match(suffix) else ''}"
        with target.open("xb") as handle:
            handle.write(content)
        return str(target)

    def delete(self, stored_path: str) -> None:
        path = Path(stored_path).resolve()
        if not path.is_relative_to(self._root):
            # Only files this storage wrote are removable; anything else is left alone.
            return
        path.unlink(missing_ok=True)
