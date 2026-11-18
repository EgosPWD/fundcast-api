from typing import Protocol


class FileStorage(Protocol):
    def save(self, filename: str, content: bytes) -> str:
        """Store bytes under a server-generated name and return the stored path."""
        ...

    def delete(self, stored_path: str) -> None:
        """Remove a stored file. Missing files are ignored."""
        ...
