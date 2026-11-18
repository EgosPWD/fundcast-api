from typing import Protocol


class FileStorage(Protocol):
    def save(self, filename: str, content: bytes) -> str:
        """Store bytes under a server-generated name and return the stored path."""
        ...
