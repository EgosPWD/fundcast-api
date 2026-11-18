from typing import Protocol


class RetrainLock(Protocol):
    """Guarantees a single retraining at a time (HTTP 409 otherwise)."""

    def acquire(self) -> bool:
        """Try to take the lock without blocking. Return False if already taken."""
        ...

    def release(self) -> None: ...
