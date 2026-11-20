"""RetrainLock adapter: one in-process, non-blocking lock.

Valid for a single API process (one uvicorn worker). Several workers or replicas would
need a shared lock, such as a PostgreSQL advisory lock.
"""

import threading


class ThreadRetrainLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        if self._lock.locked():
            self._lock.release()
