"""Local file storage and in-process retrain lock adapters."""

import threading
from pathlib import Path

import pytest

from app.adapters.outbound.storage.local_file_storage import LocalFileStorage
from app.adapters.outbound.storage.thread_retrain_lock import ThreadRetrainLock


def test_save_writes_the_bytes_under_the_uploads_dir(tmp_path):
    storage = LocalFileStorage(uploads_dir=str(tmp_path / "uploads"))

    stored = Path(storage.save("rounds.csv", b"a,b\n1,2\n"))

    assert stored.read_bytes() == b"a,b\n1,2\n"
    assert stored.parent == (tmp_path / "uploads").resolve()
    assert stored.suffix == ".csv"


@pytest.mark.parametrize("filename", ["../../etc/passwd.csv", "/etc/cron.d/evil.csv",
                                      "..\\..\\win.csv", "rounds.csv"])
def test_save_never_uses_the_client_name_as_path(tmp_path, filename):
    uploads = tmp_path / "uploads"
    storage = LocalFileStorage(uploads_dir=str(uploads))

    stored = Path(storage.save(filename, b"x"))

    assert stored.parent == uploads.resolve()
    assert "passwd" not in stored.name and "evil" not in stored.name
    assert "rounds" not in stored.name
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == [stored.name]


def test_save_generates_a_new_name_each_time(tmp_path):
    storage = LocalFileStorage(uploads_dir=str(tmp_path))

    assert storage.save("same.csv", b"1") != storage.save("same.csv", b"2")


def test_delete_removes_the_stored_file_and_ignores_missing_ones(tmp_path):
    storage = LocalFileStorage(uploads_dir=str(tmp_path))
    stored = storage.save("rounds.csv", b"x")

    storage.delete(stored)
    storage.delete(stored)

    assert not Path(stored).exists()


def test_delete_never_touches_files_outside_the_uploads_dir(tmp_path):
    outside = tmp_path / "keep.csv"
    outside.write_bytes(b"keep")
    storage = LocalFileStorage(uploads_dir=str(tmp_path / "uploads"))

    storage.delete(str(outside))
    storage.delete(str(tmp_path / "uploads" / ".." / "keep.csv"))

    assert outside.read_bytes() == b"keep"


def test_lock_is_exclusive_and_non_blocking():
    lock = ThreadRetrainLock()

    assert lock.acquire() is True
    assert lock.acquire() is False
    lock.release()
    assert lock.acquire() is True


def test_lock_rejects_a_second_thread_while_held():
    lock = ThreadRetrainLock()
    lock.acquire()
    results = []

    worker = threading.Thread(target=lambda: results.append(lock.acquire()))
    worker.start()
    worker.join(timeout=2)

    assert results == [False]


def test_releasing_a_free_lock_is_harmless():
    lock = ThreadRetrainLock()

    lock.release()

    assert lock.acquire() is True
