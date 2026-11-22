"""The teacher demo script: dataset split and multipart upload body."""

import csv
import importlib.util
from email.parser import BytesParser
from email.policy import HTTP
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "teacher_demo.py"
spec = importlib.util.spec_from_file_location("teacher_demo", SCRIPT)
assert spec and spec.loader
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def write_csv(path: Path, n_rows: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round_id", "company_name"])
        writer.writerows([[f"r{i}", f"Acme, {i}"] for i in range(n_rows)])


def read_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["round_id", "company_name"]
    return [row[0] for row in rows[1:]]


def test_prepare_splits_in_file_order_into_seed_and_two_disjoint_batches(tmp_path):
    source = tmp_path / "original.csv"
    write_csv(source, 100)

    paths = demo.prepare(source, tmp_path / "demo")

    seed, batch_1, batch_2 = (read_ids(paths[name]) for name in ("seed", "batch_1", "batch_2"))
    assert seed == [f"r{i}" for i in range(80)]
    assert batch_1 == [f"r{i}" for i in range(80, 90)]
    assert batch_2 == [f"r{i}" for i in range(90, 100)]


def test_multipart_body_carries_the_file_bytes_unchanged():
    content = b"round_id,company_name\r\nr1,\"Acme, Inc\"\n"

    body, content_type = demo.multipart_file("file", "batch_1.csv", content)

    message = BytesParser(policy=HTTP).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + body)
    (part,) = message.iter_parts()
    assert part.get_param("name", header="content-disposition") == "file"
    assert part.get_filename() == "batch_1.csv"
    assert part.get_payload(decode=True) == content


def test_read_dotenv_decodes_values_like_compose(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comment\n"
        "ADMIN_EMAIL=admin@example.com\n"
        "ADMIN_PASSWORD='demo # secret'\n"
        'DB_PASSWORD="p=ss"\n'
        "export API_PORT=8001 # inline comment\n"
        "EMPTY=\n",
        encoding="utf-8",
    )

    assert demo.read_dotenv(dotenv) == {
        "ADMIN_EMAIL": "admin@example.com", "ADMIN_PASSWORD": "demo # secret",
        "DB_PASSWORD": "p=ss", "API_PORT": "8001", "EMPTY": "",
    }
