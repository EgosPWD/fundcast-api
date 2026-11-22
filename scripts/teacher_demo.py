"""Teacher flow demo (AGENTS.md section 12, step 9).

upload data -> retrain -> upload more data -> retrain, against a running API.

    python scripts/teacher_demo.py prepare
    docker compose down -v      # the seed only loads into an EMPTY database
    SEED_DATASET_PATH=./data/demo/seed.csv docker compose up -d --build
    python scripts/teacher_demo.py run

`prepare` splits the Kaggle CSV in file order: the first 80 % seeds the database and the
remaining rows become two upload batches of 10 % each. Every uploaded row is therefore a
real round the database has not seen yet (no copies, no new round_ids invented).

Only the standard library is used, so the script runs with any Python 3.10+.
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "global_startup_funding_sample.csv"
DEFAULT_OUT = ROOT / "data" / "demo"
SEED_SHARE, BATCH_SHARE = 0.8, 0.1
RETRAIN_TIMEOUT_S = 600


def prepare(source: Path, out_dir: Path) -> dict[str, Path]:
    """Split `source` in file order into seed (80 %), batch_1 (10 %) and batch_2 (rest)."""
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    seed_end = round(len(rows) * SEED_SHARE)
    batch_end = seed_end + round(len(rows) * BATCH_SHARE)
    parts = {"seed": rows[:seed_end], "batch_1": rows[seed_end:batch_end],
             "batch_2": rows[batch_end:]}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, part in parts.items():
        paths[name] = out_dir / f"{name}.csv"
        with paths[name].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(part)
    return paths


def multipart_file(field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    """Encode one file as multipart/form-data; returns the body and its Content-Type."""
    boundary = uuid.uuid4().hex
    head = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            "Content-Type: text/csv\r\n\r\n").encode()
    return head + content + f"\r\n--{boundary}--\r\n".encode(), \
        f"multipart/form-data; boundary={boundary}"


class Api:
    def __init__(self, base_url: str) -> None:
        self._base, self._token = base_url.rstrip("/"), None

    def call(self, method: str, path: str, body: bytes | None = None,
             content_type: str = "application/json", timeout: float = 30) -> dict:
        request = urllib.request.Request(self._base + path, data=body, method=method)
        if body is not None:
            request.add_header("Content-Type", content_type)
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            sys.exit(f"{method} {path} -> {exc.code}: {detail}")

    def wait_healthy(self, attempts: int = 60) -> None:
        for _ in range(attempts):
            try:
                with urllib.request.urlopen(self._base + "/health", timeout=5):
                    return
            except OSError:
                time.sleep(2)
        sys.exit(f"the API at {self._base} did not answer /health")

    def login(self, email: str, password: str) -> None:
        body = json.dumps({"email": email, "password": password}).encode()
        self._token = self.call("POST", "/auth/login", body)["access_token"]


def read_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" not in line or line.startswith("#"):
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        values[key.strip()] = _dotenv_value(value.strip())
    return values


def _dotenv_value(raw: str) -> str:
    """Decode a value the way Compose and pydantic-settings do: enclosing quotes are
    removed (and keep a `#` inside them); an unquoted value ends at ` #`."""
    if len(raw) >= 2 and raw[0] in "'\"" and raw[-1] == raw[0]:
        return raw[1:-1]
    return raw.split(" #", 1)[0].rstrip()


def show_version(label: str, version: dict) -> None:
    m = version["metrics"]
    state = "ACTIVE" if version.get("is_active", version.get("activated")) else "inactive"
    print(f"  {label}: v{version['version']} {state}  R2={m['r2']:.4f}  "
          f"MAE_log={m['mae_log']:.4f}  n_train={version['n_train']}  n_test={version['n_test']}")


def run(api: Api, batches: list[Path]) -> None:
    api.wait_healthy()
    active = api.call("GET", "/metrics")["active"]
    print("Start")
    show_version("active model", active)
    for step, batch in enumerate(batches, start=1):
        print(f"\nStep {step}: upload {batch.name}")
        body, content_type = multipart_file("file", batch.name, batch.read_bytes())
        loaded = api.call("POST", "/training-files", body, content_type, timeout=120)
        print(f"  rows={loaded['rows']} inserted={loaded['rows_inserted']} "
              f"rejected={loaded['rows_rejected']} status={loaded['status']}")
        print(f"Step {step}: retrain on every row of funding_rounds (new 80/20 split)")
        outcome = api.call("POST", "/retrain", b"", timeout=RETRAIN_TIMEOUT_S)
        show_version("new model", outcome)
        if outcome["message"]:
            print(f"  {outcome['message']}")
    print("\nVersion history (GET /metrics)")
    for version in api.call("GET", "/metrics")["versions"]:
        show_version("version", version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="split the Kaggle CSV into seed + 2 batches")
    prep.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prep.add_argument("--out", type=Path, default=DEFAULT_OUT)
    go = commands.add_parser("run", help="upload -> retrain -> upload more -> retrain")
    go.add_argument("--url", default="http://localhost:8000")
    go.add_argument("--batches", type=Path, nargs="+",
                    default=[DEFAULT_OUT / "batch_1.csv", DEFAULT_OUT / "batch_2.csv"])
    args = parser.parse_args()

    if args.command == "prepare":
        for name, path in prepare(args.source, args.out).items():
            with path.open(newline="", encoding="utf-8") as handle:
                # csv rows, not lines: a quoted field may contain a line break.
                print(f"{name}: {sum(1 for _ in csv.reader(handle)) - 1} rows -> {path}")
        return
    env = {**read_dotenv(ROOT / ".env"), **os.environ}
    if not env.get("ADMIN_EMAIL") or not env.get("ADMIN_PASSWORD"):
        sys.exit("set ADMIN_EMAIL and ADMIN_PASSWORD (environment or .env)")
    api = Api(args.url)
    api.wait_healthy()
    api.login(env["ADMIN_EMAIL"], env["ADMIN_PASSWORD"])
    run(api, args.batches)


if __name__ == "__main__":
    main()
