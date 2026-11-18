"""Enforce the hexagonal dependency rule from AGENTS.md section 5."""

import ast
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[2] / "app"


def imported_modules(package: Path) -> list[tuple[str, str]]:
    found = []
    for file in package.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [(file.name, alias.name) for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.append((file.name, node.module))
    return found


def is_stdlib(module: str) -> bool:
    return module.split(".")[0] in sys.stdlib_module_names


@pytest.mark.parametrize(
    ("layer", "allowed_prefixes"),
    [
        ("domain", ("app.domain",)),
        ("application", ("app.domain", "app.application")),
    ],
)
def test_inner_layers_only_import_stdlib_and_inner_layers(layer, allowed_prefixes):
    violations = [
        (file, module)
        for file, module in imported_modules(APP / layer)
        if not is_stdlib(module) and not module.startswith(allowed_prefixes)
    ]

    assert violations == []
