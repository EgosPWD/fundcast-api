"""Enforce the hexagonal dependency rule from AGENTS.md section 5."""

import ast
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[2] / "app"


def imported_modules(package: Path, root: Path = APP.parent) -> list[tuple[str, str]]:
    found = []
    for file in package.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found += [(file.name, alias.name) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                found += [(file.name, m) for m in _from_targets(node, file, root)]
    return found


def _from_targets(node: ast.ImportFrom, file: Path, root: Path) -> list[str]:
    """Absolute module names an ImportFrom reaches; relative forms are resolved against
    the importing file's package, so `from ..x import y` cannot dodge the checks."""
    if node.level == 0:
        return [node.module] if node.module else []
    package = list(file.relative_to(root).with_suffix("").parts)[:-1]
    base = package[: len(package) - (node.level - 1)]
    if node.module:
        return [".".join([*base, node.module])]
    # `from . import x`: each name may be a submodule of the base package.
    return [".".join([*base, alias.name]) for alias in node.names]


def test_relative_imports_resolve_to_the_same_module_as_absolute_ones(tmp_path):
    routers = tmp_path / "app" / "adapters" / "inbound" / "api" / "routers"
    routers.mkdir(parents=True)
    (routers / "absolute.py").write_text(
        "from app.adapters.outbound.security.bcrypt_hasher import BcryptPasswordHasher\n")
    (routers / "relative.py").write_text(
        "from ....outbound.security.bcrypt_hasher import BcryptPasswordHasher\n"
        "from .. import deps\n")

    assert sorted(imported_modules(routers, root=tmp_path)) == [
        ("absolute.py", "app.adapters.outbound.security.bcrypt_hasher"),
        ("relative.py", "app.adapters.inbound.api.deps"),
        ("relative.py", "app.adapters.outbound.security.bcrypt_hasher"),
    ]


def is_stdlib(module: str) -> bool:
    return module.split(".")[0] in sys.stdlib_module_names


@pytest.mark.parametrize(
    ("layer", "allowed_prefixes"),
    [
        ("domain", ("app.domain",)),
        # Use cases may import ports, never another use case (AGENTS.md section 5).
        ("application", ("app.domain", "app.application.ports")),
    ],
)
def test_inner_layers_only_import_stdlib_and_inner_layers(layer, allowed_prefixes):
    violations = [
        (file, module)
        for file, module in imported_modules(APP / layer)
        if not is_stdlib(module) and not module.startswith(allowed_prefixes)
    ]

    assert violations == []


FRAMEWORK_DEPS = ("sqlalchemy", "sklearn", "joblib", "pandas", "numpy", "psycopg", "alembic")


def top_level(module: str) -> str:
    return module.split(".")[0]


def test_inbound_adapters_never_import_persistence_or_ml_libraries():
    violations = [
        (file, module)
        for file, module in imported_modules(APP / "adapters" / "inbound")
        if top_level(module) in FRAMEWORK_DEPS or module.startswith("app.adapters.outbound")
    ]

    assert violations == []


def test_routers_only_import_use_cases_domain_ports_and_the_api_package():
    allowed_app = ("app.domain", "app.application.use_cases", "app.application.ports",
                   "app.adapters.inbound.api")
    allowed_libs = {"fastapi", "starlette", "pydantic"}
    violations = [
        (file, module)
        for file, module in imported_modules(APP / "adapters" / "inbound" / "api" / "routers")
        if not is_stdlib(module)
        and top_level(module) not in allowed_libs
        and not module.startswith(allowed_app)
    ]

    assert violations == []


def test_config_does_not_import_persistence_or_ml_libraries():
    tree = ast.parse((APP / "config.py").read_text(encoding="utf-8"))
    modules = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names]
    modules += [node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module]

    assert [m for m in modules if top_level(m) in FRAMEWORK_DEPS] == []
