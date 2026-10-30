"""Static rules of the repository: documented code, imports at the top, no internals."""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY_FILES = sorted(p for folder in ("frozenradar", "tests") for p in (ROOT / folder).rglob("*.py"))
LIBRARY = sorted((ROOT / "frozenradar").rglob("*.py"))
OUTPUTS = {".venv", ".git", "weights", "cache", "logs", "runs", "reproduced", "wandb"}
TEXT_FILES = sorted(
    p for p in ROOT.rglob("*")
    if p.is_file() and p.suffix in {".py", ".sh", ".yaml", ".md", ".toml", ".txt", ".json"}
    and not OUTPUTS & set(p.relative_to(ROOT).parts) and p != Path(__file__))
FORBIDDEN = re.compile(r"/home/|/anvme|iwso|woody|sbatch|experiments\.|claude|anthropic", re.I)


def _tree(path: Path) -> ast.Module:
    """Parse one source file.

    Args:
        path: Python file.

    Returns:
        Its syntax tree.
    """
    return ast.parse(path.read_text(encoding="utf-8"))


def test_imports_only_at_module_level() -> None:
    """No import statement inside a function or class body."""
    for path in PY_FILES:
        for node in ast.walk(_tree(path)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for inner in ast.walk(node):
                    assert not isinstance(inner, (ast.Import, ast.ImportFrom)), (
                        f"import inside {node.name} in {path.relative_to(ROOT)}")


def test_every_module_function_and_class_is_documented() -> None:
    """Module, class and function docstrings everywhere."""
    for path in PY_FILES:
        tree = _tree(path)
        assert ast.get_docstring(tree), f"{path.relative_to(ROOT)} has no module docstring"
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (
                    not node.name.startswith("__")):
                assert ast.get_docstring(node), (
                    f"{node.name} in {path.relative_to(ROOT)} has no docstring")


def test_every_function_is_typed() -> None:
    """Return annotations on every function."""
    for path in PY_FILES:
        for node in ast.walk(_tree(path)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.returns is not None, (
                    f"{node.name} in {path.relative_to(ROOT)} has no return annotation")


def test_no_print_or_path_hacks_in_library() -> None:
    """Library code logs instead of printing and never edits sys.path."""
    for path in LIBRARY:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*print\(", text, re.M), path.relative_to(ROOT)
        assert "sys.path" not in text, path.relative_to(ROOT)


def test_no_internal_references() -> None:
    """No cluster paths, internal job names or tool attributions in any shipped file."""
    for path in TEXT_FILES:
        hit = FORBIDDEN.search(path.read_text(encoding="utf-8", errors="replace"))
        assert hit is None, f"{path.relative_to(ROOT)}: {hit.group(0) if hit else ''}"


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
def test_ruff_is_clean() -> None:
    """The strict ruff configuration in pyproject.toml reports nothing."""
    result = subprocess.run(["ruff", "check", "--no-cache", str(ROOT)], capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, result.stdout[-4000:]
