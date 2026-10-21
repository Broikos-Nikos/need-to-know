"""The suite runs where `uv sync` alone was run, as the README says: a test that needs an optional extra skips without
it (tick 80: a fresh clone failed one test that imported torch outright)."""

import ast
import tomllib

from needtoknow.settings import repo_root


def extras() -> set[str]:
    """The top level modules of every optional dependency, as pyproject.toml lists them."""
    groups = tomllib.loads((repo_root() / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    names = set()
    for requirements in groups.values():
        for requirement in requirements:
            name = requirement.split(";")[0].split("[")[0]
            for mark in "<>=!~ ":
                name = name.split(mark)[0]
            names.add(name.strip().replace("-", "_"))
    return names


def test_no_test_imports_an_optional_extra_except_through_importorskip():
    optional = extras()
    assert {"torch", "transformers"} <= optional
    found = []
    for path in sorted((repo_root() / "tests").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            found += [f"{path.name}:{node.lineno} {m}" for m in modules if m.split(".")[0] in optional]
    assert not found, f"import these through pytest.importorskip, so the suite runs without the extra: {found}"


def reason_of(call: ast.Call) -> ast.expr | None:
    """What a skip says, positionally as pytest.skip and the skip mark take it, or as reason=; an empty string says
    nothing."""
    name = ast.unparse(call.func)
    said = [k.value for k in call.keywords if k.arg in ("reason", "msg")]
    if name in ("pytest.skip", "pytest.mark.skip"):
        said += call.args[:1]
    return next((s for s in said if not (isinstance(s, ast.Constant) and not s.value)), None)


def test_every_skip_in_the_suite_says_what_it_needed():
    # MA-F2: how many tests pass is partly the machine's; a skip that says nothing would leave pytest -rs, which the
    # README gives for it, a line with no reason.
    silent = []
    for path in sorted((repo_root() / "tests").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) in ("pytest.skip", "pytest.mark.skip",
                                                                          "pytest.mark.skipif"):
                if reason_of(node) is None:
                    silent.append(f"{path.name}:{node.lineno}")
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                silent += [f"{path.name}:{d.lineno}" for d in node.decorator_list if ast.unparse(d) == "pytest.mark.skip"]
    assert not silent, f"say what the test needed: {silent}"
