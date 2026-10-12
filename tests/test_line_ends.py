"""LF everywhere this project writes: Path.write_text writes CRLF on Windows, which turned a source file and six
committed sidecars CRLF before Git normalised them (ticks 14 and 19)."""

import pathlib
import re

import needtoknow
from needtoknow.settings import repo_root


def test_no_source_file_writes_text_mode_files():
    source = pathlib.Path(needtoknow.__file__).parent
    offenders = [f"{path.name}:{n}" for path in source.rglob("*.py")
                 for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                 if re.search(r"\.write_text\(", line)]
    assert offenders == [], "write bytes, or use needtoknow.corpus.dump_json: " + ", ".join(offenders)


def test_the_committed_samples_have_lf_line_ends():
    crlf = [p.name for p in (repo_root() / "samples").iterdir() if b"\r\n" in p.read_bytes()]
    assert crlf == []
