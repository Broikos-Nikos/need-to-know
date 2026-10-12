"""The corpus: the fact registry and catalogue, the writers' assignments, assembly and a light check."""

import json
from pathlib import Path


def dump_json(path: Path, data) -> None:
    """JSON with LF line ends on every platform: Path.write_text would write CRLF on Windows."""
    path.write_bytes((json.dumps(data, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
