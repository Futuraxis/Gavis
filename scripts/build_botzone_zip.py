"""Build a Botzone-uploadable Python zip bundle.

The bundle follows Botzone's multi-file Python guidance: a zip archive
with ``__main__.py`` at the root.  For Botzone's old python3 runtimes,
the bundle is intentionally a tiny standalone Mahjong-Format-Test bot
instead of the full Gavis source tree.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "dist" / "gavis_botzone.zip"

INCLUDE_PATHS = ()

EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".git",
    ".venv",
    ".venv-1",
    ".venv-2",
    "node_modules",
    "dist",
    "models",
    "data",
    "tests",
    "archive",
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Gavis Botzone zip bundle")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output zip path")
    parser.add_argument("--remote-url", default="", help="remote Gavis Botzone API, e.g. https://host/botzone/decide")
    parser.add_argument("--remote-token", default="", help="optional bearer token sent to the remote API")
    parser.add_argument("--remote-timeout", type=float, default=0.75, help="Botzone client HTTP timeout in seconds")
    args = parser.parse_args()
    build(args.out, remote_url=args.remote_url, remote_token=args.remote_token, remote_timeout=args.remote_timeout)
    print(args.out)


def build(out_path: Path, remote_url: str = "", remote_token: str = "", remote_timeout: float = 0.75) -> None:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Botzone may run an old python3 (3.5/3.6).  The upload bundle
        # therefore uses a tiny standalone Mahjong entrypoint instead of
        # importing the full Gavis project, whose source targets Python 3.11+.
        zf.writestr(
            "__main__.py",
            _render_py36_entrypoint(remote_url=remote_url, remote_token=remote_token, remote_timeout=remote_timeout),
        )
        for include in INCLUDE_PATHS:
            path = ROOT / include
            if path.is_file():
                _write_file(zf, path)
            elif path.is_dir():
                for file_path in sorted(path.rglob("*")):
                    if file_path.is_file() and not _excluded(file_path):
                        _write_file(zf, file_path)
            else:
                raise FileNotFoundError(path)


def _write_file(zf: zipfile.ZipFile, file_path: Path) -> None:
    rel = file_path.relative_to(ROOT).as_posix()
    zf.write(file_path, rel)


def _excluded(file_path: Path) -> bool:
    rel_parts = file_path.relative_to(ROOT).parts
    if any(part in EXCLUDED_PARTS for part in rel_parts):
        return True
    name = file_path.name
    if name.startswith(".DS_Store"):
        return True
    if name.endswith((".pyc", ".pyo", ".so", ".dylib")):
        return True
    # Keep uploaded source compact and deterministic.
    return os.path.getsize(file_path) == 0 and name != "__init__.py"


def _render_py36_entrypoint(remote_url: str, remote_token: str, remote_timeout: float) -> str:
    source = (ROOT / "layer4_interface" / "botzone" / "mahjong_format_py36.py").read_text(encoding="utf-8")
    replacements = {
        'REMOTE_URL = ""': f"REMOTE_URL = {json.dumps(remote_url)}",
        'REMOTE_TOKEN = ""': f"REMOTE_TOKEN = {json.dumps(remote_token)}",
        "REMOTE_TIMEOUT = 0.75": f"REMOTE_TIMEOUT = {float(remote_timeout)!r}",
    }
    for old, new in replacements.items():
        source = source.replace(old, new, 1)
    return source


if __name__ == "__main__":
    main()
