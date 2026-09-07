"""Build a code-only release ZIP; never include observations or alter originals.

This maintenance tool is not an APR runtime dependency. All experiment code
resides in the notebook; the installable package contains generic methods only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.3.3"
REQUIRED = ["README.md", "pyproject.toml", "MANIFEST.in", "CITATION.cff", "LICENSE.txt",
            "DATA_SOURCES.md", "requirements-v9.4.txt", ".gitignore"]


def release_members() -> dict[str, bytes]:
    members = {name: (ROOT / name).read_bytes() for name in REQUIRED}
    for directory, pattern in (("src/tsaime", "*.py"), ("tests", "*.py"),
                               ("docs", "*.md"), ("tools", "*.py")):
        for path in sorted((ROOT / directory).rglob(pattern)):
            if "__pycache__" not in path.parts:
                members[str(path.relative_to(ROOT))] = path.read_bytes()
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
                cell.get("metadata", {}).pop("execution", None)
        notebook.get("metadata", {}).pop("widgets", None)
        text = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
        if "/Users/" in text or "/private/tmp/" in text:
            raise ValueError(f"Private local path in release notebook: {path.name}")
        members[str(path.relative_to(ROOT))] = text.encode()
    for pattern in (f"tsaime-{VERSION}-*.whl", f"tsaime-{VERSION}.tar.gz"):
        paths = sorted((ROOT / "dist").glob(pattern))
        if len(paths) != 1:
            raise FileNotFoundError(f"Build the current distribution first: {pattern}")
        members["dist/" + paths[0].name] = paths[0].read_bytes()
    if 'version = "'+VERSION+'"' not in members["pyproject.toml"].decode():
        raise ValueError("Release and package versions disagree")
    if "version: " + VERSION not in members["CITATION.cff"].decode():
        raise ValueError("Citation and package versions disagree")
    license_text = members["LICENSE.txt"].decode()
    if "PolyForm Noncommercial License 1.0.0" not in license_text:
        raise ValueError("Required noncommercial license text is missing")
    for name in members:
        if any(part in {"results", "private_traces", "archives", "audits", "__pycache__"} for part in Path(name).parts):
            raise ValueError(f"Forbidden release member: {name}")
    members["RELEASE_CONTENTS.md"] = (
        "# APR v9.4 code release candidate\n\n"
        "Package: tsaime 0.3.3. Workflow: notebooks/tsAIME_APR_all_experiments_v9_4.ipynb.\n"
        "No raw observations, result archives, private traces, or notebook execution outputs are included.\n"
        "Historical notebooks are clean copies; local executed originals were not modified.\n"
        "The package requires only generic numerical dependencies; APR code stays in the notebook.\n"
        "No DOI, GitHub publication, acceptance, or dataset-license clearance is asserted.\n"
        "NIES data notices require manual resolution before releasing derived research results.\n"
        "The controlling software terms are in LICENSE.txt.\n"
    ).encode()
    return members


def build_release() -> Path:
    members = release_members()
    manifest = [{"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in sorted(members.items())]
    members["RELEASE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    folder = ROOT / "release"
    folder.mkdir(exist_ok=True)
    path = folder / "timeseries-aime-v9.4-code-release.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            archive.writestr("timeseries-aime/" + name, data)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        for row in manifest:
            data = archive.read("timeseries-aime/" + row["file"])
            assert hashlib.sha256(data).hexdigest() == row["sha256"]
    print(path)
    print(f"Verified {len(manifest)} files; no observations or executed notebook outputs.")
    return path


if __name__ == "__main__":
    build_release()
