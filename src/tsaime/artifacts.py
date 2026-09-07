"""Reproducible figure, table, manifest, and ZIP output helpers."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ArtifactWriter:
    """Write a clean, hashable research artifact bundle."""

    root: Path | str
    clear_existing: bool = True

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.figure_dir = self.root / "figures"
        self.csv_dir = self.root / "csv"
        self.tex_dir = self.root / "tex"
        self.log_dir = self.root / "logs"
        for directory in (self.root, self.figure_dir, self.csv_dir, self.tex_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if self.clear_existing:
            for directory in (self.figure_dir, self.csv_dir, self.tex_dir, self.log_dir, self.root):
                for path in directory.glob("*"):
                    if path.is_file():
                        path.unlink()

    def save_table(
        self,
        frame: pd.DataFrame,
        stem: str,
        caption: str,
        label: str,
        *,
        latex_max_rows: int | None = None,
        latex_frame: pd.DataFrame | None = None,
    ) -> tuple[Path, Path]:
        """Save the complete CSV and an explicitly selected LaTeX representation.

        ``latex_frame`` is intended for a compact, manuscript-facing summary of
        a long machine-readable table.  ``latex_max_rows`` remains only as a
        guard against accidental oversized tables; it never truncates data.
        """

        csv_path = self.csv_dir / f"{stem}.csv"
        tex_path = self.tex_dir / f"{stem}.tex"
        frame.to_csv(csv_path, index=False)
        if latex_frame is not None and latex_max_rows is not None:
            raise ValueError("use latex_frame or latex_max_rows, not both")
        shown = (latex_frame if latex_frame is not None else frame).copy()
        if latex_max_rows is not None and len(shown) > latex_max_rows:
            raise ValueError(
                f"{stem} has {len(shown)} rows; provide an explicit latex_frame "
                "instead of silently truncating the table"
            )
        shown = shown.replace([np.inf, -np.inf], np.nan)
        for column in shown.select_dtypes(include=[np.number]).columns:
            if pd.api.types.is_integer_dtype(shown[column].dtype):
                continue
            shown[column] = shown[column].map(
                lambda value: np.nan if pd.isna(value) else round(float(value), 4)
            )
        body = shown.to_latex(index=False, escape=True, na_rep="--")
        summary_note = (
            f"% LaTeX summary: {len(shown)} rows; complete CSV: {len(frame)} rows.\n"
            if latex_frame is not None
            else f"% Complete table: {len(frame)} rows.\n"
        )
        tex_path.write_text(
            summary_note
            + "\\begin{table}[htbp]\n\\centering\n"
            + body
            + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n",
            encoding="utf-8",
        )
        return csv_path, tex_path

    def save_csv(self, frame: pd.DataFrame, name: str) -> Path:
        path = self.csv_dir / name
        frame.to_csv(path, index=False)
        return path

    def save_figure(self, figure, stem: str, *, dpi: int = 300) -> tuple[Path, Path]:
        png_path = self.figure_dir / f"{stem}.png"
        pdf_path = self.figure_dir / f"{stem}.pdf"
        figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
        figure.savefig(pdf_path, bbox_inches="tight")
        try:
            import matplotlib.pyplot as plt

            plt.close(figure)
        except ImportError:
            pass
        return png_path, pdf_path

    def write_text(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def manifest(self) -> pd.DataFrame:
        rows = []
        for kind, directory in (
            ("figure", self.figure_dir),
            ("csv", self.csv_dir),
            ("tex", self.tex_dir),
            ("log", self.log_dir),
        ):
            for path in sorted(directory.glob("*")):
                if path.is_file() and path.name != "artifact_manifest.csv":
                    rows.append(
                        {
                            "kind": kind,
                            "relative_path": str(path.relative_to(self.root)),
                            "bytes": path.stat().st_size,
                            "sha256": _sha256(path),
                        }
                    )
        for path in sorted(self.root.glob("*")):
            if path.is_file() and path.suffix != ".zip":
                rows.append(
                    {
                        "kind": "root",
                        "relative_path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
        table = pd.DataFrame(rows).drop_duplicates("relative_path")
        table.to_csv(self.csv_dir / "artifact_manifest.csv", index=False)
        return table

    def create_zip(self, name: str) -> Path:
        path = self.root / name
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(self.root.rglob("*")):
                if item.is_file() and item != path:
                    archive.write(item, arcname=str(item.relative_to(self.root)))
        return path
