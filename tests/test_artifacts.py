from pathlib import Path

import pandas as pd
import pytest

from tsaime.artifacts import ArtifactWriter


def test_save_table_never_silently_truncates_latex(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)
    frame = pd.DataFrame({"value": range(5)})
    with pytest.raises(ValueError, match="silently truncating"):
        writer.save_table(
            frame,
            "long_table",
            "Long table.",
            "tab:long",
            latex_max_rows=3,
        )


def test_save_table_records_explicit_summary_and_complete_csv(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)
    frame = pd.DataFrame({"value": range(5)})
    summary = pd.DataFrame({"statistic": ["median"], "value": [2]})
    csv_path, tex_path = writer.save_table(
        frame,
        "summarized_table",
        "Explicit summary.",
        "tab:summary",
        latex_frame=summary,
    )
    assert len(pd.read_csv(csv_path)) == 5
    text = tex_path.read_text(encoding="utf-8")
    assert "LaTeX summary: 1 rows; complete CSV: 5 rows" in text
    assert "median" in text
    assert "2.000000" not in text
