"""列仕様台帳 (src/b3_column_spec.py) から docs/B3_COLUMN_SPEC.md を生成する。"""
from __future__ import annotations

from pathlib import Path

from src.b3_column_spec import render_markdown

OUT = Path(__file__).resolve().parents[1] / "docs" / "B3_COLUMN_SPEC.md"


def main() -> None:
    OUT.write_text(render_markdown(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
