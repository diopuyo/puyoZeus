"""Phase Z レビュー文字列を行配列で受け取り、apply スクリプトを叩くラッパ.

利用例:
    python scripts/_apply_review_helper.py <segment_id> <rows.txt>

rows.txt は 1 行 1 シート行。空白・改行は無視。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
SEG_ROOT = _ROOT / "data" / "verify" / "phase_z_review" / "weak_video_extra"


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python scripts/_apply_review_helper.py <segment_id> <rows.txt>")
        return 2
    seg = sys.argv[1]
    rows_path = Path(sys.argv[2])
    csv_path = SEG_ROOT / seg / "violations_review" / "violations.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return 1
    text = rows_path.read_text(encoding="utf-8")
    review = "".join(text.split()).upper()
    print(f"[helper] segment={seg}  review_chars={len(review)}")
    res = subprocess.run(
        ["./venv/bin/python", "-m", "scripts.phase_z_apply_weak_review",
         "--csv", str(csv_path), "--review", review],
        cwd=str(_ROOT),
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
    )
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
