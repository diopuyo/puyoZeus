"""cv2 を 1 スレッドに固定して collect_boards_lean.main を呼ぶ非侵襲ラッパー。

memory `project_collect_indicators_v2_perf_2026-07-20` の教訓を collect_boards_lean
にも適用する。collect_boards_lean は RecognitionPipeline.load_default 経由で
score_ocr / UI mask 等の matchTemplate を internally 使うため、同じスレッドプール
競合(多プロセス並列時に 16スレッド×N で激遅化)が起きうる。
1 スレッド固定で「1プロセス=1コア」をクリーンに保ち、プロセス数を増やして
総スループットを上げる (認識結果は不変、matchTemplate はスレッド数によらず決定的)。
"""
import sys
from pathlib import Path

import cv2

# プロジェクトルートを import path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 最重要レバー: cv2 を単一スレッド化 (プロセス間競合を排除)
cv2.setNumThreads(1)

from scripts.collect_boards_lean import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
