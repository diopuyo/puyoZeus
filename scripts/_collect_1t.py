"""cv2 を 1 スレッドに固定して collect_indicators_v2.main を呼ぶ非侵襲ラッパー。

matchTemplate(認識の 62%)は極小テンプレのため cv2 のマルチスレッドが効かず
(単独 1.03fps vs 1スレッド 0.95fps とほぼ同等)、多プロセス並列時は 16スレッド×N
のスレッドプール overhead で競合するだけ。1 スレッド固定で「1プロセス=1コア」を
クリーンに保ち、プロセス数を増やして総スループットを上げる。認識結果は不変
(matchTemplate はスレッド数によらず決定的)。
"""
import sys
from pathlib import Path

import cv2

# プロジェクトルートを import path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 最重要レバー: cv2 を単一スレッド化(プロセス間競合を排除)
cv2.setNumThreads(1)

from scripts.collect_indicators_v2 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
