"""スレッド数を 1 に固定して collect_boards_lean.main を呼ぶ非侵襲ラッパー。

memory `project_collect_indicators_v2_perf_2026-07-20` の教訓を collect_boards_lean
にも適用する。collect_boards_lean は RecognitionPipeline.load_default 経由で
score_ocr / UI mask 等の matchTemplate を internally 使うため、同じスレッドプール
競合(多プロセス並列時に 16スレッド×N で激遅化)が起きうる。
1 スレッド固定で「1プロセス=1コア」をクリーンに保ち、プロセス数を増やして
総スループットを上げる (認識結果は不変、matchTemplate はスレッド数によらず決定的)。

2026-08-20 追記 (実測にもとづく重要な修正): cv2 だけを 1 スレッドにしても
**CNN 推論 (torch) と BLAS/OpenMP は制限されていなかった**。48本収集の実測で
1プロセスあたり **72 スレッド**、14並列で合計 **1,008 スレッドを16コアで**
回しており、各プロセスの CPU 使用率は 107% (実質1コア強) に留まっていた。
1本あたりの所要が単独実行 (69分) の約4.3倍に膨らみ、48本で18時間の見積もりに
なっていた。スレッド数の制御は **torch を import する前に環境変数を置く**
必要がある (OMP/MKL 系は import 時にプールを確定するため、後から
set_num_threads しても遅い)。そのため本ラッパーの冒頭で os.environ を
設定する。ここは collect_boards_lean (=torch を間接 import する側) より
必ず前に実行されるので、全経路に効く。
"""
import os

# torch / numpy / BLAS のスレッドプールは import 時に確定するため、
# **何かを import する前に**環境変数で 1 に固定する。setdefault なので
# 呼出元が明示的に別値を渡していればそれを尊重する。
for _var in (
    "OMP_NUM_THREADS",       # OpenMP (torch CPU 演算 / numpy の一部)
    "MKL_NUM_THREADS",       # Intel MKL
    "OPENBLAS_NUM_THREADS",  # OpenBLAS
    "NUMEXPR_NUM_THREADS",   # numexpr
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402

# プロジェクトルートを import path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 最重要レバー: cv2 を単一スレッド化 (プロセス間競合を排除)
cv2.setNumThreads(1)

# torch 側も明示的に 1 に固定する (環境変数だけでは interop スレッドが
# 残るため両方必要)。torch が無い環境でも動くよう ImportError は無視する。
try:
    import torch  # noqa: E402

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except (ImportError, RuntimeError):
    # RuntimeError: set_num_interop_threads は並列処理開始後は変更不可。
    # その場合も環境変数側で既に抑えられているため続行する。
    pass

from scripts.collect_boards_lean import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
