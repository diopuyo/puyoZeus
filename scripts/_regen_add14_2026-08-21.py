"""追加14本の収集 (2026-08-21)。

48本モデルの学習が完了 (AUC 0.6351、動画別中央値 0.6299) したが、前回の
148本モデル (0.6573) との差は**動画数の差** (144→47本、行数 1,035,639→248,749)
が支配的で、境界修正やRust化の効果とは切り分けられない。

そこで手元に動画がある未収集分から14本を足し、62本にしてから死に指標の
確認に進む (user 指示 2026-08-21)。

選定条件:
  - data/frames に動画が現存
  - boards_lean_model50v2_2026-08-20 に未収集
  - BROKEN_VIDEOS (c26/c30/c58/c69、score OCR破綻でwon欠損100%) を除外
    ※ 今回の48本には c58 が含まれてしまい欠損100% (4,098行) になった。
      全体の欠損2.84%のうち55%がこの1本によるもの
  - マスター級を優先

出力は既存の 48本と**同じ npz ディレクトリ**に追加する
(boards_lean_model50v2_2026-08-20)。学習側は npz ディレクトリ単位で読むため、
分けると再ビルドが二度手間になる。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_ORCH_PATH = PROJECT_ROOT / "scripts" / "_regen148_orchestrator_2026-08-11.py"
_spec = importlib.util.spec_from_file_location("_regen148_orchestrator_impl", _ORCH_PATH)
orch = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = orch
_spec.loader.exec_module(orch)


if __name__ == "__main__":
    base = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-21_add14"
    orch.MANIFEST_TSV = base / "manifest.tsv"
    orch.STATUS_TSV = base / "status.tsv"
    # 既存48本と同じディレクトリに追加する (学習は npz ディレクトリ単位で読む)
    orch.NEW_NPZ_DIR = (
        PROJECT_ROOT / "data" / "indicators_v2" / "boards_lean_model50v2_2026-08-20"
    )
    orch.LOG_DIR = PROJECT_ROOT / "logs" / "regen_add14_2026-08-21_per_video"
    # 並列14が実測最適 (2026-08-20、10並列は全体18%悪化。
    # docs/KNOWN_WEAKNESSES.md W27-c)
    orch.MAX_COLLECT_PARALLEL = int(os.environ.get("COLLECT_PARALLEL", "14"))
    orch.TOTAL_HOLD_SLOTS = orch.MAX_COLLECT_PARALLEL + orch.DOWNLOAD_QUEUE_SIZE
    orch._HOLD_SEMAPHORE = threading.Semaphore(orch.TOTAL_HOLD_SLOTS)

    if "--smoke-check" in sys.argv:
        from src.production_config import collect_flags
        cf = collect_flags()
        print("[smoke] MANIFEST =", orch.MANIFEST_TSV)
        print("[smoke] NPZ_DIR  =", orch.NEW_NPZ_DIR)
        print("[smoke] 並列     =", orch.MAX_COLLECT_PARALLEL)
        for n in ("--enable-native-hsv-classifier",
                  "--enable-score-reset-requires-zero",
                  "--enable-winner-panel-priority"):
            print(f"[smoke] {'OK ' if n in cf else 'NG '}{n}")
        raise SystemExit(0)

    raise SystemExit(orch.main())
