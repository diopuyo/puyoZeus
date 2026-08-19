"""配線是正の2本検証収集 (2026-08-20)。

目的: 2026-08-19 の境界修正3フラグ (--enable-lockdown-score-numeric-release /
--enable-lockdown-score-moving-release / --enable-boundary-newmatch-evidence)
が `production_config.collect_flags()` に登録漏れしていた事故の是正を、
**本番オーケストレータ経由で実際に収集して**確認する。

--help 突合や単体スクリプトでの確認は無意味という教訓 (memory
`feedback_wiring_check_needs_nongeneric_scripts_2026-08-18`、本事故で3回目の
再発) を踏まえ、本番と同じ `_regen148_orchestrator_2026-08-11.py` の
collect_one をそのまま通す (コマンド構築を複製しない)。

対象2本 (user方針「2本ごとに確認して進める」に従う):
  - 39: 手書きフラグ版 (boards_lean_lockfix_2026-08-19) が既にあるため、
        本番経路の結果がそれと一致すれば配線成功が確定する
  - 38: 手書き版が無い動画。新規に効果が出るかを見る

動画は2本とも手元にあるため DL は発生しない (download_video は差し替えない)。
"""
from __future__ import annotations

import importlib.util
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
    base = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-20_wirecheck"
    orch.MANIFEST_TSV = base / "manifest.tsv"
    orch.STATUS_TSV = base / "status.tsv"
    orch.NEW_NPZ_DIR = (
        PROJECT_ROOT / "data" / "indicators_v2" / "boards_lean_wirecheck_2026-08-20"
    )
    orch.LOG_DIR = PROJECT_ROOT / "logs" / "regen_wirecheck_2026-08-20_per_video"
    # 2本だけなので並列は2で足りる (他の重い検証と競合させない、
    # memory: 3件同時検証で負荷39=16コアの倍になり互いに足を引っ張った教訓)
    orch.MAX_COLLECT_PARALLEL = 2
    orch.TOTAL_HOLD_SLOTS = orch.MAX_COLLECT_PARALLEL + orch.DOWNLOAD_QUEUE_SIZE
    orch._HOLD_SEMAPHORE = threading.Semaphore(orch.TOTAL_HOLD_SLOTS)

    if "--smoke-check" in sys.argv:
        from src.production_config import collect_flags
        cf = collect_flags()
        print("[smoke] MANIFEST =", orch.MANIFEST_TSV)
        print("[smoke] NPZ_DIR  =", orch.NEW_NPZ_DIR)
        print("[smoke] 並列     =", orch.MAX_COLLECT_PARALLEL)
        for n in ("--enable-lockdown-score-numeric-release",
                  "--enable-lockdown-score-moving-release",
                  "--enable-boundary-newmatch-evidence"):
            print(f"[smoke] {'OK ' if n in cf else 'NG '}{n}")
        raise SystemExit(0)

    raise SystemExit(orch.main())
