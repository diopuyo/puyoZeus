"""現在の models/cnn_best.pt を cnn_global_best.pt として seed 登録する初回ユーティリティ。

重要: この seed は **Cycle 7 最終モデル (holdout 0.8124)** を登録する。
**Cycle 3 peak の holdout 0.8920 モデルは cnn_best.pt の毎サイクル上書き設計で既に失われており、復元不能**。
新規再学習で 0.8124 を超えた段階で自動的に cnn_global_best.pt が更新される仕組み。

再学習で cnn_best.pt が上書きされても、このシードが global best として守られる。
holdout 値は最後に記録された data/e2e_log.jsonl から読む。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODEL_DIR = Path("models")
CNN_BEST = MODEL_DIR / "cnn_best.pt"
GLOBAL_BEST = MODEL_DIR / "cnn_global_best.pt"
STATE = Path("data/global_best.json")
E2E_LOG = Path("data/e2e_log.jsonl")


def main() -> None:
    if not CNN_BEST.exists():
        print(f"{CNN_BEST} が無いので中止")
        return

    if STATE.exists() and GLOBAL_BEST.exists():
        cur = json.loads(STATE.read_text(encoding="utf-8"))
        print(f"既に global best 登録済: holdout={cur.get('holdout_acc')}")
        return

    holdout = 0.0
    internal = 0.0
    sanity = None
    symmetry = None
    if E2E_LOG.exists():
        for line in E2E_LOG.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            h = r.get("holdout_acc")
            if h is not None and h > holdout:
                holdout = float(h)
                internal = float(r.get("acc", 0.0))
                sanity = r.get("sanity_ok")
                symmetry = r.get("symmetry_ok")

    # ただし現 cnn_best.pt は Cycle 7 最終モデル (holdout 0.8124) であって
    # Cycle 3 peak 0.8920 のモデルではない。seed は Cycle 7 値を採用。
    # 最後の e2e_log レコードの holdout_acc を採用する。
    if E2E_LOG.exists():
        last = E2E_LOG.read_text(encoding="utf-8").strip().splitlines()[-1]
        r = json.loads(last)
        holdout = float(r.get("holdout_acc") or 0.0)
        internal = float(r.get("acc") or 0.0)
        sanity = r.get("sanity_ok")
        symmetry = r.get("symmetry_ok")
        print(f"e2e_log 最終レコードから seed: holdout={holdout:.4f} internal={internal:.4f}")

    shutil.copyfile(CNN_BEST, GLOBAL_BEST)
    state = {
        "holdout_acc": holdout,
        "internal_acc": internal,
        "prev_holdout_acc": 0.0,
        "sanity_ok": sanity,
        "symmetry_ok": symmetry,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cycle": None,
        "source": str(CNN_BEST),
        "note": "seed from Cycle 7 final cnn_best.pt (Cycle 3 peak 0.8920 was lost)",
    }
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"cnn_global_best.pt をシード登録: holdout={holdout:.4f}")


if __name__ == "__main__":
    main()
