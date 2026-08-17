"""_train_model() 内部の段階別タイミング診断 (2026-08-11、使い捨て)。

scan_judgment_anomalies のスモーク実走が10並列regen競合下で2時間経っても
完了せず harness に kill されたため、どの段階が重いかを切り分ける。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.model_indicator_win import (  # noqa: E402
    build_features, load_labeled_csv, pair_sides_for_win,
)
from scripts.visualize_advantage_overlay import (  # noqa: E402
    TRAIN_CSV_PATH, _add_interaction_columns, _resolve_features,
)


def main() -> int:
    t0 = time.time()
    df = load_labeled_csv(TRAIN_CSV_PATH)
    print(f"[t] load_labeled_csv: {time.time() - t0:.1f}s rows={len(df)}", flush=True)

    t1 = time.time()
    feat_cols = _resolve_features(df)
    print(f"[t] resolve_features: {time.time() - t1:.1f}s n_cols={len(feat_cols)}", flush=True)

    t2 = time.time()
    paired = pair_sides_for_win(df, max_tdiff=1.0)
    print(f"[t] pair_sides_for_win: {time.time() - t2:.1f}s paired={len(paired)}", flush=True)

    t3 = time.time()
    feat = build_features(paired, feat_cols)
    print(f"[t] build_features: {time.time() - t3:.1f}s", flush=True)

    t4 = time.time()
    feat, cols = _add_interaction_columns(feat, feat_cols, paired)
    print(f"[t] add_interaction_columns: {time.time() - t4:.1f}s n_cols={len(cols)}", flush=True)

    print(f"[t] TOTAL: {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
