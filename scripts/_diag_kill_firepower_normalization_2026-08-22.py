"""15連鎖直後の判定 (t=6717.5, 30先2セット動画 seg08) を計装する診断スクリプト。

user/coordinator 仮説: 火力 (お邪魔予告) が0-1正規化で頭打ちになり、
未正規化の構造特徴量 (連結対数差等) に「主因」表示で負けている。

本スクリプトはコードを一切変更せず、visualize_advantage_overlay の
_score_advantage_full_row を monkeypatch して呼び出しごとの
入力特徴量ベクトル (f1/f2)・モデル出力 (p_1p_wins/p_2p_wins)・
_side_feats_full が実際にセットしなかった列 (fail-silent 0.0 埋め)
を記録する。

出力: logs/_diag_kill_display_2026-08-22/*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
cv2.setNumThreads(1)

import numpy as np  # noqa: E402
import scripts.visualize_advantage_overlay as vao  # noqa: E402

VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
MODEL_DIR = PROJECT_ROOT / "data/verify/retrain_model62_2026-08-21"
OUT_DIR = PROJECT_ROOT / "logs/_diag_kill_display_2026-08-22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

records: list[dict] = []
_state = {"t": None}
_orig = vao._score_advantage_full_row
_orig_drive_ojama = vao._drive_ojama


def _patched_drive_ojama(tracker, sp1, sp2, ps1, ps2, t, *a, **kw):
    _state["t"] = t
    return _orig_drive_ojama(tracker, sp1, sp2, ps1, ps2, t, *a, **kw)


vao._drive_ojama = _patched_drive_ojama


def _patched(model, b1, b2, snap, attribution_exclude):
    cols = model._puyo_full_cols
    base1, base2 = vao._side_feats_full_base(b1), vao._side_feats_full_base(b2)
    f1 = vao._side_feats_full(base1, base2, snap.net_balance_capped, snap.forecast_p1)
    f2 = vao._side_feats_full(base2, base1, -snap.net_balance_capped, snap.forecast_p2)
    x1 = np.array([[np.nan_to_num(f1.get(c, 0.0)) for c in cols]], dtype=float)
    x2 = np.array([[np.nan_to_num(f2.get(c, 0.0)) for c in cols]], dtype=float)
    p_1p_wins = float(model.predict_proba(x1)[0, 1])
    p_2p_wins = float(model.predict_proba(x2)[0, 1])
    p1 = 0.5 * (p_1p_wins + (1.0 - p_2p_wins))
    adv = (p1 - 0.5) * 200.0
    missing_cols = [c for c in cols if c not in f1]
    all_candidates = sorted(
        ((c, vao._driver_value(c, f1, f2)) for c in vao.JP_LABEL if c in f1),
        key=lambda kv: -abs(kv[1]))
    drivers = [kv for kv in all_candidates if kv[0] not in attribution_exclude][:3]
    records.append(dict(
        t_sec=_state["t"],
        adv=adv, p1=p1, p_1p_wins=p_1p_wins, p_2p_wins=p_2p_wins,
        snap_forecast_p1=snap.forecast_p1, snap_forecast_p2=snap.forecast_p2,
        snap_pending_p1=snap.pending_p1, snap_pending_p2=snap.pending_p2,
        f1_board_ojama_count=base1["board_ojama_count"],
        f2_board_ojama_count=base2["board_ojama_count"],
        f1_ojama_forecast=f1["ojama_forecast"], f2_ojama_forecast=f2["ojama_forecast"],
        f1_current_max_chain=base1["current_max_chain"],
        f2_current_max_chain=base2["current_max_chain"],
        f1_conn_pair_count=base1["conn_pair_count"], f2_conn_pair_count=base2["conn_pair_count"],
        diff_conn_pair_count=f1["diff_conn_pair_count"],
        diff_board_ojama_count=f1["diff_board_ojama_count"],
        diff_current_max_chain=f1["diff_current_max_chain"],
        missing_cols=missing_cols,
        drivers=[(n, float(v)) for n, v in drivers],
    ))
    return _orig(model, b1, b2, snap, attribution_exclude)


vao._score_advantage_full_row = _patched


def main() -> None:
    start_sec, end_sec, warmup = 6660.0, 6760.0, 30.0
    vao.generate(
        video=VIDEO,
        out=OUT_DIR / "_dummy.mp4",
        max_sec=0.0,
        sample_interval=0.0,
        start_sec=start_sec,
        end_sec=end_sec,
        warmup_sec=warmup,
        model_dir=MODEL_DIR,
        force_in_match=False,
        layout="panel",
        panel_subtitle_h=0,
        render=False,
        dump_timeline_path=OUT_DIR / "seg_around_6717.npz",
        enable_early_fire_reaction=True,
        enable_resolved_exchange_eval=True,
        enable_resolved_decisive_amplify=True,
        enable_resolved_live_defender=True,
        enable_resolved_live_defender_strict=True,
        enable_per_side_settled=True,
        disable_score_lead_bias=True,
        disable_pressure=True,
        enable_resolved_kill_override=True,
    )
    out_path = OUT_DIR / "records_t6717.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"records: {len(records)} -> {out_path}")
    near = [r for r in records if r["t_sec"] is not None and 6690.0 <= r["t_sec"] <= 6745.0]
    print(f"near t=6717.5 window: {len(near)} records")
    for r in near:
        print(f"t={r['t_sec']:.2f} adv={r['adv']:.1f} p1={r['p1']:.3f} "
              f"fc_p1={r['snap_forecast_p1']} fc_p2={r['snap_forecast_p2']} "
              f"pend_p1={r['snap_pending_p1']} pend_p2={r['snap_pending_p2']} "
              f"board_ojama(1p,2p)=({r['f1_board_ojama_count']:.0f},{r['f2_board_ojama_count']:.0f}) "
              f"diff_board_ojama={r['diff_board_ojama_count']:.3f} "
              f"conn_pair(1p,2p)=({r['f1_conn_pair_count']:.0f},{r['f2_conn_pair_count']:.0f}) "
              f"diff_conn_pair={r['diff_conn_pair_count']:.2f} "
              f"cur_max_chain(1p,2p)=({r['f1_current_max_chain']:.3f},{r['f2_current_max_chain']:.3f}) "
              f"diff_cur_max_chain={r['diff_current_max_chain']:.3f} "
              f"drivers={r['drivers']}")


if __name__ == "__main__":
    main()
