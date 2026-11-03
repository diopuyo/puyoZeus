"""描画・検収の試験用の合成時系列 (実データではない。レビュー器の動作確認専用)。"""
from __future__ import annotations

import math
from typing import Any

from src.b3_review_input import empty_row

SYNTH_VIDEO_ID = "synthetic"
EXCHANGE_START_SEC, EXCHANGE_END_SEC = 8.0, 14.0
DEAD_FROM_SEC = 17.0


def synth_rows(duration_sec: float = 20.0, step_sec: float = 0.5) -> list[dict[str, Any]]:
    """静止 → 撃ち合い (a=1P 連鎖中) → 2P 死亡証明、の 20 秒。欠測 (None) も混ぜる。"""
    rows = []
    for i in range(int(duration_sec / step_sec) + 1):
        t = i * step_sec
        r = empty_row()
        exch = EXCHANGE_START_SEC <= t < EXCHANGE_END_SEC
        p_s = 0.5 + 0.25 * math.sin(t / 3)
        p_e = min(0.95, 0.5 + 0.05 * (t - EXCHANGE_START_SEC)) if exch else None
        sel = p_e if exch else p_s
        r.update(t_sec=t, frame_idx=int(t * 60), video_id=SYNTH_VIDEO_ID, game_idx=1,
                 phase=min(2, int(t // 8)), p_static=p_s, p_exchange=p_e, p_selected=sel,
                 p_display=0.7 * sel + 0.3 * 0.5, p_ref=0.5 + 0.2 * math.cos(t / 4),
                 p_ref_display=0.5 + 0.18 * math.cos(t / 4),
                 mode="exchange" if exch else "static",
                 state_reason="1P連鎖中: 撃ち合い" if exch else "両者構築", missing_code=0,
                 sl_rate=70.0)
        if exch:
            r.update(p1_active=1, p1_H_med=3, p1_H_fast=4, p1_Q_med=18.0, p1_Q_fast=30.0,
                     p1_I_sl=24.0, p1_I_pred=20.0, p1_I_icon=22.0, p1_M=-6.0, p1_recv=6.0,
                     p1_send=0.0, p1_a_rem=4.2, p1_b_busy=0.0, p1_window=4.2, p1_req_h=4,
                     p1_done_h=4, p1_trunc=0, p1_hand_med=1.4, p1_hand_fast=1.0,
                     p1_sl_pending=0.0, p1_sl_generated=24.0, p1_sl_offset=0.0,
                     p1_sl_dropped=0.0, p1_sl_carry=35.0, p1_icon_corr=2.0, p1_icon_raw=2.0,
                     p1_mk_ta=0, p1_mk_sl=0, p2_sl_pending=24.0, p2_icon_corr=22.0, p2_mk_ta=8)
        else:
            r.update(p1_mk_ta=8, p2_mk_ta=8)  # 連鎖なし: 時間軸は該当なし
        r["p1_death_state"] = "none"
        r["p2_death_state"] = "proven_dead" if t >= DEAD_FROM_SEC else ("suspect" if t >= 15 else "none")
        r["p2_death_basis"] = "相殺不能の予告72個以上" if t >= DEAD_FROM_SEC else None
        if 3.0 <= t < 4.0:   # 欠測区間 (補間されず途切れること)
            r.update(p_display=None, p_ref_display=None, missing_code=5, missing_note="2P盤面未観測")
        rows.append(r)
    return rows
