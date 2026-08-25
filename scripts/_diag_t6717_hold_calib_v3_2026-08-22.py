"""t=6717.5 で「CHAIN保持時間の実測較正配線 (根治②)」+「累積OFF」の効果を
計装で確認する (2026-08-22、user判断: 累積は対症療法のため外し、既存の
実測較正式 2.61+1.17×N を配線した構成だけで測る)。

受け入れ条件 (この短区間チェック用):
  1. ChainEvent.trigger_sec が連鎖中に変わらないこと (1つの連鎖で1イベント)
  2. 連鎖数 (chain_count) が正しく検知されるか (cc=5 から変わるか)。
     保持時間を伸ばしても連鎖数の誤検知自体は直らない可能性があるため、
     「断片化は消えるが連鎖数は依然5」の場合は火力が足りるかも確認する。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
cv2.setNumThreads(1)

import scripts.visualize_advantage_overlay as vao  # noqa: E402

VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
MODEL_DIR = PROJECT_ROOT / "data/verify/retrain_model62_2026-08-21"
OUT_DIR = PROJECT_ROOT / "logs/_diag_t6717_hold_calib_v3_2026-08-22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# [高速化] t=6717.5 は game_idx=9 の試合内で、その試合自体は t=6664.17 に
# score大幅減少検知でリセットされて始まっている。今回のチェック対象は
# 「保持時間が伸びて再トリガーが起きなくなるか」なので、_diag_t6717_
# kill_override_root_cause_2026-08-22 (旧設計) で確認済みの通り、この
# 短縮開始点は全編走査 (t=6131.6起点) と同一の trigger_sec 系列を再現する
# (2026-08-22 実測、v2/v3のtrigger_sec完全一致で検証済み)。
SEG_START = 6664.17
SEG_END = 6725.0
WARMUP = 30.0
WATCH_LO, WATCH_HI = 6700.0, 6720.0

_STATE: dict = {"t": None}
_detail: list[str] = []
_trigger_history_1p: list[float] = []


def main() -> None:
    orig_drive_ojama = vao._drive_ojama
    orig_accum_update = vao.ChainGenerationAccumulator.update
    orig_inputs_fn = vao._kill_override_chain_completion_inputs
    orig_kill_override = vao.kill_override

    def patched_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw):
        _STATE["t"] = t
        return orig_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw)

    def patched_accum_update(self, r_p1, r_p2, elapsed_sec):
        result = orig_accum_update(self, r_p1, r_p2, elapsed_sec)
        t = _STATE.get("t")
        ev1 = getattr(r_p1, "chain_event", None)
        if ev1 is not None:
            if not _trigger_history_1p or _trigger_history_1p[-1] != ev1.trigger_sec:
                _trigger_history_1p.append(ev1.trigger_sec)
        if t is not None and WATCH_LO <= t <= WATCH_HI:
            gen1, before1, gen2, before2 = result
            _detail.append(
                f"[accum] t={t:.3f} state1={r_p1.state} accumulate={self._accumulate} "
                f"ev1_trigger={'None' if ev1 is None else f'{ev1.trigger_sec:.3f} cc={ev1.chain_count} ts={ev1.total_score}'} "
                f"gen1={gen1:.2f} last_trigger1={self._last_trigger['1p']}"
            )
        return result

    def patched_inputs_fn(snap, b1, b2, room1, room2, gen1, before1, gen2, before2):
        t = _STATE.get("t")
        result = orig_inputs_fn(snap, b1, b2, room1, room2, gen1, before1, gen2, before2)
        if t is not None and WATCH_LO <= t <= WATCH_HI:
            r1, r2, p1, p2 = result
            _detail.append(
                f"  [inputs] t={t:.3f} gen1={gen1:.2f} gen2={gen2:.2f} "
                f"snap.pending_p1={snap.pending_p1} room1_in={room1} "
                f"-> room1_eff={r1} room2_eff={r2} pending1_eff={p1} pending2_eff={p2}"
            )
        return result

    def patched_kill_override(adv, inc1, inc2, room1, room2):
        t = _STATE.get("t")
        result = orig_kill_override(adv, inc1, inc2, room1, room2)
        if t is not None and WATCH_LO <= t <= WATCH_HI:
            l1 = inc1 / max(vao.KILL_ROOM_FLOOR, room1) if inc1 >= vao.KILL_MIN_PENDING else 0.0
            l2 = inc2 / max(vao.KILL_ROOM_FLOOR, room2) if inc2 >= vao.KILL_MIN_PENDING else 0.0
            _detail.append(
                f"  [kill_override] t={t:.3f} adv_in={adv:+.2f} inc1={inc1} inc2={inc2} "
                f"room1={room1} room2={room2} l1={l1:.3f} l2={l2:.3f} "
                f"mag={abs(l1-l2):.3f} KILL_RATIO_FULL={vao.KILL_RATIO_FULL} "
                f"-> adv_out={result:+.2f}"
            )
        return result

    vao._drive_ojama = patched_drive_ojama
    vao.ChainGenerationAccumulator.update = patched_accum_update
    vao._kill_override_chain_completion_inputs = patched_inputs_fn
    vao.kill_override = patched_kill_override
    try:
        vao.generate(
            video=VIDEO, out=OUT_DIR / "_dummy_v3.mp4",
            max_sec=0.0, sample_interval=0.0,
            start_sec=SEG_START, end_sec=SEG_END, warmup_sec=WARMUP,
            model_dir=MODEL_DIR, layout="panel", panel_subtitle_h=0,
            render=False, dump_timeline_path=None,
            enable_early_fire_reaction=True,
            enable_per_side_settled=True,
            disable_score_lead_bias=True,
            disable_pressure=True,
            enable_counter_reach=True,
            normalize_fps_30=True,
            use_production_recognition=True,
            resize_1080p=True,
            enable_resolved_live_defender_strict=True,
            enable_resolved_kill_override=True,
            enable_resolved_exchange_eval=True,
            enable_resolved_decisive_amplify=True,
            enable_resolved_live_defender=True,
            force_in_match=False,
            enable_kill_override_chain_completion=True,
            enable_kill_override_chain_gen_accumulate=False,  # 既定 (対症療法回避)
            chain_hold_base_sec=2.61,
            chain_hold_per_step_sec=1.17,
        )
    finally:
        vao._drive_ojama = orig_drive_ojama
        vao.ChainGenerationAccumulator.update = orig_accum_update
        vao._kill_override_chain_completion_inputs = orig_inputs_fn
        vao.kill_override = orig_kill_override

    out_path = OUT_DIR / "detail_v3.log"
    out_path.write_text("\n".join(_detail), encoding="utf-8")
    print(f"[detail] {len(_detail)} 行 -> {out_path}")
    print(f"[trigger_history_1p] distinct trigger_sec seen (in-order, dedup consecutive): "
          f"{_trigger_history_1p}")


if __name__ == "__main__":
    main()
