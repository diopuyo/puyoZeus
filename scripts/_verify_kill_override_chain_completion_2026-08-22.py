"""修正①②④ (kill_override 連鎖完走後是正 / EarlyFireTracker finalize連動クリア /
主因表示への安全弁理由明示) の短区間自己検収 (2026-08-22)。

対象: t=6717.5s (1Pが15連鎖 [50x386] 撃っている最中に2P有利100%と誤表示される
本命エピソード)。

設計:
  - 「旧コードの誤り」は既存の本番レンダ dump
    (data/verify/zenchi_render_2026-08-21/seg08_6131.6_7033.6.npz、
    現行本番フラグそのままの実測) に既に入っているため、旧コードの再実行は
    しない (再レンダ節約)。
  - --per-side-settled 下の会計状態は試合開始 (セグメント境界=6131.6s、
    _render_zenchi_8seg_2026-08-21.sh の分割根拠と同一=試合開始で状態機械が
    リセットされるため前試合の状態を引き継ぐ必要がない) から連続して駆動する
    必要があるため、t=6717.5 単独を切り出さず 6131.6 から通しで処理する。
  - FIXED (新フラグ3つ ON) と GUARD_OFF (新フラグ全て既定 False = 現行コード
    そのまま) を同一区間・同一コードベースで2回実行し、
      (1) FIXED で adv_ema の符号が adv_raw と一致すること (根治確認)
      (2) GUARD_OFF が旧本番dumpの実測値と (ほぼ) 一致すること
          (=新コードでもフラグ既定 False なら退行していない陽性対照)
    を数値で確認する。render=False (動画書き出しなし、長時間レンダ回避)。
"""
from __future__ import annotations

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
OUT_DIR = PROJECT_ROOT / "logs/_verify_kill_override_chain_completion_2026-08-22"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_NPZ = PROJECT_ROOT / "data/verify/zenchi_render_2026-08-21/seg08_6131.6_7033.6.npz"

SEG_START = 6131.6      # セグメント境界 (試合開始)
SEG_END = 6740.0        # 本命アンカー t=6717.5 を少し過ぎたところで打ち切り
WARMUP = 30.0
ANCHOR = 6717.500

PRODUCTION_KWARGS = dict(
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
)


def _nearest(d, t_target: float) -> int:
    return int(np.argmin(np.abs(d["t_sec"] - t_target)))


def run(label: str, extra_kwargs: dict) -> Path:
    dump_path = OUT_DIR / f"{label}.npz"
    vao.generate(
        video=VIDEO, out=OUT_DIR / f"_dummy_{label}.mp4",
        max_sec=0.0, sample_interval=0.0,
        start_sec=SEG_START, end_sec=SEG_END, warmup_sec=WARMUP,
        model_dir=MODEL_DIR, layout="panel", panel_subtitle_h=0,
        render=False, dump_timeline_path=dump_path,
        **PRODUCTION_KWARGS, **extra_kwargs,
    )
    return dump_path


def show(label: str, d, t_target: float) -> None:
    i = _nearest(d, t_target)
    print(
        f"[{label}] t_target={t_target:.3f} -> t={d['t_sec'][i]:.3f} "
        f"adv_raw={d['adv_raw'][i]:+.2f} adv_ema={d['adv_ema'][i]:+.2f} "
        f"p1={d['p1'][i]*100:.1f}% p1_raw={d['p1_raw'][i]*100:.1f}% "
        f"pending_p1={int(d['pending_p1'][i])} pending_p2={int(d['pending_p2'][i])} "
        f"room1={int(d['room1'][i])} room2={int(d['room2'][i])} "
        f"state1={d['state1'][i]} state2={d['state2'][i]}"
    )


def main() -> None:
    baseline = np.load(BASELINE_NPZ, allow_pickle=True)
    show("baseline(旧本番dump、再実行なし)", baseline, ANCHOR)

    print("\n--- FIXED (修正①②④ ON) ---")
    fixed = np.load(run("fixed_on", dict(
        enable_kill_override_chain_completion=True,
        enable_kill_override_attribution=True,
        enable_early_fire_clear_on_finalize=True,
    )), allow_pickle=True)
    show("FIXED", fixed, ANCHOR)

    print("\n--- GUARD_OFF (新フラグ全て既定 False = 現行コードそのまま) ---")
    guard_off = np.load(run("guard_off", dict()), allow_pickle=True)
    show("GUARD_OFF", guard_off, ANCHOR)

    i_fixed = _nearest(fixed, ANCHOR)
    i_guard = _nearest(guard_off, ANCHOR)
    i_base = _nearest(baseline, ANCHOR)
    print("\n=== 判定 ===")
    print("1) FIXED: adv_ema の符号が adv_raw と一致するか ->",
          (fixed["adv_raw"][i_fixed] > 0) == (fixed["adv_ema"][i_fixed] > 0))
    print("2) GUARD_OFF は旧本番dumpと同型の誤り (符号不一致) を再現するか ->",
          (guard_off["adv_raw"][i_guard] > 0) != (guard_off["adv_ema"][i_guard] > 0))
    print("3) 旧本番dump自体も符号不一致 (問題の実在確認) ->",
          (baseline["adv_raw"][i_base] > 0) != (baseline["adv_ema"][i_base] > 0))


if __name__ == "__main__":
    main()
