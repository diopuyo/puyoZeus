"""`--per-side-settled` 等 未配線5フラグの効果を短区間で測る (2026-08-22)。

## 目的
scripts/_render_zenchi_8seg_2026-08-21.sh の手書き FLAGS には
production_config.advantage_overlay_flags() の採用済みフラグのうち
--per-side-settled/--no-score-lead-bias/--no-pressure/--sample-interval 0/
--resolved-kill-override の5つが欠落していた (配線事故)。この欠落が
判定矛盾 (D0/D1a/D1b) の一部を説明するかを、3アンカー窓で
「BASELINE (欠落構成のまま、修正①②のみ適用)」と
「FULL (5フラグ全て追加)」の dump を作り、
scan_judgment_anomalies.py --from-dump に通して比較する。

フルレンダは行わない (3アンカーの短窓のみ、指示厳守)。
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
OUT_DIR = PROJECT_ROOT / "logs/_verify_per_side_settled_2026-08-22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHORS = [
    ("A_t886", 848.0, 905.0, 28.0),
    ("B_t200", 165.0, 215.0, 30.0),
    ("C_t3228", 3190.0, 3260.0, 30.0),
]

# BASELINE: 修正①②のみ適用した状態 (今回セッション以前の手書き FLAGS 相当。
# --resolved-exchange-eval 系は元々手書きで含まれていたため True のまま)。
BASELINE_KW = dict(
    enable_early_fire_reaction=True,
    enable_resolved_exchange_eval=True,
    enable_resolved_decisive_amplify=True,
    enable_resolved_live_defender=True,
    enable_resolved_live_defender_strict=True,
    # 以下、production_config.py 採用済みなのに欠落していた5フラグ = 全て
    # 既定値 (未適用) のまま。
    enable_per_side_settled=False,
    disable_score_lead_bias=False,
    disable_pressure=False,
    enable_resolved_kill_override=False,
)
FULL_SAMPLE_INTERVAL = 0.0
BASELINE_SAMPLE_INTERVAL = 0.15

# FULL: BASELINE + 欠落していた5フラグを全て追加。
FULL_KW = dict(BASELINE_KW)
FULL_KW.update(
    enable_per_side_settled=True,
    disable_score_lead_bias=True,
    disable_pressure=True,
    enable_resolved_kill_override=True,
)


def _run(window_name: str, start_sec: float, end_sec: float, warmup_sec: float,
         label: str, kw: dict, sample_interval: float) -> Path:
    dump_path = OUT_DIR / f"{window_name}_{label}.npz"
    vao.generate(
        video=VIDEO,
        out=OUT_DIR / f"_dummy_{window_name}_{label}.mp4",
        max_sec=0.0,
        sample_interval=sample_interval,
        start_sec=start_sec,
        end_sec=end_sec,
        warmup_sec=warmup_sec,
        model_dir=MODEL_DIR,
        force_in_match=False,
        layout="panel",
        panel_subtitle_h=0,
        render=False,  # 判定計算のみ (dump取得が目的、描画・エンコード不要)
        dump_timeline_path=dump_path,
        **kw,
    )
    return dump_path


def main() -> None:
    dump_dirs = {"BASELINE": OUT_DIR / "dumps_baseline", "FULL": OUT_DIR / "dumps_full"}
    for d in dump_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for name, s, e, w in ANCHORS:
        print(f"=== {name} (窓 {s}-{e}s, 暖機{w}s) ===")
        p = _run(name, s, e, w, "baseline", BASELINE_KW, BASELINE_SAMPLE_INTERVAL)
        (dump_dirs["BASELINE"] / p.name).write_bytes(p.read_bytes())
        print(f"  [BASELINE] dump -> {dump_dirs['BASELINE'] / p.name}")
        p = _run(name, s, e, w, "full", FULL_KW, FULL_SAMPLE_INTERVAL)
        (dump_dirs["FULL"] / p.name).write_bytes(p.read_bytes())
        print(f"  [FULL] dump -> {dump_dirs['FULL'] / p.name}")

    print("\n=== 走査器 (scan_judgment_anomalies.py --from-dump) ===")
    print(f"BASELINE: python -m scripts.scan_judgment_anomalies "
          f"--from-dump {dump_dirs['BASELINE']}")
    print(f"FULL:     python -m scripts.scan_judgment_anomalies "
          f"--from-dump {dump_dirs['FULL']}")


if __name__ == "__main__":
    main()
