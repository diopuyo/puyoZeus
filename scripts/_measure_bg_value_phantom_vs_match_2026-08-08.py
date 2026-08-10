"""幻盤面と正常な試合中で、 盤面背景の明るさ (HSV V) がどれだけ分離するか測る.

目的: 認識側の根治 (非試合画面を入口で弾く) の閾値を、 シーン逆算ではなく
**実測分布**から決めるための材料を作る (過学習防止則、
[[feedback-overfitting-awareness-2026-08-04]])。

背景:
  `scripts/collect_boards_lean.py` は `force_in_match=True` で
  MatchStateDetector を無効化しているため、 対戦カード紹介・ロビー・順位表
  画面でも盤面が記録される。 MatchStateDetector 自体は
  「盤面上部の背景が暗ければ試合中」 という原理で、 既定閾値
  IN_MATCH_V_MAX=170 (試合中 実測 69-156、 非試合 実測 178-231) を持つ。
  そのまま force_in_match=False にすると、 試合中の混雑時に誤って試合外と
  判定される旧バグが再発するため、 「確実に非試合」 と言える高い側の
  ハード閾値を別に置きたい。 その値を決めるための実測。

方法:
  幻盤面と判定されたスナップショットの時刻 (陽性側) と、 同じ動画で幻でない
  スナップショットの時刻 (陰性側) をサンプリングし、 実フレームの bg_value を
  測って分布を比較する。

出力: data/verify/phase_l_quality_gate_2026-08-07/bg_value_phantom_vs_match_2026-08-08.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board_quality import phantom_board_mask  # noqa: E402
from src.match_state import MatchStateDetector  # noqa: E402

NPZ_DIR = _ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-07"
OUT_TSV = (
    _ROOT / "data" / "verify" / "phase_l_quality_gate_2026-08-07"
    / "bg_value_phantom_vs_match_2026-08-08.tsv"
)
# WSL 側の動画置き場 (regen ジョブと同じ規約)。
FRAMES_DIR = Path.home() / "frames"
# 測定対象動画 (幻盤面が多い順に少数を選ぶ。 全動画を回すのは高コスト)。
TARGET_VIDEOS: tuple[str, ...] = ("c28", "c117", "32", "c12", "c27")
# 各動画・各クラスからサンプリングするフレーム数。
SAMPLES_PER_CLASS: int = 40


def _sample_times(npz_path: Path) -> tuple[list[float], list[float]]:
    """(幻盤面の時刻, 幻でない時刻) をサンプリングして返す."""
    d = np.load(npz_path, allow_pickle=True)
    grids = d["grids"]
    t_sec = np.asarray(d["t_sec"], dtype=float)
    mask = phantom_board_mask(grids)
    pos = t_sec[mask]
    neg = t_sec[~mask]
    rng_pos = np.linspace(0, len(pos) - 1, min(SAMPLES_PER_CLASS, len(pos)))
    rng_neg = np.linspace(0, len(neg) - 1, min(SAMPLES_PER_CLASS, len(neg)))
    return (
        [float(pos[int(i)]) for i in rng_pos] if len(pos) else [],
        [float(neg[int(i)]) for i in rng_neg] if len(neg) else [],
    )


def _measure_video(vid: str, det: MatchStateDetector) -> list[tuple[str, str, float, float]]:
    """1 動画分の (video, class, t_sec, bg_value) を返す."""
    npz = NPZ_DIR / f"{vid}.npz"
    src = FRAMES_DIR / f"video_{vid}.mp4"
    if not npz.exists() or not src.exists():
        print(f"  skip {vid} (npz={npz.exists()} mp4={src.exists()})")
        return []
    pos_t, neg_t = _sample_times(npz)
    cap = cv2.VideoCapture(str(src))
    rows: list[tuple[str, str, float, float]] = []
    for cls, times in (("phantom", pos_t), ("normal", neg_t)):
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if frame.shape[1] != 1920 or frame.shape[0] != 1080:
                frame = cv2.resize(frame, (1920, 1080))
            try:
                res = det.detect(frame)
            except Exception:
                continue
            rows.append((vid, cls, t, float(res.bg_value)))
    cap.release()
    return rows


def main() -> int:
    det = MatchStateDetector.load_default()
    all_rows: list[tuple[str, str, float, float]] = []
    for vid in TARGET_VIDEOS:
        print(f"[bg] {vid} ...")
        all_rows.extend(_measure_video(vid, det))
    if not all_rows:
        print("測定できたフレームが無い")
        return 1
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    lines = ["video_id\tclass\tt_sec\tbg_value"]
    lines += [f"{v}\t{c}\t{t:.2f}\t{b:.2f}" for v, c, t, b in all_rows]
    OUT_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")

    arr_p = np.array([b for _, c, _, b in all_rows if c == "phantom"])
    arr_n = np.array([b for _, c, _, b in all_rows if c == "normal"])
    print(f"\n幻盤面 n={len(arr_p)}  正常 n={len(arr_n)}")
    for name, a in (("幻盤面", arr_p), ("正常", arr_n)):
        if not len(a):
            continue
        print(
            f"  {name}: min={a.min():.1f} p5={np.percentile(a, 5):.1f} "
            f"median={np.median(a):.1f} p95={np.percentile(a, 95):.1f} "
            f"max={a.max():.1f}"
        )
    if len(arr_p) and len(arr_n):
        print(f"\n  正常の最大 = {arr_n.max():.1f} / 幻盤面の最小 = {arr_p.min():.1f}")
        print("  (この 2 値が離れているほど、ハード閾値を安全に置ける)")
    print(f"\n出力: {OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
