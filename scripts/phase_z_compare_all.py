"""Phase Z 全バリエーションの cross_video 比較。

v16 / v17 / v17b / v16+HSV / v16+anomaly / v16+HSV+anomaly などを
存在するものだけ集計表示。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_compare_all
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


VARIANTS: list[tuple[str, str]] = [
    ("v16", "cross_video"),
    ("v16c", "cross_video_v16_clean"),
    ("v17", "cross_video_v17"),
    ("v17b", "cross_video_v17b"),
    ("v18", "cross_video_v18"),
    ("v16+H", "cross_video_v16_hsv"),
    ("v16+A", "cross_video_v16_anomaly"),
    ("v16+A2", "cross_video_v16_anomaly_v2"),
    ("v16+HA", "cross_video_v16_hsv_anomaly"),
    ("v16+En", "cross_video_v16_ensemble"),
    ("v16+R", "cross_video_v16_autoroi"),
    ("v16+Co", "cross_video_v16_conn"),
    ("v16+St", "cross_video_v16_stability"),
    ("v16+PV", "cross_video_v16_per_video"),
    # 旧 sweep (env propagation バグで全て default 動作 = 結果無効)
    ("ES=80", "cross_video_v16_emS80"),
    ("ES=50", "cross_video_v16_emS50"),
    ("VS=80", "cross_video_v16_vS80"),
    ("VS=120", "cross_video_v16_vS120"),
    ("PV+R", "cross_video_v16_pv_roi"),
    ("TV=5", "cross_video_v16_tv5"),
    # 新 sweep (env propagation 修正後の真結果)
    ("emS40c", "cross_video_v16_emS40_clean"),
    ("emS50c", "cross_video_v16_emS50_clean"),
    ("emS70c", "cross_video_v16_emS70_clean"),
    ("emS80c", "cross_video_v16_emS80_clean"),
    ("vS80c", "cross_video_v16_vS80_clean"),
    ("vS90c", "cross_video_v16_vS90_clean"),
    ("vS110c", "cross_video_v16_vS110_clean"),
    ("vS130c", "cross_video_v16_vS130_clean"),
    ("tv2c", "cross_video_v16_tv2_clean"),
    ("tv4c", "cross_video_v16_tv4_clean"),
    ("tv5c", "cross_video_v16_tv5_clean"),
]


def load_summary(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                out[r["video"]] = {
                    "hard": int(r["hard"]),
                    "acc": float(r["est_accuracy"]),
                }
            except (KeyError, ValueError):
                pass
    return out


def main() -> int:
    base = _ROOT / "data/verify/phase_z_review"
    summaries: dict[str, dict] = {}
    available = []
    for label, dirname in VARIANTS:
        path = base / dirname / "summary.tsv"
        s = load_summary(path)
        if s:
            summaries[label] = s
            available.append(label)
            print(f"[loaded] {label}: {len(s)} 動画 ({dirname})")
        else:
            print(f"[miss]   {label}: not found at {dirname}")
    if not summaries:
        return 1

    # 動画一覧 (v16 ベース)
    base_label = "v16" if "v16" in summaries else available[0]
    videos = sorted(summaries[base_label].keys())

    print()
    # ヘッダ
    header = f"{'video':<5}"
    for label in available:
        header += f" {label:<8}"
    if base_label in available:
        for label in available:
            if label != base_label:
                header += f" Δ{label:<7}"
    print(header)
    print("-" * len(header))

    # 集計用
    totals: dict[str, list[float]] = {l: [] for l in available}
    deltas: dict[str, list[float]] = {
        l: [] for l in available if l != base_label
    }
    for video in videos:
        line = f"{video:<5}"
        accs = {}
        for label in available:
            d = summaries[label].get(video, {"acc": None, "hard": 0})
            accs[label] = d["acc"]
            if d["acc"] is not None:
                line += f" {d['acc']:<8.3f}"
                totals[label].append(d["acc"])
            else:
                line += f" {'-':<8}"
        # delta
        if base_label in accs and accs[base_label] is not None:
            for label in available:
                if label == base_label:
                    continue
                if accs[label] is None:
                    line += f" {'-':<8}"
                    continue
                d = accs[label] - accs[base_label]
                sign = "+" if d >= 0 else ""
                line += f" {sign}{d:<7.3f}"
                deltas[label].append(d)
        print(line)

    print("-" * len(header))
    print("\n=== 全動画平均 ===")
    for label in available:
        if totals[label]:
            avg = sum(totals[label]) / len(totals[label])
            print(f"  {label:<7}: {avg:.3f}%")
    if deltas:
        print(f"\n=== {base_label} 比改善幅 ===")
        for label in available:
            if label == base_label:
                continue
            if deltas[label]:
                avg = sum(deltas[label]) / len(deltas[label])
                n_imp = sum(1 for d in deltas[label] if d > 0)
                n_wor = sum(1 for d in deltas[label] if d < 0)
                print(
                    f"  {label:<7}: {avg:+.3f}pt "
                    f"(改善 {n_imp} / 悪化 {n_wor})"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
