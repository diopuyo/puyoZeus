"""Phase Z: 全動画 (v01-v19) で連続 frame 自動評価を回し横展開計測。

各動画について match_boundaries_* から最も適切な試合区間を取得し、
最初の 30s で phase_z_review_ui を実行 → labels.csv 生成。
phase_z_continuous_eval で hard violations を集計し、推定 accuracy を出力。

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_z_cross_video \
        --videos 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()


def get_match_2(video_id: int) -> tuple[float, float] | None:
    """matches.tsv から試合 2 (or 1) の (start, end) を返す。"""
    candidates = [
        _ROOT
        / f"data/verify/match_boundaries_v5/video_{video_id:02d}/matches.tsv",
        _ROOT
        / f"data/verify/match_boundaries_v4/video_{video_id:02d}/matches.tsv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
            if len(rows) >= 2:
                row = rows[1]  # 試合 2
            elif rows:
                row = rows[0]
            else:
                continue
            return (float(row["start_sec"]), float(row["end_sec"]))
    return None


def run_review_ui(
    video_id: int, start: float, end: float, out_dir: Path,
    cnn_model: str = "models/cnn_phase_u_v16.pt",
    use_online_hsv: bool = False,
    use_cell_anomaly: bool = False,
    use_hsv_anomaly: bool = False,
    ensemble: bool = False,
    auto_roi: bool = False,
    use_connectivity: bool = False,
    use_stability: bool = False,
) -> bool:
    cmd = [
        "./venv/bin/python", "-m", "scripts.phase_z_review_ui",
        "--video", f"data/frames/video_{video_id:02d}.mp4",
        "--start", str(start),
        "--end", str(end),
        "--bg-fp-time", str(start),
        "--out-dir", str(out_dir),
        "--cnn-model", cnn_model,
    ]
    if use_online_hsv:
        cmd.append("--use-online-hsv")
    if use_cell_anomaly:
        cmd.append("--use-cell-anomaly")
    if use_hsv_anomaly:
        cmd.append("--use-hsv-anomaly")
    if ensemble:
        cmd.append("--ensemble")
    if auto_roi:
        cmd.append("--auto-roi")
    if use_connectivity:
        cmd.append("--use-connectivity")
    if use_stability:
        cmd.append("--use-stability")
    # 修正: 完全上書きでなく親 env を継承 (PHASE_Z_* env var を subprocess に伝える)
    import os
    env = {**os.environ, "PYTHONPATH": "."}
    try:
        result = subprocess.run(
            cmd, cwd=str(_ROOT), env={**env, "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=600,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def parse_eval_output(labels_path: Path) -> dict | None:
    """phase_z_continuous_eval を呼んで主要指標を抽出。"""
    cmd = [
        "./venv/bin/python", "-m", "scripts.phase_z_continuous_eval",
        "--labels", str(labels_path),
    ]
    import os
    env = {**os.environ, "PYTHONPATH": ".", "PATH": "/usr/bin:/bin"}
    try:
        result = subprocess.run(
            cmd, cwd=str(_ROOT), env=env,
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return None
        out = result.stdout
        # 推定 accuracy 行をパース
        info = {}
        for line in out.splitlines():
            if "全 cell:" in line:
                info["total"] = int(line.split(":")[-1].strip())
            elif "連鎖中 cell" in line:
                # "連鎖中 cell (除外): N (P%)"
                parts = line.split(":")[-1].strip()
                info["chain"] = int(parts.split()[0])
            elif line.startswith("clean") or "clean (" in line:
                pass
            elif line.startswith("hard violations"):
                info["hard"] = int(line.split(":")[-1].strip().split()[0])
            elif "推定 accuracy" in line:
                # "推定 accuracy (...): 98.477%"
                pct = line.split(":")[-1].strip().rstrip("%")
                info["accuracy"] = float(pct)
        return info
    except subprocess.TimeoutExpired:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--cnn-model", default="models/cnn_phase_u_v16.pt",
    )
    parser.add_argument("--out-suffix", default="")
    parser.add_argument(
        "--use-online-hsv", action="store_true",
        help="Z-3I OnlineHsvCalibrator を有効化",
    )
    parser.add_argument(
        "--use-cell-anomaly", action="store_true",
        help="Z-3J CellAnomalyDetector を有効化",
    )
    parser.add_argument(
        "--use-hsv-anomaly", action="store_true",
        help="Z-3J' CellHsvAnomalyDetector を有効化",
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Z-X: v16+v17b Multi-CNN ensemble",
    )
    parser.add_argument(
        "--auto-roi", action="store_true",
        help="D: ROI offset auto-calibration",
    )
    parser.add_argument(
        "--use-connectivity", action="store_true",
        help="A: 孤立 cell を周囲色に補正",
    )
    parser.add_argument(
        "--use-stability", action="store_true",
        help="G: cell HSV σ stability tracker",
    )
    parser.add_argument(
        "--per-video-model", action="store_true",
        help="E: 動画別ベスト model 自動選択 (v16 or v17b)",
    )
    args = parser.parse_args()

    video_ids = [int(s) for s in args.videos.split(",") if s]
    if args.out_suffix:
        out_root = (
            _ROOT
            / f"data/verify/phase_z_review/cross_video_{args.out_suffix}"
        )
    else:
        out_root = _ROOT / "data/verify/phase_z_review/cross_video"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[list] = []
    for vid in video_ids:
        match = get_match_2(vid)
        if match is None:
            print(f"[skip] v{vid:02d}: matches not found")
            continue
        match_start, match_end = match
        start = match_start
        end = min(match_start + args.duration, match_end)
        out_dir = out_root / f"v{vid:02d}_m_{int(start)}_{int(end)}"
        print(
            f"[run] v{vid:02d} match {start:.0f}-{end:.0f}s → "
            f"{to_windows_path(out_dir)}"
        )
        # E: 動画別 model 選択
        if args.per_video_model:
            from src.per_video_model_selector import select_model_for_video
            cnn_model = select_model_for_video(
                f"data/frames/video_{vid:02d}.mp4",
            )
        else:
            cnn_model = args.cnn_model
        ok = run_review_ui(
            vid, start, end, out_dir,
            cnn_model=cnn_model,
            use_online_hsv=args.use_online_hsv,
            use_cell_anomaly=args.use_cell_anomaly,
            use_hsv_anomaly=args.use_hsv_anomaly,
            ensemble=args.ensemble,
            auto_roi=args.auto_roi,
            use_connectivity=args.use_connectivity,
            use_stability=args.use_stability,
        )
        if not ok:
            print(f"[fail] v{vid:02d}: review_ui failed")
            summary_rows.append([
                f"v{vid:02d}", start, end, "FAIL", "FAIL", "FAIL",
            ])
            continue
        info = parse_eval_output(out_dir / "labels.csv")
        if info is None:
            print(f"[fail] v{vid:02d}: eval failed")
            summary_rows.append([
                f"v{vid:02d}", start, end, "FAIL", "FAIL", "FAIL",
            ])
            continue
        summary_rows.append([
            f"v{vid:02d}", start, end,
            info.get("total", 0),
            info.get("hard", 0),
            f"{info.get('accuracy', 0.0):.3f}",
        ])
        print(
            f"  total={info.get('total')} hard={info.get('hard')} "
            f"acc={info.get('accuracy', 0):.3f}%"
        )

    summary_path = out_root / "summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "video", "start", "end", "total_cells", "hard", "est_accuracy",
        ])
        for r in summary_rows:
            writer.writerow(r)
    print()
    print(f"summary: {to_windows_path(summary_path)}")
    accs = [
        float(r[5]) for r in summary_rows
        if isinstance(r[5], str) and r[5] != "FAIL"
    ]
    if accs:
        print(f"平均推定 accuracy: {sum(accs) / len(accs):.3f}%")
        print(f"最低推定 accuracy: {min(accs):.3f}%")
        print(f"最高推定 accuracy: {max(accs):.3f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
