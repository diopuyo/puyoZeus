"""区間並列レンダの成立性 (2026-08-21、coordinator依頼)。

同じ合計ワークロード (N本 x 20秒区間) を並列度 P (=2/4/6/8) で流し、
「全区間完了までの壁時間」からスループット (処理した動画秒数/壁秒数) を
求める。ディスクI/O (動画読込・書込) を含む**本物のレンダ**で測る
(--no-render は使わない。認識+パネル描画+エンコード全部を含むのが目的)。

区間はキャッシュ効果で有利になりすぎないよう、動画内の異なる時刻から選ぶ
(同一区間を複数回読むとOSページキャッシュが効いて2回目以降が速くなり、
並列度の効果を過大評価する恐れがあるため)。

本体 (scripts/visualize_advantage_overlay.py) は変更せず、subprocess で
CLI を N プロセス起動するだけ (計装なし、壁時間のみ測る)。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VIDEO = "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
MODEL_DIR = "data/verify/retrain_model62_2026-08-21"
PYTHON = str(PROJECT_ROOT / "venv" / "bin" / "python")

# 動画内の異なる時刻 (キャッシュ効果を避けるため分散させる)。試合境界
# (regionA/B/Cで実測済み) の近くを避け、単純な等間隔スタートにする
# (区間の中身の難易度差は今回の関心の外)。
SEGMENT_STARTS = [200.0, 900.0, 1800.0, 2600.0, 3450.0, 4200.0, 5100.0, 6100.0]
SEGMENT_DUR = 15.0


def _loadavg() -> str:
    try:
        return Path("/proc/loadavg").read_text().strip()
    except OSError:
        return "N/A"


def run_batch(n_segments: int, parallelism: int, out_dir: Path, tag: str) -> dict:
    """n_segments 本を parallelism 並列で流し、全完了までの壁時間を返す。"""
    starts = SEGMENT_STARTS[:n_segments]
    procs: list[subprocess.Popen] = []
    completed_times: list[float] = []
    t_wall0 = time.perf_counter()
    pending = list(starts)
    running: list[tuple[subprocess.Popen, float]] = []
    idx = 0
    while pending or running:
        while pending and len(running) < parallelism:
            s = pending.pop(0)
            out_path = out_dir / f"seg_{tag}_{int(s)}.mp4"
            cmd = [
                PYTHON, "-m", "scripts.visualize_advantage_overlay",
                "--video", VIDEO,
                "--start-sec", str(s), "--end-sec", str(s + SEGMENT_DUR),
                "--layout", "panel", "--no-force-in-match",
                "--model-dir", MODEL_DIR,
                "--out", str(out_path),
            ]
            p = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            running.append((p, s))
            idx += 1
        still_running = []
        for p, s in running:
            if p.poll() is None:
                still_running.append((p, s))
            else:
                completed_times.append(time.perf_counter() - t_wall0)
        running = still_running
        if running or pending:
            time.sleep(0.5)
    wall_total = time.perf_counter() - t_wall0
    total_video_sec = n_segments * SEGMENT_DUR
    throughput = total_video_sec / wall_total if wall_total > 0 else float("nan")
    return {
        "n_segments": n_segments, "parallelism": parallelism,
        "wall_total": wall_total, "throughput_video_sec_per_wall_sec": throughput,
        "completed_times": completed_times,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallelism", default="2,4,6,8",
                     help="カンマ区切りで並列度を連続測定")
    ap.add_argument("--n-segments", type=int, default=8,
                     help="各並列度で流す区間数 (総ワークロードを固定して比較)")
    ap.add_argument("--out-dir", type=Path,
                     default=Path("data/verify/zenchi_parallel_sweep_2026-08-21"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for p in [int(x) for x in args.parallelism.split(",")]:
        print(f"\n[loadavg 開始前 P={p}] {_loadavg()}")
        r = run_batch(args.n_segments, p, args.out_dir, tag=f"p{p}")
        print(f"[loadavg 終了後 P={p}] {_loadavg()}")
        print(f"P={p}: {r['n_segments']}本 x {SEGMENT_DUR:.0f}秒 = "
              f"{r['n_segments']*SEGMENT_DUR:.0f}秒分を壁時間{r['wall_total']:.1f}秒で完了 "
              f"-> スループット {r['throughput_video_sec_per_wall_sec']:.4f} "
              f"(動画秒/壁秒)")
        results.append(r)

    print(f"\n{'=' * 70}\n[まとめ] 総ワークロード={args.n_segments*SEGMENT_DUR:.0f}秒分\n{'=' * 70}")
    print(f"{'並列度':>6} {'壁時間(s)':>10} {'スループット':>14} {'P=2に対する倍率':>16}")
    base = None
    for r in results:
        if base is None:
            base = r["throughput_video_sec_per_wall_sec"]
        ratio = r["throughput_video_sec_per_wall_sec"] / base if base else float("nan")
        print(f"{r['parallelism']:>6} {r['wall_total']:>10.1f} "
              f"{r['throughput_video_sec_per_wall_sec']:>14.4f} {ratio:>16.2f}")


if __name__ == "__main__":
    main()
