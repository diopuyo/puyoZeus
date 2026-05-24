"""Phase Z 夜間作業の最終化スクリプト。

cross_video 完了後に以下を順次実行:
    1. cross_video の summary.tsv を確認
    2. 各動画の violations を抽出 (phase_z_extract_all_violations.py)
    3. v18_m03 30-60s で最終 accuracy 測定 (Z-3G/3H 反映後)
    4. 結果を docs/HANDOFF_2026-05-01_PHASE_Z.md に追記する素材を生成
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()


def step_1_cross_summary() -> dict | None:
    """cross_video の summary.tsv を読み込み、動画別精度を集計。"""
    summary_path = (
        _ROOT / "data/verify/phase_z_review/cross_video/summary.tsv"
    )
    if not summary_path.exists():
        return None
    rows = []
    with summary_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        return None
    accs = []
    for r in rows:
        try:
            accs.append(float(r["est_accuracy"]))
        except (KeyError, ValueError):
            pass
    return {
        "videos": rows,
        "mean_acc": sum(accs) / len(accs) if accs else 0.0,
        "min_acc": min(accs) if accs else 0.0,
        "max_acc": max(accs) if accs else 0.0,
        "n_videos": len(accs),
    }


def step_2_extract_violations() -> bool:
    """全動画の violations を抽出。"""
    cmd = [
        "./venv/bin/python", "-m", "scripts.phase_z_extract_all_violations",
    ]
    env = {**os.environ, "PYTHONPATH": ".", "PATH": "/usr/bin:/bin"}
    try:
        result = subprocess.run(
            cmd, cwd=str(_ROOT), env=env,
            capture_output=True, text=True, timeout=3600,
        )
        for line in result.stdout.splitlines()[-15:]:
            print(line)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def step_3_v18_final_eval() -> dict | None:
    """v18_m03 30-60s で最新コードでの最終評価。"""
    out_dir = _ROOT / "data/verify/phase_z_review/v18_m03_30_60_final"
    cmd = [
        "./venv/bin/python", "-m", "scripts.phase_z_review_ui",
        "--video", "data/frames/video_18.mp4",
        "--start", "281", "--end", "311",
        "--bg-fp-time", "251",
        "--out-dir", str(out_dir),
    ]
    env = {**os.environ, "PYTHONPATH": ".", "PATH": "/usr/bin:/bin"}
    try:
        result = subprocess.run(
            cmd, cwd=str(_ROOT), env=env,
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"[v18 eval] FAIL: {result.stderr[-200:]}")
            return None
        # 連続評価
        cmd2 = [
            "./venv/bin/python", "-m", "scripts.phase_z_continuous_eval",
            "--labels", str(out_dir / "labels.csv"),
        ]
        result2 = subprocess.run(
            cmd2, cwd=str(_ROOT), env=env,
            capture_output=True, text=True, timeout=60,
        )
        if result2.returncode != 0:
            return None
        info = {}
        for line in result2.stdout.splitlines():
            if "全 cell:" in line:
                info["total"] = int(line.split(":")[-1].strip())
            elif "hard violations:" in line:
                info["hard"] = int(line.split(":")[-1].strip().split()[0])
            elif "推定 accuracy" in line:
                pct = line.split(":")[-1].strip().rstrip("%")
                info["accuracy"] = float(pct)
        return info
    except subprocess.TimeoutExpired:
        return None


def main() -> int:
    print("=" * 60)
    print("Phase Z 夜間最終化")
    print("=" * 60)

    print("\n[step 1] cross_video summary 確認")
    summary = step_1_cross_summary()
    if summary is None:
        print("ERROR: summary.tsv 不在 — cross_video 未完了の可能性")
        return 1
    print(f"  動画数: {summary['n_videos']}")
    print(f"  推定 accuracy: 平均 {summary['mean_acc']:.3f}% / "
          f"最低 {summary['min_acc']:.3f}% / "
          f"最高 {summary['max_acc']:.3f}%")

    print("\n[step 2] 全動画 violations 抽出 (約 30-40 分)")
    ok = step_2_extract_violations()
    print(f"  result: {'OK' if ok else 'FAIL'}")

    print("\n[step 3] v18_m03 最新コード評価 (Z-3G/3H + 高速化反映)")
    info = step_3_v18_final_eval()
    if info:
        print(f"  total={info.get('total')} "
              f"hard={info.get('hard')} "
              f"acc={info.get('accuracy', 0):.3f}%")
    else:
        print("  FAIL")

    print("\n=" * 60)
    print("夜間作業完了。起床時の確認事項:")
    print("- cross_video summary.tsv で動画別推定 acc")
    print("- 各動画 violations.html でレビュー (起床時)")
    print("- v18_m03_30_60_final で Z-3G/3H 反映後の accuracy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
