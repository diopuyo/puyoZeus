"""placement_override 修正版 (--enable-ojama-fall-placement-override,
コミット9dc5e35) の全域バックテスト収集 (2026-08-15)。

## 位置づけ (正直な開示)
user 承認 (2026-08-15) は「フル尺」ではなく **拡張代表サンプル** である。
手元実物16本の合計は約4,454,457 生フレームあり、フル尺 × 2構成を実測
スループット (8並列で7.9生フレーム/秒/プロセス、2026-08-14実測
`scripts/_collect_yardstick_v2_2026-08-14.py` 参照) で処理すると約39.2時間
かかる (前日に同じ16本プールで「非現実的」と判定された規模と一致)。

本スクリプトは動画あたり 序盤/中盤/終盤 の3地点 × `CHUNK_SEC` 秒のみを
全フレーム処理する (間引きなし、既存 yardstick v2 の 90秒/動画に対する拡張版、
`CHUNK_SEC=120.0` で 360秒/動画 = 4倍のカバレッジ)。真のフル尺検証は次回の
148動画再収集 (本フラグ込み) が実質的に兼ねる。

## 計装 (read-only, production コード無変更)
`collect_boards_lean.collect_lean()` は STABLE snapshot のみ npz に記録し、
OJAMA_FALL 滞在中の毎フレーム状態は捨てられる。振動 (0.35秒未満での
OJAMA_FALL<->他state往復) 計測にはこれが必要なため、`_process_side_lean`
(モジュール関数参照) を呼び出し前に薄いラッパーへ差し替え、毎フレームの
(side, t_sec, state) を副作用なく記録してから元の関数へ委譲する
(`scripts/_diag_scene1_oscillation_recheck_2026-08-15.py` と同じ read-only
計装方針)。これにより 1 動画 1 構成につき 1 パスで grids (churn/重力違反/
品質ゲート用) と状態トレース (振動/滞在時間用) の両方を取得できる
(2パスに分けるより計算コストを半分にできる)。

## 使い方
    # ドライバ (全ジョブ実行、8並列)
    PYTHONPATH=. ./venv/bin/python -m \\
        scripts._backtest_placement_override_full_2026-08-15

    # 単一ジョブ (内部で driver が subprocess 起動する用、直接使わない)
    PYTHONPATH=. ./venv/bin/python -m \\
        scripts._backtest_placement_override_full_2026-08-15 --worker \\
        --video <path> --config a --start-sec 100 --max-sec 120 \\
        --out-npz <path> --out-trace <path>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.production_config import (  # noqa: E402
    COLLECT_ONLY_ADOPTED,
    recognition_load_default_kwargs,
)

# ============================
# 定数 (マジックナンバー禁止規約)
# ============================

VIDEO_DIR: Path = Path.home() / "frames"
OUT_ROOT: Path = _ROOT / "data" / "indicators_v2" / "backtest_placement_override_2026-08-15"
LOG_ROOT: Path = _ROOT / "logs" / "backtest_placement_override_2026-08-15"
PYTHON_BIN: Path = _ROOT / "venv" / "bin" / "python"

# 手元実物動画16本 (削除禁止資産、ファイル名は WSL ~/frames/ 実物のまま。
# scripts/_collect_yardstick_v2_2026-08-14.py の VIDEO_FILENAMES と同一)
VIDEO_FILENAMES: tuple[str, ...] = (
    "video_c10.mp4", "video_c11.mp4", "video_c12.mp4", "video_c13.mp4",
    "video_c14.mp4", "video_c15.mp4", "video_c16.mp4", "video_c17.mp4",
    "video_c18.mp4", "video_c19.mp4", "video_c20.mp4", "video_c21.mp4",
    "video_c22.mp4", "video_c23.mp4", "_hold_video_c96.mp4", "video_c109.mp4",
)

# 1地点あたりの処理長 (秒)。全域バックテストの工数見積り誤り (3.7h想定が
# 実際は7.1h) を是正し、user承認の「3.7時間は許容」に収めるため
# 300秒でなく120秒を採用する (2026-08-15 訂正、報告書に明記)。
CHUNK_SEC: float = 120.0
# 動画1本あたりの地点数 (序盤/中盤/終盤)
N_CHUNKS_PER_VIDEO: int = 3
# 地点の相対位置 (yardstick v2 / regen148 と同じ慣例値)
CHUNK_OFFSET_FRACTIONS: tuple[float, ...] = (0.08, 0.45, 0.80)
# 同時実行プロセス数 (8並列時 7.9fps/proc の実測がある構成に合わせる)
MAX_PARALLEL_WORKERS: int = 8

# 構成タグ
CONFIG_A: str = "a"  # 本番採用構成 (regen148オーケストレータと同一)
CONFIG_B: str = "b"  # a + placement_override 修正版


def config_kwargs(tag: str) -> dict[str, float | bool]:
    """構成タグから collect_lean() kwargs を組み立てる。

    構成a = RECOGNITION_ADOPTED + COLLECT_ONLY_ADOPTED (production_config
    単一情報源) + capture_next/enable_phantom_board_guard (実際の148動画
    regen オーケストレータ `scripts/_regen148_orchestrator_2026-08-11.py:
    211-218` が使う本番コマンドと同一の構成)。
    構成b = a + enable_ojama_fall_placement_override=True のみ。
    """
    kwargs: dict[str, float | bool] = dict(recognition_load_default_kwargs())
    for f in COLLECT_ONLY_ADOPTED:
        parts = f.flag.split()
        name = parts[0].lstrip("-").replace("-", "_")
        kwargs[name] = True if len(parts) == 1 else float(parts[1])
    kwargs["capture_next"] = True  # --with-next 相当
    kwargs["enable_phantom_board_guard"] = True
    if tag == CONFIG_B:
        kwargs["enable_ojama_fall_placement_override"] = True
    return kwargs


def probe_duration_sec(path: Path) -> float:
    """動画の全長 [秒] を返す (ヘッダ読取のみ)。"""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if fps > 0 else 0.0


def video_id_of(filename: str) -> str:
    """ファイル名から video_id (例: "c10") を取り出す。"""
    stem = filename.removeprefix("_hold_").removesuffix(".mp4")
    return stem.removeprefix("video_")


@dataclass(frozen=True)
class BacktestJob:
    """1 (動画, 地点, 構成) 分の収集ジョブ。"""

    video_id: str
    video_path: Path
    chunk_idx: int
    start_sec: float
    config_tag: str
    out_npz: Path
    out_trace: Path
    log_path: Path


def build_jobs() -> list[BacktestJob]:
    """16動画 × 3地点 × 2構成 のジョブ一覧を組み立てる。"""
    jobs: list[BacktestJob] = []
    for filename in VIDEO_FILENAMES:
        vid = video_id_of(filename)
        video_path = VIDEO_DIR / filename
        if not video_path.exists():
            print(f"[skip] 動画が無い: {video_path}")
            continue
        duration = probe_duration_sec(video_path)
        if duration <= 0:
            print(f"[skip] 動画を開けない: {video_path}")
            continue
        for k, frac in enumerate(CHUNK_OFFSET_FRACTIONS[:N_CHUNKS_PER_VIDEO]):
            start_sec = max(0.0, frac * duration)
            for tag in (CONFIG_A, CONFIG_B):
                out_dir = OUT_ROOT / tag
                trace_dir = OUT_ROOT / f"{tag}_trace"
                log_dir = LOG_ROOT / tag
                out_dir.mkdir(parents=True, exist_ok=True)
                trace_dir.mkdir(parents=True, exist_ok=True)
                log_dir.mkdir(parents=True, exist_ok=True)
                name = f"{vid}_chunk{k}"
                jobs.append(BacktestJob(
                    video_id=vid, video_path=video_path, chunk_idx=k,
                    start_sec=start_sec, config_tag=tag,
                    out_npz=out_dir / f"{name}.npz",
                    out_trace=trace_dir / f"{name}_trace.npz",
                    log_path=log_dir / f"{name}.log",
                ))
    return jobs


def run_job(job: BacktestJob) -> tuple[str, bool, float]:
    """1ジョブを subprocess (--worker モード) で実行する。"""
    job_name = f"{job.config_tag}/{job.video_id}_chunk{job.chunk_idx}"
    cmd = [
        str(PYTHON_BIN), "-u", "-m",
        "scripts._backtest_placement_override_full_2026-08-15",
        "--worker",
        "--video", str(job.video_path),
        "--config", job.config_tag,
        "--start-sec", str(job.start_sec),
        "--max-sec", str(CHUNK_SEC),
        "--out-npz", str(job.out_npz),
        "--out-trace", str(job.out_trace),
    ]
    start = time.monotonic()
    if job.out_npz.exists() and job.out_trace.exists():
        # 再実行時の重複計算を避ける (中断からの再開に対応)。
        return job_name, True, 0.0
    with job.log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - start
    ok = proc.returncode == 0 and job.out_npz.exists() and job.out_trace.exists()
    print(f"[{'OK' if ok else 'FAIL'}] {job_name} ({elapsed:.1f}s)")
    return job_name, ok, elapsed


def run_driver() -> None:
    """全ジョブを 8並列で実行するドライバ。"""
    jobs = build_jobs()
    n_videos = len({j.video_id for j in jobs})
    print(
        f"[driver] ジョブ数: {len(jobs)} (動画{n_videos}本 × "
        f"最大{N_CHUNKS_PER_VIDEO}地点 × 2構成、CHUNK_SEC={CHUNK_SEC})"
    )
    t0 = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as ex:
        for r in ex.map(run_job, jobs):
            results.append(r)
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"[driver] 完了: {n_ok}/{len(results)} 成功 ({time.monotonic() - t0:.1f}s)")
    if n_ok < len(results):
        print("[driver] 失敗ジョブ:")
        for name, ok, _ in results:
            if not ok:
                print(f"  {name}")


# ============================
# --worker モード: 1 (動画,地点,構成) を実処理する
# ============================

def _run_worker(
    video: Path, config_tag: str, start_sec: float, max_sec: float,
    out_npz: Path, out_trace: Path,
) -> None:
    """1ジョブを実処理する (状態トレース計装込み)。

    collect_boards_lean.collect_lean() は無変更のまま呼び出し、
    _process_side_lean だけを read-only ラッパーに差し替える
    (モジュール属性の差し替えは呼び出し元プロセス内のみに閉じる、
    このプロセスは1ジョブ限りで終了するため後始末は不要)。
    """
    import scripts.collect_boards_lean as cbl
    from src.board_state_machine import BoardState

    trace: list[tuple[float, int, int]] = []
    side_code = {"1P": 0, "2P": 1}
    state_code = {s.value: i for i, s in enumerate(BoardState)}
    original = cbl._process_side_lean

    def _traced_process_side_lean(*args: object, **kwargs: object) -> None:
        # 位置引数: (acc, state, side_label, board, bstate, score, video_id, t_sec, frame_idx, ...)
        side_label = args[2]
        bstate = args[4]
        t_sec = args[7]
        trace.append((float(t_sec), side_code[side_label], state_code[bstate.value]))
        return original(*args, **kwargs)

    cbl._process_side_lean = _traced_process_side_lean
    try:
        kwargs = config_kwargs(config_tag)
        cbl.collect_lean(
            video_path=video, out_npz=out_npz,
            start_sec=start_sec, max_sec=max_sec, **kwargs,
        )
    finally:
        cbl._process_side_lean = original

    out_trace.parent.mkdir(parents=True, exist_ok=True)
    if trace:
        t_arr = np.array([t for t, _, _ in trace], dtype=np.float32)
        side_arr = np.array([s for _, s, _ in trace], dtype=np.int8)
        state_arr = np.array([st for _, _, st in trace], dtype=np.int8)
    else:
        t_arr = np.zeros(0, dtype=np.float32)
        side_arr = np.zeros(0, dtype=np.int8)
        state_arr = np.zeros(0, dtype=np.int8)
    state_names = np.array([s.value for s in BoardState])
    np.savez_compressed(
        out_trace, t_sec=t_arr, side=side_arr, state=state_arr,
        state_names=state_names,
    )
    print(f"[worker] trace {len(trace)} entries -> {out_trace}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--config", choices=[CONFIG_A, CONFIG_B])
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--max-sec", type=float, default=0.0)
    ap.add_argument("--out-npz", type=Path)
    ap.add_argument("--out-trace", type=Path)
    args = ap.parse_args()

    if args.worker:
        assert args.video and args.config and args.out_npz and args.out_trace
        _run_worker(
            args.video, args.config, args.start_sec, args.max_sec,
            args.out_npz, args.out_trace,
        )
        return 0

    run_driver()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
