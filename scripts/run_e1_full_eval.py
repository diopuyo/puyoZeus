"""Phase I E1 評価: 8 動画 × 最初 2 試合切り出し + 認識率一括測定スクリプト。

手順:
    Step 1: 各動画から最初 2 試合を切り出し (= cut_matches_by_score_next.py を内部呼び出し)
            出力: data/match_clips/v<NN>/v<NN>_match01.mp4, v<NN>_match02.mp4
            並列 3 で実行 (I/O 競合回避のため上限を設ける)
    Step 2: 全 16 clip で measure_stable_cell_acc.py を実行
            出力: data/verify/stable_cell_acc/e1_full_8videos.json

使い方:
    PYTHONPATH=. python scripts/run_e1_full_eval.py

制約:
    - Step 1 は並列 3 (OpenCV 書き出し + I/O 競合回避)
    - Step 2 は全 16 clip を measure_stable_cell_acc に渡す
    - 既に切り出し済の clip はスキップ (冪等)
"""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# === プロジェクトルート ===
BASE = Path(__file__).resolve().parent.parent
PYTHON = str(BASE / "venv" / "bin" / "python")

# === 対象 8 動画 (video_id -> 元動画ファイル名) ===
VIDEO_MAP: dict[str, str] = {
    "v29": "video_29.mp4",
    "v40": "video_40.mp4",
    "v51": "video_51.mp4",
    "v57": "video_57.mp4",
    "v70": "video_70.mp4",
    "v89": "video_89.mp4",
    "v95": "video_95.mp4",
    "v97": "video_97.mp4",
}

# === 元動画ディレクトリ ===
FRAMES_DIR = BASE / "data" / "frames"

# === 切り出し出力ルート ===
CLIP_ROOT = BASE / "data" / "match_clips"

# === 切り出し設定 ===
BUFFER_SEC: float = 5.0       # 試合開始前バッファ (5 秒)
SCAN_INTERVAL: float = 0.5    # スキャン間隔

# === 切り出し並列数 ===
CUT_MAX_PARALLEL: int = 3

# === 評価出力 ===
EVAL_OUTPUT = BASE / "data" / "verify" / "stable_cell_acc" / "e1_full_8videos.json"

# === ログ ===
LOG_DIR = BASE / "logs" / "e1_eval"

# === holdout 動画 ID (固定 3 本) ===
HOLDOUT_IDS: list[str] = ["v29", "v40", "v89"]

# === 最大処理試合数 ===
MAX_MATCHES: int = 2


def _build_env() -> dict[str, str]:
    """PYTHONPATH を含む環境変数を返す。"""
    return {**os.environ, "PYTHONPATH": str(BASE)}


def _clip_path(vid: str, match_no: int) -> Path:
    """clip ファイルパスを返す。match_no は 1 始まり。"""
    return CLIP_ROOT / vid / f"{vid}_match{match_no:02d}.mp4"


def _cut_one_video(vid: str, src_path: Path, log_path: Path) -> list[Path]:
    """1 動画から最初 2 試合を切り出す。

    既に 2 ファイルとも存在する場合はスキップ。
    1 ファイルだけ存在する場合は cut_matches_by_score_next.py を再実行し、
    不足ファイルを補う。

    Returns:
        書き出した (または既存の) clip ファイルパスのリスト。
    """
    out_dir = CLIP_ROOT / vid
    clip1 = _clip_path(vid, 1)
    clip2 = _clip_path(vid, 2)

    # 両方存在すればスキップ
    if clip1.exists() and clip2.exists():
        print(f"[cut] {vid}: 既に切り出し済 → スキップ ({clip1.name}, {clip2.name})")
        return [clip1, clip2]

    print(f"[cut] {vid}: 切り出し開始 ({src_path.name}) ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # cut_matches_by_score_next.py を subprocess 呼び出し
    # --max-matches 3 = 最初 2 試合境界確定には 3 回目シグナルが必要
    cmd = [
        PYTHON, "-m", "scripts.cut_matches_by_score_next",
        "--input", str(src_path),
        "--output-dir", str(out_dir),
        "--video-stem", vid,
        "--buffer-sec", str(BUFFER_SEC),
        "--scan-interval", str(SCAN_INTERVAL),
        # MAX_MATCHES=2 を渡す: 2 試合境界確定後に早期終了
        "--max-matches", str(MAX_MATCHES),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        ret = subprocess.run(
            cmd,
            cwd=str(BASE),
            env=_build_env(),
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    if ret.returncode != 0:
        print(f"[cut] {vid}: cut_matches 失敗 (rc={ret.returncode}), ログ: {log_path}")
    else:
        print(f"[cut] {vid}: 切り出し完了")

    # 生成された clip を最初 2 件に絞る
    results: list[Path] = []
    for i in range(1, MAX_MATCHES + 1):
        p = _clip_path(vid, i)
        if p.exists():
            results.append(p)
        else:
            print(f"[cut] {vid}: match{i:02d} が生成されなかった (動画が 1 試合のみ?)")
    return results


def _cut_task(args: tuple[str, Path]) -> list[str]:
    """並列実行ワーカ。(vid, src_path) を受け取り clip ID リストを返す。"""
    vid, src_path = args
    log_path = LOG_DIR / f"cut_{vid}.log"
    clips = _cut_one_video(vid, src_path, log_path)
    return [p.stem for p in clips]


def step1_cut_all() -> list[str]:
    """全 8 動画を並列 3 で切り出し、評価対象 clip ID リストを返す。

    clip ID 形式: "<vid>_match<NN>" (= measure_stable_cell_acc の --videos 引数と一致)
    """
    print("\n" + "=" * 60)
    print("Step 1: 試合切り出し (8 動画 × 最初 2 試合、並列 3)")
    print("=" * 60)

    tasks: list[tuple[str, Path]] = []
    for vid, filename in VIDEO_MAP.items():
        src_path = FRAMES_DIR / filename
        if not src_path.exists():
            print(f"[cut] {vid}: 元動画が存在しない ({src_path}) → スキップ")
            continue
        tasks.append((vid, src_path))

    clip_ids: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=CUT_MAX_PARALLEL) as ex:
        futs = {ex.submit(_cut_task, t): t[0] for t in tasks}
        for fut in concurrent.futures.as_completed(futs):
            vid = futs[fut]
            try:
                ids = fut.result()
                clip_ids.extend(ids)
                print(f"[cut] {vid}: 完了 clip_ids={ids}")
            except Exception as e:
                print(f"[cut] {vid}: エラー {e}")

    print(f"\n[Step 1 完了] 切り出し clip 数: {len(clip_ids)}")
    for cid in sorted(clip_ids):
        print(f"  {cid}")
    return sorted(clip_ids)


def step2_measure(clip_ids: list[str]) -> int:
    """全 clip を measure_stable_cell_acc.py で評価する。

    Returns:
        subprocess の returncode (0=PASS, 1=FAIL, 2=エラー)
    """
    print("\n" + "=" * 60)
    print("Step 2: 認識率測定 (全 clip 一括)")
    print("=" * 60)

    if not clip_ids:
        print("[measure] 評価対象 clip がゼロ件。終了。")
        return 2

    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # holdout_ids: clip_ids の中で holdout 動画に属するもの
    holdout_clip_ids = [
        cid for cid in clip_ids
        if any(cid.startswith(vid) for vid in HOLDOUT_IDS)
    ]

    videos_arg = ",".join(clip_ids)
    holdout_arg = ",".join(holdout_clip_ids)

    print(f"[measure] videos=({len(clip_ids)} 件)")
    print(f"[measure] holdout=({len(holdout_clip_ids)} 件): {holdout_arg}")
    print(f"[measure] 出力: {EVAL_OUTPUT}")

    log_path = LOG_DIR / "measure_e1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # --sample-interval 1.0 = 毎秒 1 フレーム処理 (30fps 動画では 30 フレームに 1 回)
    # デフォルト (1/30s = 毎フレーム) だと 45,000 フレームで 8 時間超かかるため
    cmd = [
        PYTHON, "-m", "scripts.measure_stable_cell_acc",
        "--videos", videos_arg,
        "--holdout", holdout_arg,
        "--video-dir", str(CLIP_ROOT),
        "--output", str(EVAL_OUTPUT),
        "--sample-interval", "1.0",
    ]

    print(f"[measure] コマンド: {' '.join(cmd[:6])} ...")
    print(f"[measure] ログ: {log_path}")
    print("[measure] 実行中 (1-2 時間かかります) ...\n")

    with log_path.open("w", encoding="utf-8") as lf:
        ret = subprocess.run(
            cmd,
            cwd=str(BASE),
            env=_build_env(),
            stdout=lf,
            stderr=subprocess.STDOUT,
        )

    # ログの末尾 50 行を標準出力に表示
    _tail_log(log_path, n=50)

    print(f"\n[measure] 完了 rc={ret.returncode}")
    print(f"[measure] 結果 JSON: {EVAL_OUTPUT}")
    return ret.returncode


def _tail_log(log_path: Path, n: int = 30) -> None:
    """ログファイルの末尾 n 行を表示する。"""
    if not log_path.exists():
        return
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-n:] if len(lines) > n else lines
    print(f"\n--- {log_path.name} (末尾 {len(tail)} 行) ---")
    for line in tail:
        print(line)
    print("---")


def _print_result_summary() -> None:
    """JSON 結果を読み込んでサマリを表示する。"""
    import json
    if not EVAL_OUTPUT.exists():
        print("[summary] 結果 JSON が存在しない。")
        return
    with EVAL_OUTPUT.open("r", encoding="utf-8") as f:
        d = json.load(f)

    print("\n" + "=" * 60)
    print("E1 評価結果サマリ")
    print("=" * 60)
    ov = d["overall"]
    print(f"全マス平均合意率: {ov['acc']:.4f}  ({ov['correct']}/{ov['total_cells']})")

    ho = d.get("holdout_summary", {})
    if ho.get("acc") is not None:
        print(f"holdout 合意率:   {ho['acc']:.4f}  ({ho.get('correct',0)}/{ho.get('total_cells',0)})")

    print("\n[色別合意率]")
    for cname, acc in sorted(d["per_color"].items()):
        mark = "OK" if acc >= 0.98 else "NG"
        print(f"  {cname:8s}: {acc:.4f}  [{mark}]")

    print("\n[動画別合意率]")
    for vid, info in sorted(d["per_video"].items()):
        flag = "holdout" if info.get("is_holdout") else "       "
        print(f"  {vid:25s} {flag}  acc={info['acc']:.4f}  stable={info['stable_frame_count']}")

    print(f"\n判定: {d['verdict']}")
    if d.get("failures"):
        print("[FAIL 理由]")
        for r in d["failures"]:
            print(f"  - {r}")
    print("=" * 60)


def main() -> int:
    """Step 1 切り出し → Step 2 測定 → サマリ表示。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[E1] Phase I 全体評価 開始")
    print(f"[E1] BASE={BASE}")
    print(f"[E1] CLIP_ROOT={CLIP_ROOT}")
    print(f"[E1] EVAL_OUTPUT={EVAL_OUTPUT}")

    clip_ids = step1_cut_all()
    rc = step2_measure(clip_ids)
    _print_result_summary()

    print(f"\n[E1] 終了 rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
