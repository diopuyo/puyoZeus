"""ツモ数 vs 認識 STABLE 時間率 比較スクリプト。

試合ごとに 2 つの指標を計測する:

[指標 1] ツモ数 (ネクスト変化回数)
  - ネクスト pair ROI のハッシュが変化した回数 (立ち上がりエッジ)
  - 1 ツモ = 1 ネクスト変化イベント
  - cut_matches_by_score_next.py の hash ロジックを流用

[指標 2] STABLE 時間率 (認識システムが STABLE を出力した frame 割合)
  - trial 区間中の全サンプル frame 数に対する STABLE frame 数の比
  - p1_stable_rate = p1_stable_samples / total_samples
  - p2_stable_rate = p2_stable_samples / total_samples
  - avg_stable_rate = (p1 + p2) / (2 * total)
  - 理想: 0.50 以上 (試合中半分以上は STABLE = 盤面確定中)

FAIL 判定: avg_stable_rate < FAIL_STABLE_RATE_THRESHOLD (= 0.30)
  = STABLE が 30% 未満 = 認識が非常に不安定

補助情報:
  - p1/p2_peak_cells: STABLE 盤面の最大 non-EMPTY cell 数 (積み上がり確認用)
  - tsumo_count: ネクスト変化回数 (ツモペース確認用)

Usage:
    PYTHONPATH=. python scripts/measure_tsumo_vs_recognition.py \\
        --video data/match_clips/v40/v40_match01.mp4

    # 全 17 clip 一括:
    PYTHONPATH=. python scripts/measure_tsumo_vs_recognition.py \\
        --video-dir data/match_clips \\
        --output data/eval/tsumo_vs_recognition.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# ============================
# 定数
# ============================

# ネクスト変化のサンプリング間隔 (秒)。 30fps 相当。
TSUMO_SCAN_INTERVAL_SEC: float = 0.033

# RecognitionPipeline 呼び出し間隔 (秒)。 visualize_recognition.py と同値。
RECOG_SCAN_INTERVAL_SEC: float = 0.033

# STABLE 時間率がこれを下回ると FAIL 判定。
# = 30% 未満の frame で STABLE = 認識が非常に不安定
FAIL_STABLE_RATE_THRESHOLD: float = 0.30

# ネクスト hash 変化のハミング距離閾値 (cut_matches_by_score_next.py と同値)。
NEXT_HASH_HAMMING_THRESHOLD: int = 20

# 認識処理の進捗ログ間隔 (秒)
PROGRESS_LOG_INTERVAL_SEC: float = 15.0

# ネクスト ROI ハッシュ縮小サイズ (cut_matches_by_score_next.py と同値)
NEXT_HASH_SIZE: int = 16

# ネクスト ROI (1P + 2P、 cut_matches_by_score_next.py と同値)
NEXT_ROI_1P: tuple[int, int, int, int] = (162, 297, 710, 785)
NEXT_ROI_2P: tuple[int, int, int, int] = (162, 297, 1135, 1210)

# 1920×1080 期待解像度
EXPECTED_HEIGHT: int = 1080
EXPECTED_WIDTH: int = 1920

# STABLE 盤面カウントに使う最小 STABLE 観測数 (= 試合序盤の空盤面を除外)
MIN_STABLE_SAMPLE_COUNT: int = 3

# ネクスト変化「立ち上がりエッジ」カウントのためのデバウンス フレーム数
# = 変化開始から N フレーム以内の再変化は同じイベントとして無視
TSUMO_DEBOUNCE_FRAMES: int = 5


# ============================
# ヘルパ: フレーム操作
# ============================


def _resize_to_1080p(frame: np.ndarray) -> np.ndarray:
    """1920×1080 以外をリサイズして返す。"""
    h, w = frame.shape[:2]
    if h == EXPECTED_HEIGHT and w == EXPECTED_WIDTH:
        return frame
    return cv2.resize(
        frame, (EXPECTED_WIDTH, EXPECTED_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def _read_frame_at(
    cap: cv2.VideoCapture,
    t_sec: float,
) -> Optional[np.ndarray]:
    """指定秒のフレームを読み込んで 1080p に変換する。 失敗時 None。"""
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return _resize_to_1080p(frame)


# ============================
# ヘルパ: ネクスト hash
# ============================


def _compute_next_hash(frame: np.ndarray) -> np.ndarray:
    """1P+2P のネクスト ROI を縮小グレースケールで hash 化する。

    cut_matches_by_score_next.py の同名関数と同一ロジック。
    ローカル定義にすることでスクリプト単体での動作を保証する。
    """
    parts: list[np.ndarray] = []
    for y1, y2, x1, x2 in (NEXT_ROI_1P, NEXT_ROI_2P):
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            parts.append(np.zeros(NEXT_HASH_SIZE * NEXT_HASH_SIZE, dtype=np.uint8))
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(
            gray, (NEXT_HASH_SIZE, NEXT_HASH_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        parts.append(small.flatten())
    return np.concatenate(parts).astype(np.uint8)


def _hash_distance(a: np.ndarray, b: np.ndarray) -> float:
    """2 ハッシュの平均絶対差分 (MAD) を返す。"""
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


# ============================
# ツモ数カウント
# ============================


def count_tsumo(
    cap: cv2.VideoCapture,
    start_sec: float,
    end_sec: float,
) -> int:
    """試合区間のネクスト変化回数 (= ツモ数) を返す。

    ネクスト ROI のハッシュが NEXT_HASH_HAMMING_THRESHOLD を超えたとき、
    前回の変化から TSUMO_DEBOUNCE_FRAMES 以上経過していれば 1 カウントする。
    これにより「変化が持続している間に複数回カウントされる」 問題をデバウンスする。

    Args:
        cap: 動画キャプチャ (シーク可能)。
        start_sec: 試合開始秒。
        end_sec: 試合終了秒。

    Returns:
        ツモ数 (ネクスト変化回数)。
    """
    prev_hash: Optional[np.ndarray] = None
    tsumo_count: int = 0
    last_tsumo_sample_idx: int = -TSUMO_DEBOUNCE_FRAMES - 1
    sample_idx: int = 0
    t = start_sec

    while t <= end_sec:
        frame = _read_frame_at(cap, t)
        if frame is None:
            t += TSUMO_SCAN_INTERVAL_SEC
            sample_idx += 1
            continue
        cur_hash = _compute_next_hash(frame)
        if prev_hash is not None:
            dist = _hash_distance(cur_hash, prev_hash)
            changed = dist > NEXT_HASH_HAMMING_THRESHOLD
            # デバウンス: 前回変化から DEBOUNCE_FRAMES 以上離れていれば新規イベント
            since_last = sample_idx - last_tsumo_sample_idx
            if changed and since_last > TSUMO_DEBOUNCE_FRAMES:
                tsumo_count += 1
                last_tsumo_sample_idx = sample_idx
        prev_hash = cur_hash
        t += TSUMO_SCAN_INTERVAL_SEC
        sample_idx += 1

    return tsumo_count


# ============================
# 認識 STABLE 率計測
# ============================


def _count_non_empty(board) -> int:
    """Board の non-EMPTY / non-UNKNOWN cell 数を返す。"""
    count = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            v = int(board.get(r, c))
            if v != COLOR_EMPTY and v != COLOR_UNKNOWN:
                count += 1
    return count


def measure_stable_rate(
    pipeline: RecognitionPipeline,
    cap: cv2.VideoCapture,
    start_sec: float,
    end_sec: float,
    fps: float,
) -> tuple[float, float, int, int, int, int, int]:
    """STABLE 時間率と peak cell 数を計測して返す。

    STABLE 時間率 = STABLE だった sample 数 / 全 sample 数。
    試合中に認識システムが盤面を「確定」できた時間の割合。

    Args:
        pipeline: RecognitionPipeline インスタンス (呼び出し元で reset 済み)。
        cap: 動画キャプチャ。
        start_sec: 試合開始秒。
        end_sec: 試合終了秒。
        fps: 動画の fps (frame_idx 計算用)。

    Returns:
        (p1_stable_rate, p2_stable_rate,
         p1_stable_count, p2_stable_count, total_count,
         p1_peak_cells, p2_peak_cells)
    """
    p1_stable_cnt: int = 0
    p2_stable_cnt: int = 0
    total_cnt: int = 0
    p1_peak: int = 0
    p2_peak: int = 0

    t = start_sec
    last_log_t = start_sec

    while t <= end_sec:
        frame = _read_frame_at(cap, t)
        if frame is None:
            t += RECOG_SCAN_INTERVAL_SEC
            continue

        frame_idx = int(t * fps)
        result = pipeline.update(frame_idx, t, frame)
        total_cnt += 1

        # 1P STABLE チェック
        if (result.p1.state == BoardState.STABLE
                and result.p1.confirmed_board is not None):
            p1_stable_cnt += 1
            if p1_stable_cnt >= MIN_STABLE_SAMPLE_COUNT:
                cnt = _count_non_empty(result.p1.confirmed_board)
                if cnt > p1_peak:
                    p1_peak = cnt

        # 2P STABLE チェック
        if (result.p2.state == BoardState.STABLE
                and result.p2.confirmed_board is not None):
            p2_stable_cnt += 1
            if p2_stable_cnt >= MIN_STABLE_SAMPLE_COUNT:
                cnt = _count_non_empty(result.p2.confirmed_board)
                if cnt > p2_peak:
                    p2_peak = cnt

        # 進捗ログ
        if t - last_log_t >= PROGRESS_LOG_INTERVAL_SEC:
            elapsed = t - start_sec
            total = end_sec - start_sec
            p1_rate = p1_stable_cnt / max(total_cnt, 1)
            p2_rate = p2_stable_cnt / max(total_cnt, 1)
            print(
                f"  [recog] t={t:.1f}s ({elapsed:.0f}/{total:.0f}s) "
                f"STABLE_rate 1P={p1_rate:.3f} 2P={p2_rate:.3f} "
                f"peak 1P={p1_peak} 2P={p2_peak}",
                flush=True,
            )
            last_log_t = t

        t += RECOG_SCAN_INTERVAL_SEC

    p1_rate = p1_stable_cnt / max(total_cnt, 1)
    p2_rate = p2_stable_cnt / max(total_cnt, 1)
    return (
        p1_rate, p2_rate,
        p1_stable_cnt, p2_stable_cnt, total_cnt,
        p1_peak, p2_peak,
    )


# ============================
# 評価: 1 試合
# ============================


def evaluate_match(
    match_idx: int,
    tsumo_count: int,
    p1_stable_rate: float,
    p2_stable_rate: float,
    p1_stable_count: int,
    p2_stable_count: int,
    total_count: int,
    p1_peak: int,
    p2_peak: int,
    start_sec: float,
    end_sec: float,
) -> dict:
    """1 試合分の評価結果 dict を返す。

    主要指標:
      avg_stable_rate = (p1_stable_rate + p2_stable_rate) / 2
        → 0.5 以上が目標。 低いほど認識不良。
    FAIL: avg_stable_rate < FAIL_STABLE_RATE_THRESHOLD

    Args:
        match_idx: 試合インデックス (1 オリジン)。
        tsumo_count: ネクスト変化回数 (= ツモ数、補助)。
        p1_stable_rate: 1P STABLE 時間率 (0-1)。
        p2_stable_rate: 2P STABLE 時間率 (0-1)。
        p1_stable_count: 1P STABLE サンプル数。
        p2_stable_count: 2P STABLE サンプル数。
        total_count: 全サンプル数。
        p1_peak: 1P peak cell 数 (補助)。
        p2_peak: 2P peak cell 数 (補助)。
        start_sec: 試合開始秒。
        end_sec: 試合終了秒。

    Returns:
        JSON シリアライズ可能な評価 dict。
    """
    avg_rate = (p1_stable_rate + p2_stable_rate) / 2.0
    verdict = "FAIL" if avg_rate < FAIL_STABLE_RATE_THRESHOLD else "PASS"

    return {
        "match_idx": match_idx,
        "start_sec": round(start_sec, 2),
        "end_sec": round(end_sec, 2),
        "duration_sec": round(end_sec - start_sec, 2),
        "tsumo_count": tsumo_count,
        "p1_stable_rate": round(p1_stable_rate, 4),
        "p2_stable_rate": round(p2_stable_rate, 4),
        "avg_stable_rate": round(avg_rate, 4),
        "p1_stable_count": p1_stable_count,
        "p2_stable_count": p2_stable_count,
        "total_sample_count": total_count,
        "p1_peak_cells": p1_peak,
        "p2_peak_cells": p2_peak,
        "verdict": verdict,
    }


# ============================
# 1 clip 動画の計測
# ============================


def _inject_per_video_hsv(pipeline: RecognitionPipeline, video_path: Path) -> None:
    """動画 ID から per-video HSV を pipeline に inject する。

    visualize_recognition.py:694-726 の inject ロジックを踏襲。
    inject 失敗は silent skip (= 本評価に影響させない)。
    """
    import re

    hsv_db_root = Path("data/per_video_hsv_ranges")
    merged_default = hsv_db_root / "_merged_default.json"

    m = re.match(r"(v\d+)", video_path.name)
    candidate = hsv_db_root / f"{m.group(1)}.json" if m else None
    hsv_path = candidate if (candidate and candidate.exists()) else merged_default

    if not hsv_path.exists():
        return

    try:
        with hsv_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        ranges = state.get("per_video_ranges", {})
        ranges_int = {int(k): tuple(int(x) for x in v) for k, v in ranges.items()}
        from src.hybrid_classifier import HybridClassifier
        hc = getattr(pipeline._reader, "_classifier", None)
        if (
            isinstance(hc, HybridClassifier)
            and hasattr(hc._hsv, "set_color_ranges_from_simple")
            and ranges_int
        ):
            hc._hsv.set_color_ranges_from_simple(ranges_int)
        print(f"  [hsv] injected from {hsv_path.name} ({len(ranges_int)} colors)")
    except Exception as e:
        print(f"  [hsv] inject failed (silent skip): {e}", file=sys.stderr)


def _measure_match_in_video(
    pipeline: RecognitionPipeline,
    cap: cv2.VideoCapture,
    fps: float,
    match_idx: int,
    start_sec: float,
    end_sec: float,
) -> dict:
    """1 試合分のツモ数 + STABLE 率を計測して評価 dict を返す。

    Args:
        pipeline: RecognitionPipeline (試合ごとに reset 済み)。
        cap: 動画キャプチャ。
        fps: 動画 fps。
        match_idx: 試合番号 (1 オリジン)。
        start_sec: 試合開始秒。
        end_sec: 試合終了秒。

    Returns:
        evaluate_match() の戻り値 dict。
    """
    print(
        f"  [match {match_idx}] t=[{start_sec:.1f}-{end_sec:.1f}s] "
        f"({end_sec-start_sec:.1f}s)",
        flush=True,
    )

    # ツモ数カウント (pipeline 使わず hash のみ)
    tsumo = count_tsumo(cap, start_sec, end_sec)
    print(f"  [match {match_idx}] tsumo_count={tsumo}", flush=True)

    # 認識処理 (pipeline は reset 済みで渡す)
    pipeline.reset()
    p1_rate, p2_rate, p1_sc, p2_sc, total, p1_peak, p2_peak = measure_stable_rate(
        pipeline, cap, start_sec, end_sec, fps,
    )
    avg_rate = (p1_rate + p2_rate) / 2.0
    print(
        f"  [match {match_idx}] STABLE_rate 1P={p1_rate:.3f} 2P={p2_rate:.3f} "
        f"avg={avg_rate:.3f} peak 1P={p1_peak} 2P={p2_peak}",
        flush=True,
    )

    return evaluate_match(
        match_idx=match_idx,
        tsumo_count=tsumo,
        p1_stable_rate=p1_rate,
        p2_stable_rate=p2_rate,
        p1_stable_count=p1_sc,
        p2_stable_count=p2_sc,
        total_count=total,
        p1_peak=p1_peak,
        p2_peak=p2_peak,
        start_sec=start_sec,
        end_sec=end_sec,
    )


def measure_video(video_path: Path) -> dict:
    """1 動画 (= 1 clip) を計測して評価 dict を返す。

    match_clips 配下の既切り出し clip を想定。
    1 clip = 1 試合として扱い、動画全体を 1 試合として計測する。

    Args:
        video_path: clip 動画パス。

    Returns:
        {video, match_count, matches, summary} の dict。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / max(fps, 1.0)
        print(
            f"[measure] {video_path.name}: {duration_sec:.1f}s fps={fps:.1f}",
            flush=True,
        )

        pipeline = RecognitionPipeline.load_default(
            stable_frame_count=3,
            load_score_ocr=True,
            enable_chain_tracker=True,
            load_next_detector=True,
            force_in_match=True,
        )
        _inject_per_video_hsv(pipeline, video_path)

        result = _measure_match_in_video(
            pipeline=pipeline,
            cap=cap,
            fps=fps,
            match_idx=1,
            start_sec=0.0,
            end_sec=duration_sec,
        )
        matches = [result]

    finally:
        cap.release()

    avg_stable = result["avg_stable_rate"]
    summary = {
        "match_count": 1,
        "avg_stable_rate": avg_stable,
        "verdict": "FAIL" if avg_stable < FAIL_STABLE_RATE_THRESHOLD else "PASS",
    }

    return {
        "video": video_path.name,
        "matches": matches,
        "summary": summary,
    }


# ============================
# CLI
# ============================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="measure_tsumo_vs_recognition",
        description="STABLE 時間率でツモ認識精度を定量化する",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="計測する動画ファイル (1 clip)")
    src.add_argument(
        "--video-dir", type=Path,
        help="計測する動画ディレクトリ (再帰的に *.mp4 を収集)",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="JSON 出力先 (省略時は stdout に出力)",
    )
    return p


def _collect_videos(video_dir: Path) -> list[Path]:
    """ディレクトリ配下の *.mp4 を再帰収集してソートして返す。"""
    return sorted(video_dir.rglob("*.mp4"))


def _compute_overall_summary(results: list[dict]) -> dict:
    """複数動画の summary から全体統計を計算する。

    avg_stable_rate の加重平均 (= total_sample_count で重み付け) を overall とする。
    """
    if not results:
        return {"match_count": 0, "avg_stable_rate": 0.0, "verdict": "N/A"}

    total_p1_stable = 0
    total_p2_stable = 0
    total_samples = 0
    total_matches = 0
    fail_count = 0

    for r in results:
        for m in r.get("matches", []):
            n = m.get("total_sample_count", 0)
            total_p1_stable += m.get("p1_stable_count", 0)
            total_p2_stable += m.get("p2_stable_count", 0)
            total_samples += n
            total_matches += 1
            if m.get("verdict") == "FAIL":
                fail_count += 1

    p1_rate = total_p1_stable / max(total_samples, 1)
    p2_rate = total_p2_stable / max(total_samples, 1)
    avg_rate = (p1_rate + p2_rate) / 2.0

    return {
        "video_count": len(results),
        "match_count": total_matches,
        "total_sample_count": total_samples,
        "overall_p1_stable_rate": round(p1_rate, 4),
        "overall_p2_stable_rate": round(p2_rate, 4),
        "overall_avg_stable_rate": round(avg_rate, 4),
        "fail_match_count": fail_count,
        "verdict": "FAIL" if avg_rate < FAIL_STABLE_RATE_THRESHOLD else "PASS",
    }


def main(argv: Optional[list[str]] = None) -> int:
    """CLI エントリポイント。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.video is not None:
        videos = [args.video]
    else:
        videos = _collect_videos(args.video_dir)
        print(f"[measure] {len(videos)} clip(s) 収集: {args.video_dir}", flush=True)

    if not videos:
        print("[ERROR] 動画ファイルが見つかりません", file=sys.stderr)
        return 1

    all_results: list[dict] = []
    for vp in videos:
        try:
            res = measure_video(vp)
            all_results.append(res)
            rate = res["summary"]["avg_stable_rate"]
            verdict = res["summary"]["verdict"]
            print(
                f"[result] {vp.name}: avg_stable_rate={rate:.4f} {verdict}",
                flush=True,
            )
        except Exception as e:
            print(f"[ERROR] {vp.name}: {e}", file=sys.stderr)
            all_results.append({
                "video": vp.name,
                "error": str(e),
                "matches": [],
                "summary": {"verdict": "ERROR"},
            })

    overall = _compute_overall_summary(all_results)

    output = {
        "results": all_results,
        "overall_summary": overall,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[done] → {args.output}", flush=True)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))

    print(
        f"\n[overall] avg_stable_rate={overall['overall_avg_stable_rate']:.4f} "
        f"({overall.get('fail_match_count', 0)}/{overall.get('match_count', 0)} FAIL) "
        f"→ {overall['verdict']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
