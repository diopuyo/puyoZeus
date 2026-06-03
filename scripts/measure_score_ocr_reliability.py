"""score OCR 信頼性測定スクリプト — 機能D基盤健全性確認。

機能D (掛け算式検知) は「連鎖発火時に score 領域へ掛け算式が出る
→ ScoreOcr.read が score=None を返す」前提で動く。本スクリプトは
この前提の健全性を 16 動画の実データで定量化する。

測定内容:
    1. score 読取り率 = score != None の frame 率 (per-side, per-video)
    2. None 率の状態別内訳 (STABLE / CHAIN / TSUMO_FALL / OJAMA_FALL / 他)
       → 機能D健全性の核心 = 「STABLE 中の None 率」が低いか
    3. 実誤発火リスク = STABLE 状態が続く中で 2 frame 連続 None かつ
       ink_ratio > SCORE_ROI_INK_RATIO_MIN が起きる回数
    4. 誤読率 = 単調性違反 (試合中 score が減少) / 全読取り件数

対象動画: data/match_clips/{v29,v40,v51,v57,v70,v89,v95,v97}/match01,match02
sample_interval = 0.0333 秒 (約 30fps 相当)

出力: data/verify/score_ocr_reliability.json

使い方:
    python scripts/measure_score_ocr_reliability.py
    python scripts/measure_score_ocr_reliability.py --workers 4
    python scripts/measure_score_ocr_reliability.py --smoke  # smoke: v29m01 のみ
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as _mp
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.board_state_machine import BoardState
from src.score_ocr import (
    SCORE_ROI_INK_RATIO_MIN,
    ScoreOcr,
    ScoreReadResult,
    _crop_score_roi,
    compute_score_roi_ink_ratio,
)
from src.recognition_pipeline import RecognitionPipeline

# ============================
# 定数
# ============================

# 処理間隔 (秒)。約 30fps 相当。
SAMPLE_INTERVAL_SEC: float = 0.0333

# 対象動画 ID と試合番号
TARGET_VIDEO_IDS: tuple[str, ...] = (
    "v29", "v40", "v51", "v57", "v70", "v89", "v95", "v97",
)
TARGET_MATCH_NUMS: tuple[int, ...] = (1, 2)

# match_clips ディレクトリ
MATCH_CLIPS_DIR: Path = Path("data/match_clips")

# 出力先
OUTPUT_PATH: Path = Path("data/verify/score_ocr_reliability.json")

# 機能D誤発火判定: STABLE 連続 None かつ ink_ratio > 閾値 の検出用
# CHAIN_FORMULA_CONSEC_FRAMES と同値
FORMULA_CONSEC_FRAMES: int = 2

# STABLE 中 None 率の警告閾値: この値を超えるとリスク動画と判定
STABLE_NONE_RATE_RISK_THRESHOLD: float = 0.10

# 単調性違反率の警告閾値
MONOTONIC_VIOLATION_RATE_THRESHOLD: float = 0.005

# 状態ラベル (JSON キー用)
STATE_LABEL_MAP: dict[str, str] = {
    "STABLE": "STABLE",
    "CHAIN": "CHAIN",
    "TSUMO_FALL": "TSUMO_FALL",
    "OJAMA_FALL": "OJAMA_FALL",
    "EFFECT": "EFFECT",
    "MENU": "MENU",
    "UNKNOWN": "UNKNOWN",
}


# ============================
# データクラス
# ============================


@dataclass
class SideStats:
    """1 サイド (1P or 2P) の集計結果."""

    side: str
    total_frames: int = 0
    readable_frames: int = 0  # score != None の件数

    # 状態別 frame 数 (None / 読取り両方含む)
    frames_by_state: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # 状態別 None frame 数
    none_by_state: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # 単調性違反
    monotonic_violations: int = 0  # score が直前より減少した件数
    last_score: int | None = None  # 前 frame の score (試合境界リセット用)

    # 機能D実誤発火リスク: STABLE 連続 2frame None + ink_ratio > 閾値
    formula_risk_events: int = 0
    # 連続カウンタ (状態追跡用, stateless 集計の外部 wrapper)
    _stable_consec_none: int = 0

    def state_none_rate(self, state_key: str) -> float:
        """指定 state の None 率を返す."""
        total = self.frames_by_state.get(state_key, 0)
        if total == 0:
            return 0.0
        return self.none_by_state.get(state_key, 0) / total

    def overall_read_rate(self) -> float:
        """全フレームの score 読取り率."""
        if self.total_frames == 0:
            return 0.0
        return self.readable_frames / self.total_frames

    def monotonic_violation_rate(self) -> float:
        """単調性違反率 = 違反 / 読取り可能フレーム数."""
        if self.readable_frames == 0:
            return 0.0
        return self.monotonic_violations / self.readable_frames


@dataclass
class VideoResult:
    """1 動画の測定結果."""

    video_id: str
    match_num: int
    video_path: str
    total_duration_sec: float
    p1: SideStats = field(default_factory=lambda: SideStats("1P"))
    p2: SideStats = field(default_factory=lambda: SideStats("2P"))
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON シリアライズ用辞書."""

        def side_to_dict(s: SideStats) -> dict[str, Any]:
            return {
                "total_frames": s.total_frames,
                "readable_frames": s.readable_frames,
                "read_rate": round(s.overall_read_rate(), 4),
                "monotonic_violations": s.monotonic_violations,
                "monotonic_violation_rate": round(s.monotonic_violation_rate(), 6),
                "formula_risk_events": s.formula_risk_events,
                "frames_by_state": dict(s.frames_by_state),
                "none_by_state": dict(s.none_by_state),
                "stable_none_rate": round(s.state_none_rate("STABLE"), 4),
                "chain_none_rate": round(s.state_none_rate("CHAIN"), 4),
                "is_risk": s.state_none_rate("STABLE") > STABLE_NONE_RATE_RISK_THRESHOLD,
            }

        return {
            "video_id": self.video_id,
            "match_num": self.match_num,
            "video_path": self.video_path,
            "total_duration_sec": self.total_duration_sec,
            "error": self.error,
            "p1": side_to_dict(self.p1),
            "p2": side_to_dict(self.p2),
        }


# ============================
# コア処理関数
# ============================


def _state_key(state: BoardState) -> str:
    """BoardState を集計キー文字列に変換する."""
    name = state.name if hasattr(state, "name") else str(state)
    return STATE_LABEL_MAP.get(name, name)


def _update_side_stats(
    stats: SideStats,
    score: int | None,
    state: BoardState,
    ink_ratio: float,
    is_match_active: bool,
) -> None:
    """1 frame 分の SideStats を更新する。stateless = 外部 wrapper で呼ぶ。

    Args:
        stats: 更新対象の SideStats (mutable)
        score: OCR 結果 (None = 読取り失敗)
        state: 現 frame の BoardState
        ink_ratio: score ROI の ink_ratio
        is_match_active: 試合中フラグ (False なら単調性チェックをスキップ)
    """
    stats.total_frames += 1
    key = _state_key(state)
    stats.frames_by_state[key] = stats.frames_by_state.get(key, 0) + 1

    if score is not None:
        stats.readable_frames += 1
        # 単調性チェック (試合中のみ、試合境界は呼出元でリセット済)
        if is_match_active and stats.last_score is not None:
            if score < stats.last_score:
                stats.monotonic_violations += 1
        stats.last_score = score
    else:
        stats.none_by_state[key] = stats.none_by_state.get(key, 0) + 1

    # 機能D実誤発火リスク: STABLE 中 None + ink_ratio > 閾値 が連続 2 frame
    if key == "STABLE" and score is None and ink_ratio > SCORE_ROI_INK_RATIO_MIN:
        stats._stable_consec_none += 1
        if stats._stable_consec_none >= FORMULA_CONSEC_FRAMES:
            stats.formula_risk_events += 1
            # カウンタをリセットしてイベントの重複カウントを防ぐ
            stats._stable_consec_none = 0
    else:
        # STABLE 連続 None 条件が崩れたらリセット
        if key != "STABLE" or score is not None:
            stats._stable_consec_none = 0


def _process_frame(
    frame: np.ndarray,
    pipeline: RecognitionPipeline,
    ocr: ScoreOcr,
    frame_idx: int,
    t_sec: float,
    result: VideoResult,
) -> None:
    """1 frame を処理して VideoResult を更新する。

    Args:
        frame: BGR フレーム画像
        pipeline: RecognitionPipeline インスタンス (state 取得用)
        ocr: ScoreOcr インスタンス
        frame_idx: フレームインデックス
        t_sec: 時刻 (秒)
        result: 更新対象の VideoResult (mutable)
    """
    # 1080p 正規化
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)

    # pipeline で状態取得
    pr = pipeline.update(frame_idx, t_sec, frame)

    # OCR 実行
    ocr_res: ScoreReadResult = ocr.read(frame)

    # ink_ratio 計算 (1P/2P 各サイド)
    roi_1p = _crop_score_roi(frame, "1P")
    roi_2p = _crop_score_roi(frame, "2P")
    ink_1p = compute_score_roi_ink_ratio(roi_1p) if roi_1p is not None else 0.0
    ink_2p = compute_score_roi_ink_ratio(roi_2p) if roi_2p is not None else 0.0

    _update_side_stats(
        result.p1,
        score=ocr_res.score_1p,
        state=pr.p1.state,
        ink_ratio=ink_1p,
        is_match_active=pr.is_match_active,
    )
    _update_side_stats(
        result.p2,
        score=ocr_res.score_2p,
        state=pr.p2.state,
        ink_ratio=ink_2p,
        is_match_active=pr.is_match_active,
    )


def _measure_one_video(video_path: Path, video_id: str, match_num: int) -> VideoResult:
    """1 動画を処理して VideoResult を返す。

    Args:
        video_path: 動画ファイルパス
        video_id: 動画 ID (例: "v29")
        match_num: 試合番号 (1 or 2)

    Returns:
        VideoResult
    """
    result = VideoResult(
        video_id=video_id,
        match_num=match_num,
        video_path=str(video_path),
        total_duration_sec=0.0,
    )

    if not video_path.is_file():
        result.error = f"ファイル不在: {video_path}"
        return result

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result.error = f"動画を開けない: {video_path}"
        return result

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames_raw / fps if fps > 0 else 0.0
    result.total_duration_sec = duration
    cap.release()

    # pipeline と ocr を生成 (重い初期化はここで 1 回)
    try:
        pipeline = RecognitionPipeline.load_default()
        pipeline.set_video_id(video_id)
    except Exception as e:
        result.error = f"pipeline 初期化失敗: {e}"
        return result

    try:
        ocr = ScoreOcr.load_default()
    except Exception as e:
        result.error = f"ScoreOcr 初期化失敗: {e}"
        return result

    # フレームループ
    cap = cv2.VideoCapture(str(video_path))
    interval_frames = max(1, int(fps * SAMPLE_INTERVAL_SEC))
    frame_idx = 0
    sample_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % interval_frames == 0:
            t_sec = frame_idx / fps
            try:
                _process_frame(frame, pipeline, ocr, sample_idx, t_sec, result)
            except Exception as e:
                print(f"  [warn] {video_id}m{match_num:02d} fi={frame_idx} エラー: {e}")
            sample_idx += 1
        frame_idx += 1

    cap.release()
    return result


def _measure_video_task(args: tuple[Path, str, int]) -> VideoResult:
    """並列 worker 用エントリポイント。"""
    video_path, video_id, match_num = args
    print(f"  [start] {video_id} match{match_num:02d}: {video_path.name}")
    res = _measure_one_video(video_path, video_id, match_num)
    stable_none_1p = res.p1.state_none_rate("STABLE")
    stable_none_2p = res.p2.state_none_rate("STABLE")
    print(
        f"  [done]  {video_id} match{match_num:02d}: "
        f"frames={res.p1.total_frames} "
        f"read_rate_1p={res.p1.overall_read_rate():.3f} "
        f"stable_none_1p={stable_none_1p:.4f} "
        f"stable_none_2p={stable_none_2p:.4f}"
        + (f" ERR={res.error}" if res.error else "")
    )
    return res


# ============================
# 集計・サマリ
# ============================


def _build_summary(results: list[VideoResult]) -> dict[str, Any]:
    """全動画の集計サマリを生成する。

    Returns:
        summary dict (JSON 書込み用)
    """
    risk_videos: list[dict] = []
    all_stable_none_1p: list[float] = []
    all_stable_none_2p: list[float] = []
    all_read_rate_1p: list[float] = []
    all_read_rate_2p: list[float] = []
    total_formula_risk = 0
    total_violations = 0
    total_readable = 0

    for r in results:
        if r.error:
            continue
        for side, stats in [("1P", r.p1), ("2P", r.p2)]:
            sn = stats.state_none_rate("STABLE")
            rr = stats.overall_read_rate()
            if side == "1P":
                all_stable_none_1p.append(sn)
                all_read_rate_1p.append(rr)
            else:
                all_stable_none_2p.append(sn)
                all_read_rate_2p.append(rr)
            if sn > STABLE_NONE_RATE_RISK_THRESHOLD:
                risk_videos.append({
                    "video_id": r.video_id,
                    "match_num": r.match_num,
                    "side": side,
                    "stable_none_rate": round(sn, 4),
                })
            total_formula_risk += stats.formula_risk_events
            total_violations += stats.monotonic_violations
            total_readable += stats.readable_frames

    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    # 機能D健全性判定
    avg_stable_none = _avg(all_stable_none_1p + all_stable_none_2p)
    is_healthy = (
        avg_stable_none < STABLE_NONE_RATE_RISK_THRESHOLD
        and len(risk_videos) == 0
    )

    return {
        "avg_stable_none_rate_1p": _avg(all_stable_none_1p),
        "avg_stable_none_rate_2p": _avg(all_stable_none_2p),
        "avg_read_rate_1p": _avg(all_read_rate_1p),
        "avg_read_rate_2p": _avg(all_read_rate_2p),
        "total_formula_risk_events": total_formula_risk,
        "total_monotonic_violations": total_violations,
        "total_readable_frames": total_readable,
        "monotonic_violation_rate_overall": round(
            total_violations / total_readable if total_readable > 0 else 0.0, 6
        ),
        "risk_videos": risk_videos,
        "feature_d_healthy": is_healthy,
        "thresholds": {
            "stable_none_rate_risk": STABLE_NONE_RATE_RISK_THRESHOLD,
            "ink_ratio_min": SCORE_ROI_INK_RATIO_MIN,
            "formula_consec_frames": FORMULA_CONSEC_FRAMES,
        },
    }


def _print_report(results: list[VideoResult], summary: dict[str, Any]) -> None:
    """コンソールに集計レポートを表示する."""
    print("\n" + "=" * 70)
    print("score OCR 信頼性測定 — 機能D基盤健全性レポート")
    print("=" * 70)

    print("\n[1] 動画別 STABLE 中 None 率 (機能D誤発火リスクの核心)")
    print(f"{'video_id':<12} {'match':<6} {'1P stable_none':>14} {'2P stable_none':>14} {'risk':<6}")
    print("-" * 55)
    for r in results:
        if r.error:
            print(f"{r.video_id:<12} {r.match_num:<6} ERROR: {r.error}")
            continue
        sn1 = r.p1.state_none_rate("STABLE")
        sn2 = r.p2.state_none_rate("STABLE")
        risk = "RISK" if (sn1 > STABLE_NONE_RATE_RISK_THRESHOLD or
                          sn2 > STABLE_NONE_RATE_RISK_THRESHOLD) else "-"
        print(f"{r.video_id:<12} {r.match_num:<6} {sn1:>14.4f} {sn2:>14.4f} {risk:<6}")

    print("\n[2] 動画別 None 状態内訳 (1P)")
    print(f"{'video_id':<12} {'match':<6} {'STABLE':>8} {'CHAIN':>8} {'TSUMO':>8} {'OJAMA':>8} {'OTHER':>8}")
    print("-" * 60)
    for r in results:
        if r.error:
            continue
        s = r.p1
        other = sum(
            s.none_by_state.get(k, 0)
            for k in s.none_by_state
            if k not in ("STABLE", "CHAIN", "TSUMO_FALL", "OJAMA_FALL")
        )
        print(
            f"{r.video_id:<12} {r.match_num:<6}"
            f" {s.none_by_state.get('STABLE', 0):>8}"
            f" {s.none_by_state.get('CHAIN', 0):>8}"
            f" {s.none_by_state.get('TSUMO_FALL', 0):>8}"
            f" {s.none_by_state.get('OJAMA_FALL', 0):>8}"
            f" {other:>8}"
        )

    print("\n[3] 誤読率 (単調性違反) per-video")
    print(f"{'video_id':<12} {'match':<6} {'1P viol':>10} {'1P rate':>10} {'2P viol':>10} {'2P rate':>10}")
    print("-" * 62)
    for r in results:
        if r.error:
            continue
        print(
            f"{r.video_id:<12} {r.match_num:<6}"
            f" {r.p1.monotonic_violations:>10}"
            f" {r.p1.monotonic_violation_rate():>10.6f}"
            f" {r.p2.monotonic_violations:>10}"
            f" {r.p2.monotonic_violation_rate():>10.6f}"
        )

    print("\n[4] 機能D実誤発火リスク (STABLE 連続 2frame None + ink_ratio > 閾値)")
    print(f"{'video_id':<12} {'match':<6} {'1P risk':>10} {'2P risk':>10}")
    print("-" * 42)
    for r in results:
        if r.error:
            continue
        print(f"{r.video_id:<12} {r.match_num:<6} {r.p1.formula_risk_events:>10} {r.p2.formula_risk_events:>10}")

    print("\n[5] サマリ")
    print(f"  avg STABLE None rate (1P): {summary['avg_stable_none_rate_1p']:.4f}")
    print(f"  avg STABLE None rate (2P): {summary['avg_stable_none_rate_2p']:.4f}")
    print(f"  avg read rate (1P):        {summary['avg_read_rate_1p']:.4f}")
    print(f"  avg read rate (2P):        {summary['avg_read_rate_2p']:.4f}")
    print(f"  total formula_risk_events: {summary['total_formula_risk_events']}")
    print(f"  monotonic_violation_rate:  {summary['monotonic_violation_rate_overall']:.6f}")
    risk = summary["risk_videos"]
    if risk:
        print(f"\n  [RISK] STABLE 中 None 率 > {STABLE_NONE_RATE_RISK_THRESHOLD:.0%} の動画:")
        for rv in risk:
            print(f"    {rv['video_id']} match{rv['match_num']} {rv['side']}: {rv['stable_none_rate']:.4f}")
    else:
        print("  [OK] STABLE 中 None 率が閾値を超える動画なし")

    healthy = summary["feature_d_healthy"]
    verdict = "HEALTHY (機能D 誤発火リスク低)" if healthy else "RISK (機能D 基盤に問題あり)"
    print(f"\n  機能D 基盤健全性: {verdict}")
    print("=" * 70)


# ============================
# エントリポイント
# ============================


def _build_tasks(
    smoke: bool,
) -> list[tuple[Path, str, int]]:
    """測定タスクリストを生成する。

    Args:
        smoke: True の場合は v29 match01 のみ

    Returns:
        (video_path, video_id, match_num) のタプルリスト
    """
    tasks: list[tuple[Path, str, int]] = []
    video_ids = ("v29",) if smoke else TARGET_VIDEO_IDS
    for vid in video_ids:
        match_nums = (1,) if smoke else TARGET_MATCH_NUMS
        for mnum in match_nums:
            clip_dir = MATCH_CLIPS_DIR / vid
            # ファイル名パターン: v29_match01.mp4 or match_v29_01.mp4 等を試みる
            candidates = [
                clip_dir / f"{vid}_match{mnum:02d}.mp4",
                clip_dir / f"match_{vid}_{mnum:02d}.mp4",
                clip_dir / f"v{vid[1:]}m{mnum:02d}.mp4",
            ]
            found: Path | None = None
            for cand in candidates:
                if cand.is_file():
                    found = cand
                    break
            if found is None:
                # glob で探す
                pat = list(clip_dir.glob(f"*match*{mnum:02d}*.mp4"))
                if pat:
                    found = pat[0]
            if found is not None:
                tasks.append((found, vid, mnum))
            else:
                print(f"  [skip] {vid} match{mnum:02d}: ファイル見つからず ({clip_dir})")
    return tasks


def main(argv: list[str] | None = None) -> int:
    """スクリプトエントリポイント."""
    parser = argparse.ArgumentParser(
        description="score OCR 信頼性測定 (機能D基盤健全性確認)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="並列ワーカー数 (default: 1)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="スモークテスト: v29 match01 のみ実行",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"JSON 出力先 (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args(argv)

    tasks = _build_tasks(smoke=args.smoke)
    if not tasks:
        print("[ERROR] 処理対象動画が 0 件。data/match_clips/ を確認してください。")
        return 1

    print(f"処理対象: {len(tasks)} 動画 (workers={args.workers})")
    print(f"sample_interval = {SAMPLE_INTERVAL_SEC} 秒")

    results: list[VideoResult] = []

    if args.workers <= 1:
        for task in tasks:
            results.append(_measure_video_task(task))
    else:
        ctx = _mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=ctx
        ) as executor:
            futures = [executor.submit(_measure_video_task, t) for t in tasks]
            for f in concurrent.futures.as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    print(f"  [ERROR] worker 例外: {e}")

    summary = _build_summary(results)
    _print_report(results, summary)

    # JSON 出力
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "summary": summary,
        "results": [r.to_dict() for r in results],
        "config": {
            "sample_interval_sec": SAMPLE_INTERVAL_SEC,
            "stable_none_rate_risk_threshold": STABLE_NONE_RATE_RISK_THRESHOLD,
            "formula_consec_frames": FORMULA_CONSEC_FRAMES,
            "ink_ratio_min": SCORE_ROI_INK_RATIO_MIN,
        },
    }
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(output_data, fp, ensure_ascii=False, indent=2)
    print(f"\nJSON 出力: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
