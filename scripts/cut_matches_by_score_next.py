"""
試合切り出しスクリプト (score=0 + ネクスト変化検出 版)

設計:
    1. 長尺動画を scan_interval 秒間隔でスキャン
    2. 試合開始検出: score=0 (両サイド) かつネクスト ROI のハッシュが前 frame と異なる
       → その瞬間の buffer_sec 秒前 (= バッファ) から切り出し開始
    3. 試合終了検出: 次の「score=0 + ネクスト変化」検出を次試合開始として扱い、
       その clip_start_sec (バッファ込み開始) = 前の試合の終了点とする。
       動画末尾で試合中だった場合は末尾まで含める。
    4. OpenCV VideoWriter でクリップ書き出し (ffmpeg 不要)

注意:
    WinPanelDetector (★ WIN ★) は試合セクション全体で表示されるため
    試合終了の検出には使用しない。代わりに「次試合開始シグナル」を
    前試合の終了とみなす設計を採用。

用途:
    v29m2/v51m2 の認識崩壊 (MENU 画面なし) を根治するため、
    必ず MENU 画面 5 秒分をバッファとして先頭に含める。

Usage:
    PYTHONPATH=. python scripts/cut_matches_by_score_next.py \\
        --input data/raw_videos/v29.mp4 \\
        --output-dir data/match_clips/v29 \\
        --buffer-sec 5
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.score_zero import ScoreZeroDetector  # noqa: E402

# ============================
# 定数
# ============================

# ネクスト ROI (1P 側, 1920×1080, next_top + next_bot 合算矩形)
# src/next_detector.py の ROI_1P_NEXT_TOP/BOT を包含する矩形
NEXT_ROI_1P: tuple[int, int, int, int] = (162, 297, 710, 785)  # (y1, y2, x1, x2)
# 2P 側
NEXT_ROI_2P: tuple[int, int, int, int] = (162, 297, 1135, 1210)

# ネクスト ROI ハッシュサイズ (pHash 相当の縮小サイズ)
NEXT_HASH_SIZE: int = 16

# スキャン間隔 (秒) - 試合開始検出精度と速度のバランス
SCAN_INTERVAL_SEC: float = 0.5

# 試合開始確定に必要な連続一致数
# score_zero + next_changed が CONFIRM 回続けば確定
# = 1 でもユーザー仕様「score=0 + ネクストが動いた瞬間」に対応
# = 2 以上はチャタリング抑制目的だが、境界フレームを見逃す可能性あり
MATCH_START_CONFIRM: int = 1

# 試合終了後に追加で含める末尾バッファ (秒)
# = 次試合開始シグナルから buffer_sec 手前まで (= 前試合の末尾余白)
MATCH_END_TAIL_SEC: float = 5.0

# VideoWriter の fourcc
OUTPUT_FOURCC: str = "mp4v"

# 進捗ログの間隔 (フレーム)
PROGRESS_LOG_INTERVAL: int = 90

# ネクストが変化したとみなすハミング距離閾値 (ピクセル差分)
NEXT_HASH_HAMMING_THRESHOLD: int = 20

# 解像度
EXPECTED_HEIGHT: int = 1080
EXPECTED_WIDTH: int = 1920


# ============================
# データクラス
# ============================


@dataclass
class MatchBoundary:
    """1 試合分の切り出し範囲。"""

    match_index: int         # 0 オリジン
    clip_start_sec: float    # バッファ込みの開始秒 (>= 0)
    clip_end_sec: float      # 末尾バッファ込みの終了秒
    trigger_sec: float       # score=0 + next 変化を検出した秒
    end_trigger_sec: float   # WIN パネル or 次試合開始を検出した秒


# ============================
# ヘルパ関数
# ============================


def _resize_to_1080p(frame: np.ndarray) -> np.ndarray:
    """1920×1080 以外をリサイズして返す。"""
    h, w = frame.shape[:2]
    if h == EXPECTED_HEIGHT and w == EXPECTED_WIDTH:
        return frame
    return cv2.resize(frame, (EXPECTED_WIDTH, EXPECTED_HEIGHT),
                      interpolation=cv2.INTER_AREA)


def _compute_next_hash(frame: np.ndarray) -> np.ndarray:
    """1P+2P のネクスト ROI を縮小グレースケール化してハッシュ化する。

    Returns:
        shape (NEXT_HASH_SIZE * NEXT_HASH_SIZE * 2,) の uint8 配列。
        1P と 2P 両方を連結して誤発火耐性を上げる。
    """
    parts: list[np.ndarray] = []
    for y1, y2, x1, x2 in (NEXT_ROI_1P, NEXT_ROI_2P):
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            parts.append(np.zeros(NEXT_HASH_SIZE * NEXT_HASH_SIZE, dtype=np.uint8))
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (NEXT_HASH_SIZE, NEXT_HASH_SIZE),
                           interpolation=cv2.INTER_AREA)
        parts.append(small.flatten())
    return np.concatenate(parts).astype(np.uint8)


def _hash_distance(a: np.ndarray, b: np.ndarray) -> float:
    """2 ハッシュの平均絶対差分 (MAD) を返す。"""
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())


def _read_frame_at(cap: cv2.VideoCapture, t_sec: float) -> Optional[np.ndarray]:
    """指定秒のフレームを読み込んで 1080p に変換する。失敗時 None。"""
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return _resize_to_1080p(frame)


# ============================
# 試合境界検出
# ============================


def detect_match_boundaries(
    video_path: Path,
    buffer_sec: float = 5.0,
    scan_interval: float = SCAN_INTERVAL_SEC,
    confirm_count: int = MATCH_START_CONFIRM,
    max_matches: int = 0,
) -> list[MatchBoundary]:
    """長尺動画から全試合の切り出し境界を検出する。

    試合終了は「次の試合開始シグナル」で判定する。
    WinPanelDetector は score 表示エリアを誤検知するため使用しない。

    Args:
        video_path: 入力動画パス。
        buffer_sec: 試合開始判定の何秒前から切り出すか。
        scan_interval: スキャン間隔 (秒)。
        confirm_count: 試合開始確定に必要な連続検出数。
        max_matches: 0 以外の場合、最初 N 試合の境界が確定した時点で
                     早期終了する。最初 N 試合だけ切り出す用途に使用。

    Returns:
        MatchBoundary のリスト (時系列順)。
    """
    zero_det = ScoreZeroDetector.load_default()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        print(f"[detect] {video_path.name}: {duration_sec:.1f}s  fps={fps:.1f}")

        boundaries = _scan_boundaries(
            cap=cap,
            zero_det=zero_det,
            duration_sec=duration_sec,
            buffer_sec=buffer_sec,
            scan_interval=scan_interval,
            confirm_count=confirm_count,
            max_matches=max_matches,
        )
    finally:
        cap.release()

    return boundaries


def _scan_boundaries(
    cap: cv2.VideoCapture,
    zero_det: ScoreZeroDetector,
    duration_sec: float,
    buffer_sec: float,
    scan_interval: float,
    confirm_count: int,
    max_matches: int = 0,
) -> list[MatchBoundary]:
    """フレームスキャンループ本体。

    状態: searching → in_match → searching (次試合開始シグナルで遷移)
    試合終了 = 次の試合開始シグナルの clip_start_sec (バッファ込み) を使用。

    Args:
        max_matches: 0 以外の場合、このエントリ数の境界が確定した時点で
                     早期終了する。最初 N 試合だけ切り出す用途に使用。
                     N 試合境界確定 = N+1 回目のトリガーを検出した後に終了。
    """
    prev_hash: Optional[np.ndarray] = None
    start_confirm_count = 0
    start_triggers: list[float] = []  # 検出した試合開始 trigger_sec
    boundaries: list[MatchBoundary] = []

    t = 0.0
    while t <= duration_sec:
        # max_matches 達成チェック: N 試合境界確定 → 終了
        if max_matches > 0 and len(boundaries) >= max_matches:
            print(
                f"[detect]  max_matches={max_matches} 達成 → "
                f"早期終了 (t={t:.1f}s)"
            )
            return boundaries

        frame = _read_frame_at(cap, t)
        if frame is None:
            t += scan_interval
            continue

        zero_result = zero_det.detect(frame)
        cur_hash = _compute_next_hash(frame)
        prev_hash, start_confirm_count = _process_one_frame(
            t=t,
            zero_result=zero_result,
            cur_hash=cur_hash,
            prev_hash=prev_hash,
            start_confirm_count=start_confirm_count,
            confirm_count=confirm_count,
            start_triggers=start_triggers,
            boundaries=boundaries,
            buffer_sec=buffer_sec,
            duration_sec=duration_sec,
        )
        t += scan_interval

    _append_final_match(
        start_triggers=start_triggers,
        boundaries=boundaries,
        buffer_sec=buffer_sec,
        duration_sec=duration_sec,
    )
    return boundaries


def _process_one_frame(
    t: float,
    zero_result,
    cur_hash: np.ndarray,
    prev_hash: Optional[np.ndarray],
    start_confirm_count: int,
    confirm_count: int,
    start_triggers: list[float],
    boundaries: list[MatchBoundary],
    buffer_sec: float,
    duration_sec: float,
) -> tuple[np.ndarray, int]:
    """1 フレーム分の試合開始シグナル評価と境界確定処理。

    Returns:
        (new_prev_hash, new_confirm_count)
    """
    new_phase, new_confirm, _ = _handle_searching(
        t=t,
        frame=None,
        zero_result=zero_result,
        cur_hash=cur_hash,
        prev_hash=prev_hash,
        start_confirm_count=start_confirm_count,
        confirm_count=confirm_count,
    )
    if new_phase == "in_match":
        start_triggers.append(t)
        print(f"[detect]  試合開始検出  t={t:.1f}s  (match #{len(start_triggers)})")
        if len(start_triggers) >= 2:
            _finalize_boundary(
                boundaries=boundaries,
                start_triggers=start_triggers,
                buffer_sec=buffer_sec,
                duration_sec=duration_sec,
                new_trigger_sec=t,
            )
        return cur_hash, 0
    return cur_hash, new_confirm


def _append_final_match(
    start_triggers: list[float],
    boundaries: list[MatchBoundary],
    buffer_sec: float,
    duration_sec: float,
) -> None:
    """最後の試合を動画末尾まで含めて boundaries に追加する。"""
    if not start_triggers:
        return
    last_trigger = start_triggers[-1]
    clip_start = max(0.0, last_trigger - buffer_sec)
    registered_triggers = {b.trigger_sec for b in boundaries}
    if last_trigger not in registered_triggers:
        b = MatchBoundary(
            match_index=len(boundaries),
            clip_start_sec=clip_start,
            clip_end_sec=duration_sec,
            trigger_sec=last_trigger,
            end_trigger_sec=duration_sec,
        )
        boundaries.append(b)
        print(
            f"[detect]  動画末尾まで  "
            f"clip=[{clip_start:.1f}-{duration_sec:.1f}s]"
            f"  (match #{len(boundaries)})"
        )


def _finalize_boundary(
    boundaries: list[MatchBoundary],
    start_triggers: list[float],
    buffer_sec: float,
    duration_sec: float,
    new_trigger_sec: float,
) -> None:
    """前の試合の境界を確定して boundaries に追加する。

    前の試合の終了点 = 今の試合の clip_start_sec (= new_trigger_sec - buffer_sec)
    ただし前の試合の開始からある程度の長さがある場合のみ確定する。
    """
    if len(start_triggers) < 2:
        return
    prev_trigger = start_triggers[-2]
    # 前の試合の clip_start
    prev_clip_start = max(0.0, prev_trigger - buffer_sec)
    # 前の試合の clip_end = 今の試合の clip_start (MENU バッファが重なる形)
    prev_clip_end = max(0.0, new_trigger_sec - buffer_sec)

    # 前の試合が極端に短い場合は誤検知とみなしてスキップ
    min_match_duration = buffer_sec * 2  # バッファの 2 倍 = 最低 10 秒
    actual_duration = prev_clip_end - prev_clip_start
    if actual_duration < min_match_duration:
        print(
            f"[detect]  試合短すぎ ({actual_duration:.1f}s < {min_match_duration:.1f}s)"
            f"  t={prev_trigger:.1f}s → スキップ"
        )
        # start_triggers から前の試合を除去 (今の試合を前の試合として扱う)
        start_triggers.pop(-2)
        return

    match_index = len(boundaries)
    b = MatchBoundary(
        match_index=match_index,
        clip_start_sec=prev_clip_start,
        clip_end_sec=prev_clip_end,
        trigger_sec=prev_trigger,
        end_trigger_sec=new_trigger_sec - buffer_sec,
    )
    boundaries.append(b)
    print(
        f"[detect]  試合境界確定  "
        f"clip=[{prev_clip_start:.1f}-{prev_clip_end:.1f}s]"
        f"  (match #{match_index + 1})"
    )


def _handle_searching(
    t: float,
    frame: Optional[np.ndarray],
    zero_result,
    cur_hash: np.ndarray,
    prev_hash: Optional[np.ndarray],
    start_confirm_count: int,
    confirm_count: int,
) -> tuple[str, int, Optional[np.ndarray]]:
    """searching フェーズのシグナル評価。

    Args:
        frame: フレーム画像 (将来の拡張用、現在は未使用)。

    Returns:
        (new_phase, new_confirm_count, new_prev_hash)
    """
    # score=0 (両サイド) かつネクスト変化 = 試合開始候補
    score_is_zero = zero_result.both_zero
    next_changed = (
        prev_hash is not None
        and _hash_distance(cur_hash, prev_hash) > NEXT_HASH_HAMMING_THRESHOLD
    )
    is_candidate = score_is_zero and next_changed

    if is_candidate:
        new_count = start_confirm_count + 1
        if new_count >= confirm_count:
            return "in_match", 0, cur_hash
        return "searching", new_count, prev_hash
    # 候補でない場合はカウントリセット
    return "searching", 0, cur_hash



# ============================
# クリップ書き出し
# ============================


def write_clips(
    video_path: Path,
    boundaries: list[MatchBoundary],
    output_dir: Path,
    video_stem: str = "",
    no_audio: bool = False,
) -> list[Path]:
    """MatchBoundary リストに従ってクリップを書き出す。

    Args:
        video_path: 元動画パス。
        boundaries: detect_match_boundaries() の結果。
        output_dir: 出力ディレクトリ。
        video_stem: 出力ファイル名のプレフィックス (省略時は video_path.stem)。
        no_audio: True でオーディオなし (OpenCV は元々音声なし)。

    Returns:
        書き出したファイルパスのリスト。
    """
    if not boundaries:
        print("[write_clips] 試合境界が 0 件のため書き出しをスキップ")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_stem or video_path.stem
    written_paths: list[Path] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_code = cv2.VideoWriter_fourcc(*OUTPUT_FOURCC)

        for b in boundaries:
            out_name = f"{stem}_match{b.match_index + 1:02d}.mp4"
            out_path = output_dir / out_name
            n_written = _write_clip_range(
                cap=cap,
                out_path=out_path,
                start_sec=b.clip_start_sec,
                end_sec=b.clip_end_sec,
                fps=fps,
                width=width,
                height=height,
                fourcc_code=fourcc_code,
            )
            print(
                f"[write]  {out_name}  "
                f"({b.clip_start_sec:.1f}-{b.clip_end_sec:.1f}s, "
                f"{n_written} frames)"
            )
            written_paths.append(out_path)
    finally:
        cap.release()

    return written_paths


def _write_clip_range(
    cap: cv2.VideoCapture,
    out_path: Path,
    start_sec: float,
    end_sec: float,
    fps: float,
    width: int,
    height: int,
    fourcc_code: int,
) -> int:
    """指定秒範囲のフレームを書き出す。書き出したフレーム数を返す。"""
    writer = cv2.VideoWriter(
        str(out_path), fourcc_code, fps, (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter を開けません: {out_path}")

    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    written = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            pos_sec = pos_ms / 1000.0 if pos_ms > 0 else (
                start_sec + written / max(fps, 1.0)
            )
            if pos_sec > end_sec:
                break
            writer.write(frame)
            written += 1
            if written % PROGRESS_LOG_INTERVAL == 0:
                print(
                    f"[write]    {out_path.name}: {written} frames "
                    f"(t={pos_sec:.1f}s)",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        writer.release()

    return written


# ============================
# CLI
# ============================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cut_matches_by_score_next",
        description="score=0 + ネクスト変化検出で試合を切り出す",
    )
    p.add_argument("--input", required=True, help="入力長尺動画パス")
    p.add_argument("--output-dir", required=True, help="出力ディレクトリ")
    p.add_argument(
        "--buffer-sec", type=float, default=5.0,
        help="試合開始の何秒前から切り出すか (デフォルト: 5.0)",
    )
    p.add_argument(
        "--scan-interval", type=float, default=SCAN_INTERVAL_SEC,
        help=f"スキャン間隔 (秒, デフォルト: {SCAN_INTERVAL_SEC})",
    )
    p.add_argument(
        "--confirm-count", type=int, default=MATCH_START_CONFIRM,
        help=f"試合開始確定に必要な連続検出数 (デフォルト: {MATCH_START_CONFIRM})",
    )
    p.add_argument(
        "--no-audio", action="store_true",
        help="オーディオなしモード (OpenCV 書き出しは常にオーディオなし)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="境界検出のみ実行、クリップ書き出しをスキップ",
    )
    p.add_argument(
        "--video-stem", default="",
        help="出力ファイル名プレフィックス (省略時は入力ファイル名 stem)",
    )
    p.add_argument(
        "--max-matches", type=int, default=0,
        help="最初 N 試合の境界が確定した時点で早期終了 (0=全試合, デフォルト: 0)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI エントリポイント。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] 入力動画が存在しない: {in_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)

    boundaries = detect_match_boundaries(
        video_path=in_path,
        buffer_sec=args.buffer_sec,
        scan_interval=args.scan_interval,
        confirm_count=args.confirm_count,
        max_matches=args.max_matches,
    )

    print(f"\n検出試合数: {len(boundaries)}")
    for b in boundaries:
        print(
            f"  match #{b.match_index + 1:2d}  "
            f"trigger={b.trigger_sec:.1f}s  "
            f"end={b.end_trigger_sec:.1f}s  "
            f"clip=[{b.clip_start_sec:.1f}-{b.clip_end_sec:.1f}s]"
        )

    if args.dry_run:
        print("\n[dry-run] クリップ書き出しをスキップ")
        return 0

    paths = write_clips(
        video_path=in_path,
        boundaries=boundaries,
        output_dir=out_dir,
        video_stem=args.video_stem,
        no_audio=args.no_audio,
    )
    print(f"\n書き出し完了: {len(paths)} ファイル → {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
