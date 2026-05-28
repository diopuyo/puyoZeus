"""連鎖補正版 セル数精度評価スクリプト。

STABLE-to-STABLE のセル数差分を追跡し、発生したゲームイベント
(ツモ落下 / 連鎖消滅 / おじゃま降下) を考慮して「期待セル数」を更新。
認識セル数との誤差を物理的に補正した指標で誤認を定量化する。

評価指標 (連鎖補正版):

  各 STABLE 確定時点で:
    expected_cells(t) = expected_cells(t-1) + delta_expected
    recognized_cells(t) = confirmed_board の non-EMPTY/non-UNKNOWN cell 数
    abs_err(t) = |expected_cells(t) - recognized_cells(t)|

  delta_expected の決定ルール:
    - TSUMO 後 STABLE: +2 (= 1 ツモ分の 2 ぷよ落下)
    - CHAIN 後 STABLE: 直前 STABLE と現 STABLE の recognized_cells 差を使用
      (連鎖消滅量は物理シミュレーション不要; 認識値で直接更新)
    - OJAMA 後 STABLE: 同様に recognized_cells 差を使用
    - STABLE→STABLE (= 置き直後): +2 想定

  最終 diff (1 試合 total):
    absolute_error_mean = mean(abs_err over all stable transitions)
    absolute_error_max  = max(abs_err)
    tsumo_err_count     = ツモ後 STABLE で |expected - recognized| > TSUMO_ERR_THRESHOLD
                          (= 2 cell が盤面に現れなかった count)

FAIL 判定:
  absolute_error_mean > FAIL_ABS_ERR_MEAN_THRESHOLD

Usage:
    PYTHONPATH=. python scripts/measure_chain_corrected_cells.py \\
        --video data/match_clips/v40/v40_match01.mp4

    # 全 clip 一括:
    PYTHONPATH=. python scripts/measure_chain_corrected_cells.py \\
        --video-dir data/match_clips \\
        --output data/eval/chain_corrected_cells.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
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

# RecognitionPipeline 呼び出し間隔 (秒)
RECOG_SCAN_INTERVAL_SEC: float = 0.033

# ツモ後 STABLE で |expected - recognized| がこれを超えるとツモ誤認とカウント
TSUMO_ERR_THRESHOLD: int = 1

# 試合内で abs_err_mean がこれを超えると FAIL 判定
FAIL_ABS_ERR_MEAN_THRESHOLD: float = 3.0

# STABLE 観測の最低回数 (試合開始直後の空盤面除外用)
MIN_STABLE_WARMUP_COUNT: int = 3

# 進捗ログ間隔 (秒)
PROGRESS_LOG_INTERVAL_SEC: float = 15.0

# 1920×1080 期待解像度
EXPECTED_HEIGHT: int = 1080
EXPECTED_WIDTH: int = 1920

# ツモ後 STABLE での期待 cell 増分
TSUMO_CELL_ADD: int = 2

# CHAIN / OJAMA 後 STABLE では expected を recognized に同期するため
# 「期待値を観測値で上書き」するモードを使う (= 物理シミュレーション不要)
_SYNC_STATES: frozenset[BoardState] = frozenset({
    BoardState.CHAIN,
    BoardState.OJAMA_FALL,
})


# ============================
# データクラス
# ============================


@dataclass
class StableTransition:
    """STABLE 確定時点の 1 サンプル。

    Attributes:
        time_sec: 発生時刻 (秒)
        prev_state: 直前の state (= このイベントの原因)
        recognized: 現 STABLE 盤面の non-EMPTY cell 数
        expected: 物理補正後の期待 cell 数
        abs_err: |expected - recognized|
        is_tsumo_transition: ツモ後 STABLE かどうか
        is_chain_transition: 連鎖後 STABLE かどうか
        is_ojama_transition: おじゃま降下後 STABLE かどうか
    """

    time_sec: float
    prev_state: str
    recognized: int
    expected: float
    abs_err: float
    is_tsumo_transition: bool
    is_chain_transition: bool
    is_ojama_transition: bool


@dataclass
class SideTracker:
    """1 サイドの STABLE 遷移を追跡するトラッカー。

    Attributes:
        side: "1P" or "2P"
        prev_state: 前 frame の BoardState
        prev_recognized: 前回 STABLE 確定時の recognized cell 数
        expected: 現時点の期待 cell 数 (累積更新)
        stable_count: STABLE 確定回数 (warmup 除外用)
        transitions: 記録済み StableTransition リスト
        _last_recorded_recognized: 前回 transition 記録時の recognized 値
            (STABLE 継続中の重複記録を防ぐため; -1 = 未記録)
    """

    side: str
    prev_state: BoardState = BoardState.MENU
    prev_recognized: int = 0
    expected: float = 0.0
    stable_count: int = 0
    transitions: list[StableTransition] = field(default_factory=list)
    _last_recorded_recognized: int = -1

    def update(
        self,
        cur_state: BoardState,
        confirmed_board,
        time_sec: float,
    ) -> None:
        """1 frame 分の更新処理。

        STABLE 確定時点でのみ遷移を記録する。
        ただし以下の条件で記録をスキップする:
          1. warmup 期間 (stable_count < MIN_STABLE_WARMUP_COUNT)
          2. STABLE 継続中かつ recognized cell 数が前回と同一
             (= 同一盤面の毎 frame 連続観測は 1 度だけ記録)

        Args:
            cur_state: 現 frame の BoardState
            confirmed_board: 現 STABLE 盤面 (STABLE 以外なら None)
            time_sec: 現 frame の時刻
        """
        if cur_state == BoardState.STABLE and confirmed_board is not None:
            self.stable_count += 1
            cur_recognized = _count_non_empty(confirmed_board)

            # STABLE 継続中に recognized が変化していない場合はスキップ
            # (= 同一盤面を毎 frame 再観測しているだけ)
            stable_continuing = (
                self.prev_state == BoardState.STABLE
                and cur_recognized == self._last_recorded_recognized
            )
            if not stable_continuing and self.stable_count >= MIN_STABLE_WARMUP_COUNT:
                prev = self.prev_state
                is_tsumo = (prev == BoardState.TSUMO_FALL)
                is_chain = (prev == BoardState.CHAIN)
                is_ojama = (prev == BoardState.OJAMA_FALL)
                # STABLE→STABLE で recognized が変化した場合は着地直後とみなす
                is_stable_to_stable = (
                    prev == BoardState.STABLE
                    or prev == BoardState.EFFECT
                )

                # expected の更新
                if is_tsumo or is_stable_to_stable:
                    # ツモ後 or 着地直後: +2 期待
                    self.expected += TSUMO_CELL_ADD
                elif is_chain or is_ojama:
                    # 連鎖後 / おじゃま後: expected を recognized に同期
                    # (消滅量・降下量を直接観測で取り込む)
                    self.expected = float(cur_recognized)

                abs_err = abs(self.expected - cur_recognized)

                tr = StableTransition(
                    time_sec=time_sec,
                    prev_state=prev.value,
                    recognized=cur_recognized,
                    expected=self.expected,
                    abs_err=abs_err,
                    is_tsumo_transition=is_tsumo,
                    is_chain_transition=is_chain,
                    is_ojama_transition=is_ojama,
                )
                self.transitions.append(tr)
                self._last_recorded_recognized = cur_recognized

            self.prev_recognized = cur_recognized

        # 状態更新: STABLE 以外の state は常に更新、
        # STABLE は「STABLE → STABLE」継続として記録する
        if cur_state != BoardState.STABLE or self.prev_state != BoardState.STABLE:
            self.prev_state = cur_state
        # STABLE が継続している間は prev_state を STABLE のまま維持


# ============================
# ヘルパー
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
    """指定秒のフレームを読み込んで 1080p に変換する。失敗時 None。"""
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return _resize_to_1080p(frame)


def _count_non_empty(board) -> int:
    """Board の non-EMPTY / non-UNKNOWN cell 数を返す。"""
    count = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            v = int(board.get(r, c))
            if v != COLOR_EMPTY and v != COLOR_UNKNOWN:
                count += 1
    return count


def _inject_per_video_hsv(pipeline: RecognitionPipeline, video_path: Path) -> None:
    """動画 ID から per-video HSV を pipeline に inject する。

    scripts/measure_tsumo_vs_recognition.py と同一ロジック。
    inject 失敗は silent skip。
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


# ============================
# 評価コア
# ============================


def _compute_transitions_stats(transitions: list[StableTransition]) -> dict:
    """StableTransition リストから統計 dict を計算する。

    Args:
        transitions: STABLE 確定イベントのリスト

    Returns:
        abs_err_mean / abs_err_max / tsumo_err_count 等を含む dict
    """
    if not transitions:
        return {
            "transition_count": 0,
            "abs_err_mean": 0.0,
            "abs_err_max": 0.0,
            "tsumo_transition_count": 0,
            "tsumo_err_count": 0,
            "chain_transition_count": 0,
            "ojama_transition_count": 0,
        }

    errs = [tr.abs_err for tr in transitions]
    tsumo_trs = [tr for tr in transitions if tr.is_tsumo_transition]
    chain_trs = [tr for tr in transitions if tr.is_chain_transition]
    ojama_trs = [tr for tr in transitions if tr.is_ojama_transition]
    tsumo_errs = [
        tr for tr in tsumo_trs if tr.abs_err > TSUMO_ERR_THRESHOLD
    ]

    return {
        "transition_count": len(transitions),
        "abs_err_mean": round(float(np.mean(errs)), 4),
        "abs_err_max": round(float(np.max(errs)), 4),
        "tsumo_transition_count": len(tsumo_trs),
        "tsumo_err_count": len(tsumo_errs),
        "chain_transition_count": len(chain_trs),
        "ojama_transition_count": len(ojama_trs),
    }


def run_recognition_and_track(
    pipeline: RecognitionPipeline,
    cap: cv2.VideoCapture,
    start_sec: float,
    end_sec: float,
    fps: float,
) -> tuple[SideTracker, SideTracker]:
    """試合区間を認識しながら 1P/2P の STABLE 遷移を追跡する。

    Args:
        pipeline: RecognitionPipeline インスタンス (呼び出し元で reset 済み)
        cap: 動画キャプチャ
        start_sec: 試合開始秒
        end_sec: 試合終了秒
        fps: 動画の fps (進捗ログ用)

    Returns:
        (tracker_1p, tracker_2p) の tuple
    """
    tracker_1p = SideTracker(side="1P")
    tracker_2p = SideTracker(side="2P")

    t = start_sec
    last_log_t = start_sec

    while t <= end_sec:
        frame = _read_frame_at(cap, t)
        if frame is None:
            t += RECOG_SCAN_INTERVAL_SEC
            continue

        frame_idx = int(t * fps)
        result = pipeline.update(frame_idx, t, frame)

        tracker_1p.update(result.p1.state, result.p1.confirmed_board, t)
        tracker_2p.update(result.p2.state, result.p2.confirmed_board, t)

        # 進捗ログ
        if t - last_log_t >= PROGRESS_LOG_INTERVAL_SEC:
            elapsed = t - start_sec
            total = end_sec - start_sec
            p1_cnt = len(tracker_1p.transitions)
            p2_cnt = len(tracker_2p.transitions)
            print(
                f"  [track] t={t:.1f}s ({elapsed:.0f}/{total:.0f}s) "
                f"1P_transitions={p1_cnt} 2P_transitions={p2_cnt}",
                flush=True,
            )
            last_log_t = t

        t += RECOG_SCAN_INTERVAL_SEC

    return tracker_1p, tracker_2p


# ============================
# 1 試合評価
# ============================


def evaluate_match_chain_corrected(
    match_idx: int,
    tracker_1p: SideTracker,
    tracker_2p: SideTracker,
    start_sec: float,
    end_sec: float,
) -> dict:
    """1 試合の連鎖補正版評価 dict を返す。

    Args:
        match_idx: 試合番号 (1 オリジン)
        tracker_1p: 1P SideTracker (計測済み)
        tracker_2p: 2P SideTracker (計測済み)
        start_sec: 試合開始秒
        end_sec: 試合終了秒

    Returns:
        JSON シリアライズ可能な評価 dict
    """
    stats_1p = _compute_transitions_stats(tracker_1p.transitions)
    stats_2p = _compute_transitions_stats(tracker_2p.transitions)

    # 両サイド統合の abs_err_mean
    all_errs: list[float] = (
        [tr.abs_err for tr in tracker_1p.transitions]
        + [tr.abs_err for tr in tracker_2p.transitions]
    )
    combined_err_mean = (
        round(float(np.mean(all_errs)), 4) if all_errs else 0.0
    )
    verdict = "FAIL" if combined_err_mean > FAIL_ABS_ERR_MEAN_THRESHOLD else "PASS"

    return {
        "match_idx": match_idx,
        "start_sec": round(start_sec, 2),
        "end_sec": round(end_sec, 2),
        "duration_sec": round(end_sec - start_sec, 2),
        "combined_abs_err_mean": combined_err_mean,
        "verdict": verdict,
        "p1": stats_1p,
        "p2": stats_2p,
    }


# ============================
# 1 clip 動画の計測
# ============================


def measure_video(video_path: Path) -> dict:
    """1 動画 (= 1 clip) を連鎖補正版評価で計測して返す。

    Args:
        video_path: clip 動画パス

    Returns:
        {video, matches, summary} の dict
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / max(fps, 1.0)
        print(
            f"[measure_cc] {video_path.name}: {duration_sec:.1f}s fps={fps:.1f}",
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
        pipeline.reset()

        tracker_1p, tracker_2p = run_recognition_and_track(
            pipeline=pipeline,
            cap=cap,
            start_sec=0.0,
            end_sec=duration_sec,
            fps=fps,
        )

        print(
            f"  [result] 1P: {len(tracker_1p.transitions)} transitions, "
            f"2P: {len(tracker_2p.transitions)} transitions",
            flush=True,
        )

        result = evaluate_match_chain_corrected(
            match_idx=1,
            tracker_1p=tracker_1p,
            tracker_2p=tracker_2p,
            start_sec=0.0,
            end_sec=duration_sec,
        )

    finally:
        cap.release()

    summary = {
        "match_count": 1,
        "combined_abs_err_mean": result["combined_abs_err_mean"],
        "verdict": result["verdict"],
    }

    return {
        "video": video_path.name,
        "matches": [result],
        "summary": summary,
    }


# ============================
# 全体統計
# ============================


def _compute_overall_summary(results: list[dict]) -> dict:
    """複数動画の summary から全体統計を計算する。"""
    if not results:
        return {
            "video_count": 0,
            "match_count": 0,
            "combined_abs_err_mean": 0.0,
            "verdict": "N/A",
        }

    all_errs: list[float] = []
    fail_count = 0
    total_matches = 0

    for r in results:
        for m in r.get("matches", []):
            # 個別 transition の abs_err は summary に含まれていないが、
            # combined_abs_err_mean を重み付き平均で集計する
            tcount = (
                m.get("p1", {}).get("transition_count", 0)
                + m.get("p2", {}).get("transition_count", 0)
            )
            if tcount > 0:
                all_errs.extend(
                    [m["combined_abs_err_mean"]] * tcount
                )
            total_matches += 1
            if m.get("verdict") == "FAIL":
                fail_count += 1

    overall_mean = round(float(np.mean(all_errs)), 4) if all_errs else 0.0

    return {
        "video_count": len(results),
        "match_count": total_matches,
        "combined_abs_err_mean": overall_mean,
        "fail_match_count": fail_count,
        "verdict": "FAIL" if overall_mean > FAIL_ABS_ERR_MEAN_THRESHOLD else "PASS",
    }


# ============================
# CLI
# ============================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="measure_chain_corrected_cells",
        description=(
            "連鎖消滅・おじゃま降下を考慮した STABLE セル数精度評価"
        ),
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


def main(argv: Optional[list[str]] = None) -> int:
    """CLI エントリポイント。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.video is not None:
        videos = [args.video]
    else:
        videos = _collect_videos(args.video_dir)
        print(
            f"[measure_cc] {len(videos)} clip(s) 収集: {args.video_dir}",
            flush=True,
        )

    if not videos:
        print("[ERROR] 動画ファイルが見つかりません", file=sys.stderr)
        return 1

    all_results: list[dict] = []
    for vp in videos:
        try:
            res = measure_video(vp)
            all_results.append(res)
            err_mean = res["summary"]["combined_abs_err_mean"]
            verdict = res["summary"]["verdict"]
            print(
                f"[result] {vp.name}: abs_err_mean={err_mean:.4f} {verdict}",
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
        print(f"\n[done] -> {args.output}", flush=True)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))

    print(
        f"\n[overall] abs_err_mean={overall['combined_abs_err_mean']:.4f} "
        f"({overall.get('fail_match_count', 0)}/{overall.get('match_count', 0)} FAIL) "
        f"-> {overall['verdict']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
