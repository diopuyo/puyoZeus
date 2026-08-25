"""STABLE持続ゲート A/B 高速診断・計測プローブ (2026-08-18、一時スクリプト)。

user指示: 3動画で高速にゲート適正値 (窓長 STABLE_PERSISTENCE_WINDOW_SEC /
閾値 STABLE_PERSISTENCE_DIFF_THRESHOLD、src/board_motion.py) を決める。

## 設計方針 (高速化の要)
1動画につき RecognitionPipeline を **1回だけ** 駆動すれば全設定パターンを
評価できるようにする。理由: `_should_emit` (scripts/collect_boards_lean.py)
の dedup 状態 (last_emitted_grid) は「実際に emit した」時にのみ更新される
ため、STABLE候補列 (bstate==STABLE かつ非空かつ非幻盤面) と生ピクセル
diff系列さえ記録しておけば、任意の (窓長, 閾値, top行除外有無) について
post-hoc に dedup を再生 (replay) して行数を再計算できる。実際の採否ロジック
(`_should_emit`) は一切書き換えず、`state.last_emitted_grid` を一時的に
None にしてから呼ぶことで dedup を無効化した「素の候補判定」を得る
(モンキーパッチはランタイムのみ、src/ や collect_boards_lean.py のファイル
自体は無改変)。

## 記録するログ (video_id ごとに 1 npz)
- candidates_side / candidates_tsec / candidates_grid: STABLE候補列
  (ゲート・dedup適用前)
- bstate_side / bstate_tsec / bstate_value: 両side・毎フレームの bstate
  (相手が連鎖中か・自分が直近連鎖中かの判定に使う)
- diff_side / diff_tsec / diff_full / diff_no_top: 生ピクセル差分系列
  (ゲートON/OFFに関わらず毎フレーム計算する。diff_no_top は盤面ROI最上段
  TOP_EXCLUDE_VISIBLE_ROWS 行を除いた版、user指摘「ツモ落下域を含む」の
  裏取り用)

分析は _ab_stable_persistence_gate_analyze_2026-08-18.py が別途行う。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.collect_boards_lean as clb  # noqa: E402
from src.board_motion import board_roi_gray, frame_diff_mean  # noqa: E402
from src.board import VISIBLE_ROWS  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION  # noqa: E402
import src.production_config as prod_config  # noqa: E402

# 盤面ROI最上段の除外行数 (2026-08-18 診断用): ツモの軸+子の2セル分。
# user仮説「ツモ落下域を含む」の検証用であり本番定数ではない。
TOP_EXCLUDE_VISIBLE_ROWS: int = 2
_CELL_HEIGHT_PX: float = DEFAULT_P1_REGION.height / VISIBLE_ROWS
TOP_EXCLUDE_ROWS_PX: int = int(round(TOP_EXCLUDE_VISIBLE_ROWS * _CELL_HEIGHT_PX))


def _production_kwargs_excluding_gate() -> dict:
    """production_config.collect_flags() を collect_lean() 用 kwargs に変換する。

    recognition_load_default_kwargs() と同じ変換規則 (フラグ名の -> _ 置換、
    値なしは True、値ありは float) を、collect専用フラグも含めて汎用適用する。
    --enable-stable-persistence-gate は本プローブが専用に制御するため除外する。
    """
    tokens = prod_config.collect_flags().split()
    kwargs: dict = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        name = tok.lstrip("-").replace("-", "_")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            kwargs[name] = float(tokens[i + 1])
            i += 2
        else:
            kwargs[name] = True
            i += 1
    kwargs.pop("enable_stable_persistence_gate", None)
    return kwargs


class _Recorder:
    """1動画分のログを保持する (video_id ごとに使い捨て)。"""

    def __init__(self) -> None:
        self.candidates: dict[str, list[tuple[float, int, bytes]]] = {"1P": [], "2P": []}
        self.bstate: dict[str, list[tuple[float, str]]] = {"1P": [], "2P": []}
        self.diffs: dict[str, list[tuple[float, float, float]]] = {"1P": [], "2P": []}


def _make_instrumented_update_raw_pixel_stable(rec: _Recorder):
    """常に diff を計算・記録し、採否は常に True (OFF相当) を返す差替え版。"""

    def _instrumented(state, frame, side_label, t_sec, enable_stable_persistence_gate):
        gray = board_roi_gray(frame, side_label)
        if state.motion_prev_gray is not None:
            diff_full = frame_diff_mean(state.motion_prev_gray, gray)
            diff_no_top = frame_diff_mean(
                state.motion_prev_gray[TOP_EXCLUDE_ROWS_PX:, :],
                gray[TOP_EXCLUDE_ROWS_PX:, :],
            )
            rec.diffs[side_label].append((t_sec, diff_full, diff_no_top))
        state.motion_prev_gray = gray
        return True

    return _instrumented


def _make_instrumented_process_side_lean(rec: _Recorder, orig_process_side, orig_should_emit):
    """候補列 (dedup前) と bstate 時系列を記録してから元関数を呼ぶ差替え版。"""

    def _instrumented(acc, state, side_label, board, bstate, score, video_id,
                       t_sec, frame_idx, **kwargs):
        rec.bstate[side_label].append((t_sec, bstate.value))
        exclude_phantom = kwargs.get("exclude_phantom", False)
        if board is not None:
            saved_last = state.last_emitted_grid
            state.last_emitted_grid = None  # dedup無効化して素の候補判定を得る
            is_candidate = orig_should_emit(
                state, board, bstate, exclude_phantom=exclude_phantom,
                raw_pixel_stable=True,
            )
            state.last_emitted_grid = saved_last
            if is_candidate:
                rec.candidates[side_label].append(
                    (t_sec, frame_idx, board._grid.tobytes())
                )
        kwargs["raw_pixel_stable"] = True  # このプローブ実行自体はOFF相当で駆動
        return orig_process_side(
            acc, state, side_label, board, bstate, score, video_id,
            t_sec, frame_idx, **kwargs,
        )

    return _instrumented


def run_probe(video_path: Path, out_pkl: Path, max_sec: float, start_sec: float) -> None:
    """1動画を1回だけ処理し、候補列・bstate系列・diff系列を pkl に保存する。"""
    rec = _Recorder()
    orig_update_raw = clb._update_raw_pixel_stable
    orig_process_side = clb._process_side_lean
    orig_should_emit = clb._should_emit
    clb._update_raw_pixel_stable = _make_instrumented_update_raw_pixel_stable(rec)
    clb._process_side_lean = _make_instrumented_process_side_lean(
        rec, orig_process_side, orig_should_emit,
    )
    try:
        kwargs = _production_kwargs_excluding_gate()
        dummy_out = out_pkl.with_suffix(".dummy.npz")
        n = clb.collect_lean(
            video_path=video_path,
            out_npz=dummy_out,
            max_sec=max_sec,
            start_sec=start_sec,
            enable_stable_persistence_gate=True,  # パッチ済のため実質無関係
            **kwargs,
        )
        if dummy_out.exists():
            dummy_out.unlink()
    finally:
        clb._update_raw_pixel_stable = orig_update_raw
        clb._process_side_lean = orig_process_side

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(
            {
                "video_id": video_path.stem,
                "max_sec": max_sec,
                "start_sec": start_sec,
                "top_exclude_rows_px": TOP_EXCLUDE_ROWS_PX,
                "off_equivalent_row_count": n,
                "candidates": rec.candidates,
                "bstate": rec.bstate,
                "diffs": rec.diffs,
            },
            f,
        )
    print(f"[probe] {video_path.name}: off_equivalent_rows={n} "
          f"candidates_1P={len(rec.candidates['1P'])} "
          f"candidates_2P={len(rec.candidates['2P'])} -> {out_pkl}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out-pkl", required=True, type=Path)
    parser.add_argument("--max-sec", type=float, default=1800.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    args = parser.parse_args()
    run_probe(args.video, args.out_pkl, args.max_sec, args.start_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
