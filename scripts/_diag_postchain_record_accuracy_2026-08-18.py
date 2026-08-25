"""「連鎖後/おじゃま着弾後に記録される盤面」が正しいかの実測 (2026-08-18)。

user の問い:「連鎖中の誤認を対策する意味ある?」→ 検討の結果「意味は薄い」
(連鎖中は記録しない方針のため)。 意味があるのは「連鎖後に記録される確定盤面
(=npz に載る値) が誤りを引きずっているか」だけ。

本スクリプトは本番構成 (src.production_config.collect_flags() 相当、
1手区切り観測スケジューラ + 持続的物理制約フィルタ + 色スワップ拒否が全て
有効) で 3 動画の対戦区間を処理し、collect_boards_lean.py の内部関数
(_SideState / _should_emit / _update_move_scheduler /
_update_physics_transition_marker) をそのままインポートして駆動する
(本番の記録タイミング判定ロジックを一切再実装しない = 測定器自体のズレを防ぐ)。

出力:
  logs/_diag_postchain_record_accuracy_2026-08-18_<tag>_frames.jsonl
      毎フレームの (side, t_sec, state, confirmed_grid, emitted有無) ログ。
      残存誤りが何秒で解消するかの事後分析に使う。
  logs/_diag_postchain_record_accuracy_2026-08-18_<tag>_moves.json
      実際に「記録された」盤面 (=本番 npz に載る値) のリスト。
      kind: "chain"=直前に CHAIN/GRAVITY_SETTLE を経由, "ojama"=OJAMA_FALL
      を経由, "normal"=どちらも経由せず (通常の1手)。
  logs/_diag_postchain_record_accuracy_2026-08-18_ALL_moves.json
      3動画分を結合したもの (分析スクリプトの入力)。

コードは変更しない (診断専用)。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.board_state_machine import BoardState  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.production_config import (  # noqa: E402
    GHOST_CHAIN_RULE_ENABLED,
    recognition_load_default_kwargs,
)
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.collect_boards_lean import (  # noqa: E402
    _SideState,
    _should_emit,
    _update_move_scheduler,
    _update_physics_transition_marker,
)

_CHAIN_STATES = {BoardState.CHAIN, BoardState.GRAVITY_SETTLE}
_TAG = "2026-08-18"


def _grid_of(board) -> "list | None":
    return board.to_dict()["grid"] if board is not None else None


def run_window(video_path: str, start_sec: float, end_sec: float, tag: str) -> tuple[list, list]:
    """1 動画・1 窓を処理し、(moves, transitions) を返す。"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(round(start_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    kwargs = recognition_load_default_kwargs()
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,  # 本番 --with-next (capture_next=True) 相当
        force_in_match=True,
        **kwargs,
    )
    vid_match = re.search(r"(v\d+|video_\d+|c\d+)", Path(video_path).stem)
    if vid_match and hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(vid_match.group(1))

    physics_sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)

    sides = {"1P": _SideState(), "2P": _SideState()}
    prev_bstate = {"1P": BoardState.MENU, "2P": BoardState.MENU}
    states_since_emit: dict[str, set] = {"1P": set(), "2P": set()}
    last_chain_end: dict[str, "tuple | None"] = {"1P": None, "2P": None}
    last_ojama_end: dict[str, "tuple | None"] = {"1P": None, "2P": None}

    frame_log_path = Path(f"logs/_diag_postchain_record_accuracy_{_TAG}_{tag}_frames.jsonl")
    frame_log_path.parent.mkdir(parents=True, exist_ok=True)
    frame_fp = open(frame_log_path, "w", encoding="utf-8")

    moves: list[dict] = []
    transitions: list[dict] = []

    fi = 0
    frame_idx_out = start_frame
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = start_sec + fi / fps
        if t_sec > end_sec:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        result = pipeline.update(frame_idx_out, t_sec, frame)
        get_tsumo_count = getattr(pipeline, "tsumo_count", None)

        for side, res in (("1P", result.p1), ("2P", result.p2)):
            bstate = res.state
            board = res.confirmed_board
            tsumo_count = get_tsumo_count(side) if callable(get_tsumo_count) else None
            next_pair = res.next_pair
            state = sides[side]

            if prev_bstate[side] in _CHAIN_STATES and bstate == BoardState.STABLE:
                last_chain_end[side] = (t_sec, frame_idx_out)
                transitions.append({
                    "video": tag, "side": side, "kind": "chain_end",
                    "t_sec": round(t_sec, 3), "frame_idx": frame_idx_out,
                })
            if prev_bstate[side] == BoardState.OJAMA_FALL and bstate == BoardState.STABLE:
                last_ojama_end[side] = (t_sec, frame_idx_out)
                transitions.append({
                    "video": tag, "side": side, "kind": "ojama_end",
                    "t_sec": round(t_sec, 3), "frame_idx": frame_idx_out,
                })

            if bstate != BoardState.STABLE:
                states_since_emit[side].add(bstate)

            # 本番と同じ呼び出し順序 (_process_side_lean と同一)
            _update_physics_transition_marker(state, bstate, True, tsumo_count=tsumo_count)
            _update_move_scheduler(state, next_pair, tsumo_count, bstate, frame_idx_out, True)

            emitted = False
            if board is not None and bstate == BoardState.STABLE:
                if _should_emit(
                    state, board, bstate, exclude_phantom=True,
                    enable_move_segmented_recording=True, frame_idx=frame_idx_out,
                    enable_physics_persistence_filter=True, physics_sim=physics_sim,
                ):
                    kinds = states_since_emit[side]
                    if BoardState.OJAMA_FALL in kinds:
                        move_kind = "ojama"
                    elif kinds & _CHAIN_STATES:
                        move_kind = "chain"
                    else:
                        move_kind = "normal"
                    moves.append({
                        "video": tag, "side": side, "t_sec": round(t_sec, 3),
                        "frame_idx": frame_idx_out, "kind": move_kind,
                        "grid": _grid_of(board), "tsumo_count": tsumo_count,
                        "chain_end_t_sec": (
                            last_chain_end[side][0]
                            if move_kind == "chain" and last_chain_end[side] else None
                        ),
                        "ojama_end_t_sec": (
                            last_ojama_end[side][0]
                            if move_kind == "ojama" and last_ojama_end[side] else None
                        ),
                        "prior_states": sorted(s.value for s in kinds),
                    })
                    state.last_emitted_grid = board._grid.tobytes()
                    state.move_window_recorded = True
                    states_since_emit[side] = set()
                    emitted = True

            frame_fp.write(json.dumps({
                "t_sec": round(t_sec, 3), "side": side, "state": bstate.value,
                "frame_idx": frame_idx_out, "emitted": emitted,
                "grid": _grid_of(board) if board is not None else None,
            }, ensure_ascii=False) + "\n")

        prev_bstate["1P"] = result.p1.state
        prev_bstate["2P"] = result.p2.state
        fi += 1
        frame_idx_out += 1
        if fi % 900 == 0:
            print(f"  [{tag}] progress t={t_sec:.1f}s ({len(moves)} moves so far)")

    cap.release()
    frame_fp.close()
    print(f"[{tag}] done: {len(moves)} moves, {len(transitions)} transitions "
          f"-> {frame_log_path}")
    return moves, transitions


def main() -> None:
    jobs = [
        ("data/frames/video_36.mp4", 118.0, 340.0, "video36_118_340"),
        ("data/frames/video_52.mp4", 129.0, 330.0, "video52_129_330"),
        ("data/frames/video_c100.mp4", 570.0, 660.0, "c100_570_660"),
    ]
    all_moves: list[dict] = []
    all_transitions: list[dict] = []
    for video_path, s, e, tag in jobs:
        if not Path(video_path).exists():
            print(f"[skip] {video_path} not found")
            continue
        print(f"=== {tag}: {video_path} [{s},{e}] ===")
        moves, transitions = run_window(video_path, s, e, tag)
        Path(f"logs/_diag_postchain_record_accuracy_{_TAG}_{tag}_moves.json").write_text(
            json.dumps(moves, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        Path(f"logs/_diag_postchain_record_accuracy_{_TAG}_{tag}_transitions.json").write_text(
            json.dumps(transitions, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        all_moves.extend(moves)
        all_transitions.extend(transitions)
    Path(f"logs/_diag_postchain_record_accuracy_{_TAG}_ALL_moves.json").write_text(
        json.dumps(all_moves, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    Path(f"logs/_diag_postchain_record_accuracy_{_TAG}_ALL_transitions.json").write_text(
        json.dumps(all_transitions, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    by_kind: dict[str, int] = {}
    for m in all_moves:
        by_kind[m["kind"]] = by_kind.get(m["kind"], 0) + 1
    print("=== 全体集計 (記録された手数の内訳) ===")
    print(json.dumps(by_kind, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
