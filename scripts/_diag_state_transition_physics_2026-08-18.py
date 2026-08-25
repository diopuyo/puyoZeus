"""状態遷移の物理制約による誤認検出率の実測 (2026-08-18)。

user指摘: 「おじゃま落下時は色ぷよが降るわけないので、状態単体でなく状態遷移
(直前の確定盤面との差分) を見れば物理制約で誤認を検出できるはず」。

3状態の許容差分ルール:
  - OJAMA_FALL: 空セル→9 のみ許容。それ以外の新規非空化・色変化は違反。
  - CHAIN / GRAVITY_SETTLE: 減る (非空→空) 以外は許容しない。増加・色変化は違反。
  - TSUMO_FALL→STABLE 着地: 新規非空化セルはちょうど2個、かつ色は次のツモ
    (next_pair) と一致するはず (best-effort、次ツモの左右2点対応まではしない)。

コードは変更しない。RecognitionPipeline を直接叩き、
raw cnn_board (会計整合フィルタ適用前、src.ojama_write_accounting の
関数をモンキーパッチして捕捉) と confirmed_board (state machine確定値、
= viz/npz に実際に載る値) を毎フレーム記録して事後分析する。

出力: logs/_diag_state_transition_physics_2026-08-18_<tag>.jsonl (frame log)
      logs/_diag_state_transition_physics_2026-08-18_<tag>_summary.json (集計)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recognition_pipeline import RecognitionPipeline, BoardState  # noqa: E402
from src.production_config import recognition_load_default_kwargs  # noqa: E402
import src.ojama_write_accounting as owa  # noqa: E402

COLOR_NAMES = {0: "空", 1: "赤", 2: "青", 3: "緑", 4: "黄", 5: "紫", 9: "お邪魔", 10: "UNK"}

# ----------------------------------------------------------------------
# monkeypatch: 会計整合フィルタの raw (フィルタ前) board を捕捉する
# ----------------------------------------------------------------------
_CAPTURE: dict = {}  # id(memory-dict) -> {"raw": grid, "filtered": grid}
_orig_apply = owa.apply_ojama_write_accounting_filter


def _wrapped_apply(cnn_board, memory, credit, duration_by_cell=None):
    raw_grid = cnn_board.to_dict()["grid"]
    filtered = _orig_apply(cnn_board, memory, credit, duration_by_cell)
    _CAPTURE[id(memory)] = {
        "raw": raw_grid,
        "filtered": filtered.to_dict()["grid"],
    }
    return filtered


owa.apply_ojama_write_accounting_filter = _wrapped_apply


def _grid_of(board) -> list:
    return board.to_dict()["grid"] if board is not None else None


def run_video(video_path: str, start_sec: float, end_sec: float, tag: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    kwargs = recognition_load_default_kwargs()
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
        **kwargs,
    )

    mem_ids = {
        "1P": id(pipeline._stable_color_memory_1p),
        "2P": id(pipeline._stable_color_memory_2p),
    }

    log_path = Path(f"logs/_diag_state_transition_physics_2026-08-18_{tag}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "w", encoding="utf-8")

    # side別の直近ベースライン(最後にSTABLEで確定したconfirmed_board grid)
    baseline = {"1P": None, "2P": None}
    # side別: 現在の非STABLE区間に入った時点のbaseline (区間中固定)
    episode_baseline = {"1P": None, "2P": None}
    episode_state = {"1P": None, "2P": None}
    prev_state = {"1P": None, "2P": None}

    violations = {
        "1P": {"ojama_fall": [], "chain": [], "tsumo_landing": []},
        "2P": {"ojama_fall": [], "chain": [], "tsumo_landing": []},
    }

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
        for side, res in (("1P", result.p1), ("2P", result.p2)):
            state = res.state
            cap_entry = _CAPTURE.get(mem_ids[side])
            raw_grid = cap_entry["raw"] if cap_entry is not None else None
            filtered_grid = cap_entry["filtered"] if cap_entry is not None else None

            # 非STABLE区間の開始検知 → episode baseline 固定
            if state != BoardState.STABLE and prev_state[side] != state:
                if baseline[side] is not None:
                    episode_baseline[side] = baseline[side]
                    episode_state[side] = state

            if state == BoardState.STABLE and res.confirmed_board is not None:
                baseline[side] = _grid_of(res.confirmed_board)
                episode_baseline[side] = None
                episode_state[side] = None

            # OJAMA_FALL 違反判定 (raw board vs episode baseline)
            if (state == BoardState.OJAMA_FALL and episode_baseline[side] is not None
                    and raw_grid is not None):
                base = episode_baseline[side]
                for r in range(13):
                    for c in range(6):
                        b = base[r][c]
                        cur = raw_grid[r][c]
                        if b == 0 and cur in (1, 2, 3, 4, 5):
                            violations[side]["ojama_fall"].append(
                                {"t_sec": round(t_sec, 3), "r": r, "c": c,
                                 "kind": "empty_to_color", "before": b, "after": cur})
                        elif b in (1, 2, 3, 4, 5) and cur != b and cur != 0:
                            violations[side]["ojama_fall"].append(
                                {"t_sec": round(t_sec, 3), "r": r, "c": c,
                                 "kind": "color_changed", "before": b, "after": cur})

            # CHAIN / GRAVITY_SETTLE 違反判定 (raw board vs episode baseline)
            if (state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE)
                    and episode_baseline[side] is not None and raw_grid is not None):
                base = episode_baseline[side]
                for r in range(13):
                    for c in range(6):
                        b = base[r][c]
                        cur = raw_grid[r][c]
                        if b == 0 and cur != 0:
                            violations[side]["chain"].append(
                                {"t_sec": round(t_sec, 3), "r": r, "c": c,
                                 "kind": "empty_to_nonempty", "before": b, "after": cur,
                                 "state": state.value})
                        elif b != 0 and cur != b and cur != 0:
                            violations[side]["chain"].append(
                                {"t_sec": round(t_sec, 3), "r": r, "c": c,
                                 "kind": "color_changed", "before": b, "after": cur,
                                 "state": state.value})

            # TSUMO_FALL→STABLE 着地チェック (best-effort)
            if (prev_state[side] == BoardState.TSUMO_FALL and state == BoardState.STABLE
                    and episode_baseline[side] is not None and res.confirmed_board is not None):
                base = episode_baseline[side]
                new_grid = _grid_of(res.confirmed_board)
                added = []
                for r in range(13):
                    for c in range(6):
                        if base[r][c] == 0 and new_grid[r][c] != 0:
                            added.append((r, c, new_grid[r][c]))
                np_pair = res.next_pair
                ok_landing = (len(added) == 2)
                if np_pair is not None and ok_landing:
                    added_colors = sorted(v for (_, _, v) in added)
                    expect_colors = sorted(np_pair)
                    if added_colors != expect_colors:
                        ok_landing = False
                elif np_pair is None:
                    ok_landing = None  # next_pair 不明のため判定不能
                if ok_landing is False:
                    violations[side]["tsumo_landing"].append(
                        {"t_sec": round(t_sec, 3), "added": added,
                         "next_pair": np_pair})

            log_fp.write(json.dumps({
                "t_sec": round(t_sec, 3), "side": side, "state": state.value,
                "raw_grid": raw_grid, "confirmed_grid": (
                    _grid_of(res.confirmed_board) if res.confirmed_board is not None else None
                ),
            }, ensure_ascii=False) + "\n")
            prev_state[side] = state
        fi += 1
        frame_idx_out += 1
        if fi % 300 == 0:
            print(f"  [{tag}] progress t={t_sec:.1f}s")

    cap.release()
    log_fp.close()

    summary = {
        "tag": tag, "video": video_path, "start_sec": start_sec, "end_sec": end_sec,
        "ojama_fall_violations": {
            s: len(violations[s]["ojama_fall"]) for s in ("1P", "2P")
        },
        "chain_violations": {
            s: len(violations[s]["chain"]) for s in ("1P", "2P")
        },
        "tsumo_landing_violations": {
            s: len(violations[s]["tsumo_landing"]) for s in ("1P", "2P")
        },
        "detail": violations,
    }
    out_path = Path(f"logs/_diag_state_transition_physics_2026-08-18_{tag}_summary.json")
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{tag}] done. summary -> {out_path}")
    return summary


def main() -> None:
    jobs = [
        ("data/frames/video_36.mp4", 100.0, 160.0, "video36_100_160"),
        ("data/frames/video_52.mp4", 100.0, 160.0, "video52_100_160"),
        ("data/frames/video_c100.mp4", 570.0, 660.0, "c100_570_660"),
    ]
    all_summary = []
    for video_path, s, e, tag in jobs:
        if not Path(video_path).exists():
            print(f"[skip] {video_path} not found")
            continue
        print(f"=== {tag}: {video_path} [{s},{e}] ===")
        summ = run_video(video_path, s, e, tag)
        all_summary.append({k: v for k, v in summ.items() if k != "detail"})
    Path("logs/_diag_state_transition_physics_2026-08-18_ALL_summary.json").write_text(
        json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
