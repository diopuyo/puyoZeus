"""1手区切り観測スケジューラの設計実測 (2026-08-18、コーダエージェント計装)。

coordinator からの3回の追加依頼をまとめて検証する:
  (1) user提案: 手の区切り信号を tsumo_count 増分でなく NEXT 繰り上がり
      (next1_a/next1_b の変化) を主軸にする案の妥当性確認。
      「盤面固定」と「NEXT繰り上がり」のどちらが先に観測されるかを実測する
      (ゲーム内部の順序も user 未確定とのことなので実測で決める)。
  (2) 落下中ツモの写り込みリスク: 着地確定 (本スクリプトでは選定した
      区切りイベント) から 0/5/10/15 フレーム後、盤面上部の「浮遊セル」
      (src.board_rules.clear_floating_above_gap と同じ列走査ロジックだが
      gap=1 も報告する拡張版) がどれだけ残っているかを測る。
  (3) NEXT履歴突き合わせによる色検証: 盤面に新規出現した2セルの色が
      直前の next_pair の色集合と一致するかを測る (不一致 = 認識誤りの
      可能性)。おじゃま着弾/相手連鎖と重なる場合を層別する。

対象: data/frames/video_{36,52,c100}.mp4 (物理制約実測と同じ3動画)。
実装しない (計装のみ)。RecognitionPipeline を毎フレーム (stride無し) 駆動する
ため、--max-sec で処理区間を絞る (既定 200 秒/本)。

実行:
    PYTHONPATH=. python scripts/_diag_move_segmentation_design_2026-08-18.py 36
    PYTHONPATH=. python scripts/_diag_move_segmentation_design_2026-08-18.py --merge
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN, HIDDEN_ROWS,
)
from src.production_config import recognition_load_default_kwargs  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

TARGET_W: int = 1920
TARGET_H: int = 1080
NPZ_DIR_NOTE = "data/frames"
FRAMES_DIR = Path("data/frames")
OUT_DIR = Path("logs")
TARGET_VIDEOS: tuple[str, ...] = ("36", "52", "c100")
DEFAULT_MAX_SEC: float = 200.0

# 着地確定からの経過フレームオフセット (coordinator指定: 0, 5, 10, 15)
FLOATING_RISK_OFFSETS: tuple[int, ...] = (0, 5, 10, 15)
# T-N イベント突合の許容窓 (フレーム、前後どちらの方向にも探索)
EVENT_MATCH_WINDOW_FRAMES: int = 45


@dataclass
class FrameRecord:
    fi: int
    t_sec: float
    bstate: str
    tsumo_count: int
    next_a: int
    next_b: int
    grid: "list[list[int]]"  # cnn_board (drift detector 入力、floating除去済み観測)
    confirmed_grid: "list[list[int]] | None"
    chain_trigger_finite: bool
    chain_mechanism: str


def _process_video(video_id: str, max_sec: float, start_sec: float = 0.0) -> dict:
    path = FRAMES_DIR / f"video_{video_id}.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[skip] {path} を開けません")
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * fps)
    if start_frame > 0:
        for _ in range(start_frame):
            ok_skip, _ = cap.read()
            if not ok_skip:
                break
    end_frame = (
        min(total_frames, start_frame + int(max_sec * fps))
        if max_sec > 0 else total_frames
    )

    # 本番採用フラグ群 (production_config.RECOGNITION_ADOPTED) を実装用に
    # 適用する (2026-08-18 追加修正)。当初これを付けずに実行したところ
    # OJAMA_FALL 滞在が82%に達し TSUMO_FALL が一度も観測されない異常な
    # 挙動になった (未修正状態は既知の振動バグの影響を受ける)。
    pipeline_kwargs = dict(recognition_load_default_kwargs())
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        load_next_detector=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        force_in_match=True,
        **pipeline_kwargs,
    )
    vid_match = __import__("re").search(r"(v\d+|video_\d+)", path.name)
    if vid_match and hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(vid_match.group(1))

    records: dict[str, list[FrameRecord]] = {"1P": [], "2P": []}
    fi = start_frame
    n_report = max(1, (end_frame - start_frame) // 10)
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)
        for side_label, side in (("1P", result.p1), ("2P", result.p2)):
            tsumo_count = pipeline.tsumo_count(side_label)
            na, nb = (side.next_pair if side.next_pair is not None else (-1, -1))
            trigger = getattr(side.chain_event, "trigger_sec", None) if side.chain_event else None
            mech = getattr(side.chain_event, "mechanism", None) if side.chain_event else ""
            records[side_label].append(FrameRecord(
                fi=fi, t_sec=t_sec, bstate=side.state.value,
                tsumo_count=int(tsumo_count),
                next_a=int(na), next_b=int(nb),
                grid=side.cnn_board._grid.tolist() if side.cnn_board is not None else None,
                confirmed_grid=(
                    side.confirmed_board._grid.tolist()
                    if side.confirmed_board is not None else None
                ),
                chain_trigger_finite=bool(trigger is not None and np.isfinite(trigger)),
                chain_mechanism=str(mech or ""),
            ))
        if fi % n_report == 0:
            print(f"  [{video_id}] frame {fi}/{end_frame} ({fi / end_frame * 100:.0f}%)")
        fi += 1
    cap.release()
    return {
        "video": video_id, "fps": fps,
        "records": {k: [asdict(r) for r in v] for k, v in records.items()},
    }


# =============================================================================
# (1) T-event (tsumo_count増分) vs N-event (next_pair変化) の順序
# =============================================================================


def _detect_t_events(recs: list[dict]) -> list[int]:
    """tsumo_count が前フレームから増分した frame index (recs内のindex) の一覧。"""
    out = []
    for i in range(1, len(recs)):
        if recs[i]["tsumo_count"] > recs[i - 1]["tsumo_count"]:
            out.append(i)
    return out


def _detect_n_events(recs: list[dict]) -> list[int]:
    """next_pair が前フレームから変化した frame index の一覧 (未検出 -1 同士は除く)。"""
    out = []
    for i in range(1, len(recs)):
        prev = (recs[i - 1]["next_a"], recs[i - 1]["next_b"])
        cur = (recs[i]["next_a"], recs[i]["next_b"])
        if prev != cur and cur != (-1, -1) and prev != (-1, -1):
            out.append(i)
    return out


def measure_ordering(all_recs: dict[str, dict[str, list[dict]]]) -> dict:
    """T-event と最近傍 N-event の frame offset (N_fi - T_fi) を全動画全sideで集計する。"""
    offsets: list[int] = []
    n_matched = 0
    n_t_total = 0
    detail: list[dict] = []
    for vid, by_side in all_recs.items():
        for side, recs in by_side.items():
            t_idxs = _detect_t_events(recs)
            n_idxs = _detect_n_events(recs)
            n_t_total += len(t_idxs)
            for ti in t_idxs:
                t_fi = recs[ti]["fi"]
                best = None
                for ni in n_idxs:
                    n_fi = recs[ni]["fi"]
                    if abs(n_fi - t_fi) <= EVENT_MATCH_WINDOW_FRAMES:
                        if best is None or abs(n_fi - t_fi) < abs(best - t_fi):
                            best = n_fi
                if best is not None:
                    n_matched += 1
                    offsets.append(best - t_fi)
                    detail.append({
                        "video": vid, "side": side, "t_fi": t_fi, "n_fi": best,
                        "offset": best - t_fi,
                    })
    arr = np.array(offsets) if offsets else np.array([])
    return {
        "n_t_events_total": n_t_total,
        "n_matched": n_matched,
        "match_rate": (n_matched / n_t_total * 100) if n_t_total else None,
        "offset_mean": float(arr.mean()) if arr.size else None,
        "offset_median": float(np.median(arr)) if arr.size else None,
        "offset_negative_rate_n_before_t": (
            float((arr < 0).sum() / arr.size * 100) if arr.size else None
        ),
        "offset_positive_rate_n_after_t": (
            float((arr > 0).sum() / arr.size * 100) if arr.size else None
        ),
        "offset_zero_rate_same_frame": (
            float((arr == 0).sum() / arr.size * 100) if arr.size else None
        ),
        "detail_sample": detail[:20],
    }


# =============================================================================
# (2) 落下中ツモの写り込みリスク (floating-cell gap 走査)
# =============================================================================


def _scan_column_gap(grid: "list[list[int]]", col: int) -> "tuple[int, int, int] | None":
    """1列を下から走査し (stack_top, gap_size, n_floating_above) を返す。

    src.board_rules.clear_floating_above_gap と同じ走査ロジックだが、
    min_gap 閾値を適用せず gap の実測値をそのまま返す拡張版。
    """
    start_row = HIDDEN_ROWS
    stack_top = BOARD_ROWS
    for row in range(BOARD_ROWS - 1, start_row - 1, -1):
        v = grid[row][col]
        if v in (COLOR_EMPTY, COLOR_UNKNOWN):
            break
        stack_top = row
    if stack_top == BOARD_ROWS:
        return None
    gap_start = stack_top - 1
    if gap_start < start_row:
        return (stack_top, 0, 0)
    gap = 0
    scan_row = gap_start
    while scan_row >= start_row and grid[scan_row][col] in (COLOR_EMPTY, COLOR_UNKNOWN):
        gap += 1
        scan_row -= 1
    n_floating = 0
    r = scan_row
    while r >= start_row:
        if grid[r][col] not in (COLOR_EMPTY, COLOR_UNKNOWN):
            n_floating += 1
        r -= 1
    return (stack_top, gap, n_floating)


def _count_floating_by_gap(grid: "list[list[int]]") -> dict:
    """盤面全体で gap==1 (残存リスク) / gap>=2 (既存フィルタで除去される) の
    浮遊セル数を数える。"""
    n_gap1 = 0
    n_gap2plus = 0
    for col in range(BOARD_COLS):
        res = _scan_column_gap(grid, col)
        if res is None:
            continue
        _stack_top, gap, n_floating = res
        if n_floating == 0:
            continue
        if gap == 1:
            n_gap1 += n_floating
        elif gap >= 2:
            n_gap2plus += n_floating
    return {"gap1_residual": n_gap1, "gap2plus_removed_by_existing_filter": n_gap2plus}


def measure_floating_risk(
    all_recs: dict[str, dict[str, list[dict]]], use_n_events: bool,
) -> dict:
    """区切りイベント (N-event優先、なければT-event) から 0/5/10/15 フレーム後の
    浮遊セルリスクを集計する。"""
    by_offset: dict[int, list[dict]] = {off: [] for off in FLOATING_RISK_OFFSETS}
    n_events_used = 0
    for vid, by_side in all_recs.items():
        for side, recs in by_side.items():
            events = _detect_n_events(recs) if use_n_events else _detect_t_events(recs)
            for ei in events:
                n_events_used += 1
                for off in FLOATING_RISK_OFFSETS:
                    j = ei + off
                    if j >= len(recs) or recs[j]["grid"] is None:
                        continue
                    stats = _count_floating_by_gap(recs[j]["grid"])
                    by_offset[off].append(stats)
    out = {"n_events_used": n_events_used, "by_offset": {}}
    for off, stats_list in by_offset.items():
        if not stats_list:
            out["by_offset"][off] = {"n": 0}
            continue
        gap1 = [s["gap1_residual"] for s in stats_list]
        gap2 = [s["gap2plus_removed_by_existing_filter"] for s in stats_list]
        out["by_offset"][off] = {
            "n": len(stats_list),
            "gap1_residual_rate_any": sum(1 for g in gap1 if g > 0) / len(gap1) * 100,
            "gap1_residual_mean": float(np.mean(gap1)),
            "gap2plus_rate_any": sum(1 for g in gap2 if g > 0) / len(gap2) * 100,
        }
    return out


# =============================================================================
# (3) NEXT履歴突き合わせによる色検証
# =============================================================================


def measure_next_color_crosscheck(all_recs: dict[str, dict[str, list[dict]]]) -> dict:
    """T-event (着地確定) 直前の next_pair 色と、新規出現した2セルの色を突合する。"""
    n_checked = 0
    n_mismatch = 0
    mismatch_categories: dict[str, int] = {
        "clean": 0, "ojama_overlap": 0, "opponent_chain_overlap": 0, "own_chain_overlap": 0,
    }
    for vid, by_side in all_recs.items():
        for side, recs in by_side.items():
            t_idxs = _detect_t_events(recs)
            for ti in t_idxs:
                if ti == 0:
                    continue
                prev = recs[ti - 1]
                cur = recs[ti]
                prev_next = (prev["next_a"], prev["next_b"])
                if -1 in prev_next:
                    continue
                prev_grid = prev["confirmed_grid"]
                cur_grid = cur["confirmed_grid"]
                if prev_grid is None or cur_grid is None:
                    continue
                new_colors = set()
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        b = prev_grid[r][c]
                        a = cur_grid[r][c]
                        if b == COLOR_EMPTY and 1 <= a <= 5:
                            new_colors.add(a)
                if not new_colors:
                    continue
                n_checked += 1
                expected = set(prev_next)
                is_mismatch = not new_colors.issubset(expected)
                if is_mismatch:
                    n_mismatch += 1
                    if cur["chain_trigger_finite"] and cur["chain_mechanism"]:
                        mismatch_categories["own_chain_overlap"] += 1
                    elif any(
                        prev_grid[r][c] == COLOR_EMPTY and cur_grid[r][c] == COLOR_OJAMA
                        for r in range(BOARD_ROWS) for c in range(BOARD_COLS)
                    ):
                        mismatch_categories["ojama_overlap"] += 1
                    else:
                        mismatch_categories["clean"] += 1
    return {
        "n_checked": n_checked,
        "n_mismatch": n_mismatch,
        "mismatch_rate": (n_mismatch / n_checked * 100) if n_checked else None,
        "mismatch_categories": mismatch_categories,
    }


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--merge":
        merged: dict[str, dict] = {}
        for vid in TARGET_VIDEOS:
            p = OUT_DIR / f"_diag_move_segmentation_design_2026-08-18_{vid}.json"
            if not p.exists():
                print(f"[skip] {p} なし")
                continue
            merged[vid] = json.loads(p.read_text(encoding="utf-8"))["records"]
        ordering = measure_ordering(merged)
        floating_n = measure_floating_risk(merged, use_n_events=True)
        floating_t = measure_floating_risk(merged, use_n_events=False)
        crosscheck = measure_next_color_crosscheck(merged)
        report = {
            "ordering": ordering,
            "floating_risk_by_n_event": floating_n,
            "floating_risk_by_t_event": floating_t,
            "next_color_crosscheck": crosscheck,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        out_path = OUT_DIR / "_diag_move_segmentation_design_2026-08-18_merged_report.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[保存] {out_path}")
        return

    if not args:
        print("使い方: python _diag_move_segmentation_design_2026-08-18.py <video_id|--merge> [max_sec]")
        sys.exit(1)
    video_id = args[0]
    max_sec = float(args[1]) if len(args) > 1 else DEFAULT_MAX_SEC
    start_sec = float(args[2]) if len(args) > 2 else 0.0
    data = _process_video(video_id, max_sec, start_sec=start_sec)
    if not data:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"_diag_move_segmentation_design_2026-08-18_{video_id}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[保存] {out_path}")


if __name__ == "__main__":
    main()
