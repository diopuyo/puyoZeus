"""指標B (連鎖数の食い違い検出器) 試作 + 測定器健全性チェック (2026-08-17)。

## これは何か
`src/chain_count_truth.select_chain_count_high_confidence_band` (得点逆算・
テロップ非依存) が「高信頼」と判定した連鎖数と、`ChainSimulator.simulate()`
(認識盤面からの連鎖数再構成) を突き合わせ、食い違いを「認識誤りの疑い」候補
として集計する。

`scripts/_measure_chain_count_truth_coverage_2026-08-14.py` (カバレッジ測定、
16動画・data/indicators_v2/boards_lean_phase_l_2026-08-11/) が既に集計値
(n_high_confidence=541, agree=100, disagree=441) を出しているため、本スクリプト
はそれを流用しつつ、
    (a) 食い違いの規模分布 (score側 - sim側の差、方向) を追加集計する
    (b) 動画ファイルが手元に残っている3本 (c109/c13/c96) に絞り、食い違いの
        大きいイベント上位を実フレーム (盤面ROI) で切り出し、目視検証できる
        形にする (測定器事故8件の教訓、数値だけで信用しない)

## 使い方
    PYTHONPATH=. ./venv/bin/python -m \\
        scripts._diag_indicatorB_chain_mismatch_2026-08-17 --summary
    PYTHONPATH=. ./venv/bin/python -m \\
        scripts._diag_indicatorB_chain_mismatch_2026-08-17 --extract-frames
"""
from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.board import Board
from src.chain import ChainSimulator
from src.chain_count_truth import select_chain_count_high_confidence_band
from src.chain_detector import ERASURE_MIN_DROP
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.production_config import GHOST_CHAIN_RULE_ENABLED

_ROOT = Path(__file__).resolve().parent.parent
_cov = importlib.import_module("scripts._measure_chain_count_truth_coverage_2026-08-14")

NPZ_DIR: Path = _cov.DEFAULT_NPZ_DIR
VIDEO_IDS: tuple[str, ...] = _cov.DEFAULT_VIDEO_IDS

# 動画ファイルが手元に残存している3本のみ (ストレージ節約ルールで大半は削除済み)。
VIDEO_FILENAME_OF: dict[str, str] = {
    "c109": "video_c109.mp4", "c13": "video_c13.mp4", "c96": "video_c96.mp4",
}
VIDEO_DIR: Path = Path.home() / "frames"
OUT_DIR: Path = _ROOT / "data" / "verify" / "chain_count_v2_2026-08-14" / "diag_indicatorB_2026-08-17"
STD_WIDTH, STD_HEIGHT = 1920, 1080
# 実フレーム抽出で目視検証する上位件数 (工数とのバランス、W8等の前例に倣う)。
N_SPOTCHECK_EVENTS: int = 6


def _events_in_file_with_frames(npz_path: Path) -> list[dict]:
    """`_measure_chain_count_truth_coverage_2026-08-14._events_in_file` の複製+拡張。

    frame_idx (トリガー行/発火前行の両方) を追加で保持する点のみが差分
    (使い捨て診断ツール間の複製慣行、元スクリプトの同名コメント参照)。
    """
    d = np.load(npz_path, allow_pickle=True)
    if "chain_trigger_sec" not in d.files or "chain_mechanism" not in d.files:
        return []
    grids = d["grids"]
    side = d["side"]
    game_idx = d["game_idx"]
    t_sec = d["t_sec"]
    score = d["score"]
    frame_idx = d["frame_idx"]
    trigger = d["chain_trigger_sec"]
    mechanism = d["chain_mechanism"]
    nz_counts = (grids != 0).sum(axis=(1, 2)).tolist()

    groups: dict[tuple, list[int]] = {}
    for i in range(len(grids)):
        key = (str(side[i]), int(game_idx[i]))
        groups.setdefault(key, []).append(i)

    sim = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    events: list[dict] = []
    for (side_key, game_idx_key), idxs in groups.items():
        idxs.sort(key=lambda i: float(t_sec[i]))
        prev_trigger_sec: float | None = None
        for pos in range(len(idxs)):
            i = idxs[pos]
            if not np.isfinite(trigger[i]):
                prev_trigger_sec = None
                continue
            tag = str(mechanism[i]).strip().lower()
            if tag in _cov._NO_CHAIN_TAG_VALUES:
                prev_trigger_sec = None
                continue
            if prev_trigger_sec is not None and float(trigger[i]) == prev_trigger_sec:
                continue
            prev_trigger_sec = float(trigger[i])
            before_i = _cov._find_before_board_index(nz_counts, idxs, pos)
            if before_i is None:
                continue
            before_board = Board.from_list(grids[before_i].tolist())
            if before_board.is_dead():
                continue
            sim_result = sim.simulate(before_board)
            if sim_result.chain_count < 1:
                continue
            delta_score = int(score[i]) - int(score[before_i])
            events.append({
                "side": side_key,
                "game_idx": int(game_idx_key),
                "t_sec": float(t_sec[i]),
                "delta_score": delta_score,
                "sim_chain_count": int(sim_result.chain_count),
                "trigger_frame_idx": int(frame_idx[i]),
                "before_frame_idx": int(frame_idx[before_i]),
                "before_grid": grids[before_i].tolist(),
            })
    return events


def _collect_high_confidence_mismatches(video_ids: tuple[str, ...]) -> list[dict]:
    """全動画から高信頼帯イベントを集め、score/sim 双方の連鎖数を付与して返す。"""
    out: list[dict] = []
    for vid in video_ids:
        p = NPZ_DIR / f"{vid}.npz"
        if not p.is_file():
            continue
        for ev in _events_in_file_with_frames(p):
            hc = select_chain_count_high_confidence_band(ev["delta_score"])
            if hc.reason != "high_confidence":
                continue
            ev = {**ev, "video_id": vid, "score_chain_count": hc.chain_count, "score_ratio": hc.ratio}
            out.append(ev)
    return out


def cmd_summary() -> None:
    """(a) 食い違いの規模分布・方向を集計する。"""
    events = _collect_high_confidence_mismatches(VIDEO_IDS)
    print(f"[指標B] 高信頼帯イベント総数: {len(events)}")
    diffs: Counter[int] = Counter()
    n_agree = n_score_gt_sim = n_score_lt_sim = 0
    for ev in events:
        diff = ev["score_chain_count"] - ev["sim_chain_count"]
        diffs[diff] += 1
        if diff == 0:
            n_agree += 1
        elif diff > 0:
            n_score_gt_sim += 1
        else:
            n_score_lt_sim += 1
    n_total = len(events)
    print(f"  一致(diff=0): {n_agree} ({n_agree / n_total:.1%})")
    print(f"  score側 > sim側 (simが過小評価の疑い): {n_score_gt_sim} ({n_score_gt_sim / n_total:.1%})")
    print(f"  score側 < sim側 (simが過大評価の疑い): {n_score_lt_sim} ({n_score_lt_sim / n_total:.1%})")
    print("  diff(score-sim)分布 (件数):")
    for diff in sorted(diffs):
        print(f"    diff={diff:+d}: {diffs[diff]}")
    out = {
        "n_total_high_confidence": n_total,
        "n_agree": n_agree,
        "n_score_gt_sim": n_score_gt_sim,
        "n_score_lt_sim": n_score_lt_sim,
        "diff_histogram": {str(k): v for k, v in sorted(diffs.items())},
    }
    out_path = OUT_DIR / "summary.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {out_path}")


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> "np.ndarray | None":
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (STD_HEIGHT, STD_WIDTH):
        frame = cv2.resize(frame, (STD_WIDTH, STD_HEIGHT), interpolation=cv2.INTER_AREA)
    return frame


def _grid_to_ascii(grid: list) -> str:
    """13x6 グリッドを目視しやすい ascii に変換 (行0=隠し段込み)。"""
    lines = []
    for r, row in enumerate(grid):
        lines.append(f"  r{r:>2}: " + " ".join(str(v) for v in row))
    return "\n".join(lines)


def cmd_extract_frames() -> None:
    """(b) 手元に動画がある3本に絞り、食い違い上位イベントの実フレームを保存する。"""
    events = _collect_high_confidence_mismatches(tuple(VIDEO_FILENAME_OF))
    mismatched = [ev for ev in events if ev["score_chain_count"] != ev["sim_chain_count"]]
    mismatched.sort(key=lambda ev: -abs(ev["score_chain_count"] - ev["sim_chain_count"]))
    top = mismatched[:N_SPOTCHECK_EVENTS]
    print(f"[指標B] 手元動画3本内の食い違いイベント: {len(mismatched)}件、"
          f"上位{len(top)}件を実フレーム抽出する")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    caps: dict[str, cv2.VideoCapture] = {}
    manifest = []
    for n, ev in enumerate(top):
        vid = ev["video_id"]
        vpath = VIDEO_DIR / VIDEO_FILENAME_OF[vid]
        cap = caps.setdefault(vid, cv2.VideoCapture(str(vpath)))
        reg = DEFAULT_P1_REGION if ev["side"] == "1P" else DEFAULT_P2_REGION

        tag = f"{n:02d}_{vid}_{ev['side']}_g{ev['game_idx']}_t{ev['t_sec']:.1f}"
        diff = ev["score_chain_count"] - ev["sim_chain_count"]
        print(f"\n=== {tag} ===")
        print(f"  delta_score={ev['delta_score']} score_ratio={ev['score_ratio']:.3f} "
              f"score_chain_count={ev['score_chain_count']} sim_chain_count={ev['sim_chain_count']} "
              f"diff={diff:+d}")
        print("  before_board (発火前盤面, simの入力):")
        print(_grid_to_ascii(ev["before_grid"]))

        frame_before = _read_frame(cap, ev["before_frame_idx"])
        if frame_before is not None:
            roi = frame_before[reg.y: reg.y + reg.height, reg.x: reg.x + reg.width]
            cv2.imwrite(str(OUT_DIR / f"{tag}_before_frame_f{ev['before_frame_idx']}.png"), frame_before)
            cv2.imwrite(str(OUT_DIR / f"{tag}_before_roi_f{ev['before_frame_idx']}.png"), roi)
        frame_trigger = _read_frame(cap, ev["trigger_frame_idx"])
        if frame_trigger is not None:
            roi_t = frame_trigger[reg.y: reg.y + reg.height, reg.x: reg.x + reg.width]
            cv2.imwrite(str(OUT_DIR / f"{tag}_trigger_frame_f{ev['trigger_frame_idx']}.png"), frame_trigger)
            cv2.imwrite(str(OUT_DIR / f"{tag}_trigger_roi_f{ev['trigger_frame_idx']}.png"), roi_t)

        manifest.append({**{k: v for k, v in ev.items() if k != "before_grid"}, "tag": tag, "diff": diff})

    for cap in caps.values():
        cap.release()
    (OUT_DIR / "spotcheck_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] -> {OUT_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--extract-frames", action="store_true")
    args = ap.parse_args()
    if args.summary:
        cmd_summary()
        return
    if args.extract_frames:
        cmd_extract_frames()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
