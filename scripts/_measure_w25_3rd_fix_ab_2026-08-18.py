"""W25根治 第3弾・最終 (2026-08-18) の28チャンクA/B測定。

アーキ指定の検証観点 (d) を計測する:
  - 物差しv2 stage1/stage2 acc は別スクリプト
    (scripts/_score_yardstick_v2_w25_2026-08-17.py 相当を第3弾用に流用)
    で計測する (本スクリプトの対象外)。
  - 終端おじゃま数 proxy (false negative チェック、第2弾と同型)。
  - OJAMA_FALL entry/exit 回数 (振動率不変の確認)。
  - 検証観点(b): 本物のおじゃま着弾 (空セル起点) の反映遅延分布
    (OJAMA_FALL entry frame_idx から、そのセルが最終的に9で確定する
    までの frame 数) をF/F+第3弾で比較する (8フレーム基準からの退行有無)。

src/ は一切変更しない (計装は外部からの直接読み出しのみ)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._measure_w25_3rd_fix_ab_2026-08-18
"""
from __future__ import annotations

import importlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent

_bc = importlib.import_module("scripts._collect_yardstick_v2_bc_2026-08-15")
_diag = importlib.import_module("scripts._diag_c13c22_recheck_2026-08-17")

from src.board import COLOR_EMPTY, COLOR_OJAMA, HIDDEN_ROWS, BOARD_ROWS, BOARD_COLS  # noqa: E402

OUT_PATH = _ROOT / "data" / "verify" / "diag_c13c22_recheck_2026-08-17" / "w25_3rd_fix_ab.json"

MAX_PARALLEL_WORKERS: int = 12


def _count_ojama(board) -> int:  # noqa: ANN001
    n = 0
    for r in range(HIDDEN_ROWS, BOARD_ROWS):
        for c in range(BOARD_COLS):
            if int(board.get(r, c)) == COLOR_OJAMA:
                n += 1
    return n


def build_pipeline_3rd():  # noqa: ANN201
    from src.recognition_pipeline import RecognitionPipeline
    return RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        load_next_detector=True,
        temporal_smoothing=1,
        force_in_match=False,
        enable_effect_gate=True,
        enable_burst_guard_v2=True,
        enable_transition_merge_guard=True,
        burst_gate_open_threshold=0.954,
        enable_hidden_row_burst_guard=True,
        enable_match_transition_debounce=True,
        enable_ojama_fall_placement_override=True,
        enable_patch_fp_hsv_guard=True,
        enable_chain_tracker=True,
        enable_floating_gap_restore=True,
        enable_landing_color_guard=True,
        enable_override_color_guard=True,
        enable_ojama_column_stack_fix=True,
        enable_next_history_starvation_fix=True,
        enable_ojama_write_accounting_guard=True,
    )


def analyze_chunk(video_id: str, chunk_idx: int, start_sec: float, use_flag: bool) -> dict:
    filename = _bc.video_filename_of(video_id)
    video_path = _bc.VIDEO_DIR / filename
    cap = cv2.VideoCapture(str(video_path))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * real_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pipeline = build_pipeline_3rd() if use_flag else _diag.build_pipeline()

    n_ojama_entry = {"1P": 0, "2P": 0}
    n_ojama_exit = {"1P": 0, "2P": 0}
    prev_state = {"1P": None, "2P": None}
    final_ojama_count = {"1P": 0, "2P": 0}

    # 検証観点(b): 空セル起点のおじゃま着弾反映遅延 (OJAMA_FALL entry から
    # そのセルが 9 で確定するまでの frame 数) を全セル・全episodeで収集する。
    entry_frame_idx: dict[str, int | None] = {"1P": None, "2P": None}
    prev_confirmed_snapshot: dict[str, object] = {"1P": None, "2P": None}
    reflection_delays: dict[str, list[int]] = {"1P": [], "2P": []}
    pending_new_ojama_cells: dict[str, dict] = {"1P": {}, "2P": {}}

    frame_idx = start_frame
    t_sec = start_sec
    n_frames = 0
    CHUNK_SEC = 30.0
    while t_sec < start_sec + CHUNK_SEC:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        res = pipeline.update(frame_idx, t_sec, frame)
        for side, side_res in (("1P", res.p1), ("2P", res.p2)):
            cur = side_res.state
            prv = prev_state[side]
            if prv is not None and prv.name != "OJAMA_FALL" and cur.name == "OJAMA_FALL":
                n_ojama_entry[side] += 1
                entry_frame_idx[side] = frame_idx
            if prv is not None and prv.name == "OJAMA_FALL" and cur.name != "OJAMA_FALL":
                n_ojama_exit[side] += 1
            prev_state[side] = cur

            if side_res.confirmed_board is not None:
                final_ojama_count[side] = _count_ojama(side_res.confirmed_board)
                prev_c = prev_confirmed_snapshot[side]
                if prev_c is not None and entry_frame_idx[side] is not None:
                    for r in range(HIDDEN_ROWS, BOARD_ROWS):
                        for c in range(BOARD_COLS):
                            key = (r, c)
                            was_empty_before_entry = (
                                key not in pending_new_ojama_cells[side]
                                and int(prev_c.get(r, c)) == COLOR_EMPTY
                            )
                            cur_v = int(side_res.confirmed_board.get(r, c))
                            if was_empty_before_entry and cur_v == COLOR_OJAMA:
                                # 空セルが今回初めて9になった → 反映遅延を記録。
                                delay = frame_idx - entry_frame_idx[side]
                                reflection_delays[side].append(delay)
                                pending_new_ojama_cells[side][key] = True
                prev_confirmed_snapshot[side] = side_res.confirmed_board.copy()

        n_frames += 1
        frame_idx += 1
        t_sec = frame_idx / real_fps

    cap.release()
    return {
        "video_id": video_id, "chunk_idx": chunk_idx, "use_flag": use_flag,
        "n_frames": n_frames,
        "ojama_fall_entry_1p": n_ojama_entry["1P"],
        "ojama_fall_entry_2p": n_ojama_entry["2P"],
        "ojama_fall_exit_1p": n_ojama_exit["1P"],
        "ojama_fall_exit_2p": n_ojama_exit["2P"],
        "final_ojama_count_1p": final_ojama_count["1P"],
        "final_ojama_count_2p": final_ojama_count["2P"],
        "reflection_delays_1p": reflection_delays["1P"],
        "reflection_delays_2p": reflection_delays["2P"],
    }


def build_jobs() -> list[tuple[str, int, float, bool]]:
    jobs = []
    duration_cache: dict[str, float] = {}
    for video_id, chunk_idx in _bc.NEEDED_CHUNKS:
        filename = _bc.video_filename_of(video_id)
        video_path = _bc.VIDEO_DIR / filename
        if not video_path.exists():
            continue
        if video_id not in duration_cache:
            duration_cache[video_id] = _bc.probe_duration_sec(video_path)
        duration = duration_cache[video_id]
        frac = _bc.CHUNK_OFFSET_FRACTIONS[chunk_idx]
        start_sec = max(0.0, frac * duration)
        for use_flag in (False, True):
            jobs.append((video_id, chunk_idx, start_sec, use_flag))
    return jobs


def _run_job(job: tuple[str, int, float, bool]) -> dict:
    video_id, chunk_idx, start_sec, use_flag = job
    try:
        r = analyze_chunk(video_id, chunk_idx, start_sec, use_flag)
        print(f"[OK] {video_id}_chunk{chunk_idx} flag={use_flag}", flush=True)
        return r
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {video_id}_chunk{chunk_idx} flag={use_flag}: {e}", flush=True)
        return {"video_id": video_id, "chunk_idx": chunk_idx, "use_flag": use_flag, "error": str(e)}


def main() -> None:
    jobs = build_jobs()
    print(f"ジョブ数: {len(jobs)} (workers={MAX_PARALLEL_WORKERS})", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as ex:
        for r in ex.map(_run_job, jobs):
            results.append(r)
            print(f"  進捗: {len(results)}/{len(jobs)}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT_PATH}")

    by_key = {}
    for r in results:
        if "error" in r:
            continue
        key = (r["video_id"], r["chunk_idx"])
        by_key.setdefault(key, {})[r["use_flag"]] = r

    print("\n=== (d) false negative proxy: 終端おじゃまセル数比較 (チャンク単位) ===")
    n_decreased = n_increased = n_same = 0
    for key, v in sorted(by_key.items()):
        if False not in v or True not in v:
            continue
        f_total = v[False]["final_ojama_count_1p"] + v[False]["final_ojama_count_2p"]
        w_total = v[True]["final_ojama_count_1p"] + v[True]["final_ojama_count_2p"]
        if w_total < f_total:
            n_decreased += 1
            print(f"  {key}: F={f_total} -> F+3rd={w_total} (下振れ、要確認)")
        elif w_total > f_total:
            n_increased += 1
            print(f"  {key}: F={f_total} -> F+3rd={w_total} (上振れ)")
        else:
            n_same += 1
    print(f"  下振れ={n_decreased} 上振れ={n_increased} 同一={n_same} (全{len(by_key)}チャンク)")

    print("\n=== (d) OJAMA_FALL 遷移回数/振動率 (entry/exit) F vs F+3rd ===")
    n_transition_diff = 0
    for key, v in sorted(by_key.items()):
        if False not in v or True not in v:
            continue
        f_entry = v[False]["ojama_fall_entry_1p"] + v[False]["ojama_fall_entry_2p"]
        w_entry = v[True]["ojama_fall_entry_1p"] + v[True]["ojama_fall_entry_2p"]
        f_exit = v[False]["ojama_fall_exit_1p"] + v[False]["ojama_fall_exit_2p"]
        w_exit = v[True]["ojama_fall_exit_1p"] + v[True]["ojama_fall_exit_2p"]
        if f_entry != w_entry or f_exit != w_exit:
            n_transition_diff += 1
            print(f"  {key}: entry F={f_entry}/W={w_entry} exit F={f_exit}/W={w_exit} (差異あり)")
    print(f"  entry/exit回数が F と F+3rd で異なるチャンク数: {n_transition_diff}/{len(by_key)}")

    print("\n=== (b) 反映遅延分布 (空セル起点の新規おじゃま確定まで, frame数) ===")
    for use_flag, label in ((False, "F"), (True, "F+3rd")):
        all_delays = []
        for r in results:
            if r.get("use_flag") != use_flag or "error" in r:
                continue
            all_delays.extend(r.get("reflection_delays_1p", []))
            all_delays.extend(r.get("reflection_delays_2p", []))
        if all_delays:
            all_delays.sort()
            n = len(all_delays)
            median = all_delays[n // 2]
            p90 = all_delays[int(n * 0.9)] if n > 1 else all_delays[0]
            n_over_8 = sum(1 for d in all_delays if d > 8)
            print(f"  [{label}] n={n} median={median} p90={p90} max={max(all_delays)} "
                  f"8フレーム超過={n_over_8} ({100*n_over_8/n:.1f}%)")
        else:
            print(f"  [{label}] n=0 (該当イベント無し)")


if __name__ == "__main__":
    main()
