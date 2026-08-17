"""W25根治 第2弾 (2026-08-17) の追加観点A/B測定。

アーキ指定の3観点を28チャンク (scripts/_collect_yardstick_v2_bc_2026-08-15
.NEEDED_CHUNKS と同一) で計測する:

  1. drift-resync 発火回数の比較 (F vs F+本フラグ、ON/OFF)。
     sm.reset(keep_match_state=True) の実発火回数 (monkeypatchで直接計数、
     _check_baseline_broken_reset 由来の sm.reset(keep_match_state=False)
     とは kwargs で区別する) + 新カウンタ
     (_drift_resync_ojama_warmup_suppressed_*) の発火分布。
  2. false negative チェック (資金会計 vs confirmed_board のおじゃま個数):
     各チャンク終了時点の confirmed_board 内おじゃまセル総数を F / F+flag で
     比較する。resync 抑制が実着弾の反映を破壊していれば F+flag 側が
     系統的に下振れするはず (proxy 指標、厳密な会計突合ではない)。
  3. OJAMA_FALL 遷移回数/振動率 (entry回数・exit回数) が F/F+flag 間で
     不変であること (本ガードは detector ロジックに触れないため理論的に
     不変のはずだが実測で確認する)。

src/ は一切変更しない (monkeypatch による外部計数のみ)。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._measure_w25_2nd_fix_ab_2026-08-17
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
_verify = importlib.import_module("scripts._verify_w25_fix_2026-08-17")

import src.board_state_machine as bsm  # noqa: E402
from src.board import COLOR_OJAMA, HIDDEN_ROWS, BOARD_ROWS, BOARD_COLS  # noqa: E402

OUT_PATH = _ROOT / "data" / "verify" / "diag_c13c22_recheck_2026-08-17" / "w25_2nd_fix_ab.json"

CHUNK_SEC: float = 30.0
MAX_PARALLEL_WORKERS: int = 12


def _count_ojama(board) -> int:  # noqa: ANN001
    n = 0
    for r in range(HIDDEN_ROWS, BOARD_ROWS):
        for c in range(BOARD_COLS):
            if int(board.get(r, c)) == COLOR_OJAMA:
                n += 1
    return n


def analyze_chunk(video_id: str, chunk_idx: int, start_sec: float, use_flag: bool) -> dict:
    filename = _bc.video_filename_of(video_id)
    video_path = _bc.VIDEO_DIR / filename
    fps = _bc.probe_duration_sec  # unused directly; get fps via cv2 below
    cap = cv2.VideoCapture(str(video_path))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * real_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pipeline = (
        _verify.build_pipeline_w25flag() if use_flag else _diag.build_pipeline()
    )

    # sm.reset(keep_match_state=True) の実発火回数を monkeypatch で計数する
    # (keep_match_state=False の _check_baseline_broken_reset 経路とは区別)。
    _resync_reset_calls = {"1P": 0, "2P": 0}
    for side, sm in (("1P", pipeline._sm_1p), ("2P", pipeline._sm_2p)):
        orig_reset = sm.reset

        def _wrapped_reset(*a, _orig=orig_reset, _side=side, **kw):  # noqa: ANN002, ANN003
            if kw.get("keep_match_state") is True:
                _resync_reset_calls[_side] += 1
            return _orig(*a, **kw)

        sm.reset = _wrapped_reset

    n_ojama_entry = {"1P": 0, "2P": 0}
    n_ojama_exit = {"1P": 0, "2P": 0}
    prev_state = {"1P": None, "2P": None}

    frame_idx = start_frame
    t_sec = start_sec
    n_frames = 0
    final_ojama_count = {"1P": 0, "2P": 0}
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
            if prv is not None and prv.name == "OJAMA_FALL" and cur.name != "OJAMA_FALL":
                n_ojama_exit[side] += 1
            prev_state[side] = cur
            if side_res.confirmed_board is not None:
                final_ojama_count[side] = _count_ojama(side_res.confirmed_board)
        n_frames += 1
        frame_idx += 1
        t_sec = frame_idx / real_fps

    cap.release()
    return {
        "video_id": video_id, "chunk_idx": chunk_idx, "use_flag": use_flag,
        "n_frames": n_frames,
        "resync_reset_calls_1p": _resync_reset_calls["1P"],
        "resync_reset_calls_2p": _resync_reset_calls["2P"],
        "ojama_warmup_suppressed_1p": pipeline._drift_resync_ojama_warmup_suppressed_1p,
        "ojama_warmup_suppressed_2p": pipeline._drift_resync_ojama_warmup_suppressed_2p,
        "start_guard_suppressed_1p": pipeline._drift_resync_start_guard_suppressed_1p,
        "start_guard_suppressed_2p": pipeline._drift_resync_start_guard_suppressed_2p,
        "hsv_gate_suppressed_1p": pipeline._drift_resync_hsv_gate_suppressed_1p,
        "hsv_gate_suppressed_2p": pipeline._drift_resync_hsv_gate_suppressed_2p,
        "ojama_fall_entry_1p": n_ojama_entry["1P"],
        "ojama_fall_entry_2p": n_ojama_entry["2P"],
        "ojama_fall_exit_1p": n_ojama_exit["1P"],
        "ojama_fall_exit_2p": n_ojama_exit["2P"],
        "final_ojama_count_1p": final_ojama_count["1P"],
        "final_ojama_count_2p": final_ojama_count["2P"],
    }


def build_jobs() -> list[tuple[str, int, float, bool]]:
    jobs = []
    duration_cache: dict[str, float] = {}
    for video_id, chunk_idx in _bc.NEEDED_CHUNKS:
        filename = _bc.video_filename_of(video_id)
        video_path = _bc.VIDEO_DIR / filename
        if not video_path.exists():
            print(f"[skip] 動画が無い: {video_path}")
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
        print(f"[OK] {video_id}_chunk{chunk_idx} flag={use_flag}")
        return r
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {video_id}_chunk{chunk_idx} flag={use_flag}: {e}")
        return {"video_id": video_id, "chunk_idx": chunk_idx, "use_flag": use_flag, "error": str(e)}


def main() -> None:
    jobs = build_jobs()
    print(f"ジョブ数: {len(jobs)} (workers={MAX_PARALLEL_WORKERS})", flush=True)
    results = []
    # ProcessPoolExecutor (GIL 回避、真の並列化): 各ジョブは独立プロセスで
    # pipeline を構築するため、monkeypatch はプロセス内に閉じる (安全)。
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as ex:
        for r in ex.map(_run_job, jobs):
            results.append(r)
            print(f"  進捗: {len(results)}/{len(jobs)}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT_PATH}")

    # 集計
    by_key = {}
    for r in results:
        if "error" in r:
            continue
        key = (r["video_id"], r["chunk_idx"])
        by_key.setdefault(key, {})[r["use_flag"]] = r

    print("\n=== (1) drift-resync 発火回数比較 (F vs F+flag、全チャンク合計) ===")
    tot_reset_f = sum(
        v[False]["resync_reset_calls_1p"] + v[False]["resync_reset_calls_2p"]
        for v in by_key.values() if False in v
    )
    tot_reset_w25 = sum(
        v[True]["resync_reset_calls_1p"] + v[True]["resync_reset_calls_2p"]
        for v in by_key.values() if True in v
    )
    tot_ojama_warmup_suppressed = sum(
        v[True]["ojama_warmup_suppressed_1p"] + v[True]["ojama_warmup_suppressed_2p"]
        for v in by_key.values() if True in v
    )
    tot_start_guard_f = sum(
        v[False]["start_guard_suppressed_1p"] + v[False]["start_guard_suppressed_2p"]
        for v in by_key.values() if False in v
    )
    tot_start_guard_w25 = sum(
        v[True]["start_guard_suppressed_1p"] + v[True]["start_guard_suppressed_2p"]
        for v in by_key.values() if True in v
    )
    tot_hsv_gate_f = sum(
        v[False]["hsv_gate_suppressed_1p"] + v[False]["hsv_gate_suppressed_2p"]
        for v in by_key.values() if False in v
    )
    tot_hsv_gate_w25 = sum(
        v[True]["hsv_gate_suppressed_1p"] + v[True]["hsv_gate_suppressed_2p"]
        for v in by_key.values() if True in v
    )
    print(f"  実 sm.reset(keep_match_state=True) 発火数: F={tot_reset_f} F+flag={tot_reset_w25}")
    print(f"  新カウンタ (ojama_warmup_suppressed) 合計: F+flag={tot_ojama_warmup_suppressed} (Fは常に0)")
    print(f"  既存ガード1 (start_guard) 合計: F={tot_start_guard_f} F+flag={tot_start_guard_w25}")
    print(f"  既存ガード2 (hsv_gate) 合計: F={tot_hsv_gate_f} F+flag={tot_hsv_gate_w25}")

    print("\n=== (2) false negative proxy: 終端おじゃまセル数比較 (チャンク単位) ===")
    n_decreased = 0
    n_increased = 0
    n_same = 0
    for key, v in sorted(by_key.items()):
        if False not in v or True not in v:
            continue
        f_total = v[False]["final_ojama_count_1p"] + v[False]["final_ojama_count_2p"]
        w_total = v[True]["final_ojama_count_1p"] + v[True]["final_ojama_count_2p"]
        if w_total < f_total:
            n_decreased += 1
            print(f"  {key}: F={f_total} -> F+flag={w_total} (下振れ、要確認)")
        elif w_total > f_total:
            n_increased += 1
        else:
            n_same += 1
    print(f"  下振れ(下振れ=false negative懸念)={n_decreased} 上振れ={n_increased} 同一={n_same} "
          f"(全{len(by_key)}チャンク)")

    print("\n=== (3) OJAMA_FALL 遷移回数/振動率 (entry/exit) F vs F+flag ===")
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
    print(f"  entry/exit回数が F と F+flag で異なるチャンク数: {n_transition_diff}/{len(by_key)} "
          "(0 が期待値、本ガードは detector ロジックに触れないため)")


if __name__ == "__main__":
    main()
