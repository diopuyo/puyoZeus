"""W25根治 案4 (2026-08-17) の直接効果確認: c13_chunk0 対象9セルの解消チェック。

`scripts/_diag_c13c22_recheck_2026-08-17.py` の RUNS["c13_chunk0"] と同じ
video/window/cell定義をそのまま再利用し、以下2構成で該当区間 (t=210-296、
対象窓 t=288.5-293.0) を通し処理する:

    構成F      : _diag_c13c22_recheck_2026-08-17.build_pipeline() そのまま
                 (= 統一測定 構成F、docstring 記載の通り既に W23 まで含む)
    構成F+本フラグ: 構成F + enable_ojama_cnn_override_warmup=True

対象9セル (docs/KNOWN_WEAKNESSES.md W25、c13_chunk0 の 12 セルから
r2c2/r3c2/r9c3 の3件 [別系統・別窓] を除いた9件):
    2P r9c1, r9c4, r9c5, r10c1, r10c3, r10c4, r10c5, r11c0, r11c4

各セルについて、対象窓終了時点 (t≈293.0 直前の最終フレーム) の
confirmed_board 値を2構成で比較し、解消件数 (F:誤り → F+flag:正解) と
新規悪化件数 (F:正解 → F+flag:誤り) を報告する。

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._verify_w25_fix_2026-08-17
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent

# ファイル名にハイフンを含むため import 文でなく importlib で動的 import する。
_diag = importlib.import_module("scripts._diag_c13c22_recheck_2026-08-17")

OUT_PATH = _ROOT / "data" / "verify" / "diag_c13c22_recheck_2026-08-17" / "w25_fix_verify.json"

# 対象9セル (docs/KNOWN_WEAKNESSES.md W25)。RUNS["c13_chunk0"] の12セルから
# r2c2/r3c2 (r9c3 は別sheet/別窓のため元々対象外) を除いた集合。
TARGET_CELLS: tuple[tuple[int, int], ...] = (
    (9, 1), (9, 4), (9, 5), (10, 1), (10, 3), (10, 4), (10, 5), (11, 0), (11, 4),
)


def build_pipeline_w25flag() -> "_diag.RecognitionPipeline":  # type: ignore[name-defined]
    """構成F + enable_ojama_cnn_override_warmup=True。"""
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
        enable_ojama_cnn_override_warmup=True,
    )


def run_config(config_name: str, use_flag: bool) -> dict:
    spec = _diag.RUNS["c13_chunk0"]
    video_path = _diag.video_path_of(spec.video_id)
    fps = _diag.probe_fps(video_path)
    print(f"[{config_name}] video={video_path.name} fps={fps:.3f} "
          f"start={spec.start_sec} end={spec.end_sec}")

    cap = cv2.VideoCapture(str(video_path))
    start_frame = int(spec.start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pipeline = build_pipeline_w25flag() if use_flag else _diag.build_pipeline()

    # target_cells のうち RUNS["c13_chunk0"].cells に定義がある分だけ dense window
    # (t_lo/t_hi) を引き継ぐ (定義自体は流用、判定ロジックは新規)。
    cell_specs = {
        (t.r, t.c): t for t in spec.cells if (t.r, t.c) in TARGET_CELLS
    }
    assert len(cell_specs) == len(TARGET_CELLS), (
        f"対象9セルの一部が RUNS['c13_chunk0'] に見つからない: "
        f"{set(TARGET_CELLS) - set(cell_specs)}"
    )

    # 各セルの「対象窓終了直前の最終 confirmed 値」を記録する。
    last_in_window: dict[str, int | None] = {
        f"r{r}c{c}": None for (r, c) in TARGET_CELLS
    }

    frame_idx = start_frame
    t_sec = spec.start_sec
    n_frames = 0
    while t_sec < spec.end_sec:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        res = pipeline.update(frame_idx, t_sec, frame)
        side_res = res.p2  # 対象9セルは全て2P側

        for (r, c) in TARGET_CELLS:
            t = cell_specs[(r, c)]
            if t.t_lo <= t_sec <= t.t_hi and side_res.confirmed_board is not None:
                last_in_window[f"r{r}c{c}"] = int(side_res.confirmed_board.get(r, c))

        n_frames += 1
        frame_idx += 1
        t_sec = frame_idx / fps

    cap.release()
    print(f"[{config_name}] 処理完了 n_frames={n_frames}")
    return {"n_frames": n_frames, "last_in_window": last_in_window}


def main() -> None:
    spec = _diag.RUNS["c13_chunk0"]
    wrong_correct: dict[str, tuple[int, int]] = {
        f"r{t.r}c{t.c}": (t.wrong, t.correct)
        for t in spec.cells if (t.r, t.c) in TARGET_CELLS
    }

    result_f = run_config("F(baseline)", use_flag=False)
    result_w25 = run_config("F+w25flag", use_flag=True)

    n_resolved = 0
    n_regressed = 0
    n_unchanged_wrong = 0
    n_unchanged_correct = 0
    detail = []
    for key, (wrong, correct) in wrong_correct.items():
        v_f = result_f["last_in_window"][key]
        v_w25 = result_w25["last_in_window"][key]
        was_wrong = v_f == wrong
        was_correct = v_f == correct
        now_wrong = v_w25 == wrong
        now_correct = v_w25 == correct
        if was_wrong and now_correct:
            n_resolved += 1
            status = "RESOLVED"
        elif was_correct and now_wrong:
            n_regressed += 1
            status = "REGRESSED"
        elif was_wrong and now_wrong:
            n_unchanged_wrong += 1
            status = "STILL_WRONG"
        elif was_correct and now_correct:
            n_unchanged_correct += 1
            status = "STILL_CORRECT"
        else:
            status = "OTHER"
        detail.append({
            "cell": key, "wrong_label": wrong, "correct_label": correct,
            "f_value": v_f, "w25_value": v_w25, "status": status,
        })
        print(f"  {key}: F={v_f} W25={v_w25} (wrong={wrong} correct={correct}) -> {status}")

    print(f"\n[summary] 解消(RESOLVED)={n_resolved}/9 新規悪化(REGRESSED)={n_regressed}/9 "
          f"未解消(STILL_WRONG)={n_unchanged_wrong}/9 元々正解(STILL_CORRECT)={n_unchanged_correct}/9")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "n_resolved": n_resolved, "n_regressed": n_regressed,
        "n_still_wrong": n_unchanged_wrong, "n_still_correct": n_unchanged_correct,
        "detail": detail,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] -> {OUT_PATH}")


if __name__ == "__main__":
    main()
