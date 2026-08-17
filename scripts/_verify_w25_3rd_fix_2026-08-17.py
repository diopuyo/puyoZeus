"""W25根治 第3弾・最終 (2026-08-18) の直接効果確認: c13_chunk0 対象9セルの解消チェック。

`scripts/_verify_w25_fix_2026-08-17.py` (第2弾検証) と同一の video/window/
cell定義を再利用し、以下2構成で該当区間 (t=210-296、対象窓 t=288.5-293.0)
を通し処理する:

    構成F        : _diag_c13c22_recheck_2026-08-17.build_pipeline() そのまま
                   (= 統一測定 構成F、W23まで含む。W25関連フラグは全てOFF)
    構成F+第3弾   : 構成F + enable_ojama_write_accounting_guard=True
                   (enable_ojama_cnn_override_warmup は意図的に OFF のまま
                    にして、第3弾単体での解消力を切り分けて確認する)

使い方:
    PYTHONPATH=. ./venv/bin/python -m scripts._verify_w25_3rd_fix_2026-08-17
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parent.parent

_diag = importlib.import_module("scripts._diag_c13c22_recheck_2026-08-17")

OUT_PATH = (
    _ROOT / "data" / "verify" / "diag_c13c22_recheck_2026-08-17"
    / "w25_3rd_fix_verify.json"
)

TARGET_CELLS: tuple[tuple[int, int], ...] = (
    (9, 1), (9, 4), (9, 5), (10, 1), (10, 3), (10, 4), (10, 5), (11, 0), (11, 4),
)


def build_pipeline_w25_3rd(*, also_warmup: bool = False) -> "_diag.RecognitionPipeline":  # type: ignore[name-defined]
    """構成F + enable_ojama_write_accounting_guard=True。

    also_warmup=True で enable_ojama_cnn_override_warmup も併用する
    (アーキ設計「廃止せず併用・フェイルセーフ」の組合せ確認用)。
    """
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
        enable_ojama_cnn_override_warmup=also_warmup,
        enable_ojama_write_accounting_guard=True,
    )


def run_config(config_name: str, pipeline_factory) -> dict:  # noqa: ANN001
    spec = _diag.RUNS["c13_chunk0"]
    video_path = _diag.video_path_of(spec.video_id)
    fps = _diag.probe_fps(video_path)
    print(f"[{config_name}] video={video_path.name} fps={fps:.3f} "
          f"start={spec.start_sec} end={spec.end_sec}")

    cap = cv2.VideoCapture(str(video_path))
    start_frame = int(spec.start_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pipeline = pipeline_factory()

    cell_specs = {
        (t.r, t.c): t for t in spec.cells if (t.r, t.c) in TARGET_CELLS
    }
    assert len(cell_specs) == len(TARGET_CELLS), (
        f"対象9セルの一部が RUNS['c13_chunk0'] に見つからない: "
        f"{set(TARGET_CELLS) - set(cell_specs)}"
    )

    last_in_window: dict[str, int | None] = {
        f"r{r}c{c}": None for (r, c) in TARGET_CELLS
    }
    # 反映遅延測定 (検証観点b用): 各セルで t_lo..t_hi 内、値が最終値に
    # 一致した最初の frame_idx を記録する (真の着弾からの遅延推定に使う)。
    first_frame_at_final: dict[str, int | None] = {
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
                key = f"r{r}c{c}"
                v = int(side_res.confirmed_board.get(r, c))
                last_in_window[key] = v

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

    result_f = run_config("F(baseline)", _diag.build_pipeline)
    result_3rd = run_config(
        "F+3rd(accounting_guard_only)",
        lambda: build_pipeline_w25_3rd(also_warmup=False),
    )
    result_3rd_both = run_config(
        "F+3rd+warmup(both)",
        lambda: build_pipeline_w25_3rd(also_warmup=True),
    )

    def _summarize(result_on: dict, label: str) -> dict:
        n_resolved = n_regressed = n_still_wrong = n_still_correct = 0
        detail = []
        for key, (wrong, correct) in wrong_correct.items():
            v_f = result_f["last_in_window"][key]
            v_on = result_on["last_in_window"][key]
            was_wrong, was_correct = v_f == wrong, v_f == correct
            now_wrong, now_correct = v_on == wrong, v_on == correct
            if was_wrong and now_correct:
                n_resolved += 1
                status = "RESOLVED"
            elif was_correct and now_wrong:
                n_regressed += 1
                status = "REGRESSED"
            elif was_wrong and now_wrong:
                n_still_wrong += 1
                status = "STILL_WRONG"
            elif was_correct and now_correct:
                n_still_correct += 1
                status = "STILL_CORRECT"
            else:
                status = "OTHER"
            detail.append({
                "cell": key, "wrong_label": wrong, "correct_label": correct,
                "f_value": v_f, "on_value": v_on, "status": status,
            })
            print(f"  [{label}] {key}: F={v_f} ON={v_on} "
                  f"(wrong={wrong} correct={correct}) -> {status}")
        print(f"\n[summary:{label}] 解消={n_resolved}/9 新規悪化={n_regressed}/9 "
              f"未解消={n_still_wrong}/9 元々正解={n_still_correct}/9")
        return {
            "n_resolved": n_resolved, "n_regressed": n_regressed,
            "n_still_wrong": n_still_wrong, "n_still_correct": n_still_correct,
            "detail": detail,
        }

    summary_3rd = _summarize(result_3rd, "accounting_guard_only")
    summary_both = _summarize(result_3rd_both, "accounting_guard+warmup")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "accounting_guard_only": summary_3rd,
        "accounting_guard_plus_warmup": summary_both,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] -> {OUT_PATH}")


if __name__ == "__main__":
    main()
