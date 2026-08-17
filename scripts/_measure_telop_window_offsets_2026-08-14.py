"""窓ズレ機構の実測 (続行指示 item1)。

npz の chain_trigger_sec (mechanism='baseline' が大半) が実テロップ表示区間の
どこに位置するかを、得点逆算高信頼帯で expected_n が判明済みの実イベント
10件以上で実測する。

## 測定方法

各イベントについて、`expected_n` の単一クラステンプレのみを使い (他クラスとの
argmax競合を避けるため、_capture_chain_telop_templates_multisource と同じ
単一クラス matchTemplate 方式)、広い探索窓 [before_t_sec - PRE_MARGIN,
trigger_sec + POST_MARGIN] をスキャンして最終ステップ (expected_n) の
ポップアップの実ピーク位置を探す。

    offset_settle_sec = trigger_sec - t(最終ステップのピーク)

を全イベントで測り、その分布 (中央値・最大値) から「baseline機構の
trigger_sec は実際のポップアップ終了から何秒後に来るか」を実測値として
定義する (シーン逆算ではなく複数イベントの分布から定数を決める)。

同時に、before_t_sec (発火前STABLE行) から「1れんさ!」の最初のポップアップ
(sustained plateau、>=3連続samples>=PLATEAU_MIN_CONFIDENCE) までの遅延も
測る (offset_placement_to_first_popup_sec)。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from src.chain_count_ocr import DEFAULT_CHAIN_TEMPLATE_DIR, _crop_search_roi, _to_gray

_spec = importlib.util.spec_from_file_location(
    "_review20_for_offsets",
    Path(__file__).resolve().parent / "_build_review20_chain_count_v2_2026-08-14.py",
)
assert _spec is not None and _spec.loader is not None
_review20_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_review20_mod)

VIDEO_DIR = Path.home() / "frames"
OUT_PATH = Path("data/verify/chain_count_v2_2026-08-14/telop_window_offsets.json")

PRE_MARGIN_SEC = 1.0          # before_t_sec より少し前まで広げる
SAMPLE_INTERVAL_SEC = 0.05    # 生産既定と同じ細かさ (フェード/縮小の取りこぼし対策)
PLATEAU_SAMPLE_INTERVAL_SEC = 0.1  # plateau検出は粗くても十分 (実行コスト優先)
PLATEAU_MIN_CONFIDENCE = 0.65
PLATEAU_MIN_RUN = 3           # 連続 samples 数
# 前面実行の時間予算のため、上限イベント数を絞る (userタスク指定「10件以上」を満たす範囲)
MAX_EVENTS = 12


def _scan_single_class(
    video_path: Path, side: str, tpl_gray: np.ndarray, t_start: float, t_end: float,
    sample_interval_sec: float = SAMPLE_INTERVAL_SEC,
) -> list[tuple[float, float]]:
    """[t_start, t_end] を単一クラステンプレでスキャンし (t, score) 列を返す。"""
    cap = cv2.VideoCapture(str(video_path))
    out: list[tuple[float, float]] = []
    if not cap.isOpened():
        return out
    t = t_start
    while t <= t_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            roi = _crop_search_roi(frame, side)
            if roi is not None and roi.size > 0:
                gray = _to_gray(roi)
                if gray.shape[0] >= tpl_gray.shape[0] and gray.shape[1] >= tpl_gray.shape[1]:
                    res = cv2.matchTemplate(gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
                    _, mv, _, _ = cv2.minMaxLoc(res)
                    out.append((t, float(mv)))
        t += sample_interval_sec
    cap.release()
    return out


def _find_first_plateau_start(
    scores: list[tuple[float, float]],
    min_conf: float = PLATEAU_MIN_CONFIDENCE,
    min_run: int = PLATEAU_MIN_RUN,
) -> float | None:
    run_start: float | None = None
    run_len = 0
    for t, s in scores:
        if s >= min_conf:
            if run_len == 0:
                run_start = t
            run_len += 1
            if run_len >= min_run:
                return run_start
        else:
            run_len = 0
            run_start = None
    return None


def main() -> None:
    events = _review20_mod._select_review_events()[:MAX_EVENTS]
    tpl1_gray = _to_gray(cv2.imread(str(DEFAULT_CHAIN_TEMPLATE_DIR / "digit_1.png")))

    rows: list[dict] = []
    for ev in events:
        video_path = VIDEO_DIR / f"video_{ev['video_id']}.mp4"
        if not video_path.is_file():
            continue
        tpl_n_gray = _to_gray(cv2.imread(
            str(DEFAULT_CHAIN_TEMPLATE_DIR / f"digit_{ev['expected_n']}.png"),
        ))
        if tpl_n_gray is None:
            continue

        # (a) 最終ステップ (expected_n) のピーク位置 (trigger_sec までの広域探索)
        final_scores = _scan_single_class(
            video_path, ev["side"], tpl_n_gray,
            ev["before_t_sec"] - PRE_MARGIN_SEC, ev["t_sec"],
        )
        if not final_scores:
            continue
        peak_t, peak_score = max(final_scores, key=lambda x: x[1])

        # (b) 「1れんさ!」最初のポップアップ (plateau) 位置
        one_scores = _scan_single_class(
            video_path, ev["side"], tpl1_gray,
            ev["before_t_sec"] - PRE_MARGIN_SEC, ev["t_sec"],
            sample_interval_sec=PLATEAU_SAMPLE_INTERVAL_SEC,
        )
        first_one_t = _find_first_plateau_start(one_scores)

        row = {
            "video_id": ev["video_id"], "side": ev["side"], "game_idx": ev["game_idx"],
            "expected_n": ev["expected_n"],
            "before_t_sec": round(ev["before_t_sec"], 2),
            "trigger_sec": round(ev["t_sec"], 2),
            "final_step_peak_t": round(peak_t, 2),
            "final_step_peak_score": round(peak_score, 3),
            "offset_settle_sec": round(ev["t_sec"] - peak_t, 3),
            "first_one_popup_t": round(first_one_t, 2) if first_one_t is not None else None,
            "offset_placement_to_first_popup_sec": (
                round(first_one_t - ev["before_t_sec"], 3) if first_one_t is not None else None
            ),
        }
        rows.append(row)
        print(f"[offsets] {row['video_id']} {row['side']} g{row['game_idx']} "
              f"n={row['expected_n']} offset_settle={row['offset_settle_sec']} "
              f"offset_placement_to_first={row['offset_placement_to_first_popup_sec']} "
              f"peak_score={row['final_step_peak_score']}")

    settle_vals = [r["offset_settle_sec"] for r in rows if r["final_step_peak_score"] >= 0.6]
    placement_vals = [
        r["offset_placement_to_first_popup_sec"] for r in rows
        if r["offset_placement_to_first_popup_sec"] is not None
    ]
    summary = {
        "n_events": len(rows),
        "n_events_with_confident_final_peak": len(settle_vals),
        "offset_settle_sec_median": float(np.median(settle_vals)) if settle_vals else None,
        "offset_settle_sec_max": float(np.max(settle_vals)) if settle_vals else None,
        "offset_settle_sec_values": settle_vals,
        "offset_placement_to_first_popup_median": (
            float(np.median(placement_vals)) if placement_vals else None
        ),
        "offset_placement_to_first_popup_max": (
            float(np.max(placement_vals)) if placement_vals else None
        ),
        "rows": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[offsets] n={len(rows)} settle_median={summary['offset_settle_sec_median']} "
          f"settle_max={summary['offset_settle_sec_max']} "
          f"placement_median={summary['offset_placement_to_first_popup_median']} "
          f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
