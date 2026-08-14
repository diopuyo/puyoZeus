"""タスク#7 item3+4: 人手検証20件のレビューシート作成 + 実測カバレッジ測定。

## 目的

`src.chain_count_truth.resolve_chain_count_truth` (テロップ×得点逆算の二重照合)
を実データ (16動画、削除禁止資産 `~/frames/`) の実イベントに適用し、
(a) 単一動画テンプレ (旧構成、baseline) と
(b) 複数動画から採取した追加テンプレを加えた構成 (2026-08-14 タスク#7)
の二重照合の解決率 (二系統一致率) を比較する。

対象20件はランダム抽出ではなく「得点逆算の高信頼帯 (タスク#7新設) で
expected_n が判明済み」のイベントから選定する (完全にランダムだと大半が
低信頼のまま何も判定できない事象になり、レビューする意味のある事例に
絞れないため)。選定バイアスがあることを正直に明記する。

各イベントについて実画面フレーム (テロップが映っている瞬間、またはテロップ
非検出の場合は tag 行付近のフレーム) を切り出し、判定結果と並べて保存する
(feedback_review_actual_screen_frames_2026-07-24.md 準拠)。

## 出力
    data/verify/chain_count_v2_2026-08-14/review20/
        event_XX_<video>_<side>_g<game_idx>.png   (実画面フレーム)
        review_sheet.json                          (判定結果+メタデータ)
        coverage_before_after.json                 (baseline vs 拡張後の解決率)
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from src.chain_count_ocr import ChainCountOcr, DEFAULT_CHAIN_TEMPLATE_DIR
from src.chain_count_truth import compute_telop_search_window, resolve_chain_count_truth
from src.scoring import is_pure_chain_score_delta

_spec = importlib.util.spec_from_file_location(
    "_coverage_for_review20",
    Path(__file__).resolve().parent / "_measure_chain_count_truth_coverage_2026-08-14.py",
)
assert _spec is not None and _spec.loader is not None
_coverage_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_coverage_mod)

NPZ_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-11")
VIDEO_DIR = Path.home() / "frames"
OUT_DIR = Path("data/verify/chain_count_v2_2026-08-14/review20")
CANDIDATE_TEMPLATE_DIR = Path("data/verify/chain_count_v2_2026-08-14/candidate_templates")

# 目標イベント件数 (item 3 要求「20件」)
TARGET_N_EVENTS = 20

# レビュー対象動画 (16動画のうち、事件数の多い順に選んだサブセット。
# 全16本を通しで再生すると動画I/Oが重いため、多様性を保ちつつ絞る)。
CANDIDATE_VIDEOS = ["c13", "c11", "c17", "c20", "c12", "c16"]

# 得点逆算高信頼帯 (score band) の対象連鎖数レンジ (1桁のみ、視認性のため)。
MIN_N_FOR_REVIEW = 2
MAX_N_FOR_REVIEW = 9

# 探索窓は `compute_telop_search_window` (src/chain_count_truth.py) に一元化。
# 【2026-08-14 続行タスクで撤回】旧実装はここで独自に
# WINDOW_BEFORE_PAD_SEC/WINDOW_AFTER_PAD_SEC/MAX_WINDOW_SPAN_SEC=4.5秒 という
# 実測に基づかないキャップを持っていたが、実データ (video_c13 game_idx=12)
# でこのキャップが本物のポップアップ区間を切り落としていたことが判明した
# (docs/KNOWN_WEAKNESSES.md W3、src.chain_count_truth モジュール docstring
# 「テロップ探索窓の設計」参照)。sampling 間隔も生産既定 (0.05秒) に戻す
# (フェード/縮小アニメーションでスコアが鋭く変化するため粗い間隔は
# ピークを逃す実測あり)。
REVIEW_SAMPLE_INTERVAL_SEC = 0.05


def _events_with_before_t(npz_path: Path) -> list[dict]:
    """_measure_chain_count_truth_coverage_2026-08-14._events_in_file の拡張版。

    before_i の t_sec も追加で持たせる (window の開始点として使うため)。
    """
    d = np.load(npz_path, allow_pickle=True)
    if "chain_trigger_sec" not in d.files or "chain_mechanism" not in d.files:
        return []
    grids = d["grids"]
    side = d["side"]
    game_idx = d["game_idx"]
    t_sec = d["t_sec"]
    score = d["score"]
    trigger = d["chain_trigger_sec"]
    mechanism = d["chain_mechanism"]
    nz_counts = (grids != 0).sum(axis=(1, 2)).tolist()

    groups: dict[tuple, list[int]] = {}
    for i in range(len(grids)):
        key = (str(side[i]), int(game_idx[i]))
        groups.setdefault(key, []).append(i)

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
            if tag in _coverage_mod._NO_CHAIN_TAG_VALUES:
                prev_trigger_sec = None
                continue
            if prev_trigger_sec is not None and float(trigger[i]) == prev_trigger_sec:
                continue
            prev_trigger_sec = float(trigger[i])
            before_i = _coverage_mod._find_before_board_index(nz_counts, idxs, pos)
            if before_i is None:
                continue
            delta_score = int(score[i]) - int(score[before_i])
            events.append({
                "side": side_key,
                "game_idx": int(game_idx_key),
                "t_sec": float(t_sec[i]),
                "before_t_sec": float(t_sec[before_i]),
                "delta_score": delta_score,
            })
    return events


def _select_review_events() -> list[dict]:
    """高信頼帯で expected_n が判明した実イベントを、動画をまたいで選ぶ。"""
    from src.chain_count_truth import select_chain_count_high_confidence_band

    picked: list[dict] = []
    for vid in CANDIDATE_VIDEOS:
        p = NPZ_DIR / f"{vid}.npz"
        if not p.is_file():
            continue
        events = _events_with_before_t(p)
        for ev in events:
            hc = select_chain_count_high_confidence_band(ev["delta_score"])
            if hc.reason != "high_confidence":
                continue
            if not (MIN_N_FOR_REVIEW <= hc.chain_count <= MAX_N_FOR_REVIEW):
                continue
            picked.append({**ev, "video_id": vid, "expected_n": hc.chain_count,
                           "score_ratio": hc.ratio})
        if len(picked) >= TARGET_N_EVENTS * 2:
            break
    # 動画ごとに偏らないよう、動画をラウンドロビンして TARGET_N_EVENTS 件に絞る
    by_video: dict[str, list[dict]] = {}
    for ev in picked:
        by_video.setdefault(ev["video_id"], []).append(ev)
    selected: list[dict] = []
    while len(selected) < TARGET_N_EVENTS and any(by_video.values()):
        for vid in list(by_video.keys()):
            if by_video[vid] and len(selected) < TARGET_N_EVENTS:
                selected.append(by_video[vid].pop(0))
    return selected


def _load_extra_templates() -> dict[int, list[np.ndarray]]:
    """candidate_templates/ の検証済みクロップのみを拡張テンプレとして使う。

    2026-08-14 実測で目視確認済みなのは digit_3 (video_c13、2件、
    NCC 0.637/0.687 で自己一致) のみ。他クラス (4/6) は同バッチでテロップを
    捉えられておらず (ピークスコアが低くテロップ自体を含まない背景クロップ)、
    未検証のまま拡張に混ぜないよう明示的にホワイトリスト化する。
    """
    whitelist = [
        (3, "digit_3_src_c13_g12.png"),
        (3, "digit_3_src_c13_g20.png"),
    ]
    out: dict[int, list[np.ndarray]] = {}
    for label, fname in whitelist:
        p = CANDIDATE_TEMPLATE_DIR / fname
        if p.is_file():
            img = cv2.imread(str(p))
            if img is not None:
                out.setdefault(label, []).append(img)
    return out


def _save_review_frame(video_path: Path, side: str, t_sec: float, out_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return True


def main() -> None:
    events = _select_review_events()
    print(f"[review20] 選定イベント数={len(events)}")

    ocr_baseline = ChainCountOcr.load_default()  # 単一動画テンプレ (旧構成)
    extra = _load_extra_templates()
    primary, _ = ChainCountOcr._load_template_sources_from_dir(DEFAULT_CHAIN_TEMPLATE_DIR)
    ocr_extended = ChainCountOcr(templates=primary, extra_templates=extra)

    n_resolved_baseline = 0
    n_resolved_extended = 0
    review_rows: list[dict] = []
    for idx, ev in enumerate(events):
        video_path = VIDEO_DIR / f"video_{ev['video_id']}.mp4"
        if not video_path.is_file():
            print(f"[review20] SKIP {ev['video_id']}: 動画不在")
            continue
        t_start, t_end = compute_telop_search_window(ev["before_t_sec"], ev["t_sec"])

        # 系統① (テロップ単独、連続列方式) を読む。delta_score は渡さない
        # (`resolve_chain_count_truth` 内で系統②を独立に計算するため、ここで
        # score_backed 方式を混ぜると系統①/②の独立性が壊れる)。
        cap_b = cv2.VideoCapture(str(video_path))
        win_baseline = ocr_baseline.read_max_in_window(
            cap_b, ev["side"], t_start, t_end, sample_interval_sec=REVIEW_SAMPLE_INTERVAL_SEC,
        )
        cap_b.release()
        cap_e = cv2.VideoCapture(str(video_path))
        win_extended = ocr_extended.read_max_in_window(
            cap_e, ev["side"], t_start, t_end, sample_interval_sec=REVIEW_SAMPLE_INTERVAL_SEC,
        )
        cap_e.release()

        truth_baseline = resolve_chain_count_truth(win_baseline, ev["delta_score"])
        truth_extended = resolve_chain_count_truth(win_extended, ev["delta_score"])
        n_resolved_baseline += int(truth_baseline.chain_count is not None)
        n_resolved_extended += int(truth_extended.chain_count is not None)

        # レビュー用の実画面フレームは「拡張構成で最も確信度が高かった瞬間」を
        # 優先して切り出す (テロップが見えている可能性が高い瞬間、user目視用)。
        # 見つからない場合は発火タグ行時刻 (ev["t_sec"]) にフォールバックする。
        best_frame_t = ev["t_sec"]
        best_conf = -1.0
        for i, s in enumerate(win_extended.samples):
            if s.chain_count is not None and s.confidence > best_conf:
                best_conf = s.confidence
                best_frame_t = t_start + i * REVIEW_SAMPLE_INTERVAL_SEC

        frame_name = f"event_{idx:02d}_{ev['video_id']}_{ev['side']}_g{ev['game_idx']}.png"
        _save_review_frame(video_path, ev["side"], best_frame_t, OUT_DIR / frame_name)

        review_rows.append({
            "idx": idx,
            "video_id": ev["video_id"],
            "side": ev["side"],
            "game_idx": ev["game_idx"],
            "trigger_sec": round(ev["t_sec"], 2),
            "search_window": [round(t_start, 2), round(t_end, 2)],
            "frame_captured_at_sec": round(best_frame_t, 2),
            "delta_score": ev["delta_score"],
            "expected_n_from_score_band": ev["expected_n"],
            "score_ratio": ev["score_ratio"],
            "is_pure_chain_score_delta": is_pure_chain_score_delta(ev["delta_score"]),
            "telop_baseline_max": win_baseline.max_chain_count,
            "telop_extended_max": win_extended.max_chain_count,
            "truth_baseline": truth_baseline.chain_count,
            "truth_extended": truth_extended.chain_count,
            "frame_png": frame_name,
        })
        print(f"[review20] {idx:02d} {ev['video_id']} {ev['side']} g{ev['game_idx']} "
              f"expected={ev['expected_n']} baseline_truth={truth_baseline.chain_count} "
              f"extended_truth={truth_extended.chain_count}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "review_sheet.json").write_text(
        json.dumps(review_rows, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    n_total = len(review_rows)
    coverage = {
        "n_events": n_total,
        "n_resolved_baseline_single_source_template": n_resolved_baseline,
        "n_resolved_extended_multisource_template": n_resolved_extended,
        "resolution_rate_baseline": (n_resolved_baseline / n_total) if n_total else 0.0,
        "resolution_rate_extended": (n_resolved_extended / n_total) if n_total else 0.0,
        "note": (
            "selection bias: 対象イベントは得点逆算高信頼帯で expected_n が"
            "判明済みのものに限定 (完全ランダム抽出ではない、正直に明記)。"
            "この20件は『二重照合の解決率』の実測サンプルであり、"
            "母集団全体 (2826イベント) への外挿はできない。"
        ),
    }
    (OUT_DIR / "coverage_before_after.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[review20] baseline resolution={coverage['resolution_rate_baseline']:.3f} "
          f"extended resolution={coverage['resolution_rate_extended']:.3f}")


if __name__ == "__main__":
    main()
