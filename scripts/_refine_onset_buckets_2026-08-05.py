"""onset バケット (曖昧1-5 / 着弾6+ / 着弾なし0) の自動絞り込み (2026-08-05)。

`scripts/build_error_onset_sheet_2026-08-04.py` の診断結果 (no_time_window
84件) について、user目視レビュー (次は明日夜) までに自動で2段の切り分けを
行う。シート生成器 (`build_error_onset_sheet_2026-08-04.py`) は変更しない。

## タスク1: お邪魔会計による着弾の裏取り (曖昧35件+着弾22件)
物理則 (reference_ojama_landing_gated_by_placement): 本物のお邪魔着弾には
「相手が事前の連鎖でおじゃまを送っている」という会計上の裏付けが必須。
アンカー npz の score 列 (信頼性 user 確認済み、project_score_ocr_reliability_
confirmed) から、onset 前 30 秒以内の相手側スコア増分をお邪魔換算し、
観測された `ojama_delta_clean` と整合するか判定する:
    - `landing_confirmed`  : 換算量 >= clean増分 (会計上の裏付けあり)
    - `landing_partial`    : 0 < 換算量 < clean増分 (量が不整合)
    - `landing_unsupported`: 換算量 == 0 (裏付けなし → 誤読の疑い)
お邪魔換算は既存実装をそのまま import して使う (再定義禁止):
    scripts/label_exchange_outcome.py の `_delta_to_ojama_standard`
    (標準レート70点/個、src.scoring.OJAMA_RATE_STANDARD)。

## タスク2: 0バケット27件 + タスク1でunsupportedになった件の特徴分析
- 動画・side・game への集中度 (groupby 件数)
- onset フレームの対象セル HSV 統計 (V平均/S平均/白比率/暗部比率/おじゃま灰色比率)
  を既存の較正済み閾値で分類 (再定義禁止、既存実装から import):
    - `SPECULAR_V_MIN` / `SPECULAR_S_MAX` (src/image_reader.py、光沢ハイライト)
    - `DARK_V_MAX` (src/ojama_warning.py、暗部)
    - `OJM_RECOVERY_S_MAX` / `OJM_RECOVERY_V_MIN` / `OJM_RECOVERY_V_MAX`
      (src/cell_recovery_refiner.py、おじゃま灰色署名)
- 誤値→正値の遷移パターン別クロス集計 (`scripts/measure_effect_gate_c_2026-08-04.py`
  の `classify_error_category` を再利用: color_to_ojama/ojama_to_other/
  empty_confusion/color_confusion)

## 出力
- `data/verify/error_onset_sheet_2026-08-04/index_refined.md` (index.md + refine列)
- 分類別集計・仮説と該当セルリストを stdout に出力

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._refine_onset_buckets_2026-08-05
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cell_recovery_refiner import (  # noqa: E402
    OJM_RECOVERY_S_MAX,
    OJM_RECOVERY_V_MAX,
    OJM_RECOVERY_V_MIN,
)
from src.image_reader import SPECULAR_S_MAX, SPECULAR_V_MIN  # noqa: E402
from src.ojama_warning import DARK_V_MAX  # noqa: E402
from scripts.label_exchange_outcome import _delta_to_ojama_standard  # noqa: E402

# ファイル名にハイフンを含むため動的 import (コピペ禁止指示への対応)。
_SHEET = importlib.import_module("scripts.build_error_onset_sheet_2026-08-04")
_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")
_DIAG = importlib.import_module("scripts._diag_c_zero_effect_2026-08-04")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化、既存の較正済み閾値は import 再利用)
# =============================================================================

INDEX_REFINED_MD_PATH: Path = _SHEET.OUTPUT_DIR / "index_refined.md"

# タスク1: 相手のお邪魔送付を裏取りする探索窓 (onset 前 30 秒)。
OJAMA_ACCOUNTING_WINDOW_SEC: float = 30.0

# タスク2: HSV 署名判定用のパッチ内一致率閾値 (新規、既存閾値そのものは
# 全て import 再利用、この「何割一致したら署名とみなすか」のみ新規定数)。
HSV_SIGNATURE_MATCH_RATIO_MIN: float = 0.3

LANDING_CONFIRMED: str = "landing_confirmed"
LANDING_PARTIAL: str = "landing_partial"
LANDING_UNSUPPORTED: str = "landing_unsupported"

SIG_SPECULAR: str = "specular_highlight"
SIG_DARK: str = "dark_shadow"
SIG_GRAY_OJAMA: str = "gray_ojama_like"
SIG_UNCLEAR: str = "unclear"


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class RefinedInfo:
    """1誤りセル分のタスク1/タスク2追加情報 (元の CellOnsetRecord に紐づく)。"""

    record: "object"  # _SHEET.CellOnsetRecord
    error_category: str
    landing_status: "str | None" = None
    opp_supply_ojama: "int | None" = None
    hsv_signature: "str | None" = None
    v_mean: "float | None" = None
    s_mean: "float | None" = None
    white_ratio: "float | None" = None
    dark_ratio: "float | None" = None
    gray_ojama_ratio: "float | None" = None


# =============================================================================
# 1. タスク1: お邪魔会計による着弾の裏取り
# =============================================================================


def _load_scores(npz_path: Path) -> "np.ndarray":
    """npz から score 列を生配列で読む (_NpzIndex は本列を持たないため別読込)。"""
    data = np.load(npz_path, allow_pickle=True)
    return data["score"].astype(np.int64)


def _opponent_ojama_supply(
    idx: "object", scores: "np.ndarray", opp_side: str, game_idx: int,
    onset_t: float, window_sec: float,
) -> int:
    """相手 (opp_side) が onset 前 window_sec 秒で送った推定お邪魔量。"""
    rows = _DIAG._side_game_row_indices(idx, opp_side, game_idx)
    if len(rows) == 0:
        return 0
    times = idx.t_secs[rows]
    vals = scores[rows]
    now_mask = times <= onset_t
    if not now_mask.any():
        return 0
    now_score = int(vals[now_mask][-1])
    before_mask = times <= (onset_t - window_sec)
    before_score = int(vals[before_mask][-1]) if before_mask.any() else 0
    return int(_delta_to_ojama_standard(max(0, now_score - before_score)))


def classify_landing(clean_delta: int, opp_supply: int) -> str:
    """会計換算量と観測増分を比較し、着弾の裏取り結果を分類する。"""
    if opp_supply == 0:
        return LANDING_UNSUPPORTED
    if opp_supply >= clean_delta:
        return LANDING_CONFIRMED
    return LANDING_PARTIAL


def build_task1_infos(
    records_all: list["object"], no_window_ids: set, anchor_cache: dict, score_cache: dict,
) -> dict[int, RefinedInfo]:
    """全93セルに誤り分類を付与し、no_time_window かつ clean増分>0 のセルのみ
    タスク1 (お邪魔会計の裏取り) 判定も付与する (仕様の対象範囲を厳密に守る)。
    """
    infos: dict[int, RefinedInfo] = {}
    for r in records_all:
        cat = _MC.classify_error_category(r.wrong_value, r.correct_value)
        info = RefinedInfo(record=r, error_category=cat)
        in_scope = (
            id(r) in no_window_ids and r.ojama_delta_clean and r.ojama_delta_clean > 0
            and r.onset_t_sec is not None
        )
        if in_scope:
            opp_side = "2P" if r.side == "1P" else "1P"
            supply = _opponent_ojama_supply(
                anchor_cache[r.video], score_cache[r.video], opp_side, r.game_idx,
                r.onset_t_sec, OJAMA_ACCOUNTING_WINDOW_SEC,
            )
            info.opp_supply_ojama = supply
            info.landing_status = classify_landing(r.ojama_delta_clean, supply)
        infos[id(r)] = info
    return infos


# =============================================================================
# 2. タスク2: HSV 署名 + 特徴分析
# =============================================================================


def _patch_hsv_stats(frame_bgr: "np.ndarray", region: "object", row: int, col: int) -> dict:
    """対象セルパッチの V平均/S平均/白比率/暗部比率/おじゃま灰色比率を計算する。"""
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    patch = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1].astype(np.float64)
    v_ch = hsv[:, :, 2].astype(np.float64)
    white = (v_ch >= SPECULAR_V_MIN) & (s_ch <= SPECULAR_S_MAX)
    dark = v_ch <= DARK_V_MAX
    gray_ojama = (s_ch <= OJM_RECOVERY_S_MAX) & (v_ch >= OJM_RECOVERY_V_MIN) & (v_ch <= OJM_RECOVERY_V_MAX)
    return {
        "v_mean": float(np.mean(v_ch)), "s_mean": float(np.mean(s_ch)),
        "white_ratio": float(np.mean(white)), "dark_ratio": float(np.mean(dark)),
        "gray_ojama_ratio": float(np.mean(gray_ojama)),
    }


def _classify_hsv_signature(stats: dict) -> str:
    """パッチ HSV 統計を既知の誤読署名カテゴリに分類する (既存較正閾値ベース)。"""
    if stats["white_ratio"] >= HSV_SIGNATURE_MATCH_RATIO_MIN:
        return SIG_SPECULAR
    if stats["gray_ojama_ratio"] >= HSV_SIGNATURE_MATCH_RATIO_MIN:
        return SIG_GRAY_OJAMA
    if stats["dark_ratio"] >= HSV_SIGNATURE_MATCH_RATIO_MIN:
        return SIG_DARK
    return SIG_UNCLEAR


def run_task2(target_infos: list[RefinedInfo], cap_cache: dict, fps_cache: dict) -> None:
    """task2対象セルに HSV 署名分類を付与する (in-place)。"""
    for info in target_infos:
        r = info.record
        if r.onset_t_sec is None:
            info.hsv_signature = SIG_UNCLEAR
            continue
        cap = cap_cache[r.video]
        fps = fps_cache[r.video]
        frame = _DIAG._read_frame_at(cap, fps, max(r.onset_t_sec, 0.0))
        if frame is None:
            info.hsv_signature = SIG_UNCLEAR
            continue
        region = _SHEET._region_for_side(r.side)
        stats = _patch_hsv_stats(frame, region, r.row, r.col)
        info.v_mean = stats["v_mean"]
        info.s_mean = stats["s_mean"]
        info.white_ratio = stats["white_ratio"]
        info.dark_ratio = stats["dark_ratio"]
        info.gray_ojama_ratio = stats["gray_ojama_ratio"]
        info.hsv_signature = _classify_hsv_signature(stats)


# =============================================================================
# 3. キャッシュ構築 (動画ごとに npz/score/動画capを1回だけ開く)
# =============================================================================


def build_caches(records: list["object"]) -> tuple[dict, dict, dict, dict]:
    """動画ごとに anchor_idx / score配列 / cv2.VideoCapture / fps をキャッシュする。"""
    anchor_cache: dict[str, "object"] = {}
    score_cache: dict[str, "np.ndarray"] = {}
    cap_cache: dict[str, "cv2.VideoCapture"] = {}
    fps_cache: dict[str, float] = {}
    for r in records:
        if r.video in anchor_cache:
            continue
        npz_path = _MC.ANCHOR_NPZ_DIR / f"{r.video}.npz"
        anchor_cache[r.video] = _MC._load_npz_index(npz_path)
        score_cache[r.video] = _load_scores(npz_path)
        video_path = _DIAG.VIDEO_DIR / f"video_{r.video}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        cap_cache[r.video] = cap
        fps_cache[r.video] = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return anchor_cache, score_cache, cap_cache, fps_cache


def release_caps(cap_cache: dict) -> None:
    """VideoCapture を全て解放する。"""
    for cap in cap_cache.values():
        cap.release()


# =============================================================================
# 4. 集計レポート
# =============================================================================


def _cell_label(r: "object") -> str:
    """セル識別用の短い表記 (video/side/t_sec/row/col)。"""
    return f"{r.video} {r.side} t={r.label_t_sec:.1f} row{r.row}col{r.col}"


def report_task1(no_window: list["object"], infos: dict[int, RefinedInfo]) -> str:
    """タスク1: landing_confirmed/partial/unsupported の件数 + セルリスト。"""
    scoped = [
        infos[id(r)] for r in no_window
        if r.ojama_delta_clean and r.ojama_delta_clean > 0
    ]
    lines = [f"--- タスク1: お邪魔会計の裏取り (対象 {len(scoped)} 件) ---"]
    for status in (LANDING_CONFIRMED, LANDING_PARTIAL, LANDING_UNSUPPORTED):
        matched = [i for i in scoped if i.landing_status == status]
        lines.append(f"{status}: {len(matched)} 件")
        for i in matched:
            lines.append(f"    {_cell_label(i.record)} clean={i.record.ojama_delta_clean} supply={i.opp_supply_ojama}")
    return "\n".join(lines)


def report_task2_concentration(target_infos: list[RefinedInfo]) -> str:
    """タスク2: 動画・side への集中度。"""
    lines = [f"--- タスク2: 集中度分析 (対象 {len(target_infos)} 件) ---"]
    by_video: dict[str, int] = {}
    by_side: dict[str, int] = {}
    for info in target_infos:
        by_video[info.record.video] = by_video.get(info.record.video, 0) + 1
        by_side[info.record.side] = by_side.get(info.record.side, 0) + 1
    lines.append(f"動画別: {dict(sorted(by_video.items(), key=lambda kv: -kv[1]))}")
    lines.append(f"side別: {by_side}")
    return "\n".join(lines)


def report_task2_signatures(target_infos: list[RefinedInfo]) -> str:
    """タスク2: HSV署名別の件数 + 該当セルリスト。"""
    lines = ["--- タスク2: HSV署名分類 ---"]
    for sig in (SIG_SPECULAR, SIG_GRAY_OJAMA, SIG_DARK, SIG_UNCLEAR):
        matched = [i for i in target_infos if i.hsv_signature == sig]
        lines.append(f"{sig}: {len(matched)} 件")
        for i in matched:
            r = i.record
            lines.append(
                f"    {_cell_label(r)} wrong={r.wrong_value}->correct={r.correct_value} "
                f"V={i.v_mean:.0f} S={i.s_mean:.0f}"
            )
    return "\n".join(lines)


def report_transition_crosstab(target_infos: list[RefinedInfo]) -> str:
    """タスク2: 誤値→正値の遷移パターン (classify_error_category 再利用) 別集計。"""
    lines = ["--- タスク2: 遷移パターン別クロス集計 ---"]
    by_cat: dict[str, int] = {}
    for info in target_infos:
        by_cat[info.error_category] = by_cat.get(info.error_category, 0) + 1
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        lines.append(f"{cat}: {n} 件")
    return "\n".join(lines)


# =============================================================================
# 5. index_refined.md 出力
# =============================================================================


def _refined_row(info: RefinedInfo) -> str:
    """index_refined.md 用の1セル分の行。"""
    r = info.record
    sig = info.hsv_signature or ""
    landing = info.landing_status or ""
    supply = "" if info.opp_supply_ojama is None else str(info.opp_supply_ojama)
    v = "" if info.v_mean is None else f"{info.v_mean:.0f}"
    s = "" if info.s_mean is None else f"{info.s_mean:.0f}"
    return (
        f"| {r.video} | {r.side} | {r.label_t_sec:.1f} | {r.row} | {r.col} | "
        f"{r.wrong_value}→{r.correct_value} | {info.error_category} | "
        f"{r.ojama_delta_clean} | {landing} | {supply} | {sig} | {v} | {s} |"
    )


def write_index_refined(records_all: list["object"], infos: dict[int, RefinedInfo]) -> None:
    """全93セルの index_refined.md (index.md + refine列) を出力する。"""
    header = (
        "| video | side | label_t | row | col | wrong→correct | error_category | "
        "ojama_delta_clean | landing_status | opp_supply_ojama | hsv_signature | "
        "v_mean | s_mean |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    ordered = sorted(records_all, key=lambda r: (r.video, r.side, r.label_t_sec, r.row, r.col))
    body = "\n".join(_refined_row(infos[id(r)]) for r in ordered)
    INDEX_REFINED_MD_PATH.write_text(header + body + "\n", encoding="utf-8")


# =============================================================================
# 6. main
# =============================================================================


def main() -> None:
    records_all = _SHEET.diagnose_all_samples()
    no_window = _SHEET._no_time_window_records(records_all)
    no_window_ids = {id(r) for r in no_window}
    print(f"[1/4] 診断対象: 全{len(records_all)}件 / no_time_window {len(no_window)}件")

    anchor_cache, score_cache, cap_cache, fps_cache = build_caches(records_all)
    infos = build_task1_infos(records_all, no_window_ids, anchor_cache, score_cache)
    print("\n[2/4] " + report_task1(no_window, infos))

    task2_targets = [
        infos[id(r)] for r in no_window
        if not r.ojama_delta_clean or r.ojama_delta_clean == 0
        or infos[id(r)].landing_status == LANDING_UNSUPPORTED
    ]
    run_task2(task2_targets, cap_cache, fps_cache)
    release_caps(cap_cache)
    print("\n[3/4] " + report_task2_concentration(task2_targets))
    print()
    print(report_task2_signatures(task2_targets))
    print()
    print(report_transition_crosstab(task2_targets))

    write_index_refined(records_all, infos)
    print(f"\n[4/4] index_refined.md: {INDEX_REFINED_MD_PATH}")


if __name__ == "__main__":
    main()
