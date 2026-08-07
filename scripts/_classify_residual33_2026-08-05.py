"""残存33セル (既知93セル→v2_err_known) の敗因分類 (2026-08-05、使い捨て分析)。

本命構成 (Stage1.5 + BURST_GATE_OPEN_THRESHOLD=0.954、
`data/verify/burst_guard_2026-08-05/on_v2_full` 全31動画着弾) の最終値=
既知93セル→残存33セル (`measure_effect_gate_c_2026-08-04.py --v2-npz-dir
.../on_v2_full` の `v2_err_known` 合計) について、セル単位でなぜ残ったのかを
4分類する:

- below_threshold: onset±0.2秒の compute_effect_glow_score (rows1-3) が
  0.954 未満 = 弱い光でバースト窓が開かなかった (Stage2 較正拡張の対象)
- window_open_but_survived: 上記スコアが0.954以上なのに誤りが残った
  (別経路の疑い、要個別調査)
- smoke_layer: row4-12 (バースト検出row1-3のスコープ外、Stage2=着弾窓の担当)
- other: 上記以外 (row0等、想定外パターンとして明示フラグ)

## 過学習防止・既存資産の再利用 (コピペ禁止指示への対応)
- `scripts/measure_effect_gate_c_2026-08-04.py` (importlib動的import):
  `load_all_samples` / `_load_npz_index` / `_lookup_anchor_row` /
  `_find_by_frame_idx_exact` / `_find_bit_exact_match` / `_find_effective_row` /
  `_known_error_cells` / `ANCHOR_NPZ_DIR` / `ANCHOR_MATCH_WINDOW_SEC` /
  `V2_MATCH_MODE_*` をそのまま再利用し、v2グリッド解決の3段フォールバック
  ((exact→bit_exact→effective)) は `compute_v2_stage_result` と完全に同じ
  優先順位を保つ (件数計算は既存関数、本スクリプトはセル位置が必要なため
  グリッド自体を取得する薄いラッパーのみ追加)。
- `scripts/build_error_onset_sheet_2026-08-04.py` (importlib動的import):
  `diagnose_all_samples` (93セルの onset_t_sec を特定済みのロジックをそのまま
  再利用、onset検出ロジック自体は再実装しない) / `_region_for_side`。
- `scripts/_diag_c_zero_effect_2026-08-04.py` (importlib動的import):
  `_read_frame_at` / `VIDEO_DIR`。
- `src/effect_glow_detector.py::compute_effect_glow_score` (Stage0抽出済み、
  本番と同一ロジック)。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._classify_residual33_2026-08-05
"""
from __future__ import annotations

import csv
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board_state_machine import EFFECT_GATE_TOP_ROWS  # noqa: E402
from src.effect_glow_detector import compute_effect_glow_score  # noqa: E402

# ファイル名にハイフンを含むため通常の `from ... import` は不可 (動的import)。
_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")
_SHEET = importlib.import_module("scripts.build_error_onset_sheet_2026-08-04")
_DIAG = importlib.import_module("scripts._diag_c_zero_effect_2026-08-04")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

# 本命構成の着弾先 (Stage1.5+0.954、31動画フル着弾)。
V2_LANDED_DIR: Path = Path("data/verify/burst_guard_2026-08-05/on_v2_full")

# on_v2_full backtest で実際に使われた閾値 (モジュール既定
# `src.recognition_pipeline.BURST_GATE_OPEN_THRESHOLD`=0.97 は互換維持のため
# 未変更、この0.954は --burst-gate-open-threshold で明示上書きされた値)。
LANDED_BURST_GATE_OPEN_THRESHOLD: float = 0.954

# smoke layer (row4-12、Stage2=着弾窓の担当スコープ)。EFFECT_GATE_TOP_ROWS
# ({1,2,3}) が burst layer、それ以外の row1〜12 を smoke layer とする
# (row0=隠し段はOFF誤り0件と実測済みのため別枠 = other 分類で捕捉する)。
SMOKE_LAYER_ROWS: "frozenset[int]" = frozenset(range(4, 13))

# onset±0.2秒の3点で compute_effect_glow_score を実測し最大値を取る
# (コーディネータ指示 "onset±0.2秒" をそのまま3点実測として解釈)。
ONSET_SCORE_PROBE_OFFSETS_SEC: "tuple[float, ...]" = (-0.2, 0.0, 0.2)

# c5実測時に確認済みの弱グロー帯 (below_thresholdの集中度を報告する際の目安)。
C5_BAND_LOW: float = 0.78
C5_BAND_HIGH: float = 0.90

OUTPUT_CSV_PATH: Path = Path(
    "data/verify/burst_guard_2026-08-05/residual33_cell_classification.csv"
)

CLASS_BELOW_THRESHOLD: str = "below_threshold"
CLASS_WINDOW_OPEN_SURVIVED: str = "window_open_but_survived"
CLASS_SMOKE_LAYER: str = "smoke_layer"
CLASS_OTHER: str = "other"


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class ResidualCell:
    """1件の残存既知誤りセル (敗因分類済み)。"""

    video: str
    side: str
    label_t_sec: float
    row: int
    col: int
    off_value: int
    correct_value: int
    v2_match_mode: str
    v2_value: int
    onset_t_sec: "float | None"
    max_burst_score: "float | None"
    classification: str
    note: str = ""


# =============================================================================
# 1. v2グリッド解決 (compute_v2_stage_result と同一優先順位、グリッド自体を返す薄いラッパー)
# =============================================================================


def _resolve_v2_grid(
    v2_idx: "object | None", side: str, anchor: "object",
) -> "tuple[np.ndarray, str] | None":
    """(exact → bit_exact_fallback → effective) の順で v2 グリッドを解決する。"""
    if v2_idx is None:
        return None
    match = _MC._find_by_frame_idx_exact(v2_idx, side, anchor.frame_idx)
    if match is not None:
        return match[0], _MC.V2_MATCH_MODE_EXACT
    match = _MC._find_bit_exact_match(
        v2_idx, side, anchor.grid, anchor.t_sec, _MC.ANCHOR_MATCH_WINDOW_SEC,
    )
    if match is not None:
        return match[0], _MC.V2_MATCH_MODE_EXACT
    match = _MC._find_effective_row(v2_idx, side, anchor.t_sec)
    if match is not None:
        return match[0], _MC.V2_MATCH_MODE_EFFECTIVE
    return None


# =============================================================================
# 2. onset記録の索引付け (build_error_onset_sheet の結果をそのまま再利用)
# =============================================================================


def _onset_key(video: str, side: str, label_t_sec: float, row: int, col: int) -> tuple:
    """onset記録の索引キー (label_t は0.05秒丸めで同一サンプルを一意識別)。"""
    return (video, side, round(label_t_sec, 1), row, col)


def build_onset_index(records: list) -> dict:
    """CellOnsetRecord のリストから (video,side,label_t,row,col) -> record の索引を作る。"""
    return {
        _onset_key(r.video, r.side, r.label_t_sec, r.row, r.col): r
        for r in records
    }


# =============================================================================
# 3. 残存セルの列挙 (measure_effect_gate_c の v2_err_known と同一集合になるはず)
# =============================================================================


def find_residual_cells(samples: list, onset_idx: dict) -> list[ResidualCell]:
    """fixed サンプル全件について、既知誤りセルのうち v2 でも残ったものを列挙する。"""
    out: list[ResidualCell] = []
    anchor_cache: dict = {}
    v2_cache: dict = {}
    for s in samples:
        if s.status != "fixed":
            continue
        if s.video_stem not in anchor_cache:
            anchor_cache[s.video_stem] = _MC._load_npz_index(
                _MC.ANCHOR_NPZ_DIR / f"{s.video_stem}.npz"
            )
            v2_cache[s.video_stem] = _MC._load_npz_index(
                V2_LANDED_DIR / f"{s.video_stem}.npz"
            )
        anchor_idx = anchor_cache[s.video_stem]
        anchor = _MC._lookup_anchor_row(
            anchor_idx, s.side, s.t_sec, s.game_idx, s.anchor_recognized_grid,
        )
        if anchor is None:
            continue
        resolved = _resolve_v2_grid(v2_cache[s.video_stem], s.side, anchor)
        if resolved is None:
            continue
        v2_grid, match_mode = resolved
        out.extend(_residual_cells_in_sample(s, anchor, v2_grid, match_mode, onset_idx))
    return out


def _residual_cells_in_sample(
    s: "object", anchor: "object", v2_grid: "np.ndarray", match_mode: str, onset_idx: dict,
) -> list[ResidualCell]:
    """1サンプル分: 既知誤りセルのうち v2 でも誤りが残ったものだけを抜き出す。"""
    known_cells = _MC._known_error_cells(anchor.grid, s.correct_grid)
    cells: list[ResidualCell] = []
    for row, col in sorted(known_cells):
        correct_v = int(s.correct_grid[row, col])
        v2_v = int(v2_grid[row, col])
        if v2_v == correct_v:
            continue
        key = _onset_key(s.video_stem, s.side, s.t_sec, row, col)
        onset_rec = onset_idx.get(key)
        cells.append(ResidualCell(
            video=s.video_stem, side=s.side, label_t_sec=s.t_sec, row=row, col=col,
            off_value=int(anchor.grid[row, col]), correct_value=correct_v,
            v2_match_mode=match_mode, v2_value=v2_v,
            onset_t_sec=(onset_rec.onset_t_sec if onset_rec else None),
            max_burst_score=None, classification="",
        ))
    return cells


# =============================================================================
# 4. burst score 実測 (onset±0.2秒) + 分類判定
# =============================================================================


def _measure_max_burst_score(
    video: str, side: str, onset_t_sec: float, cap_cache: dict, fps_cache: dict,
) -> "float | None":
    """onset±0.2秒の3点で compute_effect_glow_score (rows1-3) の最大値を測る。"""
    region = _SHEET._region_for_side(side)
    if video not in cap_cache:
        video_path = _DIAG.VIDEO_DIR / f"video_{video}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        cap_cache[video] = cap
        fps_cache[video] = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap = cap_cache[video]
    fps = fps_cache[video]
    best = 0.0
    found = False
    for offset in ONSET_SCORE_PROBE_OFFSETS_SEC:
        frame = _DIAG._read_frame_at(cap, fps, max(onset_t_sec + offset, 0.0))
        if frame is None:
            continue
        found = True
        score = compute_effect_glow_score(frame, region, EFFECT_GATE_TOP_ROWS)
        best = max(best, score)
    return best if found else None


def classify_cell(cell: ResidualCell, cap_cache: dict, fps_cache: dict) -> None:
    """1セルを4分類のいずれかに割り当て、cell を破壊的に更新する。"""
    if cell.row in SMOKE_LAYER_ROWS:
        cell.classification = CLASS_SMOKE_LAYER
        return
    if cell.row not in EFFECT_GATE_TOP_ROWS:
        cell.classification = CLASS_OTHER
        cell.note = "burst/smoke レイヤーいずれにも属さない行 (row0隠し段等、想定外)"
        return
    if cell.onset_t_sec is None:
        cell.classification = CLASS_OTHER
        cell.note = "onset特定不能 (build_error_onset_sheet で pre_existing 等)"
        return
    score = _measure_max_burst_score(cell.video, cell.side, cell.onset_t_sec, cap_cache, fps_cache)
    cell.max_burst_score = score
    if score is None:
        cell.classification = CLASS_OTHER
        cell.note = "フレーム取得不能でスコア実測不可"
    elif score < LANDED_BURST_GATE_OPEN_THRESHOLD:
        cell.classification = CLASS_BELOW_THRESHOLD
    else:
        cell.classification = CLASS_WINDOW_OPEN_SURVIVED
        cell.note = "要個別調査: 窓は開いたはずなのに誤りが残存"


def classify_all(cells: list[ResidualCell]) -> None:
    """burstレイヤー対象セルのみ動画キャプチャを開いてスコア実測する。"""
    cap_cache: dict = {}
    fps_cache: dict = {}
    for cell in cells:
        classify_cell(cell, cap_cache, fps_cache)
    for cap in cap_cache.values():
        cap.release()


# =============================================================================
# 5. 出力 (セル単位CSV + 分類集計レポート)
# =============================================================================


def write_cell_csv(cells: list[ResidualCell], out_path: Path) -> None:
    """残存セル明細を1行1セルのCSVに書き出す。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video", "side", "label_t_sec", "row", "col", "off_value", "correct_value",
        "v2_match_mode", "v2_value", "onset_t_sec", "max_burst_score",
        "classification", "note",
    ]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in cells:
            writer.writerow({
                "video": c.video, "side": c.side, "label_t_sec": c.label_t_sec,
                "row": c.row, "col": c.col, "off_value": c.off_value,
                "correct_value": c.correct_value, "v2_match_mode": c.v2_match_mode,
                "v2_value": c.v2_value,
                "onset_t_sec": "" if c.onset_t_sec is None else f"{c.onset_t_sec:.3f}",
                "max_burst_score": (
                    "" if c.max_burst_score is None else f"{c.max_burst_score:.3f}"
                ),
                "classification": c.classification, "note": c.note,
            })


def _score_distribution_report(scores: list[float]) -> str:
    """below_threshold セルの score 分布 (Stage2較正拡張の効果上限を決める数字)。"""
    if not scores:
        return "  分布: データなし"
    arr = np.array(scores)
    n_c5_band = int(np.sum((arr >= C5_BAND_LOW) & (arr < C5_BAND_HIGH)))
    n_below_c5 = int(np.sum(arr < C5_BAND_LOW))
    n_above_c5_below_thr = int(np.sum(
        (arr >= C5_BAND_HIGH) & (arr < LANDED_BURST_GATE_OPEN_THRESHOLD)
    ))
    return (
        f"  score分布: min={arr.min():.3f} p25={np.percentile(arr,25):.3f} "
        f"median={np.median(arr):.3f} p75={np.percentile(arr,75):.3f} max={arr.max():.3f}\n"
        f"  帯別: <{C5_BAND_LOW}帯={n_below_c5}件 / "
        f"[{C5_BAND_LOW},{C5_BAND_HIGH})帯(c5実測帯)={n_c5_band}件 / "
        f"[{C5_BAND_HIGH},{LANDED_BURST_GATE_OPEN_THRESHOLD})帯={n_above_c5_below_thr}件"
    )


def build_summary_report(cells: list[ResidualCell]) -> str:
    """分類集計 + below_threshold分布 + window_open_but_survived個別例。"""
    lines = [f"--- 残存既知誤りセル分類 (合計 {len(cells)} 件) ---"]
    by_class: dict[str, list[ResidualCell]] = {}
    for c in cells:
        by_class.setdefault(c.classification, []).append(c)
    for name in (CLASS_BELOW_THRESHOLD, CLASS_WINDOW_OPEN_SURVIVED, CLASS_SMOKE_LAYER, CLASS_OTHER):
        group = by_class.get(name, [])
        lines.append(f"  {name}: {len(group)} 件")
    lines.append("")
    below = by_class.get(CLASS_BELOW_THRESHOLD, [])
    lines.append(f"[below_threshold の score分布 (n={len(below)})]")
    lines.append(_score_distribution_report([c.max_burst_score for c in below if c.max_burst_score is not None]))
    survived = by_class.get(CLASS_WINDOW_OPEN_SURVIVED, [])
    lines.append("")
    lines.append(f"[window_open_but_survived の個別例 (n={len(survived)})]")
    for c in survived:
        lines.append(
            f"  {c.video} {c.side} t={c.label_t_sec} row{c.row}col{c.col}: "
            f"score={c.max_burst_score:.3f} off={c.off_value}→v2={c.v2_value}(正解{c.correct_value}) "
            f"mode={c.v2_match_mode}"
        )
    other = by_class.get(CLASS_OTHER, [])
    lines.append("")
    lines.append(f"[other (想定外) の個別例 (n={len(other)})]")
    for c in other:
        lines.append(
            f"  {c.video} {c.side} t={c.label_t_sec} row{c.row}col{c.col}: {c.note}"
        )
    return "\n".join(lines)


# =============================================================================
# 6. main
# =============================================================================


def main() -> None:
    cv2.setNumThreads(1)
    samples = _MC.load_all_samples()
    print(f"[1/4] ラベル読込: {len(samples)} 件 (fixed分のみ以降で対象)")

    onset_records = _SHEET.diagnose_all_samples()
    onset_idx = build_onset_index(onset_records)
    print(f"[2/4] onset記録読込: {len(onset_records)} 件 (93セル相当)")

    cells = find_residual_cells(samples, onset_idx)
    print(f"[3/4] 残存既知誤りセル特定: {len(cells)} 件 (measure_effect_gate_c の v2_err_known 合計と照合)")

    classify_all(cells)
    write_cell_csv(cells, OUTPUT_CSV_PATH)
    print(f"[出力] セル単位明細 CSV: {OUTPUT_CSV_PATH}")

    print("\n[4/4] " + build_summary_report(cells))


if __name__ == "__main__":
    main()
