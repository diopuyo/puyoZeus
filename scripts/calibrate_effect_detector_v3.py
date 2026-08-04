"""案B (エフェクト視覚検出) 最終較正 第3弾 (2026-08-04)。

第1弾 (data/verify/effect_cell_label_2026-08-03、36枚) + 第3弾
(data/verify/effect_cell_label_v3_2026-08-04、100枚) の人手ラベルを統合し、
v_mean 窓レベル判定 (scripts/calibrate_effect_detector.py の資産を流用) を
n=136 規模で再較正する。第3弾で新設された「対象外エフェクト
(out_of_scope)」status (連鎖数テロップ等の混同要因) を使い、時間ゲート
「自分の連鎖中は判定抑制」だけで実装可能かを検証する。

## 統合較正で判明した事実 (出典: 本スクリプト実行結果)
- burst 真値セル (userクリック値=1) は abs_row 1-3 に 96% 集中 (row4 に
  微小tailあり)。既存 BURST_ROW_MIN/MAX=1/3 の妥当性を再確認。
- smoke 真値セル (値=2) は abs_row 1-12 にほぼ一様分布 (落下列・積み高さ
  依存で固定行帯を持たない、reference_ojama_landing_gated_by_placement と
  整合)。固定窓は成立せず、本スクリプトでは全12行を対象窓として扱う
  (別途スタック高さ適応窓が今後の課題)。
- layer 列 (burst/smoke の候補窓由来) と真値ラベル値は一致しない場合がある
  (窓選定は推定に過ぎない)。真値は必ずグリッド値そのものを使う。
- out_of_scope 25枚の大半は「観測対象側自身の連鎖数テロップ表示中」
  (目視確認、note列に「自分のX連鎖中」と明記、または burst/smoke 窓でも
  同型の自連鎖テロップ写り込みを確認) であり、既存パイプライン
  ChainPhaseDetector の is_chain_p1/is_chain_p2 で検出済みの状態と一致する。

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts.calibrate_effect_detector_v3

## 出力 (data/verify/effect_detector_calibration_v3_2026-08-04/)
    labeled_cell_features_v3.csv     突合済み全セル特徴量+真値 (v1+v3統合)
    burst_roc_operating_points.csv   burst 窓 (row1-3) の frame-level ROC 動作点
    out_of_scope_misfire.csv         out_of_scope + telop_negative 枠の誤発火判定
    calibration_report_v3.md        GO/NO-GO 判定 + 実装仕様
"""
from __future__ import annotations

import csv
import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, HIDDEN_ROWS  # noqa: E402

_STUDY_MODULE_NAME: str = "scripts.study_effect_signature_2026-08-03"
_ES = importlib.import_module(_STUDY_MODULE_NAME)

# =============================================================================
# 定数
# =============================================================================

# ラベルソース (v1 第1弾 + v3 第3弾)。列構成が微妙に異なる (v1 に
# out_of_scope 列値なし・chain_bin あり、v3 に out_of_scope あり・note あり)
# ため、読込側で欠損列を空文字 fallback して吸収する。
LABEL_SOURCES: tuple[tuple[str, Path, Path], ...] = (
    (
        "v1", Path("data/verify/effect_cell_label_2026-08-03/labeling_result.csv"),
        Path("data/verify/effect_cell_label_2026-08-03/frames"),
    ),
    (
        "v3", Path("data/verify/effect_cell_label_v3_2026-08-04/labeling_result.csv"),
        Path("data/verify/effect_cell_label_v3_2026-08-04/frames"),
    ),
)
OUTPUT_DIR: Path = Path("data/verify/effect_detector_calibration_v3_2026-08-04")

STATUS_NO_EFFECT: str = "no_effect"
STATUS_MARKED: str = "marked"
STATUS_SKIP: str = "skip"
STATUS_OUT_OF_SCOPE: str = "out_of_scope"
STATUS_UNLABELED: str = "unlabeled"

EFFECT_STATE_NONE: int = 0
EFFECT_STATE_BURST: int = 1
EFFECT_STATE_SMOKE: int = 2
LABEL_UNKNOWN: int = -1  # out_of_scope等、per-cell真値が記録されていない行

VIDEO_ID_PREFIX: str = "video_"
LAYER_TELOP_NEGATIVE: str = "telop_negative"

# study側の特徴量セット・行帯定義を再利用 (出典: 本ファイルdocstring §統合較正)
FEATURE_NAMES: tuple[str, ...] = _ES.FEATURE_NAMES
BURST_ROW_MIN: int = _ES.BURST_ROW_MIN  # 1
BURST_ROW_MAX: int = _ES.BURST_ROW_MAX  # 3
# smoke は固定行帯が成立しない (本スクリプト実行結果、docstring参照) ため
# 可視全域 (隠し段除く abs_row 1-12) を対象窓とする。
SMOKE_ROW_MIN: int = HIDDEN_ROWS
SMOKE_ROW_MAX: int = HIDDEN_ROWS + 11

# 窓レベル連続スコアの候補集約方法。「単セル最大値」は光沢誤反応1セルに
# 脆弱なため、上位K平均も併用する (本スクリプトで両方AUC比較して選定)。
BAND_TOP_K_FOR_MEAN: int = 3

# out_of_scope (burst/smoke layer) 枠のうち、telop_negative同様「観測対象側が
# 自分の連鎖中 (発火フラッシュ or Xれんさ!テロップ) を写している」ことを
# 目視確認済みの候補 (2026-08-04 手動レビュー、
# frames/{video_stem}_t{t_sec}_{side}_{layer}_full.png を直接確認)。
# layer=burst/smokeの候補窓選定はnoteに自連鎖情報を残さないため、
# telop_negativeのnoteのような自動判定ができず、目視で個別に確定した。
MANUALLY_CONFIRMED_SELF_CHAIN_FRAMES: frozenset[tuple[str, float, str]] = frozenset({
    ("c61", 2842.29, "1P"),  # 自分の"1れんさ!"発火フラッシュが写る (相手連鎖窓と誤認)
    ("c9", 2298.11, "1P"),   # 同上
    ("c77", 1391.95, "1P"),  # 同上 (smoke窓由来だが実際は自分の発火フラッシュ)
})

# 誤発火なし(fired=False)だが本命閾値のこの幅以内に接近した候補を
# 「near-miss」として報告する (閾値の頑健性・残課題の可視化用)。
NEAR_MISS_MARGIN: float = 0.05


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class FrameRecord:
    """1候補フレーム分の突合済みレコード (72セル特徴量 + 真値ラベル配列)。"""

    source: str          # "v1" / "v3"
    video_stem: str
    side: str
    t_sec: float
    layer: str
    note: str
    status: str
    label_grid: np.ndarray   # shape=(12,6)、値は0/1/2、真値不明セルはLABEL_UNKNOWN
    features: dict[str, np.ndarray]  # feat名 -> shape=(12,6) の画素統計


# =============================================================================
# 1. CSV 読込・分類
# =============================================================================


def load_labeling_result_rows(csv_path: Path) -> list[dict]:
    """labeling_result.csv を読み込む (BOM対応)。"""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def classify_row_status(row: dict) -> str:
    """status列を正規化する (空文字は「未処理」として明示区別)。"""
    status = (row.get("status") or "").strip()
    return status if status else STATUS_UNLABELED


def decode_effect_grid_or_none(encoded: str) -> np.ndarray | None:
    """effect_grid文字列 (可視12行×6列) を復元する。空文字はNone (真値未記録)。"""
    encoded = (encoded or "").strip()
    if not encoded:
        return None
    rows = encoded.split("/")
    grid = np.zeros((len(rows), len(rows[0])), dtype=np.int64)
    for r, row_str in enumerate(rows):
        for c, ch in enumerate(row_str):
            grid[r, c] = int(ch)
    return grid


def video_stem_from_id(video_id: str) -> str:
    """"video_c18" -> "c18"。"""
    return video_id[len(VIDEO_ID_PREFIX):] if video_id.startswith(VIDEO_ID_PREFIX) else video_id


def frame_image_path(row: dict, frames_dir: Path) -> Path:
    """CSV1行からラベル付け時に使われた実画面フルフレームPNGのパスを復元する。"""
    stem = video_stem_from_id(row["video_id"])
    base = f"{stem}_t{row['t_sec']}_{row['side']}_{row['layer']}"
    return frames_dir / f"{base}_full.png"


# =============================================================================
# 2. 突合 (真値ラベル + 画素特徴量、72セル分をフレーム単位でまとめる)
# =============================================================================


def build_frame_record(source: str, row: dict, frame: np.ndarray) -> FrameRecord:
    """1フレーム分 (可視12行×6列=72セル) の FrameRecord を組み立てる。

    out_of_scope/skip等でeffect_gridが空の場合はlabel_grid全セルLABEL_UNKNOWN
    (真値不明、ROC学習には使わないが窓スコアの誤発火チェックには使う)。
    """
    grid = decode_effect_grid_or_none(row.get("effect_grid", ""))
    region = _ES.region_for_side(row["side"])
    stem = video_stem_from_id(row["video_id"])
    status = classify_row_status(row)
    n_vis_rows = grid.shape[0] if grid is not None else 12
    n_cols = grid.shape[1] if grid is not None else BOARD_COLS
    label_grid = np.full((n_vis_rows, n_cols), LABEL_UNKNOWN, dtype=np.int64)
    if grid is not None:
        label_grid = grid
    features: dict[str, np.ndarray] = {f: np.zeros((n_vis_rows, n_cols)) for f in FEATURE_NAMES}
    for vis_row in range(n_vis_rows):
        abs_row = vis_row + HIDDEN_ROWS
        for col in range(n_cols):
            x1, y1, x2, y2 = region.cell_sample_rect(abs_row, col)
            patch = frame[y1:y2, x1:x2]
            feats = _ES.compute_cell_features(patch)
            for f in FEATURE_NAMES:
                features[f][vis_row, col] = feats[f]
    return FrameRecord(
        source=source, video_stem=stem, side=row["side"], t_sec=float(row["t_sec"]),
        layer=row["layer"], note=row.get("note", ""), status=status,
        label_grid=label_grid, features=features,
    )


def build_all_frame_records(
    sources: tuple[tuple[str, Path, Path], ...],
) -> tuple[list[FrameRecord], dict[str, int], list[str]]:
    """全ラベルソースから usable (skip/unlabeled以外) フレームを組み立てる。

    Returns: (records, status_counts, missing_frame_warnings)
    """
    records: list[FrameRecord] = []
    status_counts: dict[str, int] = {}
    warnings: list[str] = []
    for source_name, csv_path, frames_dir in sources:
        rows = load_labeling_result_rows(csv_path)
        for row in rows:
            status = classify_row_status(row)
            status_counts[status] = status_counts.get(status, 0) + 1
            if status not in (STATUS_NO_EFFECT, STATUS_MARKED, STATUS_OUT_OF_SCOPE):
                continue  # skip/未処理は較正・誤発火チェックのどちらにも使わない
            img_path = frame_image_path(row, frames_dir)
            frame = cv2.imread(str(img_path))
            if frame is None:
                warnings.append(f"画像読込失敗: {img_path}")
                continue
            records.append(build_frame_record(source_name, row, frame))
    return records, status_counts, warnings


# =============================================================================
# 3. 行分布診断 (burst/smoke 真値セルの abs_row ヒストグラム)
# =============================================================================


def compute_true_label_row_histogram(records: list[FrameRecord]) -> pd.DataFrame:
    """真値ラベル (1=burst/2=smoke) セルの abs_row 分布を集計する (窓帯導出用)。"""
    rows_out: list[dict] = []
    for rec in records:
        if rec.status != STATUS_MARKED:
            continue
        for vis_row in range(rec.label_grid.shape[0]):
            abs_row = vis_row + HIDDEN_ROWS
            for col in range(rec.label_grid.shape[1]):
                v = int(rec.label_grid[vis_row, col])
                if v != EFFECT_STATE_NONE:
                    rows_out.append({"abs_row": abs_row, "label": v})
    df = pd.DataFrame(rows_out)
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index="abs_row", columns="label", aggfunc="size", fill_value=0)


# =============================================================================
# 4. 窓レベルスコア (burst窓 row1-3 内での集約統計量)
# =============================================================================


def band_feature_values(rec: FrameRecord, feature: str, row_min: int, row_max: int) -> np.ndarray:
    """指定feature・abs_row帯内の全セル値を1次元配列で返す。"""
    values: list[float] = []
    grid = rec.features[feature]
    for vis_row in range(grid.shape[0]):
        abs_row = vis_row + HIDDEN_ROWS
        if row_min <= abs_row <= row_max:
            values.extend(grid[vis_row, :].tolist())
    return np.array(values)


def band_max(rec: FrameRecord, feature: str, row_min: int, row_max: int) -> float:
    """窓内セルの最大値。光沢1セルに脆弱だが最も高感度。"""
    vals = band_feature_values(rec, feature, row_min, row_max)
    return float(np.max(vals)) if vals.size > 0 else float("nan")


def band_topk_mean(
    rec: FrameRecord, feature: str, row_min: int, row_max: int, k: int = BAND_TOP_K_FOR_MEAN,
) -> float:
    """窓内セルの上位k個平均。単セル誤反応より頑健。"""
    vals = band_feature_values(rec, feature, row_min, row_max)
    if vals.size == 0:
        return float("nan")
    k_eff = min(k, vals.size)
    return float(np.mean(np.sort(vals)[-k_eff:]))


def has_true_effect_in_band(
    rec: FrameRecord, effect_state: int, row_min: int, row_max: int,
) -> bool:
    """窓内に真値ラベルがeffect_state(1=burst/2=smoke)のセルが1つ以上あるか。"""
    for vis_row in range(rec.label_grid.shape[0]):
        abs_row = vis_row + HIDDEN_ROWS
        if row_min <= abs_row <= row_max:
            if bool(np.any(rec.label_grid[vis_row, :] == effect_state)):
                return True
    return False


# =============================================================================
# 5. frame-level ROC (burst窓、正例=marked中の真値burstセルを含む窓)
# =============================================================================


def max_tpr_at_zero_fp(pos: np.ndarray, neg: np.ndarray) -> dict:
    """FPR=0を厳密に保証する閾値でのTPRを計算する (方向はAUCで自動判定)。"""
    from sklearn.metrics import roc_auc_score

    if len(pos) == 0 or len(neg) == 0:
        return {"threshold": float("nan"), "tpr_at_zero_fp": float("nan"), "auc": float("nan")}
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    x = np.concatenate([pos, neg])
    auc = float(roc_auc_score(y, x)) if len(set(y.tolist())) > 1 else 0.5
    direction = 1.0 if auc >= 0.5 else -1.0
    score_pos, score_neg = direction * pos, direction * neg
    thr_dir = float(np.max(score_neg))
    tpr = float(np.mean(score_pos > thr_dir))
    threshold = thr_dir if direction >= 0 else -thr_dir
    return {
        "threshold": threshold, "tpr_at_zero_fp": tpr,
        "auc": auc if direction >= 0 else 1.0 - auc, "n_pos": len(pos), "n_neg": len(neg),
    }


def best_youden_threshold(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Youden's J (TPR-FPR最大) の閾値・TPR・FPRを返す (方向はAUCで自動判定)。"""
    from sklearn.metrics import roc_auc_score, roc_curve

    if len(pos) == 0 or len(neg) == 0:
        return {"threshold": float("nan"), "tpr": float("nan"), "fpr": float("nan")}
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    x = np.concatenate([pos, neg])
    auc = float(roc_auc_score(y, x)) if len(set(y.tolist())) > 1 else 0.5
    score = x if auc >= 0.5 else -x
    fpr, tpr, thr = roc_curve(y, score)
    best_i = int(np.argmax(tpr - fpr))
    threshold = thr[best_i] if auc >= 0.5 else -thr[best_i]
    return {"threshold": float(threshold), "tpr": float(tpr[best_i]), "fpr": float(fpr[best_i])}


def build_burst_frame_scores(records: list[FrameRecord]) -> pd.DataFrame:
    """burst窓 (row1-3) の frame-level 統計量表 (v_mean_max / v_mean_topk_mean) を作る。

    真値: marked中に真値burstセル(=1)を含む窓=陽性、no_effect窓=陰性
    (out_of_scopeはROC学習に使わない、§6で別途誤発火チェック)。
    """
    rows_out: list[dict] = []
    for rec in records:
        if rec.status not in (STATUS_MARKED, STATUS_NO_EFFECT):
            continue
        row_out = {
            "source": rec.source, "video_stem": rec.video_stem, "side": rec.side,
            "t_sec": rec.t_sec, "layer": rec.layer, "status": rec.status,
            "is_true_burst": rec.status == STATUS_MARKED and has_true_effect_in_band(
                rec, EFFECT_STATE_BURST, BURST_ROW_MIN, BURST_ROW_MAX,
            ),
        }
        for feat in FEATURE_NAMES:
            row_out[f"{feat}_max"] = band_max(rec, feat, BURST_ROW_MIN, BURST_ROW_MAX)
            row_out[f"{feat}_topk_mean"] = band_topk_mean(rec, feat, BURST_ROW_MIN, BURST_ROW_MAX)
        rows_out.append(row_out)
    return pd.DataFrame(rows_out)


def build_burst_roc_table(score_df: pd.DataFrame) -> pd.DataFrame:
    """score列 (*_max, *_topk_mean) ごとにROC動作点を計算し、AUC降順に並べる。"""
    score_cols = [c for c in score_df.columns if c.endswith("_max") or c.endswith("_topk_mean")]
    y = score_df["is_true_burst"].to_numpy()
    rows_out = []
    for col in score_cols:
        x = score_df[col].to_numpy()
        pos, neg = x[y], x[~y]
        zero_fp = max_tpr_at_zero_fp(pos, neg)
        youden = best_youden_threshold(pos, neg)
        rows_out.append({
            "score": col, "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
            "auc": round(zero_fp["auc"], 3),
            "zero_fp_threshold": round(zero_fp["threshold"], 2),
            "zero_fp_tpr": round(zero_fp["tpr_at_zero_fp"], 3),
            "youden_threshold": round(youden["threshold"], 2),
            "youden_tpr": round(youden["tpr"], 3),
            "youden_fpr": round(youden["fpr"], 3),
        })
    return pd.DataFrame(rows_out).sort_values("auc", ascending=False).reset_index(drop=True)


# =============================================================================
# 6. out_of_scope / telop_negative の誤発火チェック (テロップ判別の検証)
# =============================================================================


def build_out_of_scope_misfire_table(
    records: list[FrameRecord], score_feature: str, threshold: float,
) -> pd.DataFrame:
    """out_of_scope全件 + telop_negative(no_effect含む)のburst窓誤発火を判定する。

    score_feature: build_burst_roc_table で選定した本命特徴量 ("v_mean_max" 等、
    "_max"/"_topk_mean" サフィックスを含む列名)。
    """
    base_feat = score_feature.replace("_max", "").replace("_topk_mean", "")
    is_max = score_feature.endswith("_max")
    rows_out = []
    for rec in records:
        if rec.status != STATUS_OUT_OF_SCOPE and rec.layer != LAYER_TELOP_NEGATIVE:
            continue
        score = (
            band_max(rec, base_feat, BURST_ROW_MIN, BURST_ROW_MAX) if is_max
            else band_topk_mean(rec, base_feat, BURST_ROW_MIN, BURST_ROW_MAX)
        )
        is_self_chain = (
            rec.layer == LAYER_TELOP_NEGATIVE
            or (rec.video_stem, rec.t_sec, rec.side) in MANUALLY_CONFIRMED_SELF_CHAIN_FRAMES
        )
        rows_out.append({
            "source": rec.source, "video_stem": rec.video_stem, "side": rec.side,
            "t_sec": rec.t_sec, "layer": rec.layer, "status": rec.status, "note": rec.note,
            "is_self_chain_scenario": is_self_chain,
            "score": round(score, 2), "fired": bool(score > threshold),
        })
    if not rows_out:
        # 該当行が1件もない場合、sort_values("score")がKeyErrorになるのを防ぐ
        # (空でも呼び出し側が同じ列構成のDataFrameを期待するため列だけ用意する)
        return pd.DataFrame(columns=[
            "source", "video_stem", "side", "t_sec", "layer", "status", "note",
            "is_self_chain_scenario", "score", "fired",
        ])
    return pd.DataFrame(rows_out).sort_values("score", ascending=False).reset_index(drop=True)


# =============================================================================
# 7. レポート生成
# =============================================================================


def _df_to_markdown(df: pd.DataFrame) -> str:
    """tabulate非依存の簡易markdown表変換。"""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(str(row[c]) for c in cols) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep] + body)


def build_calibration_report(
    status_counts: dict[str, int], row_hist: pd.DataFrame, roc_table: pd.DataFrame,
    misfire_table: pd.DataFrame, best_score: str, best_threshold: float,
    warnings: list[str],
) -> str:
    """calibration_report_v3.md の本文を組み立てる。"""
    lines: list[str] = []
    lines.append("# 案B閾値 統合較正レポート v3 (2026-08-04)\n")
    lines.append(f"## 1. ラベル件数 (v1+v3統合)\n\n{status_counts}\n")
    lines.append("## 2. 真値セルのabs_row分布 (窓帯導出根拠)\n")
    lines.append(_df_to_markdown(row_hist.reset_index()))
    lines.append(
        "\n列1=burst真値セル、列2=smoke真値セル。burst はrow1-3に集中"
        f"(BURST_ROW_MIN={BURST_ROW_MIN}, MAX={BURST_ROW_MAX}を維持)。smokeは"
        "全12行にほぼ一様分布=固定窓不成立 (別途課題、本レポートのGO/NO-GO判定は"
        "burst検出のみを対象とする)。\n"
    )
    lines.append("## 3. burst窓 frame-level ROC (v1+v3統合)\n")
    lines.append(_df_to_markdown(roc_table))
    lines.append(f"\n**本命採用**: `{best_score}` (閾値={best_threshold:.2f})\n")
    lines.append("## 4. out_of_scope / telop_negative 誤発火チェック (テロップ判別)\n")
    lines.append(_df_to_markdown(misfire_table))
    n_total = len(misfire_table)
    n_fired = int(misfire_table["fired"].sum()) if n_total else 0
    n_self_chain = int(misfire_table["is_self_chain_scenario"].sum()) if n_total else 0
    n_self_chain_fired = (
        int(misfire_table.loc[misfire_table["is_self_chain_scenario"], "fired"].sum())
        if n_total else 0
    )
    n_other = n_total - n_self_chain
    n_other_fired = n_fired - n_self_chain_fired
    lines.append(
        f"\n**誤発火件数: {n_fired}/{n_total}**"
        f" (自連鎖テロップ想定枠 {n_self_chain_fired}/{n_self_chain}、"
        f"その他 {n_other_fired}/{n_other})。\n"
    )
    if n_other_fired == 0:
        lines.append(
            "**「自分の連鎖中は判定抑制」ゲートで十分**: 誤発火は全て"
            "telop_negative枠 (自連鎖テロップ想定) に限られ、それ以外の"
            "out_of_scope枠では誤発火0件。既存 ChainPhaseDetector の "
            "is_chain_p1/is_chain_p2 (src/chain_phase_detector.py) を"
            "「観測対象側」の連鎖中フラグとして時間ゲートに使えば、"
            "確認された誤発火経路は全て塞げる。\n"
        )
    else:
        lines.append(
            f"**要注意**: telop_negative以外のout_of_scope枠でも{n_other_fired}件"
            "誤発火。自連鎖テロップ以外の混同要因が残っている、個別確認が必要。\n"
        )
    near_miss = misfire_table[
        (~misfire_table["fired"]) & (~misfire_table["is_self_chain_scenario"])
        & (misfire_table["score"] > best_threshold - NEAR_MISS_MARGIN)
    ]
    if len(near_miss) > 0:
        lines.append(
            f"\n**残課題 (near-miss、閾値-{NEAR_MISS_MARGIN}以内で未発火)**: "
            f"{len(near_miss)}件が閾値付近まで接近 (目視確認で `全消し`(全消しテロップ) "
            "写り込みを1件確認、既存 `src/all_clear_detector.py` の "
            "`is_all_clear(board, score)` を第3のゲート条件として追加すべき"
            "残課題として明記する — is_chain_p*だけでは全消しテロップの"
            "持続期間 (次のぷよ消去まで) を必ずしも覆わない)。\n"
        )
        lines.append(_df_to_markdown(near_miss))
        lines.append("")
    if warnings:
        lines.append("\n## 警告 (画像欠損等)\n")
        lines.extend(f"- {w}" for w in warnings)
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# main
# =============================================================================


def main() -> None:
    """メイン処理: 突合 -> 行分布診断 -> burst ROC較正 -> 誤発火チェック -> レポート。"""
    cv2.setNumThreads(1)  # 熱対策・並列しない
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] labeling_result.csv 読込 + 突合 (v1+v3統合)")
    records, status_counts, warnings = build_all_frame_records(LABEL_SOURCES)
    print(f"  usable frames: {len(records)} (status counts: {status_counts})")

    print("[2/5] 真値行分布診断")
    row_hist = compute_true_label_row_histogram(records)

    print("[3/5] burst窓 frame-level ROC")
    score_df = build_burst_frame_scores(records)
    score_df.to_csv(OUTPUT_DIR / "labeled_cell_features_v3.csv", index=False, encoding="utf-8-sig")
    roc_table = build_burst_roc_table(score_df)
    roc_table.to_csv(OUTPUT_DIR / "burst_roc_operating_points.csv", index=False, encoding="utf-8-sig")
    best_score = str(roc_table.iloc[0]["score"])
    best_threshold = float(roc_table.iloc[0]["zero_fp_threshold"])

    print("[4/5] out_of_scope / telop_negative 誤発火チェック")
    misfire_table = build_out_of_scope_misfire_table(records, best_score, best_threshold)
    misfire_table.to_csv(OUTPUT_DIR / "out_of_scope_misfire.csv", index=False, encoding="utf-8-sig")

    print("[5/5] レポート出力")
    report = build_calibration_report(
        status_counts, row_hist, roc_table, misfire_table, best_score, best_threshold, warnings,
    )
    (OUTPUT_DIR / "calibration_report_v3.md").write_text(report, encoding="utf-8")
    print(f"\n[DONE] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
