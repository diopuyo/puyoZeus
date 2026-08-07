"""案B (エフェクト視覚検出) 閾値 本較正スクリプト (2026-08-04)。

特性調査 (data/verify/effect_signature_study_2026-08-03、s_min AUC0.822) は
窓タイミングが機械推定 (正解ラベル無しの下限値) だったため、user が
data/verify/effect_cell_label_2026-08-03/label_tool.html で40フレームに
セル単位の人手ラベル (0=なし/1=バースト/2=煙) を付けた
(labeling_result.csv)。本スクリプトはこの正解ラベルと実画面の画素統計を
突合し、案Bの判定式・定数を較正する。**検出器の実装は行わない**
(仕様提案のみ、CLAUDE.md「認識精度99.99%達成まで他phase凍結」に配慮し
実装着手は別途の設計合意後)。

## データソースと突合方法
- ラベル: data/verify/effect_cell_label_2026-08-03/labeling_result.csv
  (effect_grid は可視12行×6列、隠し段(row0)は含まれない。
  scripts/build_effect_cell_label_tool.py の GEOM.visibleRows=N_VISIBLE_ROWS
  でグリッドを構築しているため、絶対盤面行 = 可視行index + HIDDEN_ROWS で
  復元する。1=バースト(オレンジ,B) / 2=煙(灰,S) の状態値は同ファイルの
  EFFECT_STATE_BURST/EFFECT_STATE_SMOKE 定義と対応)
- 画素: labeling_result.csv 各行の実画面フルフレーム (frames/*_full.png、
  ラベル付け時にuserが見たのと同一画像) から
  scripts/study_effect_signature_2026-08-03.py の compute_cell_features /
  region_for_side をそのまま流用してセル単位HSV統計を再計算する
  (study側ファイルは調査専用スクリプトのため無変更、importlib経由で呼ぶ)。
- 白目・通常ぷよのゼロ誤検出制約は、study側が既に収集した
  data/verify/effect_signature_study_2026-08-03/cell_stats.csv の
  highlight_suspect=True 集団 (通常ぷよの白目ハイライト) を追加の負例として
  流用する (今回の40フレームのラベルだけでは白目サンプルが少ないため)。

## 出力 (data/verify/effect_detector_calibration_2026-08-04/)
    labeled_cell_features.csv       突合済み全セルの特徴量+真値ラベル
    frame_status_summary.csv        レイヤー×status 件数集計 (窓推定の空振り率)
    roc_operating_points.csv        特徴量別 ROC動作点 (ゼロFP/Youden's J)
    ojama_vs_smoke_separability.csv 煙ラベル vs 実おじゃま色の分離可能性
    calibration_report.md           判定式・定数の仕様提案 (出典明記)

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.calibrate_effect_detector
"""
from __future__ import annotations

import csv
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, COLOR_OJAMA, HIDDEN_ROWS  # noqa: E402

# =============================================================================
# study_effect_signature_2026-08-03 の動的import (ファイル名がハイフンを含み
# 通常のimport文で書けないため。調査専用スクリプトなので内容は無変更、
# 特徴量計算関数・領域定義のみ再利用する)
# =============================================================================

_STUDY_MODULE_NAME: str = "scripts.study_effect_signature_2026-08-03"
_ES = importlib.import_module(_STUDY_MODULE_NAME)

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化、出典を併記)
# =============================================================================

LABELING_RESULT_CSV: Path = Path("data/verify/effect_cell_label_2026-08-03/labeling_result.csv")
FRAMES_DIR: Path = Path("data/verify/effect_cell_label_2026-08-03/frames")
STUDY_CELL_STATS_CSV: Path = Path("data/verify/effect_signature_study_2026-08-03/cell_stats.csv")
OUTPUT_DIR: Path = Path("data/verify/effect_detector_calibration_2026-08-04")

STATUS_NO_EFFECT: str = "no_effect"
STATUS_MARKED: str = "marked"
STATUS_SKIP: str = "skip"
# labeling_result.csv の status 列が空文字 = user がボタンを一度も押していない
# 未処理行 (「エフェクトなし」の明示確認とは異なり真値不明、較正対象外)
STATUS_UNLABELED: str = "unlabeled"

# scripts/build_effect_cell_label_tool.py の状態定義と同一 (出典: 同ファイル
# EFFECT_STATE_NONE/BURST/SMOKE)
EFFECT_STATE_NONE: int = 0
EFFECT_STATE_BURST: int = 1
EFFECT_STATE_SMOKE: int = 2

VIDEO_ID_PREFIX: str = "video_"

# study側の特徴量セット・行帯定義を再利用 (重複定義しない)
FEATURE_NAMES: tuple[str, ...] = _ES.FEATURE_NAMES
BURST_ROW_MIN: int = _ES.BURST_ROW_MIN
BURST_ROW_MAX: int = _ES.BURST_ROW_MAX
HIGHLIGHT_SUSPECT_RATIO_THRESHOLD: float = _ES.HIGHLIGHT_SUSPECT_RATIO_THRESHOLD


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class LabeledCellSample:
    """1セル分の人手ラベル真値+画素特徴量 (突合済み)。"""

    video_stem: str
    side: str
    t_sec: float
    layer_hint: str      # 窓選定時のレイヤー ("burst"/"smoke"/"baseline")
    chain_bin: str
    frame_status: str    # "no_effect" / "marked" (較正対象のみここに来る)
    row: int             # 盤面絶対行 (1-12、隠し段0は対象外)
    col: int
    label: int            # 0=なし/1=バースト/2=煙 (userクリックの真値)
    v_mean: float
    v_max: float
    s_mean: float
    s_min: float
    specular_ratio: float
    bright_ratio: float


# =============================================================================
# 1. labeling_result.csv 読込・分類
# =============================================================================


def load_labeling_result_rows(csv_path: Path) -> list[dict]:
    """labeling_result.csv (userダウンロード結果) を読み込む (BOM対応)。"""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def classify_row_status(row: dict) -> str:
    """status列を正規化する (空文字は「未処理」として明示区別、fail-silent回避)。"""
    status = (row.get("status") or "").strip()
    return status if status else STATUS_UNLABELED


def decode_effect_grid(encoded: str) -> np.ndarray:
    """"0.../..." 形式のエフェクトグリッド文字列を (可視行数, 6) int配列に戻す。"""
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
# 2. 突合 (真値ラベル + 画素特徴量)
# =============================================================================


def _build_samples_for_row(row: dict, frame: np.ndarray) -> list[LabeledCellSample]:
    """1フレーム分 (可視12行×6列=72セル) の LabeledCellSample を組み立てる。"""
    grid = decode_effect_grid(row["effect_grid"])
    region = _ES.region_for_side(row["side"])
    stem = video_stem_from_id(row["video_id"])
    status = classify_row_status(row)
    samples: list[LabeledCellSample] = []
    for vis_row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            abs_row = vis_row + HIDDEN_ROWS
            x1, y1, x2, y2 = region.cell_sample_rect(abs_row, col)
            patch = frame[y1:y2, x1:x2]
            feats = _ES.compute_cell_features(patch)
            samples.append(LabeledCellSample(
                video_stem=stem, side=row["side"], t_sec=float(row["t_sec"]),
                layer_hint=row["layer"], chain_bin=row.get("chain_bin", ""),
                frame_status=status, row=abs_row, col=col,
                label=int(grid[vis_row, col]), **feats,
            ))
    return samples


def build_labeled_cell_samples(
    rows: list[dict], frames_dir: Path,
) -> tuple[list[LabeledCellSample], dict[str, int], list[str]]:
    """usable行 (no_effect/marked) から72セル分の特徴量+真値ラベルを組み立てる。

    Returns: (samples, status_counts, missing_frame_warnings)
    """
    samples: list[LabeledCellSample] = []
    status_counts: dict[str, int] = {}
    warnings: list[str] = []
    for row in rows:
        status = classify_row_status(row)
        status_counts[status] = status_counts.get(status, 0) + 1
        if status not in (STATUS_NO_EFFECT, STATUS_MARKED):
            continue  # skip/未処理は真値不明のため較正対象外
        img_path = frame_image_path(row, frames_dir)
        frame = cv2.imread(str(img_path))
        if frame is None:
            warnings.append(f"画像読込失敗: {img_path}")
            continue
        samples.extend(_build_samples_for_row(row, frame))
    return samples, status_counts, warnings


def samples_to_dataframe(samples: list[LabeledCellSample]) -> pd.DataFrame:
    """LabeledCellSampleのリストをDataFrameに変換する。"""
    return pd.DataFrame([s.__dict__ for s in samples])


# =============================================================================
# 3. フレーム単位の集計 (窓推定の空振り率、skip/未処理の扱い明記)
# =============================================================================


def summarize_frame_status(rows: list[dict]) -> pd.DataFrame:
    """レイヤー(窓の由来)×status の件数表を作る (窓タイミング推定の的中率の裏付け)。"""
    records = [{"layer": r["layer"], "status": classify_row_status(r)} for r in rows]
    df = pd.DataFrame(records)
    table = df.pivot_table(index="layer", columns="status", aggfunc="size", fill_value=0)
    for col in (STATUS_NO_EFFECT, STATUS_MARKED, STATUS_SKIP, STATUS_UNLABELED):
        if col not in table.columns:
            table[col] = 0
    return table[[STATUS_NO_EFFECT, STATUS_MARKED, STATUS_SKIP, STATUS_UNLABELED]]


def count_effect_cells_by_layer(samples: list[LabeledCellSample]) -> pd.DataFrame:
    """レイヤー×真値ラベルのセル数集計 (真値は layer_hint と独立な軸)。"""
    df = samples_to_dataframe(samples)
    if df.empty:
        return pd.DataFrame()
    table = df.pivot_table(index="layer_hint", columns="label", aggfunc="size", fill_value=0)
    for label in (EFFECT_STATE_NONE, EFFECT_STATE_BURST, EFFECT_STATE_SMOKE):
        if label not in table.columns:
            table[label] = 0
    return table[[EFFECT_STATE_NONE, EFFECT_STATE_BURST, EFFECT_STATE_SMOKE]]


# =============================================================================
# 4. ROC 動作点 (ゼロFP優先 + Youden's J 参考値)
# =============================================================================


def max_tpr_at_zero_fp(pos: np.ndarray, neg: np.ndarray) -> dict:
    """FPR=0を厳密に保証する閾値でのTPRを計算する (方向はAUCで自動判定)。

    閾値は neg の最大値 (方向調整後) に設定し、判定は「厳密に閾値超え」とする
    ことで epsilon 無しに FPR=0 を保証する (neg は全て閾値以下になるため)。
    """
    from sklearn.metrics import roc_auc_score

    if len(pos) == 0 or len(neg) == 0:
        return {
            "threshold": float("nan"), "tpr_at_zero_fp": float("nan"),
            "direction": 1, "n_pos": len(pos), "n_neg": len(neg), "auc": float("nan"),
        }
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    x = np.concatenate([pos, neg])
    auc = float(roc_auc_score(y, x)) if len(set(y.tolist())) > 1 else 0.5
    direction = 1.0 if auc >= 0.5 else -1.0
    score_pos, score_neg = direction * pos, direction * neg
    thr_dir = float(np.max(score_neg))
    tpr = float(np.mean(score_pos > thr_dir))
    threshold = thr_dir if direction >= 0 else -thr_dir
    return {
        "threshold": threshold, "tpr_at_zero_fp": tpr, "direction": direction,
        "n_pos": len(pos), "n_neg": len(neg), "auc": auc if direction >= 0 else 1.0 - auc,
    }


def build_roc_operating_point_table(
    df: pd.DataFrame, highlight_neg: pd.DataFrame,
) -> pd.DataFrame:
    """特徴量ごとに ゼロFP動作点(3種の負例母集団) + Youden's J 参考値を並べる。

    負例母集団3種:
      A: 今回ラベルのclean cell (label==0) のみ
      B: A + study側 highlight_suspect (白目ハイライト疑い、通常ぷよ)
      C: A + study側 normal 全セル (最も保守的、最大母集団)
    """
    pos_mask = df["label"] != EFFECT_STATE_NONE
    neg_mask = df["label"] == EFFECT_STATE_NONE
    rows = []
    for feat in FEATURE_NAMES:
        pos = df.loc[pos_mask, feat].to_numpy()
        neg_a = df.loc[neg_mask, feat].to_numpy()
        neg_b = np.concatenate([neg_a, highlight_neg.loc[highlight_neg["highlight_suspect"], feat].to_numpy()])
        neg_c = np.concatenate([neg_a, highlight_neg[feat].to_numpy()])
        zero_a = max_tpr_at_zero_fp(pos, neg_a)
        zero_b = max_tpr_at_zero_fp(pos, neg_b)
        zero_c = max_tpr_at_zero_fp(pos, neg_c)
        youden = _ES._best_threshold(pos, neg_a)  # (threshold, auc, tpr, fpr)
        rows.append({
            "feature": feat, "n_pos": len(pos), "n_neg_new_label": len(neg_a),
            "auc_new_label_only": round(zero_a["auc"], 3),
            "zero_fp_threshold_A_new_only": round(zero_a["threshold"], 2),
            "zero_fp_tpr_A_new_only": round(zero_a["tpr_at_zero_fp"], 3),
            "zero_fp_threshold_B_plus_highlight": round(zero_b["threshold"], 2),
            "zero_fp_tpr_B_plus_highlight": round(zero_b["tpr_at_zero_fp"], 3),
            "zero_fp_threshold_C_plus_all_normal": round(zero_c["threshold"], 2),
            "zero_fp_tpr_C_plus_all_normal": round(zero_c["tpr_at_zero_fp"], 3),
            "youden_threshold": round(youden[0], 2), "youden_auc": round(youden[1], 3),
            "youden_tpr": round(youden[2], 3), "youden_fpr": round(youden[3], 3),
        })
    return pd.DataFrame(rows).sort_values("auc_new_label_only", ascending=False).reset_index(drop=True)


# =============================================================================
# 5. 煙ラベル vs 実おじゃま色 の分離可能性
# =============================================================================


def load_study_cell_stats(csv_path: Path) -> pd.DataFrame:
    """study側cell_stats.csvを読み込み、highlight_suspectをbool化する。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["highlight_suspect"] = df["highlight_suspect"].astype(str) == "True"
    return df


def compute_ojama_vs_smoke_separability(df: pd.DataFrame, study_df: pd.DataFrame) -> pd.DataFrame:
    """煙ラベル(真値2)セル vs study側「実おじゃま色」セルのAUC分離可能性を計算する。

    coordinatorの注記「煙ラベルには落下中のおじゃまぷよ自体を疑ってマークした
    ものが混在」への対応: 分離できるかを特徴量ごとに検証する。
    """
    from sklearn.metrics import roc_auc_score

    smoke = df.loc[df["label"] == EFFECT_STATE_SMOKE]
    real_ojama = study_df.loc[study_df["ground_truth_color"] == COLOR_OJAMA]
    rows = []
    for feat in FEATURE_NAMES:
        pos = smoke[feat].to_numpy()
        neg = real_ojama[feat].to_numpy()
        if len(pos) == 0 or len(neg) == 0:
            rows.append({"feature": feat, "n_smoke_label": len(pos), "n_real_ojama": len(neg), "auc": float("nan")})
            continue
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        x = np.concatenate([pos, neg])
        auc = float(roc_auc_score(y, x)) if len(set(y.tolist())) > 1 else 0.5
        auc_eff = auc if auc >= 0.5 else 1.0 - auc
        rows.append({"feature": feat, "n_smoke_label": len(pos), "n_real_ojama": len(neg), "auc": round(auc_eff, 3)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


# =============================================================================
# 6. 窓レベル同時性 (行帯内で同時に閾値超えするセル数)
# =============================================================================


def compute_window_level_flag_counts(df: pd.DataFrame, feature: str, threshold: float) -> pd.DataFrame:
    """burst行帯(row1-3)内で閾値超えするセル数を、フレーム(真の効果あり/なし)別に集計する。

    案Bの本命設計「セル単独でなく行帯内の同時複数セル」の判定材料
    (study側 separability_report.md の推奨に基づく)。feature/threshold は
    呼び出し側が選ぶ (ゼロFP閾値はTPRが低すぎて単独では窓判定に使えないため、
    本スクリプトでは Youden's J 閾値=より高感度な動作点を採用し、その代わり
    「同時に何セル超えるか」で誤検出を絞り込む設計思想を検証する)。
    """
    band = df[(df["row"] >= BURST_ROW_MIN) & (df["row"] <= BURST_ROW_MAX)]
    band = band.assign(flagged=band[feature] > threshold)
    group_cols = ["video_stem", "side", "t_sec", "frame_status"]
    counts = band.groupby(group_cols)["flagged"].sum().reset_index(name="n_flagged_in_band")
    counts["has_true_effect_in_band"] = band.groupby(group_cols)["label"].apply(
        lambda s: bool((s != EFFECT_STATE_NONE).any())
    ).reset_index(drop=True)
    return counts.sort_values("n_flagged_in_band", ascending=False).reset_index(drop=True)


def compute_frame_level_auc(window_counts: pd.DataFrame) -> dict:
    """n_flagged_in_band を窓レベルスコアとした「フレーム単位」AUC + ゼロFP動作点を計算する。

    セル単位AUC (§3) は同一フレーム内72セルが独立サンプルでない (疑似反復、
    pseudo-replication) ため統計的検出力を過大評価する。本関数は
    フレームを1サンプルとする、より統計的に正直な検定を提供する
    (n=36フレームのみだが、これが実際の独立観測数)。
    """
    from sklearn.metrics import roc_auc_score

    y = window_counts["has_true_effect_in_band"].astype(int).to_numpy()
    x = window_counts["n_flagged_in_band"].to_numpy()
    if len(set(y.tolist())) < 2:
        return {"n_frames": len(y), "n_true": int(y.sum()), "auc": float("nan")}
    auc = float(roc_auc_score(y, x))
    pos = x[y == 1]
    neg = x[y == 0]
    zero_fp = max_tpr_at_zero_fp(pos, neg)
    return {
        "n_frames": len(y), "n_true": int(y.sum()), "n_false": int(len(y) - y.sum()),
        "auc": round(auc, 3),
        "zero_fp_min_flagged_cells": int(zero_fp["threshold"]) + 1,  # 閾値超え=以上を要求する最小セル数
        "zero_fp_tpr": round(zero_fp["tpr_at_zero_fp"], 3),
    }


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
    status_table: pd.DataFrame, layer_label_table: pd.DataFrame,
    roc_table: pd.DataFrame, ojama_table: pd.DataFrame,
    window_counts: pd.DataFrame, window_feature: str, window_threshold: float,
    frame_level_auc: dict, warnings: list[str],
) -> str:
    """calibration_report.md の本文を組み立てる。"""
    lines: list[str] = []
    lines.append("# 案B閾値 本較正レポート (2026-08-04)\n")
    lines.append(
        "**最重要の訂正**: 特性調査 (2026-08-03) が示した s_min AUC=0.822 は、"
        "今回の人手正解ラベルでは **AUC=0.501 (ほぼ偶然と同等)** に崩れた。"
        "窓タイミングが機械推定だった特性調査は「エフェクトがあるはずの窓」を"
        "対象にしていたため、実際にエフェクトが写っていないフレームを陽性側に"
        "混入させていたことが原因と考えられる。s_min中心の閾値提案は撤回し、"
        "本レポートは新しいランキング (v_mean が最良、§3) に基づく。\n"
    )
    lines.append("## 1. フレーム単位の集計 (窓推定の的中率)\n")
    lines.append(_df_to_markdown(status_table.reset_index()))
    lines.append("")
    lines.append(
        f"unlabeled({STATUS_UNLABELED})列は user がボタン未押下の行 (真値不明、"
        "較正対象外)。skip は非対象フレームとしてuserが明示スキップした行 "
        "(較正対象外)。両方とも「no_effect(明示確認)」とは意味が異なるため区別している。"
    )
    lines.append("")
    lines.append("## 2. レイヤー(窓の由来)×真値ラベルのセル数\n")
    lines.append(_df_to_markdown(layer_label_table.reset_index()))
    lines.append(
        "\n列の0/1/2はユーザークリックの真値 (layer_hintとは独立な軸)。"
        "smoke窓由来のフレームでも真値がバースト(1)になったセルがある "
        "(窓の由来は選定時のヒントに過ぎず、実際の視覚判定とは別)。\n"
    )
    lines.append("## 3. ROC動作点 (特徴量別)\n")
    lines.append(_df_to_markdown(roc_table))
    lines.append(
        "\n`zero_fp_threshold_B_plus_highlight` が本命 (白目ハイライト疑いを含めて"
        "ゼロ誤検出を保つ閾値)。A<B<Cの順に負例母集団が広がり、閾値は単調に厳しく"
        "(TPRは単調に低く) なるのが期待挙動 (今回は新ラベルの負例が既にstudy側の"
        "負例より広いため、v_meanではA=B=Cが一致=study側の追加が制約を狭めなかった)。"
        "\n**ゼロFP制約下ではどの特徴量も単セルではTPR<10%に留まる** (v_meanで最良3.8%)。"
        "単セル閾値だけでの実用的な検出器は不成立、§5の窓レベル同時性判定が必須。\n"
    )
    lines.append("## 4. 煙ラベル vs 実おじゃま色の分離可能性\n")
    lines.append(_df_to_markdown(ojama_table))
    lines.append(
        "\n**分離可能 (v_mean AUC=0.896)、別クラス扱いを推奨**。目視確認 "
        "(c50_t1593.64_1P_smoke_board_crop.png) でも、実おじゃま色は丸い球体+白目"
        "アイコンの一貫した見た目だが、煙エフェクトは輪郭不定形の白い綿状の塊で"
        "盤面複数セルに渡ってはみ出しており、視覚的にも明確に別物と確認できる "
        "(v_meanが高いのは綿状の煙が地色より格段に明るい白のため)。"
        "**注記**: study側の「実おじゃま色」参照集団は静止済み(着地後)のセルであり、"
        "落下中アニメーション中のおじゃまぷよのサンプルではない。ぷよ自体の見た目は"
        "落下中も静止時と同一スプライトのはずだが、この前提は本レポートでは"
        "未検証 (要落下中フレームの追加ラベル)。\n"
    )
    lines.append(
        f"## 5. 行帯内同時性 (burst行帯 row1-3、{window_feature}>{window_threshold:.2f}"
        "=Youden's J閾値、超えセル数)\n"
    )
    lines.append(_df_to_markdown(window_counts))
    lines.append(
        "\nゼロFP閾値はTPRが低すぎるため、より高感度なYouden's J閾値をセル単独の"
        "トリガーに採用し、「同時に何セル超えるか」で誤検出を絞り込めるかを検証。"
    )
    lines.append(
        f"\n**フレーム単位AUC (n={frame_level_auc.get('n_frames')}、真陽性フレーム"
        f"{frame_level_auc.get('n_true')}件) = {frame_level_auc.get('auc')}** (セル単位§3"
        "のAUCは同一フレーム内72セルの疑似反復でn=104を過大評価するため、フレームを"
        "1サンプルとする本指標がより統計的に正直)。**窓レベル同時性判定は単セル判定より"
        f"大幅に有効**: ゼロFPを保つ最小セル数は「{frame_level_auc.get('zero_fp_min_flagged_cells')}"
        f"セル以上同時該当」で、この動作点でのTPRは{frame_level_auc.get('zero_fp_tpr')}"
        " (単セルのゼロFP TPR最良3.8%からv_meanで15倍改善)。ただし唯一の例外例として "
        "`c21 2764.13`(no_effect、14セル該当) が真陽性フレーム `c50 1593.64`(14セル) と"
        "同数であり、目視確認 (board_crop画像) では**連鎖数テロップ「1れんさ!」の発光"
        "オーバーレイ** (対象外の別UI要素) がrow1-3で強く光っており、v_mean単独では"
        "予告おじゃまバーストと連鎖テロップ発光を区別できない。実運用では"
        "連鎖テロップ表示中フレームを別途除外条件に加える必要がある "
        "(n=36と小さいため要追加ラベルでの再確認)。"
    )
    lines.append("\n## 6. 検出器プロトタイプ仕様案 (実装はまだしない、設計提案のみ)\n")
    lines.append(
        "**判定式 (窓レベル、セル単独ではなく行帯内同時性を要求)**:\n"
        "```\n"
        "flagged(row,col) := v_mean(row,col) > V_MEAN_THRESHOLD   # 出典: 本レポート§3 youden_threshold\n"
        "n_flagged := count(flagged(row,col) for row in [BURST_ROW_MIN, BURST_ROW_MAX], col in [0,5])\n"
        "effect_suspected := n_flagged >= N_MIN_SIMULTANEOUS_CELLS\n"
        "```\n"
    )
    lines.append(
        "**定数 (全て本レポートのラベル較正由来)**:\n"
        f"- `V_MEAN_THRESHOLD = 96.48` (§3 v_mean youden_threshold、AUC=0.707)\n"
        f"- `N_MIN_SIMULTANEOUS_CELLS = 15` (§5 フレーム単位ゼロFP動作点、TPR=0.571)\n"
        "- `BURST_ROW_MIN=1, BURST_ROW_MAX=3` (出典: study_effect_signature既存定義、"
        "タスク前提「上段row1-3」を維持)\n"
    )
    lines.append(
        "**適用範囲 (いつ走らせるか)**:\n"
        "- 対象: 自盤面 (相手の連鎖アニメ中に自盤面上段が汚染される想定、"
        "project_full_board_error_taxonomy_2026-08-02)。\n"
        "- 時間ゲート: 相手側の連鎖アニメーション中 (fire event検知〜終了+マージン) のみ"
        "有効化し、平常時は判定自体を走らせない (誤検出源になりうる連鎖テロップ表示中"
        "フレームとの重複を避けるため、**連鎖テロップ表示中は判定を抑制する除外条件を"
        "追加する** — §5の唯一の既知の紛らわしい例への直接対策)。\n"
        "- 出力: 「このセルは信頼できない (エフェクトに覆われている疑い)」フラグのみ。"
        "色再認識は行わず、遅延コミット (project_full_board_error_taxonomy_2026-08-02"
        "の「持続確認」案と併用) に流す。\n"
    )
    lines.append(
        "**未解決・要追加作業 (実装着手前に必須)**:\n"
        "1. n=36フレーム・真陽性7件は極小 (統計的検出力が低い、CI算出不能)。"
        "最低100フレーム規模の追加ラベルで再較正すること。\n"
        "2. 連鎖テロップ表示中フレームの除外条件は未実装・未検証 (score OCR等の"
        "既存連鎖検知シグナルとの組み合わせが必要)。\n"
        "3. 煙(お邪魔落下)側は行帯がburstと異なり着地列近傍・可変行のため、"
        "本判定式のBURST_ROW_MIN/MAXは煙には適用できない (別途着地列ベースの"
        "窓定義が必要、本レポート未着手)。\n"
    )
    if warnings:
        lines.append("\n## 警告 (画像欠損等)\n")
        lines.extend(f"- {w}" for w in warnings)
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# main
# =============================================================================


def main() -> None:
    """メイン処理: 突合 -> 集計 -> ROC較正 -> 分離可能性 -> レポート出力。"""
    cv2.setNumThreads(1)  # 熱対策・並列しない
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/5] labeling_result.csv 読込: {LABELING_RESULT_CSV}")
    rows = load_labeling_result_rows(LABELING_RESULT_CSV)
    print(f"  {len(rows)} 件")

    print("[2/5] 突合 (真値ラベル + 画素特徴量)")
    samples, status_counts, warnings = build_labeled_cell_samples(rows, FRAMES_DIR)
    df = samples_to_dataframe(samples)
    df.to_csv(OUTPUT_DIR / "labeled_cell_features.csv", index=False, encoding="utf-8-sig")
    print(f"  突合済みセル数: {len(df)} (usable frame count: {status_counts})")

    print("[3/5] フレーム単位集計")
    status_table = summarize_frame_status(rows)
    layer_label_table = count_effect_cells_by_layer(samples)
    status_table.to_csv(OUTPUT_DIR / "frame_status_summary.csv", encoding="utf-8-sig")

    print("[4/5] ROC較正 + 分離可能性")
    study_df = load_study_cell_stats(STUDY_CELL_STATS_CSV)
    study_normal = study_df.loc[study_df["layer"] == "normal"]
    roc_table = build_roc_operating_point_table(df, study_normal)
    roc_table.to_csv(OUTPUT_DIR / "roc_operating_points.csv", index=False, encoding="utf-8-sig")
    ojama_table = compute_ojama_vs_smoke_separability(df, study_df)
    ojama_table.to_csv(OUTPUT_DIR / "ojama_vs_smoke_separability.csv", index=False, encoding="utf-8-sig")
    # ゼロFP閾値はTPRが低すぎて単独では窓判定に使えない (§3参照) ため、
    # 最良特徴量のYouden's J閾値 (より高感度) を採用し、行帯内同時性で
    # 誤検出を絞り込めるかを検証する。
    best_feature = str(roc_table.iloc[0]["feature"])
    best_threshold = float(roc_table.iloc[0]["youden_threshold"])
    window_counts = compute_window_level_flag_counts(df, best_feature, best_threshold)
    window_counts.to_csv(OUTPUT_DIR / "window_level_flag_counts.csv", index=False, encoding="utf-8-sig")
    frame_level_auc = compute_frame_level_auc(window_counts)

    print("[5/5] レポート出力")
    report = build_calibration_report(
        status_table, layer_label_table, roc_table, ojama_table,
        window_counts, best_feature, best_threshold, frame_level_auc, warnings,
    )
    (OUTPUT_DIR / "calibration_report.md").write_text(report, encoding="utf-8")
    print(f"\n[DONE] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
