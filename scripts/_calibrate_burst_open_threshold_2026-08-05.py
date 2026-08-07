"""BURST_GATE_OPEN_THRESHOLD 緊急再較正 (2026-08-05)。

バーストガードv2 中間結果「31盤面ペアで改善ゼロ・row0に新規誤り3セル」の
主因調査。c5計装実測でバースト中の bright_ratio_max が 0.78〜0.90 に留まり
既存閾値 0.97 (zero-FP点、telop_negative枠を含む無条件ROC由来) が実際の
バースト強度より厳しすぎて窓がほとんど開いていない疑いを検証する。
設計書 `docs/BURST_GUARD_DESIGN_2026-08-05.md` Stage2 で予定していた
`BURST_GATE_CLOSE_THRESHOLD` 較正 (§8 Stage2-1) を前倒しする。

## 過学習防止 (最重要、`feedback_overfitting_awareness_2026-08-04` 準拠)
**閾値の選定には較正セット (v1+v3統合136枚、`labeled_cell_features_v3.csv`
104行) のみを使う。93セルonset (`build_error_onset_sheet_2026-08-04.py`) は
検証専用であり、閾値選定 (ROC計算・動作点決定) には一切使わない。** 93セル
onsetは「較正セットから決めた閾値が実運用でどの程度カバレッジを持つか」を
測る事後検証としてのみ使用し、この分離を本ファイル全体で維持する。

## 条件付きROC (`is_self_chain_scenario` 相当の除外)
v2実装では `own_chain_active` による force_close が「観測対象側自身が
連鎖中」の経路を既に塞いでいる (`docs/BURST_GUARD_DESIGN_2026-08-05.md` §2.3)。
そのため視覚閾値がこの経路の負例まで弾く必要はなく、条件付きROC (この負例を
除外したROC) の方が Stage1 の実際の運用条件に整合する。
`labeled_cell_features_v3.csv` の `layer=="telop_negative"` 行 (8件、全て
`is_true_burst=False`) が `is_self_chain_scenario=True` 相当であることは
`calibration_report_v3.md` §4 (`out_of_scope_misfire.csv` の
`is_self_chain_scenario` 列) で既に確認済みの対応関係を再利用する
(較正データを再ラベルせず、既存の layer 列で機械的に判定できる)。

## 既存資産の再利用 (コピペ禁止指示への対応)
- `scripts/calibrate_effect_detector_v3.py::max_tpr_at_zero_fp` /
  `best_youden_threshold` (ROC動作点計算、通常import・ハイフン無しファイル名)
- `scripts/build_error_onset_sheet_2026-08-04.py::diagnose_all_samples`
  (93セルonset取得、importlib動的import)
- `scripts/_diag_c_zero_effect_2026-08-04.py::_read_frame_at` /
  `scripts/measure_effect_gate_c_2026-08-04.py::_lookup_anchor_row` /
  `_find_by_frame_idx_exact` / `_find_bit_exact_match` (row0調査用)
- `src/effect_glow_detector.py::compute_effect_glow_score` (Stage0で抽出済み
  のスコア関数、本番と完全同一ロジック)

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._calibrate_burst_open_threshold_2026-08-05
"""
from __future__ import annotations

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

from src.board_state_machine import EFFECT_GATE_TOP_ROWS  # noqa: E402
from src.effect_glow_detector import compute_effect_glow_score  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from scripts.calibrate_effect_detector_v3 import (  # noqa: E402
    best_youden_threshold,
    max_tpr_at_zero_fp,
)

# ファイル名にハイフンを含むため動的 import (コピペ禁止指示への対応)。
_SHEET = importlib.import_module("scripts.build_error_onset_sheet_2026-08-04")
_DIAG = importlib.import_module("scripts._diag_c_zero_effect_2026-08-04")
_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

CALIBRATION_CSV: Path = Path(
    "data/verify/effect_detector_calibration_v3_2026-08-04/labeled_cell_features_v3.csv"
)
LAYER_TELOP_NEGATIVE: str = "telop_negative"  # is_self_chain_scenario 相当
SCORE_COLUMN: str = "bright_ratio_max"

# FPR目標値 (窓トリガー用途、FPコストが低いため 5% を許容予算とする)。
TARGET_FPR: float = 0.05

# row0副作用調査対象 (coordinator指定の3セル)。
V2_LANDED_DIR: Path = Path("data/verify/burst_guard_2026-08-05/on_v2")
ROW0_SUSPECTS: "tuple[tuple[str, str, float, int], ...]" = (
    ("c13", "2P", 2524.2, 1),
    ("c13", "2P", 2524.2, 5),
    ("c15", "1P", 760.3, 1),
)
# row0異常調査時の burst score 実測窓 (onset近傍で高スコアが無かったかを
# 広めに確認するためのオフセット、秒)。
ROW0_PROBE_OFFSETS_SEC: "tuple[float, ...]" = (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0)


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class OperatingPoint:
    """1つの閾値候補の動作点。"""

    name: str  # "zero_fp" / "fpr5" / "youden"
    threshold: float
    tpr: float
    fpr: float


# =============================================================================
# 1. 較正セット読込 (93セルonsetは一切参照しない)
# =============================================================================


def load_calibration_frame() -> "pd.DataFrame":
    """labeled_cell_features_v3.csv を読み込む (較正専用データ、選定にのみ使用)。"""
    return pd.read_csv(CALIBRATION_CSV, encoding="utf-8-sig")


def split_pos_neg(
    df: "pd.DataFrame", exclude_self_chain: bool,
) -> "tuple[np.ndarray, np.ndarray]":
    """is_true_burst で pos/neg に分割する。exclude_self_chain=True なら
    layer=="telop_negative" (is_self_chain_scenario 相当) を負例から除外する
    (条件付きROC、docstring 差分参照)。
    """
    work = df
    if exclude_self_chain:
        work = df[df["layer"] != LAYER_TELOP_NEGATIVE]
    pos = work.loc[work["is_true_burst"], SCORE_COLUMN].to_numpy()
    neg = work.loc[~work["is_true_burst"], SCORE_COLUMN].to_numpy()
    return pos, neg


# =============================================================================
# 2. ROC動作点計算 (既存資産再利用 + FPR目標点は新規)
# =============================================================================


def operating_point_at_target_fpr(
    pos: "np.ndarray", neg: "np.ndarray", target_fpr: float,
) -> "dict":
    """FPR<=target_fpr を満たす最大TPRの動作点のうち、最も閾値が低い (=FPR
    予算を使い切る) ものを返す (新規、既存資産に無い唯一の計算)。

    2026-08-05: 較正セットの真陽性がbimodal (0.78-0.97が空隙) なため、
    単純な argmax(tpr) では tie-break で「予算内で最初に見つかる最高閾値」
    を返してしまい、FPR予算を活用しきれない。同じ最大TPRの候補群からは
    **最も低い閾値** (=空隙内でFPR予算を最大限使う点) を選ぶ。
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    x = np.concatenate([pos, neg])
    auc = float(roc_auc_score(y, x)) if len(set(y.tolist())) > 1 else 0.5
    direction = 1.0 if auc >= 0.5 else -1.0
    score = direction * x
    # drop_intermediate=False: 既定Trueだと「TPR不変・FPRのみ増加」の水平
    # 区間の中間点が凸包表示のために間引かれ、bimodal空隙内のFPR予算消化点が
    # 消えてしまう (2026-08-05 発見・修正、本関数のみの既知差異)。
    fpr, tpr, thr = roc_curve(y, score, drop_intermediate=False)
    ok = fpr <= target_fpr
    if not ok.any():
        return {"threshold": float("nan"), "tpr": float("nan"), "fpr": float("nan")}
    max_tpr = tpr[ok].max()
    tie_idx = np.where(ok & (tpr == max_tpr))[0]
    best_i = int(tie_idx[np.argmin(thr[tie_idx])])
    threshold = thr[best_i] if direction >= 0 else -thr[best_i]
    return {"threshold": float(threshold), "tpr": float(tpr[best_i]), "fpr": float(fpr[best_i])}


def build_operating_points(pos: "np.ndarray", neg: "np.ndarray") -> list[OperatingPoint]:
    """zero-FP / FPR5% / Youden の3候補を計算する。"""
    zero_fp = max_tpr_at_zero_fp(pos, neg)
    fpr5 = operating_point_at_target_fpr(pos, neg, TARGET_FPR)
    youden = best_youden_threshold(pos, neg)
    return [
        OperatingPoint("zero_fp", zero_fp["threshold"], zero_fp["tpr_at_zero_fp"], 0.0),
        OperatingPoint("fpr5", fpr5["threshold"], fpr5["tpr"], fpr5["fpr"]),
        OperatingPoint("youden", youden["threshold"], youden["tpr"], youden["fpr"]),
    ]


def build_roc_comparison_report(df: "pd.DataFrame") -> "tuple[str, list[OperatingPoint]]":
    """無条件/条件付きROCの比較表 + 条件付きROCの3候補を返す。"""
    pos_u, neg_u = split_pos_neg(df, exclude_self_chain=False)
    pos_c, neg_c = split_pos_neg(df, exclude_self_chain=True)
    from sklearn.metrics import roc_auc_score
    y_u = np.concatenate([np.ones(len(pos_u)), np.zeros(len(neg_u))])
    x_u = np.concatenate([pos_u, neg_u])
    y_c = np.concatenate([np.ones(len(pos_c)), np.zeros(len(neg_c))])
    x_c = np.concatenate([pos_c, neg_c])
    auc_u = float(roc_auc_score(y_u, x_u))
    auc_c = float(roc_auc_score(y_c, x_c))
    points_u = build_operating_points(pos_u, neg_u)
    points_c = build_operating_points(pos_c, neg_c)
    lines = [
        "--- 無条件ROC vs 条件付きROC (self_chain負例除外) ---",
        f"無条件: n_pos={len(pos_u)} n_neg={len(neg_u)} AUC={auc_u:.3f}",
        f"条件付き: n_pos={len(pos_c)} n_neg={len(neg_c)} AUC={auc_c:.3f} "
        f"(除外負例={len(neg_u) - len(neg_c)}件)",
        "",
        "動作点  | 無条件(閾値/TPR/FPR)          | 条件付き(閾値/TPR/FPR)",
    ]
    for pu, pc in zip(points_u, points_c):
        lines.append(
            f"{pu.name:8}| {pu.threshold:.3f}/{pu.tpr:.3f}/{pu.fpr:.3f}"
            f"          | {pc.threshold:.3f}/{pc.tpr:.3f}/{pc.fpr:.3f}"
        )
    return "\n".join(lines), points_c


def recommend_threshold(points: list[OperatingPoint]) -> "tuple[OperatingPoint, str]":
    """3候補のうち1つを推奨する (較正セットの真陽性分布の形を踏まえる)。

    重要な発見: 較正セットの真陽性はbimodal (強いバースト≒0.97-1.0 と、
    弱いバースト0.48-0.77 の2群、その間0.78-0.97は真陽性が1件も無い空隙)。
    この空隙内はどの閾値を選んでもTPRは不変 (0.647固定) で、FPRのみ単調に
    増える。zero_fp/youdenは「TPR向上が無いのにFPRだけ増やす」ことを避ける
    ため空隙の最上端 (≒0.97) に留まり、これは各基準として正しい振る舞い。
    一方 fpr5 は「予算(5%)内でFPRを使い切る」設計にしたため、この空隙の
    下端付近まで下げる (=較正セットのTPRを一切犠牲にせず、より広いスコア帯
    を窓トリガー対象にできる、フィールドデータへの逆算ではなく較正セット
    自身のFPR予算のみで導出、過学習ではない)。
    判定基準: fpr5 が youden より低い閾値かつ FPR>0 (=予算を実際に使えている)
    なら bimodal空隙が存在する証拠とみなし fpr5 を推奨する。空隙が無ければ
    fpr5もyoudenと同じ点に収束する (FPR>0の余地が無いため)。
    """
    by_name = {p.name: p for p in points}
    youden = by_name["youden"]
    fpr5 = by_name["fpr5"]
    if fpr5.threshold < youden.threshold and fpr5.fpr > 0.0:
        return fpr5, (
            f"fpr5点 (閾値={fpr5.threshold:.3f}, TPR={fpr5.tpr:.3f}, "
            f"FPR={fpr5.fpr:.3f}) を推奨。較正セットの真陽性が bimodal で "
            f"{fpr5.threshold:.3f}〜{youden.threshold:.3f} の間に真陽性が"
            "存在しないため、この帯でTPRを犠牲にせずFPR予算(5%)まで閾値を"
            "下げられる (窓トリガーのFPコストが低いこととも整合)。"
        )
    return youden, (
        f"youden点 (閾値={youden.threshold:.3f}, TPR={youden.tpr:.3f}, "
        f"FPR={youden.fpr:.3f}) を推奨 (fpr5との有意差なし、bimodal空隙なし)。"
    )


# =============================================================================
# 3. 93セルonset カバレッジ検証 (選定には使わない、事後検証のみ)
# =============================================================================


def _burst_layer_onset_records() -> list["object"]:
    """93セルonsetのうち burst layer (row1-3) のみを抜き出す (70件想定)。

    smoke layer (row4-12、23セル) はStage1のスコープ外であり本検証では
    対象外として明示的に除外する (coordinator指示、Stage2の担当)。
    """
    records = _SHEET.diagnose_all_samples()
    return [r for r in records if r.row in EFFECT_GATE_TOP_ROWS]


def _measure_onset_burst_score(
    r: "object", cap_cache: dict, fps_cache: dict,
) -> "float | None":
    """1セル分の onset_t / onset+0.2秒 の2点で compute_effect_glow_score の最大値を測る。"""
    if r.onset_t_sec is None:
        return None
    region = _SHEET._region_for_side(r.side)
    if r.video not in cap_cache:
        video_path = _DIAG.VIDEO_DIR / f"video_{r.video}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        cap_cache[r.video] = cap
        fps_cache[r.video] = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap = cap_cache[r.video]
    fps = fps_cache[r.video]
    best = 0.0
    found = False
    for t in (r.onset_t_sec, r.onset_t_sec + _SHEET.CAPTURE_POST_ONSET_SEC):
        frame = _DIAG._read_frame_at(cap, fps, max(t, 0.0))
        if frame is None:
            continue
        found = True
        score = compute_effect_glow_score(frame, region, EFFECT_GATE_TOP_ROWS)
        best = max(best, score)
    return best if found else None


def measure_93cell_coverage(
    candidates: list[OperatingPoint],
) -> "tuple[str, list[float]]":
    """burst layer 70セルについて onset時の実測スコアと各候補のカバレッジを計算する。

    93セルonsetは検証専用であり、この結果は閾値選定には使わない
    (docstring 冒頭「過学習防止」参照)。
    """
    burst_records = _burst_layer_onset_records()
    cap_cache: dict = {}
    fps_cache: dict = {}
    scores: list[float] = []
    n_no_frame = 0
    for r in burst_records:
        s = _measure_onset_burst_score(r, cap_cache, fps_cache)
        if s is None:
            n_no_frame += 1
            continue
        scores.append(s)
    for cap in cap_cache.values():
        cap.release()
    lines = [
        f"--- 93セルonset検証 (burst layer {len(burst_records)}セル中 "
        f"実測可能{len(scores)}件、smoke layer 23セルは対象外) ---",
    ]
    for cand in candidates:
        covered = sum(1 for s in scores if s > cand.threshold)
        pct = (covered / len(scores) * 100) if scores else 0.0
        lines.append(
            f"  {cand.name} (閾値={cand.threshold:.3f}): "
            f"カバレッジ={covered}/{len(scores)} ({pct:.1f}%)"
        )
    if n_no_frame:
        lines.append(f"  [注] フレーム取得不能で測定不可: {n_no_frame} 件")
    lines.append(_score_distribution_summary(scores))
    return "\n".join(lines), scores


def _score_distribution_summary(scores: list[float]) -> str:
    """未カバー帯の分布 (c5実測0.78-0.90帯が実際に多いかを確認する)。"""
    if not scores:
        return "  分布: データなし"
    arr = np.array(scores)
    n_c5_band = int(np.sum((arr >= 0.78) & (arr < 0.90)))
    return (
        f"  スコア分布: min={arr.min():.3f} p25={np.percentile(arr,25):.3f} "
        f"median={np.median(arr):.3f} p75={np.percentile(arr,75):.3f} "
        f"max={arr.max():.3f} / c5実測帯[0.78,0.90)に該当: {n_c5_band}/{len(scores)}件"
    )


# =============================================================================
# 4. row0 副作用の切り分け (悪化3セルが「窓が開いた副作用」か「無関係な揺らぎ」か)
# =============================================================================


def _find_sample_for_suspect(video: str, side: str, t_sec: float) -> "object | None":
    """(video, side, t_sec) に一致する LabelSample を返す (fixed/ok問わず)。"""
    for s in _MC.load_all_samples():
        if s.video_stem == video and s.side == side and abs(s.t_sec - t_sec) < 1e-3:
            return s
    return None


def _probe_burst_scores_near(
    video: str, side: str, center_t_sec: float,
) -> "list[tuple[float, float]]":
    """center_t_sec 近傍の複数オフセットで compute_effect_glow_score を実測する。"""
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    video_path = _DIAG.VIDEO_DIR / f"video_{video}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out: list[tuple[float, float]] = []
    for offset in ROW0_PROBE_OFFSETS_SEC:
        t = max(center_t_sec + offset, 0.0)
        frame = _DIAG._read_frame_at(cap, fps, t)
        if frame is None:
            continue
        score = compute_effect_glow_score(frame, region, EFFECT_GATE_TOP_ROWS)
        out.append((offset, score))
    cap.release()
    return out


def investigate_row0_suspect(video: str, side: str, t_sec: float, col: int) -> str:
    """1件の row0 副作用疑いセルを調査する (anchor vs v2 の差分 + burst score近傍実測)。"""
    sample = _find_sample_for_suspect(video, side, t_sec)
    if sample is None:
        return f"  {video} {side} t={t_sec} row0col{col}: [警告] ラベルサンプル特定失敗"
    anchor_idx = _MC._load_npz_index(_MC.ANCHOR_NPZ_DIR / f"{video}.npz")
    anchor = _MC._lookup_anchor_row(
        anchor_idx, sample.side, sample.t_sec, sample.game_idx, sample.anchor_recognized_grid,
    )
    if anchor is None:
        return f"  {video} {side} t={t_sec} row0col{col}: [警告] アンカー突合失敗"
    v2_path = V2_LANDED_DIR / f"{video}.npz"
    if not v2_path.exists():
        return f"  {video} {side} t={t_sec} row0col{col}: (v2) 未着弾"
    v2_idx = _MC._load_npz_index(v2_path)
    match = _MC._find_by_frame_idx_exact(v2_idx, side, anchor.frame_idx)
    if match is None:
        match = _MC._find_bit_exact_match(
            v2_idx, side, anchor.grid, anchor.t_sec, _MC.ANCHOR_MATCH_WINDOW_SEC,
        )
    if match is None:
        return f"  {video} {side} t={t_sec} row0col{col}: (v2) no_match"
    v2_grid, _v2_t = match
    off_val = int(anchor.grid[0, col])
    v2_val = int(v2_grid[0, col])
    scores = _probe_burst_scores_near(video, side, anchor.t_sec)
    max_score = max((s for _o, s in scores), default=0.0)
    return (
        f"  {video} {side} t={t_sec} row0col{col}: OFF={off_val} → (v2)={v2_val} "
        f"(差分={'あり' if off_val != v2_val else 'なし'}) / "
        f"近傍burst_score最大={max_score:.3f} (詳細={scores})"
    )


def build_row0_report() -> str:
    """row0副作用3セルの調査レポート。"""
    lines = ["--- row0副作用の切り分け (悪化3セル) ---"]
    for video, side, t_sec, col in ROW0_SUSPECTS:
        lines.append(investigate_row0_suspect(video, side, t_sec, col))
    return "\n".join(lines)


# =============================================================================
# 5. main
# =============================================================================


def main() -> None:
    cv2.setNumThreads(1)
    df = load_calibration_frame()
    print(f"[1/4] 較正セット読込: {len(df)} 行 (v1+v3統合、93セルonsetは未参照)")

    roc_report, candidates = build_roc_comparison_report(df)
    print("\n[2/4] " + roc_report)

    recommended, reason = recommend_threshold(candidates)
    print(f"\n[推奨] {reason}")

    print("\n[3/4] 93セルonset カバレッジ検証 (選定には使わない事後検証)")
    coverage_report, _scores = measure_93cell_coverage(candidates)
    print(coverage_report)

    print("\n[4/4] " + build_row0_report())


if __name__ == "__main__":
    main()
