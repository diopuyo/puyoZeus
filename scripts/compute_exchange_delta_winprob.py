"""#24 打ち合い計測器 ΔWinProb接続 (アーキ設計 案C: 仮想盤面2回評価) Step3。

## 背景
Step1 (src/exchange_predictor.py) が案D単体モデルで発火イベントの正味おじゃま
予測を出し、Step2 (src/exchange_virtual_board.py) がその予測から「発火後の
仮想盤面ペア」を再構成した。本 Step3 は仮想盤面ペアの前後で「45指標→勝率」
モデルを2回評価し、ΔWinProb (-100〜+100) を出す最終接続部分。

## 設計方針 (CLAUDE.md 準拠)
- **既存資産の再利用・再実装禁止**: 盤面再構成は `src.exchange_virtual_board`
  (Step2)、発火イベント検出・盤面復元は `scripts.label_exchange_outcome` の
  内部関数、併用スタッキング予測は `scripts.run_exchange_triple_comparison`
  をそのまま import して使う (コピペ再実装しない)。
- **stateless**: 指標計算関数はすべて `board: Board` を受け取る純関数呼び出し。

## スコープ決定 (正直な注記、fail-silent回避のため明記)
「45指標→勝率」モデルは本来 `data/verify/win_eval_combined66_2026-07-29/
labeled_win_combined66.csv` の全指標 (96列) を使うが、その多くは
next/dnext ツモ・経過時間・手数等 **盤面グリッド以外の文脈** を必要とする
(near_future_fire_k*, expected_fire_k*, tsumo_count_rate 等)。Step2 が返す
「仮想盤面」はグリッドのみの純粋シミュレーション結果でありこれらの文脈を
持たない (次ツモが未知の仮想将来盤面のため next 情報は原理的に存在しない)。
そのため本スクリプトは **盤面グリッドのみから計算可能な指標
(BOARD_ONLY_INDICATOR_BASES, 23種)** に限定したLRを学習し、それを
「45指標→勝率」モデルの代わりに使う。実際の値/前後比較は一貫してこの
限定モデルで行うため比較は公平 (STABLE盤面のタイムラインも同じモデルで
評価する)。将来 next 情報を仮想盤面に付与できるようになれば拡張可能。

## 使い方 (本走行)
    PYTHONPATH=. python -m scripts.compute_exchange_delta_winprob \\
        --aug-csv data/indicators_v2/exchange_labels_regen_step3_aug_2026-08-02.csv \\
        --model-d-dir data/verify/exchange_model_d_step3_2026-08-02 \\
        --npz-dir data/indicators_v2/boards_lean_regen_2026-07-31 \\
        --labeled-win-csv data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv \\
        --out-dir data/verify/exchange_delta_winprob_step3_2026-08-02

## 動作確認 (--limit、軽量チェック用)
    PYTHONPATH=. python -m scripts.compute_exchange_delta_winprob --limit 300
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.board import Board
from src.chain import ChainSimulator
import src.indicators_v2 as iv
from src.exchange_virtual_board import reconstruct_virtual_board_pair
from src.ojama_accounting import cancel_own_pending_then_send_surplus
from scripts.label_exchange_outcome import (
    NpzRecord,
    _board_from_grid,
    _detect_fire_events,
    _load_npz,
    _merge_fire_event_clusters,
)
from scripts.measure_exchange_effectiveness import estimate_available_hands
from scripts.model_indicator_win import (
    LR_PARAMS,
    build_features,
    load_labeled_csv,
    pair_sides_for_win,
)
from scripts.run_exchange_triple_comparison import (
    align_aug_with_model_d,
    build_stacking_oof_predictions,
    filter_nan_sim_rows,
    load_model_d_oof,
)
from scripts.train_exchange_model_d import load_exchange_labels
from scripts.visualize_advantage_overlay import board_room

# =============================================================================
# 定数定義
# =============================================================================

DEFAULT_AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_step3_aug_2026-08-02.csv")
DEFAULT_MODEL_D_DIR = Path("data/verify/exchange_model_d_step3_2026-08-02")
DEFAULT_NPZ_DIR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
DEFAULT_LABELED_WIN_CSV = Path("data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv")
DEFAULT_OUT_DIR = Path("data/verify/exchange_delta_winprob_step3_2026-08-02")

# デモviz対象動画 (c34/c11/c21/c28/c23/c27/c5/Step0のc11/c21/c28 は使用済みのため
# それ以外を選定、2026-08-02時点で未使用と確認済み)
DEFAULT_DEMO_VIDEO_ID = "c50"

# 位相ラベル一覧 (label_exchange_outcome.py の _classify_phase と同じ表記)
PHASES: tuple[str, ...] = ("序", "中", "終")

# 勝率 (確率0〜1) を 0〜100 (%) に変換する尺度。
#
# 2026-08-02 修正 (main検品指摘): 旧実装は (p-0.5)*200 で「-100(確実敗北)
# 〜+100(確実勝利)」の対称アドバンテージスコアにしていたが、delta_winprob
# = after - before を取るとこの±100範囲を持つ値同士の差になり理論上
# ±200まで発生してしまう (実測 max=190.10 で確認、いわば200%スケールの
# 二重計上バグ)。delta_winprob の定義は「発火側から見た勝率(0-100%)の
# 前→後 差分」であるべき (これなら理論上も±100を超えない) ため、
# winprob_before/after 自体を「0〜100%の勝率」表現に変更する
# (p=0 -> 0%, p=1 -> 100%)。
WINPROB_PERCENT_SCALE: float = 100.0

# 1P/2P ペアリングの最大時刻差 (秒、model_indicator_win.DEFAULT_MAX_TDIFF と同値)
PAIR_MAX_TDIFF_SEC: float = 1.0

# 校正器 (isotonic) を学習可能とみなす inner-OOF 最小件数
# (_calibration_fit_2026-07-29.py の MIN_CALIB_FIT_N と同じ思想の安全弁)
MIN_ISOTONIC_FIT_N: int = 200

# 位相別モデルを学習可能とみなす最小サンプル数
MIN_PHASE_TRAIN_N: int = 30

# 位相別LR学習の内側 GroupKFold 分割数 (isotonic 用 OOF 生成)
N_INNER_FOLDS: int = 5

# npz 内 t_sec とラベルCSVの t_sec 突合許容誤差 (float32往復の丸め誤差を吸収)
T_SEC_MATCH_TOL_SEC: float = 0.02

# 健全性チェック: |ΔWinProb| がこれを超えたら「大きな変動」とみなす
LARGE_DELTA_ABS_THRESHOLD: float = 20.0

# 健全性チェック: 上位/下位何件を個別表示するか
N_SANITY_TOP_ROWS: int = 5

# aug CSV の video_id 列は "video_c10" 形式だが npz ファイル名は "c10.npz"
# (prefix無し) のため、ファイル探索時のみ剥がす (label_exchange_outcome.py
# 側の _load_npz が返す NpzRecord.video_id は "video_c10" のまま扱う点に注意)。
VIDEO_ID_NPZ_PREFIX: str = "video_"

# 並列ワーカー数 (既定 = 論理コア数と8の小さい方、feedback_parallelize_evals_by_default)
DEFAULT_N_WORKERS: int = min(8, mp.cpu_count())

# デモviz: 折れ線の見た目調整
DEMO_FIG_SIZE: tuple[float, float] = (16.0, 6.0)
DEMO_FIRE_LINE_ALPHA: float = 0.35


# =============================================================================
# 1. 盤面グリッドのみで計算可能な指標 (board-only) の一覧
# =============================================================================
#
# 45指標のうち next/dnext/経過時間/手数を必要としない (=仮想盤面にも適用できる)
# サブセット。src.indicators_v2 の各関数シグネチャを確認し `board` (+ 任意で
# `simulator`) のみで呼べるものだけを採用 (absorption_capacity は
# scripts/model_indicator_win.py REDUNDANT_COLS で board_puyo_total と完全重複と
# 既知のため除外、既存の除外方針を踏襲)。

_BOARD_ONLY_FUNCS: dict[str, Callable[[Board, ChainSimulator], "iv.IndicatorV2Value"]] = {
    "board_puyo_total": lambda b, s: iv.board_puyo_total(b),
    "board_color_puyo_total": lambda b, s: iv.board_color_puyo_total(b),
    "max_column_height": lambda b, s: iv.max_column_height(b),
    "column_bumpiness": lambda b, s: iv.column_bumpiness(b),
    "death_margin": lambda b, s: iv.death_margin(b),
    "death_margin_neighbor": lambda b, s: iv.death_margin_neighbor(b),
    "board_ojama_count": lambda b, s: iv.board_ojama_count(b),
    "current_max_chain": lambda b, s: iv.current_max_chain(b, simulator=s),
    "immediate_fire_power": lambda b, s: iv.immediate_fire_power(b, simulator=s),
    "chain_efficiency": lambda b, s: iv.chain_efficiency(b, simulator=s),
    "min_puyos_to_ignite": lambda b, s: iv.min_puyos_to_ignite(b, simulator=s),
    "second_chain_potential": lambda b, s: iv.second_chain_potential(b, simulator=s),
    "dig_resistance": lambda b, s: iv.dig_resistance(b, simulator=s),
    "ojama_disruption": lambda b, s: iv.ojama_disruption(b, simulator=s),
    "main_linked_pair_count": lambda b, s: iv.main_linked_pair_count(b, simulator=s),
    "isolated_pair_count": lambda b, s: iv.isolated_pair_count(b, simulator=s),
    "main_linked_ratio": lambda b, s: iv.main_linked_ratio(b, simulator=s),
    "ukeyasusa": lambda b, s: iv.ukeyasusa(b, simulator=s),
    "saturated_chain_count": lambda b, s: iv.saturated_chain_count(b, simulator=s),
    "ignition_point_count": lambda b, s: iv.ignition_point_count(b, simulator=s),
    "multi_color_ignition": lambda b, s: iv.multi_color_ignition(b, simulator=s),
    "sub_chain_count": lambda b, s: iv.sub_chain_count(b, simulator=s),
    "simultaneous_pop_richness": lambda b, s: iv.simultaneous_pop_richness(b, simulator=s),
}
BOARD_ONLY_INDICATOR_BASES: tuple[str, ...] = tuple(_BOARD_ONLY_FUNCS.keys())


def compute_board_only_features(board: Board, simulator: ChainSimulator) -> dict[str, float]:
    """盤面グリッドのみから計算可能な23指標のスコア (0〜1) を返す。

    実盤面・仮想盤面のどちらにも同一ロジックで適用できる (次ツモ等の文脈を
    要求する指標は含まない、モジュール冒頭のスコープ決定を参照)。
    """
    return {name: fn(board, simulator).score for name, fn in _BOARD_ONLY_FUNCS.items()}


# =============================================================================
# 2. 位相別 勝率モデル (LR + 入れ子 isotonic 校正)
# =============================================================================

@dataclass(frozen=True)
class PhaseWinprobModel:
    """1位相分の勝率モデル (標準化器 + LR + isotonic校正器)。

    isotonic が None の場合は inner-OOF 不足で校正をスキップしたことを示す
    (raw 確率をそのまま使う、_calibration_fit_2026-07-29.py と同じ安全弁)。
    """
    scaler: StandardScaler
    lr: LogisticRegression
    isotonic: "IsotonicRegression | None"
    oof_auc: float
    n_train: int


def _nested_fit_one_phase(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> PhaseWinprobModel:
    """1位相分: 全データで最終LRを学習しつつ、内側GroupKFoldのOOFでisotonicを学習する。

    最終LR (=将来の未知盤面を予測する本体) は全データで学習する一方、
    isotonic校正器は「そのLRが学習に使ったのと同じサンプルへの予測」では
    なく、内側 GroupKFold の held-out 予測 (inner-OOF) だけで学習する。
    そのため校正器はどのサンプルについても「そのサンプルを学習に使った
    LR」の予測を直接は見ない (リーク防止、nested構成)。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(**LR_PARAMS)
    lr.fit(X_scaled, y)

    n_groups = len(np.unique(groups))
    eff_folds = min(N_INNER_FOLDS, max(2, n_groups))
    gkf = GroupKFold(n_splits=eff_folds)
    oof = np.full(len(y), np.nan)
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr_idx])) < 2:
            continue
        fold_scaler = StandardScaler().fit(X[tr_idx])
        fold_lr = LogisticRegression(**LR_PARAMS)
        fold_lr.fit(fold_scaler.transform(X[tr_idx]), y[tr_idx])
        oof[te_idx] = fold_lr.predict_proba(fold_scaler.transform(X[te_idx]))[:, 1]

    valid = ~np.isnan(oof)
    oof_auc = (
        float(roc_auc_score(y[valid], oof[valid]))
        if valid.sum() > 0 and len(np.unique(y[valid])) > 1
        else float("nan")
    )
    isotonic = None
    if valid.sum() >= MIN_ISOTONIC_FIT_N and len(np.unique(y[valid])) > 1:
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        isotonic.fit(oof[valid], y[valid])
    return PhaseWinprobModel(scaler=scaler, lr=lr, isotonic=isotonic,
                              oof_auc=oof_auc, n_train=len(y))


def _assign_phase_by_puyo_tertile(phase_metric: np.ndarray) -> tuple[np.ndarray, float, float]:
    """盤面の埋まり具合を表す指標の3分位で 序/中/終 を割り当てる。

    label_exchange_outcome.py の phase 定義 (盤面ぷよ合計の3分位) と同じ
    「盤面の埋まり具合」の考え方を、labeled_win データセット側で独立に
    再定義したもの (元データが別収集のため分位境界値は一致しない、
    どちらも「盤面ぷよ合計3分位」という同一コンセプトである点は共通)。

    2026-08-03 指摘1対処: 呼び出し側 (train_winprob_models) は 1P+2P の
    合計値 (1P/2P入替に対して不変な対称量) を渡す。1P単独の値を渡すと
    鏡像複製後に元サンプルと鏡像とで異なる位相バケツに分かれてしまい
    (元: 1P値が低いので「序」、鏡像: その値が2P側に回るため「序」と
    判定されない、等)、位相別モデルごとの対称化が崩れて空盤面が
    ちょうど50%にならない実害が out (テストで確認済み)。引数名を
    board_puyo_total_1p から phase_metric に変更 (呼び出し側は位置引数の
    ため後方互換に影響なし)。
    """
    q_low = float(np.quantile(phase_metric, 0.33))
    q_high = float(np.quantile(phase_metric, 0.67))
    labels = np.full(len(phase_metric), "中", dtype=object)
    labels[phase_metric <= q_low] = "序"
    labels[phase_metric > q_high] = "終"
    return labels, q_low, q_high


def _build_mirror_paired(paired: pd.DataFrame) -> pd.DataFrame:
    """1P/2P の `_1p`/`_2p` 列を丸ごと入れ替えた鏡像複製 DataFrame を作る。

    2026-08-03 userレビュー指摘1 (空盤面で1P勝率52%から始まる) 対処。
    671試合の1P勝率52.0% (二項検定 p=0.32、偶然の範囲、userドメイン確認済み
    「1P有利の実在事情は無い」) をモデルが事前確率として学習してしまって
    いたため、学習データを対称化する。`build_features` は列名の `_1p`/`_2p`
    suffix のみを見て `_diff` (=1p-2p) を機械的に再構築するため、全列を
    丸ごと入れ替えるだけで「2P視点から見た対称サンプル」が自動的に手に入る
    (diff の符号反転も build_features 側で自然に起きる、二重実装しない)。
    video_id_1p/2p 等の非指標列も入れ替わるが、呼び出し側は fold 分割に
    元の (非入れ替え) video_id_1p を使うため実害はない。

    数学的根拠 (テストで検証): 鏡像複製込みで学習した L2正則化ロジスティック
    回帰は、1P/2P 入替 + ラベル反転で損失関数が不変になるため、一意最適解は
    この対称変換の不動点になる (w_1p = -w_2p, bias = 0)。この結果、
    1P/2P の指標値が完全に一致する対称局面 (空盤面等) の予測確率は
    厳密に 0.5 になる。
    """
    mirror = paired.copy()
    for col in paired.columns:
        if not col.endswith("_1p"):
            continue
        col_2p = f"{col[:-3]}_2p"
        if col_2p in paired.columns:
            mirror[col] = paired[col_2p].values
            mirror[col_2p] = paired[col].values
    return mirror


def train_winprob_models(labeled_win_csv: Path) -> dict[str, PhaseWinprobModel]:
    """labeled_win CSV から位相別の勝率モデル (LR + isotonic校正) を学習する。

    「45指標→勝率」の代わりに BOARD_ONLY_INDICATOR_BASES (23指標) に限定する
    (モジュール冒頭のスコープ決定を参照)。2026-08-03 指摘1対処: 学習データを
    1P/2P鏡像複製で対称化する (_build_mirror_paired docstring 参照)。
    """
    print(f"[winprob] labeled_win読込: {labeled_win_csv}")
    df = load_labeled_csv(str(labeled_win_csv))
    paired = pair_sides_for_win(df, PAIR_MAX_TDIFF_SEC)
    feat_df = build_features(paired, list(BOARD_ONLY_INDICATOR_BASES))
    X_orig = feat_df.fillna(0.0).values.astype(float)
    y_orig = paired["won_1p"].astype(int).values
    groups_orig = paired["video_id_1p"].values
    # 位相判定量: 1P+2P合計 (1P/2P入替に対して不変な対称量、
    # _assign_phase_by_puyo_tertile docstring の実害説明を参照)。
    phase_metric_orig = (paired["board_puyo_total_1p"].astype(float).values
                         + paired["board_puyo_total_2p"].astype(float).values)

    # 対称化 (指摘1修正): 1P/2P入替鏡像複製を追加する。fold分割は video_id
    # 単位のまま保つため groups は元の video_id を鏡像側にもそのまま使う
    # (鏡像は必ず元サンプルと同じfoldに入る = GroupKFoldはユニークな group
    # 値単位で分割するため自動的に満たされる)。
    paired_mirror = _build_mirror_paired(paired)
    feat_df_mirror = build_features(paired_mirror, list(BOARD_ONLY_INDICATOR_BASES))
    X_mirror = feat_df_mirror.fillna(0.0).values.astype(float)
    y_mirror = 1 - y_orig
    # 合計値は入替で不変なので鏡像でも同じ値 (=元と必ず同じ位相バケツに入る)。
    phase_metric_mirror = phase_metric_orig

    X_all = np.vstack([X_orig, X_mirror])
    y_all = np.concatenate([y_orig, y_mirror])
    groups_all = np.concatenate([groups_orig, groups_orig])
    phase_metric_all = np.concatenate([phase_metric_orig, phase_metric_mirror])
    print(f"[winprob] 対称化 (指摘1対処): {len(y_orig)}行 -> {len(y_all)}行"
          f" (1P勝率 元={y_orig.mean():.3f} 対称化後={y_all.mean():.3f})")

    phase_labels, q_low, q_high = _assign_phase_by_puyo_tertile(phase_metric_all)
    print(f"[winprob] 位相境界 (board_puyo_total 1P+2P合計): 序<={q_low:.3f} 終>{q_high:.3f}")

    models: dict[str, PhaseWinprobModel] = {}
    for phase in PHASES:
        mask = phase_labels == phase
        n = int(mask.sum())
        if n < MIN_PHASE_TRAIN_N:
            print(f"[winprob] {phase}: データ不足 (n={n}) -> skip")
            continue
        model = _nested_fit_one_phase(X_all[mask], y_all[mask], groups_all[mask])
        calib_state = "isotonic校正あり" if model.isotonic is not None else "校正スキップ(raw)"
        print(f"[winprob] {phase}: n={n} inner-OOF AUC={model.oof_auc:.3f} ({calib_state})")
        models[phase] = model
    return models


def winprob_attacker(
    models: dict[str, PhaseWinprobModel], phase: str,
    attacker_feats: dict[str, float], opponent_feats: dict[str, float],
) -> float:
    """発火側 (attacker) の勝率 P(attacker win) を校正済みで返す (0〜1)。

    attacker を "1P枠"、opponent を "2P枠" として build_features と同じ
    列順 (base_1p, base_2p, base_diff の3つ組) でベクトルを組む。特徴量に
    座席 (1P/2P) を区別する情報が無いため、どちらの盤面を1P枠に置いても
    「その盤面の持ち主が勝つ確率」を返す対称な関数として使える。
    """
    if phase not in models:
        raise KeyError(f"位相 '{phase}' の勝率モデルが未学習です (学習データ不足)")
    model = models[phase]
    values: list[float] = []
    for base in BOARD_ONLY_INDICATOR_BASES:
        v1, v2 = attacker_feats[base], opponent_feats[base]
        values.extend([v1, v2, v1 - v2])
    x = np.asarray(values, dtype=float).reshape(1, -1)
    raw_p = float(model.lr.predict_proba(model.scaler.transform(x))[0, 1])
    if model.isotonic is not None:
        return float(model.isotonic.predict([raw_p])[0])
    return raw_p


def winprob_to_score100(p: float) -> float:
    """勝率確率 (0〜1) を 0〜100 (%) スケールに変換する (発火側視点の勝率%)。

    この値同士の差 (delta_winprob) は理論上も必ず -100〜+100 に収まる
    (0〜100 の値同士の差のため、2026-08-02 修正)。
    """
    return p * WINPROB_PERCENT_SCALE


# =============================================================================
# 3. npz からの盤面再構成 (label_exchange_outcome.py の内部関数を再利用)
# =============================================================================

@dataclass
class _VideoNpzCache:
    """1動画分の 1P/2P NpzRecord (video_id をキーにしたキャッシュ単位)。"""
    r1p: NpzRecord
    r2p: NpzRecord


def _npz_stem_from_video_id(video_id: str) -> str:
    """aug CSV の video_id ("video_c10") から npz ファイル名の幹 ("c10") を作る。"""
    if video_id.startswith(VIDEO_ID_NPZ_PREFIX):
        return video_id[len(VIDEO_ID_NPZ_PREFIX):]
    return video_id


def _load_video_npz(video_id: str, npz_dir: Path) -> "_VideoNpzCache | None":
    """1動画分の npz を読み込む (存在しない/片側欠損なら None)。"""
    path = npz_dir / f"{_npz_stem_from_video_id(video_id)}.npz"
    if not path.exists():
        return None
    records = _load_npz(path)
    by_side = {r.side: r for r in records}
    if "1P" not in by_side or "2P" not in by_side:
        return None
    return _VideoNpzCache(r1p=by_side["1P"], r2p=by_side["2P"])


def reconstruct_event_board_pair(
    video_cache: _VideoNpzCache, game_idx: int, t_sec: float, fire_side: str,
) -> "tuple[Board, Board] | None":
    """1イベント分の (発火直前攻撃側盤面, 発火直前相手側盤面) を npz から復元する。

    label_exchange_outcome._process_game と同一の検出・クラスタリング関数を
    再利用し、CSV の t_sec に最も近い発火クラスタを突合する (突合失敗は
    None を返し、呼び出し側で件数をログする、silent drop しない)。
    """
    fire_rec, opp_rec = (
        (video_cache.r1p, video_cache.r2p) if fire_side == "1P"
        else (video_cache.r2p, video_cache.r1p)
    )
    game_mask = fire_rec.game_idx == game_idx
    if not game_mask.any():
        return None
    t_g, grids_g, score_g = fire_rec.t_sec[game_mask], fire_rec.grids[game_mask], fire_rec.score[game_mask]
    fire_indices = _detect_fire_events(t_g, score_g)
    clusters = _merge_fire_event_clusters(t_g, score_g, grids_g, fire_indices)
    if not clusters:
        return None
    cluster_t = np.array([t_g[c.fire_index] for c in clusters])
    idx = int(np.argmin(np.abs(cluster_t - t_sec)))
    if abs(float(cluster_t[idx]) - t_sec) > T_SEC_MATCH_TOL_SEC:
        return None
    fire_board = _board_from_grid(grids_g[clusters[idx].board_ref_index])

    opp_mask = opp_rec.game_idx == game_idx
    opp_t, opp_grids = opp_rec.t_sec[opp_mask], opp_rec.grids[opp_mask]
    if len(opp_t) == 0:
        return None
    nearest = int(np.argmin(np.abs(opp_t - t_sec)))
    opp_board = _board_from_grid(opp_grids[nearest])
    return fire_board, opp_board


# =============================================================================
# 4. 1イベント分の ΔWinProb 計算
# =============================================================================

@dataclass(frozen=True)
class DeltaWinProbResult:
    """1発火イベント分の ΔWinProb 計算結果。

    Attributes:
        winprob_before: 発火側 (attacker) 視点の勝率 (0〜100%、発火直前)。
        winprob_after: 発火側視点の勝率 (0〜100%、Step2仮想盤面ペア評価後)。
        delta_winprob: winprob_after - winprob_before (0〜100%同士の差の
            ため理論上も必ず -100〜+100 に収まる)。
        attacker_dead_after: 仮想盤面ペアで攻撃側が窒息判定なら True。
        opponent_dead_after: 仮想盤面ペアで相手側が窒息判定なら True。
    """
    winprob_before: float
    winprob_after: float
    delta_winprob: float
    attacker_dead_after: bool
    opponent_dead_after: bool


def compute_delta_winprob_for_event(
    fire_board: Board, opp_board: Board, phase: str,
    net_ojama_after_pred: float,
    models: dict[str, PhaseWinprobModel], simulator: ChainSimulator,
) -> DeltaWinProbResult:
    """発火直前盤面ペア→(Step2仮想盤面ペア)→発火後、の前後で勝率を評価しΔを返す。

    net_ojama_after_pred には「併用スタッキングモデルの予測値」を渡す想定
    (実運用では正解ラベルが未知のため、Step1/併用モデルの予測を使う設計。
    正解ラベル net_ojama_after をそのまま使うとリーク相当になるため使わない)。
    """
    before_fire_feats = compute_board_only_features(fire_board, simulator)
    before_opp_feats = compute_board_only_features(opp_board, simulator)
    p_before = winprob_attacker(models, phase, before_fire_feats, before_opp_feats)

    vpair = reconstruct_virtual_board_pair(
        fire_board, opp_board, net_ojama_after_pred, simulator=simulator,
    )
    after_fire_feats = compute_board_only_features(vpair.attacker_board_after, simulator)
    after_opp_feats = compute_board_only_features(vpair.opponent_board_after, simulator)
    p_after = winprob_attacker(models, phase, after_fire_feats, after_opp_feats)

    wb, wa = winprob_to_score100(p_before), winprob_to_score100(p_after)
    return DeltaWinProbResult(
        winprob_before=wb, winprob_after=wa, delta_winprob=wa - wb,
        attacker_dead_after=vpair.attacker_dead, opponent_dead_after=vpair.opponent_dead,
    )


# =============================================================================
# 4b. 連鎖中の仮想盤面ウィンドウ (2026-08-03 指摘2/Fix B)
# =============================================================================
#
# 背景 (main実測、userレビュー match_02 2996.5s〜3006.9s): _build_stable_timeline
# (従来ライン) は各サイドの「直近STABLE盤面」を前方保持するだけのため、発火側が
# 連鎖実行中(非STABLE)の間、その側の直近STABLE値は「連鎖発火**前**の盤面
# (=消化していない満載の連鎖がまだ乗っている状態)」のまま据え置かれる。
# これは (a) 発火側の火力を「まだ使っていない」ものとして二重計上し、
# (b) 相手側への着弾 (お邪魔) もまだ反映しない、という2つの誤りを生む。
# 修正: 発火検知〜(発火側自身の次のSTABLE確定)の間は、Step2の仮想盤面
# (reconstruct_virtual_board_pair: 連鎖消化後+予測正味おじゃま着弾済み) を
# 両サイドの評価に使う。

# ウィンドウ終端のフォールバック (発火側の「次のSTABLE」が試合終了まで
# 見つからない場合の打ち切り秒数、マジックナンバー回避)
CHAIN_WINDOW_FALLBACK_TAIL_SEC: float = 20.0


def _prev_own_stable_time(
    video_cache: _VideoNpzCache, fire_side: str, game_idx: int, t_fire: float,
) -> float:
    """発火側自身の「直前の(t_fireより厳密に前の)STABLE確定時刻」を返す。

    見つからなければ t_fire そのものを返す (連鎖前STABLEが記録されていない
    境界ケースの安全弁、ウィンドウ長ゼロ相当)。
    """
    own_rec = video_cache.r1p if fire_side == "1P" else video_cache.r2p
    own_mask = own_rec.game_idx == game_idx
    own_t = own_rec.t_sec[own_mask]
    earlier = own_t[own_t < t_fire]
    if len(earlier) == 0:
        return t_fire
    return float(earlier.max())


def ignition_time_for_event(
    video_cache: _VideoNpzCache, fire_side: str, game_idx: int, t_sec: float,
) -> float:
    """発火側自身の直前STABLE確定時刻を「発火検知(近似)時刻」として返す (データ駆動)。

    2026-08-03 指摘 (欠陥C) 対処: 旧実装は
    src.indicators_v2.estimate_chain_anim_duration_sec (CHAIN_ANIM_PER_STEP_SEC
    =0.4秒/連鎖) による近似式だったが、これは「着弾遅延の物差し」であり
    「連鎖アニメの実時間」ではない (memory
    project_exchange_measurement_foundation_2026-07-22: 8連鎖実測14.5秒、
    0.4秒/連鎖=3.2秒は大幅な過小評価)。main実測 (match_01) で大型連鎖ほど
    最大9秒近似が遅れることが確認されたため、近似式を廃止し
    「発火側自身の直前STABLE行の実時刻」で置き換える (連鎖構築完了直後の
    確定盤面がそのまま連鎖開始点になる、というデータ駆動の定義)。
    scripts.render_delta_winprob_demo と共有するためここに定義する
    (両モジュールでの重複実装を避ける、2026-08-03 移設)。
    """
    return _prev_own_stable_time(video_cache, fire_side, game_idx, t_sec)


@dataclass(frozen=True)
class ChainInProgressWindow:
    """連鎖中(発火検知〜発火側の次のSTABLE確定)に従来ラインが使うべき仮想盤面。

    2026-08-03 指摘 (欠陥E-2) 対処: 固定するのは**発火側自身**の仮想盤面
    (連鎖消化後+Fix E-1で1ターン上限適用済みの着弾) のみ。相手側は
    「連鎖中も自由に積んでいる」(userレビュー実況: 「1Pの連鎖の時間で
    2Pがぷよ量有利を取る」) ため、ウィンドウ内でも各評価時点の最新実測
    STABLE盤面を使う (_build_stable_timeline 側で live 参照する、
    本データクラスは相手側盤面を保持しない)。
    """
    fire_side: str      # "1P" or "2P" (この仮想盤面がどちら側のものか)
    ignition_sec: float
    window_end_sec: float
    board_after: Board  # 発火側自身の仮想盤面 (固定表示対象)


def _next_own_stable_time(
    video_cache: _VideoNpzCache, fire_side: str, game_idx: int, t_fire: float,
) -> float:
    """発火側自身の「次のSTABLE確定時刻」(=連鎖後に実際に落ち着いた時刻) を返す。

    見つからなければ CHAIN_WINDOW_FALLBACK_TAIL_SEC 後で打ち切る (末尾ガード、
    試合末尾のイベント等で「次」が存在しないケースの安全弁)。
    """
    own_rec = video_cache.r1p if fire_side == "1P" else video_cache.r2p
    own_mask = own_rec.game_idx == game_idx
    own_t = own_rec.t_sec[own_mask]
    later = own_t[own_t > t_fire]
    if len(later) == 0:
        return t_fire + CHAIN_WINDOW_FALLBACK_TAIL_SEC
    return float(later.min())


def build_chain_in_progress_windows(
    events_df: pd.DataFrame, video_cache: _VideoNpzCache, simulator: ChainSimulator,
) -> list[ChainInProgressWindow]:
    """1試合分のΔWinProbイベントから、連鎖中に使う仮想盤面ウィンドウ一覧を作る。

    events_df は最低限 t_sec/fire_side/approx_fire_chains/game_idx/
    stack_net_ojama_after_pred 列を持つこと (match_failed 行は除外済みで渡す)。
    ignition_sec 昇順にソートして返す (_find_active_chain_window の探索前提)。

    2026-08-03 指摘 (欠陥E-2) 対処: 固定表示するのは発火側自身の仮想盤面
    (vpair.attacker_board_after) だけであり、相手側 (vpair.opponent_board_after)
    はここでは使わない (_build_stable_timeline 側で相手側は毎回 live 参照
    する、ChainInProgressWindow docstring 参照)。相手側盤面はそれでも
    `reconstruct_virtual_board_pair` の入力として必要 (連鎖消化計算とは
    独立だが、ojama_after_pred が負値=攻撃側が受け取る場合の計算に使う)。
    """
    windows: list[ChainInProgressWindow] = []
    for _, ev in events_df.iterrows():
        fire_side = str(ev["fire_side"])
        game_idx = int(ev["game_idx"])
        t_fire = float(ev["t_sec"])
        pair = reconstruct_event_board_pair(video_cache, game_idx, t_fire, fire_side)
        if pair is None:
            continue
        fire_board, opp_board = pair
        net_ojama_pred = float(ev["stack_net_ojama_after_pred"])
        vpair = reconstruct_virtual_board_pair(fire_board, opp_board, net_ojama_pred, simulator=simulator)
        window_end = _next_own_stable_time(video_cache, fire_side, game_idx, t_fire)
        windows.append(ChainInProgressWindow(
            fire_side=fire_side,
            ignition_sec=ignition_time_for_event(video_cache, fire_side, game_idx, t_fire),
            window_end_sec=window_end, board_after=vpair.attacker_board_after,
        ))
    windows.sort(key=lambda w: w.ignition_sec)
    return windows


def _find_active_chain_window(
    windows: list[ChainInProgressWindow], t: float,
) -> "ChainInProgressWindow | None":
    """時刻tを含む最新のウィンドウを返す (ignition_sec昇順ソート済み前提、無ければNone)。"""
    active: "ChainInProgressWindow | None" = None
    for w in windows:
        if w.ignition_sec <= t < w.window_end_sec:
            active = w
    return active


# =============================================================================
# 4c. 相打ち (時間的因果関係のある発火) の相殺 (2026-08-03 指摘 欠陥D/Fix F)
# =============================================================================
#
# 背景 (main実測、match_01 2939.17s): 2P発火の瞬間、1Pの8連鎖が進行中
# (OPP_CHAINING) だったため sim は「相手の反撃力=0」として2P火力を丸ごと
# 加点していた。しかし1Pの連鎖は既に実測データとして分かっている(オフライン
# レンダでは両イベントとも既知)ため、実測 net_ojama_after を相殺した正味を使う。
#
# 2026-08-03 指摘 欠陥F: 初版は「活動窓 [ignition_sec, window_end_sec) が
# 重なっていれば相殺」という対称な判定だったが、window_end_sec (=次の自分の
# STABLE確定、Fix B/E-2の表示用の概念) を終端に使うと、実際にはほぼ逐次的な
# 2発火 (match_05 3168.73s/3175.53s、重なりはわずか0.6秒) まで相打ちと
# 誤判定し、「まだ起きていない未来の反撃」を先取りして相殺してしまう
# (時間的因果に反する)。修正: 「自分の点火時刻に、相手の連鎖が既に
# 空中にある (点火済みだが未完了) 場合のみ」相殺対象とする一方向の因果判定
# に変更する。相手の飛行区間は [相手のignition_sec, 相手のt_sec) (=点火から
# 実際の完了(スコア確定)まで、window_end_sec ではない) とする。

@dataclass(frozen=True)
class EventActivityWindow:
    """1発火イベント分の「非STABLE活動窓」(相打ち検出用 + 欠陥G改の予告台帳用)。"""
    row_index: object  # DataFrame の元インデックス (突合用)
    fire_side: str
    t_sec: float             # 発火完了(スコア確定)時刻
    ignition_sec: float      # 発火検知(=直前STABLE、Fix C)時刻
    window_end_sec: float    # 次の自分のSTABLE確定時刻 (Fix B)
    net_ojama_after: float   # 実測(ground truth) 正味お邪魔
    receiver_baseline_ojama: float = 0.0  # t_sec時点の受け手盤面の実測おじゃま数
    #   (欠陥G改「着弾済み分の控除」用基準値。受け手が t_sec 以降に実際に
    #   受け取ったおじゃま数 = 評価時点の受け手盤面おじゃま数 - この基準値、
    #   として二重計上を防ぐ、_aggregate_known_pending_net_ojama 参照)
    fire_chain_count: float = 0.0  # 発火側の連鎖数 (欠陥G2「構築済み返し能力」用。
    #   着弾までの遅延 = estimate_available_hands(この値) から、受け側が
    #   何手分の応手時間を持つかを見積もる、_realizable_counter_ojama 参照)


def build_event_activity_windows(
    game_events: pd.DataFrame, video_cache: _VideoNpzCache,
) -> list[EventActivityWindow]:
    """1試合(1 game_idx)分の全発火イベントの活動窓一覧を作る。

    receiver_baseline_ojama は「発火完了(t_sec)時点で、受け手側の盤面に
    実測で何個のおじゃまが既にあったか」(時刻最近傍で復元、
    reconstruct_event_board_pair と同じ突合方式)。評価時点の受け手盤面の
    おじゃま数からこの基準値を引いた分だけが「この発火が着弾済み」とみなせる
    (欠陥G改、main実測 match_02 の再現に必要な会計修正)。
    """
    windows: list[EventActivityWindow] = []
    for idx, ev in game_events.iterrows():
        fire_side = str(ev["fire_side"])
        game_idx = int(ev["game_idx"])
        t_fire = float(ev["t_sec"])
        receiver_side = "2P" if fire_side == "1P" else "1P"
        receiver_rec = video_cache.r2p if receiver_side == "2P" else video_cache.r1p
        receiver_mask = receiver_rec.game_idx == game_idx
        receiver_t = receiver_rec.t_sec[receiver_mask]
        receiver_grids = receiver_rec.grids[receiver_mask]
        baseline_ojama = 0.0
        if len(receiver_t) > 0:
            nearest = int(np.argmin(np.abs(receiver_t - t_fire)))
            baseline_board = _board_from_grid(receiver_grids[nearest])
            baseline_ojama = float(iv.board_ojama_count(baseline_board).raw)
        windows.append(EventActivityWindow(
            row_index=idx, fire_side=fire_side, t_sec=t_fire,
            ignition_sec=ignition_time_for_event(video_cache, fire_side, game_idx, t_fire),
            window_end_sec=_next_own_stable_time(video_cache, fire_side, game_idx, t_fire),
            net_ojama_after=float(ev["net_ojama_after"]),
            receiver_baseline_ojama=baseline_ojama,
            fire_chain_count=float(ev["approx_fire_chains"])
            if "approx_fire_chains" in game_events.columns and pd.notna(ev["approx_fire_chains"])
            else 0.0,
        ))
    return windows


def _is_airborne_at(window: EventActivityWindow, moment: float) -> bool:
    """momentの時点で window (発火イベント) が「まだ空中」(点火済み・未完了) か。

    飛行区間は [ignition_sec, t_sec) (点火から実際の完了=スコア確定まで)。
    window_end_sec (次の自分のSTABLE確定、Fix B/E-2の表示用) は使わない
    (欠陥Fの根因、モジュール冒頭コメント参照)。
    """
    return window.ignition_sec <= moment < window.t_sec


def find_mutual_exchange_partner(
    target: EventActivityWindow, all_windows: list[EventActivityWindow],
) -> "EventActivityWindow | None":
    """target の点火時刻に、既に空中にある「相手側 (fire_side が異なる)」の
    イベントを返す (時間的因果制約、欠陥F対処)。

    「自分の点火時刻に相手の連鎖が既に空中にある」場合のみを対象とする
    一方向判定 (対称な重なり判定ではない、_is_airborne_at 参照)。
    未来にまだ起きていない相手の反撃を先取りして相殺することはない。
    複数該当する場合は ignition_sec が最も近い(直近に点火した)ものを採用する
    (無ければ None、= 相打ちではない通常イベント)。
    """
    candidates = [
        w for w in all_windows
        if w.row_index != target.row_index and w.fire_side != target.fire_side
        and _is_airborne_at(w, target.ignition_sec)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda w: abs(w.ignition_sec - target.ignition_sec))


def apply_mutual_exchange_adjustment(
    events_df: pd.DataFrame, video_cache: _VideoNpzCache,
    net_ojama_pred_col: str = "stack_net_ojama_after_pred",
) -> pd.DataFrame:
    """相打ちを検出し、実測 net_ojama_after の相殺値で net_ojama_pred_col を上書きする。

    上書き対象は呼び出し時点で net_ojama_pred_col に入っている値 (通常は
    モデル予測値) で、相打ちが検出された行のみ「自分の実測net_ojama_after −
    相手の実測net_ojama_after」に置き換える (相打ちが無い行は既存の予測値の
    まま、後方互換)。新規列 is_mutual_exchange (bool) / mutual_partner_t_sec
    (float、無ければNaN) を追加する。events_df は t_sec/fire_side/game_idx/
    net_ojama_after 列を持つこと (呼び出し前提、無ければ KeyError で早期検出)。
    """
    out = events_df.copy()
    out["is_mutual_exchange"] = False
    out["mutual_partner_t_sec"] = np.nan
    for _game_idx, game_events in out.groupby("game_idx"):
        windows = build_event_activity_windows(game_events, video_cache)
        for w in windows:
            partner = find_mutual_exchange_partner(w, windows)
            if partner is None:
                continue
            net = w.net_ojama_after - partner.net_ojama_after
            out.at[w.row_index, net_ojama_pred_col] = net
            out.at[w.row_index, "is_mutual_exchange"] = True
            out.at[w.row_index, "mutual_partner_t_sec"] = partner.t_sec
    return out


# =============================================================================
# 4d. 予告台帳の相殺会計 + 受け切れ判定 (2026-08-03 指摘 欠陥G→欠陥G改→欠陥G2)
# =============================================================================
#
# 背景 (main実測 match_01/match_02): 欠陥D/Fの相打ち相殺は「重なる2イベント
# ペアのみ」を対象にしていたが、実際の勝率へは反映されず (欠陥E-2で相手側は
# live評価にしたのみ)、かつ「受け切れない (物理的に埋まる)」ケースを明示的に
# 判定していなかった。
#
# 初版 (欠陥G) は「点火済み・未着弾 (=t_sec未到達) の空中イベントのみ」を
# 相殺対象にしていたが、これは「送付済み・着弾待ちの予告 (=t_sec到達後、
# 受け手がまだ物理的に受け取っていない分)」を勘定に入れておらず、
# match_02 (2988s、2Pの349は送付済み・未着弾なのに勘定から漏れて
# 1Pの317がそのまま2Pへの脅威になった) で誤判定した。
#
# 修正 (欠陥G改): 「点火済み (ignition_sec<=t)」の全イベントを対象に、
# t < t_sec (未生成・空中) は実測net_ojama_afterを全額、t >= t_sec
# (生成済み・着弾待ち) は「受け手盤面の実測おじゃま増分」を観測して
# 着弾済み分を控除した残りを「予告台帳」として集計する (二重計上防止)。
# 台帳を src.ojama_accounting.cancel_own_pending_then_send_surplus (既存の
# 会計ロジックそのまま再利用) で相殺し、正味予告が残る側の空き容量のみと
# 比較して「受け切れない」と判定していたが、match_04 (3093-3101、2Pの217
# を受けた1Pが実は既に8連鎖相当を構築済みで即座に返せる状況) で
# 「5%平坦→3101で90pt垂直反転」という不自然な挙動になった (main実測、
# 受け切れない判定が「構築済みの返し能力」を無視していたため)。
#
# 修正 (欠陥G2): 受け切れ判定に「受け側の構築済み返し能力」を**台帳側の
# 控除**として組み込む (容量側への加算ではない、以前の懸念の解消)。
# realizable_counter = expected_fire_power(受け側の現在ライブ盤面,
# k_hands=estimate_available_hands(攻撃側連鎖数)) の raw値 (#24 sim部品の
# 再利用、estimate_expected_net_damage と同じ構造)。P' = P - realizable_
# counter を room と比較する (P'<=0 なら完全に返せる、P'>room なら
# 受け切れない)。
# =============================================================================

# 受け切れない判定時の勝率クランプ値 (生存側=予告を送った側の視点、%)。
# user指定「例:92-97%帯」の中央値を採用 (isotonic出力とのmaxで下限として使う、
# 通常評価がこれを超えていればクランプは効かない=より強い方を採用)。
LETHAL_CLAMP_FAVOR_PCT: float = 95.0


def _aggregate_known_pending_net_ojama(
    activity_windows: list[EventActivityWindow], t: float, board_1p: Board, board_2p: Board,
) -> tuple[float, float, float, float]:
    """時刻tまでに点火済み(ignition_sec<=t)の全イベントについて、実測
    net_ojama_after から「着弾済み分」を控除した残存予告を両側で集計する
    (欠陥G改)。

    t < event.t_sec (まだ生成されていない、空中): 全額をそのまま計上する
    (オフラインでは将来の確定値が既知のため先取りする、Fix C/Eと同じ思想)。
    t >= event.t_sec (生成済み、着弾待ち): 受け手盤面 (board_1p/board_2p、
    E-2のライブ評価と同じもの) の実測おじゃま数から
    event.receiver_baseline_ojama (発火完了時点の基準値) を引いた増分を
    「着弾済み」とみなし、実測net_ojama_afterから控除した残りだけを計上する
    (二重計上防止: E-2で既にライブ盤面に反映されている分を台帳からも
    引かないと過大評価になる)。

    まだ残存予告のある (remaining>0) イベントに限り、その発火側の
    fire_chain_count の最大値も併せて追跡する (欠陥G2、_realizable_counter_
    ojama で「着弾までに受け側が持てる手数」の見積もりに使う)。

    Returns:
        (1P発火分の残存予告合計, 2P発火分の残存予告合計,
         1P発火分の最大連鎖数(remaining>0限定), 2P発火分の最大連鎖数) の4値。
    """
    current_ojama_1p = float(iv.board_ojama_count(board_1p).raw)
    current_ojama_2p = float(iv.board_ojama_count(board_2p).raw)
    pending_from_1p = 0.0
    pending_from_2p = 0.0
    chain_from_1p = 0.0
    chain_from_2p = 0.0
    for w in activity_windows:
        if w.ignition_sec > t:
            continue  # まだ点火していない (時間的因果、Fix F と同じ考え方)
        if t < w.t_sec:
            remaining = max(0.0, w.net_ojama_after)
        else:
            receiver_current = current_ojama_2p if w.fire_side == "1P" else current_ojama_1p
            landed = max(0.0, receiver_current - w.receiver_baseline_ojama)
            remaining = max(0.0, w.net_ojama_after - landed)
        if w.fire_side == "1P":
            pending_from_1p += remaining
            if remaining > 0.0:
                chain_from_1p = max(chain_from_1p, w.fire_chain_count)
        else:
            pending_from_2p += remaining
            if remaining > 0.0:
                chain_from_2p = max(chain_from_2p, w.fire_chain_count)
    return pending_from_1p, pending_from_2p, chain_from_1p, chain_from_2p


def _net_pending_after_cancellation(
    attack_from_1p: float, attack_from_2p: float,
) -> tuple[float, float]:
    """両側の予告台帳を相殺し、高々片側にのみ残る正味予告を返す。

    src.ojama_accounting.cancel_own_pending_then_send_surplus をそのまま
    再利用する (「1Pの生成が1P自身の受信中pending(=2P発火分)を相殺し、
    余剰を2Pのpendingへ送る」という既存の会計モデルを、オフライン集計値に
    適用する)。整数専用の関数のため四捨五入してから渡す。

    Returns:
        (1Pが最終的に受ける正味予告, 2Pが最終的に受ける正味予告) のタプル
        (どちらか一方は必ず0)。
    """
    pending_on_1p, pending_on_2p = cancel_own_pending_then_send_surplus(
        gen=int(round(attack_from_1p)), own_pending=int(round(attack_from_2p)), other_pending=0,
    )
    return float(pending_on_1p), float(pending_on_2p)


# expected_fire_power は K=3,4 でモンテカルロ近似 (重い) のため、
# タイムライン評価 (0.2秒刻み) で同一盤面×同一k_handsを何度も再計算しない
# ようにメモ化する (main指摘「発火イベント近傍窓のみ計算で可」への対処、
# 値は変えず計算回数だけ減らす)。盤面内容が同じなら結果は必ず同じ
# (stateless、Board.grid_bytes() で内容ベースのキーを作る)。
_REALIZABLE_COUNTER_CACHE: dict[tuple[bytes, int], float] = {}


def _realizable_counter_ojama(defender_board: Board, attacker_chain_count: float) -> float:
    """受け側の「構築済み返し能力」(欠陥G2)。既存 #24 sim 部品の組み替えのみ
    (新規ロジック最小、estimate_expected_net_damage と同じ構造)。

    k_hands = estimate_available_hands(攻撃側連鎖数) で「着弾までに受け側が
    打てる手数」を見積もり、受け側の現在ライブ盤面から
    expected_fire_power(k_hands) の raw 値 (お邪魔換算の期待火力) を返す。
    未来の情報は使わない (受け側の現在盤面のみで判定する、時間的因果を
    満たす)。attacker_chain_count=0.0 (連鎖数不明) の場合は
    estimate_available_hands(0)>=1 が返るため常に有効な k_hands になる。
    """
    k_hands = estimate_available_hands(int(round(attacker_chain_count)))
    cache_key = (defender_board.grid_bytes(), k_hands)
    cached = _REALIZABLE_COUNTER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    result = iv.expected_fire_power(defender_board, k_levels=(k_hands,))
    value = float(result.values[k_hands].raw)
    _REALIZABLE_COUNTER_CACHE[cache_key] = value
    return value


def _lethal_readout_clamp(
    activity_windows: list[EventActivityWindow], t: float,
    board_1p: Board, board_2p: Board, model_winprob_1p: float, simulator: ChainSimulator,
) -> float:
    """受け切れ判定 (欠陥G2)。通常評価 (model_winprob_1p) をそのまま返すか、
    受け切れないと判定した側に不利なクランプを適用して返す。

    正味予告 P (相殺後、_net_pending_after_cancellation) から、受け側の
    構築済み返し能力 realizable_counter (_realizable_counter_ojama) を
    控除した P' を room と比較する (欠陥G改の「room単独」判定に、反撃力を
    容量側でなく**台帳側の控除**として組み込んだもの、main承認済み設計)。
    P' <= 0 (完全に返せる) なら容量比較すら行わずクランプしない。
    容量内 (受け切れる) の場合や空中おじゃまが無い場合は model_winprob_1p を
    そのまま返す (上書きしない、既存の盤面ライブ評価=欠陥E-2の結果を優先)。
    simulator は将来拡張用の引数として残すが、本関数では未使用。
    """
    attack_from_1p, attack_from_2p, chain_from_1p, chain_from_2p = _aggregate_known_pending_net_ojama(
        activity_windows, t, board_1p, board_2p,
    )
    if attack_from_1p == 0.0 and attack_from_2p == 0.0:
        return model_winprob_1p
    pending_on_1p, pending_on_2p = _net_pending_after_cancellation(attack_from_1p, attack_from_2p)
    if pending_on_1p <= 0.0 and pending_on_2p <= 0.0:
        return model_winprob_1p  # 完全相殺、脅威側なし

    if pending_on_1p > 0.0:
        threatened_board, pending, attacker_chain = board_1p, pending_on_1p, chain_from_2p
    else:
        threatened_board, pending, attacker_chain = board_2p, pending_on_2p, chain_from_1p
    realizable_counter = _realizable_counter_ojama(threatened_board, attacker_chain)
    net_pending = pending - realizable_counter
    if net_pending <= 0.0:
        return model_winprob_1p  # 構築済み返しだけで完全に返せる、通常評価のまま
    room = board_room(threatened_board)
    if net_pending <= room:
        return model_winprob_1p  # 容量内 = 受け切れる、通常評価のまま

    # 受け切れない: 予告を送った側 (脅かされていない側) へ強くクランプする。
    if pending_on_1p > 0.0:
        return min(model_winprob_1p, 100.0 - LETHAL_CLAMP_FAVOR_PCT)  # 1P脅威 -> 2P有利
    return max(model_winprob_1p, LETHAL_CLAMP_FAVOR_PCT)  # 2P脅威 -> 1P有利


# =============================================================================
# 5. 全イベント処理 (動画単位でグルーピングし並列化)
# =============================================================================

# fork (Linux/WSL既定) の COW を使い、Pool生成前にセットした値を子プロセスへ
# 引き継がせるためのモジュールグローバル (再pickleコストを避ける)。
_WORKER_MODELS: "dict[str, PhaseWinprobModel] | None" = None
_WORKER_NPZ_DIR: "Path | None" = None


def _process_one_video(args: tuple[str, list[dict]]) -> list[dict]:
    """1動画分のイベント行を処理し、ΔWinProb列を付けたdictのリストを返す。"""
    video_id, rows = args
    models, npz_dir = _WORKER_MODELS, _WORKER_NPZ_DIR
    assert models is not None and npz_dir is not None, "ワーカー初期化前に呼ばれた"
    cache = _load_video_npz(video_id, npz_dir)
    sim = ChainSimulator()
    results: list[dict] = []
    for row in rows:
        out = dict(row)
        if cache is None:
            out["match_failed"] = True
            results.append(out)
            continue
        pair = reconstruct_event_board_pair(cache, row["game_idx"], row["t_sec"], row["fire_side"])
        if pair is None:
            out["match_failed"] = True
            results.append(out)
            continue
        fire_board, opp_board = pair
        delta = compute_delta_winprob_for_event(
            fire_board, opp_board, row["phase"], row["stack_net_ojama_after_pred"], models, sim,
        )
        out.update(
            match_failed=False,
            winprob_before=delta.winprob_before,
            winprob_after=delta.winprob_after,
            delta_winprob=delta.delta_winprob,
            attacker_dead_after=delta.attacker_dead_after,
            opponent_dead_after=delta.opponent_dead_after,
        )
        results.append(out)
    return results


def compute_all_delta_winprob(
    df: pd.DataFrame, models: dict[str, PhaseWinprobModel], npz_dir: Path, n_workers: int,
) -> pd.DataFrame:
    """全イベント行を動画単位で並列処理し、ΔWinProb列付き DataFrame を返す。"""
    global _WORKER_MODELS, _WORKER_NPZ_DIR
    _WORKER_MODELS, _WORKER_NPZ_DIR = models, npz_dir

    groups = [
        (vid, sub.to_dict("records"))
        for vid, sub in df.groupby("video_id", sort=False)
    ]
    print(f"[compute_all] 動画数={len(groups)} イベント総数={len(df)} workers={n_workers}")
    if n_workers <= 1:
        all_rows = [r for g in groups for r in _process_one_video(g)]
    else:
        with mp.Pool(processes=n_workers) as pool:
            nested = pool.map(_process_one_video, groups)
        all_rows = [r for chunk in nested for r in chunk]
    out_df = pd.DataFrame(all_rows)
    n_failed = int(out_df["match_failed"].sum())
    print(f"[compute_all] 盤面突合失敗={n_failed}/{len(out_df)}行 "
          f"({n_failed / len(out_df):.2%})")
    return out_df


# =============================================================================
# 6. 併用モデルOOF予測の付与 (run_exchange_triple_comparison を再利用)
# =============================================================================

def load_aug_with_stacking_predictions(aug_csv: Path, model_d_dir: Path, n_folds: int) -> pd.DataFrame:
    """aug CSV + 案D OOF を突合し、併用スタッキングOOF予測列を付けて返す。

    run_exchange_triple_comparison.py の手順1-4と同一処理を再利用する
    (コピペ再実装しない)。
    """
    aug_df = load_exchange_labels(str(aug_csv))
    oof_df = load_model_d_oof(model_d_dir)
    merged = align_aug_with_model_d(aug_df, oof_df)
    merged = filter_nan_sim_rows(merged)
    oof_proba_stack, oof_pred_stack, _feat_names = build_stacking_oof_predictions(merged, n_folds)
    merged = merged.copy()
    merged["stack_prob_taiou_success"] = oof_proba_stack
    merged["stack_net_ojama_after_pred"] = oof_pred_stack
    return merged


# =============================================================================
# 7. 健全性チェック (stdout)
# =============================================================================

def print_sanity_checks(df: pd.DataFrame) -> None:
    """ΔWinProbの分布・発火側有利率・大変動割合・窒息率・逆行ケースを表示する。"""
    valid = df.loc[~df["match_failed"]].copy()
    print(f"\n=== 健全性チェック (突合成功 {len(valid)}/{len(df)} 行) ===")
    print("--- 位相別 ΔWinProb 分布 ---")
    for phase in PHASES:
        sub = valid.loc[valid["phase"] == phase, "delta_winprob"]
        if len(sub) == 0:
            continue
        print(f"  {phase}: n={len(sub)} 平均={sub.mean():+.2f} 中央値={sub.median():+.2f}"
              f" 標準偏差={sub.std():.2f}")
    favorable_rate = float((valid["delta_winprob"] > 0).mean())
    print(f"--- 発火側有利方向 (ΔWinProb>0) の比率: {favorable_rate:.1%} ---")
    large_rate = float((valid["delta_winprob"].abs() > LARGE_DELTA_ABS_THRESHOLD).mean())
    print(f"--- |ΔWinProb|>{LARGE_DELTA_ABS_THRESHOLD:.0f} の割合: {large_rate:.1%} ---")
    dead_rate_atk = float(valid["attacker_dead_after"].mean())
    dead_rate_opp = float(valid["opponent_dead_after"].mean())
    print(f"--- 窒息フラグ発生率: 攻撃側後={dead_rate_atk:.2%}  相手側後={dead_rate_opp:.2%} ---")

    print(f"--- 「発火したのに勝率が下がる」上位{N_SANITY_TOP_ROWS}件 ---")
    worst = valid.sort_values("delta_winprob").head(N_SANITY_TOP_ROWS)
    for _, r in worst.iterrows():
        print(f"  video={r['video_id']} game={r['game_idx']} t_sec={r['t_sec']:.1f}"
              f" fire_side={r['fire_side']} phase={r['phase']}"
              f" ΔWinProb={r['delta_winprob']:+.2f}"
              f" (before={r['winprob_before']:+.2f} after={r['winprob_after']:+.2f})")


# =============================================================================
# 8. デモviz (従来STABLE推移 vs 発火直後速報の重ね書き)
# =============================================================================

def _build_stable_timeline(
    video_cache: _VideoNpzCache, game_idx: int, models: dict[str, PhaseWinprobModel],
    simulator: ChainSimulator,
    chain_windows: "list[ChainInProgressWindow] | None" = None,
    activity_windows: "list[EventActivityWindow] | None" = None,
) -> pd.DataFrame:
    """1試合分、両サイドの全STABLEスナップショット時刻の和集合で勝率推移を作る。

    2026-08-03 userレビュー指摘2 (発火時しか指標値が動かない) 対処。
    旧実装は npz が既に「STABLE確定時のみ記録される疎な配列」であるにも
    関わらず、さらに DEMO_FRAME_STRIDE=15 の間引きを二重適用しており、
    実測 c61 game_idx=16 (84秒) でわずか7点にしかならないバグだった
    (main実測、scratchpad/diag_timeline_granularity2.py 参照)。

    新実装は間引きを行わず、各サイドの「直近STABLE盤面」を前方保持
    (forward-fill) で独立に組み合わせ、両サイドの全STABLEスナップショット
    時刻の和集合の各点で評価する (設置のたびに動く連続的なラインになる)。
    どちらか一方がまだ最初のSTABLEに達していない時刻 (前方保持不能) は
    スキップする。

    chain_windows: optional (2026-08-03 指摘2/Fix B、指摘 欠陥E-2 で修正)。
    渡すと、評価時刻がいずれかのウィンドウ内にある場合、**発火側自身のみ**
    素の前方保持グリッドではなく Step2 仮想盤面 (連鎖消化後+1ターン上限
    適用済み着弾、build_chain_in_progress_windows docstring 参照) に
    差し替える (発火側の「使用済み連鎖をまだ持っている」二重計上を防ぐ)。
    **相手側は常に素の前方保持 (live) を使う** (相手は連鎖中も自由に積んで
    いるため、固定盤面では反映できない、欠陥E-2)。既定 None = 従来通り
    両側とも素の前方保持のみ (後方互換)。

    activity_windows: optional (2026-08-03 指摘 欠陥G)。渡すと、各評価時刻で
    「点火済み・未着弾」の全イベントの実測 net_ojama_after を相殺会計し、
    受け切れない (空き容量+近い将来の返し能力を超える) と判定した場合、
    盤面ライブ評価の結果を予告側 (攻撃側) へ強くクランプする
    (_lethal_readout_clamp docstring 参照)。既定 None = クランプ無効
    (後方互換)。
    """
    mask1, mask2 = video_cache.r1p.game_idx == game_idx, video_cache.r2p.game_idx == game_idx
    t1, g1 = video_cache.r1p.t_sec[mask1], video_cache.r1p.grids[mask1]
    t2, g2 = video_cache.r2p.t_sec[mask2], video_cache.r2p.grids[mask2]
    if len(t1) == 0 or len(t2) == 0:
        return pd.DataFrame(columns=["t_sec", "winprob_1p"])
    puyo_totals = np.array([int((g != 0).sum()) for g in g1], dtype=float)
    q_low, q_high = np.quantile(puyo_totals, 0.33), np.quantile(puyo_totals, 0.67)
    windows = chain_windows or []
    activity = activity_windows or []

    eval_times = np.union1d(t1, t2)  # 両サイドの全STABLE時刻の和集合 (昇順・重複除去)
    rows: list[dict] = []
    for t in eval_times:
        idx1 = int(np.searchsorted(t1, t, side="right")) - 1
        idx2 = int(np.searchsorted(t2, t, side="right")) - 1
        if idx1 < 0 or idx2 < 0:
            continue
        phase = "序" if puyo_totals[idx1] <= q_low else ("終" if puyo_totals[idx1] > q_high else "中")
        if phase not in models:
            continue
        b1, b2 = _board_from_grid(g1[idx1]), _board_from_grid(g2[idx2])
        active = _find_active_chain_window(windows, float(t))
        if active is not None:
            # 発火側自身のみ仮想盤面に差し替える (相手側は live のまま、欠陥E-2)。
            if active.fire_side == "1P":
                b1 = active.board_after
            else:
                b2 = active.board_after
        f1, f2 = compute_board_only_features(b1, simulator), compute_board_only_features(b2, simulator)
        p1 = winprob_attacker(models, phase, f1, f2)
        winprob_1p = winprob_to_score100(p1)
        if activity:
            winprob_1p = _lethal_readout_clamp(activity, float(t), b1, b2, winprob_1p, simulator)
        rows.append({"t_sec": float(t), "winprob_1p": winprob_1p})
    return pd.DataFrame(rows)


def render_demo_viz(
    video_id: str, game_idx: int, timeline: pd.DataFrame, events: pd.DataFrame, out_path: Path,
) -> None:
    """従来STABLE推移 (実線) + 発火直後速報 (前後ジャンプ) を1枚のPNGに描く。"""
    # 日本語ラベルの文字化け対策 (feedback_terminal_font_mojibake、
    # exchange_meter_eval_harness.plot_reliability_diagrams と同一パターン)。
    meiryo_path = "/mnt/c/Windows/Fonts/meiryo.ttc"
    if Path(meiryo_path).exists():
        font_manager.fontManager.addfont(meiryo_path)
        plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(figsize=DEMO_FIG_SIZE)
    ax.plot(timeline["t_sec"], timeline["winprob_1p"], color="steelblue",
            linewidth=1.5, label="従来の勝率推移 (STABLE評価のみ、1P視点)")
    ax.axhline(50.0, color="gray", linewidth=0.8, linestyle=":")

    for _, ev in events.iterrows():
        # winprob_before/after は「発火側 (attacker) 視点の勝率 0〜100%」。
        # 1P視点に変換するには、fire_side=2P の場合は補数 (100-値) を取る
        # (符号反転ではない、2026-08-02 修正: 0〜100%スケールは0を中心に
        # 対称ではないため sign 反転は誤り)。
        is_1p = ev["fire_side"] == "1P"
        wb_1p = ev["winprob_before"] if is_1p else 100.0 - ev["winprob_before"]
        wa_1p = ev["winprob_after"] if is_1p else 100.0 - ev["winprob_after"]
        t = ev["t_sec"]
        ax.axvline(t, color="orange", alpha=DEMO_FIRE_LINE_ALPHA, linewidth=1.0)
        ax.plot([t, t], [wb_1p, wa_1p], color="crimson", marker="o", markersize=4, linewidth=2.0)
        ax.annotate(f"{ev['delta_winprob']:+.0f}", xy=(t, wa_1p),
                    fontsize=7, color="crimson", xytext=(2, 2), textcoords="offset points")

    ax.set_xlabel("動画内 経過秒 (t_sec)")
    ax.set_ylabel("1P視点 勝率 (0〜100%)")
    ax.set_title(f"{video_id} game_idx={game_idx}: 従来推移 vs 発火直後速報 (ΔWinProb, 案C Step3)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(-5, 105)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[demo_viz] 保存: {out_path}")


def build_demo_viz(
    video_id: str, npz_dir: Path, models: dict[str, PhaseWinprobModel],
    delta_df: pd.DataFrame, out_dir: Path,
) -> "Path | None":
    """デモviz対象動画の1試合を選び、タイムライン生成 + 描画までを行う。

    delta_df["video_id"] は aug CSV 由来の "video_c50" 形式のため、
    --demo-video に "c50" (npzファイル名幹と同じprefix無し表記) を渡された
    場合でも一致するよう正規化する。
    """
    cache = _load_video_npz(video_id, npz_dir)
    if cache is None:
        print(f"[demo_viz] {video_id} の npz が見つからないためスキップ")
        return None
    full_video_id = (
        video_id if video_id.startswith(VIDEO_ID_NPZ_PREFIX) else VIDEO_ID_NPZ_PREFIX + video_id
    )
    events = delta_df.loc[(delta_df["video_id"] == full_video_id) & (~delta_df["match_failed"])].copy()
    if len(events) == 0:
        print(f"[demo_viz] {video_id} に突合成功イベントが無いためスキップ")
        return None
    # 発火イベント数が最も多い試合 (=見応えのある展開) を選ぶ
    game_idx = int(events["game_idx"].value_counts().idxmax())
    events = events.loc[events["game_idx"] == game_idx].sort_values("t_sec")
    sim = ChainSimulator()
    timeline = _build_stable_timeline(cache, game_idx, models, sim)
    if len(timeline) == 0:
        print(f"[demo_viz] {video_id} game={game_idx} のタイムライン生成に失敗")
        return None
    out_path = out_dir / f"demo_viz_{video_id}_game{game_idx}.png"
    render_demo_viz(video_id, game_idx, timeline, events, out_path)
    return out_path


# =============================================================================
# 9. メイン
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する (main を50行以内に保つための分割)。"""
    parser = argparse.ArgumentParser(description="#24 打ち合い計測器 ΔWinProb接続 Step3")
    parser.add_argument("--aug-csv", type=Path, default=DEFAULT_AUG_CSV)
    parser.add_argument("--model-d-dir", type=Path, default=DEFAULT_MODEL_D_DIR)
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--labeled-win-csv", type=Path, default=DEFAULT_LABELED_WIN_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--demo-video", type=str, default=DEFAULT_DEMO_VIDEO_ID)
    parser.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS)
    parser.add_argument("--limit", type=int, default=None,
                        help="先頭N行だけで動作確認する (本走行では未指定)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. 併用モデルOOF予測付き aug データ読込 ===")
    merged = load_aug_with_stacking_predictions(args.aug_csv, args.model_d_dir, n_folds=5)
    if args.limit is not None:
        merged = merged.head(args.limit).reset_index(drop=True)
        print(f"[--limit] 先頭{args.limit}行のみ処理")

    print("\n=== 2. 勝率モデル (board-only指標 + 位相別isotonic校正) 学習 ===")
    models = train_winprob_models(args.labeled_win_csv)

    print("\n=== 3. 全イベント ΔWinProb 計算 ===")
    delta_df = compute_all_delta_winprob(merged, models, args.npz_dir, args.n_workers)
    out_csv = out_dir / "exchange_delta_winprob.csv"
    delta_df.to_csv(out_csv, index=False)
    print(f"[main] CSV保存: {out_csv} ({len(delta_df)}行)")

    print_sanity_checks(delta_df)

    print("\n=== 4. デモviz ===")
    build_demo_viz(args.demo_video, args.npz_dir, models, delta_df, out_dir)
    print(f"\n出力先: {out_dir}")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
