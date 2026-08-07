"""ΔWinProb接続 Step4: 新指標2列 (直近発火イベント由来) のアブレーション評価。

## 背景
Step3 (scripts/compute_exchange_delta_winprob.py) が全16,470発火イベントに
ΔWinProb (発火直前→仮想盤面ペア後の勝率変化) を付けた。本 Step4 は、この
ΔWinProb 系列を「評価時点より前の直近発火イベント由来の特徴量」として
labeled_win 側の勝率予測モデルに追加した場合、既存45指標(実測44指標)だけの
モデルより勝率予測が改善するかを検証するアブレーションである。

## 新指標2列 (時系列特徴量化)
- prob_taiou_success_last: 直近発火イベント (同一動画・同side) の
  対応成功確率予測 (併用スタッキングOOF予測 stack_prob_taiou_success)。
- delta_winprob_last: 同じく直近発火イベントの delta_winprob。

これらは各 labeled_win 評価時点 (video_id, side, t_sec) に対して「その時点
より前で最も新しい同一 video_id・同 side の発火イベント」を実時刻窓で突合
して埋め込む (**game_idx は信用しない**、labeled_win 側の game_idx は
ジョブ窓内相対インデックスで試合と対応しないことが既知
[[project_game_idx_desync_bug_2026-07-29]] のため、video_id 全体を1本の
連続タイムラインとして扱う。これは exchange_delta_winprob.csv 側の
game_idx が試合単位で単調増加するのとは別のパイプライン由来であることの
帰結でもあり、本アブレーションの前提として明記する)。

## 非発火時の既定値 (2方式)
- (a) neutral: 直近発火イベントの値をそのまま時刻減衰なしで保持する
  (直近発火が無ければ中立値 0.5 / 0.0)。
- (b) decay: 直近発火からの経過秒 gap に対し weight=exp(-gap/tau) で
  中立値へ指数減衰する (直近発火が無ければ weight=0 相当で中立値に一致)。
  既定 tau=10秒、tau感度は 10/20/40 で報告する。

## 評価方法
- 45指標(実測44) のみ / +2列(neutral) / +2列(decay, tau=10) の3構成で
  LogisticRegression を GroupKFold(video_id) OOF学習する
  (scripts/model_indicator_win.py の run_oof_lr をそのまま再利用)。
- 3構成は同一 paired DataFrame (行順・GroupKFold分割が完全一致) から
  特徴量列だけを差し替えて作るため、フェアなペアード比較になる。
- AUC差の有意性は動画クラスタ・ブートストラップCI
  (scripts/exchange_meter_eval_harness.py の bootstrap_diff_ci_by_video を再利用)。
  CIが0を跨ぐ場合は「効果を確認できず」と正直に報告する (盛らない)。
- 校正曲線 (reliability diagram) も同ハーネスの関数を再利用して出力する。

## 使い方
    PYTHONPATH=. python -m scripts.ablate_exchange_indicators
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.exchange_meter_eval_harness import (
    EXCHANGE_PHASES,
    N_BOOTSTRAP_RESAMPLES,
    bootstrap_diff_ci_by_video,
    compute_reliability_table,
    exact_auc,
    phase_power_flag,
    plot_reliability_diagrams,
)
from scripts.model_indicator_win import (
    DEFAULT_MAX_TDIFF,
    N_FOLDS,
    TSUMO_EARLY_RATIO,
    TSUMO_LATE_RATIO,
    _get_indicator_cols,
    build_features,
    load_labeled_csv,
    pair_sides_for_win,
    run_oof_lr,
)

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

DEFAULT_LABELED_WIN_CSV = Path("data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv")
DEFAULT_DELTA_WINPROB_CSV = Path("data/verify/exchange_delta_winprob_step3_2026-08-02/exchange_delta_winprob.csv")
DEFAULT_OUT_DIR = Path("data/verify/exchange_indicator_ablation_2026-08-03")

# exchange_delta_winprob.csv 側の列名 (Step3 の実列名、コメントに合わせて明示)
EVENT_VIDEO_COL: str = "video_id"
EVENT_SIDE_COL: str = "fire_side"
EVENT_T_SEC_COL: str = "t_sec"
EVENT_PROB_COL: str = "stack_prob_taiou_success"
EVENT_DELTA_COL: str = "delta_winprob"
EVENT_MATCH_FAILED_COL: str = "match_failed"

# 非発火時の中立値 (0.5=五分五分、0.0=変化なし)
NEUTRAL_PROB: float = 0.5
NEUTRAL_DELTA: float = 0.0

# 時間減衰の既定 tau (秒) + 感度確認用の3値
DEFAULT_TAU_SEC: float = 10.0
TAU_SENSITIVITY_VALUES: tuple[float, ...] = (10.0, 20.0, 40.0)

# 新指標2列のベース名 (build_features に渡す base 名、suffix無し)
PROB_LAST_BASE_FMT: str = "prob_taiou_success_last__{mode}"
DELTA_LAST_BASE_FMT: str = "delta_winprob_last__{mode}"

# 位相ラベル一覧 (exchange_meter_eval_harness と同一表記、"全体"含む拡張)
SCOPES: tuple[str, ...] = ("全体",) + EXCHANGE_PHASES

# ブートストラップCIの有意判定: 0を跨がない (CI下限>0 or CI上限<0) 場合のみ有意とみなす。
# 「有意」でも符号 (改善/悪化) を必ず区別する (悪化を「条件付き採用推奨」に混ぜない、盛らない)。
CI_STRADDLES_ZERO_MESSAGE: str = "効果を確認できず (CIが0を跨ぐ)"
CI_SIGNIFICANT_IMPROVE_MESSAGE: str = "有意に改善 (CIが0より上)"
CI_SIGNIFICANT_WORSEN_MESSAGE: str = "有意に悪化 (CIが0より下)"


# =============================================================================
# 1. 直近発火イベントの読込 + ルックアップテーブル構築
# =============================================================================

def load_delta_winprob_events(path: Path) -> pd.DataFrame:
    """突合成功イベントのみ抽出し (video_id, side, t_sec, prob, delta) に整形する。"""
    df = pd.read_csv(path)
    df = df.loc[~df[EVENT_MATCH_FAILED_COL].astype(bool)].copy()
    out = pd.DataFrame({
        "video_id": df[EVENT_VIDEO_COL].astype(str).values,
        "side": df[EVENT_SIDE_COL].astype(str).values,
        "t_sec": df[EVENT_T_SEC_COL].astype(float).values,
        "prob": df[EVENT_PROB_COL].astype(float).values,
        "delta": df[EVENT_DELTA_COL].astype(float).values,
    })
    print(f"[events] 突合成功イベント読込: {len(out)}/{len(pd.read_csv(path))} 行"
          f" (動画数={out['video_id'].nunique()})")
    return out


def _build_lookup_tables(events: pd.DataFrame) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(video_id, side) キーで t_sec 昇順ソート済み配列を返す (searchsorted用)。"""
    tables: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (vid, side), g in events.groupby(["video_id", "side"], sort=False):
        g_sorted = g.sort_values("t_sec")
        tables[(vid, side)] = (
            g_sorted["t_sec"].values, g_sorted["prob"].values, g_sorted["delta"].values,
        )
    return tables


def lookup_last_events(
    video_ids: np.ndarray, sides: np.ndarray, t_secs: np.ndarray, events: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """各行について「その時刻より前で最も新しい同一 video_id・同side イベント」を返す。

    未来のイベントは絶対に参照しない (searchsorted の backward-only 突合)。
    該当イベントが無い行は NaN を返す (呼び出し側で既定値処理する)。
    """
    tables = _build_lookup_tables(events)
    n = len(video_ids)
    gap = np.full(n, np.nan)
    prob = np.full(n, np.nan)
    delta = np.full(n, np.nan)
    df_idx = pd.DataFrame({"video_id": video_ids, "side": sides, "t_sec": t_secs})
    for (vid, side), g in df_idx.groupby(["video_id", "side"], sort=False):
        key = (vid, side)
        if key not in tables:
            continue
        ev_t, ev_prob, ev_delta = tables[key]
        idx = np.searchsorted(ev_t, g["t_sec"].values, side="right") - 1
        found = idx >= 0
        rows = g.index.values
        gap[rows[found]] = g["t_sec"].values[found] - ev_t[idx[found]]
        prob[rows[found]] = ev_prob[idx[found]]
        delta[rows[found]] = ev_delta[idx[found]]
    return gap, prob, delta


# =============================================================================
# 2. 直近発火イベント特徴量の合成 (neutral方式 + decay方式)
# =============================================================================

def _decay_weight(gap_sec: np.ndarray, tau_sec: float) -> np.ndarray:
    """weight=exp(-gap/tau) を返す (イベント無し行は呼び出し側で0に上書きされる前提)。"""
    return np.exp(-np.clip(gap_sec, 0.0, None) / tau_sec)


def attach_last_event_features(
    df: pd.DataFrame, events: pd.DataFrame, tau_values: tuple[float, ...],
) -> pd.DataFrame:
    """labeled_win 生データに neutral方式 + 各tauのdecay方式の列を付与して返す。

    df は video_id/side/t_sec 列を持つこと (load_labeled_csv 出力を想定)。
    追加列は pair_sides_for_win が全列を汎用的にコピーするため、そのまま
    渡せば "_1p"/"_2p" 付きで自動的にペアリングされる (既存パイプライン再利用)。
    """
    gap, prob_raw, delta_raw = lookup_last_events(
        df["video_id"].values, df["side"].values, df["t_sec"].astype(float).values, events,
    )
    has_event = ~np.isnan(gap)
    out = df.copy()

    # (a) neutral: 直近値をそのまま保持、イベント無しなら中立値
    prob_neutral = np.where(has_event, prob_raw, NEUTRAL_PROB)
    delta_neutral = np.where(has_event, delta_raw, NEUTRAL_DELTA)
    out[PROB_LAST_BASE_FMT.format(mode="neutral")] = prob_neutral
    out[DELTA_LAST_BASE_FMT.format(mode="neutral")] = delta_neutral

    # (b) decay: イベント無しは gap=NaN -> weight=0 として中立値に一致させる
    prob_filled = np.where(has_event, prob_raw, NEUTRAL_PROB)
    delta_filled = np.where(has_event, delta_raw, NEUTRAL_DELTA)
    for tau in tau_values:
        weight = np.where(has_event, _decay_weight(gap, tau), 0.0)
        mode = f"decay_tau{int(tau)}"
        out[PROB_LAST_BASE_FMT.format(mode=mode)] = NEUTRAL_PROB + weight * (prob_filled - NEUTRAL_PROB)
        out[DELTA_LAST_BASE_FMT.format(mode=mode)] = NEUTRAL_DELTA + weight * (delta_filled - NEUTRAL_DELTA)
    return out


# =============================================================================
# 3. 位相割当 (tsumo三分位、model_indicator_win.py と同一の考え方を再利用)
# =============================================================================

def assign_phase_by_tsumo_tertile(tsumo_1p: np.ndarray) -> tuple[np.ndarray, float, float]:
    """tsumo_1p (手数) の3分位で 序/中/終 を割り当てる (既存 run_phase_models と同じ境界定義)。"""
    q_low = float(np.quantile(tsumo_1p, TSUMO_EARLY_RATIO))
    q_high = float(np.quantile(tsumo_1p, TSUMO_LATE_RATIO))
    labels = np.full(len(tsumo_1p), "中", dtype=object)
    labels[tsumo_1p <= q_low] = "序"
    labels[tsumo_1p > q_high] = "終"
    return labels, q_low, q_high


# =============================================================================
# 4. 構成別 OOF LR 評価
# =============================================================================

@dataclass(frozen=True)
class ConfigResult:
    """1構成 (指標セット) 分の OOF 評価結果。"""
    name: str
    n_features: int
    oof_proba: np.ndarray  # shape (n,) 陽性確率
    y: np.ndarray
    video_ids: np.ndarray
    phase_labels: np.ndarray


def run_config(
    name: str, paired: pd.DataFrame, indicator_cols: list[str],
    y: np.ndarray, groups: np.ndarray, phase_labels: np.ndarray, n_folds: int,
) -> ConfigResult:
    """1構成分: 特徴量構築 -> GroupKFold OOF LogisticRegression。"""
    feat_df = build_features(paired, indicator_cols)
    X = feat_df.fillna(0.0).values.astype(float)
    oof_proba = run_oof_lr(X, y, groups, n_folds)
    print(f"[config={name}] n_features={X.shape[1]}")
    return ConfigResult(
        name=name, n_features=X.shape[1], oof_proba=oof_proba[:, 1],
        y=y, video_ids=groups, phase_labels=phase_labels,
    )


def _scope_mask(phase_labels: np.ndarray, scope: str) -> np.ndarray:
    """"全体" or 位相ラベルに応じたブールマスクを返す。"""
    if scope == "全体":
        return np.ones(len(phase_labels), dtype=bool)
    return phase_labels == scope


def build_auc_table(results: list[ConfigResult]) -> pd.DataFrame:
    """構成×範囲(全体/序/中/終) の AUC 一覧表を作る。"""
    rows: list[dict] = []
    for res in results:
        for scope in SCOPES:
            mask = _scope_mask(res.phase_labels, scope)
            n = int(mask.sum())
            auc = exact_auc(res.y[mask], res.oof_proba[mask])
            rows.append({
                "構成": res.name, "範囲": scope, "n": n, "OOF_AUC": auc,
                "備考": phase_power_flag(n) if scope != "全体" else "",
            })
    return pd.DataFrame(rows)


def _auc_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """bootstrap_diff_ci_by_video に渡す AUC ラッパー (harness.exact_auc の再利用)。"""
    return exact_auc(y_true, y_pred)


def build_auc_diff_table(
    baseline: ConfigResult, others: list[ConfigResult], n_resamples: int,
) -> pd.DataFrame:
    """baseline vs 各構成の AUC差 (動画クラスタ・ブートストラップCI) を範囲別に作る。"""
    rows: list[dict] = []
    for other in others:
        for scope in SCOPES:
            mask = _scope_mask(baseline.phase_labels, scope)
            if int(mask.sum()) == 0:
                # silent drop 禁止: 件数ゼロも明示的に1行として報告する
                rows.append({
                    "比較": f"{other.name} - {baseline.name}", "範囲": scope,
                    "AUC差(点推定)": float("nan"), "CI_low": float("nan"), "CI_high": float("nan"),
                    "判定": "データなし (n=0)",
                })
                continue
            ci = bootstrap_diff_ci_by_video(
                _auc_metric, baseline.video_ids[mask],
                {"y_true": other.y[mask], "y_pred": other.oof_proba[mask]},
                {"y_true": baseline.y[mask], "y_pred": baseline.oof_proba[mask]},
                n_resamples=n_resamples,
            )
            if ci.ci_low <= 0.0 <= ci.ci_high:
                verdict = CI_STRADDLES_ZERO_MESSAGE
            elif ci.ci_low > 0.0:
                verdict = CI_SIGNIFICANT_IMPROVE_MESSAGE
            else:
                verdict = CI_SIGNIFICANT_WORSEN_MESSAGE
            rows.append({
                "比較": f"{other.name} - {baseline.name}", "範囲": scope,
                "AUC差(点推定)": ci.point, "CI_low": ci.ci_low, "CI_high": ci.ci_high,
                "判定": verdict,
            })
    return pd.DataFrame(rows)


# =============================================================================
# 5. 校正曲線 + markdown レポート出力
# =============================================================================

def build_calibration_tables(results: list[ConfigResult], n_bins: int = 10) -> dict[str, pd.DataFrame]:
    """構成×範囲 の reliability table 辞書を返す (harness.compute_reliability_table 再利用)。"""
    tables: dict[str, pd.DataFrame] = {}
    for res in results:
        for scope in SCOPES:
            mask = _scope_mask(res.phase_labels, scope)
            tables[f"{res.name} / {scope}"] = compute_reliability_table(
                res.y[mask], res.oof_proba[mask], n_bins,
            )
    return tables


def _df_to_markdown(df: pd.DataFrame) -> str:
    """tabulate 非依存の最小 markdown テーブル変換 (harness と同一パターン)。"""
    if df.empty:
        return "(データなし)\n"
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    )
    return f"{header}\n{sep}\n{body}\n"


def write_markdown_report(
    out_dir: Path, auc_table: pd.DataFrame, diff_table_main: pd.DataFrame,
    diff_table_tau: pd.DataFrame, verdict: str,
) -> Path:
    """AUC表・差分CI表・所見を1枚のmarkdownにまとめる。"""
    lines = [
        "# ΔWinProb接続 Step4: 新指標2列アブレーション",
        "",
        "45指標(実測44指標)のみ / +2列(neutral) / +2列(decay,tau=10) の3構成比較。",
        "AUC差はCIが0を跨げば「効果を確認できず」と明記する(盛らない)。",
        "",
        "## 構成×範囲 OOF AUC 一覧",
        _df_to_markdown(auc_table),
        "## baseline比 AUC差 (動画クラスタ・ブートストラップCI)",
        _df_to_markdown(diff_table_main),
        "## tau感度 (decay方式、tau=10/20/40)",
        _df_to_markdown(diff_table_tau),
        "## 所見",
        verdict,
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ablation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _build_verdict(diff_table_main: pd.DataFrame) -> str:
    """全体スコープのCI跨ぎ判定から採用推奨/非推奨/条件付きの所見文を作る (改善/悪化を必ず区別)。"""
    overall = diff_table_main.loc[diff_table_main["範囲"] == "全体"]
    if overall.empty:
        return "判定不能 (全体スコープの行が無い)。"
    any_improve = (overall["判定"] == CI_SIGNIFICANT_IMPROVE_MESSAGE).any()
    any_worsen = (overall["判定"] == CI_SIGNIFICANT_WORSEN_MESSAGE).any()
    if any_worsen and not any_improve:
        return "非推奨: 全体スコープで有意な悪化が確認された構成がある (CIが0より下)。新指標2列は追加すべきでない。"
    if any_improve:
        return "条件付き採用推奨: 全体スコープで有意な改善が確認された構成がある。位相別の内訳・悪化構成の有無を要確認。"
    return "非推奨 (現状): 全体スコープではいずれの構成もCIが0を跨ぎ、効果を確認できなかった。"


# =============================================================================
# 6. メイン
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する (main を50行以内に保つための分割)。"""
    parser = argparse.ArgumentParser(description="ΔWinProb接続 Step4 新指標アブレーション")
    parser.add_argument("--labeled-win-csv", type=Path, default=DEFAULT_LABELED_WIN_CSV)
    parser.add_argument("--delta-winprob-csv", type=Path, default=DEFAULT_DELTA_WINPROB_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--max-tdiff", type=float, default=DEFAULT_MAX_TDIFF)
    return parser.parse_args()


def _build_configs(paired: pd.DataFrame) -> dict[str, list[str]]:
    """構成名 -> indicator_cols (base名リスト) の辞書を作る (main の分割)。"""
    new_bases = {
        PROB_LAST_BASE_FMT.format(mode="neutral"), DELTA_LAST_BASE_FMT.format(mode="neutral"),
        PROB_LAST_BASE_FMT.format(mode="decay_tau10"), DELTA_LAST_BASE_FMT.format(mode="decay_tau10"),
        PROB_LAST_BASE_FMT.format(mode="decay_tau20"), DELTA_LAST_BASE_FMT.format(mode="decay_tau20"),
        PROB_LAST_BASE_FMT.format(mode="decay_tau40"), DELTA_LAST_BASE_FMT.format(mode="decay_tau40"),
    }
    baseline_cols = [c for c in _get_indicator_cols(paired) if c not in new_bases]
    configs = {"baseline_44": baseline_cols}
    for mode in ("neutral", "decay_tau10", "decay_tau20", "decay_tau40"):
        configs[f"plus2_{mode}"] = baseline_cols + [
            PROB_LAST_BASE_FMT.format(mode=mode), DELTA_LAST_BASE_FMT.format(mode=mode),
        ]
    return configs


def main() -> None:
    args = _parse_args()
    print("=== 1. データ読込 + 直近発火イベント特徴量付与 ===")
    raw = load_labeled_csv(str(args.labeled_win_csv))
    events = load_delta_winprob_events(args.delta_winprob_csv)
    raw_with_events = attach_last_event_features(raw, events, TAU_SENSITIVITY_VALUES)

    print("\n=== 2. 1P/2P ペアリング ===")
    paired = pair_sides_for_win(raw_with_events, args.max_tdiff)
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].values
    phase_labels, q_low, q_high = assign_phase_by_tsumo_tertile(paired["tsumo_1p"].astype(float).values)
    print(f"[phase] 手数境界: 序<={q_low:.1f} 終>{q_high:.1f}")

    print("\n=== 3. 構成別 GroupKFold OOF LogisticRegression ===")
    configs = _build_configs(paired)
    results = {
        name: run_config(name, paired, cols, y, groups, phase_labels, args.n_folds)
        for name, cols in configs.items()
    }

    print("\n=== 4. AUC表 + ブートストラップCI ===")
    auc_table = build_auc_table(list(results.values()))
    main_others = [results["plus2_neutral"], results["plus2_decay_tau10"]]
    diff_main = build_auc_diff_table(results["baseline_44"], main_others, args.n_bootstrap)
    tau_others = [results["plus2_decay_tau20"], results["plus2_decay_tau40"]]
    diff_tau = build_auc_diff_table(results["plus2_decay_tau10"], tau_others, args.n_bootstrap)
    verdict = _build_verdict(diff_main)
    print(diff_main.to_string())
    print(f"\n[所見] {verdict}")

    print("\n=== 5. 校正図 + markdown レポート出力 ===")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    main_results = [results["baseline_44"], results["plus2_neutral"], results["plus2_decay_tau10"]]
    calib_tables = build_calibration_tables(main_results)
    plot_reliability_diagrams(calib_tables, args.out_dir / "calibration_curves.png")
    report_path = write_markdown_report(args.out_dir, auc_table, diff_main, diff_tau, verdict)
    print(f"[main] レポート保存: {report_path}")
    print(f"出力先: {args.out_dir}")
    print("=== 完了 ===")


if __name__ == "__main__":
    main()
