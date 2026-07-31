"""v2プローブ: 「有効な催促保持」(effective_saisoku_hold) の安価先行検証。

催促モデル条件1(相手が催促を返せる見込み)を関係的に測る新指標の
本実装前プローブ。v1 `saisoku_hold` (自分側保持のみ) は win-AUC 信号なし
(`data/indicators_v2/study/saisoku_hold_auc.csv`, 全体AUC=0.487) と判明済み。

定義 (user 2026-07-22 依頼準拠、プローブ簡略化あり):
    1. 自分の催促候補 (v1 `saisoku_hold` の 0〜1手ヒット、
       `src.indicators_v2._saisoku_hold_hits`/`_saisoku_hold_eval` を再利用) の中で
       最大送りお邪魔のものを「代表催促」とする。
    2. 着弾までの時間窓 W = `chain_to_time(代表催促の連鎖数)`、
       相手が置ける手数 K = floor(W / SEC_PER_HAND) (1〜MC_MAX_HANDS にクランプ)。
    3. 相手盤面 (boards_lean_fixed を t_sec で直近過去ペアリング、
       `scripts.prescreen_boardsim_auc._find_nearest_past_grid` 再利用) に
       K手ぶんの4色等確率ランダムぷよを greedy に積み、各手 simulate。
       降ってくるお邪魔 (代表催促のお邪魔数) 以上を相殺できる連鎖に
       到達した試行の割合 = P_return (モンテカルロ、既定 N=15 試行)。
    4. 「有効な催促」 = saisoku_hold_flag==1 かつ P_return <= 0.5。

割り切り (プローブなので):
    - ネクストは lean_fixed に無いため MC は全て4色等確率 (既知ネクスト不使用)。
    - greedy 積みは「発火で足りれば即返す、足りなければ非発火の中で
      最も安全 (列高さ最小) な配置を選ぶ」簡易ヒューリスティック。
    - コスト削減のため saisoku_hold_flag==1 の行のみ MC 対象
      (flag==0 は effective=0 が自明なので MC 不要)。
    - コストゲート必須: 全量 (7895行) 前に --cost-gate でサンプル実測すること。

使い方:
    PYTHONPATH=. python -m scripts.proto_effective_saisoku --cost-gate
    PYTHONPATH=. python -m scripts.proto_effective_saisoku --sample-n 1500
    PYTHONPATH=. python -m scripts.proto_effective_saisoku --full

出力:
    data/indicators_v2/study/effective_saisoku_features.csv (行毎の指標値)
    data/indicators_v2/study/effective_saisoku_auc.csv       (AUC 結果表)
    data/indicators_v2/study/effective_saisoku_cost_gate.csv (コストゲート実測ログ)
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# スレッド制限 (熱暴走防止、feedback_thermal_safety_mandatory 準拠)
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "2")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import (  # noqa: E402
    Board, COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW,
)
from src.chain import ChainSimulator  # noqa: E402
from src.scoring import OJAMA_RATE_STANDARD, calculate_chain_score, score_to_ojama  # noqa: E402
from src.indicators_v2 import (  # noqa: E402
    SAISOKU_CONSUME_RATIO, SAISOKU_OJAMA_MIN, SEC_PER_HAND,
    _count_color_puyos, _enumerate_placements, _saisoku_hold_eval,
    _saisoku_hold_hits, chain_to_time, max_column_height,
)
from scripts.measure_saisoku_hold_auc import _grid_to_board  # noqa: E402  (ojama保持版を再利用)
from scripts.prescreen_boardsim_auc import _find_nearest_past_grid  # noqa: E402
from scripts.prescreen_candidates import (  # noqa: E402
    PHASE_ALL, PHASE_EARLY, PHASE_LATE, PHASE_MID, univariate_auc,
)

# ============================
# 定数
# ============================

NPZ_DIR: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed"
OUT_DIR: Path = PROJ_ROOT / "data" / "indicators_v2" / "study"
FEATURES_V1_CSV: Path = OUT_DIR / "saisoku_hold_features.csv"
FEATURES_V2_CSV: Path = OUT_DIR / "effective_saisoku_features.csv"
AUC_CSV: Path = OUT_DIR / "effective_saisoku_auc.csv"
COST_LOG_CSV: Path = OUT_DIR / "effective_saisoku_cost_gate.csv"

# 4色等確率 (プローブ簡略化: 既知ネクスト不使用、紫は除外)
FOUR_COLORS: tuple[int, ...] = (COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW)

# モンテカルロ試行数 (プローブ用、暫定・コストゲート結果次第で調整)
MC_N_TRIALS: int = 15
# K手の上限キャップ (時間窓が長すぎる場合の打ち切り)
MC_MAX_HANDS: int = 4
# 有効な催促の判定閾値 (P_return <= この値)
P_RETURN_THRESHOLD: float = 0.5

# コストゲートのデフォルトサンプル数
DEFAULT_COST_GATE_N: int = 300
RANDOM_SEED: int = 42

TIER_CHALLENGER: tuple[str, ...] = (
    "c5", "c6", "c7", "c11", "c16", "c21", "c22", "c28", "c30", "c31",
)
TIER_MASTER: tuple[str, ...] = (
    "c44", "c51", "c53", "c54", "c59", "c62", "c68", "c73", "c78", "c80",
)
TIER_S_CLASS: tuple[str, ...] = ("c82", "c83", "c84")
ALL_VIDEOS: tuple[str, ...] = TIER_CHALLENGER + TIER_MASTER + TIER_S_CLASS


# ============================
# セクション1: 1手窓の推定
# ============================


def _hands_window(chain_count: float) -> int:
    """代表催促の連鎖数から相手が置ける手数 K (1〜MC_MAX_HANDS) を返す。"""
    window = chain_to_time(chain_count)
    hands = int(math.floor(window / SEC_PER_HAND))
    return max(1, min(MC_MAX_HANDS, hands))


# ============================
# セクション2: モンテカルロ返し確率
# ============================


def _mc_trial(
    opp_board: Board, required_ojama: int, k_hands: int,
    sim: ChainSimulator, rng: random.Random,
) -> bool:
    """1 MC 試行: K手ぶんランダム4色ぷよを greedy に積み、返せたか判定する。

    greedy 方針: 発火で required_ojama 以上賄えれば即返す。
    足りなければ非発火 (chain=0) 候補のうち最も列高さが低い (安全な) ものを選ぶ。
    非発火候補が無ければ (強制発火) 最良連鎖を採用して続行する。
    """
    board = opp_board.copy()
    for _ in range(k_hands):
        if board.is_dead():
            return False
        pair = (rng.choice(FOUR_COLORS), rng.choice(FOUR_COLORS))
        candidates = _enumerate_placements(board, pair, sim)
        if not candidates:
            return False  # 全列満杯で置けない = 返せない
        best_chain, best_placed = candidates[0]
        if best_chain > 0:
            result = sim.simulate(best_placed)
            score = calculate_chain_score(result).total_score
            ojama = score_to_ojama(
                score=score, prev_leftover=0, elapsed_sec=0.0,
                rate_base=OJAMA_RATE_STANDARD,
            ).ojama_count
            if ojama >= required_ojama:
                return True
        zero_candidates = [(c, b) for c, b in candidates if c == 0]
        if zero_candidates:
            _, board = min(
                zero_candidates, key=lambda cb: max_column_height(cb[1]).raw,
            )
        else:
            # 非発火候補が無い (強制発火): 最良連鎖を採用し発火後盤面へ継続。
            board = sim.simulate(best_placed).final_board if best_chain > 0 else best_placed
    return False


def _effective_probe_for_row(
    self_board: Board, opp_board: "Board | None",
    sim: ChainSimulator, rng: random.Random,
) -> "dict[str, float] | None":
    """1盤面分の (required_ojama, chain_count, k_hands, p_return) を計算する。

    saisoku_hold の該当候補が無い場合は None (呼び出し側は flag==1 のみ渡す前提)。
    opp_board が None (ペアリング失敗) の場合 p_return は NaN。
    """
    color_count = _count_color_puyos(self_board)
    hits = _saisoku_hold_hits(self_board, sim)
    matched: list[tuple[int, int]] = []  # (ojama, chain_count)
    for result in hits:
        ojama, consume_ratio = _saisoku_hold_eval(color_count, result)
        if consume_ratio < SAISOKU_CONSUME_RATIO and ojama > SAISOKU_OJAMA_MIN:
            matched.append((ojama, result.chain_count))
    if not matched:
        return None
    required_ojama, chain_count = max(matched, key=lambda x: x[0])
    k_hands = _hands_window(float(chain_count))
    if opp_board is None:
        return {
            "required_ojama": float(required_ojama), "chain_count": float(chain_count),
            "k_hands": float(k_hands), "p_return": float("nan"),
        }
    successes = sum(
        _mc_trial(opp_board, required_ojama, k_hands, sim, rng)
        for _ in range(MC_N_TRIALS)
    )
    p_return = successes / float(MC_N_TRIALS)
    return {
        "required_ojama": float(required_ojama), "chain_count": float(chain_count),
        "k_hands": float(k_hands), "p_return": p_return,
    }


# ============================
# セクション3: バッチ処理
# ============================


def _load_v1_features_with_pos() -> pd.DataFrame:
    """v1 features CSV を読み、動画内 npz 位置 (npz_pos) 列を付与する。"""
    df = pd.read_csv(FEATURES_V1_CSV)
    df["npz_pos"] = df.groupby("video_id").cumcount()
    return df


def compute_v2_for_sample(
    sample_df: pd.DataFrame, seed: int, log_every: int = 50,
) -> pd.DataFrame:
    """flag==1 サンプル行について MC 指標を計算し、元の index 付き DataFrame を返す。"""
    sim = ChainSimulator()
    rng = random.Random(seed)
    npz_cache: dict[str, dict[str, np.ndarray]] = {}
    rows_out: list[dict] = []
    orig_indices: list[int] = []
    total = len(sample_df)
    t0 = time.time()

    for i, (idx, row) in enumerate(sample_df.iterrows()):
        vid = str(row["video_id"])
        pos = int(row["npz_pos"])
        if vid not in npz_cache:
            npz = np.load(NPZ_DIR / f"{vid}.npz")
            npz_cache[vid] = {
                "side": npz["side"], "game_idx": npz["game_idx"],
                "t_sec": npz["t_sec"].astype(float), "grids": npz["grids"],
            }
        vdata = npz_cache[vid]
        self_grid = vdata["grids"][pos]
        self_side = str(vdata["side"][pos])
        game_idx = int(vdata["game_idx"][pos])
        t_sec = float(vdata["t_sec"][pos])
        opp_side = "2P" if self_side == "1P" else "1P"

        self_board = _grid_to_board(self_grid)
        opp_grid, opp_t_diff = _find_nearest_past_grid(vdata, opp_side, game_idx, t_sec)
        opp_board = _grid_to_board(opp_grid) if opp_grid is not None else None

        probe = _effective_probe_for_row(self_board, opp_board, sim, rng)
        rec: dict[str, float] = {"opp_t_diff_sec": opp_t_diff}
        if probe is None:
            rec.update({"required_ojama": float("nan"), "chain_count": float("nan"),
                        "k_hands": float("nan"), "p_return": float("nan")})
        else:
            rec.update(probe)
        rows_out.append(rec)
        orig_indices.append(idx)

        if (i + 1) % log_every == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            per_row_ms = elapsed / (i + 1) * 1000.0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"[{i + 1}/{total}] elapsed={elapsed:.1f}s "
                  f"per_row={per_row_ms:.1f}ms eta={eta:.1f}s")

    out = pd.DataFrame(rows_out, index=orig_indices)
    return out


def run_cost_gate(n_sample: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """flag==1 行から n_sample 件をランダム抽出し MC 実測時間を計測する。"""
    df_v1 = _load_v1_features_with_pos()
    flag_rows = df_v1[df_v1["saisoku_hold_flag"] == 1.0]
    print(f"[cost-gate] flag==1 行数: {len(flag_rows)} / 全 {len(df_v1)} 行")
    n = min(n_sample, len(flag_rows))
    sample = flag_rows.sample(n=n, random_state=seed)

    t0 = time.time()
    computed = compute_v2_for_sample(sample, seed, log_every=max(1, n // 10))
    elapsed = time.time() - t0

    per_row_sec = elapsed / n
    full_n = len(flag_rows)
    est_full_sec = per_row_sec * full_n
    print(f"\n[cost-gate結果] n={n} 件 elapsed={elapsed:.1f}s "
          f"per_row={per_row_sec * 1000:.1f}ms")
    print(f"[cost-gate結果] flag==1 全量 {full_n} 件の見込み時間: "
          f"{est_full_sec:.0f}秒 (={est_full_sec / 60:.1f}分)")

    log_df = pd.DataFrame([{
        "n_sample": n, "elapsed_sec": elapsed, "per_row_sec": per_row_sec,
        "full_flag1_n": full_n, "est_full_sec": est_full_sec,
        "mc_n_trials": MC_N_TRIALS, "mc_max_hands": MC_MAX_HANDS,
    }])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(COST_LOG_CSV, index=False)
    print(f"[save] {COST_LOG_CSV}")
    return log_df


# ============================
# セクション4: AUC 計算
# ============================


def build_full_v2_features(sample_n: "int | None", seed: int = RANDOM_SEED) -> pd.DataFrame:
    """全行 (flag==0 は自明に0) + サンプル (flag==1) の MC 結果を結合した DataFrame。"""
    df = _load_v1_features_with_pos()
    df["effective_saisoku_hold"] = np.where(df["saisoku_hold_flag"] == 0.0, 0.0, np.nan)
    for col in ("p_return", "required_ojama", "chain_count", "k_hands", "opp_t_diff_sec"):
        df[col] = np.nan

    flag_rows = df[df["saisoku_hold_flag"] == 1.0]
    n = len(flag_rows) if sample_n is None else min(sample_n, len(flag_rows))
    sample = flag_rows if sample_n is None else flag_rows.sample(n=n, random_state=seed)
    print(f"[v2計算] flag==1 行 {len(flag_rows)} 件中 {len(sample)} 件を MC 計算します。")

    computed = compute_v2_for_sample(sample, seed)
    for col in ("p_return", "required_ojama", "chain_count", "k_hands", "opp_t_diff_sec"):
        df.loc[computed.index, col] = computed[col].values
    df.loc[computed.index, "effective_saisoku_hold"] = np.where(
        computed["p_return"] <= P_RETURN_THRESHOLD, 1.0,
        np.where(computed["p_return"].notna(), 0.0, np.nan),
    )
    return df


def _phase_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        PHASE_ALL: pd.Series(True, index=df.index),
        PHASE_EARLY: df["phase"] == "序盤",
        PHASE_MID: df["phase"] == "中盤",
        PHASE_LATE: df["phase"] == "終盤",
    }


def compute_auc_table(df: pd.DataFrame) -> pd.DataFrame:
    """v2 (effective_saisoku_hold) と v1/sub_chain_count/current_max_chain の
    win-AUC を ティア別 + 全体 + 位相別 (video holdout) で比較する。"""
    feature_cols = [
        "effective_saisoku_hold", "saisoku_hold_flag",
        "saisoku_hold_max_ojama", "sub_chain_count", "current_max_chain",
    ]
    y = df["won"].astype(float)
    groups = df["video_id"]
    phase_masks = _phase_masks(df)
    tier_subsets = {
        "全体(23本)": pd.Series(True, index=df.index),
        "チャレンジャー": df["tier"] == "チャレンジャー",
        "マスター": df["tier"] == "マスター",
        "S級": df["tier"] == "S級",
    }
    records = []
    for tier_name, tier_mask in tier_subsets.items():
        for phase_name, phase_mask in phase_masks.items():
            mask = tier_mask & phase_mask
            for col in feature_cols:
                auc = univariate_auc(df[col], y, groups, mask)
                n_valid = int((mask & df[col].notna() & y.notna()).sum())
                records.append({
                    "tier": tier_name, "phase": phase_name,
                    "feature": col, "auc": auc, "n": n_valid,
                })
    return pd.DataFrame(records)


def _report_coverage(df: pd.DataFrame) -> None:
    """opp 被覆率 (flag==1 のうち MC 計算対象 + ペアリング成功) を報告する。"""
    flag1 = df[df["saisoku_hold_flag"] == 1.0]
    n_flag1 = len(flag1)
    n_computed = int(flag1["p_return"].notna().sum() + flag1["p_return"].isna().sum()
                      - flag1["required_ojama"].isna().sum())
    n_attempted = int(flag1["required_ojama"].notna().sum())  # MC対象(サンプル)行数
    n_paired = int(flag1["p_return"].notna().sum())  # うちopp盤面ペアリング成功
    print(f"\n[opp被覆率] flag==1 {n_flag1}行中、MC計算対象(サンプル) {n_attempted}行、"
          f"うちopp盤面ペアリング成功 {n_paired}行 "
          f"({(n_paired / n_attempted * 100.0) if n_attempted else 0.0:.1f}%)。")


def _print_auc_summary(auc_df: pd.DataFrame) -> None:
    print("\n## win-AUC サマリ (全体・全位相)")
    sub = auc_df[auc_df["tier"] == "全体(23本)"]
    piv = sub.pivot(index="feature", columns="phase", values="auc")
    cols = [c for c in [PHASE_ALL, PHASE_EARLY, PHASE_MID, PHASE_LATE] if c in piv.columns]
    print(piv[cols].to_string())

    print("\n## win-AUC サマリ (ティア別・中盤)")
    mid = auc_df[auc_df["phase"] == PHASE_MID]
    piv2 = mid.pivot(index="feature", columns="tier", values="auc")
    print(piv2.to_string())


# ============================
# メイン
# ============================


def main() -> None:
    """エントリポイント: --cost-gate / --sample-n / --full を切り替えて実行する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost-gate", action="store_true",
                         help="小サンプルでコスト実測のみ行う (全量実行しない)")
    parser.add_argument("--cost-gate-n", type=int, default=DEFAULT_COST_GATE_N,
                         help="コストゲートのサンプル数 (既定 %(default)s)")
    parser.add_argument("--sample-n", type=int, default=None,
                         help="flag==1 行からのMC計算対象サンプル数 (未指定なら--fullで全量)")
    parser.add_argument("--full", action="store_true",
                         help="flag==1 全行 (約7895件) を MC 計算する")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.cost_gate:
        print("=== proto_effective_saisoku コストゲート ===")
        run_cost_gate(args.cost_gate_n)
        return

    sample_n = None if args.full else args.sample_n
    if sample_n is None and not args.full:
        print("[ERROR] --cost-gate / --sample-n N / --full のいずれかを指定してください。")
        sys.exit(1)

    print("=== proto_effective_saisoku 本計算開始 ===")
    df = build_full_v2_features(sample_n)
    df.to_csv(FEATURES_V2_CSV, index=False)
    print(f"[save] {FEATURES_V2_CSV} ({len(df)}行)")

    _report_coverage(df)

    print("\n[step] win-AUC 計算中 (video holdout, GroupKFold)...")
    auc_df = compute_auc_table(df)
    auc_df.to_csv(AUC_CSV, index=False)
    print(f"[save] {AUC_CSV}")
    _print_auc_summary(auc_df)

    print("\n=== proto_effective_saisoku 完了 ===")


if __name__ == "__main__":
    main()
