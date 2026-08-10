"""アブレーション1: near_future_fire_power の K=1,2 (実ツモのみ) を
「ビーム」と「実ツモ厳密全探索」の2通りで計算し、win-AUC を横並び比較する。

## 背景
near_future_fire_power (src/indicators_v2.py) はビームサーチ (既定幅8) で
K=1..5 を近似する。ビーム値でも win-AUC 中盤+0.12〜0.17 が観測されているが、
ビーム幅8では2手目 (dnext) 展開時に1手目 (next) 22配置中トップ8にしか
展開しないため、取りこぼしが生じうる (scripts/_verify_beam_miss_fair_2026-08-09.py
の理想ツモ版で実測87.5%取りこぼし)。本スクリプトは**実ツモのみ**
(next 22通り・next+dnext 484通り) で、ビーム値と厳密全探索値を並べ、
win-AUC が変わるかを確認する (投資判断の材料作り)。

## K の定義 (注意: near_future_fire_power 本体の k_levels とは意味が違う)
- K=1: next のみを置いた場合の最大得点 (22配置)
- K=2: next→(発火後の残骸に)dnext を置いた場合の最大得点 (最大22×22=484配置)
  ※ 得点は「その手番の単発点火」であり、複数手の点数を合算しない
    (near_future_fire_power の checkpoint 定義 = 実行中の最大値 と同じ規約)。
  ※ ideal (理想ツモ) 代用のフォールバックは使わない。next/dnext のどちらかが
    実色でない行は K1/K2 とも計算せず除外する (分母つきで報告)。

## 手法
1. `--mode scan`  : 全npzの next/dnext 有効性を軽量スキャン (simulate 呼び出しなし)。
2. `--mode bench` : 1盤面あたりの厳密/ビームの実測msを計測し全体見積もりを出す。
3. `--mode extract`: (層化サンプル可) board 再構成 + current_max_chain/beam/exact
   を計算し、per_row_values.tsv に保存する。
4. `--mode auc`   : per_row_values.tsv から 1P/2P ペアリング + GroupKFold(video_id)
   OOF LogisticRegression で3構成 (current_max_chain のみ / +beam K1,K2 / +exact
   K1,K2) の win-AUC を比較し、動画クラスタ・ブートストラップCIで差を判定する。

## 使う既存資産 (車輪の再発明禁止)
- src/indicators_v2.py: _enumerate_placement_boards, _near_future_known_expand,
  _near_future_is_valid_pair, current_max_chain, NEAR_FUTURE_BEAM_WIDTH,
  NEAR_FUTURE_FIRE_NORM, _clamp01 (いずれも import のみ、無改変)
- src/chain.py: ChainSimulator (import のみ、無改変)
- scripts/model_indicator_win.py: pair_sides_for_win, run_oof_lr, DEFAULT_MAX_TDIFF,
  N_FOLDS, TSUMO_EARLY_RATIO, TSUMO_LATE_RATIO
- scripts/ablate_exchange_indicators.py: ConfigResult, run_config, build_auc_table,
  build_auc_diff_table, assign_phase_by_tsumo_tertile, SCOPES
- scripts/exchange_meter_eval_harness.py: exact_auc, bootstrap_diff_ci_by_video,
  N_BOOTSTRAP_RESAMPLES, phase_power_flag

## 注意 (データ制約)
- boards_lean_phase_l npz には「手数(tsumo)」列が無い。本スクリプトは
  (video_id, game_idx, side) 内の frame_idx 昇順順位を手数プロキシとして使う
  (真の手数ではない近似、AUC比較のための位相三分位分割にのみ使用)。
- 実行はまだしない (本走行は main が WSL detach で出す)。本ファイルは
  スクリプト完成 + 小規模スモークまでを担当する。

## 使い方 (スモーク例)
    PYTHONPATH=. python -m scripts._ablate_exact_k12_2026-08-09 \
        --mode bench --limit-files 1
    PYTHONPATH=. python -m scripts._ablate_exact_k12_2026-08-09 \
        --mode extract --limit-files 1 --sample-per-group 50
    PYTHONPATH=. python -m scripts._ablate_exact_k12_2026-08-09 \
        --mode auc
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import Board
from src.chain import ChainSimulator
from src.console_init import init_console

init_console()
import src.indicators_v2 as iv
from src.scoring import OJAMA_RATE_STANDARD, calculate_chain_score

from scripts.ablate_exchange_indicators import (
    ConfigResult,
    SCOPES,
    assign_phase_by_tsumo_tertile,
    build_auc_diff_table,
    build_auc_table,
    run_config,
)
from scripts.model_indicator_win import (
    DEFAULT_MAX_TDIFF,
    N_FOLDS,
    pair_sides_for_win,
)
from scripts.exchange_meter_eval_harness import N_BOOTSTRAP_RESAMPLES

# =============================================================================
# 定数 (マジックナンバー禁止)
# =============================================================================
DEFAULT_NPZ_DIR: Path = _ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-07"
DEFAULT_OUT_DIR: Path = _ROOT / "data" / "verify" / "ablate_exact_k12_2026-08-09"

PER_ROW_TSV: str = "per_row_values.tsv"
SKIP_SUMMARY_TSV: str = "skip_summary.tsv"
AUC_TABLE_TSV: str = "auc_table.tsv"
AUC_DIFF_TSV: str = "auc_diff_ci.tsv"
SCAN_SUMMARY_TSV: str = "scan_summary.tsv"

# IGNITION_TRIAL_COLORS (src/indicators_v2.py) と同一の実色集合 (1〜5)。
REAL_COLORS: tuple[int, ...] = (1, 2, 3, 4, 5)

FIRE_NORM: int = iv.NEAR_FUTURE_FIRE_NORM  # =72、既存火力系指標と同一正規化分母
BEAM_WIDTH: int = iv.NEAR_FUTURE_BEAM_WIDTH  # =8、既定ビーム幅 (near_future_fire_power既定と同一)

SAMPLE_PHASE_BUCKETS: int = 3  # 層化サンプリングの位相バケツ数 (序/中/終近似)
SAMPLE_RANDOM_SEED: int = 42  # 再現性確保
BENCH_N_BOARDS: int = 40  # --mode bench の計測件数


# =============================================================================
# 1. 厳密全探索 / ビーム探索 (K=1,2、実ツモのみ)
# =============================================================================

def _score_and_final(board: Board, sim: ChainSimulator) -> "tuple[float, Board]":
    """盤面を1回シミュレートし、(お邪魔換算raw得点, 発火後残骸盤面) を返す。"""
    result = sim.simulate(board)
    raw = float(calculate_chain_score(result).total_score) / OJAMA_RATE_STANDARD
    return raw, result.final_board


def exact_k1_k2(
    board: Board, next_pair: "tuple[int, int]", dnext_pair: "tuple[int, int]",
    sim: ChainSimulator,
) -> "tuple[float, float, int]":
    """実ツモ next/dnext の厳密全探索で K1 (22通り)・K2 (最大484通り) を求める。

    Returns:
        (k1_raw, k2_raw, n_dnext_sims): raw はお邪魔換算個数。n_dnext_sims は
        dnext 展開で実際に simulate した回数 (性能・取りこぼし検証の参考値)。
    """
    placements1 = iv._enumerate_placement_boards(board, next_pair)
    if not placements1:
        return 0.0, 0.0, 0
    scored1 = [_score_and_final(p, sim) for p in placements1]
    k1_raw = max(raw for raw, _ in scored1)
    best2 = k1_raw
    n_dnext_sims = 0
    for _, final1 in scored1:
        for placed2 in iv._enumerate_placement_boards(final1, dnext_pair):
            raw2, _ = _score_and_final(placed2, sim)
            n_dnext_sims += 1
            if raw2 > best2:
                best2 = raw2
    return k1_raw, best2, n_dnext_sims


def beam_k1_k2(
    board: Board, next_pair: "tuple[int, int]", dnext_pair: "tuple[int, int]",
    sim: ChainSimulator, beam_width: int,
) -> "tuple[float, float]":
    """near_future_fire_power と同じビーム machinery (_near_future_known_expand)
    を「next のみ」「next+dnext」で止めて呼ぶ (既知手のみ、理想ツモ展開なし)。

    _near_future_search と同じ「途中経過の最大値を保持する」規約に従う。
    """
    frontier: "list[tuple[float, Board]]" = [(0.0, board)]
    expanded1 = iv._near_future_known_expand(frontier, next_pair, sim)
    if not expanded1:
        return 0.0, 0.0
    k1_score = expanded1[0][0]
    frontier2 = [(s, b) for s, b, _c in expanded1[:beam_width]]
    expanded2 = iv._near_future_known_expand(frontier2, dnext_pair, sim)
    k2_score = max(k1_score, expanded2[0][0]) if expanded2 else k1_score
    # _near_future_known_expand の score 単位は calculate_chain_score 生値
    # (OJAMA_RATE_STANDARD 未除算) のため、exact_k1_k2 と単位を揃える。
    return k1_score / OJAMA_RATE_STANDARD, k2_score / OJAMA_RATE_STANDARD


# =============================================================================
# 2. 軽量メタデータスキャン (simulate 呼び出しなし)
# =============================================================================

def _valid_pair_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """next1_a/b, dnext_a/b が実色ペアとして使えるかの配列版判定。"""
    return np.isin(a, REAL_COLORS) & np.isin(b, REAL_COLORS)


def scan_metadata(npz_paths: "list[Path]") -> pd.DataFrame:
    """全npzの next/dnext 有効性を軽量スキャンする (Board構成・simulate なし)。"""
    frames: "list[pd.DataFrame]" = []
    for path in npz_paths:
        d = np.load(str(path), allow_pickle=True)
        n = len(d["grids"])
        ok_next = _valid_pair_mask(d["next1_a"], d["next1_b"])
        ok_dnext = _valid_pair_mask(d["dnext_a"], d["dnext_b"])
        frames.append(pd.DataFrame({
            "npz_path": str(path), "row_idx": np.arange(n),
            "video_id": d["video_id"], "side": d["side"],
            "t_sec": d["t_sec"].astype(float), "game_idx": d["game_idx"].astype(int),
            "frame_idx": d["frame_idx"].astype(int), "won": d["won"].astype(float),
            "next_valid": ok_next, "dnext_valid": ok_dnext,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_tsumo_proxy(meta: pd.DataFrame) -> pd.DataFrame:
    """(video_id, game_idx, side) 内の frame_idx 昇順順位を「手数プロキシ」とする。

    boards_lean npz に真の手数(tsumo)累積カウンタ列が無いための近似。
    位相三分位分割 (序/中/終) の代用にのみ使う。
    """
    meta = meta.sort_values(["video_id", "game_idx", "side", "frame_idx"]).copy()
    meta["tsumo"] = meta.groupby(["video_id", "game_idx", "side"]).cumcount()
    return meta


def print_scan_summary(meta: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """スキップ内訳 (next無効/dnext無効/両方有効) を分母つきで表示・保存する。"""
    n_total = len(meta)
    n_next_ok = int(meta["next_valid"].sum())
    n_both_ok = int((meta["next_valid"] & meta["dnext_valid"]).sum())
    summary = pd.DataFrame([{
        "n_total": n_total,
        "n_next_valid": n_next_ok, "next_valid_rate": n_next_ok / n_total if n_total else 0.0,
        "n_both_valid": n_both_ok, "both_valid_rate": n_both_ok / n_total if n_total else 0.0,
        "n_skip_missing_next_or_dnext": n_total - n_both_ok,
    }])
    print(f"[scan] 総行数={n_total} next有効={n_next_ok} "
          f"({n_next_ok/n_total:.1%}) next+dnext両方有効={n_both_ok} "
          f"({n_both_ok/n_total:.1%})")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / SCAN_SUMMARY_TSV, sep="\t", index=False)
    return summary


# =============================================================================
# 3. 層化サンプリング (--sample-per-group>0 で有効化、既定0=全行使用)
# =============================================================================

def stratified_sample(meta_valid: pd.DataFrame, sample_per_group: int, seed: int) -> pd.DataFrame:
    """video_id×位相バケツ(手数プロキシ三分位近似)×side で層化サンプルする。

    sample_per_group<=0 の場合は全行をそのまま返す (silent cap 禁止のため
    既定は無効側、有効時は必ず呼び出し側でサンプル件数を報告すること)。
    """
    if sample_per_group <= 0:
        return meta_valid
    pool = meta_valid.copy()
    pool["_phase_bucket"] = pool.groupby("video_id")["tsumo"].transform(
        lambda s: pd.qcut(s, SAMPLE_PHASE_BUCKETS, labels=False, duplicates="drop")
    )
    # pandas 2.x の groupby().apply() はグループ化列を戻り値から除外することが
    # あるため (video_id が消える事故を実機確認済)、手動ループ+concatで組み立てる。
    rng = np.random.RandomState(seed)
    parts: "list[pd.DataFrame]" = []
    for _, g in pool.groupby(["video_id", "_phase_bucket", "side"]):
        parts.append(g.sample(min(len(g), sample_per_group), random_state=rng))
    sampled = pd.concat(parts, ignore_index=True) if parts else pool.iloc[0:0]
    return sampled.drop(columns=["_phase_bucket"])


# =============================================================================
# 4. 本計算 (board再構成 + current_max_chain/beam/exact)
# =============================================================================

def _build_row_record(
    meta_row: "pd.Series", board: Board, next_pair: "tuple[int, int]",
    dnext_pair: "tuple[int, int]", sim: ChainSimulator, compute_exact: bool,
) -> dict:
    """1行分の指標値レコードを組み立てる (0〜1正規化スコア + raw個数を両方保存)。"""
    cmc = iv.current_max_chain(board, sim)
    beam_k1_raw, beam_k2_raw = beam_k1_k2(board, next_pair, dnext_pair, sim, BEAM_WIDTH)
    record: dict = {
        "video_id": meta_row["video_id"], "side": meta_row["side"],
        "t_sec": meta_row["t_sec"], "game_idx": meta_row["game_idx"],
        "tsumo": meta_row["tsumo"], "won": meta_row["won"],
        "current_max_chain": cmc.score, "current_max_chain_raw": cmc.raw,
        "beam_k1": iv._clamp01(beam_k1_raw / FIRE_NORM), "beam_k1_raw": beam_k1_raw,
        "beam_k2": iv._clamp01(beam_k2_raw / FIRE_NORM), "beam_k2_raw": beam_k2_raw,
    }
    if compute_exact:
        exact_k1_raw, exact_k2_raw, n_sims = exact_k1_k2(board, next_pair, dnext_pair, sim)
        record.update({
            "exact_k1": iv._clamp01(exact_k1_raw / FIRE_NORM), "exact_k1_raw": exact_k1_raw,
            "exact_k2": iv._clamp01(exact_k2_raw / FIRE_NORM), "exact_k2_raw": exact_k2_raw,
            "n_dnext_sims": n_sims,
        })
    return record


def extract_indicator_values(sample: pd.DataFrame, compute_exact: bool) -> pd.DataFrame:
    """サンプル済み行に対し board 再構成 + 指標計算を行う (npz毎にまとめて再読込)。"""
    sim = ChainSimulator()
    out_rows: "list[dict]" = []
    n_dead, n_no_placement = 0, 0
    for npz_path, group in sample.groupby("npz_path"):
        d = np.load(npz_path, allow_pickle=True)
        grids = d["grids"]
        na, nb, da, db = d["next1_a"], d["next1_b"], d["dnext_a"], d["dnext_b"]
        for _, row in group.iterrows():
            i = int(row["row_idx"])
            board = Board.from_list(grids[i].tolist())
            if board.is_dead():
                n_dead += 1
                continue
            next_pair = (int(na[i]), int(nb[i]))
            dnext_pair = (int(da[i]), int(db[i]))
            if not iv._enumerate_placement_boards(board, next_pair):
                n_no_placement += 1
                continue
            out_rows.append(_build_row_record(row, board, next_pair, dnext_pair, sim, compute_exact))
    print(f"[extract] 窒息盤面スキップ={n_dead}  next配置不能スキップ={n_no_placement}  "
          f"抽出成功={len(out_rows)}")
    return pd.DataFrame(out_rows)


# =============================================================================
# 5. AUC比較 (a: current_max_chain / b: +beam K1,K2 / c: +exact K1,K2)
# =============================================================================

def _build_configs() -> "dict[str, list[str]]":
    """3構成の特徴量列 (base名) を返す。"""
    return {
        "a_baseline_cmc": ["current_max_chain"],
        "b_plus_beam_k12": ["current_max_chain", "beam_k1", "beam_k2"],
        "c_plus_exact_k12": ["current_max_chain", "exact_k1", "exact_k2"],
    }


def run_auc_comparison(
    per_row_df: pd.DataFrame, max_tdiff: float, n_folds: int, n_bootstrap: int,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """1P/2Pペアリング -> 3構成 GroupKFold OOF LR -> AUC表 + 差分CI表を返す。"""
    paired = pair_sides_for_win(per_row_df, max_tdiff)
    if len(paired) == 0:
        raise ValueError("1P/2Pペアが成立しなかった (--max-tdiff を確認)")
    y = paired["won_1p"].astype(int).values
    groups = paired["video_id_1p"].values
    phase_labels, q_low, q_high = assign_phase_by_tsumo_tertile(
        paired["tsumo_1p"].astype(float).values
    )
    print(f"[auc] 手数プロキシ境界(近似): 序<={q_low:.1f} 終>{q_high:.1f}")
    configs = _build_configs()
    results = {
        name: run_config(name, paired, cols, y, groups, phase_labels, n_folds)
        for name, cols in configs.items()
    }
    auc_table = build_auc_table(list(results.values()))
    diff_ba = build_auc_diff_table(
        results["a_baseline_cmc"],
        [results["b_plus_beam_k12"], results["c_plus_exact_k12"]],
        n_bootstrap,
    )
    diff_cb = build_auc_diff_table(
        results["b_plus_beam_k12"], [results["c_plus_exact_k12"]], n_bootstrap,
    )
    diff_table = pd.concat([diff_ba, diff_cb], ignore_index=True)
    return auc_table, diff_table


# =============================================================================
# 6. ベンチマーク (1盤面あたり実測ms + 全体見積もり)
# =============================================================================

def _pick_bench_rows(meta_valid: pd.DataFrame, n_boards: int) -> pd.DataFrame:
    """ベンチ計測用に最初のnpzから両方有効な行を先頭からn_boards件取る。"""
    first_path = meta_valid["npz_path"].iloc[0]
    return meta_valid[meta_valid["npz_path"] == first_path].head(n_boards)


def run_bench(npz_dir: Path, n_boards: int) -> None:
    """厳密/ビームの1盤面あたり実測msを計測し、全npz見積もりを表示する。"""
    paths = sorted(npz_dir.glob("*.npz"))
    if not paths:
        print(f"[bench] npzファイルが見つからない: {npz_dir}")
        return
    meta = scan_metadata(paths[:1])
    meta_valid = meta[meta["next_valid"] & meta["dnext_valid"]]
    bench_rows = _pick_bench_rows(meta_valid, n_boards)
    sim = ChainSimulator()
    t_exact = _time_exact(bench_rows, sim)
    t_beam = _time_beam(bench_rows, sim)
    full_meta = scan_metadata(paths)
    n_full_valid = int((full_meta["next_valid"] & full_meta["dnext_valid"]).sum())
    _print_bench_report(t_exact, t_beam, len(bench_rows), n_full_valid, len(paths))


def _time_exact(bench_rows: pd.DataFrame, sim: ChainSimulator) -> float:
    """厳密探索の平均秒/盤面を計測する。"""
    d = np.load(bench_rows["npz_path"].iloc[0], allow_pickle=True)
    grids, na, nb, da, db = d["grids"], d["next1_a"], d["next1_b"], d["dnext_a"], d["dnext_b"]
    t0 = time.perf_counter()
    for _, row in bench_rows.iterrows():
        i = int(row["row_idx"])
        board = Board.from_list(grids[i].tolist())
        exact_k1_k2(board, (int(na[i]), int(nb[i])), (int(da[i]), int(db[i])), sim)
    return (time.perf_counter() - t0) / len(bench_rows)


def _time_beam(bench_rows: pd.DataFrame, sim: ChainSimulator) -> float:
    """ビーム探索の平均秒/盤面を計測する。"""
    d = np.load(bench_rows["npz_path"].iloc[0], allow_pickle=True)
    grids, na, nb, da, db = d["grids"], d["next1_a"], d["next1_b"], d["dnext_a"], d["dnext_b"]
    t0 = time.perf_counter()
    for _, row in bench_rows.iterrows():
        i = int(row["row_idx"])
        board = Board.from_list(grids[i].tolist())
        beam_k1_k2(board, (int(na[i]), int(nb[i])), (int(da[i]), int(db[i])), sim, BEAM_WIDTH)
    return (time.perf_counter() - t0) / len(bench_rows)


def _print_bench_report(
    t_exact_sec: float, t_beam_sec: float, n_bench: int, n_full_valid: int, n_files_total: int,
) -> None:
    """実測ms/盤面 + 全npz全体の見積もり時間 (直列/14並列) を表示する。"""
    print(f"[bench] 計測件数={n_bench}  厳密={t_exact_sec*1000:.2f}ms/盤面  "
          f"ビーム={t_beam_sec*1000:.2f}ms/盤面")
    est_exact_sec = t_exact_sec * n_full_valid
    est_beam_sec = t_beam_sec * n_full_valid
    print(f"[bench] 全{n_files_total}ファイル中 next+dnext両方有効={n_full_valid}行 (推定)")
    print(f"[bench] 厳密 全量見積り: 直列={est_exact_sec/3600:.2f}h  "
          f"14並列={est_exact_sec/3600/14*60:.1f}分")
    print(f"[bench] ビーム全量見積り: 直列={est_beam_sec/3600:.2f}h  "
          f"14並列={est_beam_sec/3600/14*60:.1f}分")


# =============================================================================
# 7. CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する。"""
    parser = argparse.ArgumentParser(description="アブレーション1: near_future K1,K2 beam vs exact")
    parser.add_argument("--mode", choices=["scan", "bench", "extract", "auc"], required=True)
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit-files", type=int, default=0, help="0=全ファイル")
    parser.add_argument("--sample-per-group", type=int, default=0, help="0=サンプリングなし(全行)")
    parser.add_argument("--skip-exact", action="store_true", help="ビームのみ計算(厳密は計算しない)")
    parser.add_argument("--bench-n-boards", type=int, default=BENCH_N_BOARDS)
    parser.add_argument("--max-tdiff", type=float, default=DEFAULT_MAX_TDIFF)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_RESAMPLES)
    return parser.parse_args()


def _select_npz_paths(npz_dir: Path, limit_files: int) -> "list[Path]":
    """処理対象npzファイル一覧を返す (limit_files>0ならスモーク用に先頭N件)。"""
    paths = sorted(npz_dir.glob("*.npz"))
    return paths[:limit_files] if limit_files > 0 else paths


def _run_extract_mode(args: argparse.Namespace) -> None:
    """--mode extract: スキャン -> (層化)サンプル -> 抽出 -> per_row_values.tsv 保存。"""
    paths = _select_npz_paths(args.npz_dir, args.limit_files)
    meta = add_tsumo_proxy(scan_metadata(paths))
    print_scan_summary(meta, args.out_dir)
    meta_valid = meta[meta["next_valid"] & meta["dnext_valid"]]
    sample = stratified_sample(meta_valid, args.sample_per_group, SAMPLE_RANDOM_SEED)
    print(f"[extract] 抽出対象行数={len(sample)} "
          f"(sample_per_group={args.sample_per_group}, 有効行母集団={len(meta_valid)})")
    per_row_df = extract_indicator_values(sample, compute_exact=not args.skip_exact)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / PER_ROW_TSV
    per_row_df.to_csv(out_path, sep="\t", index=False)
    print(f"[extract] 保存: {out_path} ({len(per_row_df)}行)")


def _run_auc_mode(args: argparse.Namespace) -> None:
    """--mode auc: per_row_values.tsv を読み込み AUC比較を実行・保存する。"""
    in_path = args.out_dir / PER_ROW_TSV
    per_row_df = pd.read_csv(in_path, sep="\t")
    if "exact_k1" not in per_row_df.columns:
        raise ValueError(
            f"{in_path} に exact_k1 列が無い (--skip-exact で抽出した可能性)。"
            " exact有りで --mode extract をやり直すこと。"
        )
    auc_table, diff_table = run_auc_comparison(
        per_row_df, args.max_tdiff, args.n_folds, args.n_bootstrap,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    auc_table.to_csv(args.out_dir / AUC_TABLE_TSV, sep="\t", index=False)
    diff_table.to_csv(args.out_dir / AUC_DIFF_TSV, sep="\t", index=False)
    print(auc_table.to_string())
    print(diff_table.to_string())


def main() -> int:
    args = _parse_args()
    if args.mode == "scan":
        paths = _select_npz_paths(args.npz_dir, args.limit_files)
        print_scan_summary(add_tsumo_proxy(scan_metadata(paths)), args.out_dir)
    elif args.mode == "bench":
        run_bench(args.npz_dir, args.bench_n_boards)
    elif args.mode == "extract":
        _run_extract_mode(args)
    elif args.mode == "auc":
        _run_auc_mode(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
