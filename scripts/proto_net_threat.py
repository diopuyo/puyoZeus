"""発火時ネット脅威 (net_threat) の試作検証。

指標定義 (user承認済み設計):
    net_threat = 送りお邪魔量(確定) - 相手の予測相殺量(K手以内)

    - 送りお邪魔量(確定): 発火した瞬間のΔscoreを標準レート(70点/個)で
      お邪魔換算した値。連鎖は自動再生されるため発火時点で確定する。
    - 予測相殺量: 発火から着弾までの猶予(T_guard)内に相手が置ける手数 K を
      見積り、相手盤面の到達連鎖ポテンシャルを K でキャップして近似する。

近似の限界 (要注意・レポート必読):
    1. 本データセットの npz (boards_lean_fixed) には相手のネクスト情報が
       保存されていない (video_id/side/t_sec/game_idx/grids/won/score のみ)。
       そのため「相手が実際に見えているネクストで積む」経路は検証不能。
    2. 代替として `potential_fire_power` (ツモ非依存・任意色/列の greedy 2手
       ビーム探索) を流用し、K手をこの探索の深さ上限にキャップする。
       ただし `potential_fire_power` は実装上 max_add=1(1手)か
       max_add>=2(常に2手固定、3手以上は深化しない)の2値しか区別しない
       (src/indicators_v2.py:973-1018)。よって K>=2 は全て2に丸める。
       この近似は「相手は何でも都合よく積める」上振れ仮定であり、
       実戦のネクスト制約より寛容(＝相殺量を過大評価しがち)な点に注意。

再利用方針: 発火イベント検出・盤面復元・ラベル計算 (opp_buried/taiou_success/
survived/phase分類) は scripts/label_exchange_outcome.py の関数をそのまま
import して使う(車輪の再発明を避ける・ラベル定義を exchange_labels.csv と
完全一致させるため)。マージンタイム経過秒補正 (_game_relative_elapsed) も
同モジュールが正本 (2026-07-21 修正でバグ本体を label_exchange_outcome.py
側に直接修正、本モジュールはそれを import するのみ)。

実行方法:
    python -m scripts.proto_net_threat            # サンプル(間引き)実行
    python -m scripts.proto_net_threat --full      # 全 7960 発火イベント実行
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# スレッド制限 (CLAUDE.md 熱暴走対策)
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.indicators_v2 import (  # noqa: E402
    ON_FIELD_CAP,
    SEC_PER_HAND,
    _clamp01,
    chain_to_time,
    current_max_chain,
    potential_fire_power,
)
from scripts.label_exchange_outcome import (  # noqa: E402
    NPZ_DIR,
    NpzRecord,
    _board_from_grid,
    _classify_phase,
    _compute_opp_buried,
    _compute_taiou_success,
    _delta_to_ojama_standard,
    _detect_fire_events,
    _estimate_puyo_quantiles,
    _game_relative_elapsed,
    _load_npz,
)

# ============================
# 定数定義
# ============================
OUTPUT_CSV = PROJ_ROOT / "data" / "indicators_v2" / "proto_net_threat.csv"
EXCHANGE_LABELS_CSV = PROJ_ROOT / "data" / "indicators_v2" / "exchange_labels.csv"

# サンプル実行時の間引き幅 (7960件のうち 1/SAMPLE_STRIDE を処理)
SAMPLE_STRIDE: int = 10

# K手→potential_fire_power の探索深さ(max_add)キャップ。
# K_raw < この閾値なら1手、それ以上は2手固定 (実装上2手が上限のため)。
ONE_HAND_THRESHOLD: float = 1.5
POTENTIAL_DEPTH_CAP: int = 2

# net_threat の 0-1 正規化分母 (ON_FIELD_CAP=72、収支の半値幅として使用)
NET_THREAT_NORM: float = float(ON_FIELD_CAP)


def _hands_cap(fire_chain: float) -> int:
    """発火連鎖の着弾猶予から相手の手数 K を概算し、探索深さにキャップする。

    T_guard = chain_to_time(fire_chain) + SEC_PER_HAND
    (label_exchange_outcome.py の taiou_success 定義と同一の猶予窓)。
    K_raw = T_guard / SEC_PER_HAND。
    """
    t_guard = chain_to_time(max(1.0, fire_chain)) + SEC_PER_HAND
    k_raw = t_guard / SEC_PER_HAND
    return 1 if k_raw < ONE_HAND_THRESHOLD else POTENTIAL_DEPTH_CAP


def _predicted_counter_ojama(
    opp_board: Board, elapsed_sec_in_game: float, k: int, sim: ChainSimulator,
) -> float:
    """相手が K 手以内に到達できる最大お邪魔量(近似値)。

    ネクスト非依存の potential_fire_power を流用。上振れ近似である点は
    モジュール docstring 参照。

    ⚠️ elapsed_sec_in_game は必ず「試合開始からの経過秒」(マージンタイム計算用)
    を渡すこと。npz の t_sec は動画絶対時刻であり、複数試合を含む動画では
    そのまま渡すとマージンタイム減衰が過剰発生し raw が桁違いに膨張する
    (実データ確認済みバグ、本モジュールでは _game_relative_elapsed で補正)。
    """
    return potential_fire_power(
        opp_board, elapsed_sec=elapsed_sec_in_game, simulator=sim, max_add=k,
    ).raw


def _net_threat(ojama_sent: float, predicted_counter: float) -> tuple[float, float]:
    """net_threat = 送りお邪魔量 - 相手予測相殺量。(raw, 0-1正規化) を返す。"""
    raw = ojama_sent - predicted_counter
    norm = _clamp01((raw + NET_THREAT_NORM) / (2.0 * NET_THREAT_NORM))
    return raw, norm


def _delta_score_at_fire(score: np.ndarray, fi: int) -> int | None:
    """発火フレーム fi の直前有効scoreとの差分を返す(欠損ならNone)。"""
    s_fire = int(score[fi])
    for j in range(fi - 1, -1, -1):
        if score[j] >= 0:
            return s_fire - int(score[j])
    return None


def _nearest_board(
    t_arr: np.ndarray, boards: list[tuple[float, Board]], t: float,
) -> Board:
    """時刻 t に最も近い相手盤面を返す。"""
    idx = int(np.argmin(np.abs(t_arr - t)))
    return boards[idx][1]


def _process_one_event(
    fi: int,
    fire_rec: NpzRecord,
    opp_rec: NpzRecord,
    opp_boards: list[tuple[float, Board]],
    fire_side: str,
    sim: ChainSimulator,
    q_low: float,
    q_high: float,
    game_start_t: float,
) -> dict | None:
    """1発火イベントを処理して net_threat 行を返す(欠損等はNone)。"""
    delta_score = _delta_score_at_fire(fire_rec.score, fi)
    if delta_score is None:
        return None

    fi_board_idx = max(0, fi - 1)
    fire_board = _board_from_grid(fire_rec.grids[fi_board_idx])
    t_fire = float(fire_rec.t_sec[fi])
    elapsed_in_game = _game_relative_elapsed(t_fire, game_start_t)

    fire_chain = current_max_chain(fire_board, sim).raw
    k = _hands_cap(fire_chain)

    opp_board = _nearest_board(opp_rec.t_sec, opp_boards, t_fire)
    ojama_sent = float(_delta_to_ojama_standard(delta_score))
    predicted_counter = _predicted_counter_ojama(opp_board, elapsed_in_game, k, sim)
    net_raw, net_norm = _net_threat(ojama_sent, predicted_counter)

    opp_buried = _compute_opp_buried(t_fire, opp_boards, sim)
    taiou_success, survived = _compute_taiou_success(
        t_fire, max(1.0, fire_chain), opp_rec.t_sec, opp_rec.score, opp_boards,
    )
    phase = _classify_phase(float(fire_board.count_puyos()), q_low, q_high)

    return {
        "video_id": fire_rec.video_id,
        "game_idx": int(fire_rec.game_idx[fi]),
        "t_sec": t_fire,
        "fire_side": fire_side,
        "phase": phase,
        "fire_chain": float(fire_chain),
        "hands_k": k,
        "ojama_sent": ojama_sent,
        "predicted_counter": predicted_counter,
        "net_threat_raw": net_raw,
        "net_threat_norm": net_norm,
        "opp_buried": opp_buried,
        "taiou_success": taiou_success,
        "survived": survived,
        "won": float(fire_rec.won[fi]),
    }


def _process_game_side(
    fire_rec: NpzRecord,
    opp_rec: NpzRecord,
    fire_side: str,
    sim: ChainSimulator,
    q_low: float,
    q_high: float,
    stride: int,
    counter: list[int],
) -> list[dict]:
    """1ゲーム・1サイドの発火イベントを処理する(stride間引き付き)。"""
    fire_events = _detect_fire_events(fire_rec.t_sec, fire_rec.score)
    if not fire_events:
        return []
    opp_boards = [
        (float(opp_rec.t_sec[i]), _board_from_grid(opp_rec.grids[i]))
        for i in range(len(opp_rec.t_sec))
    ]
    # 試合開始時刻の近似 = このゲームで記録された最初のフレーム時刻
    game_start_t = float(fire_rec.t_sec[0])
    rows: list[dict] = []
    for fi in fire_events:
        counter[0] += 1
        if counter[0] % stride != 0:
            continue
        row = _process_one_event(
            fi, fire_rec, opp_rec, opp_boards, fire_side, sim, q_low, q_high, game_start_t,
        )
        if row is not None:
            rows.append(row)
    return rows


def _collect_rows(npz_paths: list[Path], stride: int) -> pd.DataFrame:
    """全 npz を走査して net_threat 行データフレームを構築する。"""
    q_low, q_high = _estimate_puyo_quantiles(npz_paths)
    print(f"[INFO] 位相閾値: 序≤{q_low:.1f} 中≤{q_high:.1f} 終>{q_high:.1f}")
    sim = ChainSimulator()
    counter = [0]
    all_rows: list[dict] = []
    for npz_path in npz_paths:
        records = _load_npz(npz_path)
        by_side = {r.side: r for r in records}
        if "1P" not in by_side or "2P" not in by_side:
            continue
        r1p, r2p = by_side["1P"], by_side["2P"]
        for gid in np.unique(r1p.game_idx):
            m1, m2 = r1p.game_idx == gid, r2p.game_idx == gid
            if not m1.any() or not m2.any():
                continue
            g1p = NpzRecord(r1p.video_id, "1P", r1p.t_sec[m1], r1p.game_idx[m1], r1p.grids[m1], r1p.won[m1], r1p.score[m1])
            g2p = NpzRecord(r2p.video_id, "2P", r2p.t_sec[m2], r2p.game_idx[m2], r2p.grids[m2], r2p.won[m2], r2p.score[m2])
            for side, fr, opp in (("1P", g1p, g2p), ("2P", g2p, g1p)):
                try:
                    all_rows.extend(_process_game_side(fr, opp, side, sim, q_low, q_high, stride, counter))
                except Exception as e:
                    print(f"[WARN] {npz_path.stem} game={gid} side={side}: {e}", file=sys.stderr)
        print(f"  {npz_path.stem}: 累計 {len(all_rows)} 行 (走査済み発火 {counter[0]})")
    return pd.DataFrame(all_rows)


def _auc_line(df: pd.DataFrame, label: str, metric: str) -> str:
    """1ラベル×1指標のAUCを整形して返す(単クラスはskip表記)。"""
    y = df[label].astype(int).values
    if len(np.unique(y)) < 2:
        return f"    {metric:18s} -> n/a (単一クラス, n={len(y)})"
    auc = roc_auc_score(y, df[metric].values)
    return f"    {metric:18s} -> AUC={auc:.4f}  (n={len(y)}, 正例率={y.mean():.3f})"


def _auc_report(df: pd.DataFrame) -> None:
    """net_threat_raw vs ojama_sent の単変量AUCを全体・位相別に出力する。"""
    labels = ["opp_buried", "taiou_success", "won"]
    metrics = ["ojama_sent", "net_threat_raw"]
    print("\n=== 単変量AUC (全体) ===")
    for label in labels:
        print(f"  [{label}]")
        for metric in metrics:
            print(_auc_line(df, label, metric))
    for phase in ["序", "中", "終"]:
        sub = df[df["phase"] == phase]
        if len(sub) < 20:
            print(f"\n=== 位相={phase}: n={len(sub)} (20未満のためskip) ===")
            continue
        print(f"\n=== 単変量AUC (位相={phase}, n={len(sub)}) ===")
        for label in labels:
            print(f"  [{label}]")
            for metric in metrics:
                print(_auc_line(sub, label, metric))


def _redundancy_report(df: pd.DataFrame) -> None:
    """既存 exchange_labels.csv の主要指標との相関(冗長性チェック)を出力する。"""
    if not EXCHANGE_LABELS_CSV.exists():
        print("\n[WARN] exchange_labels.csv が見つからず冗長性チェックをskipします")
        return
    ex = pd.read_csv(EXCHANGE_LABELS_CSV)
    ex["t_key"] = ex["t_sec"].round(2)
    df2 = df.copy()
    df2["t_key"] = df2["t_sec"].round(2)
    merged = df2.merge(
        ex[["video_id", "game_idx", "t_key", "fire_side", "diff_absorption_capacity",
            "diff_potential_fire_power", "diff_honsen_output", "net_ojama_after"]],
        on=["video_id", "game_idx", "t_key", "fire_side"], how="inner",
    )
    print(f"\n=== 既存指標との相関 (冗長性チェック, 結合 n={len(merged)}) ===")
    for col in ("diff_absorption_capacity", "diff_potential_fire_power", "diff_honsen_output", "net_ojama_after"):
        if len(merged) < 5:
            continue
        corr = merged["net_threat_raw"].corr(merged[col])
        print(f"    net_threat_raw vs {col:26s} -> r={corr:+.3f}")


def main() -> None:
    """メイン処理。"""
    warnings.filterwarnings("ignore")
    full_run = "--full" in sys.argv
    stride = 1 if full_run else SAMPLE_STRIDE
    npz_paths = sorted(NPZ_DIR.glob("c*.npz"))
    if not npz_paths:
        print(f"[ERROR] npz が見つかりません: {NPZ_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] npz {len(npz_paths)} 本 / stride={stride} ({'全件' if full_run else 'サンプル'})")

    df = _collect_rows(npz_paths, stride)
    if df.empty:
        print("[ERROR] 発火イベントが0件でした。", file=sys.stderr)
        sys.exit(1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_CSV if full_run else OUTPUT_CSV.with_name("proto_net_threat_sample.csv")
    df.to_csv(out_path, index=False)
    print(f"[DONE] {len(df)} 行を {out_path} に保存しました")
    print(f"  video数={df['video_id'].nunique()}  phase内訳={df['phase'].value_counts().to_dict()}")
    print(f"  ojama_sent mean={df['ojama_sent'].mean():.2f}  predicted_counter mean={df['predicted_counter'].mean():.2f}"
          f"  net_threat_raw mean={df['net_threat_raw'].mean():.2f}")

    _auc_report(df)
    _redundancy_report(df)


if __name__ == "__main__":
    main()
