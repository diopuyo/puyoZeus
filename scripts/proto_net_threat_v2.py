"""発火時ネット脅威 (net_threat) 本命版 — 実ネクスト情報を使った相殺予測。

proto_net_threat.py (near-blind 版) との違い:
    近似版は「相手は何色でも都合よく積める」楽観近似 (potential_fire_power の
    任意色探索) を使っていた。本モジュールは NextDetector が実際に読み取った
    相手の next_pair / dnext_pair (色 1-5) を使い、「相手に見えている実際の
    ネクストで置いた場合」の相殺連鎖を探索する (本来の指標設計)。

前提 (2026-07 フェーズ1で追加):
    scripts/collect_boards_lean.py --with-next で収集した npz
    (data/indicators_v2/boards_lean_next/) に next1_a/next1_b/dnext_a/dnext_b
    (色 1-5、未検出/未取得は -1 or NextDetector 誤検出値) が保存されている。
    label_exchange_outcome.py の NpzRecord / _load_npz を拡張して読み込む
    (2026-07 フェーズ3、既存 95 本の boards_lean_fixed も同じ関数で読める
    ままにする後方互換設計)。

近似の限界 (要注意):
    1. K手(=相手が着弾までに打てる実ツモ数)は 1 または 2 にキャップする
       (_hands_cap、chain_to_time+SEC_PER_HAND から概算)。
       K>=3 相当の猶予がある場合、3手目以降は色不明のため
       potential_fire_power の任意色近似にフォールバックする
       (=本命版でも全区間を実ネクストでは追えない)。
    2. next1_a 等に color 1-5 以外の値 (=-1 未検出、または NextDetector の
       誤検出、実データ確認で ojama=9 混入を確認済) が入っている場合は
       「ネクスト不明」として扱い、その手は全面的に近似版にフォールバックする。
    3. 22 通りの配置 (縦6列×2色順 + 横5組×2色順) は「軸/子が独立落下する
       横置き」「同列2段積みの縦置き」という標準ぷよぷよ物理を反映するが、
       回転猶予 (= 設置直前の回転で軸/子の上下が入れ替わる余地) や
       壁蹴り等の高度なテクニックは考慮しない (簡易版)。

実行方法:
    python -m scripts.proto_net_threat_v2            # サンプル(間引き)実行
    python -m scripts.proto_net_threat_v2 --full      # 全発火イベント実行
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

from src.board import (  # noqa: E402
    BOARD_COLS,
    BOARD_ROWS,
    Board,
)
from src.chain import ChainSimulator  # noqa: E402
from src.indicators_v2 import (  # noqa: E402
    ON_FIELD_CAP,
    SEC_PER_HAND,
    _board_fire_ojama,
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
from scripts.proto_net_threat import _hands_cap  # noqa: E402

# ============================
# 定数定義
# ============================
# フェーズ2で収集した実ネクスト付き npz の格納先 (boards_lean_fixed とは別、
# 既存資産を破壊しない)。
NPZ_DIR_NEXT = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_next"
OUTPUT_CSV = PROJ_ROOT / "data" / "indicators_v2" / "proto_net_threat_v2.csv"
EXCHANGE_LABELS_CSV = PROJ_ROOT / "data" / "indicators_v2" / "exchange_labels.csv"

SAMPLE_STRIDE: int = 5  # 25本ぶんはイベント数が少ないため v1 より緩い間引き

# 有効なぷよ色 (1-5)。-1 (未検出) や 9 (ojama、NextDetector 誤検出で混入する
# ことを実データで確認済) はネクスト不明として扱う。
VALID_PUYO_COLORS: frozenset[int] = frozenset({1, 2, 3, 4, 5})

# collect_boards_lean.NEXT_COLOR_UNKNOWN / label_exchange_outcome.NEXT_COLOR_UNKNOWN
# と同値 (「未検出・未収集」の sentinel)。-1 以外の無効値は NextDetector 誤検出とみなす。
NEXT_COLOR_UNKNOWN_SENTINEL: int = -1

# 本命探索で「実ネクストで埋める」実ツモ数の上限 (=最大2組=4個、タスク仕様)。
MAX_REAL_PAIRS: int = 2

# 1 手目候補のうち 2 手目探索に持ち越す上位数 (ビーム幅、計算コスト抑制)。
PAIR_BEAM_K: int = 5

NET_THREAT_NORM: float = float(ON_FIELD_CAP)


def _is_valid_next_pair(pair: tuple[int, int] | None) -> bool:
    """next_pair/dnext_pair が実ネクストとして使える値かを判定する。

    None、または要素が VALID_PUYO_COLORS (1-5) 外なら False
    (=NextDetector 未検出・誤検出、フォールバック対象)。
    """
    if pair is None:
        return False
    return all(int(c) in VALID_PUYO_COLORS for c in pair)


def _enumerate_pair_placements(
    board: Board, color_a: int, color_b: int,
) -> list[Board]:
    """既知の2色ぷよペアを実物理配置(22通り)で置いた盤面群を返す。

    標準ぷよぷよの設置パターン:
      - 縦置き: 同一列に2段 (下段/上段の色順で2通り) × 6列 = 12通り
      - 横置き: 隣接2列に軸/子が各列の積み上がり最上段まで落下
        (色順2通り) × 5組 = 10通り
    列が満杯で置けない配置は候補から除外する (board 非破壊)。

    Args:
        board: 対象盤面 (相手側、破壊しない)。
        color_a: 軸ぷよ色 (1-5)。
        color_b: 子ぷよ色 (1-5)。

    Returns:
        配置後盤面のリスト (最大22件)。
    """
    placements: list[Board] = []
    placements.extend(_vertical_placements(board, color_a, color_b))
    placements.extend(_horizontal_placements(board, color_a, color_b))
    return placements


def _vertical_placements(board: Board, color_a: int, color_b: int) -> list[Board]:
    """縦置き (同一列2段、色順2通り) の配置を列挙する。最大 6×2=12 件。"""
    out: list[Board] = []
    for col in range(BOARD_COLS):
        height = board.height_of(col)
        if height + 2 > BOARD_ROWS:
            continue  # 2段積む余地なし
        row_bottom = BOARD_ROWS - 1 - height
        row_top = row_bottom - 1
        for bottom_color, top_color in ((color_a, color_b), (color_b, color_a)):
            work = board.copy()
            work.set(row_bottom, col, bottom_color)
            work.set(row_top, col, top_color)
            out.append(work)
    return out


def _horizontal_placements(board: Board, color_a: int, color_b: int) -> list[Board]:
    """横置き (隣接2列、色順2通り) の配置を列挙する。最大 5×2=10 件。

    軸/子は各列の現在の積み上がり最上段に独立して着地する
    (標準ぷよぷよ物理: 横向きペアの2セルは同じ行に揃うとは限らない)。
    """
    out: list[Board] = []
    for col in range(BOARD_COLS - 1):
        col2 = col + 1
        h1, h2 = board.height_of(col), board.height_of(col2)
        if h1 >= BOARD_ROWS or h2 >= BOARD_ROWS:
            continue
        row1 = BOARD_ROWS - 1 - h1
        row2 = BOARD_ROWS - 1 - h2
        for c1, c2 in ((color_a, color_b), (color_b, color_a)):
            work = board.copy()
            work.set(row1, col, c1)
            work.set(row2, col2, c2)
            out.append(work)
    return out


def _best_after_known_pair(
    board: Board, color_a: int, color_b: int, sim: ChainSimulator, beam_k: int,
) -> list[tuple[int, Board]]:
    """既知ペアの22配置を simulate し chain_count 降順で上位 beam_k を返す。"""
    candidates: list[tuple[int, Board]] = []
    for placed in _enumerate_pair_placements(board, color_a, color_b):
        chain = sim.simulate(placed).chain_count
        candidates.append((chain, placed))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:beam_k]


def _predicted_counter_ojama_v2(
    opp_board: Board,
    elapsed_sec_in_game: float,
    k_hands: int,
    next_pair: tuple[int, int] | None,
    dnext_pair: tuple[int, int] | None,
    sim: ChainSimulator,
) -> tuple[float, bool]:
    """本命版: 実ネクストで置いた場合の相手最大相殺お邪魔量を返す。

    次ツモ (next_pair) が有効なら1手目を実色22通りで固定する。
    2手目 (dnext_pair) も有効なら同様に固定する。K>=3相当分や
    ネクスト不明な手は potential_fire_power の任意色近似にフォールバックする。

    Returns:
        (predicted_counter_ojama, used_real_next)。
        used_real_next=True は next_pair が実際に使えた行(=本命版が
        近似版と異なる計算をした行)を示す。フォールバック行は
        v1 と同一ロジックになるため AUC 比較で区別する必要がある。
    """
    if not _is_valid_next_pair(next_pair):
        # 1手目からネクスト不明 → 全面フォールバック (v1 と同じ)
        raw = potential_fire_power(
            opp_board, elapsed_sec=elapsed_sec_in_game, simulator=sim,
            max_add=min(k_hands, 2),
        ).raw
        return raw, False

    real_pairs = min(k_hands, MAX_REAL_PAIRS)
    top1 = _best_after_known_pair(opp_board, next_pair[0], next_pair[1], sim, PAIR_BEAM_K)
    if not top1:
        return 0.0, True  # 全列満杯

    if real_pairs == 1 or not _is_valid_next_pair(dnext_pair):
        best = max(_board_fire_ojama(b, elapsed_sec_in_game, sim) for _, b in top1)
        return float(best), True

    # 2手目もネクスト既知 → 各候補から dnext_pair で再展開
    best = 0
    for _, b1 in top1:
        for _, b2 in _best_after_known_pair(b1, dnext_pair[0], dnext_pair[1], sim, PAIR_BEAM_K):
            ojama = _board_fire_ojama(b2, elapsed_sec_in_game, sim)
            if ojama > best:
                best = ojama
    return float(best), True


def _net_threat_v2(ojama_sent: float, predicted_counter: float) -> tuple[float, float]:
    """net_threat_v2 = 送りお邪魔量 - 本命版相手予測相殺量。(raw, 0-1正規化)。"""
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


def _nearest_index(t_arr: np.ndarray, t: float) -> int:
    """時刻 t に最も近いインデックスを返す。"""
    return int(np.argmin(np.abs(t_arr - t)))


def _opp_next_at(
    opp_rec: NpzRecord, idx: int,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """相手 npz の idx 番目スナップショットの (next_pair, dnext_pair) を返す。

    next1_a/next1_b/dnext_a/dnext_b が npz に存在しない(=boards_lean_fixed
    等の旧形式)場合は _load_npz が -1 埋め配列を返すため、この関数は
    常に安全に呼べる (後方互換)。
    """
    next_pair = (int(opp_rec.next1_a[idx]), int(opp_rec.next1_b[idx]))
    dnext_pair = (int(opp_rec.dnext_a[idx]), int(opp_rec.dnext_b[idx]))
    return next_pair, dnext_pair


# next_pair 無効時の原因分類 (opp_next_known 率の内訳報告用)。
NEXT_STATUS_VALID: str = "valid"       # 色1-5、実ネクストとして使用
NEXT_STATUS_ABSENT: str = "absent"    # -1 (NextDetector未検出/未収集)
NEXT_STATUS_MISDETECT: str = "misdetect"  # 1-5 でも -1 でもない値 (例: 9=おじゃま誤検出)


def _next_pair_status(pair: tuple[int, int]) -> str:
    """next_pair の無効原因を分類する (集計レポート用、判定ロジック自体は変えない)。"""
    if _is_valid_next_pair(pair):
        return NEXT_STATUS_VALID
    if all(int(c) == NEXT_COLOR_UNKNOWN_SENTINEL for c in pair):
        return NEXT_STATUS_ABSENT
    return NEXT_STATUS_MISDETECT


def _compute_counter_predictions(
    opp_board: Board,
    elapsed_in_game: float,
    k: int,
    next_pair: tuple[int, int] | None,
    dnext_pair: tuple[int, int] | None,
    sim: ChainSimulator,
) -> dict:
    """v1(近似) / v2(本命) 両方の相殺予測 + net_threat を計算して辞書で返す。

    _process_one_event_v2 の 50 行制約分割用ヘルパ。
    """
    predicted_v1 = potential_fire_power(
        opp_board, elapsed_sec=elapsed_in_game, simulator=sim, max_add=min(k, 2),
    ).raw
    predicted_v2, used_real_next = _predicted_counter_ojama_v2(
        opp_board, elapsed_in_game, k, next_pair, dnext_pair, sim,
    )
    return {
        "predicted_v1": predicted_v1,
        "predicted_v2": predicted_v2,
        "used_real_next": used_real_next,
    }


def _fire_context(
    fi: int, fire_rec: NpzRecord, game_start_t: float, sim: ChainSimulator,
) -> dict | None:
    """発火側の文脈 (盤面/時刻/連鎖数/K手) を抽出する。欠損時 None。"""
    delta_score = _delta_score_at_fire(fire_rec.score, fi)
    if delta_score is None:
        return None
    fi_board_idx = max(0, fi - 1)
    fire_board = _board_from_grid(fire_rec.grids[fi_board_idx])
    t_fire = float(fire_rec.t_sec[fi])
    elapsed_in_game = _game_relative_elapsed(t_fire, game_start_t)
    fire_chain = current_max_chain(fire_board, sim).raw
    return {
        "fire_board": fire_board,
        "t_fire": t_fire,
        "elapsed_in_game": elapsed_in_game,
        "fire_chain": fire_chain,
        "k": _hands_cap(fire_chain),
        "ojama_sent": float(_delta_to_ojama_standard(delta_score)),
    }


def _opp_context(
    opp_rec: NpzRecord, opp_boards: list[tuple[float, Board]], t_fire: float,
) -> dict:
    """相手側の文脈 (最近傍盤面/next_pair/dnext_pair/無効原因) を抽出する。"""
    opp_t_arr = np.array([x[0] for x in opp_boards])
    opp_idx = _nearest_index(opp_t_arr, t_fire)
    next_pair, dnext_pair = _opp_next_at(opp_rec, opp_idx)
    return {
        "opp_board": opp_boards[opp_idx][1],
        "next_pair": next_pair,
        "dnext_pair": dnext_pair,
        "next_pair_status": _next_pair_status(next_pair),
    }


def _assemble_event_row(
    fire_rec: NpzRecord, fi: int, fire_side: str,
    phase: str, fctx: dict, octx: dict, pred: dict, net_v1_raw: float,
    net_v2_raw: float, net_v2_norm: float, opp_buried: int,
    taiou_success: int, survived: int,
) -> dict:
    """net_threat_v2 出力1行を組み立てる (_process_one_event_v2 の分割用)。"""
    return {
        "video_id": fire_rec.video_id,
        "game_idx": int(fire_rec.game_idx[fi]),
        "t_sec": fctx["t_fire"],
        "fire_side": fire_side,
        "phase": phase,
        "fire_chain": float(fctx["fire_chain"]),
        "hands_k": fctx["k"],
        "ojama_sent": fctx["ojama_sent"],
        "opp_next_known": pred["used_real_next"],
        "next_pair_status": octx["next_pair_status"],
        "predicted_counter": pred["predicted_v1"],
        "predicted_counter_v2": pred["predicted_v2"],
        "net_threat_raw": net_v1_raw,
        "net_threat_v2_raw": net_v2_raw,
        "net_threat_v2_norm": net_v2_norm,
        "opp_buried": opp_buried,
        "taiou_success": taiou_success,
        "survived": survived,
        "won": float(fire_rec.won[fi]),
    }


def _process_one_event_v2(
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
    """1発火イベントを処理して net_threat_v2 行を返す(欠損等はNone)。"""
    fctx = _fire_context(fi, fire_rec, game_start_t, sim)
    if fctx is None:
        return None
    octx = _opp_context(opp_rec, opp_boards, fctx["t_fire"])

    pred = _compute_counter_predictions(
        octx["opp_board"], fctx["elapsed_in_game"], fctx["k"],
        octx["next_pair"], octx["dnext_pair"], sim,
    )
    net_v1_raw, _ = _net_threat_v2(fctx["ojama_sent"], pred["predicted_v1"])
    net_v2_raw, net_v2_norm = _net_threat_v2(fctx["ojama_sent"], pred["predicted_v2"])

    opp_buried = _compute_opp_buried(fctx["t_fire"], opp_boards, sim)
    taiou_success, survived = _compute_taiou_success(
        fctx["t_fire"], max(1.0, fctx["fire_chain"]),
        opp_rec.t_sec, opp_rec.score, opp_boards,
    )
    phase = _classify_phase(float(fctx["fire_board"].count_puyos()), q_low, q_high)
    return _assemble_event_row(
        fire_rec, fi, fire_side, phase, fctx, octx, pred,
        net_v1_raw, net_v2_raw, net_v2_norm, opp_buried, taiou_success, survived,
    )


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
    game_start_t = float(fire_rec.t_sec[0])
    rows: list[dict] = []
    for fi in fire_events:
        counter[0] += 1
        if counter[0] % stride != 0:
            continue
        row = _process_one_event_v2(
            fi, fire_rec, opp_rec, opp_boards, fire_side, sim, q_low, q_high, game_start_t,
        )
        if row is not None:
            rows.append(row)
    return rows


def _collect_rows(npz_paths: list[Path], stride: int) -> pd.DataFrame:
    """全 npz を走査して net_threat_v2 行データフレームを構築する。"""
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
            g1p = _slice_record(r1p, m1)
            g2p = _slice_record(r2p, m2)
            for side, fr, opp in (("1P", g1p, g2p), ("2P", g2p, g1p)):
                try:
                    all_rows.extend(_process_game_side(fr, opp, side, sim, q_low, q_high, stride, counter))
                except Exception as e:
                    print(f"[WARN] {npz_path.stem} game={gid} side={side}: {e}", file=sys.stderr)
        print(f"  {npz_path.stem}: 累計 {len(all_rows)} 行 (走査済み発火 {counter[0]})")
    return pd.DataFrame(all_rows)


def _slice_record(rec: NpzRecord, mask: np.ndarray) -> NpzRecord:
    """NpzRecord を game_idx マスクでスライスする (next 列含む全フィールド)。"""
    return NpzRecord(
        video_id=rec.video_id, side=rec.side,
        t_sec=rec.t_sec[mask], game_idx=rec.game_idx[mask],
        grids=rec.grids[mask], won=rec.won[mask], score=rec.score[mask],
        next1_a=rec.next1_a[mask], next1_b=rec.next1_b[mask],
        dnext_a=rec.dnext_a[mask], dnext_b=rec.dnext_b[mask],
    )


def _auc_line(df: pd.DataFrame, label: str, metric: str) -> str:
    """1ラベル×1指標のAUCを整形して返す(単クラスはskip表記)。"""
    y = df[label].astype(int).values
    if len(np.unique(y)) < 2:
        return f"    {metric:22s} -> n/a (単一クラス, n={len(y)})"
    auc = roc_auc_score(y, df[metric].values)
    return f"    {metric:22s} -> AUC={auc:.4f}  (n={len(y)}, 正例率={y.mean():.3f})"


_AUC_LABELS: tuple[str, ...] = ("opp_buried", "taiou_success", "won")
_AUC_METRICS: tuple[str, ...] = ("ojama_sent", "net_threat_raw", "net_threat_v2_raw")

# 位相別AUC出力の最小サンプル数 (これ未満はskip、v1と同じ閾値)
PHASE_AUC_MIN_N: int = 20


def _auc_block(sub_df: pd.DataFrame, title: str) -> None:
    """1条件(全体 or 位相別サブセット)ぶんのAUCブロックを出力する。"""
    print(f"\n=== 単変量AUC ({title}, n={len(sub_df)}) ===")
    if len(sub_df) < PHASE_AUC_MIN_N:
        print(f"    (n={len(sub_df)} < {PHASE_AUC_MIN_N} のためskip)")
        return
    for label in _AUC_LABELS:
        print(f"  [{label}]")
        for metric in _AUC_METRICS:
            print(_auc_line(sub_df, label, metric))


def _auc_report_condition(df: pd.DataFrame, cond_name: str) -> None:
    """1条件(全体 or opp_next_known=True)ぶんに対し、全体+位相別AUCを出す。"""
    _auc_block(df, f"{cond_name} 全体")
    for phase in ("序", "中", "終"):
        _auc_block(df[df["phase"] == phase], f"{cond_name} 位相={phase}")


def _auc_report(df: pd.DataFrame) -> None:
    """ojama_sent / net_threat_raw(v1) / net_threat_v2_raw の単変量AUCを比較する。

    全体と「opp_next_known=True (=本命版が実際に実ネクストで計算した行)」の
    2条件 × (全体+序/中/終位相別) で出力する。本命版の真価は後者の subset
    (特に中盤) で見るべき (フォールバック行は v1 と数値が一致するため差が出ない)。
    """
    _auc_report_condition(df, "全体条件")
    known_df = df[df["opp_next_known"]]
    _auc_report_condition(known_df, f"opp_next_known=True(n={len(known_df)})条件")


def _next_pair_status_report(df: pd.DataFrame) -> None:
    """opp_next_known率と、無効原因(未検出absent/誤検出misdetect)の内訳を出す。"""
    counts = df["next_pair_status"].value_counts()
    total = len(df)
    print(f"\n=== opp_next_known 内訳 (n={total}) ===")
    for status in (NEXT_STATUS_VALID, NEXT_STATUS_ABSENT, NEXT_STATUS_MISDETECT):
        n = int(counts.get(status, 0))
        print(f"    {status:10s}: {n:6d} ({n / total * 100:5.1f}%)")


def main() -> None:
    """メイン処理。"""
    warnings.filterwarnings("ignore")
    full_run = "--full" in sys.argv
    stride = 1 if full_run else SAMPLE_STRIDE
    npz_paths = sorted(NPZ_DIR_NEXT.glob("c*.npz"))
    if not npz_paths:
        print(f"[ERROR] npz が見つかりません: {NPZ_DIR_NEXT}", file=sys.stderr)
        print("  (先に scripts.collect_boards_lean --with-next で収集すること)", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] npz {len(npz_paths)} 本 / stride={stride} ({'全件' if full_run else 'サンプル'})")

    df = _collect_rows(npz_paths, stride)
    if df.empty:
        print("[ERROR] 発火イベントが0件でした。", file=sys.stderr)
        sys.exit(1)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_CSV if full_run else OUTPUT_CSV.with_name("proto_net_threat_v2_sample.csv")
    df.to_csv(out_path, index=False)
    print(f"[DONE] {len(df)} 行を {out_path} に保存しました")
    print(f"  video数={df['video_id'].nunique()}  opp_next_known率={df['opp_next_known'].mean():.3f}")
    print(f"  ojama_sent mean={df['ojama_sent'].mean():.2f}"
          f"  predicted_counter(v1) mean={df['predicted_counter'].mean():.2f}"
          f"  predicted_counter_v2 mean={df['predicted_counter_v2'].mean():.2f}")

    _next_pair_status_report(df)
    _auc_report(df)


if __name__ == "__main__":
    main()
