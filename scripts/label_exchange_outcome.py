"""発火イベント検出 + 打ち合い結果ラベル生成スクリプト。

boards_lean_fixed の npz (95本) から:
  1. 同一(video_id, game_idx)で 1P/2P を時刻整列
  2. スコア増分が閾値超え = 発火イベントと判定
  3. 発火直前盤面から指標を計算
  4. 発火後の「打ち合い結果」ラベルを生成
  5. exchange_labels.csv として出力

v2 追加ラベル:
  - taiou_success: 対応成功(受け手が T_guard 内に発火 かつ 埋まらず生存)
  - survived: 受け手が T_guard+TAIOU_CHECK_AFTER_SEC 後も生存(単純生存)
  - net_ojama_after: 攻撃お邪魔 − 受け手の猶予内相殺お邪魔(個数、連続値)

CPU 節度: OMP_NUM_THREADS=3 等はプロセス起動前にセットすること(CLAUDE.md推奨)。
重い指標(potential_fire_power等)は発火イベント時点だけ計算する。
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# スレッド制限(ここで設定しておく)
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

# プロジェクトルート追加
PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.indicators_v2 import (  # noqa: E402
    absorption_capacity,
    board_ojama_count,
    column_bumpiness,
    current_max_chain,
    death_margin,
    dig_resistance,
    honsen_output,
    honsen_tempo_output,
    immediate_fire_power,
    max_column_height,
    potential_fire_power,
    second_chain_potential,
    estimate_chain_anim_duration_sec,
)
from src.scoring import score_to_ojama  # noqa: E402

# ============================
# 定数定義
# ============================
NPZ_DIR = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed"
OUTPUT_PATH = PROJ_ROOT / "data" / "indicators_v2" / "exchange_labels.csv"

# score OCR が完全破綻し発火検出(スコア増分ベース)が信用できない動画
# (memory project_video_difficulty_3broken_2026-07-29 + c69 追加確認、
#  build_board_pairs_lean.py の won=NaN 全滅動画と一致)。
# --exclude-videos 省略時はこの定数を使わず全動画処理する(既定=旧挙動と完全一致)。
SCORE_OCR_BROKEN_VIDEOS: frozenset[str] = frozenset({"c26", "c30", "c58", "c69"})

# 発火検出: 短窓でのスコア増分閾値
# TODO(2026-08-01 Step0): scripts/measure_exchange_dynamics.py 側の同種閾値
# (=40) と不一致 (本ファイルは80)。Step0では意図的に触らない
# (user/アーキ判断待ち、統一するかは別途要判断)。
SCORE_DELTA_FIRE: int = 80

# 返し窓: score=-1 補間なし。連鎖数不明時のデフォルト秒数
RETURN_WINDOW_DEFAULT_SEC: float = 6.0

# 位相分類: 盤面ぷよ合計の3分位境界(実データから暫定)
PHASE_QUANTILE_LOW: float = 0.33
PHASE_QUANTILE_HIGH: float = 0.67

# 相手の death_margin 急悪化を検出する時間窓(秒)
BURIAL_WINDOW_SEC: float = 8.0
# death_margin スコアが閾値以下 = 危険 (opp_buried)
DEATH_MARGIN_DANGER_THRESHOLD: float = 0.2

# A-1: returned_competitive の判定係数
# 返しお邪魔 >= 発火側お邪魔 × RETURN_FACTOR を「同等以上の返し」と定義
RETURN_FACTOR: float = 0.8

# taiou_success (対応成功) 判定用定数
# T_guard = estimate_chain_anim_duration_sec(攻撃側連鎖数) + SEC_PER_HAND で定義
# (2026-08-01 Step0: 旧 chain_to_time=TIME_PER_CHAIN_SEC=0.30 から一本化)
# SEC_PER_HAND は indicators_v2 から import するが、念のため本スクリプト内にも定義
_SEC_PER_HAND_LOCAL: float = 0.733  # indicators_v2.SEC_PER_HAND と同値(実測中央値)
# T_guard 終端から何秒後まで「埋まっていないか」を確認する窓(秒)
TAIOU_CHECK_AFTER_SEC: float = 3.0
# 受け手が「対応を出した」と見なすスコア増分閾値(発火検出閾値と統一)
TAIOU_RESPONSE_SCORE_THRESHOLD: int = SCORE_DELTA_FIRE


class NpzRecord(NamedTuple):
    """1本分の npz データ。

    next1_a/next1_b/dnext_a/dnext_b (指標①本命版検証用、2026-07 追加):
    boards_lean_next (--with-next 収集) には実値、boards_lean_fixed 等の
    旧形式には存在しないため _load_npz が -1 (NEXT_COLOR_UNKNOWN) 埋め配列を
    補完する。末尾の optional フィールドのため既存の位置引数呼び出し
    (NpzRecord(video_id, side, t_sec, ...)) は後方互換で動作する。
    """
    video_id: str
    side: str
    t_sec: np.ndarray     # shape(N,)
    game_idx: np.ndarray  # shape(N,)
    grids: np.ndarray     # shape(N,13,6)
    won: np.ndarray       # shape(N,)
    score: np.ndarray     # shape(N,) int32, -1=欠損
    next1_a: np.ndarray = np.empty(0, dtype=np.int8)   # shape(N,) int8, -1=未検出
    next1_b: np.ndarray = np.empty(0, dtype=np.int8)
    dnext_a: np.ndarray = np.empty(0, dtype=np.int8)
    dnext_b: np.ndarray = np.empty(0, dtype=np.int8)


# next_pair/dnext_pair が npz に存在しない場合の埋め値
# (scripts/collect_boards_lean.py の NEXT_COLOR_UNKNOWN と同値)。
NEXT_COLOR_UNKNOWN: int = -1


def _load_npz(path: Path) -> list[NpzRecord]:
    """1つの npz から NpzRecord を側ごとに返す。

    next1_a/next1_b/dnext_a/dnext_b (指標①本命版用) が npz に存在すれば
    実値を、存在しなければ (boards_lean_fixed 等の旧形式) NEXT_COLOR_UNKNOWN
    で埋めた配列を格納する (後方互換、呼び出し側は常に同じ型で扱える)。
    """
    with np.load(path, allow_pickle=True) as d:
        video_ids = d["video_id"]  # shape(N,) <U8
        sides = d["side"]          # shape(N,) <U2
        t_secs = d["t_sec"].astype(np.float32)
        game_idxs = d["game_idx"].astype(np.int32)
        grids = d["grids"].astype(np.int8)
        wons = d["won"].astype(np.float32)
        scores = d["score"].astype(np.int32)
        n = len(scores)
        next1_a = _read_next_column(d, "next1_a", n)
        next1_b = _read_next_column(d, "next1_b", n)
        dnext_a = _read_next_column(d, "dnext_a", n)
        dnext_b = _read_next_column(d, "dnext_b", n)

    # 側ごとに分割
    records: list[NpzRecord] = []
    for side_val in ("1P", "2P"):
        mask = sides == side_val
        if not mask.any():
            continue
        vid = str(video_ids[mask][0])
        records.append(NpzRecord(
            video_id=vid,
            side=side_val,
            t_sec=t_secs[mask],
            game_idx=game_idxs[mask],
            grids=grids[mask],
            won=wons[mask],
            score=scores[mask],
            next1_a=next1_a[mask],
            next1_b=next1_b[mask],
            dnext_a=dnext_a[mask],
            dnext_b=dnext_b[mask],
        ))
    return records


def _read_next_column(npz_data: object, key: str, n: int) -> np.ndarray:
    """npz からネクスト列を読む。存在しなければ NEXT_COLOR_UNKNOWN 埋め配列を返す。

    boards_lean_fixed 等の旧形式 npz (next1_a 等が無い) との後方互換のため。
    """
    if key in npz_data:  # type: ignore[operator]
        return npz_data[key].astype(np.int8)  # type: ignore[index]
    return np.full(n, NEXT_COLOR_UNKNOWN, dtype=np.int8)


def _detect_fire_events(
    t_sec: np.ndarray,
    score: np.ndarray,
) -> list[int]:
    """スコア増分が SCORE_DELTA_FIRE を超えるフレームのインデックス一覧。

    score=-1 はスキップ。直前の有効 score との差分を使う。
    """
    fire_indices: list[int] = []
    prev_score = -1
    for i in range(len(score)):
        s = int(score[i])
        if s < 0:
            continue
        if prev_score >= 0:
            delta = s - prev_score
            if delta >= SCORE_DELTA_FIRE:
                fire_indices.append(i)
        prev_score = s
    return fire_indices


def _board_from_grid(grid: np.ndarray) -> Board:
    """(13,6) の int8 array から Board を生成する。"""
    return Board.from_list(grid.tolist())


def _game_relative_elapsed(t_fire: float, game_start_t: float) -> float:
    """試合開始からの経過秒を返す(マージンタイム計算用)。

    npz の t_sec は動画絶対時刻 (1動画に複数試合を含む場合、game_idx>=1 の
    試合では数百〜千秒に達する)。そのまま score_to_ojama 系の elapsed_sec に
    渡すとマージンタイム減衰 (MARGIN_TIME_START_SEC=96s 以降 16秒毎に×0.75、
    最大14回) が過剰発火し potential_fire_power / immediate_fire_power の
    raw が桁違いに膨張するバグがあった (実データ確認済み、修正版)。

    その試合内での最初のフレーム時刻をゼロ点として補正する。
    ⚠️ game_idx==0 (先頭試合) でも動画によっては録画開始が試合開始より
    大幅に遅れる (実測で最大 200 秒超) ケースがあり、無条件に「先頭試合は
    ほぼ0点」とは言えない。よって game_idx の値に関わらず常にこの補正を
    適用する (特別扱いしない)。
    数秒のバッファ誤差はマージンタイム閾値96秒に対し軽微なため許容する
    (scripts/proto_net_threat.py の同名関数と同等ロジック)。

    Args:
        t_fire: 対象フレームの動画絶対時刻(秒)。
        game_start_t: その (video_id, game_idx, side) グループの
            最初のフレーム時刻(秒)。

    Returns:
        試合開始からの経過秒(0以上)。
    """
    return max(0.0, t_fire - game_start_t)


def _compute_features(
    board: Board,
    elapsed_sec: float,
    sim: ChainSimulator,
) -> dict[str, float]:
    """発火直前盤面から指標を計算して辞書で返す。

    Args:
        board: 対象盤面。
        elapsed_sec: 試合開始からの経過秒(マージンタイム計算用)。
            ⚠️ 動画絶対時刻 (npz の t_sec) をそのまま渡さないこと。
            呼び出し側で _game_relative_elapsed() を通した値を渡す。
        sim: ChainSimulator。
    """
    cmc = current_max_chain(board, sim)
    pfp = potential_fire_power(board, elapsed_sec, sim)
    ifp = immediate_fire_power(board, elapsed_sec, sim)
    ho = honsen_output(board, sim)
    # honsen_tempo_output: opp_chain は外部から渡せないのでここでは opp=0 仮置き
    # → 呼び出し側で opp_chain を使って再計算する
    dm = death_margin(board)
    scp = second_chain_potential(board, sim)
    boc = board_ojama_count(board)
    mch = max_column_height(board)
    cb = column_bumpiness(board)
    ac = absorption_capacity(board)
    dig = dig_resistance(board, sim)
    return {
        "current_max_chain": cmc.raw,
        "potential_fire_power": pfp.raw,
        "immediate_fire_power": ifp.raw,
        "honsen_output": ho.raw,
        "death_margin": dm.raw,
        "second_chain_potential": scp.raw,
        "board_ojama_count": boc.raw,
        "max_column_height": mch.raw,
        "column_bumpiness": cb.raw,
        "absorption_capacity": ac.raw,
        "dig_resistance": dig.raw,
    }


def _delta_to_ojama_standard(delta_score: int) -> int:
    """Δscoreを標準レート固定(70点/個)でお邪魔換算する。

    マージンタイム補正を無効化して比較可能な単位にする。
    (マージンタイムは試合状況依存で rate が 3 まで下がり換算が歪む)
    """
    from src.scoring import OJAMA_RATE_STANDARD
    return max(0, delta_score) // OJAMA_RATE_STANDARD


def _compute_net_ojama(
    fire_score_delta: int,
    fire_t: float,
    opp_t_sec: np.ndarray,
    opp_score: np.ndarray,
    return_window_sec: float,
) -> float:
    """発火側が送ったお邪魔 - 相手が返し窓内に送ったお邪魔の正味。

    標準レート(70点/個)固定でお邪魔換算する。マージンタイム補正は除外。
    相手の返し窓 [fire_t, fire_t + return_window_sec] のΔscoreも同様換算。
    返し窓の基準スコアは窓直前の有効スコアを使う。
    """
    fire_ojama = _delta_to_ojama_standard(fire_score_delta)

    # 返し窓の直前スコア(基準値)を取得
    pre_window = (opp_t_sec < fire_t) & (opp_score >= 0)
    if pre_window.any():
        baseline_opp = int(opp_score[pre_window][-1])
    else:
        baseline_opp = -1

    # 返し窓内の最大スコア
    in_window = (opp_t_sec >= fire_t) & (opp_t_sec <= fire_t + return_window_sec)
    opp_scores_w = opp_score[in_window]
    valid = opp_scores_w[opp_scores_w >= 0]
    if len(valid) >= 1 and baseline_opp >= 0:
        opp_delta = max(0, int(valid.max()) - baseline_opp)
    else:
        opp_delta = 0
    opp_ojama = _delta_to_ojama_standard(opp_delta)
    return float(fire_ojama - opp_ojama)


def _compute_returned(
    fire_t: float,
    opp_t_sec: np.ndarray,
    opp_score: np.ndarray,
    return_window_sec: float,
) -> int:
    """相手が返し窓内に SCORE_DELTA_FIRE 以上の発火をしたか(2値)。

    返し窓の直前スコアを基準に、窓内の最大スコア増分で判定する。
    """
    # 返し窓直前の有効スコアを基準として取得
    pre_window = (opp_t_sec < fire_t) & (opp_score >= 0)
    if pre_window.any():
        prev_s = int(opp_score[pre_window][-1])
    else:
        prev_s = -1

    in_window = (opp_t_sec >= fire_t) & (opp_t_sec <= fire_t + return_window_sec)
    valid_idx = np.where(in_window)[0]
    for idx in valid_idx:
        s = int(opp_score[idx])
        if s < 0:
            continue
        if prev_s >= 0 and (s - prev_s) >= SCORE_DELTA_FIRE:
            return 1
        if s >= 0:
            prev_s = s
    return 0


def _delta_score_to_ojama_count(delta_score: int) -> int:
    """Δscoreを標準レート固定でお邪魔個数に換算する(マージン補正なし)。

    score_to_ojama(prev_leftover=0, elapsed_sec=0) と等価だが
    直接 OJAMA_RATE_STANDARD 除算で明示的に補正を排除する。
    """
    from src.scoring import OJAMA_RATE_STANDARD
    return max(0, delta_score) // OJAMA_RATE_STANDARD


def _compute_returned_competitive(
    fire_delta_score: int,
    fire_t: float,
    return_window_sec: float,
    opp_t_sec: np.ndarray,
    opp_score: np.ndarray,
) -> int:
    """A-1: 競合返し判定。

    相手が返し窓 W = estimate_chain_anim_duration_sec(発火側連鎖数) 内に発火し、
    かつ 返しお邪魔 >= 発火側お邪魔 × RETURN_FACTOR を満たすか。
    (2026-08-01 Step0: 旧 chain_to_time=TIME_PER_CHAIN_SEC=0.30 から一本化)

    - お邪魔換算は標準レート固定(マージン補正バグ回避)。
    - 発火側の連鎖数は current_max_chain.raw で近似(呼び出し元で渡す)。
    - return_window_sec = estimate_chain_anim_duration_sec(approx_chains) が
      呼び出し元で計算済み。
    """
    fire_ojama = _delta_score_to_ojama_count(fire_delta_score)

    # 返し窓の直前スコアを基準値として取得
    pre_mask = (opp_t_sec < fire_t) & (opp_score >= 0)
    baseline_opp = int(opp_score[pre_mask][-1]) if pre_mask.any() else -1

    # 返し窓内の最大スコア増分 → お邪魔換算
    in_mask = (opp_t_sec >= fire_t) & (opp_t_sec <= fire_t + return_window_sec)
    valid_in = opp_score[in_mask]
    valid_in = valid_in[valid_in >= 0]
    if len(valid_in) < 1 or baseline_opp < 0:
        return 0

    opp_delta = max(0, int(valid_in.max()) - baseline_opp)
    opp_ojama = _delta_score_to_ojama_count(opp_delta)

    # 判定: 返しが発火側の RETURN_FACTOR 倍以上
    required = fire_ojama * RETURN_FACTOR
    return 1 if opp_ojama >= required else 0


def _compute_opp_buried(
    fire_t: float,
    opp_boards: list[tuple[float, Board]],
    sim: ChainSimulator,
) -> int:
    """打ち合い後窓で相手の death_margin が急悪化 or is_dead() か(2値)。

    opp_boards: (t_sec, Board) のリスト(t昇順)。
    """
    burial_end = fire_t + BURIAL_WINDOW_SEC
    boards_in_window = [
        (t, b) for (t, b) in opp_boards
        if fire_t <= t <= burial_end
    ]
    if not boards_in_window:
        return 0
    for _, b in boards_in_window:
        if b.is_dead():
            return 1
        dm = death_margin(b)
        if dm.score <= DEATH_MARGIN_DANGER_THRESHOLD:
            return 1
    return 0


def _opp_is_safe_after_tguard(
    fire_t: float,
    t_guard: float,
    opp_boards: list[tuple[float, Board]],
) -> bool:
    """T_guard 終端〜+TAIOU_CHECK_AFTER_SEC の受け手盤面が安全か(死んでいないか)。

    安全の条件: is_dead() が False かつ death_margin.score > DEATH_MARGIN_DANGER_THRESHOLD。
    窓内にフレームがない場合は「安全」とみなす(データ欠損で判定不能 → 保守側)。
    """
    check_start = fire_t + t_guard
    check_end = check_start + TAIOU_CHECK_AFTER_SEC
    frames_in_window = [
        (t, b) for (t, b) in opp_boards
        if check_start <= t <= check_end
    ]
    if not frames_in_window:
        return True  # データ欠損 → 保守的に安全とみなす
    for _, b in frames_in_window:
        if b.is_dead():
            return False
        dm = death_margin(b)
        if dm.score <= DEATH_MARGIN_DANGER_THRESHOLD:
            return False
    return True


def _opp_fired_in_tguard(
    fire_t: float,
    t_guard: float,
    opp_t_sec: np.ndarray,
    opp_score: np.ndarray,
) -> bool:
    """受け手が T_guard 内(fire_t 〜 fire_t+t_guard)に発火したか。

    発火 = スコア増分 >= TAIOU_RESPONSE_SCORE_THRESHOLD。
    直前の有効スコアからの増分で判定する。
    """
    # T_guard 直前の有効スコアを基準として取得
    pre_mask = (opp_t_sec < fire_t) & (opp_score >= 0)
    prev_s = int(opp_score[pre_mask][-1]) if pre_mask.any() else -1

    in_window = (opp_t_sec >= fire_t) & (opp_t_sec <= fire_t + t_guard)
    for idx in np.where(in_window)[0]:
        s = int(opp_score[idx])
        if s < 0:
            continue
        if prev_s >= 0 and (s - prev_s) >= TAIOU_RESPONSE_SCORE_THRESHOLD:
            return True
        if s >= 0:
            prev_s = s
    return False


def _compute_taiou_success(
    fire_t: float,
    approx_chains: float,
    opp_t_sec: np.ndarray,
    opp_score: np.ndarray,
    opp_boards: list[tuple[float, Board]],
) -> tuple[int, int]:
    """対応成功(taiou_success)と単純生存(survived)を計算して返す。

    taiou_success=1 の条件(両方満たす):
      1. 受け手が T_guard 内に発火した
      2. T_guard 終端後も受け手が埋まっていない
    survived=1 の条件:
      T_guard 終端後も受け手が埋まっていない(発火有無問わず)

    Args:
        fire_t: 攻撃側発火時刻(秒)。
        approx_chains: 攻撃側の近似連鎖数。
        opp_t_sec: 受け手の時刻配列。
        opp_score: 受け手のスコア配列。
        opp_boards: 受け手の (t, Board) リスト。

    Returns:
        (taiou_success, survived) の 2値タプル。
    """
    from src.indicators_v2 import SEC_PER_HAND
    # 2026-08-01 Step0: 着弾遅延は estimate_chain_anim_duration_sec
    # (CHAIN_ANIM_PER_STEP_SEC=0.4秒/連鎖、23動画418イベント実測ベース) に
    # 一本化 (旧 chain_to_time=TIME_PER_CHAIN_SEC=0.30 は過小評価と判明済み)。
    t_guard = estimate_chain_anim_duration_sec(max(1.0, approx_chains)) + SEC_PER_HAND
    fired = _opp_fired_in_tguard(fire_t, t_guard, opp_t_sec, opp_score)
    safe = _opp_is_safe_after_tguard(fire_t, t_guard, opp_boards)
    taiou_success = int(fired and safe)
    survived = int(safe)
    return taiou_success, survived


def _compute_net_ojama_after(
    fire_delta_score: int,
    fire_t: float,
    approx_chains: float,
    opp_t_sec: np.ndarray,
    opp_score: np.ndarray,
) -> float:
    """攻撃お邪魔 − 受け手の猶予(T_guard)内相殺お邪魔(個数、連続値)。

    net_ojama_sign の代替: 符号でなく量(正値=攻撃側が残る正味お邪魔)。
    お邪魔換算は標準レート OJAMA_RATE_STANDARD(70点/個)固定。

    Args:
        fire_delta_score: 攻撃側のΔスコア(点)。
        fire_t: 攻撃側発火時刻(秒)。
        approx_chains: 攻撃側の近似連鎖数(T_guard 計算用)。
        opp_t_sec: 受け手の時刻配列。
        opp_score: 受け手のスコア配列。

    Returns:
        正味お邪魔個数(float)。攻撃−相殺。負値=相殺超過。
    """
    from src.indicators_v2 import SEC_PER_HAND
    # 2026-08-01 Step0: 着弾遅延は estimate_chain_anim_duration_sec
    # (CHAIN_ANIM_PER_STEP_SEC=0.4秒/連鎖、23動画418イベント実測ベース) に
    # 一本化 (旧 chain_to_time=TIME_PER_CHAIN_SEC=0.30 は過小評価と判明済み)。
    t_guard = estimate_chain_anim_duration_sec(max(1.0, approx_chains)) + SEC_PER_HAND

    attack_ojama = _delta_to_ojama_standard(fire_delta_score)

    # 受け手の T_guard 内相殺お邪魔を計算
    pre_mask = (opp_t_sec < fire_t) & (opp_score >= 0)
    baseline_opp = int(opp_score[pre_mask][-1]) if pre_mask.any() else -1

    in_mask = (opp_t_sec >= fire_t) & (opp_t_sec <= fire_t + t_guard)
    valid_in = opp_score[in_mask]
    valid_in = valid_in[valid_in >= 0]

    if len(valid_in) >= 1 and baseline_opp >= 0:
        opp_delta = max(0, int(valid_in.max()) - baseline_opp)
    else:
        opp_delta = 0
    opp_ojama = _delta_to_ojama_standard(opp_delta)
    return float(attack_ojama - opp_ojama)


def _classify_phase(puyo_total: float, q_low: float, q_high: float) -> str:
    """盤面ぷよ合計を3分位で序/中/終に分類する。"""
    if puyo_total <= q_low:
        return "序"
    elif puyo_total <= q_high:
        return "中"
    return "終"


def _process_game(
    game_1p: NpzRecord,
    game_2p: NpzRecord,
    fire_side: str,
    sim: ChainSimulator,
    puyo_q_low: float,
    puyo_q_high: float,
) -> list[dict]:
    """1ゲーム・1サイドの発火イベントを処理してレコードリストを返す。"""
    if fire_side == "1P":
        fire_rec = game_1p
        opp_rec = game_2p
    else:
        fire_rec = game_2p
        opp_rec = game_1p

    fire_events = _detect_fire_events(fire_rec.t_sec, fire_rec.score)
    if not fire_events:
        return []

    # 試合開始時刻の近似 = この (video_id, game_idx, fire_side) で
    # 記録された最初のフレーム時刻(マージンタイム計算の基準点、バグ修正)
    game_start_t = float(fire_rec.t_sec[0])

    # 相手盤面を (t, Board) ペアで保持(重い計算は発火時点のみ)
    opp_boards: list[tuple[float, Board]] = []
    for i in range(len(opp_rec.t_sec)):
        t = float(opp_rec.t_sec[i])
        opp_boards.append((t, _board_from_grid(opp_rec.grids[i])))

    rows: list[dict] = []
    prev_fire_score = -1

    for fi in fire_events:
        t_fire = float(fire_rec.t_sec[fi])
        s_fire = int(fire_rec.score[fi])
        # ΔscoreはSTABLE前の有効scoreとの差
        # 直前有効score
        prev_valid = -1
        for j in range(fi - 1, -1, -1):
            if fire_rec.score[j] >= 0:
                prev_valid = int(fire_rec.score[j])
                break
        if prev_valid < 0:
            continue
        delta_score = s_fire - prev_valid
        if delta_score < SCORE_DELTA_FIRE:
            continue

        # 発火直前盤面
        fi_board_idx = max(0, fi - 1)
        fire_board = _board_from_grid(fire_rec.grids[fi_board_idx])

        # 相手の発火直前盤面(時刻が最も近いもの)
        opp_t_arr = np.array([x[0] for x in opp_boards])
        nearest_opp = int(np.argmin(np.abs(opp_t_arr - t_fire)))
        opp_board = opp_boards[nearest_opp][1]

        # 試合開始からの経過秒(マージンタイム計算用、動画絶対時刻ではない)
        elapsed_in_game = _game_relative_elapsed(t_fire, game_start_t)

        # 発火側指標
        fire_feats = _compute_features(fire_board, elapsed_in_game, sim)
        # 相手側指標
        opp_feats = _compute_features(opp_board, elapsed_in_game, sim)

        # honsen_tempo_output: 両者の current_max_chain で計算
        hto_fire = honsen_tempo_output(
            fire_feats["current_max_chain"],
            0.0,  # achievable 不明 → fallback current+2
            opp_feats["current_max_chain"],
        )
        hto_opp = honsen_tempo_output(
            opp_feats["current_max_chain"],
            0.0,
            fire_feats["current_max_chain"],
        )

        # 返し窓: 発火連鎖数から推定(current_max_chain を近似として使用)
        # 2026-08-01 Step0: chain_to_time (TIME_PER_CHAIN_SEC=0.30、過小評価
        # 判明済み) から estimate_chain_anim_duration_sec (CHAIN_ANIM_PER_STEP_SEC
        # =0.4、23動画418イベント実測ベース) に一本化。
        approx_chains = max(1.0, fire_feats["current_max_chain"])
        return_window = estimate_chain_anim_duration_sec(approx_chains)
        if return_window <= 0:
            return_window = RETURN_WINDOW_DEFAULT_SEC

        # ラベル計算
        net_oj = _compute_net_ojama(
            delta_score, t_fire,
            opp_rec.t_sec, opp_rec.score,
            return_window,
        )
        returned = _compute_returned(
            t_fire, opp_rec.t_sec, opp_rec.score, return_window,
        )
        # A-1: 競合返し(同等以上のお邪魔を返したか)
        returned_competitive = _compute_returned_competitive(
            delta_score, t_fire, return_window,
            opp_rec.t_sec, opp_rec.score,
        )
        opp_buried = _compute_opp_buried(t_fire, opp_boards, sim)

        # user 確定定義: taiou_success (対応成功) ラベル
        # T_guard = estimate_chain_anim_duration_sec(攻撃側連鎖数) + SEC_PER_HAND
        # で猶予時間を計算 (2026-08-01 Step0 一本化)
        taiou_success, survived = _compute_taiou_success(
            t_fire, approx_chains,
            opp_rec.t_sec, opp_rec.score,
            opp_boards,
        )
        # net_ojama_after: T_guard 内の相殺後正味お邪魔(個数、連続値)
        net_ojama_after = _compute_net_ojama_after(
            delta_score, t_fire, approx_chains,
            opp_rec.t_sec, opp_rec.score,
        )

        # 盤面ぷよ合計(発火側)
        fire_puyo = fire_board.count_puyos()
        phase = _classify_phase(float(fire_puyo), puyo_q_low, puyo_q_high)

        # won は発火側視点
        won_val = float(fire_rec.won[fi])

        row = {
            "video_id": fire_rec.video_id,
            "game_idx": int(fire_rec.game_idx[fi]),
            "t_sec": t_fire,
            "fire_side": fire_side,
            "phase": phase,
            "won": won_val,
            # 発火側特徴
            **{f"fire_{k}": v for k, v in fire_feats.items()},
            "fire_honsen_tempo_output": hto_fire.raw,
            # 相手側特徴
            **{f"opp_{k}": v for k, v in opp_feats.items()},
            "opp_honsen_tempo_output": hto_opp.raw,
            # 差分特徴
            **{f"diff_{k}": fire_feats[k] - opp_feats[k] for k in fire_feats},
            "diff_honsen_tempo_output": hto_fire.raw - hto_opp.raw,
            # 既存ラベル(互換維持)
            "net_ojama": net_oj,
            "returned": returned,
            # A-1: 競合返し / 返し窓メタ情報
            "returned_competitive": returned_competitive,
            "return_window_sec": float(return_window),
            "approx_fire_chains": float(approx_chains),
            "opp_buried": opp_buried,
            # user 確定定義: taiou_success (v2 新規ラベル)
            "taiou_success": taiou_success,
            "survived": survived,
            "net_ojama_after": net_ojama_after,
        }
        rows.append(row)
    return rows


def _estimate_puyo_quantiles(npz_paths: list[Path]) -> tuple[float, float]:
    """全npzから盤面ぷよ合計の33%/67%分位点を概算する(軽量サンプリング)。"""
    totals: list[float] = []
    for p in npz_paths[:30]:  # 最初の30本で推定(全数は重い)
        with np.load(p, allow_pickle=True) as d:
            grids = d["grids"].astype(np.int8)
        for g in grids[::5]:  # 5フレームごとサンプル
            cnt = int((g != 0).sum())
            totals.append(float(cnt))
    arr = np.array(totals)
    return float(np.quantile(arr, PHASE_QUANTILE_LOW)), float(np.quantile(arr, PHASE_QUANTILE_HIGH))


def _parse_args() -> "argparse.Namespace":
    """CLI 引数をパースする。省略時は既定 (boards_lean_fixed) で従来挙動と完全一致。"""
    import argparse
    parser = argparse.ArgumentParser(description="発火イベント+打ち合い結果ラベル生成")
    parser.add_argument(
        "--npz-dir", type=Path, default=NPZ_DIR,
        help=f"入力 npz ディレクトリ (既定: {NPZ_DIR})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"出力 CSV パス (既定: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--exclude-videos", type=str, default="",
        help=(
            "除外する動画 ID をカンマ区切りで指定 (例: c26,c30,c58,c69)。"
            " 既定は空文字列 = 除外なし(旧挙動と完全一致)。"
            f" score OCR 破綻動画の定数は SCORE_OCR_BROKEN_VIDEOS を参照。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """メイン処理。"""
    warnings.filterwarnings("ignore")
    args = _parse_args()
    npz_dir: Path = args.npz_dir
    output_path: Path = args.output
    exclude_ids: set[str] = {
        v.strip() for v in args.exclude_videos.split(",") if v.strip()
    }
    npz_paths = sorted(npz_dir.glob("c*.npz"))
    if exclude_ids:
        before = len(npz_paths)
        npz_paths = [p for p in npz_paths if p.stem not in exclude_ids]
        print(f"[INFO] --exclude-videos で {before - len(npz_paths)} 本除外: "
              f"{sorted(exclude_ids)}")
    if not npz_paths:
        print(f"[ERROR] npz が見つかりません: {npz_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] npz {len(npz_paths)} 本を処理します ({npz_dir})")

    # 位相分位点を事前推定
    q_low, q_high = _estimate_puyo_quantiles(npz_paths)
    print(f"[INFO] 位相閾値: 序≤{q_low:.1f} 中≤{q_high:.1f} 終>{q_high:.1f}")

    sim = ChainSimulator()
    all_rows: list[dict] = []

    for npz_path in npz_paths:
        records = _load_npz(npz_path)
        # game_idx ごとに 1P/2P をペアにする
        by_side: dict[str, NpzRecord] = {r.side: r for r in records}
        if "1P" not in by_side or "2P" not in by_side:
            continue

        r1p = by_side["1P"]
        r2p = by_side["2P"]

        # game_idx の unique リスト
        game_ids = np.unique(r1p.game_idx)
        for gid in game_ids:
            mask1 = r1p.game_idx == gid
            mask2 = r2p.game_idx == gid
            if not mask1.any() or not mask2.any():
                continue
            g1p = NpzRecord(
                video_id=r1p.video_id, side="1P",
                t_sec=r1p.t_sec[mask1], game_idx=r1p.game_idx[mask1],
                grids=r1p.grids[mask1], won=r1p.won[mask1],
                score=r1p.score[mask1],
                next1_a=r1p.next1_a[mask1], next1_b=r1p.next1_b[mask1],
                dnext_a=r1p.dnext_a[mask1], dnext_b=r1p.dnext_b[mask1],
            )
            g2p = NpzRecord(
                video_id=r2p.video_id, side="2P",
                t_sec=r2p.t_sec[mask2], game_idx=r2p.game_idx[mask2],
                grids=r2p.grids[mask2], won=r2p.won[mask2],
                score=r2p.score[mask2],
                next1_a=r2p.next1_a[mask2], next1_b=r2p.next1_b[mask2],
                dnext_a=r2p.dnext_a[mask2], dnext_b=r2p.dnext_b[mask2],
            )
            for side in ("1P", "2P"):
                try:
                    rows = _process_game(g1p, g2p, side, sim, q_low, q_high)
                    all_rows.extend(rows)
                except Exception as e:
                    print(f"[WARN] {npz_path.stem} game={gid} side={side}: {e}", file=sys.stderr)

        print(f"  {npz_path.stem}: 累計 {len(all_rows)} 発火イベント")

    if not all_rows:
        print("[ERROR] 発火イベントが 0 件でした。score データを確認してください。", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[DONE] {len(df)} 行を {output_path} に保存しました")
    print(f"  位相別: {df['phase'].value_counts().to_dict()}")
    print(f"  fire_side: {df['fire_side'].value_counts().to_dict()}")
    print(f"  net_ojama mean={df['net_ojama'].mean():.2f}  returned={df['returned'].mean():.2f}"
          f"  returned_competitive={df['returned_competitive'].mean():.3f}"
          f"  opp_buried={df['opp_buried'].mean():.3f}")
    print(f"  return_window_sec mean={df['return_window_sec'].mean():.2f}"
          f"  approx_fire_chains mean={df['approx_fire_chains'].mean():.2f}")
    # v2 新規ラベル統計
    print(f"  [v2] taiou_success={df['taiou_success'].mean():.3f}"
          f"  survived={df['survived'].mean():.3f}"
          f"  net_ojama_after mean={df['net_ojama_after'].mean():.2f}")
    # 位相別 taiou_success 発生率
    for ph in ["序", "中", "終"]:
        sub = df[df["phase"] == ph]
        if len(sub) > 0:
            print(f"    {ph}: taiou_success={sub['taiou_success'].mean():.3f}"
                  f"  n={len(sub)}")


if __name__ == "__main__":
    main()
