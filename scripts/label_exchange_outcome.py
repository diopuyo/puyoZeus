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
from dataclasses import dataclass
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

from src.board import Board, COLOR_EMPTY, COLOR_UNKNOWN  # noqa: E402
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
    SEC_PER_HAND,
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

# 発火イベント分裂統合 (測定器事故5件目、2026-08-02 v2: user確定回答2件を反映)
# 連鎖アニメ中もスコアOCRが部分合計を拾える瞬間があり、1つの連鎖が
# _detect_fire_events で複数イベントに分裂する (実データ c27.npz 1P game12:
# 1269.2秒score=27590の"326個イベント"と1271.3秒score=35274の"109個イベント"が
# 実際は同一9連鎖、実画面確認済み。「連鎖中に一瞬通常表示のスコアが出るのは
# ゲーム仕様」とuser確定)。
#
# v2 (2026-08-02): user確定回答「同一プレイヤーの連鎖の最短間隔は2秒くらい」を
# 受け、副信号のgap閾値を2.5秒→1.5秒に引き下げ (物理最短2秒を確実に下回る
# 保険のみに限定、2.0-2.5秒帯域の本物の連続発火 [高速の撃ち合い] を誤マージ
# しないため)。
#
# v3 (2026-08-02、main実測診断で確定): v2の主判定「盤面凍結走査」(候補検出j
# 自身の盤面が参照と完全一致するか) は、連鎖終了→盤面が連鎖後STABLEに更新
# →最終スコア確定 (=検出j) という正当な順序でも j の盤面が参照と変わる
# ため、gap1.5-2.2秒帯域152件中148件を誤って分離していた (main実測)。
# v3は主判定を「設置の署名 (ぷよ総数が+2以上増え、かつ自己修復せず持続する
# 瞬間) の有無」に全面変更する。増加が無い (凍結のまま or 連鎖後の減少のみ)
# なら盤面が参照と異なっていてもマージする。gap≤1.5秒の無条件マージ (副信号)
# は維持する。
FIRE_EVENT_MERGE_GAP_SEC: float = 1.5

# 設置の署名: ぷよ総数がこの個数以上増えたら「新しい手が置かれた」証拠とする
# (1手=2ぷよ、user確定「ぷよ総数+2以上」)。
PLACEMENT_SIGNATURE_MIN_INCREASE: int = 2

# 設置署名の持続判定秒数。
# ⚠️ 実装検証中に発見した重要な補正: 「認識ノイズの自己修復秒数 (project_
# yardstick_first_results_2026-07-31 の≤1秒)」をそのまま流用すると、実データ
# c27.npz 1P game12 の index47→52 区間 (本物の設置+連鎖: ぷよ数49→53が0.9秒
# かけて増加した直後、連鎖が発火して37まで急減する) を「1秒以内に解消した
# ノイズ」と誤判定し、本物の別イベントをマージしてしまう回帰が発生した
# (自己テストで発見)。設置は連鎖発火より先に完了するはずなので、判定秒数は
# 「1手の所要時間」(SEC_PER_HAND、実測中央値) を使うのが物理的に正しい
# (1手分の設置動作が完了する前に連鎖でぷよが消えることは無い)。実データの
# 認識ノイズ (0.2秒で解消) はこの秒数を大きく下回るため誤判定しない。
PLACEMENT_SIGNATURE_PERSIST_SEC: float = SEC_PER_HAND

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


@dataclass(frozen=True)
class FireEventCluster:
    """分裂した発火検出インデックスを1連鎖に統合したクラスタ (測定器事故5件目対策)。

    Attributes:
        fire_index: クラスタ最後の検出インデックス (連鎖終了時点、t_fire・opp側
            突合の基準に使う)。
        board_ref_index: クラスタ先頭の検出インデックスの1つ前 (真の連鎖開始前
            盤面、連鎖中の中間/凍結スナップショットではない点に注意)。
        baseline_score: クラスタ先頭直前の有効スコア (部分加算を合算した
            delta_score 計算の基準値)。
    """
    fire_index: int
    board_ref_index: int
    baseline_score: int


def _last_valid_score_before(score: np.ndarray, index: int) -> int:
    """index より前で最後に有効 (>=0) だったスコアを返す (無ければ-1)。"""
    for j in range(index - 1, -1, -1):
        if score[j] >= 0:
            return int(score[j])
    return -1


def _count_concrete_puyos(grid: np.ndarray) -> int:
    """gridの「確定した色」セル数を返す (空(0)とUNKNOWN(10)を除外)。

    Board.height_of() と同じ方針 (UNKNOWNは判定不能につき除外)。遷移汚染で
    UNKNOWNセルが湧いても、それを非ゼロ扱いして偽の増加署名を作らないため
    (main指摘事項、v3で追加検討)。
    """
    return int(np.sum((grid != COLOR_EMPTY) & (grid != COLOR_UNKNOWN)))


def _has_placement_signature(
    t_sec: np.ndarray,
    grids: np.ndarray,
    prev_idx: int,
    idx: int,
) -> bool:
    """prev_idx と idx の間で「設置の署名」(ぷよ総数の持続的な+2以上増加) が

    あったか判定する (v3主判定、測定器事故5件目の再修正)。

    v2の反省 (main実測診断で確定): 候補検出 idx 自身の盤面が参照盤面と
    一致するかで判定すると、連鎖終了→盤面が連鎖後STABLEに更新→最終スコア
    確定 (=検出idx) という正当な順序でも「不一致」と誤判定してしまう
    (gap1.5-2.2秒帯域152件中148件が誤って分離されていた)。v3はぷよ総数の
    「増加」のみに着目する (連鎖後の更新は通常ぷよが減る側なので誤判定しない)。

    認識ノイズによる短時間 (PLACEMENT_SIGNATURE_PERSIST_SEC 以下) の一時的な
    増加 (写り込み) は無視する (project_yardstick_first_results_2026-07-31の
    既知の自己修復パターン、実データ c27.npz 1P game12 で確認済み: +5個の
    一時的な写り込みが0.53秒で解消)。増加がこの秒数を超えて持続して初めて
    設置ありと確定する。走査範囲が終わるまで確定できなければ (=証拠不十分)
    保守的に「設置なし (マージ)」側に倒す (v2で「候補自身の差分を安易に
    確定証拠にした」反省を踏まえ、あえて逆方向の既定値にする)。
    """
    counts = [_count_concrete_puyos(grids[k]) for k in range(prev_idx, idx + 1)]
    baseline = counts[0]
    jump_start_t: "float | None" = None
    for offset in range(1, len(counts)):
        k = prev_idx + offset
        increase = counts[offset] - baseline
        if increase >= PLACEMENT_SIGNATURE_MIN_INCREASE:
            if jump_start_t is None:
                jump_start_t = float(t_sec[k])
            elif float(t_sec[k]) - jump_start_t > PLACEMENT_SIGNATURE_PERSIST_SEC:
                return True
        else:
            jump_start_t = None
            baseline = counts[offset]
    return False


def _merge_fire_event_clusters(
    t_sec: np.ndarray,
    score: np.ndarray,
    grids: np.ndarray,
    fire_indices: list[int],
) -> list[FireEventCluster]:
    """分裂した発火検出を連鎖単位にクラスタリングする (測定器事故5件目対策、v3)。

    分離 (=別イベント) と判定するのは、副信号 (短ギャップ) に該当せず、かつ
    検出 i (直前に採用した検出) と検出 j (候補) の間に「設置の署名」
    (_has_placement_signature、ぷよ総数の持続的な+2以上増加) がある場合のみ。
    署名が無ければ (凍結のまま、または連鎖後の更新で盤面が変わっていても)
    同一連鎖としてマージする。
      1. 副信号 (短ギャップ、v2でuser確定回答により2.5秒→1.5秒に引き下げ):
         同一プレイヤーの連鎖最短間隔は約2秒 (user確定) のため、それを
         確実に下回る1.5秒以下の隣接検出は無条件マージ。
      2. 主判定 (設置の署名の有無): 副信号に該当しない場合、設置の署名が
         無ければマージ、あれば分離する。
    """
    if not fire_indices:
        return []
    clusters: list[list[int]] = [[fire_indices[0]]]
    for idx in fire_indices[1:]:
        prev_idx = clusters[-1][-1]
        gap_sec = float(t_sec[idx] - t_sec[prev_idx])
        placement_found = _has_placement_signature(t_sec, grids, prev_idx, idx)
        if gap_sec <= FIRE_EVENT_MERGE_GAP_SEC or not placement_found:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])

    result: list[FireEventCluster] = []
    for cluster in clusters:
        first_idx, last_idx = cluster[0], cluster[-1]
        result.append(FireEventCluster(
            fire_index=last_idx,
            board_ref_index=max(0, first_idx - 1),
            baseline_score=_last_valid_score_before(score, first_idx),
        ))
    return result


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


# =============================================================================
# 終局イベント合成 (2026-08-03 main発注、方針(a)、既定OFF)
# =============================================================================
#
# 背景 (main実測 match_02、_diag_match02_underclamp_2026-08-03.py / _measure_
# terminal_chain_gap_2026-08-03.py で確定): 試合終了直前の大型連鎖は、連鎖
# アニメ中に「掛け算式」スコア表示が続いてOCRできず、その確定フレーム
# (=次の有効数値スコア) が記録区間の終端後に来ることがある。この場合、
# 発火検出器 (_detect_fire_events、スコア差分ベース) は最後に検出した
# クラスタ以降のスコア増分を一切イベント化できず、66動画で18.9%
# (試合×サイド) の「明確な終局連鎖の欠落」が定量化されている。
#
# 本関数は「最後に検出したクラスタ以降、試合末尾側の最後に有効なscoreまで
# の差分がSCORE_DELTA_FIRE以上」の場合に、1件の合成イベントを追加する。
# 盤面特徴は「最後に記録された実STABLE盤面」(=連鎖確定より前、最大で
# 試合終了までの全期間ぶん古い可能性がある) を使うしかないため、
# is_synthetic_terminal_event=1 列で必ず明示し、盤面特徴の信頼性が低い旨を
# 呼び出し側が判別できるようにする (fail-silent回避、CLAUDE.md原則)。

def _last_valid_score_index(score: np.ndarray) -> "int | None":
    """score配列内で最後に有効 (>=0) だったインデックスを返す (無ければNone)。"""
    valid_idx = np.where(score >= 0)[0]
    return int(valid_idx[-1]) if len(valid_idx) > 0 else None


def _synthesize_terminal_event_row(
    fire_rec: NpzRecord, opp_rec: NpzRecord, fire_side: str,
    fire_clusters: list[FireEventCluster], sim: ChainSimulator,
    game_start_t: float, puyo_q_low: float, puyo_q_high: float,
) -> "dict | None":
    """終局連鎖の欠落を1件の合成イベント行として補完する (検出できなければNone)。

    net_ojama_after は「相殺なしの全量」(_delta_to_ojama_standard) とする
    (試合終了直後で相手の反撃猶予窓を定義できないため)。taiou_success 等の
    「その後の相手の応手」に依存するラベルは判定不能として0/NaNにする
    (盤面特徴も古いため、合成行はモデル学習で除外/層別することを想定)。
    """
    last_valid_idx = _last_valid_score_index(fire_rec.score)
    if last_valid_idx is None:
        return None
    last_event_idx = fire_clusters[-1].fire_index if fire_clusters else -1
    baseline_score = (int(fire_rec.score[last_event_idx]) if fire_clusters
                       else int(fire_rec.score[np.where(fire_rec.score >= 0)[0][0]]))
    last_valid_score = int(fire_rec.score[last_valid_idx])
    gap = last_valid_score - baseline_score
    if gap < SCORE_DELTA_FIRE or last_valid_idx <= last_event_idx:
        return None

    t_fire = float(fire_rec.t_sec[last_valid_idx])
    fire_board = _board_from_grid(fire_rec.grids[last_valid_idx])
    opp_t_arr = opp_rec.t_sec
    nearest_opp = int(np.argmin(np.abs(opp_t_arr - t_fire)))
    opp_board = _board_from_grid(opp_rec.grids[nearest_opp])
    elapsed_in_game = _game_relative_elapsed(t_fire, game_start_t)
    fire_feats = _compute_features(fire_board, elapsed_in_game, sim)
    opp_feats = _compute_features(opp_board, elapsed_in_game, sim)
    fire_puyo = fire_board.count_puyos()
    phase = _classify_phase(float(fire_puyo), puyo_q_low, puyo_q_high)
    net_ojama_after = float(_delta_to_ojama_standard(gap))

    return {
        "video_id": fire_rec.video_id, "game_idx": int(fire_rec.game_idx[last_valid_idx]),
        "t_sec": t_fire, "fire_side": fire_side, "phase": phase,
        "won": float(fire_rec.won[last_valid_idx]),
        **{f"fire_{k}": v for k, v in fire_feats.items()},
        "fire_honsen_tempo_output": float("nan"),
        **{f"opp_{k}": v for k, v in opp_feats.items()},
        "opp_honsen_tempo_output": float("nan"),
        **{f"diff_{k}": fire_feats[k] - opp_feats[k] for k in fire_feats},
        "diff_honsen_tempo_output": float("nan"),
        "net_ojama": float(_delta_to_ojama_standard(gap)),
        "returned": 0, "returned_competitive": 0,
        "return_window_sec": float("nan"),
        # approx_fire_chains: 合成行は真の連鎖数が不明 (盤面が古いため
        # current_max_chain を実測できない) だが、augment_exchange_labels_
        # with_sim.py が estimate_available_hands(int(round(NaN))) で例外に
        # なるためNaNは使えない。fire_feats["current_max_chain"] (古い盤面
        # からの近似、通常行と同じ下限1.0のフォールバック) を代用する。
        "approx_fire_chains": max(1.0, fire_feats["current_max_chain"]),
        "opp_buried": 0, "taiou_success": 0, "survived": 0,
        "net_ojama_after": net_ojama_after,
        "is_synthetic_terminal_event": 1,
    }


def _process_game(
    game_1p: NpzRecord,
    game_2p: NpzRecord,
    fire_side: str,
    sim: ChainSimulator,
    puyo_q_low: float,
    puyo_q_high: float,
    synthesize_terminal_events: bool = False,
) -> list[dict]:
    """1ゲーム・1サイドの発火イベントを処理してレコードリストを返す。

    synthesize_terminal_events: optional (既定False、後方互換)。Trueなら
    終局連鎖の欠落 (_synthesize_terminal_event_row) を1件追加で補完する。
    """
    if fire_side == "1P":
        fire_rec = game_1p
        opp_rec = game_2p
    else:
        fire_rec = game_2p
        opp_rec = game_1p

    game_start_t = float(fire_rec.t_sec[0]) if len(fire_rec.t_sec) > 0 else 0.0
    fire_indices = _detect_fire_events(fire_rec.t_sec, fire_rec.score)
    if not fire_indices:
        if not synthesize_terminal_events:
            return []
        # 発火が一度も検出されなかった試合でも終局連鎖が欠落している
        # 可能性はある (has_prior_event=False、_measure_terminal_chain_gap_
        # 2026-08-03.py の誠実性チェック参照、小刻み蓄積との混同に注意)。
        synth = _synthesize_terminal_event_row(
            fire_rec, opp_rec, fire_side, [], sim, game_start_t, puyo_q_low, puyo_q_high,
        )
        return [synth] if synth is not None else []
    # 測定器事故5件目対策: 連鎖アニメ中の分裂検出を連鎖単位に統合する
    # (_detect_fire_events 自体は他スクリプト [proto_net_threat.py 等] からも
    # 呼ばれる共有関数のため無改変、統合は本関数内でのみ行う)。
    fire_clusters = _merge_fire_event_clusters(
        fire_rec.t_sec, fire_rec.score, fire_rec.grids, fire_indices,
    )

    # 相手盤面を (t, Board) ペアで保持(重い計算は発火時点のみ)
    opp_boards: list[tuple[float, Board]] = []
    for i in range(len(opp_rec.t_sec)):
        t = float(opp_rec.t_sec[i])
        opp_boards.append((t, _board_from_grid(opp_rec.grids[i])))

    rows: list[dict] = []
    prev_fire_score = -1

    for cluster in fire_clusters:
        fi = cluster.fire_index
        t_fire = float(fire_rec.t_sec[fi])
        s_fire = int(fire_rec.score[fi])
        # Δscoreはクラスタ先頭直前の有効scoreとの差 (分裂した部分加算の合算)
        prev_valid = cluster.baseline_score
        if prev_valid < 0:
            continue
        delta_score = s_fire - prev_valid
        if delta_score < SCORE_DELTA_FIRE:
            continue

        # 発火直前盤面 (クラスタ先頭直前=真の連鎖開始前盤面。連鎖終了時点
        # [fi] の1つ前は連鎖中の中間/凍結スナップショットの場合があり不適)
        fire_board = _board_from_grid(fire_rec.grids[cluster.board_ref_index])

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
            "is_synthetic_terminal_event": 0,
        }
        rows.append(row)

    if synthesize_terminal_events:
        synth = _synthesize_terminal_event_row(
            fire_rec, opp_rec, fire_side, fire_clusters, sim, game_start_t, puyo_q_low, puyo_q_high,
        )
        if synth is not None:
            rows.append(synth)
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
    parser.add_argument(
        "--glob-pattern", type=str, default="c*.npz",
        help=(
            "npz ファイル名の glob パターン (既定 'c*.npz' = 従来のc系動画命名"
            "規約、後方互換)。2026-08-03 追加: c系以外の命名 (未知動画の汎化"
            "テスト等) の npz を処理する場合に '*.npz' 等へ変更する。"
        ),
    )
    parser.add_argument(
        "--synthesize-terminal-events", action="store_true", default=False,
        help=(
            "終局連鎖の欠落 (2026-08-03 main発注、方針(a)) を1件の合成イベント"
            "として補完する (既定OFF、後方互換)。合成行は is_synthetic_"
            "terminal_event=1 列で明示される (盤面特徴は最後の実STABLE盤面"
            "由来のため信頼性が低い、docstring参照)。"
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
    npz_paths = sorted(npz_dir.glob(args.glob_pattern))
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
                    rows = _process_game(g1p, g2p, side, sim, q_low, q_high,
                                          synthesize_terminal_events=args.synthesize_terminal_events)
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
    if args.synthesize_terminal_events:
        n_synth = int(df["is_synthetic_terminal_event"].sum())
        print(f"  [方針(a)] 合成した終局イベント: {n_synth} 行 ({n_synth / len(df):.1%})")
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
