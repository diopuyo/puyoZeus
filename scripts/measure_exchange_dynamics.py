"""打ち合い分析の計測器 (催促/対応/本線モデルの土台統計)。

memory `reference_saisoku_exchange_model_2026-07-22` (userと確定した催促モデル)
に基づき、指標本体を作る前に実データから土台統計を測る「計測器」。

処理内容:
    1. 発火イベント検出 (scripts/label_exchange_outcome.py の
       スコア増分ベース検出ロジックを土台に流用)。
    2. 各発火を「催促」(消費色ぷよ比 <60%) / 「本線」(>=60%) に分類。
    3. 応酬シーケンス追跡 (相手側の「催促」応答が仮の時間窓以内に
       続く限り1シーケンスとして連結)。
    4. (a) 時間係数 (b) 再帰深さ分布 (c) 返り量分布 + 催促/本線比率を
       ティア別 (チャレンジャー/マスター/S級) + 全体で集計・出力。

対象: チャレンジャー10・マスター10・S級3 = 計23動画 (ティア均等サンプル、
userタスク指定で固定)。c30 は既存 exchange_labels.csv に未収載のため、
本スクリプトは全動画で npz から自前で発火検出する (既存CSVには依存しない)。

⚠️ 本スクリプトは統計計測が目的であり指標本体 (0〜1正規化スコア) は
まだ実装しない。応酬窓の係数は仮値であり、(a) の実測結果を見て
別途キャリブレーションする前提 (memory 内 "感度を見て" の指示通り)。

使い方:
    PYTHONPATH=. python -m scripts.measure_exchange_dynamics
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

# スレッド制限 (熱暴走防止、feedback_thermal_safety_mandatory 準拠)
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "2")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import Board  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.indicators_v2 import SEC_PER_HAND as EXISTING_SEC_PER_HAND  # noqa: E402
from src.scoring import score_to_ojama  # noqa: E402

# ============================
# 定数定義
# ============================
NPZ_DIR: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed"
OUTPUT_CSV: Path = PROJ_ROOT / "data" / "indicators_v2" / "exchange_dynamics_stats.csv"

# 対象23動画のティア対応表 (userタスク指定、固定リスト)
TIER_CHALLENGER: tuple[str, ...] = (
    "c5", "c6", "c7", "c11", "c16", "c21", "c22", "c28", "c30", "c31",
)
TIER_MASTER: tuple[str, ...] = (
    "c44", "c51", "c53", "c54", "c59", "c62", "c68", "c73", "c78", "c80",
)
TIER_S_CLASS: tuple[str, ...] = ("c82", "c83", "c84")
TIER_MAP: dict[str, str] = {
    **{v: "チャレンジャー" for v in TIER_CHALLENGER},
    **{v: "マスター" for v in TIER_MASTER},
    **{v: "S級" for v in TIER_S_CLASS},
}

# 発火検出: スコア増分閾値。
# 2026-07-22 de-frag強化: 消去色ぷよ数ガード (MIN_CONSUMED_PUYO_FOR_FIRE) 併用を
# 前提に、旧 80 (label_exchange_outcome.py 由来) から 1連鎖の理論最小得点相当の
# 40 に引き下げ。ガードなしで 40/20 まで下げると score OCR ノイズを大量に拾って
# 送りお邪魔 median が 0 に潰れる副作用を実測済 (ガードで解消)。20 まで更に下げても
# 汚染率改善は誤差レベル (17.1% -> 17.0%) で頭打ちのため 40 を採用。
SCORE_DELTA_FIRE: int = 40

# score が全欠損 (c30等) の動画向けフォールバック検出時の delta_score センチネル値。
# 換算不能を明示するため -1 を使う (実データの delta_score は常に >=SCORE_DELTA_FIRE>0)。
SCORE_MISSING_SENTINEL: int = -1

# 催促/本線分類の閾値 (memory reference_saisoku_exchange_model_2026-07-22 確定値)
HONSEN_RATIO_THRESHOLD: float = 0.6

# 色ぷよ判定範囲 (COLOR_RED=1 〜 COLOR_PURPLE=5。おじゃま9・空0・不明10は除外)
COLOR_PUYO_MIN: int = 1
COLOR_PUYO_MAX: int = 5

# 応酬窓の仮係数 (userタスク指定の仮値。(a) の実測結果を見て別途調整する前提)
PROVISIONAL_WINDOW_CHAIN_COEF_SEC: float = 0.5
PROVISIONAL_WINDOW_BASE_SEC: float = 2.0

# 時間係数計測: 「次の相手発火」をペアリングする際の上限秒数 (無関係な発火除外)
MAX_PAIR_GAP_SEC: float = 60.0

# お邪魔は6列均等配分 (reference_ojama_landing_pattern.md 準拠) → 1段 = 6個。
# 2026-07-22 訂正: 以前 "列" とラベル付けしていたが誤り (userレビューで発覚)。
# 正しい単位は「個」(お邪魔の個数、生値) および「段」(個数/6、6列均等分配時に
# 各列に積まれる高さ)。"列" という表現は使わない。
OJAMA_PER_DAN: int = 6

# 返り量分類の閾値 (段単位。端数許容: ちょうど1段/2段を四捨五入的に許容する)
RETURN_ONE_DAN_UPPER: float = 1.2
RETURN_TWO_DAN_LOWER: float = 1.8

# 連鎖数ビン上限 (これ以上は "8+" にまとめる、サンプル数希薄化を防ぐ)
CHAIN_BIN_CAP: int = 8

# ノイズガード: 発火候補の消去色ぷよ数がこの数未満なら棄却する。
# score OCR の桁誤読・NCC ノイズによる偽の delta_score 増分は盤面上の色ぷよが
# 実際には消えていないため、このガードで機械的に除外できる (2026-07-22 強化)。
MIN_CONSUMED_PUYO_FOR_FIRE: int = 1

# 本物の連鎖判定ゲート (2026-07-22 追加): 発火前静止盤面を simulate() した
# chain_count がこの値未満なら「疑似発火」として棄却する。
#
# 背景: scripts/measure_ojama_landing_delay.py の着弾遅延実測で、検出された
# 発火イベントの 72.9% が simulate(発火前静止盤面).chain_count==0 (= 4連結
# ポップが実在しない) であることが判明した。MIN_CONSUMED_PUYO_FOR_FIRE=1 の
# ガードだけでは「色ぷよが1個以上消えた」ことしか保証せず、実際に連鎖として
# ポップしたかどうかは見ていなかったことが原因。
#
# 実データ比較 (23動画・1308発火イベント、scratchpad診断で実測):
#   Option A: chain_count>=1               -> 残存 355件 (27.1%)
#   Option B: consumed_color_puyos>=4      -> 残存 1020件 (78.0%、絞り込み不足)
# consumed>=4 でも chain_count==0 のままの疑似発火が 686 件残ってしまい
# (例: 消費57個なのに simulate()は非発火判定)、これは「発火前静止盤面」の
# フレーム選定タイミングのズレ (認識上のframe選択問題) が主因と推測される。
# 一方 chain_count>=1 は simulate() による連結判定そのものであり、
# 既知の大連鎖 (c5=14連鎖, c82=12連鎖, c62=10連鎖, c68=10連鎖, c83=13連鎖、
# いずれも simulate() 実測値。userレビュー値と一部ズレがあるが、本ゲートで
# 全て生存確認済み) を落とさずに疑似発火を大幅削減できるため、Option A を採用。
MIN_CHAIN_COUNT_FOR_FIRE: int = 1

# 得点増加要因は (1)落下ボーナス (ぷよ設置/落下ごとの+1程度の微増)
# (2)連鎖 (段ごとの大きな跳ね) (3)全消し (2100点上乗せ、連鎖に付随) の3つ
# (userドメイン知識、2026-07-22)。「連鎖の跳ね」とみなす増分の下限として、
# 1連鎖の理論最小得点 (4個消し=40点) より十分小さい安全域を設定する。
# 実データの delta 分布は 1〜50 点でなだらかでギャップが無いため、本値単独では
# なく後述の消去色ぷよ数ガードとの AND 条件で「跳ね」を判定する。
SMALL_INCREMENT_MAX: int = 8

# 連鎖跳ね判定の時間差上限 (秒)。score OCR 欠損等でこれを超えて離れた
# フレームペアは「跳ね」とみなさない (欠損中の複数連鎖の誤結合を防ぐ安全弁、
# 2026-07-22 バグ修正で導入)。実データで大連鎖 (15連鎖) が STABLE 復帰なく
# 9.4秒連続 NON-STABLE だった実例を確認したため、それを許容しつつ score OCR
# 誤読による誤結合 (23秒ギャップの実例あり) はできるだけ弾く妥協値として 12 秒
# を採用。この閾値だけでは完全に区別できないケースは動画クリップで user
# レビューに回す (ANOMALY_DELTA_SCORE_MIN 参照)。
MAX_CHAIN_JUMP_GAP_SEC: float = 12.0

# 要レビューフラグ: delta_score がこれを超えるイベントは、時間ギャップ判定
# だけでは真偽を機械的に確定できないため「要検証」候補として抽出する
# (2026-07-22、userレビュー方針)。19連鎖級の理論値を踏まえた閾値。
ANOMALY_DELTA_SCORE_MIN: int = 30000


class NpzRecord(NamedTuple):
    """1本・1サイド分の npz データ (このスクリプトで必要な列のみ)。"""
    video_id: str
    side: str
    t_sec: np.ndarray
    game_idx: np.ndarray
    grids: np.ndarray
    score: np.ndarray


@dataclass
class FireEvent:
    """1発火イベントのレコード (催促/本線分類 + シーケンス情報つき)。"""
    video_stem: str
    tier: str
    game_idx: int
    fire_side: str
    fi_idx: int
    t_fire: float
    delta_score: int  # -1 = スコアデータ欠損 (c30等、盤面ベース検出フォールバック時)
    chain_count: int
    ratio: float
    label: str  # "催促" / "本線" / "不明" (発火前色ぷよ0で分類不能)
    ojama_sent_count: float  # 送りお邪魔の個数 (生値)。スコア欠損時は NaN (換算不能)
    time_to_next_opp_fire_sec: float = float("nan")
    seq_id: int = -1
    seq_position: int = -1
    seq_depth: int = -1
    # de-frag (後処理連鎖分断解消): この発火イベントが何個の生断片から
    # 統合されたか。1 = 単独 (de-frag 対象外)。旧統計 (de-frag 前) では
    # 常に暗黙 1 として扱われる (フィールド自体が存在しないため)。
    frag_count: int = 1
    # pre-chain静止盤面 (連鎖開始直前、simulate入力) の時刻。
    # t_fire (= 発火確定後/post-chain時刻) と区別し、動画クリップ切り出し等で
    # 「連鎖の本当の開始」を指すために使う (2026-07-22 追加)。
    t_chain_start: float = float("nan")
    # pre-chain静止盤面の grids インデックス (de-frag マージ時に正しい
    # before を参照するため保持。fi_idx-1 という旧仮定は新方式 (window方式)
    # では成立しないため、明示的に保存する: 2026-07-22 バグ修正)。
    before_idx: int = -1


# ============================
# npz ロード
# ============================


def _load_npz(path: Path) -> list[NpzRecord]:
    """1つの npz から NpzRecord を側ごとに返す。"""
    with np.load(path, allow_pickle=True) as d:
        video_ids = d["video_id"]
        sides = d["side"]
        t_secs = d["t_sec"].astype(np.float32)
        game_idxs = d["game_idx"].astype(np.int32)
        grids = d["grids"].astype(np.int8)
        scores = d["score"].astype(np.int32)
    records: list[NpzRecord] = []
    for side_val in ("1P", "2P"):
        mask = sides == side_val
        if not mask.any():
            continue
        vid = str(video_ids[mask][0])
        records.append(NpzRecord(
            video_id=vid, side=side_val,
            t_sec=t_secs[mask], game_idx=game_idxs[mask],
            grids=grids[mask], score=scores[mask],
        ))
    return records


def _subset(rec: NpzRecord, mask: np.ndarray) -> NpzRecord:
    """NpzRecord をブールマスクで部分抽出する。"""
    return NpzRecord(
        video_id=rec.video_id, side=rec.side,
        t_sec=rec.t_sec[mask], game_idx=rec.game_idx[mask],
        grids=rec.grids[mask], score=rec.score[mask],
    )


# ============================
# 催促/本線分類
# ============================


def _count_color_puyos(grid: np.ndarray) -> int:
    """盤面グリッドの色ぷよ数 (おじゃま・空・不明を除く) を返す。"""
    return int(((grid >= COLOR_PUYO_MIN) & (grid <= COLOR_PUYO_MAX)).sum())


def _consumed_color_puyos(before_grid: np.ndarray, after_grid: np.ndarray) -> int:
    """発火前後の色ぷよ消費数 (負値は 0 にクランプ)。

    ノイズガード用: score OCR ノイズ由来の偽発火は盤面上の色ぷよが実際には
    消えていない (= 0) ため、これで機械的に検出・除外できる。
    """
    return max(0, _count_color_puyos(before_grid) - _count_color_puyos(after_grid))


def _classify_exchange(before_grid: np.ndarray, after_grid: np.ndarray) -> tuple[str, float]:
    """発火前後の色ぷよ消費比から催促/本線を分類する。

    ratio = 消費色ぷよ数 / 発火前色ぷよ数。0.6 未満なら催促、以上なら本線。
    発火前色ぷよ数が0の場合は分類不能 ("不明") として返す。
    """
    before_n = _count_color_puyos(before_grid)
    after_n = _count_color_puyos(after_grid)
    if before_n <= 0:
        return "不明", float("nan")
    consumed = max(0, before_n - after_n)
    ratio = consumed / before_n
    label = "催促" if ratio < HONSEN_RATIO_THRESHOLD else "本線"
    return label, ratio


def _ojama_sent_count(delta_score: int, elapsed_sec: float) -> float:
    """Δscore をマージンタイム考慮でお邪魔個数 (生値) に換算する。

    段単位が欲しい場合は呼び出し側で /OJAMA_PER_DAN する (2026-07-22 訂正:
    以前は本関数内で /6 していたが "列" という誤ラベルの温床だったため、
    生値 (個) を返す設計に変更した)。
    """
    result = score_to_ojama(max(0, delta_score), prev_leftover=0, elapsed_sec=elapsed_sec)
    return float(result.ojama_count)


def _provisional_window_sec(chain_count: float) -> float:
    """応酬追跡用の仮窓秒数 (x連鎖×係数+基礎秒、感度確認用の仮値)。"""
    return chain_count * PROVISIONAL_WINDOW_CHAIN_COEF_SEC + PROVISIONAL_WINDOW_BASE_SEC


# ============================
# 1 (video, game, side) 単位の処理: pre-chain静止盤面アンカー方式
# ============================
#
# naiveなスコア増分アンカー (score が閾値を超えたフレームをそのまま発火とみなす)
# は、大連鎖の後半 (得点が跳ね上がる段) だけを検出し、連鎖の途中から誤って
# アンカリングしてしまう「末尾アンカー」問題があった (2026-07-22 userレビューで
# 15/7/13連鎖が末尾のみで検出されていたことが判明)。
#
# 対策: VideoChainTracker (src/chain_detector.py:210-243) と同じ考え方で、
# 「発火直前の静止盤面 (pre-chain snapshot) まで遡り、そこから simulate() で
# 連鎖全体を一括復元」する方式に変更する。


def _passes_consumed_guard(rec: NpzRecord, pre_idx: int, post_idx: int) -> bool:
    """ノイズガード: pre_idx -> post_idx 間の消去色ぷよ数が閾値未満なら False。

    落下ボーナスは色ぷよを消さないため、このガードだけで機械的に除外できる。
    """
    consumed = _consumed_color_puyos(rec.grids[pre_idx], rec.grids[post_idx])
    return consumed >= MIN_CONSUMED_PUYO_FOR_FIRE


def _is_chain_jump(rec: NpzRecord, idx_a: int, idx_b: int) -> bool:
    """idx_a -> idx_b の得点増分が「連鎖の跳ね」かどうかを判定する。

    得点増加要因は (1)落下ボーナス (+1程度の微増、色ぷよ非消費)
    (2)連鎖 (段ごとの大跳ね、色ぷよ消費) (3)全消し (連鎖に付随) の3つ
    (userドメイン知識)。「跳ね」= delta > SMALL_INCREMENT_MAX **かつ**
    消去色ぷよ数 >= 1。どちらか一方でも満たさなければ静止 (落下ボーナスの
    累積) とみなす。

    ⚠️ 安全弁 (2026-07-22 バグ修正): score OCR が一時的に読めず idx_a/idx_b
    間が長時間 (例: NON-STABLE や連鎖中エフェクトで数十秒) 離れているのに
    合算 delta だけで判定すると、その間に起きた複数回の連鎖分を1つの巨大な
    跳ねとして誤検出する (実測で delta_score>10万点の異常値を確認、
    修正済)。時間差が MAX_CHAIN_JUMP_GAP_SEC を超えるペアは跳ねとみなさない。
    """
    time_gap = float(rec.t_sec[idx_b]) - float(rec.t_sec[idx_a])
    if time_gap > MAX_CHAIN_JUMP_GAP_SEC:
        return False
    delta = int(rec.score[idx_b]) - int(rec.score[idx_a])
    if delta <= SMALL_INCREMENT_MAX:
        return False
    return _passes_consumed_guard(rec, idx_a, idx_b)


def _find_chain_windows(rec: NpzRecord) -> list[tuple[int, int]]:
    """pre-chain静止盤面と post-chain確定盤面の (pre_idx, post_idx) 一覧を返す。

    VideoChainTracker (src/chain_detector.py:186-250) と同じ設計を採用する:
    「直前の静止フレームとの比較のみ」で跳ねを判定し (跳ねの連鎖的追跡は
    しない)、跳ね検知のたびに直前静止を pre、現在を post として1イベント
    確定 → その場で静止フレームを現在に更新して次へ進む。

    ⚠️ 過去の実装ミス (2026-07-22): 「跳ねが継続する限り window を延長し
    続ける」設計を一時試したが、対戦中盤に断続的な得点上昇が続くケースで
    window が試合の大半を飲み込み delta_score が10万点超に暴走する重大
    バグを起こした (実測で確認、即修正)。本関数は都度リセットする設計に
    より、この暴走を構造的に防ぐ。
    """
    score = rec.score
    valid_idx = [i for i in range(len(score)) if int(score[i]) >= 0]
    windows: list[tuple[int, int]] = []
    if len(valid_idx) < 2:
        return windows
    stable_idx = valid_idx[0]
    for cur in valid_idx[1:]:
        if _is_chain_jump(rec, stable_idx, cur):
            windows.append((stable_idx, cur))
        stable_idx = cur
    return windows


def _find_chain_windows_by_board(rec: NpzRecord) -> list[tuple[int, int]]:
    """色ぷよ数の変化から (pre_idx, post_idx) を検出する (score全欠損動画向け)。

    _find_chain_windows のスコア版と同じ設計 (直前の静止フレームとの比較
    のみで判定、都度更新)。念のため時間差ガードも併用する。
    """
    grids = rec.grids
    windows: list[tuple[int, int]] = []
    n = len(grids)
    if n < 2:
        return windows
    stable_idx = 0
    stable_count = _count_color_puyos(grids[0])
    for cur in range(1, n):
        cur_count = _count_color_puyos(grids[cur])
        time_gap = float(rec.t_sec[cur]) - float(rec.t_sec[stable_idx])
        if (
            stable_count - cur_count >= MIN_CONSUMED_PUYO_FOR_FIRE
            and time_gap <= MAX_CHAIN_JUMP_GAP_SEC
        ):
            windows.append((stable_idx, cur))
        stable_idx = cur
        stable_count = cur_count
    return windows


def _build_fire_event(
    rec: NpzRecord,
    fi: int,
    delta_score: int,
    sim: ChainSimulator,
    game_start_t: float,
    tier: str,
    video_stem: str,
    before_idx: int | None = None,
) -> FireEvent:
    """1発火インデックス分の FireEvent を組み立てる (score有無どちらの検出でも共通)。

    Args:
        before_idx: pre-chain静止盤面 (simulate 入力) の grids インデックス。
            省略時は従来通り fi-1 (既存呼び出しは省略のままなので挙動不変、
            backwards compat)。
    """
    b_idx = before_idx if before_idx is not None else max(0, fi - 1)
    before_grid = rec.grids[b_idx]
    after_grid = rec.grids[fi]
    label, ratio = _classify_exchange(before_grid, after_grid)

    try:
        before_board = Board.from_list(before_grid.tolist())
        chain_count = sim.simulate(before_board).chain_count
    except Exception:
        chain_count = 0

    t_fire = float(rec.t_sec[fi])
    t_chain_start = float(rec.t_sec[b_idx])
    elapsed = max(0.0, t_fire - game_start_t)
    has_score = delta_score != SCORE_MISSING_SENTINEL
    ojama_count = _ojama_sent_count(delta_score, elapsed) if has_score else float("nan")

    return FireEvent(
        video_stem=video_stem, tier=tier,
        game_idx=int(rec.game_idx[fi]), fire_side=rec.side,
        fi_idx=int(fi), t_fire=t_fire, delta_score=delta_score,
        chain_count=chain_count, ratio=ratio, label=label,
        ojama_sent_count=ojama_count, t_chain_start=t_chain_start,
        before_idx=int(b_idx),
    )


def _passes_real_chain_gate(event: FireEvent) -> bool:
    """本物の連鎖判定ゲート (MIN_CHAIN_COUNT_FOR_FIRE 参照)。

    simulate(発火前静止盤面) が実際に連結ポップを検出した (chain_count>=1)
    発火のみを「本物」とみなし、疑似発火 (score/消費数ノイズ由来) を棄却する。
    """
    return event.chain_count >= MIN_CHAIN_COUNT_FOR_FIRE


def _process_side_game(
    rec: NpzRecord,
    sim: ChainSimulator,
    game_start_t: float,
    tier: str,
    video_stem: str,
) -> list[FireEvent]:
    """1 (video, game, side) 分の発火イベントを検出し FireEvent を返す。

    pre-chain静止盤面アンカー方式 (_find_chain_windows) で発火の開始/終了を
    検出し、そこから simulate() で連鎖全体を復元する。score が1件でも有効なら
    スコア方式、全欠損 (c30等) なら色ぷよ数方式のフォールバックを使う。
    いずれの方式でも、最終的に _passes_real_chain_gate (chain_count>=1) を
    満たさない疑似発火は棄却する (2026-07-22 追加)。
    """
    has_score = bool((rec.score >= 0).any())
    events: list[FireEvent] = []
    if has_score:
        for pre_idx, post_idx in _find_chain_windows(rec):
            delta_score = int(rec.score[post_idx]) - int(rec.score[pre_idx])
            if delta_score < SCORE_DELTA_FIRE:
                continue
            if not _passes_consumed_guard(rec, pre_idx, post_idx):
                continue
            event = _build_fire_event(
                rec, post_idx, delta_score, sim, game_start_t, tier, video_stem,
                before_idx=pre_idx,
            )
            if _passes_real_chain_gate(event):
                events.append(event)
    else:
        for pre_idx, post_idx in _find_chain_windows_by_board(rec):
            if not _passes_consumed_guard(rec, pre_idx, post_idx):
                continue
            event = _build_fire_event(
                rec, post_idx, SCORE_MISSING_SENTINEL, sim, game_start_t, tier, video_stem,
                before_idx=pre_idx,
            )
            if _passes_real_chain_gate(event):
                events.append(event)
    return events


# ============================
# de-frag (後処理連鎖分断解消)
# ============================
#
# 発火直前盤面 (grids[fi-1]) を simulate() した結果 chain_count>=1 なら
# 「前フレームでまだ消去アニメ途中の残存連鎖」を捉えた誤検出であり、
# 新規発火ではなく直前発火の継続とみなす。継続分は直前イベントに畳み込み、
# 1本の連鎖 = 1 FireEvent として統合し直す。


def _merge_fragment_group(
    group: list[FireEvent],
    rec: NpzRecord,
    sim: ChainSimulator,
    game_start_t: float,
    tier: str,
    video_stem: str,
) -> FireEvent:
    """1 グループ (継続断片込み) を 1 つの FireEvent に統合する。

    before は先頭断片の発火直前盤面、after は末尾断片の発火後盤面を使い、
    delta_score は各断片の合算 (SCORE_MISSING_SENTINEL 混在時は欠損扱い)。
    chain_count/ratio/label/ojama_sent_count は統合後の盤面ペアで再計算する
    (_build_fire_event に委譲、 fi_idx/before_idx のみ差し替え)。
    """
    if len(group) == 1:
        group[0].frag_count = 1
        return group[0]
    # 2026-07-22 バグ修正: 旧実装は before_idx = fi_idx-1 と仮定していたが、
    # window方式では before (pre-chain静止盤面) は fi_idx から大きく離れうる
    # ため、FireEvent.before_idx (実際に simulate に使った値) を直接使う。
    before_idx = group[0].before_idx
    after_idx = group[-1].fi_idx
    deltas = [e.delta_score for e in group]
    has_missing = any(d == SCORE_MISSING_SENTINEL for d in deltas)
    merged_delta = SCORE_MISSING_SENTINEL if has_missing else sum(deltas)
    merged = _build_fire_event(
        rec, after_idx, merged_delta, sim, game_start_t, tier, video_stem,
        before_idx=before_idx,
    )
    merged.frag_count = len(group)
    return merged


# de-frag マージの時間差上限 (秒)。chain_count>=1 でもこれを超える間隔なら
# 無関係な発火とみなしマージしない (安全弁)。実データで t_chain_start と
# 直前発火の t_fire が数十秒離れているのに chain_count>=1 判定だけで
# 誤って結合される暴走を実測で確認したため導入 (2026-07-22)。
DEFRAG_MAX_GAP_SEC: float = 5.0


def _defrag_events(
    raw_events: list[FireEvent],
    rec: NpzRecord,
    sim: ChainSimulator,
    game_start_t: float,
    tier: str,
    video_stem: str,
) -> list[FireEvent]:
    """raw_events (t_fire 昇順、同一 side) を de-frag ルールで畳み込む。

    判定: raw_events[i].chain_count (= 発火直前盤面の simulate 結果) が
    >=1 **かつ** 直前グループ末尾との時間差が DEFRAG_MAX_GAP_SEC 以内なら
    「残存連鎖の続き」とみなし直前グループに連結。それ以外は新規グループを
    開始する (先頭要素は無条件で新規グループ)。

    ⚠️ 既知の副作用 (2026-07-22、MIN_CHAIN_COUNT_FOR_FIRE 導入時に判明):
    raw_events は既に _passes_real_chain_gate (chain_count>=1) を通過済みの
    ため、本関数内の `ev.chain_count >= 1` は常に真になり、実質的に
    DEFRAG_MAX_GAP_SEC のみが判定基準になっている。既知の大連鎖5本
    (c5/c62/c68/c82/c83) は正しく1イベントに統合され chain_count も保たれる
    ことを実測確認済みだが、理論上は「5秒以内に発生した無関係な2つの独立
    した本物の連鎖」を誤って1つに畳み込むリスクが残る (frag_count>1の大半は
    大連鎖=正当な分断再結合と実測で確認済みだが、完全な保証ではない)。
    """
    if not raw_events:
        return []
    groups: list[list[FireEvent]] = [[raw_events[0]]]
    for ev in raw_events[1:]:
        gap = ev.t_chain_start - groups[-1][-1].t_fire
        if ev.chain_count >= 1 and gap <= DEFRAG_MAX_GAP_SEC:
            groups[-1].append(ev)
        else:
            groups.append([ev])
    return [
        _merge_fragment_group(grp, rec, sim, game_start_t, tier, video_stem)
        for grp in groups
    ]


def _annotate_next_opp_fire(events: list[FireEvent]) -> None:
    """各イベントに「次の相手側発火までの秒数」を付与する (in-place、events は t_fire 昇順)。

    MAX_PAIR_GAP_SEC を超える、または該当なしの場合は NaN のまま。
    """
    for i, ev in enumerate(events):
        for j in range(i + 1, len(events)):
            other = events[j]
            if other.fire_side == ev.fire_side:
                continue
            gap = other.t_fire - ev.t_fire
            if gap <= MAX_PAIR_GAP_SEC:
                ev.time_to_next_opp_fire_sec = gap
            break  # 直近の相手側発火のみ見る (それ以降はさらに遠い)


def _build_sequences(events: list[FireEvent], start_seq_id: int) -> int:
    """催促の応酬シーケンスを検出し、各イベントに seq_id/seq_position/seq_depth を付与する。

    定義 (memory reference_saisoku_exchange_model_2026-07-22 準拠):
    直前発火の相手側 かつ「催促」ラベル かつ 仮窓秒数以内の発火のみを
    「対応」として連結する。本線での返しや窓超過はシーケンスを終端する。

    Args:
        events: 1ゲーム分の発火イベント (t_fire 昇順)。
        start_seq_id: このゲームで使い始める seq_id (全体で一意にするため)。

    Returns:
        次に使う seq_id (呼び出し元で引き継ぐ)。
    """
    seq_id = start_seq_id
    i = 0
    n = len(events)
    while i < n:
        seq = [events[i]]
        j = i + 1
        while j < n:
            prev = seq[-1]
            cur = events[j]
            if cur.fire_side == prev.fire_side:
                break
            window = _provisional_window_sec(float(prev.chain_count))
            gap = cur.t_fire - prev.t_fire
            if gap <= window and cur.label == "催促":
                seq.append(cur)
                j += 1
            else:
                break
        depth = len(seq)
        for pos, ev in enumerate(seq, start=1):
            ev.seq_id = seq_id
            ev.seq_position = pos
            ev.seq_depth = depth
        seq_id += 1
        i = j if j > i else i + 1
    return seq_id


# ============================
# 1 動画単位の処理
# ============================


def _process_video(
    npz_path: Path, sim: ChainSimulator, seq_id_start: int,
) -> tuple[list[FireEvent], list[FireEvent], int]:
    """1 npz (1動画) 分を処理し (raw, de-frag後) FireEvent 一覧と次の seq_id を返す。

    raw は de-frag 前 (旧ロジック相当、比較レポート専用)、
    de-frag 後の一覧に対してのみ応酬シーケンス付与 (_build_sequences) を行う。
    """
    video_stem = npz_path.stem
    tier = TIER_MAP.get(video_stem, "不明")
    records = _load_npz(npz_path)
    by_side = {r.side: r for r in records}
    if "1P" not in by_side or "2P" not in by_side:
        return [], [], seq_id_start

    r1p, r2p = by_side["1P"], by_side["2P"]
    game_ids = np.unique(r1p.game_idx)
    all_raw: list[FireEvent] = []
    all_defrag: list[FireEvent] = []
    seq_id = seq_id_start
    for gid in game_ids:
        m1 = r1p.game_idx == gid
        m2 = r2p.game_idx == gid
        if not m1.any() or not m2.any():
            continue
        g1p = _subset(r1p, m1)
        g2p = _subset(r2p, m2)
        game_start_t = float(min(g1p.t_sec[0], g2p.t_sec[0]))

        raw1 = _process_side_game(g1p, sim, game_start_t, tier, video_stem)
        raw2 = _process_side_game(g2p, sim, game_start_t, tier, video_stem)
        all_raw.extend(raw1)
        all_raw.extend(raw2)

        defrag1 = _defrag_events(raw1, g1p, sim, game_start_t, tier, video_stem)
        defrag2 = _defrag_events(raw2, g2p, sim, game_start_t, tier, video_stem)
        game_events = sorted(defrag1 + defrag2, key=lambda e: e.t_fire)
        _annotate_next_opp_fire(game_events)
        seq_id = _build_sequences(game_events, seq_id)
        all_defrag.extend(game_events)
    return all_raw, all_defrag, seq_id


# ============================
# レポート集計
# ============================


def _report_label_ratio(sub: pd.DataFrame) -> pd.Series:
    """催促/本線/不明の出現比率を返す。"""
    return sub["label"].value_counts(normalize=True)


def _report_time_coefficient(sub: pd.DataFrame) -> pd.DataFrame:
    """連鎖数 x 次相手発火までの時間 の集計表を返す (連鎖数ビンごとの中央値等)。

    (a) 「x連鎖 ≈ y秒 ≈ z手」の実測。手数は既存 SEC_PER_HAND (0.733秒/手)
    で換算 (指標側の既存仮定との比較用)。
    """
    valid = sub.dropna(subset=["time_to_next_opp_fire_sec"])
    if valid.empty:
        return pd.DataFrame()
    valid = valid.copy()
    valid["chain_bin"] = valid["chain_count"].clip(upper=CHAIN_BIN_CAP)
    g = valid.groupby("chain_bin")["time_to_next_opp_fire_sec"].agg(["mean", "median", "count"])
    g["hands_mean"] = g["mean"] / EXISTING_SEC_PER_HAND
    return g


def _report_depth_distribution(sub: pd.DataFrame) -> pd.Series:
    """応酬シーケンス深さのヒストグラム (seq_id 単位で重複排除)。"""
    seqs = sub.drop_duplicates(subset=["video_stem", "game_idx", "seq_id"])
    return seqs["seq_depth"].value_counts(normalize=True).sort_index()


def _report_return_distribution(sub: pd.DataFrame) -> dict[str, float]:
    """対応 (seq_position>=2) イベントの返り量分布を集計する。

    単位は「個」(お邪魔の個数、生値) を主とし、「段」(個数/OJAMA_PER_DAN) を
    併記する (2026-07-22 訂正: 以前 "列" とラベル付けしていたのは誤り)。
    mean は外れ値に強く引きずられるため median も併記し、median を主指標とする。
    """
    responses = sub[sub["seq_position"] >= 2]
    if responses.empty:
        return {
            "n": 0, "mean_count": float("nan"), "median_count": float("nan"),
            "mean_dan": float("nan"), "median_dan": float("nan"),
            "pct_le_1dan": float("nan"), "pct_ge_2dan": float("nan"),
        }
    counts = responses["ojama_sent_count"]
    dans = counts / OJAMA_PER_DAN
    return {
        "n": int(len(counts)),
        "mean_count": float(counts.mean()),
        "median_count": float(counts.median()),
        "mean_dan": float(dans.mean()),
        "median_dan": float(dans.median()),
        "pct_le_1dan": float((dans <= RETURN_ONE_DAN_UPPER).mean()),
        "pct_ge_2dan": float((dans >= RETURN_TWO_DAN_LOWER).mean()),
    }


def _report_outlier_note(sub: pd.DataFrame) -> str:
    """検出限界メモ (score欠損フォールバック件数) を返す。

    ⚠️ 2026-07-22 更新: MIN_CHAIN_COUNT_FOR_FIRE 導入により、本関数に渡る
    sub は既に chain_count>=1 の行のみ (疑似発火は _process_side_game で
    棄却済み) のため、「chain_count>=1 の発生率」は常に 1.000 に潰れ
    診断的価値を失った (旧仕様の出力のみ残す、後方互換)。カスケード分断
    そのものの目安は代わりに frag_count 分布 (_print_defrag_examples 等)
    を参照すること。
    """
    suspect_rate = float((sub["chain_count"] >= 1).mean())
    missing_score_n = int((sub["delta_score"] == SCORE_MISSING_SENTINEL).sum())
    return (f"chain_count>=1 の発火割合: {suspect_rate:.3f} (常に1.000、フィルタ済のため無意味) / "
            f"score欠損フォールバック検出イベント数: {missing_score_n}")


def _print_report(df: pd.DataFrame) -> None:
    """ティア別 + 全体で (a)(b)(c) と催促/本線比率を出力する。"""
    for tier in ("チャレンジャー", "マスター", "S級", "全体"):
        sub = df if tier == "全体" else df[df["tier"] == tier]
        if sub.empty:
            continue
        print(f"\n=== {tier} (n_fire={len(sub)}) ===")
        print("[催促/本線 比率]")
        print(_report_label_ratio(sub))
        dan = sub["ojama_sent_count"] / OJAMA_PER_DAN
        print(f"[送りお邪魔 個数: mean={sub['ojama_sent_count'].mean():.1f} "
              f"median={sub['ojama_sent_count'].median():.1f} / "
              f"段: mean={dan.mean():.2f} median={dan.median():.2f}]")
        print(f"[検出限界メモ] {_report_outlier_note(sub)}")
        print("[(a) 連鎖数 -> 次相手発火までの時間]")
        print(_report_time_coefficient(sub))
        print("[(b) 応酬シーケンス深さ分布]")
        print(_report_depth_distribution(sub))
        print("[(c) 対応の返り量分布]")
        print(_report_return_distribution(sub))


# ============================
# de-frag 比較レポート (旧 vs 新)
# ============================


def _print_defrag_counts_by_video(df_raw: pd.DataFrame, df_defrag: pd.DataFrame) -> None:
    """動画別イベント数の比較 (旧=de-frag前 -> 新=de-frag後)。"""
    raw_n = df_raw.groupby("video_stem").size()
    defrag_n = df_defrag.groupby("video_stem").size()
    print("\n[動画別イベント数] 旧(分断) -> 新(統合) (畳み込み数)")
    for stem in sorted(set(raw_n.index) | set(defrag_n.index)):
        r = int(raw_n.get(stem, 0))
        d = int(defrag_n.get(stem, 0))
        print(f"  {stem}: {r} -> {d} (-{r - d})")


def _print_defrag_counts_by_tier(df_raw: pd.DataFrame, df_defrag: pd.DataFrame) -> None:
    """ティア別イベント数の比較 (旧 -> 新)。"""
    print("\n[ティア別イベント数] 旧(分断) -> 新(統合)")
    for tier in ("チャレンジャー", "マスター", "S級", "全体"):
        sub_r = df_raw if tier == "全体" else df_raw[df_raw["tier"] == tier]
        sub_d = df_defrag if tier == "全体" else df_defrag[df_defrag["tier"] == tier]
        if sub_r.empty:
            continue
        print(f"  {tier}: {len(sub_r)} -> {len(sub_d)} (-{len(sub_r) - len(sub_d)})")


def _print_defrag_contamination(df_raw: pd.DataFrame, df_defrag: pd.DataFrame) -> None:
    """chain_count>=1 率の比較 (旧 -> 新)。

    ⚠️ 2026-07-22 更新: MIN_CHAIN_COUNT_FOR_FIRE 導入により df_raw/df_defrag
    は共に既にフィルタ済で常に 1.000 -> 1.000 になる (後方互換のため出力形式
    は残す。診断的価値は _report_outlier_note と同様に失われている)。
    """
    print("\n[chain_count>=1 率 (常に1.000、フィルタ済のため参考情報)] 旧 -> 新")
    for tier in ("チャレンジャー", "マスター", "S級", "全体"):
        sub_r = df_raw if tier == "全体" else df_raw[df_raw["tier"] == tier]
        sub_d = df_defrag if tier == "全体" else df_defrag[df_defrag["tier"] == tier]
        if sub_r.empty:
            continue
        rate_r = float((sub_r["chain_count"] >= 1).mean())
        rate_d = float((sub_d["chain_count"] >= 1).mean())
        print(f"  {tier}: {rate_r:.3f} -> {rate_d:.3f}")


def _print_defrag_examples(df_defrag: pd.DataFrame, n: int = 3) -> None:
    """畳み込み具体例 (frag_count>1 の上位 n 件) を表示する。"""
    frags = df_defrag[df_defrag["frag_count"] > 1].sort_values(
        "frag_count", ascending=False,
    )
    print(f"\n[畳み込み具体例] (frag_count>1 の上位{n}件、n_total={len(frags)})")
    if frags.empty:
        print("  該当なし (de-fragで畳み込まれた発火なし)")
        return
    for _, row in frags.head(n).iterrows():
        print(
            f"  {row['video_stem']} game={row['game_idx']} side={row['fire_side']} "
            f"t_fire={row['t_fire']:.2f}s chain_count={row['chain_count']} "
            f"frag_count={row['frag_count']} delta_score={row['delta_score']}",
        )


def _print_defrag_comparison(df_raw: pd.DataFrame, df_defrag: pd.DataFrame) -> None:
    """de-frag 前後の比較レポート一式を出力する。"""
    print("\n" + "=" * 60)
    print("[de-frag 比較] 旧(連鎖分断あり) vs 新(1本の連鎖=1イベントに統合)")
    print("=" * 60)
    _print_defrag_counts_by_video(df_raw, df_defrag)
    _print_defrag_counts_by_tier(df_raw, df_defrag)
    _print_defrag_contamination(df_raw, df_defrag)
    _print_defrag_examples(df_defrag)


# ============================
# メイン
# ============================


def main() -> None:
    """メイン処理: 23動画を処理し de-frag 後 CSV 保存 + 比較レポート出力する。"""
    warnings.filterwarnings("ignore")
    npz_paths = [NPZ_DIR / f"{stem}.npz" for stem in TIER_MAP]
    missing = [p for p in npz_paths if not p.exists()]
    if missing:
        print(f"[ERROR] npz不足: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 対象 {len(npz_paths)} 動画 (チャレンジャー{len(TIER_CHALLENGER)}"
          f"/マスター{len(TIER_MASTER)}/S級{len(TIER_S_CLASS)})")

    sim = ChainSimulator()
    all_raw: list[FireEvent] = []
    all_defrag: list[FireEvent] = []
    seq_id = 0
    for npz_path in sorted(npz_paths, key=lambda p: p.stem):
        raw_events, defrag_events, seq_id = _process_video(npz_path, sim, seq_id)
        all_raw.extend(raw_events)
        all_defrag.extend(defrag_events)
        print(
            f"  {npz_path.stem} ({TIER_MAP[npz_path.stem]}): "
            f"raw={len(raw_events)} defrag={len(defrag_events)} "
            f"(累計 raw={len(all_raw)} defrag={len(all_defrag)})",
        )

    if not all_defrag:
        print("[ERROR] 発火イベントが0件でした。", file=sys.stderr)
        sys.exit(1)

    df_raw = pd.DataFrame([vars(e) for e in all_raw])
    df = pd.DataFrame([vars(e) for e in all_defrag])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(
        f"\n[DONE] de-frag後 {len(df)} 行 (旧 {len(df_raw)} 行) を "
        f"{OUTPUT_CSV} に保存しました",
    )

    _print_defrag_comparison(df_raw, df)
    _print_report(df)


if __name__ == "__main__":
    main()
