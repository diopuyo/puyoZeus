"""
ぷよぷよ公式得点・おじゃまぷよ計算モジュール。

参考資料（ウェブ調査済み、2026-04-25）:
    - 壱大整域「得点」: alg-d.com/game/puyo/taisen11.html
    - ぷよブロ「得点計算」: puyo-euphonic.com/puyo-word-score-calculation
    - ニコニコ大百科「おじゃまぷよ算」: dic.nicovideo.jp/a/おじゃまぷよ算
    - ぷよぷよキャンプ「得点計算考察」: puyo-camp.jp/posts/186726

## 公式得点式（通ルール）

```
step_score = (消したぷよ数) × 10 × max(1, 連鎖ボーナス + 連結ボーナス + 色数ボーナス)
total_score = Σ step_score + 全消しボーナス
```

## ボーナステーブル（eスポーツ仕様）

- 連鎖ボーナス: [0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
  - 1連鎖目=0、2連鎖目=8、以降 +32 ずつ（19連鎖目=512 で頭打ち → 以降は +32 継続）
- 連結ボーナス: 4→0, 5→2, 6→3, 7→4, 8→5, 9→6, 10→7, 11+→10
  - 1ステップで複数グループ消す場合は和を取る
- 色数ボーナス: 1色→0, 2色→3, 3色→6, 4色→12, 5色→24

## おじゃまぷよ発生量

```
ojama_sent = (total_score + 繰越) / rate
余り = (total_score + 繰越) % rate   # 次の得点に繰越
```

- 通常モード: rate = 70 点 / 1 個
- フィーバーモード: rate = 120 点 / 1 個（本モジュールでは未対応、通常のみ）
- マージンタイム: 試合開始 96 秒後から rate が段階的に減少（攻撃力増強）
    - 16 秒ごとに rate を 75% に（≒ 0.75^n で減少）
    - 下限レートまで下降（公式は 4 点 / 個）

## 全消しボーナス

全消し達成時、次の連鎖に 2100 点相当のボーナス（1ライン＝ 30 個相当を即時加算）。
本モジュールでは `ALL_CLEAR_BONUS = 2100` として定義。

使い方:
    from src.chain import ChainSimulator
    from src.scoring import calculate_chain_score, score_to_ojama

    result = ChainSimulator().simulate(board)
    score = calculate_chain_score(result)
    ojama, leftover = score_to_ojama(score, prev_leftover=0)
"""
from __future__ import annotations

from dataclasses import dataclass

from src.board import COLOR_OJAMA
from src.chain import ChainResult, ChainStep

# ============================
# 公式ボーナステーブル
# ============================

# 連鎖ボーナス（1連鎖目 index 0）
CHAIN_POWER_TABLE: tuple[int, ...] = (
    0, 8, 16, 32, 64, 96, 128, 160, 192, 224,
    256, 288, 320, 352, 384, 416, 448, 480, 512,
)
# 19連鎖超は +32 ずつ線形延長
CHAIN_POWER_INCREMENT: int = 32


def chain_power(chain_idx_1based: int) -> int:
    """n 連鎖目の連鎖ボーナスを返す（1-indexed）。"""
    if chain_idx_1based < 1:
        return 0
    zero_idx = chain_idx_1based - 1
    if zero_idx < len(CHAIN_POWER_TABLE):
        return CHAIN_POWER_TABLE[zero_idx]
    # 19連鎖超は線形延長
    extra = zero_idx - (len(CHAIN_POWER_TABLE) - 1)
    return CHAIN_POWER_TABLE[-1] + extra * CHAIN_POWER_INCREMENT


# 連結ボーナス
CONNECTION_BONUS_TABLE: dict[int, int] = {
    4: 0, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
}
CONNECTION_BONUS_MAX: int = 10  # 11連結以上は 10 で頭打ち


def connection_bonus(group_size: int) -> int:
    """1 グループの連結ボーナスを返す（size は 4 以上を想定）。"""
    if group_size < 4:
        return 0
    if group_size >= 11:
        return CONNECTION_BONUS_MAX
    return CONNECTION_BONUS_TABLE[group_size]


# 色数ボーナス
COLOR_BONUS_TABLE: dict[int, int] = {1: 0, 2: 3, 3: 6, 4: 12, 5: 24}


def color_bonus(num_colors: int) -> int:
    """同時に消した色数のボーナスを返す。"""
    if num_colors <= 1:
        return 0
    if num_colors >= 5:
        return COLOR_BONUS_TABLE[5]
    return COLOR_BONUS_TABLE[num_colors]


# ============================
# その他の定数
# ============================

# 1ぷよあたりの基本得点
BASE_SCORE_PER_PUYO: int = 10

# ボーナス係数の最小値・最大値（公式: 1 ≤ (CP+CB+GB) ≤ 999）
MIN_BONUS_MULTIPLIER: int = 1
MAX_BONUS_MULTIPLIER: int = 999

# 全消しボーナス（通ルール）
ALL_CLEAR_BONUS: int = 2100

# おじゃまぷよ基準レート
OJAMA_RATE_STANDARD: int = 70
OJAMA_RATE_FEVER: int = 120

# マージンタイム設定（eスポーツ通常ルール、Puyo Nexus 公式準拠）
MARGIN_TIME_START_SEC: float = 96.0
MARGIN_TIME_DECAY_INTERVAL_SEC: float = 16.0
MARGIN_TIME_DECAY_FACTOR: float = 0.75
# 公式: 最大 14 回減衰、または rate=1 で停止のいずれか早い方
MARGIN_TIME_MAX_DECAYS: int = 14
OJAMA_RATE_MIN: int = 1  # 下限


# ============================
# ステップ単位の得点計算
# ============================


@dataclass(frozen=True)
class StepScoreResult:
    """1 連鎖ステップの得点計算結果。"""
    chain_idx: int          # 1-indexed
    erased_count: int
    chain_bonus: int
    connection_bonus_total: int
    color_bonus: int
    bonus_multiplier: int   # max(1, chain+connection+color)
    score: int


def calculate_step_score(step: ChainStep) -> StepScoreResult:
    """ChainStep 1 つ分の得点を返す。"""
    # ぷよぷよ通では erased_groups におじゃまは含まないが保険で除外
    groups = [g for g in step.erased_groups if g.color != COLOR_OJAMA]
    if not groups or step.erased_count <= 0:
        return StepScoreResult(
            chain_idx=step.chain_index,
            erased_count=0,
            chain_bonus=0,
            connection_bonus_total=0,
            color_bonus=0,
            bonus_multiplier=MIN_BONUS_MULTIPLIER,
            score=0,
        )
    cb = chain_power(step.chain_index)
    conn = sum(connection_bonus(g.size) for g in groups)
    colors = len({g.color for g in groups})
    col_bonus = color_bonus(colors)
    # 公式仕様: (CP + CB + GB) は 1 以上 999 以下にクランプ
    raw_bonus = cb + conn + col_bonus
    total_bonus = max(
        MIN_BONUS_MULTIPLIER,
        min(MAX_BONUS_MULTIPLIER, raw_bonus),
    )
    score = step.erased_count * BASE_SCORE_PER_PUYO * total_bonus
    return StepScoreResult(
        chain_idx=step.chain_index,
        erased_count=step.erased_count,
        chain_bonus=cb,
        connection_bonus_total=conn,
        color_bonus=col_bonus,
        bonus_multiplier=total_bonus,
        score=score,
    )


# ============================
# 連鎖全体の得点計算
# ============================


@dataclass(frozen=True)
class ChainScoreResult:
    """
    ChainResult 全体の得点計算。

    Attributes:
        steps: 各連鎖ステップの得点詳細。
        total_score: 連鎖の素点合計（全消しボーナスは含まない）。
        is_all_clear: 連鎖終了後の盤面が全消し状態か。
            公式仕様では全消しボーナスは「次の連鎖発火時」に加算されるため、
            このフラグを参照して呼び出し側で持ち越し処理を行う。
    """
    steps: tuple[StepScoreResult, ...]
    total_score: int
    is_all_clear: bool


def calculate_chain_score(result: ChainResult) -> ChainScoreResult:
    """
    連鎖シミュレーション結果から素点を計算する。

    全消しボーナス (ALL_CLEAR_BONUS) は本関数では加算しない。
    Puyo Nexus 公式仕様より、全消しは「次の連鎖発火時」に追加される。
    呼び出し側（VideoChainTracker 等）で is_all_clear を見て次回に持ち越す。

    Args:
        result: ChainSimulator().simulate() の結果。

    Returns:
        ChainScoreResult: 各ステップ、素点合計、全消しフラグ。
    """
    step_scores = tuple(calculate_step_score(s) for s in result.steps)
    total = sum(s.score for s in step_scores)
    return ChainScoreResult(
        steps=step_scores,
        total_score=total,
        is_all_clear=_is_all_clear(result),
    )


def _is_all_clear(result: ChainResult) -> bool:
    """連鎖後の盤面が全消し状態か。"""
    board = result.final_board
    # board.copy の方法は board.py 参照。全セル empty 判定は count_puyos で代用。
    try:
        from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                if board.get(r, c) != COLOR_EMPTY:
                    return False
        return True
    except Exception:
        return False


# ============================
# 得点 → おじゃま換算
# ============================


@dataclass(frozen=True)
class OjamaResult:
    """score_to_ojama() の結果。"""
    ojama_count: int
    leftover_score: int      # 次回へ繰越す余り得点
    effective_rate: int


def score_to_ojama(
    score: int,
    prev_leftover: int = 0,
    elapsed_sec: float = 0.0,
    rate_base: int = OJAMA_RATE_STANDARD,
) -> OjamaResult:
    """
    得点をおじゃまぷよ数に換算する。

    Args:
        score: 今回の発火得点。
        prev_leftover: 前回の繰越余り得点。
        elapsed_sec: 試合開始からの経過秒（マージンタイム計算用）。
        rate_base: 基本レート（通常 70、フィーバー 120）。

    Returns:
        OjamaResult: 換算されたおじゃま数・繰越・有効レート。
    """
    rate = compute_effective_rate(elapsed_sec, rate_base)
    pool = score + prev_leftover
    ojama = pool // rate
    leftover = pool % rate
    return OjamaResult(
        ojama_count=ojama,
        leftover_score=leftover,
        effective_rate=rate,
    )


## ============================
## 予告アイコン分解 (2026-04-27 ユーザ仕様確定)
## ============================
##
## ぷよぷよeスポーツの予告お邪魔ぷよ表示仕様:
##   - 予告は最大 6 アイコンまで表示
##   - アイコン分解は上位優先 (crown > moon > star > rock > large > small)
##   - 6 アイコンに収まらない端数は「表示落ち」(個数だけ持って表示しない)
##   - 1 ターン (ぷよ設置) ごとに最大 30 個 (= rock 1 個分) がフィールドに落下
##   - 30 個超過分は次ターンに繰越し
OJAMA_ICON_VALUES: tuple[tuple[str, int], ...] = (
    ("crown", 720),
    ("moon", 360),
    ("star", 180),
    ("rock", 30),
    ("large", 6),
    ("small", 1),
)
OJAMA_MAX_ICONS_DISPLAY: int = 6
OJAMA_MAX_DROP_PER_TURN: int = 30  # 1 ターンに落ちる ojama 個数上限


def ojama_count_to_icons(count: int) -> list[tuple[str, int]]:
    """ojama 総個数を表示アイコン (最大 6 個) に分解する。

    上位アイコン優先で詰め、6 アイコンを超えた分は表示落ち (返り値に含まない)。

    Args:
        count: ojama 個数 (>= 0)

    Returns:
        [(icon_name, icon_count), ...]、合計 icon_count <= 6
    """
    if count <= 0:
        return []
    icons: list[tuple[str, int]] = []
    remaining = int(count)
    used = 0
    for name, value in OJAMA_ICON_VALUES:
        if remaining < value or used >= OJAMA_MAX_ICONS_DISPLAY:
            continue
        max_n = remaining // value
        room = OJAMA_MAX_ICONS_DISPLAY - used
        n = min(max_n, room)
        if n > 0:
            icons.append((name, n))
            remaining -= n * value
            used += n
    return icons


def icons_to_ojama_count(icons: list[tuple[str, int]]) -> int:
    """アイコン構成 → ojama 個数 (アイコン分解の逆)。"""
    value_map = dict(OJAMA_ICON_VALUES)
    return sum(value_map.get(name, 0) * cnt for name, cnt in icons)


def split_ojama_drop_per_turn(
    pending: int, drop_max: int = OJAMA_MAX_DROP_PER_TURN
) -> tuple[int, int]:
    """予告 ojama を「今ターン落下分」と「次ターン繰越分」に分割する。

    Args:
        pending: 現在の予告 ojama 個数
        drop_max: 1 ターン落下上限 (デフォルト 30)

    Returns:
        (今ターン落下する個数, 次ターン繰越分)
    """
    p = max(0, int(pending))
    drop_now = min(p, drop_max)
    return drop_now, p - drop_now


def compute_effective_rate(
    elapsed_sec: float,
    rate_base: int = OJAMA_RATE_STANDARD,
) -> int:
    """
    マージンタイム経過後の有効レートを返す。

    仕様（Puyo Nexus 準拠）:
        - 試合開始〜MARGIN_TIME_START_SEC (96s): rate_base
        - 以降 MARGIN_TIME_DECAY_INTERVAL_SEC (16s) ごとに 0.75 倍
        - 最大 MARGIN_TIME_MAX_DECAYS (14) 回まで減衰、それ以降は固定
        - OJAMA_RATE_MIN (=1) で下限頭打ち（切り捨て）
    """
    if elapsed_sec <= MARGIN_TIME_START_SEC:
        return rate_base
    decay_steps = int((elapsed_sec - MARGIN_TIME_START_SEC) // MARGIN_TIME_DECAY_INTERVAL_SEC) + 1
    decay_steps = min(decay_steps, MARGIN_TIME_MAX_DECAYS)
    factor = MARGIN_TIME_DECAY_FACTOR ** decay_steps
    rate = int(rate_base * factor)
    return max(OJAMA_RATE_MIN, rate)
