"""発火前の最善手探索 (2026-10-01、Phase 3、状態を持たない)。

user 決定 (2026-10-01): 撃つかどうかは指し手の選択にすぎない。表示勝率は各側の **最善の選択肢の値**
とし、発火タイミングの学習 (hazard) は使わない。人のミスは考慮しない。
本モジュールは「既知のツモで撃てる最善の発火」と「受け側の最善応手の量」と「確実な勝ち (死亡証明)」
だけを返す。勝率への変換 (既存の S3 / 仮想着弾 G_fe) は `src/prefire_best_play_layer.py` が行う。

使う既存部品 (新規に書いたのはそれらをつなぐ部分だけ):
- 既知ツモの配置列挙: `puyo_core_bridge.enumerate_and_simulate_placements` (Phase 2 の `prefire_exchange_sim._split` 経由)
- 3手目の探索: `puyo_core_bridge.beam_search_continue` (native、静かな2手の全盤面を初期集団にする)
- 受け側の最善応手: `scripts/mc_counter_estimator.estimate_counter_distribution` (既知ツモは全列挙、その先は試合の4色で MC)
- 確実な勝ち: `post_counter_death_bound.prove_post_counter` (E35 の打ち返し後死亡上限、ツモは試合4色の全組合せで保守的に証明)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src import prefire_exchange_sim as sim
from src import puyo_core_bridge as native
from src.board import Board
from src.post_counter_death_bound import GAME_COLORS, prove_post_counter
from src.production_config import GHOST_CHAIN_RULE_ENABLED

KNOWN_HANDS = 3                     # 手に持つ組 + NEXT + NEXT2 (記録の NEXT は1つ先。known_pairs が並べ替える)
PAIR = 2
COUNTER_ROLLOUTS = 8                # 受け側の未来ツモ (NEXT2 より先) の標本数 (Phase 2 構成B と同じ)
COUNTER_BEAM_WIDTH = 8              # 応手探索のビーム幅 (既存 NEAR_FUTURE_BEAM_WIDTH と同じ)
MAX_COUNTER_HANDS = 16              # 応手探索の深さ上限 (Phase 2 と同じ。最大級の連鎖演出中に置ける手数の上限)
CONSERVATIVE_QUEUE: tuple[int, ...] = ()   # 死亡証明では NEXT を使わず試合4色の全組合せを認める (保守側)
CACHE_SIZE = 16384
PLACEMENTS_PER_PAIR = 22            # 1組の置き方 (縦2向き×6列 + 横2向き×5列)
THIRD_HAND_BEAM_WIDTH = 1           # 3手目の探索の幅 (_third_hand の docstring)
TWO_HAND_CACHE_SIZE = 512           # 静かな2手の盤面群 (最大 484 盤面) を持つので小さくする


@dataclass(frozen=True)
class FireLine:
    """既知ツモで撃てる1つの発火。hand=何手目で撃つか (1=手に持つ組)。final は連鎖後の盤面。"""

    hand: int
    score: int
    chain_count: int
    final: bytes


def _pairs(known: tuple[int, ...]) -> list[tuple[int, int]]:
    """見えている組を先頭から返す (最初の未読で打ち切る)。"""
    out = []
    for i in range(0, min(len(known), KNOWN_HANDS * PAIR) - 1, PAIR):
        pair = (int(known[i]), int(known[i + 1]))
        if not all(c in sim.PLAYABLE_COLORS for c in pair):
            break
        out.append(pair)
    return out


def _valid(raw: bytes) -> bool:
    """置く前から窒息セルが埋まっている・消える群がある盤面 (誤読・連鎖途中) は探索しない (Phase 2 と同じ規則)。"""
    board = sim._board(raw)
    if sim._dead(board._grid):
        return False
    return native.simulate_chain(board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED).chain_count == 0


def _line(hand: int, placement) -> FireLine:
    """配置結果を発火へ変換する。"""
    result = placement.chain_result
    return FireLine(hand, int(result.exact_score), int(result.chain_count),
                    result.final_board._grid.astype(np.int8).tobytes())


def _best(placements: list) -> object | None:
    """得点最大の配置 (同点は列挙順の先頭)。"""
    return max(placements, key=lambda p: p.chain_result.exact_score, default=None)


def _third_hand(groups: tuple, pair: tuple[int, int]) -> FireLine | None:
    """静かな2手の盤面群から3手目の最大発火を探す。native ビームで群ごとの最大を出し、最良群だけ列挙する。

    ビーム幅は1でよい: 1手だけの探索では最大得点は全候補の展開時に記録され、幅は次の深さへ残す数にしか効かない。
    幅 (群の大きさ×22) と幅1の最大得点は記録の6,039群で全件一致し、所要は 1/7.5 (Phase 4、2026-10-01)。
    幅を大きくすると最終候補を全部 Board に戻すため遅い。
    """
    best_group, best_score = None, 0
    for group in groups:
        if not group:
            continue
        frontier = [native.FrontierEntry(board=p.placed_board, running_best=0) for p in group]
        result = native.beam_search_continue(frontier, 0, [pair], THIRD_HAND_BEAM_WIDTH,
                                             exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
                                             use_exact_score=True)
        if result.best_score > best_score:
            best_group, best_score = group, int(result.best_score)
    if best_group is None:
        return None
    fires = [f for p in best_group for f in sim._split(sim._placements(p.placed_board, pair))[0]]
    top = _best(fires)
    return _line(3, top) if top is not None else None


@lru_cache(maxsize=TWO_HAND_CACHE_SIZE)
def _two_hands(raw: bytes, first: tuple[int, int], second: tuple[int, int] | None) -> tuple:
    """1〜2手目の最大発火・撃たないと窒息・静かな2手の盤面群。NEXT2 が後から読めても使い回す (純関数)。"""
    board = sim._board(raw)
    fires1, quiet1, has_quiet = sim._split(sim._placements(board, first))
    lines = [_line(1, _best(fires1))] if fires1 else []
    forced = bool(fires1) and has_quiet and not quiet1
    if second is None:
        return tuple(lines), forced, ()
    groups, fires2 = [], []
    for placed in quiet1:
        fires, quiet, _ = sim._split(sim._placements(placed.placed_board, second))
        fires2 += fires
        groups.append(tuple(quiet))
    if fires2:
        lines.append(_line(2, _best(fires2)))
    return tuple(lines), forced, tuple(groups)


@lru_cache(maxsize=CACHE_SIZE)
def fire_lines(raw: bytes, known: tuple[int, ...]) -> tuple[tuple[FireLine, ...], bool]:
    """既知の3手以内で撃てる各手数の最大発火と、撃たないと窒息するか (純関数)。

    発火は連鎖後の盤面、静かな手は設置直後の盤面で窒息を判定する (Phase 2 の `_split` と同じ)。
    盤面が不正 (置く前から窒息・消える群がある) か、手に持つ組が未読なら発火なし。
    """
    pairs = _pairs(known)
    if not pairs or not _valid(raw):
        return (), False
    lines, forced, groups = _two_hands(raw, pairs[0], pairs[1] if len(pairs) >= 2 else None)
    if len(pairs) >= KNOWN_HANDS:
        third = _third_hand(groups, pairs[2])
        if third is not None:
            lines = (*lines, third)
    return tuple(lines), forced


_RATE_CACHE: dict[tuple, object] = {}


def _by_rate(name: str, args: tuple, elapsed: float, compute):
    """経過秒を換算率に置き換えた鍵で結果を使い回す (出力は換算率だけに依存するので値は変わらない)。"""
    key = (name, args, sim.effective_rate(elapsed))
    if key not in _RATE_CACHE:
        if len(_RATE_CACHE) >= CACHE_SIZE:
            _RATE_CACHE.pop(next(iter(_RATE_CACHE)))
        _RATE_CACHE[key] = compute()
    return _RATE_CACHE[key]


def _counter_quantiles(raw: bytes, known: tuple[int, ...], hands: int, colors: tuple[int, ...],
                      elapsed: float) -> tuple[float, ...]:
    """受け側が hands 手以内に返せる最大のおじゃま個数。既知ツモで決まれば1点、MC なら (p25, 平均, p75)。

    既存の `estimate_counter_distribution` (auto: 3手以内は既知ツモの完全列挙、その先は試合の色で MC)。
    標本の種は盤面と手数から決まる (再現可能)。NEXT 未読・盤面不正なら応手なし (0)。
    """
    from scripts import mc_counter_estimator as mc
    pairs = _pairs(known)
    if not pairs or not colors or not _valid(raw):
        return (0.0,)
    depth = int(min(max(1, hands), MAX_COUNTER_HANDS))
    exact = depth <= len(pairs)
    dist = mc.estimate_counter_distribution(
        sim._board(raw), depth * mc.BEAM_ROLLOUT_AVG_STEP_TIME_SEC, tuple(pairs),
        n_rollouts=1 if exact else COUNTER_ROLLOUTS, active_colors=tuple(colors), elapsed_sec=elapsed,
        rollout_mode='auto', beam_width=COUNTER_BEAM_WIDTH)
    return (float(dist.mean),) if exact else (float(dist.p25), float(dist.mean), float(dist.p75))


def counter_quantiles(raw: bytes, known: tuple[int, ...], hands: int, colors: tuple[int, ...],
                      elapsed: float) -> tuple[float, ...]:
    """_counter_quantiles の結果を (盤面・既知の組・手数・色・換算率) で使い回す。"""
    return _by_rate('counter', (raw, tuple(known), int(hands), tuple(colors)), elapsed,
                    lambda: _counter_quantiles(raw, known, hands, colors, elapsed))


def _lethal(raw: bytes, incoming: int, hands: int, colors: tuple[int, ...], elapsed: float) -> bool:
    """受け側がどう応手しても窒息することの証明 (E35 上限、ツモは試合4色の全組合せ)。4色が未確定なら証明しない。"""
    if incoming <= 0 or len(colors) != GAME_COLORS:
        return False
    proof = prove_post_counter(sim._board(raw), CONSERVATIVE_QUEUE, int(incoming), int(hands),
                               float(elapsed), tuple(colors))
    return bool(proof['dead'])


def lethal(raw: bytes, incoming: int, hands: int, colors: tuple[int, ...], elapsed: float) -> bool:
    """_lethal の結果を (盤面・受け量・手数・色・換算率) で使い回す。"""
    return _by_rate('lethal', (raw, int(incoming), int(hands), tuple(colors)), elapsed,
                    lambda: _lethal(raw, incoming, hands, colors, elapsed))


def feature_simulator(grid: np.ndarray) -> object | None:
    """発火前の側特徴に native 連鎖計算を使ってよい盤面なら NativeSimulator、それ以外は None (Python 版)。

    浮きぷよ (重力違反) や UNKNOWN を含む盤面では native と Python の結果が一致しない (記録で2件確認)。
    既存の mc_counter_estimator と同じ安全弁 (`_board_is_gravity_consistent`) を使い、その盤面は Python 版へ戻す。
    """
    from scripts.mc_counter_estimator import _board_is_gravity_consistent
    from src.board import COLOR_UNKNOWN
    from src.exchange_event_count_features import NativeSimulator
    board = sim._board(np.asarray(grid, dtype=np.int8).tobytes())
    if np.any(board._grid == COLOR_UNKNOWN) or not _board_is_gravity_consistent(board):
        return None
    return NativeSimulator()


def board_from(raw: bytes) -> Board:
    """int8 の生盤面から Board を作る (layer 側で仮想着弾に使う)。"""
    return sim._board(raw)
