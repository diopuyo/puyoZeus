"""発火前予測に使える既存部品の1呼び出しあたり所要を、実記録の発火直前盤面で測る (Phase 1b)。

入力: 本番構成の記録 (logs/pending_expiry/full/records/*.jsonl.gz) から、各撃ち合いの発火前
3 秒以内で両者 STABLE の確定盤面と NEXT/NEXT2 を取り出す (1記録あたり上限 SAMPLE_LIMIT 盤面)。
単位: 「片側1盤面への1呼び出し」。速度は単一プロセス・nice 19・他ジョブ並走下の参考値
(背景負荷は結果の JSON に loadavg として残す)。
使い方: PYTHONPATH=. python -m scripts.prefire_asset_bench_20260930
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from src import indicators_v2 as iv
from src import puyo_core_bridge as native
from src.board import Board
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.exchange_event_record import read_records

EXEV = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer')
RECORDS = EXEV / 'logs/pending_expiry/full/records'
EVENTS = EXEV / 'logs/pending_expiry/e36b_on/on'
SOURCES = {'q_7gc4TgFig': 'renders/q_7gc4TgFig/on', 'zenchi': 'zenchi'}
OUT = Path('logs/prefire_prediction/asset_bench.json')
PREFIRE_SEC = 3.0
SAMPLE_LIMIT = 60
FOUR_COLORS = (1, 2, 3, 4)
PLAYABLE_COLORS = (1, 2, 3, 4, 5)
DEATH_ROW, DEATH_COL = 1, 2   # 窒息判定セル (CLAUDE.md: 3列目の可視最上段)
RANDOM_SEED = 20260930
BEAM_CONFIGS = ((8, 4), (50, 6), (250, 13))   # (幅, 深さ)。深さ13幅250は既存ベンチの目標条件
BEAM_HEAVY_LIMIT = 10                          # 幅250深さ13は重いので盤面数を絞る


def triggers(source: str) -> list[float]:
    """イベント原票から発火時刻を得る (発火より前の盤面だけ使うため)。"""
    lines = (EVENTS / SOURCES[source] / 'events.jsonl').read_text().splitlines()
    return sorted(json.loads(line)['trigger_sec'] for line in lines)


def near_trigger(t: float, fires: list[float]) -> bool:
    """t が次の発火の PREFIRE_SEC 秒以内の直前か。"""
    idx = int(np.searchsorted(fires, t, side='right'))
    return idx < len(fires) and 0 < fires[idx] - t <= PREFIRE_SEC


def side_sample(side: Any) -> tuple[Board, tuple] | None:
    """STABLE で盤面と NEXT/NEXT2 が揃った側だけ (盤面, 既知2組) を返す。"""
    state = getattr(getattr(side, 'state', None), 'value', None)
    board, nxt, dnxt = side.confirmed_board, side.next_pair, side.dnext_pair
    if state != 'stable' or board is None or nxt is None or dnxt is None:
        return None
    if not all(c in PLAYABLE_COLORS for c in (*nxt, *dnxt)):
        return None   # (9,9) 等は NEXT 未読の番兵。探索に使うと即終了して速度が嘘になる
    if not isinstance(board, Board):
        board = Board.from_list(np.asarray(board).tolist())
    if board._grid[DEATH_ROW, DEATH_COL] != 0:
        return None
    return board, (tuple(nxt), tuple(dnxt))


def sample_key(sample: tuple[Board, tuple]) -> bytes:
    """同じ盤面・同じ NEXT の重複を除くためのキー (連続フレームは同一盤面が続く)。"""
    return sample[0]._grid.astype(np.int8).tobytes() + bytes(c for pair in sample[1] for c in pair)


def collect(source: str, counts: dict) -> list[tuple[Board, tuple]]:
    """記録から発火直前の片側盤面 (重複除去) を最大 SAMPLE_LIMIT 件集め、NEXT 可読率の母数も数える。"""
    fires, samples, seen, pending = triggers(source), [], set(), None
    for row in read_records(RECORDS / f'{source}.jsonl.gz'):
        if row['kind'] == 'update':
            pending = row['args'][0]
        elif row['kind'] == 'display' and pending is not None:
            if near_trigger(row['t_sec'], fires):
                for side in (pending.p1, pending.p2):
                    count_side(side, counts)
                    sample = side_sample(side)
                    if sample and sample_key(sample) not in seen and len(samples) < SAMPLE_LIMIT:
                        seen.add(sample_key(sample))
                        samples.append(sample)
            pending = None
    return samples


def count_side(side: Any, counts: dict) -> None:
    """発火直前の片側フレームで、STABLE 盤面と NEXT/NEXT2 の可読がどれだけ揃うかを数える。"""
    counts['side_frames'] = counts.get('side_frames', 0) + 1
    if getattr(getattr(side, 'state', None), 'value', None) == 'stable' and side.confirmed_board is not None:
        counts['stable_board'] = counts.get('stable_board', 0) + 1
        if side_sample(side) is not None:
            counts['stable_with_next'] = counts.get('stable_with_next', 0) + 1


def timed(fn: Callable[[], Any], items: int) -> list[float]:
    """1呼び出しの所要 (ms) を items 回測る。"""
    out = []
    for _ in range(items):
        start = perf_counter()
        fn()
        out.append((perf_counter() - start) * 1e3)
    return out


def two_hand_exhaustive(board: Board, pairs: tuple) -> int:
    """既知 NEXT/NEXT2 の2手を全列挙し、発火の最大得点を返す (E30 の1〜2手列挙と同じ規模)。"""
    best = 0
    for first in native.enumerate_and_simulate_placements(board, pairs[0], filter_dead=True,
                                                           exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED):
        best = max(best, first.chain_result.exact_score)
        if first.chain_result.chain_count:
            continue
        for second in native.enumerate_and_simulate_placements(first.placed_board, pairs[1], filter_dead=True,
                                                                exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED):
            best = max(best, second.chain_result.exact_score)
    return best


def random_pairs(rng: np.random.Generator, depth: int, known: tuple) -> list[tuple[int, int]]:
    """既知2組の後ろへ4色等確率の仮ツモを足す (MC 1本分)。"""
    extra = [tuple(int(c) for c in rng.choice(FOUR_COLORS, 2)) for _ in range(max(0, depth - len(known)))]
    return list(known)[:depth] + extra


class CountNFSimulator:
    """学習時と同じ native 連鎖計算 (exchange_event_count_features.NativeSimulator と同じ規則)。"""

    def simulate(self, board: Board) -> Any:
        """採用済みの隠し段規則で厳密得点を返す。"""
        return native.simulate_chain(board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)


def near_future(board: Board, pairs: tuple) -> Any:
    """S1′/count 特徴が使う近未来火力 (K=1..5、幅8、理想ツモ) を1回計算する (キャッシュなし)。"""
    return iv.near_future_fire_power(board, pairs[0], pairs[1], 300.0, simulator=CountNFSimulator(),
                                     active_colors=iv._near_future_active_colors(board), use_exact_score=True)


def quantiles(values: list[float]) -> dict:
    """母数つきの P50/P95/最大。"""
    arr = np.asarray(values)
    return dict(n=len(arr), p50=float(np.percentile(arr, 50)), p95=float(np.percentile(arr, 95)),
                max=float(arr.max()))


def bench(samples: list[tuple[Board, tuple]]) -> dict:
    """部品ごとに1呼び出しの所要を測る。"""
    rng = np.random.default_rng(RANDOM_SEED)
    rule = GHOST_CHAIN_RULE_ENABLED
    cases: dict[str, list[float]] = {k: [] for k in ('simulate_chain', 'placements_1hand',
                                                     'exhaustive_2hand_known', 'near_future_K1_5')}
    for board, pairs in samples:
        cases['simulate_chain'] += timed(lambda: native.simulate_chain(board, exclude_hidden_row_from_pop=rule), 1)
        cases['placements_1hand'] += timed(lambda: native.enumerate_and_simulate_placements(
            board, pairs[0], filter_dead=True, exclude_hidden_row_from_pop=rule), 1)
        cases['exhaustive_2hand_known'] += timed(lambda: two_hand_exhaustive(board, pairs), 1)
        cases['near_future_K1_5'] += timed(lambda: near_future(board, pairs), 1)
    for width, depth in BEAM_CONFIGS:
        subset = samples if width < max(w for w, _ in BEAM_CONFIGS) else samples[:BEAM_HEAVY_LIMIT]
        key = f'beam_w{width}_d{depth}_mc1'
        cases[key] = []
        for board, pairs in subset:
            seq = random_pairs(rng, depth, pairs)
            cases[key] += timed(lambda: native.beam_search(board, seq, width, exclude_hidden_row_from_pop=rule,
                                                           use_exact_score=True), 1)
    return {k: quantiles(v) for k, v in cases.items() if v}


def main() -> None:
    """記録ごとに盤面を集めて測り、背景負荷と一緒に保存する。"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict] = {source: {} for source in SOURCES}
    samples = [s for source in SOURCES for s in collect(source, counts[source])]
    result = dict(samples=len(samples), prefire_side_frame_counts=counts,
                  native_available=native.NATIVE_AVAILABLE,
                  loadavg_before=os.getloadavg(), timings=bench(samples), loadavg_after=os.getloadavg(),
                  unit='片側1盤面への1呼び出し (ms)、単一プロセス nice 19')
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1), flush=True)


if __name__ == '__main__':
    main()
