"""診断・合否外: reviewだけを再生し、E35の未知手持ちを同一入力で対比する。"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

from scripts.falsedeath_replay_20261001 import OUT, enabled_flags, limit_replay
from scripts._diag_set2_falsedeath_replay import install_progress, plain
from src.exchange_post_counter_bound import PostCounterDeathBound

DEST = OUT / 'v2b_ablate'
START, END, STOP = 2760., 2771., 2775.2
RECEIVER = 1
MIN_MEMORY_KIB = 6 * 1024**2
REMOVED = {'A': '--post-counter-current-puyo-integrity',
           'B': '--score-attribution-v2', 'control': None}


def save(path: Path, value: Any) -> None:
    """新規ファイルだけを書き込む。"""
    with path.open('x', encoding='utf-8') as stream:
        json.dump(plain(value), stream, ensure_ascii=False, indent=1)


def state_row(engine: Any, projection: Any, overlay: Any, stamp: float) -> dict:
    """物理状態と終了待ち、配置観測を区別して保存する。"""
    side = engine._diagnostic_side
    chain = overlay.tracker.latest_chain('2P')
    hands = projection.hands_observation
    names = ('chain_id', 'trigger_sec', 'end_signal_sec', 'end_confirmed',
             'end_reason', 'post_end_drop_sec', 'predicted_final_score')
    return dict(t_sec=stamp, state=side.state.name,
                physical_chain=side.state.name == 'CHAIN',
                chaining=projection._chaining(overlay.tracker, RECEIVER),
                placed=None if hands is None else hands.placed[RECEIVER],
                last_placement=None if hands is None else hands.last_placement[RECEIVER],
                placement_state=('発火手設置済' if side.state.name == 'CHAIN' else
                                 '確定盤面・手持ち不明' if side.state.name == 'STABLE' else
                                 'ツモ落下中' if side.state.name == 'TSUMO_FALL' else '未確定'),
                next1=side.next_pair, next2=side.dnext_pair,
                next_motion=side.next_slide_motion,
                chain={n: getattr(chain, n, None) for n in names})


def shadow(engine: Any, current: bool) -> Any:
    """元エンジンの入力を複製し、独立キャッシュで反実仮想を計算する。"""
    cloned = copy.copy(engine)
    cloned.current_puyo_integrity = current
    cloned.cache, cloned.board_cache, cloned.audit = {}, {}, []
    return cloned


def install_probe(rows: list[dict]) -> None:
    """E35の全呼出しを観測し、表示側へ反実仮想を戻さない。"""
    original_observe = PostCounterDeathBound.observe
    original_prove = PostCounterDeathBound.prove

    def observe(self: Any, result: Any, game: int) -> None:
        self._diagnostic_side = result.p2
        original_observe(self, result, game)

    def prove(self: Any, projection: Any, overlay: Any, idx: int, latest: tuple,
              incoming: int, hands: int, context: dict, stamp: float) -> dict:
        actual = original_prove(self, projection, overlay, idx, latest,
                                incoming, hands, context, stamp)
        if idx != RECEIVER or not START <= stamp <= END:
            return actual
        row = state_row(self, projection, overlay, stamp)
        row.update(call=len(rows) + 1, incoming=incoming, hands=hands, actual=actual,
                   elapsed=overlay.tracker._score_elapsed, palette=self.colors[idx].active(),
                   queue=self.queues[idx], context=plain(context))
        for name, current in (('with_unknown', True), ('without_unknown', False)):
            engine = shadow(self, current)
            result = original_prove(engine, projection, overlay, idx, latest,
                                    incoming, hands, context, stamp)
            row[name] = result
            row[name + '_audit'] = engine.audit
            if current == self.current_puyo_integrity:
                assert result == actual, '同条件の証明再計算が不一致'
        rows.append(row)
        return actual

    PostCounterDeathBound.observe = observe
    PostCounterDeathBound.prove = prove


def run(mode: str) -> None:
    """前回と同じ記録先頭・終了秒・CLIで、指定の一要素だけを外す。"""
    directory = DEST / mode
    directory.mkdir(parents=True, exist_ok=False)
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0])
    assert os.getpriority(os.PRIO_PROCESS, 0) == 10 and available >= MIN_MEMORY_KIB
    save(directory / 'resources.json', dict(nice=10, memory_available_kib=available))
    rows: list[dict] = []
    install_probe(rows)
    install_progress()
    from scripts import replay_exchange_event_20260926 as replay
    limit_replay(replay, STOP)
    flags = [f for f in enabled_flags('v2b') if f != REMOVED[mode]]
    sys.argv = ['replay', str(OUT / 'collect/records/review.jsonl.gz'),
                '--out', str(directory), '--production-exchange-event', *flags]
    save(directory / 'COMMAND.json', sys.argv)
    replay.main()
    save(directory / 'calls.json', rows)
    if mode == 'control':
        save(directory / 'equivalence.json', replay.compare(OUT / 'v2b/short/review', directory))


def main() -> None:
    """診断構成を一つずつ別プロセスで実行する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=tuple(REMOVED))
    run(parser.parse_args().mode)


if __name__ == '__main__':
    main()
