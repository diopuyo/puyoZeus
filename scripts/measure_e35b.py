"""E35bの固定入力採取・段別計測・旧実装との厳密比較。"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from scripts import run_e35
from scripts.replay_exchange_event_20260926 import compare
from scripts.run_e3_exchange_eval_20260926 import save_json
import src.exchange_post_counter_bound as integration

OUT = Path('logs/e35b')


def directory(root: Path, source: str) -> Path:
    """既存の5記録の配置を共有する。"""
    suffix = source if source in ('review', 'zenchi') else f'renders/{source}/on'
    return root/'on'/suffix


def plain(value: Any) -> Any:
    """JSONのtuple/list表現差と計測時間だけを除く。"""
    return json.loads(json.dumps({k: v for k, v in value.items() if k != 'elapsed_sec'}))


class Capture:
    """既存監査の証明を再生し、同じ評価経路から全入力を採取する。"""

    def __init__(self, source: str) -> None:
        self.expected = json.loads((directory(Path('logs/e35'), source)/
                                    'post_counter_bound_audit.json').read_text())['rows']
        self.rows: list[dict] = []
        self.proof_index = 0

    def proof(self, board: Any, queue: tuple, incoming: int, hands: int,
              elapsed: float, palette: tuple) -> dict:
        """旧証明を順番通り返す。計算時間の採取は別工程で行う。"""
        expected = self.expected[len(self.rows)]
        assert [incoming, hands, list(queue), list(palette)] == [expected[k] for k in
                                                                 ('incoming', 'hands', 'queue', 'palette')]
        result = expected['proofs'][self.proof_index]
        self.proof_index += 1
        return result

    def install(self) -> None:
        """本体を編集せず、採取プロセス内だけで証明結果を注入する。"""
        original = integration.PostCounterDeathBound.prove
        def traced(engine: Any, projection: Any, overlay: Any, idx: int, latest: tuple,
                   incoming: int, hands: int, context: dict, stamp: float) -> dict:
            before = len(engine.audit)
            self.proof_index = 0
            value = original(engine, projection, overlay, idx, latest, incoming, hands, context, stamp)
            if len(engine.audit) == before:
                return value
            actual, expected = engine.audit[-1], self.expected[len(self.rows)]
            assert plain(actual) == plain(expected), (len(self.rows), actual, expected)
            hidden = context['hidden'][idx]
            boards = ([v['board'] for v in hidden['options']] if hidden else
                      [context['replies'][idx]._grid.tolist()])
            assert self.proof_index == (0 if expected['cached'] else len(boards))
            self.rows.append(dict(audit=expected, boards=boards, elapsed=overlay.tracker._score_elapsed))
            return value
        integration.prove_post_counter = self.proof
        integration.PostCounterDeathBound.prove = traced
        # 採取では旧実装と同じ候補単位の呼出順を保ち、新キャッシュを使わない。
        integration.PostCounterDeathBound.proofs_for = lambda engine, boards, *args: [
            self.proof(board, *args) for board in boards]


def capture(source: str) -> None:
    """全行の表示・イベント一致まで確認してから入力を確定する。"""
    trace = Capture(source)
    trace.install()
    run_e35.OUT = OUT/'capture'
    run_e35.worker(source)
    assert len(trace.rows) == len(trace.expected)
    check = compare(directory(Path('logs/e35'), source), directory(run_e35.OUT, source))
    with gzip.open(OUT/f'{source}.inputs.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(trace.rows, stream, ensure_ascii=False)
    save_json(OUT/f'{source}.capture.json', dict(rows=len(trace.rows), **check))


def main() -> None:
    """記録単位で再開可能な入力採取を行う。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=run_e35.SOURCES, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    capture(args.source)


if __name__ == '__main__':
    main()
