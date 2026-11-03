"""D-v2eの戻り値を変更せず初回証明と内部履歴を採取する。"""
from __future__ import annotations

import json
import sys
from enum import Enum
from collections import deque
from typing import Any

from scripts._diag_mia7_inspect import BASE, OUT, END, guard, save

TARGET = 691.4333333333333
TOLERANCE = 0.0001


def plain(value: Any) -> Any:
    """列挙型を含む入力を監査JSONへ変換する。"""
    import numpy as np
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [plain(v) for v in value]
    if hasattr(value, '_grid'):
        return plain(value._grid)
    if hasattr(value, '__dict__'):
        return plain(vars(value))
    return value


def install(stream: Any) -> None:
    """証明と更新は元関数を一回だけ呼び、値には触れない。"""
    from src.exchange_event_overlay import ExchangeEventOverlay
    from src.exchange_post_counter_bound import PostCounterDeathBound
    update, prove = ExchangeEventOverlay.update, PostCounterDeathBound.prove

    def traced_prove(self: Any, *args: Any, **kwargs: Any) -> dict:
        result = prove(self, *args, **kwargs)
        projection, overlay, idx, latest, incoming, hands, context, stamp = args
        if abs(stamp-TARGET) < TOLERANCE:
            save('first_proof_inputs.json', plain(dict(t=stamp, idx=idx, latest=latest,
                 incoming=incoming, hands=hands, context=context, result=result,
                 safety=projection.safety, chains=[c for r in overlay.tracker.records for c in r.chains],
                 hands_observer=vars(overlay).get('_hands'))))
        return result

    def traced_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        guard()
        result = update(self, *args, **kwargs)
        stamp = args[3]
        if 680 <= stamp <= END:
            row = dict(t=stamp, source=self.tracker.source, probability=self.tracker.probability,
                chains=[c for r in self.tracker.records for c in r.chains],
                ledger=self._landing_projection.safety.ledger,
                hands={k: v for k, v in vars(self).items() if 'hands' in k})
            stream.write(json.dumps(plain(row), ensure_ascii=False, default=str)+'\n')
        return result

    ExchangeEventOverlay.update, PostCounterDeathBound.prove = traced_update, traced_prove


def main() -> None:
    """記録先頭から706秒まで同じ構成で再生する。"""
    guard()
    from scripts.falsedeath_replay_20261001 import enabled_flags, limit_replay
    from scripts import replay_exchange_event_20260926 as replay
    limit_replay(replay, END)
    sys.argv = ['replay', str(BASE / 'collect/records/mia8KCjr52g.jsonl.gz'), '--out', str(OUT / 'replay'),
                '--production-exchange-event', *enabled_flags('v2e')]
    with (OUT / 'trace.jsonl').open('w', encoding='utf-8') as stream:
        install(stream)
        replay.main()
    save('resources_after_replay.json', dict(available_bytes=guard(), nice=10))


if __name__ == '__main__':
    main()
