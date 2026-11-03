"""全保存通知を二つの実評価器へ直列再投入し、内部キー・死亡・EMAを照合する。"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Iterator

from scripts import replay_exchange_event_20260926 as replay
from src.exchange_event_record import decode, encode
from src.phase_j.live_cache import bounded_chain_caches
from src.phase_j.live_eval_worker import boundary_tracker
from src.phase_j.live_notification_eval import NotificationExchangeOverlay
from src.phase_j.live_match_number import PublicMatchNumber

COUNTS = dict(updates=0, boundaries=0, old_death_cleared=0)
DIGEST = hashlib.sha256()
NUMBERS = PublicMatchNumber()
PROGRESS_INTERVAL = 5000


def complete_lines(path: Path) -> Iterator[dict]:
    """未完了gzipは完全なJSON行まで。欠損末尾を捏造しない。"""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        while True:
            try:
                line = stream.readline()
            except EOFError:
                return
            if not line:
                return
            try:
                yield decode(json.loads(line))
            except json.JSONDecodeError:
                return


def signature(overlay: Any) -> bytes:
    layers = overlay._e16
    values = (overlay._game, overlay.display, overlay.smoothing_state(),
        sorted(layers.dead_sides) if layers else None,
        overlay.tracker.source, overlay.tracker.probability,
        overlay._landing_projection.death, overlay._landing_projection.death_record,
        overlay.tracker.sealed)
    return pickle.dumps(encode(values))


class PairedOverlay(NotificationExchangeOverlay):
    """公開番号を別管理しても実評価器の死亡状態・数値列が一致することを検査する。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.shadow = NotificationExchangeOverlay(*args, **kwargs)
        for owner in (self, self.shadow):
            owner.tracker = boundary_tracker(owner.tracker.models, owner.tracker)

    def update(self, *args: Any, **kwargs: Any) -> None:
        game = args[4]
        boundary = self._game is not None and self._game != game
        was_dead = bool(self._e16 and self._e16.dead_sides)
        os.environ['PUYO_LIVE_PUBLIC_MATCH_NUMBER'] = '0'
        super().update(*args, **kwargs)
        os.environ['PUYO_LIVE_PUBLIC_MATCH_NUMBER'] = '1'
        # 公開用コピーを実際に作り、内部キーをそのまま評価器へ渡す。
        public = NUMBERS.decorate(dict(game=game))
        self.shadow.update(*args[:4], public['game'], *args[5:], **kwargs)
        before, after = signature(self), signature(self.shadow)
        assert before == after, (args[3], game)
        if boundary and was_dead and not getattr(args[0], 'confirmed_dead_sides', ()):
            assert not self.shadow._e16.dead_sides, (args[3], game)
            COUNTS['old_death_cleared'] += 1
        COUNTS['updates'] += 1
        COUNTS['boundaries'] += int(boundary)
        DIGEST.update(before)
        for owner in (self, self.shadow):
            owner.tracker.sealed, owner.tracker.sealed_diagnostics = [], []
        if COUNTS['updates'] % PROGRESS_INTERVAL == 0:
            print(json.dumps(COUNTS), flush=True)


def main() -> None:
    source, result_path, assets = map(Path, sys.argv[1:])
    metadata = json.loads(result_path.read_text(encoding='utf-8'))
    for bound in metadata['boundaries']:
        NUMBERS.boundary(bound['game'], bound['forced'], bound['scored'], bound['recovery'])
    out = result_path.with_suffix('')
    out.mkdir(exist_ok=True)
    os.chdir(assets)
    for key in ('CPU_THREADS', 'REFERENCE_TIMES_ONLY', 'NEAR_FUTURE_CACHE',
                'EXACT_CONNECTIVITY', 'EXACT_NATIVE'):
        os.environ['PUYO_LIVE_'+key] = '1'
    from src.phase_j.live_cpu import apply_runtime
    apply_runtime('event-evaluation')
    replay.read_records, replay.ExchangeEventOverlay = complete_lines, PairedOverlay
    sys.argv = ['replay', str(source/'inputs.jsonl.gz'), '--out', str(out), '--production-exchange-event']
    with bounded_chain_caches(future_search=True):
        replay.main()
    result = dict(**COUNTS, sha256=DIGEST.hexdigest(), bytes_equal=True)
    result_path.with_name(result_path.stem+'_evaluation.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
