"""保存済み発火起点だけをOFFへ差し替える境界介入（認識無効化とは別）。"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Iterator
import numpy as np
from scripts import replay_exchange_event_20260926 as replay_module
from scripts._d4_loss import OUT, Q, directory, save
from scripts.r1_measure_helpers import lines
from src.exchange_event_record import read_records
from src.exchange_event_overlay import ExchangeEventOverlay
from src.board_state_machine import BoardState
from types import SimpleNamespace

COHORT_END = 10704/30
KINDS = ('next', 'ojama', 'formula')


def board_equal(left: object, right: object) -> bool:
    """欠測を実盤面と混同しない。"""
    if left is None or right is None:
        return left is right
    return bool(np.array_equal(left._grid, right._grid))


class Intervention:
    """修正の直後に変わった発火起点の経路だけを分離する。"""
    def __init__(self, signal: str) -> None:
        self.signal = signal
        self.before = {r['args'][3]: r['args'] for r in read_records(Path('logs/e31/records')/f'{Q}.jsonl.gz') if r['kind']=='update'}
        self.corrections = [r for r in lines(Path('logs/r1/capture')/Q/'placement_signal_reconcile.jsonl') if r['corrections']]
        self.audit: list[dict] = []
        self.stable = {(game, side): [] for game in {v[4] for v in self.before.values()} for side in range(2)}
        for values in self.before.values():
            for side, label in enumerate(('p1', 'p2')):
                value = getattr(values[0], label)
                if value.state == BoardState.STABLE and value.confirmed_board is not None:
                    self.stable[values[4], side].append((values[3], value.confirmed_board))

    def install(self) -> None:
        """before_board欠測時の履歴選択も含め、物理完走の起点だけを介入する。"""
        original = ExchangeEventOverlay._predict_completion_base
        def predict(overlay: object, chain: object, event: object, idx: int, history: list | None = None) -> None:
            recent = [r for r in self.corrections if r['side']==idx and r['t_sec'] <= chain.trigger_sec]
            last = recent[-1] if recent else None
            candidates = [r for r in self.stable.get((overlay._game, idx), []) if r[0] < chain.trigger_sec]
            if last is not None and last['signal']==self.signal and candidates:
                selected = (overlay._history[idx] if history is None else history)
                saved = next((s for s in reversed(selected) if s.t_sec < chain.trigger_sec), None)
                board = getattr(event, 'before_board', None)
                board = board if board is not None else saved.board if saved else None
                if not board_equal(board, candidates[-1][1]):
                    self.audit.append(dict(game=overlay._game, trigger=chain.trigger_sec,
                        side=idx, chain=chain.chain_id, correction_t=last['t_sec'], signal=self.signal,
                        off_board=candidates[-1][1]._grid.tolist(), on_board=board._grid.tolist() if board else None))
                    event = SimpleNamespace(before_board=candidates[-1][1].copy())
            original(overlay, chain, event, idx, history)
        ExchangeEventOverlay._predict_completion_base = predict

    def replace(self, args: tuple) -> None:
        """同時刻・同側・同連鎖通知のbefore_boardだけを差し替える。"""
        old = self.before[args[3]]
        assert old[4] == args[4]
        for side, label in enumerate(('p1', 'p2')):
            on, off = getattr(args[0], label).chain_event, getattr(old[0], label).chain_event
            if on is None or off is None or on.trigger_sec != off.trigger_sec:
                continue
            if board_equal(getattr(on, 'before_board', None), getattr(off, 'before_board', None)):
                continue
            recent = [r for r in self.corrections if r['side']==side and r['t_sec'] <= on.trigger_sec]
            last = recent[-1] if recent else None
            if last is None or last['signal'] != self.signal:
                continue
            self.audit.append(dict(t=args[3], trigger=on.trigger_sec, side=side,
                correction_t=last['t_sec'], signal=self.signal))
            on.before_board = deepcopy(getattr(off, 'before_board', None))

    def records(self, path: Path) -> Iterator[dict]:
        """全採点行まで元の順序を維持し、介入外の値を保持する。"""
        frames = 0
        for row in read_records(path):
            if row['kind'] == 'update':
                if row['args'][3] >= COHORT_END:
                    yield dict(kind='complete', frames=frames)
                    return
                frames += 1
            yield row


def main() -> None:
    """元の本番CLIを使い、起点介入と直接の合図無効化を明示的に区別する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--signal', choices=KINDS, required=True)
    args = parser.parse_args()
    intervention = Intervention(args.signal)
    dest = OUT/'completion_ablation'/args.signal
    dest.mkdir(parents=True, exist_ok=True)
    builder = replay_module.static_builder(Path(f'logs/r1/records/{Q}.jsonl.gz'))
    replay_module.static_builder = lambda path: builder
    replay_module.read_records = intervention.records
    intervention.install()
    sys.argv = ['d4', f'logs/r1/records/{Q}.jsonl.gz', '--out', str(dest), '--production-exchange-event']
    replay_module.main()
    rows = json.loads((OUT/'q_rows.json').read_text())
    data = dict(np.load(dest/'display.npz'))
    p = np.clip(data['display_p1'][[r['i'] for r in rows]], 1e-7, 1-1e-7)
    y = np.array([r['y'] for r in rows])
    loss = float(np.mean(-np.log(np.where(y, p, 1-p))))
    result = dict(signal=args.signal, n=len(rows), log_loss=loss,
        changed_notifications=len(intervention.audit), interventions=intervention.audit,
        scope='同側直前合図別の物理完走起点だけをOFFへ差替え。認識の合図無効化ではない')
    (dest/'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(result, flush=True)


if __name__ == '__main__':
    main()
