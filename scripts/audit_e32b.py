"""E32b: 評価処理を変更せず、因果性・一致率差分・得点誤差を監査する。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import numpy as np

from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.run_e32 import OPTIONS

OUT = Path('logs/e32b')
RECORDS = Path('logs/e31/records')
SOURCES = ('q_7gc4TgFig', 'fcXG83vInDY', 'mia8KCjr52g')
TOP_INTERVALS = 10
TOP_ERRORS = 5
WORKERS = 2


def save(path: Path, value: Any) -> None:
    """監査結果をUTF-8の原票として保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prior(version: str, source: str) -> Path:
    """既存成果物の場所だけを解決する。"""
    suffix = source if source == 'zenchi' else f'renders/{source}/on'
    return Path(f'logs/{version}/on') / suffix


def load_rows(source: str, version: str = 'e32') -> list[dict]:
    """事後採点済みの採否原票を読む。"""
    return json.loads((prior(version, source)/'snapshot_final_audit.json').read_text())['rows']


def differences() -> dict:
    """元の一致率定義を保ち、連続する反転行を区間へまとめる。"""
    old, new = [np.load(prior(v, 'zenchi')/'display.npz') for v in ('e31', 'e32')]
    np.testing.assert_array_equal(old['t_sec'], new['t_sec'])
    times = new['t_sec']
    games = json.loads(Path('logs/review_zenchi_part3/official_games.json').read_text())
    intervals, gross = [], Counter()
    for game in games:
        sign = 1 if game['winner'] == '1P' else -1
        mask = (times >= game['start']+(game['end']-game['start'])*2/3) & (times < game['end'])
        left, right = old['display_adv']*sign > 0, new['display_adv']*sign > 0
        delta = right.astype(int)-left.astype(int)
        gross.update(n=int(mask.sum()), old=int(left[mask].sum()), new=int(right[mask].sum()),
                     gain=int(((delta == 1)&mask).sum()), loss=int(((delta == -1)&mask).sum()))
        selected = np.flatnonzero(mask & (delta != 0))
        groups = np.split(selected, np.flatnonzero((np.diff(selected) != 1) |
                          (np.diff(delta[selected]) != 0))+1)
        for group in groups:
            if not len(group):
                continue
            a, b = int(group[0]), int(group[-1])
            intervals.append(dict(game=game['game'], internal_game=int(new['game_idx'][a]),
                start=float(times[a]), end=float(times[b]), n=len(group), delta=int(delta[a]),
                first_row=a, last_row=b, old_source=str(old['source'][a]), new_source=str(new['source'][a]),
                old_p1=float(old['display_p1'][a]), new_p1=float(new['display_p1'][a])))
    return dict(counts=dict(gross), intervals=sorted(intervals, key=lambda r:r['n'], reverse=True))


def prepare() -> None:
    """既存原票から対象を選び、未来の静止特徴も含めず入力を切る。"""
    diff = differences()
    errors = sorted([dict(r, source=s) for s in SOURCES for r in load_rows(s)
                     if r['accepted'] and r['error'] is not None], key=lambda r:abs(r['error']), reverse=True)
    accepted = [r for r in load_rows('zenchi') if r['accepted']]
    save(OUT/'preliminary.json', dict(differences=diff, top_errors=errors[:TOP_ERRORS], accepted=accepted))
    print(json.dumps(dict(counts=diff['counts'],intervals=len(diff['intervals']),
        errors=[{k:r[k] for k in ('source','game','side','chain_id','error')} for r in errors[:TOP_ERRORS]]),indent=2))


def truncate(cutoffs: list[float]) -> None:
    """元ファイルの連続接頭辞のみを使い、完了行の件数だけ置き換える。"""
    for cutoff in cutoffs:
        dest = OUT/'inputs'/f'zenchi_{cutoff:.3f}.jsonl.gz'
        dest.parent.mkdir(parents=True, exist_ok=True)
        frames = 0
        with gzip.open(RECORDS/'zenchi.jsonl.gz', 'rt') as source, gzip.open(dest, 'wt') as target:
            for line in source:
                row = json.loads(line)
                if row['kind'] == 'complete':
                    break
                if row['kind'] == 'update':
                    if row['args']['tuple'][3] > cutoff:
                        break
                    frames += 1
                target.write(line)
            target.write(json.dumps(dict(kind='complete', frames=frames))+'\n')
    save(OUT/'cutoffs.json', cutoffs)


def serial(value: Any) -> Any:
    """dataclassを複製せずJSONへ渡し、全フィールドを厳密に比較する。"""
    if is_dataclass(value):
        return vars(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class Trace:
    """評価器を読むだけで、各入力時点の原票と候補重みを記録する。"""
    def __init__(self, out: Path, cutoffs: list[float], event_hashes: bool = True) -> None:
        self.out, self.cutoffs = out, cutoffs
        self.event_hashes = event_hashes
        self.stream = gzip.open(out/'trace.jsonl.gz', 'wt', encoding='utf-8')
        self.changed: dict = {}
        self.chains: dict = {}
        self.options: list = []
        self.last_overlay: Any = None

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        """後から変わるイベントも、そのフレームの全内容をハッシュ化する。"""
        stamp, game = inputs[3:5]
        tracker, engine = overlay.tracker, overlay._prefire
        if self.event_hashes:
            payload = json.dumps(dict(records=tracker.records, diagnostics=tracker.diagnostics),
                                 default=serial, ensure_ascii=False, allow_nan=False, separators=(',',':'))
            state = dict(t=stamp, game=game, hash=hashlib.sha256(payload.encode()).hexdigest(),
                         records=len(tracker.records), probability=tracker.probability, source=tracker.source)
            self.stream.write(json.dumps(state, ensure_ascii=False)+'\n')
        if self.cutoffs and stamp > self.cutoffs[0] and self.last_overlay is not None:
            raise AssertionError('打ち切り時刻は実在フレームから選ぶこと')
        if self.cutoffs and stamp == self.cutoffs[0]:
            tracker.save(self.out/f'checkpoint_{self.cutoffs.pop(0):.3f}.jsonl')
        for key, entry in engine.entries.items():
            identity = (game, key)
            self.chains[identity] = entry['chain']
            signature = (id(entry['options']), entry['chain'].end_signal_sec, entry['chain'].score_delta)
            if self.changed.get(identity) != signature:
                self.capture(entry, inputs)
                self.changed[identity] = signature
        self.last_overlay = overlay

    def capture(self, entry: dict, inputs: tuple) -> None:
        """全候補の得点系列ごとに事後重みをまとめ、使用観測を残す。"""
        chain = entry['chain']
        mass: Counter = Counter()
        for option in entry['options']:
            mass[tuple(option['prefix'])] += option['weight']
        idx = int(chain.side == '2P')
        event = (inputs[0].p1, inputs[0].p2)[idx].chain_event
        self.options.append(dict(t=inputs[3], game=inputs[4], side=chain.side, chain_id=chain.chain_id,
            accepted=entry['audit']['accepted'], withdrawn=entry['audit']['withdrawn'],
            formula_visible=inputs[7][idx], formula_total=inputs[5][idx],
            event=None if event is None else {k:getattr(event,k) for k in ('trigger_sec','chain_count','total_score','mechanism')},
            score_delta=chain.score_delta, end_signal_sec=chain.end_signal_sec,
            predicted_score=chain.predicted_final_score,
            distribution=[dict(prefix=k, weight=v) for k,v in sorted(mass.items())]))

    def close(self) -> None:
        """実行完了後の情報は別ファイルに隔離する。"""
        self.stream.close()
        save(self.out/'candidates.json', self.options)
        save(self.out/'chains.json', [dict(game=k[0], **asdict(c)) for k,c in self.chains.items()])


def worker(source: str, cutoff: float | None) -> None:
    """既存replayに観測専用フックを渡す。"""
    name = source if cutoff is None else f'{source}_{cutoff:.3f}'
    out = OUT/'replays'/name
    out.mkdir(parents=True, exist_ok=True)
    record = RECORDS/f'{source}.jsonl.gz' if cutoff is None else OUT/'inputs'/f'{name}.jsonl.gz'
    cutoffs = json.loads((OUT/'cutoffs.json').read_text()) if source == 'zenchi' and cutoff is None else []
    trace = Trace(out, cutoffs, event_hashes=source == 'zenchi')
    status = replay(record, out, Path('models/exchange_event_v3'), True, trace, **OPTIONS)
    trace.close()
    if cutoff is None:
        status['saved_e32_equivalence'] = compare(prior('e32', source), out)
    save(out/'DONE.json', status)


def launch(task: tuple[str, float | None]) -> None:
    """監査再生を独立プロセスで起動する。"""
    source, cutoff = task
    name = source if cutoff is None else f'{source}_{cutoff:.3f}'
    command = [sys.executable, '-m', 'scripts.audit_e32b', '--worker', source]
    if cutoff is not None:
        command += ['--cutoff', str(cutoff)]
    with (OUT/f'{name}.log').open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    """選定・切断・再生を明示的な監査段階に分ける。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--cutoffs', nargs='+', type=float)
    parser.add_argument('--worker')
    parser.add_argument('--cutoff', type=float)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare()
    if args.cutoffs:
        truncate(args.cutoffs)
    if args.worker:
        worker(args.worker, args.cutoff)
    if args.run:
        cutoffs = json.loads((OUT/'cutoffs.json').read_text())
        tasks = [('zenchi', None), *[(s,None) for s in SOURCES[:2]], *[('zenchi',t) for t in cutoffs]]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(launch, tasks))


if __name__ == '__main__':
    main()
