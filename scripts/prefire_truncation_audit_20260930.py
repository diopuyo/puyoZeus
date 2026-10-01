"""リーク監査: q の記録を時刻 T で打ち切って ON 再生し、全長 ON 再生の T 未満と一致するかを見る (Phase 2)。

予測層が時刻 t より後の情報を使っていれば、打ち切りの有無で T 未満の表示が変わる。
使い方: PYTHONPATH=. python -m scripts.prefire_truncation_audit_20260930 --cut 300
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.run_prefire_replay_20260930 import OUT as REPLAY, RECORDS, options

SOURCE = 'q_7gc4TgFig'
AUDIT = Path('logs/prefire_prediction/truncation')
DISPLAY_COLUMNS = ('t_sec', 'display_p1', 'display_adv', 'source')


def truncate(cut_sec: float, dest: Path) -> int:
    """記録を最初に t ≥ cut_sec となる display 行の直前で切り、完了行を足す。update 行数を返す。"""
    frames = 0
    with gzip.open(RECORDS / f'{SOURCE}.jsonl.gz', 'rt', encoding='utf-8') as src, \
            gzip.open(dest, 'wt', encoding='utf-8') as out:
        pending: list[str] = []
        for line in src:
            row = json.loads(line)
            if row.get('kind') == 'display' and row['t_sec'] >= cut_sec:
                break
            if row.get('kind') == 'complete':
                break
            pending.append(line)
            if row.get('kind') == 'display':
                out.writelines(pending)
                frames += sum(json.loads(p).get('kind') == 'update' for p in pending)
                pending = []
        out.write(json.dumps(dict(kind='complete', frames=frames)) + '\n')
    return frames


def compare_prefix(full: Path, cut: Path, cut_sec: float) -> dict:
    """T 未満の表示列が全ビット一致するか。"""
    with np.load(full / 'display.npz') as left, np.load(cut / 'display.npz') as right:
        n = int((left['t_sec'] < cut_sec).sum())
        same = {c: bool(np.array_equal(left[c][:n], right[c][:n])) for c in DISPLAY_COLUMNS}
        return dict(rows=n, cut_rows=int(len(right['t_sec'])), identical=same, passed=all(same.values()))


def compare_trace(full: Path, cut: Path, cut_sec: float) -> dict:
    """予測層の内訳 (所要時間の列を除く) も T 未満で一致するか。所要は壁時計なので比べない。"""
    left_path, right_path = full / 'prefire_trace.npz', cut / 'prefire_trace.npz'
    if not left_path.exists() or not right_path.exists():
        return dict(rows=0, passed=left_path.exists() == right_path.exists())
    with np.load(left_path) as left, np.load(right_path) as right:
        columns = [str(c) for c in left['columns']]
        keep = [i for i, c in enumerate(columns) if c != 'compute_ms']
        a, b = left['values'], right['values']
        n = int((a[:, 0] < cut_sec).sum())
        same = bool(len(b) >= n and np.array_equal(a[:n][:, keep], b[:n][:, keep], equal_nan=True))
        return dict(rows=n, cut_rows=int(len(b)), passed=same)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cut', type=float, required=True)
    parser.add_argument('--variant', default='on', choices=('on', 'on_mc', 'bestplay'))
    parser.add_argument('--latency', type=float, default=0.0)
    parser.add_argument('--tag', default='')
    parser.add_argument('--evaluator', default='full', choices=('full', 'fast', 's3', 'gfe'))
    args = parser.parse_args()
    name = args.variant + args.tag
    from scripts import d5_runtime
    d5_runtime.OUT = REPLAY / f'runtime_audit_{name}_{args.cut:g}'
    d5_runtime.install()
    from scripts.replay_exchange_event_20260926 import replay
    root = AUDIT / f'{name}_cut_{args.cut:g}'
    root.mkdir(parents=True, exist_ok=True)
    record = root / f'{SOURCE}.jsonl.gz'
    frames = truncate(args.cut, record)
    replay(record, root, Path('models/exchange_event_v3'), True, None, **options(args.variant, args.latency, args.evaluator))
    result = dict(cut_sec=args.cut, frames=frames, **compare_prefix(REPLAY / name / SOURCE, root, args.cut))
    result['trace'] = compare_trace(REPLAY / name / SOURCE, root, args.cut)
    result['passed'] = bool(result['passed'] and result['trace']['passed'])
    (root / 'AUDIT.json').write_text(json.dumps(result, indent=1))
    record.unlink()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
