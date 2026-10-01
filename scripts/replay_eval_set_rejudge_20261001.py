"""評価セット: 再判定の10モデル (旧/補正 × random_state 0〜4) を3パートの記録で再生する (2026-10-01)。

wt_nextfix の再生器 (scripts/next_shift_replay_20261001.run) をそのまま使い、出力先と記録の
置き場所だけ差し替える。実行は PYTHONPATH=/mnt/d/puyo_analyzer/wt_nextfix・cwd=/mnt/d/puyo_analyzer/nextfix_run
(q .505〜.518 を出した再生と同じコード・同じ cwd)。本ファイルは wt_evalset に置き、ファイルパスで起動する。
使い方: python <本ファイル> --variant J_orig_rs0 --source p1|p2|zenchi
"""
from __future__ import annotations

import argparse
from pathlib import Path

EVALSET = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set')
OUT_ROOT = EVALSET/'replay_rejudge'
PART_RECORDS = EVALSET/'collect/records'
PART3_RECORDS = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/pending_expiry/full/records')
SOURCES = ('c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'zenchi')


def install_firing_guard() -> list[dict]:
    """wt_nextfix の weighted_landing を実行時だけ包み、発火入力が無い時刻は加重なしへ落とす。

    wt_evalset の src 修正 (exchange_hidden_row_probability.weighted_landing の firing=None 分岐) と
    同じ挙動。ディスクの wt_nextfix は変えない。発動した時刻を返り値のリストへ記録する (0件も母数として残す)。
    """
    import inspect
    from src import exchange_hidden_row_probability as module
    original = module.weighted_landing
    source = inspect.getsource(original)
    assert source.count('ExchangeEndInput(tracker.firing') == 1 and 'tracker.firing is None' not in source
    fired: list[dict] = []

    def guarded(projection, overlay, snapshot, latest, hands, base, stamp):
        tracker = overlay.tracker
        if tracker.firing is None and getattr(tracker, 'hidden_row_belief', None) is not None:
            record = tracker.current or projection.death_record
            if record is not None and any(tracker.hidden_row_belief.active(c) for c in record.chains):
                fired.append(dict(t_sec=float(stamp), current_is_none=tracker.current is None,
                                  death_record_is_none=projection.death_record is None))
                return None
        return original(projection, overlay, snapshot, latest, hands, base, stamp)
    module.weighted_landing = guarded
    return fired


def main() -> None:
    """1構成×1記録を1プロセスで再生する。"""
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--source', choices=SOURCES, required=True)
    args = parser.parse_args()
    from scripts import next_shift_replay_20261001 as base
    assert Path(base.__file__).resolve().is_relative_to(Path('/mnt/d/puyo_analyzer/wt_nextfix'))
    fired = install_firing_guard()
    base.OUT_ROOT = OUT_ROOT
    base.RECORDS = PART3_RECORDS if args.source == 'zenchi' else PART_RECORDS
    base.run(args.variant, args.source)
    (OUT_ROOT/args.variant/f'firing_guard_{args.source}.json').write_text(
        json.dumps(dict(count=len(fired), rows=fired), indent=1))


if __name__ == '__main__':
    main()
