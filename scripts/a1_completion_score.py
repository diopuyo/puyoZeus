"""A1事前登録の18記録・115試合を既存採点定義で比較する。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

from scripts.a1_completion_replay import OUT, INPUT
from scripts import falsedeath_score_20261001 as prior
from scripts import falsedeath_v2_score_20261002 as v2
from scripts import falsedeath_v2d_display as display
from scripts.falsedeath_replay_20261001 import records


def configure_labels() -> None:
    """固定ラベルの基準イベントは元worktreeの既存e19記録から読む。"""
    from scripts import eval_set2_regression_20261001 as regression
    regression.OUT = INPUT.parents[2]/'set2'


def save(name: str, value: Any) -> None:
    """採点原票をA1専用出力先だけに保存する。"""
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')


def compatibility() -> dict:
    """HEADの本番と修正後OFFの全表示列・イベントのバイトを比較する。"""
    output = {}
    for name in prior.SOURCES:
        folders = [OUT/'replay'/mode/name for mode in ('original', 'prod')]
        files = {}
        for filename in ('display.npz', 'events.jsonl'):
            values = [(folder/filename).read_bytes() for folder in folders]
            files[filename] = dict(identical=values[0] == values[1],
                sha256=[hashlib.sha256(value).hexdigest() for value in values])
            assert files[filename]['identical'], (name, filename)
        output[name] = files
    return output


def histories() -> dict:
    """同じ18記録の試合開始未観測数を集計する。"""
    result = {}
    for mode in ('prod', 'a1'):
        rows = {}
        for name in records():
            games = json.loads((OUT/'replay'/mode/name/'history_integrity_audit.json').read_text())['games']
            rows[name] = dict(games=len(games), incomplete=sum(not g['history_complete'] for g in games))
        result[mode] = dict(records=rows, games=sum(r['games'] for r in rows.values()),
                            incomplete=sum(r['incomplete'] for r in rows.values()))
    return result


def score() -> dict:
    """再収集入力を両構成へ使い、門v2の不等号とbootstrapをそのまま適用する。"""
    original_out = prior.OUT
    configure_labels()
    cached_terminals = json.loads((original_out/'TERMINALS.json').read_text())['all']
    prior.OUT = display.OUT = OUT
    prior.input_record = lambda mode, part: INPUT/f'{part}.jsonl.gz'
    terminals = {name: prior.terminal_rows('a1', name) for name in records()}
    save('TERMINALS.json', terminals)
    assert terminals == cached_terminals, '観測死亡の原票が既存採点と不一致'
    v2.MODE = 'a1'
    sets, existing = v2.evaluate_visible(terminals)
    combined = v2.combined(sets)
    regression = prior.regression('a1')
    frames = {mode: {name: display.scan(name, mode, display.fixed_labels(), terminals[name])
                    for name in prior.SOURCES} for mode in ('prod', 'a1')}
    clean = (existing['false'] == existing['unresolved'] == 0
        and all(s['deaths']['after']['false'] == s['deaths']['after']['false_frames'] == 0 for s in sets.values())
        and all(r['false_frames'] == r['unknown_frames'] == 0 for r in frames['a1'].values()))
    return dict(sets=sets, existing5=existing, combined=combined, regression=regression,
        baseline_regression=prior.regression('prod'),
        display=frames, histories=histories(), reversals=v2.reversals(sets),
        gates=dict(false_death=clean, regression=regression['passed'], **v2.metric_gates(combined)),
        warmup={mode: v2.warmup_rows(mode) for mode in ('prod', 'a1')})


def main() -> None:
    """全41再生の完了後に互換・本番対差・火力誤差を一度に保存する。"""
    save('COMPATIBILITY.json', compatibility())
    result = score()
    save('RESULT.json', result)
    print(json.dumps(result['gates']), flush=True)


if __name__ == '__main__':
    main()
