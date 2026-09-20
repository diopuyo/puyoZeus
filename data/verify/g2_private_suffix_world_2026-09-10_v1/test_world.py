"""新58保存の正常/改変対照。原Link quietの遡及認証はしない。"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
import pytest
import world as W

SOURCE = W.ROOT.parent/'g2_hidden_tail_suffix_runtime_2026-09-10_v1/prefix_cpu_v5'


def read(name: str) -> Any:
    path = SOURCE/name
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines()]
    return json.loads(path.read_text())


@pytest.fixture
def saved() -> Any:
    hidden = read('COMBINED_HIDDEN.json')
    return dict(history_rows=read('directional_history.jsonl'), journal_rows=read('atomic_journal.jsonl'),
        conditional_rows=[], hidden_events=hidden['events'], hidden_history=hidden['history'],
        hidden_lifetime=hidden['lifetimes'], outer_rows=read('POSTCOMMIT_CONSUMER_ROWS.json'))


def test_saved_58_world(saved: Any) -> None:
    result = W.verify_world(**saved)
    assert result['prepared_worlds'] == 3 and result['conditional_frames'] == [34882, 34884, 34886]
    assert result['world_PB_verified'] and result['private_suffix_geometry_verified']
    assert not result['runtime_finalization_allowed'] and not result['actual_live_scope_verified']


def test_original_86_equal() -> None:
    root = W.ROOT.parent/'g2_conditional_shared_runtime_candidate_2026-09-10_v1/prefix_cpu_v2'
    def load(name: str) -> Any:
        path = root/name
        text = path.read_text()
        return [json.loads(line) for line in text.splitlines()] if path.suffix == '.jsonl' else json.loads(text)
    hidden = load('COMBINED_HIDDEN.json')
    args = dict(history_rows=load('directional_history.jsonl'), journal_rows=load('atomic_journal.jsonl'),
        conditional_rows=load('ADMISSION.json')['admission'], hidden_events=hidden['events'],
        hidden_history=hidden['history'], hidden_lifetime=hidden['lifetimes'],
        outer_rows=load('POSTCOMMIT_CONSUMER_ROWS.json'))
    before = W.libraries().verify_world(**args)
    after = W.verify_world(**args)
    assert after.pop('private_suffix_geometry_verified') is False
    assert before == after


@pytest.mark.parametrize('key,value', [('previous_private_placement', 'foreign'),
    ('previous_private_frame', 34850), ('current_permission', True), ('physical_certified', True)])
def test_bad_private_source(saved: Any, key: str, value: Any) -> None:
    row = next(r for r in saved['history_rows'] if r['prepared'] and r['prepared']['kind'] == W.PRIVATE)
    row['prepared'][key] = value
    with pytest.raises((ValueError, KeyError, AssertionError)):
        W.verify_world(**saved)


@pytest.mark.parametrize('mutation', ['missing_source', 'source_digest', 'missing_current', 'missing_history'])
def test_bad_join(saved: Any, mutation: str) -> None:
    event = next(r for r in saved['hidden_events'] if r['kind'] == 'conditional_current_after_original_J')
    if mutation in ('missing_source', 'source_digest'):
        proof = json.loads(event['current']['evidence_json'])
        if mutation == 'missing_source':
            proof.pop(W.SOURCE_KEY)
        else:
            proof[W.SOURCE_KEY]['previous_private_placement'] = 'foreign'
        event['current']['evidence_json'] = json.dumps(proof, sort_keys=True, separators=(',', ':'))
    elif mutation == 'missing_current':
        saved['hidden_events'].remove(event)
    else:
        saved['hidden_history'] = [r for r in saved['hidden_history'] if r['kind'] != W.PRIVATE]
    with pytest.raises((ValueError, KeyError, AssertionError)):
        W.verify_world(**saved)
