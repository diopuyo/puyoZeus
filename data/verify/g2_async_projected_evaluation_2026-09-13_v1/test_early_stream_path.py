"""原空tail回復の保存先検査を、実ファイルと早期Streamで再現する。"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as N
from typing import Any
import pytest
import stream_witness as W

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location('early_path_proof',
    ROOT / 'g2_empty_tail_reset_live_adapter_2026-09-11_v1/proof.py')
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)
FRAME = 35158


def test_original_recovery_rejects_missing_stream_name(tmp_path: Any) -> None:
    history_path = tmp_path / 'directional_history.jsonl'
    history_path.write_text(json.dumps({'scope': {'frame_idx': FRAME}}) + '\n')
    journal_path = tmp_path / 'atomic_journal.jsonl'
    journal_path.write_text('{}\n')
    metadata_path = tmp_path / 'collector_metadata.jsonl'
    metadata_path.write_text('{}\n')
    with history_path.open('a') as hs, journal_path.open('a') as js, metadata_path.open('a') as ms:
        witness = N(original=js)
        proxy = W.Stream(witness)
        state = {'output': tmp_path,
                 'live_history_sink': N(closed=False, errors=[], stream=hs),
                 'collector_metadata_sink': N(closed=False, errors=[], busy=False, rows=N(stream=ms))}
        with pytest.raises(AttributeError, match='name'):
            P.history(state, N(stream=proxy), FRAME)
        assert not js.closed and proxy.witness.original is js
        assert P.history(state, N(stream=js), FRAME)[0][-1]['scope']['frame_idx'] == FRAME
