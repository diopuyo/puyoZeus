"""原complete→v2→学習済み3seed→専用票。pipe/2P反映は人工fixture。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import torch
from test_live_binding import saved, live
from test_live_binding_v2 import connected
import trained_sidecar as P

ROOT = Path(__file__).resolve().parent


def test_complete_connected_save(connected: Any) -> None:
    torch.set_num_threads(2)
    loader = P.T.backend().load_loader()
    members = loader.load_members(loader.SOURCE_ID, 'cpu')
    path = ROOT/'TRAINED_SIDECAR_v1.json'
    receipt = P.run(connected.capture, members, path, sample_count=32, seed=17)
    packet = json.loads(path.read_bytes())
    assert receipt['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert packet['model_artifact_kind'] == 'ensemble_manifest'
    manifest = json.dumps(packet['trained_details']['members'], sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(manifest).hexdigest() == packet['model_artifact_sha256']
    assert tuple(P.S.S.decode(state) for state in packet['states']) == connected.capture().values
    assert len(packet['states'][0]['hidden_worlds']) == 49 and len(packet['trained_details']['members']) == 3
    assert not packet['quality_gate_clear'] and not packet['accounting_permission']
    with (ROOT/'TRAINED_SIDECAR_RECEIPT_v1.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
