"""移送前の全出力と検証済み末尾を連結し、指紋と再開位置を保持する。"""
import json
from pathlib import Path

from scripts.image_free_transfer_worker import promote
import pytest


def test_promote_preserves_prefix_and_verified_midpoint_chunks(tmp_path: Path) -> None:
    snapshot, proof = tmp_path/'snapshot', tmp_path/'proof'
    snapshot.mkdir()
    proof.mkdir()
    (snapshot/'old.gz').write_bytes(b'original')
    (snapshot/'checkpoint.json').write_text(json.dumps(dict(chunks=['old.gz'], frame=300)))
    for name in ('mid.gz', 'end.gz'):
        (proof/name).write_bytes(name.encode())
    (proof/'checkpoint.json').write_text(json.dumps(dict(chunks=['mid.gz', 'end.gz'], frame=600,
        state='600.pkl', fingerprint='local-valid')))
    state = dict(frame=600, processed_frames=300, fingerprint='local-valid', runtimes=[])
    promote(dict(id='prefix'), dict(production_output=str(tmp_path/'production')), snapshot, proof, state)
    target = tmp_path/'production/prefix'
    cp = json.loads((target/'checkpoint.json').read_text())
    assert cp['chunks'] == ['old.gz', 'mid.gz', 'end.gz']
    assert cp['frame'] == 600 and cp['fingerprint'] == 'local-valid'
    assert (target/'old.gz').read_bytes() == b'original'
    assert state['processed_frames'] == 600


def test_interrupted_proof_is_not_appended_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from scripts.image_free_transfer_worker import run
    output = tmp_path/'prefix'
    output.mkdir()
    manifest = output/'checkpoint.json'
    manifest.write_text('preserve-existing-proof')
    with pytest.raises(RuntimeError, match='中断済み'):
        run(dict(id='prefix'), dict(output=str(tmp_path)), object())
    assert manifest.read_text() == 'preserve-existing-proof'
