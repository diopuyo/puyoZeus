"""保存票由来の人工fixtureで有理時計・子process・失敗保存を検査する。GTではない。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest
from scripts import g3_model_process as P

REGISTRY = P.VERIFY / 'g3_repair_2026-09-15_v1/MODEL_PROCESS_REGISTRY.json'
REGISTRY_SHA = '51cd5547c8223aee8c82f6036dd2f44139c004ab63ba06c43b7d188bbf1a5551'


def item(source: str) -> dict[str, Any]:
    return P.route(REGISTRY, REGISTRY_SHA, source)


@pytest.mark.parametrize('source', ['video_39', 'video_c50'])
def test_exact_clock_and_scope_rejections(source: str) -> None:
    value = item(source)
    frame = 100
    num, den = value['time_base']
    row = dict(source_id=value['source_id'], frame_idx=frame, time_sec=frame*num/den)
    request = dict(row=row, frame=frame)
    P.request_gate(request, value)
    for bad in (dict(source_id='other'), dict(time_sec=frame/60), dict(frame_idx=frame+1)):
        with pytest.raises(ValueError):
            P.request_gate(dict(row=row | bad, frame=frame), value)
    with pytest.raises(ValueError):
        P.request_gate(dict(row=row, frame=value['end_frame_exclusive']), value)


def artificial_request(source: str) -> dict[str, Any]:
    """明示的に人工source/時計へ変換。別runの真値として扱わない。"""
    pub = P.M.VERIFY / 'g2_belief_live_publication_2026-09-11_v1'
    reference = json.loads((pub / 'TRAINED_SIDECAR_v1.json').read_bytes())
    raw = P.M.VERIFY / 'g2_empty_tail_reset_integration_2026-09-11_v1/prefix_cpu_v45/provisional_context.jsonl'
    with raw.open() as stream:
        row = next(value for line in stream if (value := json.loads(line))['frame_idx'] == reference['frame'])
    value = item(source)
    old_source, old_run = row['source_id'], row['run_id']
    def rewrite(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: rewrite(val) for key, val in obj.items()}
        if isinstance(obj, list):
            return [rewrite(val) for val in obj]
        if obj == old_source:
            return value['source_id']
        if obj == old_run:
            return 'g3-artificial-process-fixture'
        return obj
    row, states = rewrite(row), rewrite(reference['states'])
    num, den = value['time_base']
    row['time_sec'] = row['frame_idx'] * num / den
    row['update']['time_sec'] = row['update']['returned_time_sec'] = row['time_sec']
    return dict(schema='belief-model-process-request/v1', frame=row['frame_idx'],
                tokens=reference['journal_tokens'], context_digest=hashlib.sha256(P.encoded(row)).hexdigest(),
                row=row, states=states, observed=[row['sides'][s]['before_hold']['confirmed']['grid']
                for s in ('1P', '2P')], seed=17, sample_count=2)


@pytest.fixture(autouse=True)
def process_root(monkeypatch: Any, tmp_path: Path) -> Path:
    """子プロセスの出力先の根を試験用の一時領域へ差し替える (2026-09-17、W45)。

    `exchange` は出力が verify の根の下にあることを要求する
    (`scripts/g3_model_process.py` の `process_output_root`)。本番の安全弁なので
    **緩めない**。根そのものを差し替えて契約を生かしたまま通す。
    これが無かったため、3件が `process_output_root` で止まり、
    本来確かめたい `model_process_failed` まで到達していなかった。
    """
    monkeypatch.setattr(P, "VERIFY", tmp_path)
    return tmp_path


def test_output_outside_the_root_is_rejected(tmp_path: Path) -> None:
    """根の外へは書かせない。差し替えても契約そのものは生きている。"""
    outside = tmp_path.parent / (tmp_path.name + "_outside")
    with pytest.raises(ValueError, match="g3_source_models:process_output_root"):
        P.exchange(artificial_request("video_c50"), REGISTRY, REGISTRY_SHA,
                   "video_c50", outside)


def test_real_child_c50_and_parent_noninterference(tmp_path: Path) -> None:
    request = artificial_request('video_c50')
    before = {k: v for k, v in sys.modules.items() if k == 'src' or k.startswith('src.')}
    packet = P.exchange(request, REGISTRY, REGISTRY_SHA, 'video_c50', tmp_path / 'child')
    after = {k: v for k, v in sys.modules.items() if k == 'src' or k.startswith('src.')}
    assert before.keys() == after.keys() and all(after[k] is v for k, v in before.items())
    assert packet['frame'] == request['frame'] and packet['quality_gate_clear'] is False
    result = json.loads((tmp_path / 'child/RESULT.json').read_text())
    assert result['actual_wait'] is True and result['exit_code'] == 0


def test_corrupt_child_request_saved(tmp_path: Path) -> None:
    request = artificial_request('video_c50')
    request['tokens'][1] = request['tokens'][0]
    with pytest.raises(ValueError, match='model_process_failed'):
        P.exchange(request, REGISTRY, REGISTRY_SHA, 'video_c50', tmp_path / 'failed')
    result = json.loads((tmp_path / 'failed/RESULT.json').read_text())
    assert result['status'] == 'FAILED' and result['exit_code'] != 0
    assert (tmp_path / 'failed/stderr.log').stat().st_size > 0
    assert not (tmp_path / 'failed/PACKET.json').exists()


def test_frozen_parent_video38_normal_control(tmp_path: Path) -> None:
    """元frozen fixtureを再利用し、元v38保存数値と母環境の非干渉を確認。"""
    from tests import test_diagnose_video38_next_enqueue_live_shadow_v1 as fixture
    generator = fixture.frozen.__wrapped__()
    collector = next(generator)
    update = collector.RecognitionPipeline.update
    try:
        before = {k: v for k, v in sys.modules.items() if k == 'src' or k.startswith('src.')}
        request = artificial_request('video_38')
        request['sample_count'] = 32
        packet = P.exchange(request, REGISTRY, REGISTRY_SHA, 'video_38', tmp_path / 'normal')
        expected = json.loads((P.M.VERIFY / 'g2_belief_live_publication_2026-09-11_v1/TRAINED_SIDECAR_v1.json').read_bytes())
        assert packet['evaluation']['probability_p1'] == pytest.approx(
            expected['evaluation']['probability_p1'], abs=1e-8, rel=0)
        assert packet['model_artifact_sha256'] == expected['model_artifact_sha256']
        after = {k: v for k, v in sys.modules.items() if k == 'src' or k.startswith('src.')}
        assert before.keys() == after.keys() and all(after[k] is v for k, v in before.items())
        assert collector.RecognitionPipeline.update is update
    finally:
        generator.close()
