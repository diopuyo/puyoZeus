"""生captureを再検査した候補票だけ保存する。整数公開・本番採用は行わない。"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable
from parent_transport import Client, P
from parent_validation import digest


def publish(client: Client, capture: Callable[[], Any], producer: Callable[[], dict],
            config: Any, observation_request: Callable, joint: Any, path: Path, *,
            count: int = 256, seed: int = 0) -> dict:
    if client.error is not None:
        raise client.error
    try:
        before = capture()
        request = P.decoded(P.encoded(observation_request(before, config, count, seed)))
        snapshot = P.decoded(P.encoded(producer()))
        P.require(snapshot['identity'] == dict(source_id=client.identity[0], run_id=client.identity[1]), 'publish_identity')
        result = client.join(snapshot, request)
        used = 1 if all(len(v.worlds) == 1 for v in before.values) else count
        samples = joint.draw(before.values, used, seed)
        P.require(result['details']['sample_digest'] == digest(samples.tolist()), 'publish_sample_digest')
        after, current = capture(), producer()
        P.require(after.digest == before.digest and after.tokens == before.tokens
                  and all(a is b for a, b in zip(after.values, before.values, strict=True)), 'publish_recapture_changed')
        P.require(P.encoded(observation_request(after, config, count, seed)) == P.encoded(request)
                  and P.encoded(current) == P.encoded(snapshot), 'publish_inputs_changed')
        packet = dict(schema='g2-joint-ledger-parent-candidate/v2', observation=request,
                      producer_sha256=digest(snapshot), result=result,
                      parent_recapture_verified=True, actual_video=False,
                      session_completion_required=str(client.part.with_suffix(client.part.suffix + '.complete.json')),
                      production_permission=False, quality_gate_clear=False)
        raw = P.encoded(packet)
        with path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
        P.require(path.read_bytes() == raw, 'publish_saved_bytes')
        return packet
    except BaseException as error:
        client.fail(error)
        raise
