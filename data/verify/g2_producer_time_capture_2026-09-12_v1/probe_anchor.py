"""実保存会計を元consumerへ渡し、途中開始の既知ゼロ誤資格リスクを測る。"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
from src.event_accounting_adapter_v1 import parse_event_accounting_sidecar_bytes
from src.event_snapshot_adapter_v1 import StableBoardSnapshot, VideoTimeBase, build_stable_board_batches
from src.event_observation_adapter_v1 import merge_and_resequence_batches
from src import advantage_m1_zero_counterfactual_v3 as MODEL
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'g2_committed_ledger_bridge_2026-09-12_v1'))
import committed as K
sys.path.insert(0, str(ROOT.parent / 'g2_accounting_prefix_adapter_2026-09-11_v1'))
import continuation as C
IDENTITY = ('video_38', 'capture-diagnostic', 'attempt-v1')


def plain(value: Any) -> Any:
    if isinstance(value, dict) and 'sequence_type' in value:
        return [plain(v) for v in value['items']]
    if isinstance(value, dict) and 'ndarray_dtype' in value:
        return plain(value['values'])
    return value


def inspect(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    frame = value['last_frame']
    sidecar = parse_event_accounting_sidecar_bytes(json.dumps(value['accounting']).encode())
    events = C.build_event_accounting_events(sidecar, source_video_id=IDENTITY[0], build_id=IDENTITY[1],
        attempt_id=IDENTITY[2], time_base=VideoTimeBase(1, 60), finalize_processing=False)
    snapshots = tuple(StableBoardSnapshot(IDENTITY[0], frame, row['side'],
        tuple(tuple(r) for r in plain(row['arguments']['board'])), 'observed') for row in value['metadata'])
    boards = build_stable_board_batches(snapshots, build_id=IDENTITY[1], attempt_id=IDENTITY[2], time_base=VideoTimeBase(1, 60))
    batches = merge_and_resequence_batches(boards, events, build_id=IDENTITY[1], attempt_id=IDENTITY[2])
    timing = batches[-1][0]['timing']
    cutoff = K.V3.CanonicalCutoff(timing['available_frame'], timing['available_ms'])
    committed = K.validated(batches, IDENTITY, cutoff)
    observation = K.V3.canonical_observation_from_committed_prefix(committed, cutoff)
    tensor = K.M.advantage_m1_inputs_from_canonical(observation)
    supported = bool(MODEL._ledger_rows_supported(torch.tensor(tensor.ledger_availability[None])).item())
    return {'source': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'first_frame': value['first_frame'],
        'last_frame': frame, 'caller_anchor_qualified': value['game_anchor_qualified'],
        'canonical_masks_supported': supported, 'ledger_values': tensor.ledger_values.tolist(),
        'ledger_availability': tensor.ledger_availability.tolist(), 'actual_video': False,
        'quality_gate_clear': False, 'note': '診断は会計と現盤面のみ。境界票/開始証拠/Jはまだ結合しない'}


def main() -> None:
    values = [inspect(ROOT / folder / 'CAPTURE.json') for folder in ('actual_34772_v1', 'full_boundary_v2')]
    with (ROOT / 'ANCHOR_PROBE_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(values, stream, ensure_ascii=False, indent=2)
    print(json.dumps([{k: v[k] for k in ('first_frame', 'caller_anchor_qualified', 'canonical_masks_supported')} for v in values]))


if __name__ == '__main__':
    main()
