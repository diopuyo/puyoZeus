"""観測窓だけ拡張する。新Publisherは接続せず、旧採録の全byte一致を要求。"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
WINDOWS = ((29052, 36298),)
OLD_WINDOWS = ((32640, 32800), (34700, 36000))
REFERENCE = ROOT.parent / 'video38_provisional_context_live_2026-09-09_v1'
FRAME_SHA = '60821115eec48c8b5769b1a6848b8a0e805f2e176ab0b9450bcd37caa8971b9e'
TARGETS = {
    '_hsv_live_observer': (ROOT.parent / 'g2_hsv_correction_witness_2026-09-08_v1/observer.py',
        '7b3407ccbc46bd5ad48fdcc1fc15810ab3dce2c39866ab42b0d5da3d1084642d'),
    '_hidden_capture_live_observer': (ROOT.parent / 'g2_hidden_probability_capture_2026-09-08_v1/observer.py',
        'cea8f92da47c19210a3c9a36dc877bfddd07d102f57575caecc766c8ee2692e9'),
}
RECEIPT = 'FULL_OBSERVATION_SCOPE_RECEIPT.json'
OWN = ('scope.py', 'live_cli.py', 'preflight.py', 'launcher.sh', 'test_scope.py', 'run_cpu.py', 'CONTRACT.md')


def require(value: bool, reason: str) -> None:
    if not value:
        raise RuntimeError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def guards() -> dict[str, str]:
    fixed = {str(path): value for path, value in TARGETS.values()}
    fixed[str(REFERENCE / 'frames.jsonl')] = FRAME_SHA
    require(all(sha(Path(p)) == h for p, h in fixed.items()), 'scope_fixed_input_changed')
    return fixed | {str(ROOT / name): sha(ROOT / name) for name in OWN}


@contextmanager
def configured() -> Iterator[None]:
    """実bootstrap済み2観測moduleのみ変更し、例外でも元の窓へ復元。"""
    modules = []
    for alias, (path, expected_sha) in TARGETS.items():
        module = sys.modules.get(alias)
        require(module is not None and Path(module.__file__).resolve() == path
            and sha(path) == expected_sha, 'scope_actual_observer_required')
        require(module.WINDOWS == OLD_WINDOWS, 'scope_reentry_or_foreign_windows')
        modules.append(module)
    try:
        for module in modules:
            module.WINDOWS = WINDOWS
        yield
    finally:
        for module in modules:
            module.WINDOWS = OLD_WINDOWS


def checks(output: Path) -> dict[str, Any]:
    """窓の実採録と認識・旧collector全列の非干渉を別々に検査。"""
    require(sha(REFERENCE / 'frames.jsonl') == FRAME_SHA, 'scope_reference_changed')
    require(sha(output / 'frames.jsonl') == FRAME_SHA, 'scope_changed_legacy_frames')
    frames = list(range(WINDOWS[0][0], WINDOWS[0][1] + 1, 2))
    expected = [[f, s] for f in frames for s in ('1P', '2P')]
    for name in ('HIDDEN_PROBABILITY_RECEIPT.json', 'CURRENT_SCOPE_RECEIPT.json',
                 'PROVISIONAL_CURRENT_RECEIPT.json'):
        value = json.loads((output / name).read_text())
        require(value['expected_scopes'] == expected, 'scope_saved_coverage:' + name)
    context = json.loads((output / 'PROVISIONAL_CONTEXT_RECEIPT.json').read_text())
    require(context['expected_frames'] == frames, 'scope_outer_coverage')
    return {'windows': WINDOWS, 'updates': len(frames), 'both_side_scopes': len(expected),
        'legacy_frames_sha256': FRAME_SHA, 'legacy_frames_byte_identical': True,
        'publisher_connected': False, 'extra_candidates_are_private_only': True,
        'quality_gate_clear': False, 'training_permission': False, 'accounting_permission': False}


def finish(state: dict[str, Any]) -> None:
    output = state['output']
    write(output / RECEIPT, checks(output) | {'guards': guards()})


def verify(output: Path) -> None:
    value = json.loads((output / RECEIPT).read_text())
    expected = json.loads(json.dumps(checks(output)))
    require(value == expected | {'guards': guards()}, 'scope_receipt_changed')
