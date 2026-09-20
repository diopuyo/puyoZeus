"""限定prefixの値・排他保存・内容整合。SHAを署名とは呼ばない。"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
VERIFY, PROJECT = ROOT.parent, ROOT.parents[2]
FULL = VERIFY / 'g2_directional_history_full_runtime_2026-09-09_v1'
NORMAL = VERIFY / 'g2_normal_completion_runtime_2026-09-09_v1'
FIRST, END, STRIDE, FPS, HISTORY_FIRST = 29052, 36300, 2, 60, 34796
SIDES = ('1P', '2P')
FRAMES = tuple(range(FIRST, END, STRIDE))
SOURCE = 'b3728078cc2e8282065e5dd78ca1c13e6a443ffdf1dc4333e3da06f0852757d3'
GUARD = VERIFY / 'g2_bounded_publication_runtime_2026-09-09_v1/resource_guard.py'
PREFLIGHT = VERIFY / 'g2_split_runtime_evidence_adapter_2026-09-08_v1/prepare_preflight.py'
FIXED = {FULL / 'deps_full.py': '5d408357290750417532a9b8936a3949ccd2c0eb654c8b687267ca59bfc4789a',
    FULL / 'assembly_history.py': 'dd6777ed41eb8ad5cc56f60a3906e669cb0891a2fbb839a3f7533359e7972189',
    NORMAL / 'assembly.py': 'd4ee64f99ef8aa62ccd744437aa02e59d14484f949974a4dd8b607e82223a441',
    NORMAL / 'reporting.py': 'fcc242ea04f2f8b65e7007cacc4bd8e77f4a7c6e92039dfea35d806ac025ff7c',
    PREFLIGHT: '88722830ec8f34a07b4f4bc95bce187fff8c59e9287fb9be1fbc42525e908332',
    GUARD: '5c0682ed751d806c435e2c11a5359fffd7f2121b74d9289255107c5cc064914c',
    VERIFY / 'g2_directional_next_candidate_2026-09-09_v1/candidate.py': 'c224066f92f43c6fa095d7cb53e4da578d070500d3b6907169b62a0f0b121667',
    VERIFY / 'g2_next_motion_observables_2026-09-09_v1/observables.py': '31b29d44669cc62727c18dac0bbd235a7382eba25139ed224995edba50a6606c',
    VERIFY / 'g2_directional_next_parent_holdout_2026-09-09_v1/replay_candidate.py': 'fec515b6402bf1314bef5bc52e224aeaf6f723f8bb99ae8f280069ffd4af3d29'}
FIXED[VERIFY / 'g2_directional_raw_input_2026-09-09_v3/bridge.py'] = 'cb4d17b5a9787fde4bb656686b6988999b1cdc2bd4801992408bb4e378a84151'
OLD = VERIFY / 'g2_directional_history_probe_runtime_2026-09-09_v4'
CURRENT = VERIFY / 'g2_history_current_full_runtime_2026-09-09_v2'
PUBLICATION = VERIFY / 'g2_history_publication_full_runtime_2026-09-09_v1'
CONSUMER = VERIFY / 'g2_history_publication_consumer_2026-09-09_v2'
CATALOGS = {OLD: '401185f75a9bbe792fcf1f14ef289b7830a51ac1402e9fd47bb6dc52c8905f85',
    PUBLICATION: 'd327fea72b201660bf0b22be62e86972018e62e629e6681c58bbff9522ef5f6b',
    CONSUMER: 'ae086fa5b28b3d930568fc9f53f24ba690de13d56fc34036f7a546bb21112fdb'}
for dependency, catalog_sha in CATALOGS.items():
    if hashlib.sha256((dependency / 'FREEZE.json').read_bytes()).hexdigest() != catalog_sha:
        raise RuntimeError('probe_fixed_catalog_changed')
    FIXED[dependency / 'FREEZE.json'] = catalog_sha
    record = json.loads((dependency / 'FREEZE.json').read_bytes())['source_sha256']
    FIXED.update({dependency / name: digest for name, digest in record.items()})
FIXED[CURRENT / 'assembly_current.py'] = 'ca20b57bdd248b845337262bfe2004d1a5d713d91950c6342197e4991e3c7038'
FIXED[CURRENT / 'run_cpu.py'] = '21608c0e1023def836865423813faa48ff5301510e6d1560ccf7860e21cbcde4'
FIXED[VERIFY / 'g2_history_current_recovery_2026-09-09_v1/FREEZE.json'] = '1f20e2a370f0b13f689ebf0dd9ea1e218a6b0dc85efbf108c1c2a6f0ae1e236f'
FIXED[VERIFY / 'g2_directional_raw_input_independent_2026-09-09_v1/smoke.py'] = 'b5050aa457faec2b5aae68f88ea15ab4b07ab864ce2f4b27e4fb4c14e0818eca'
FIXED[VERIFY / 'g2_directional_raw_input_2026-09-09_v1/test_bridge.py'] = '24721f6bbdbf60239aaa7422d29f1c9aabd11140bad752b89302df0b64d86b18'

ADOPTION = VERIFY / 'g2_history_baseline_adoption_2026-09-10_v4'
FIXED[ADOPTION / 'witness.py'] = '703c4b24450d8299a1cb934c8ec06b710fbc206e4b634f5164643e309f34f3b4'
FIXED[ADOPTION / 'adoption.py'] = '9ec086d887433c23a783e36d91306edbdc43d5e53a8539520208c765e17eddd1'

MISSING = VERIFY / 'g2_history_missing_observation_2026-09-10_v1'
FIXED[MISSING / 'connection.py'] = 'cb5f501efafc91c1ebc47accea739b7d47226ad10487a1c0db2a2a2c7dcc46a0'

OWN = ('missing_connection.py', 'test_missing_connection.py', 'test_continuation_scope.py', 'initial_hand_connection.py', 'test_initial_hand_connection.py', 'ast_connection.py', 'test_ast_connection.py', 'adoption_connection.py', 'common.py', 'session.py', 'recording.py', 'closure.py', 'live_cli.py',
       'test_probe.py', 'run_cpu.py', 'launcher.sh', 'PLAN.md', 'cpu_smoke.py', 'assembly_publication_probe.py', 'goals.py')


REPEATED_MANIFEST = ROOT/'REPEATED_DEPENDENCIES.json'
REPEATED_SHA = '41074e61ef7f6c8ac20343994fc30190ed5f7ada90ebb6f96cc6a27f58144424'
if hashlib.sha256(REPEATED_MANIFEST.read_bytes()).hexdigest() != REPEATED_SHA:
    raise RuntimeError('repeated_dependency_manifest_changed')
FIXED[REPEATED_MANIFEST] = REPEATED_SHA
FIXED.update({Path(path): digest for path, digest in json.loads(REPEATED_MANIFEST.read_bytes())['guards'].items()})
OWN += ('repeated_connection.py', 'REPEATED_DEPENDENCIES.json', 'derivation_check.py', 'test_scope_connection.py',
    'finalizer_connection.py', 'test_finalizer_connection.py', 'saved_finalizer_cpu.py')


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def guards() -> dict[str, str]:
    require(all(sha(p) == h for p, h in FIXED.items()), 'probe_dependency_changed')
    return {str(p): h for p, h in FIXED.items()} | {str(ROOT / n): sha(ROOT / n) for n in OWN} | {str(GUARD): sha(GUARD)}


def load(alias: str, path: Path, stack: Any) -> Any:
    require(alias not in sys.modules, 'probe_alias_collision:' + alias)
    spec = importlib.util.spec_from_file_location(alias, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[alias] = value
    stack.callback(sys.modules.pop, alias, None)
    spec.loader.exec_module(value)
    return value


def bounds() -> dict[str, Any]:
    return dict(first_frame=FIRST, end_exclusive=END, last_frame=FRAMES[-1], stride=STRIDE,
        expected_updates=len(FRAMES), expected_side_steps=len(FRAMES) * len(SIDES), fps=FPS,
        start_sec=FIRST / FPS, max_sec=(END - FIRST) / FPS,
        original_end_sec=605.0, old_reference_commit_frames=[34898, 34940], forced_commit_clock=False)


def actual_kwargs(original: dict[str, Any]) -> dict[str, Any]:
    require(original.get('enable_ojama_write_accounting_guard') is True, 'raw_capture_constructor_flag')
    require(original.get('normalize_fps_30') is True and original.get('sample_interval_sec') == 0,
            'original_stride_flags')
    result = dict(original, start_sec=FIRST / FPS, max_sec=(END - FIRST) / FPS, precise_seek=False)
    require(int(result['start_sec'] * FPS) == FIRST and
        FIRST + int(result['max_sec'] * FPS) == END, 'collector_integer_bounds')
    return result
