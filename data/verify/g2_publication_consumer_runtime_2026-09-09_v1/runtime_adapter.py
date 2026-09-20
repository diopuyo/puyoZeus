"""旧factoryを保ったまま実公開返値と私有collectorを接続する。"""
from __future__ import annotations
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
DEATH = ROOT.parent / 'g2_death_availability_boundary_2026-09-09_v1'
STATE_KEY = 'private_publication_consumer'
ENTRIES, COMPARISON = 'consumer_publication.jsonl', 'PRIVATE_CONSUMER_COMPARISON.json'
STATUS, RECEIPT = 'PRIVATE_CONSUMER_STATUS.json', 'PRIVATE_CONSUMER_RECEIPT.json'
REQUIRED = frozenset((ENTRIES, COMPARISON, STATUS, RECEIPT))
FIXED = {
    ROOT / 'connected.py': '65f664798b8f910b2deaa59d4c67145b694c06dbeabedb78c8ef208233f26cbe',
    ROOT.parent / 'g2_current_consumer_release_2026-09-09_v1/release.py':
        '083693a073b5765b52ebb779a8b5522bcd39ea0ffe4aaac6d92c827b2903301e',
    ROOT.parent / 'g2_current_consumer_release_2026-09-09_v1/consumer_comparison.py':
        '232a69c933761200358f2bf429fb01a7a465bc07910b5f0620b0d9260e5e7e61',
    ROOT.parent / 'g2_publication_consistency_2026-09-09_v1/gate.py':
        '567cf26dc8a709625c0e4b5a6f6b288e675a67ddeaec5cd73f5670c7e2df4420',
    DEATH / 'adapter.py': 'cb15008d8212f97956bd01b2252485ec31ed8a609c4c427d56f359c87ffaf161',
    DEATH / 'isolated_runtime.py': '59df6300af2c4475dae30e12980f1bcd8c7ad33877f0ba68d7f523112f3b44ce',
    PROJECT / 'src/death_confirmation.py': '7937cacf841caa305901383b9686310eeb30a24f1ec4150e1584ea0d4f0cb88a',
    PROJECT / 'src/event_death_observer_v1.py': 'd0869120237e05c20614312295065201e0ea6161c1b4daa58e673dc1848b4fda',
    PROJECT / 'scripts/collect_boards_lean.py': '097e8c0920679b2f3ebe29ffb103ce87d5c9c6b370d0eec986972ad506d8b50c',
}
MODEL_SOURCES = {
    'projected_set_residual_cnn_v1': '4cabe8eed85198901cca812d4f6f796914c9ab28d35c99ba05201acfd74c84b0',
    'advantage_m0_current_cnn_v1': '84af708bfc348570b651834aa2e3f51f96e83551ccd11285c9f3b530347c1554',
    'canonical_observation_v2': '2aac290fe94c171b4c0acdfee462a06f4a22fdd2ff6ec93788df31025e887209',
    'canonical_observation_v3': '0cb5779612ee3daa1389da931dfbf004767681d4835ecb99f859622a5091dd44',
    'advantage_m1_causal_ledger_v3': '6fec4a6b8ff68cd5a65b69b8f10f4d5b974f5c64d7f33b6a2f3fb579d015e5ce',
}
FIXED.update({PROJECT / 'src' / (name + '.py'): value for name, value in MODEL_SOURCES.items()})


def require(value: bool, reason: str) -> None:
    if not value:
        raise RuntimeError(reason)


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path: Path, value: Any) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def guards() -> dict[str, str]:
    require(all(sha(path) == value for path, value in FIXED.items()), 'consumer_fixed_source_changed')
    paths = [ROOT / name for name in ('runtime_adapter.py', 'live_cli.py', 'preflight.py', 'launcher.sh',
                                    'test_connected.py', 'run_cpu.py')]
    paths += [ROOT.parent / 'g2_provisional_context_capture_2026-09-09_v1' / name
              for name in ('saved_context.py', 'binding.py')]
    return {str(path): value for path, value in FIXED.items()} | {str(p): sha(p) for p in paths}


def load(name: str, path: Path, *, injection: dict[str, Any] | None = None,
         restore_path: bool = True) -> Any:
    require(name not in sys.modules, 'consumer_private_module_collision')
    original_import, replacements = builtins.__import__, injection or {}
    def importing(key: str, globals: Any = None, locals: Any = None,
                  fromlist: Any = (), level: int = 0) -> Any:
        if level == 0 and key in replacements:
            return replacements[key]
        return original_import(key, globals, locals, fromlist, level)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'consumer_module_spec')
    module = importlib.util.module_from_spec(spec)
    module.__dict__['__builtins__'] = dict(vars(builtins), __import__=importing)
    previous = list(sys.path)
    sys.modules[name] = module
    try:
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
    finally:
        if restore_path:
            sys.path[:] = previous
    return module


def registration(recorder: Any) -> dict[str, Any]:
    plan = json.loads((recorder.output / 'PLAN.json').read_text())
    source = plan['original_source']
    source_sha = source['source_video_sha256']
    require(plan['input_and_code_sha256'][plan['video_path']] == source_sha, 'consumer_source_sha')
    require(recorder.source_id == 'sha256:' + source_sha
            and recorder.run_id == recorder.output.resolve().as_posix(), 'consumer_live_registration')
    return {'source_id': recorder.source_id, 'run_id': recorder.run_id, 'source_sha256': source_sha,
        'time_base_numerator': source['time_base_numerator'],
        'time_base_denominator': source['time_base_denominator'], 'ledger_connection': 'NOT_CONNECTED',
        'plan_sha256': sha(recorder.output / 'PLAN.json'), 'registration_kind': 'actual_live_plan_not_complete'}


def components() -> Any:
    """本物の型/関数を私有別名で読込。src正本の名前空間は広げない。"""
    guards()
    mapped = {}
    for stem in MODEL_SOURCES:
        mapped['src.' + stem] = load('_consumer_contract_' + stem,
            PROJECT / 'src' / (stem + '.py'), injection=mapped)
    paths = {
        'binding': ROOT.parent / 'g2_provisional_context_capture_2026-09-09_v1/binding.py',
        'saved_context': ROOT.parent / 'g2_provisional_context_capture_2026-09-09_v1/saved_context.py',
        'release': ROOT.parent / 'g2_current_consumer_release_2026-09-09_v1/release.py',
        'gate': ROOT.parent / 'g2_publication_consistency_2026-09-09_v1/gate.py',
        'consumer_comparison': ROOT.parent / 'g2_current_consumer_release_2026-09-09_v1/consumer_comparison.py',
        'connected': ROOT / 'connected.py',
    }
    for key, path in paths.items():
        mapped[key] = load('_consumer_contract_' + key, path, injection=mapped)
    return mapped['connected']


class Sink:
    def __init__(self, publisher: Any, recorder: Any, before: dict[str, str]) -> None:
        self.publisher, self.recorder, self.before = publisher, recorder, before
        self.closed, self.failures = False, []

    def wrapper(self, original: Any) -> Any:
        connected = self.publisher.wrap(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return connected(*args, **kwargs)
            except BaseException as error:
                self.failures.append({'type': type(error).__name__, 'message': str(error)})
                raise
        return wrapped

    def close(self) -> None:
        output = self.recorder.output
        with (output / ENTRIES).open('x') as stream:
            for entry in self.publisher.entries:
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
        if self.publisher.entries and not self.failures:
            write(output / COMPARISON, self.publisher.comparison.snapshot())
        self.closed = True
        write(output / STATUS, {'closed': True, 'failures': self.failures,
            'updates': len(self.publisher.entries), 'publication_changed': True,
            'private_collector_helpers_not_training_data': True})


def install(stack: Any, collector: Any, history: Any, state: dict[str, Any]) -> None:
    before = guards()
    require(STATE_KEY not in state, 'consumer_install_reentry')
    rec = state['provisional_context_observer']
    require(not rec.rows and not rec.errors, 'consumer_must_install_before_updates')
    connected = components()
    death = load('_consumer_death_runtime', DEATH / 'adapter.py')
    isolated = load('_consumer_isolated_runtime', DEATH / 'isolated_runtime.py', injection={'adapter': death})
    observer = stack.enter_context(isolated.loaded_observer())
    private = stack.enter_context(isolated.loaded_source('scripts/collect_boards_lean.py',
        injections={'src.event_death_observer_v1': observer}))
    pipeline = sys.modules['src.recognition_pipeline']
    require(private.RecognitionPipeline is collector.RecognitionPipeline is pipeline.RecognitionPipeline,
            'consumer_actual_pipeline_identity')
    bound = registration(rec)
    comparison = connected.AvailableComparison(private, bound, death, observer, pipeline)
    publisher = connected.Publisher(rec, bound, comparison)
    sink = Sink(publisher, rec, before)
    old = collector.RecognitionPipeline.update
    stack.callback(sink.close)
    stack.callback(setattr, collector.RecognitionPipeline, 'update', old)
    collector.RecognitionPipeline.update = sink.wrapper(old)
    state[STATE_KEY] = sink


def finish(state: dict[str, Any]) -> None:
    sink = state[STATE_KEY]
    require(sink.closed and not sink.failures and not sink.publisher.busy, 'consumer_failed_or_unclosed')
    frames = [row['frame_idx'] for row in sink.publisher.entries]
    require(frames == sink.recorder.expected, 'consumer_update_coverage')
    require(sink.before == guards(), 'consumer_source_changed_during_run')
    comparison = json.loads((sink.recorder.output / COMPARISON).read_text())
    require(comparison['all_consumers_ready'] is True, 'consumer_not_ready')
    write(sink.recorder.output / RECEIPT, {'expected_frames': frames, 'guards': sink.before,
        'sha256': {name: sha(sink.recorder.output / name) for name in (ENTRIES, COMPARISON, STATUS)},
        'source_id': sink.recorder.source_id, 'run_id': sink.recorder.run_id,
        'actual_pipeline_executed': True, 'current_collector_helpers_used': True,
        'training_permission': False, 'accounting_permission': False, 'quality_gate_clear': False})


def verify(output: Path) -> None:
    receipt = json.loads((output / RECEIPT).read_text())
    status = json.loads((output / STATUS).read_text())
    require(status['closed'] is True and not status['failures'], 'consumer_saved_failure')
    require(receipt['guards'] == guards(), 'consumer_saved_guards')
    require(set(receipt['sha256']) == {ENTRIES, COMPARISON, STATUS}, 'consumer_artifact_set')
    require(all(sha(output / name) == value for name, value in receipt['sha256'].items()), 'consumer_artifact_changed')
    with (output / ENTRIES).open() as stream:
        frames = [json.loads(line)['frame_idx'] for line in stream]
    require(frames == receipt['expected_frames'] and len(frames) == status['updates'], 'consumer_saved_coverage')
    require(all(receipt[key] is False for key in ('training_permission', 'accounting_permission', 'quality_gate_clear')),
            'consumer_private_permissions')
