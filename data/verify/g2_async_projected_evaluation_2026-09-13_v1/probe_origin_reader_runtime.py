"""元A23 configured環境の実Recorder型を照合する。動画/モデル実走はしない。"""
from contextlib import ExitStack
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'g2_split_tail_runtime_2026-09-13_v23'
REQUIRED = ('ledger', 'ledger_module', 'active_handles', 'generation_recorder',
            '_clock_active', '_current_generation')


def main() -> None:
    sys.path.insert(0, str(RUNTIME))
    import owned_adapter as A
    with ExitStack() as stack:
        selected = A.configured(stack)
        session = selected.__globals__['S']
        with session.configured() as env:
            history = env['runtime'].M.A.entry.previous.history
            recorder = history.HistoryRecorder(io.StringIO(), {})
            missing = [name for name in REQUIRED if not hasattr(recorder, name)]
            result = dict(recorder_class=f'{type(recorder).__module__}.{type(recorder).__name__}',
                mro=[f'{cls.__module__}.{cls.__name__}' for cls in type(recorder).__mro__],
                direct_reader_missing_fields=missing, history_module_file=history.__file__,
                video_updates=0, model_evaluated=False, live_hook_verified=False,
                quality_gate_clear=False)
    result['configured_context_exited'] = True
    with (ROOT / 'ORIGIN_READER_RUNTIME_v1.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
