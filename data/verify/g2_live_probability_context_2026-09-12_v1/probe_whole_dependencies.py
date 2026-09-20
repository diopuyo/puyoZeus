"""coldな原snapshot importと修復observer選択を確認。モデル生成/動画更新はしない。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import whole_collector_capture as W

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT.parents[2] / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'


def main() -> None:
    assert not any(name == 'src' or name.startswith('src.') for name in sys.modules)
    sys.path.insert(0, str(SNAPSHOT))
    path = SNAPSHOT / 'scripts/collect_boards_lean.py'
    spec = importlib.util.spec_from_file_location('_whole_actual_collector', path)
    collector = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = collector
    spec.loader.exec_module(collector)
    original = collector.EventAccountingRecorder
    assert W.authentic_code(collector) is collector.collect_lean.__code__
    sys.path.insert(0, str(ROOT.parent / 'g2_frozen_accounting_upgrade_2026-09-12_v1'))
    import upgrade
    modern = upgrade.modern()
    recorder = modern()
    assert type(recorder) is modern and modern is not original
    assert recorder._observed_frame_count == 0 and 'observe' not in vars(recorder)
    assert collector.EventAccountingRecorder is original
    report = dict(original_collect_code_authenticated=True, repaired_observer_selected=True,
                  observed_frames=0, original_factory_unchanged=True, model_generations=0,
                  actual_bridge_install=False, quality_gate_clear=False)
    with (ROOT / 'WHOLE_DEPENDENCIES_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
