"""原凍結collectorの実signature/ASTで引数導出だけ確認。モデル生成はしない。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import constructor_observation as C

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
SNAPSHOT = PROJECT / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'


def main() -> None:
    assert not any(name == 'src' or name.startswith('src.') for name in sys.modules)
    sys.path.insert(0, str(SNAPSHOT))
    source = SNAPSHOT / 'scripts/collect_boards_lean.py'
    spec = importlib.util.spec_from_file_location('_actual_constructor_kwargs_collector', source)
    collector = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = collector
    spec.loader.exec_module(collector)
    prior = ROOT.parent / 'video38_history_publication_probe_live_2026-09-11_v12/PLAN.json'
    receipt = json.loads(prior.read_bytes())
    kwargs = receipt['actual_collector_kwargs']
    # 旧JSONはPathを文字列化している。実collection_argumentsのPath型へ戻す。
    kwargs['score_region_calibration_path'] = Path(kwargs['score_region_calibration_path'])
    value = C.expected_kwargs(collector, Path(receipt['video_path']), ROOT / 'unused', kwargs)
    assert value['enable_ojama_write_accounting_guard'] is True
    assert value['stable_frame_count'] == 3 and value['score_region_offsets'] is not None
    assert value['stable_majority_window'] == kwargs['enable_stable_majority_window']
    result = dict(actual_collector_source=str(source), prior_config_input=str(prior),
                  keyword_count=len(value), keys=sorted(value), alias_mapping_verified=True,
                  calibration_loaded=True, factory_calls=0, updates=0, quality_gate_clear=False)
    with (ROOT / 'CONSTRUCTOR_KWARGS_v1.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
