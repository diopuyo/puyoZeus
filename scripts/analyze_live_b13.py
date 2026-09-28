"""B12保存記録の捨て内訳と、B8方式の全区間認識差分を再集計する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_live_b8 import board_comparison, distribution, load_audit, summarize_recognition
from scripts.analyze_live_b12 import compare_physics, normalized_slots
from scripts.measure_live_b6 import save
from scripts.measure_live_b12 import START, LONG_END

MILLISECONDS = 1000.0
WINDOW_SECONDS = 300.0
TIME_TOLERANCE = 1e-6
B12_BUILD = 'sha256:088721d643f5594112bc7a4c742042a1d26e88ea178dee624cf15ee83bf8ea95'


def drop_rows(data: dict, dropped: list[float], bounds: dict) -> list[dict]:
    """記録されていない選択時刻を補完せず、直前の認識完了から遅れの下限を求める。"""
    origin = float(np.median(data['captured_at']-data['t_sec']))
    period = bounds['stride']/bounds['fps']
    rows = []
    for stamp in dropped:
        previous = int(np.searchsorted(data['t_sec'], stamp))-1
        deadline = origin+stamp+period
        recognized = float(data['recognized_at'][previous]) if previous >= 0 else None
        queued = (recognized+float(data['queue_put_ms'][previous])/MILLISECONDS
                  if recognized is not None else None)
        evidence = ('recognition_already_late' if recognized is not None and recognized >= deadline else
                    'queue_wait_sufficient' if queued is not None and queued >= deadline else
                    'unobserved_source_or_other_work')
        rows.append(dict(t_sec=stamp, prior_recognized_at=recognized, next_slot_deadline=deadline,
                         evidence=evidence, category='deadline_catch_up'))
    return rows


def drop_window(data: dict, slots: list[float], rows: list[dict], start: float, end: float) -> dict:
    selected = [r for r in rows if start <= r['t_sec'] < end]
    expected = sum(start <= t < end for t in slots)
    mask = (data['t_sec'] >= start) & (data['t_sec'] < end)
    return dict(start_sec=start, end_sec=end, expected=expected, recognized=int(mask.sum()),
        dropped=len(selected), drop_fraction=len(selected)/expected if expected else None,
        event_policy_extra_skip=0, deadline_catch_up=len(selected),
        evidence={key: sum(r['evidence'] == key for r in selected) for key in
                  ('recognition_already_late', 'queue_wait_sufficient', 'unobserved_source_or_other_work')},
        recognition_ms=distribution(data['recognition_ms'][mask].tolist()),
        capture_to_acquire_ms=distribution(((data['acquired_at']-data['captured_at'])[mask]*MILLISECONDS).tolist()))


def breakdown(path: Path, start: float = START, end: float = LONG_END) -> dict:
    metrics = json.loads((path/'metrics.json').read_text())
    validate_policy(metrics)
    bounds = metrics['frame_bounds']
    if abs(end-bounds['end']/bounds['fps']) > TIME_TOLERANCE:
        raise ValueError('入力終端と集計終端を合わせてください')
    data = load_audit(path/'recognition.npz', bounds['start']/bounds['fps'], end)
    dropped = json.loads((path/'recognition.json').read_text())['dropped_times']
    slots = normalized_slots(bounds, start)
    dropped = [t for t in dropped if start <= t < end]
    rows = drop_rows(data, dropped, bounds)
    total = drop_window(data, slots, rows, start, end)
    if len(slots) != metrics['expected_frames'] or len(dropped) != metrics['capture_dropped_in_measured_window']:
        raise ValueError('計測器の母数・捨て数と一致しません')
    actual = [float(t) for t in data['t_sec'] if start <= t < end]
    expected_ids = {round(t*bounds['fps']) for t in slots}
    observed_ids = [round(t*bounds['fps']) for t in actual+dropped]
    if len(observed_ids) != len(expected_ids) or set(observed_ids) != expected_ids:
        raise ValueError('認識済みと捨て記録で全対象frameを一意に被覆できません')
    windows = [drop_window(data, slots, rows, float(lo), min(float(lo)+WINDOW_SECONDS, end))
               for lo in np.arange(start, end, WINDOW_SECONDS)]
    raw_frames = bounds['end']-int(np.ceil(start*bounds['fps']))
    return dict(total=total, windows=windows, rows=rows, assets=metrics['assets'],
        normalization_excluded=raw_frames-len(slots), raw_input_frames=raw_frames,
        classification_basis='4c4b365: select(next,due) <= due。イベント優先による追加skipは0。全dropは時刻追従経路。',
        limitation='選択時刻と反実仮想c OFFの記録はない。CPU不足・I/O等の厳密な因果配分はできない。')


def validate_policy(metrics: dict) -> None:
    """未知の版や入力へ、確認済みコードの分類根拠を誤って適用しない。"""
    if (metrics.get('assets', {}).get('app_build_id') != B12_BUILD or
            metrics.get('source') != 'video' or not metrics.get('realtime')):
        raise ValueError('この内訳分類は4c4b365の実時間動画記録専用です')


def impact(reference: Path, candidate: Path, start: float = START, end: float = LONG_END) -> dict:
    metrics = [json.loads((p/'metrics.json').read_text()) for p in (reference, candidate)]
    if metrics[0]['assets'] != metrics[1]['assets']:
        raise ValueError('参照と対象のコード・モデル・認識設定・較正が一致しません')
    configs = [json.loads((p/'recognition_config.json').read_text()) for p in (reference, candidate)]
    if configs[0] != configs[1]:
        raise ValueError('認識器へ渡した実行パラメータが一致しません')
    if metrics[0]['realtime'] or metrics[0]['capture_dropped_in_measured_window']:
        raise ValueError('参照は間引きなしのオフライン再生が必要です')
    left, right = (load_audit(p/'recognition.npz', start, end) for p in (reference, candidate))
    expected = normalized_slots(metrics[1]['frame_bounds'], start)
    if not np.array_equal(left['t_sec'], expected):
        raise ValueError('参照に対象区間の全frameが揃っていません')
    return dict(start_sec=start, end_sec=end, boards=board_comparison(left, right),
                physics=compare_physics(reference, candidate, start, end),
                recognition=summarize_recognition(candidate, start, end),
                method='B8: 同時刻±2秒の盤面対応、純増2色ぷよ。未対応は見落し/未確定を分離。')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=Path('logs/live_b12/long'))
    parser.add_argument('--reference', type=Path, default=Path('logs/live_b13/reference'))
    parser.add_argument('--output', type=Path, default=Path('logs/live_b13'))
    parser.add_argument('--drops-only', action='store_true')
    options = parser.parse_args()
    result = breakdown(options.candidate)
    save(options.output/'drop_breakdown.json', result)
    print(json.dumps(result['total'], ensure_ascii=False), flush=True)
    if not options.drops_only:
        result = impact(options.reference, options.candidate)
        save(options.output/'impact.json', result)
        print(json.dumps(dict(placements=result['boards']['placements'],
                             physics={key: {k: v for k, v in row.items() if k != 'rows'}
                                      for key, row in result['physics'].items()}), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
