"""合計処理量と区別し、取得からHTTP公開済みになるまでの実時間を数える。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_live_obs_time_gate import read, stats

DURATION_SEC = 1200
MS = 1000.


def records_path(run: Path) -> Path:
    """本人試験の記録フォルダと、ABBAの外側フォルダの両方を受け付ける。"""
    direct = run/'b3_time_gate.jsonl'
    return direct if direct.exists() else run/'output/b3_time_gate.jsonl'


def publication_delays(run: Path, rows: list[dict] | None = None) -> list[dict[str, Any]]:
    """同じ取得フレームの再公開を重複計上せず、最初の公開完了を使用する。"""
    seen: set[tuple] = set()
    result = []
    for row in rows if rows is not None else read(records_path(run)):
        if row.get('kind') != 'publication' or not 0 <= row['t_sec'] < DURATION_SEC:
            continue
        captured = row.get('captured_at')
        key = (row['frame'], captured)
        if captured is None or key in seen:
            continue
        seen.add(key)
        result.append(dict(frame=row['frame'], t_sec=row['t_sec'], captured_at=captured,
            http_available_at=row['monotonic'], actual_ms=(row['monotonic']-captured)*MS))
    return result


def b3_input_delays(rows: list[dict]) -> list[dict]:
    """最新認識の時刻へ付け替えず、表示したB3計算の元入力から測る。"""
    inputs = {(r['game'], r['frame']): r for r in rows
              if r.get('kind') == 'result' and r.get('accepted')}
    result, seen = [], set()
    for row in rows:
        if row.get('kind') != 'publication':
            continue
        key = (row['game'], row.get('state_frame'))
        source = inputs.get(key)
        if source is None or key in seen or not 0 <= source['t_sec'] < DURATION_SEC:
            continue
        captured = source.get('captured_at')
        if captured is not None:
            seen.add(key)
            result.append(dict(frame=source['frame'], t_sec=source['t_sec'], captured_at=captured,
                http_available_at=row['monotonic'], actual_ms=(row['monotonic']-captured)*MS))
    return result


def summarize(run: Path) -> dict:
    records = read(records_path(run))
    rows = publication_delays(run, records)
    b3 = b3_input_delays(records)
    value = dict(capture_to_http_available_ms=stats([r['actual_ms'] for r in rows]),
        b3_input_to_http_available_ms=stats([r['actual_ms'] for r in b3]), b3_rows=b3,
        definition='state.publish後の監査monotonic−同じフレームのcaptured_at。'
                   '最初の公開だけ。実時間のため低スペック係数を掛けない。',
        rows=rows)
    (run/'actual_latency.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', type=Path)
    for run in parser.parse_args().runs:
        print(run.name, summarize(run)['capture_to_http_available_ms'])
