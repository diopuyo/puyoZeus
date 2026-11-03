"""24fの煙試験で、モデル・高速化の実効件数・終了の組の観測を検収する。"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from statistics import median
from typing import Any

SPEED_MARK = 'RECOG_SPEED_PY '
REQUIRED_COUNTS = ('hsv_only_hit', 'slide_gray_reused', 'display_scores_shared',
                   'image_crops_retained', 'pair_lazy_crops')
BASELINE_RUN = Path('D:/puyo_analyzer/packaging/build_20261025_obs_d/work/smoke_c2/output')


def json_rows(path: Path) -> list[dict[str, Any]]:
    """欠落や破損を0件の成功にしない。"""
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def terminal_rows(path: Path) -> list[dict[str, Any]]:
    """自分の煙試験が閉じたDiskRowsを読む。途中で壊れたレコードは失敗にする。"""
    rows = []
    with path.open('rb') as stream:
        while stream.peek(1):
            rows.append(pickle.load(stream))
    return rows


def update_summary(run: Path) -> dict[str, Any]:
    rows = json_rows(run/'b3_updates.jsonl')
    values = [float(row['inference_ms']) for row in rows if row.get('inference_ms') is not None]
    errors = sum(bool(row.get('error')) for row in rows)
    return dict(updates=len(rows), errors=errors, inference_n=len(values),
                inference_p50_ms=median(values) if values else None,
                passed=bool(rows) and bool(values) and errors == 0)


def recognition_summary(out: Path, arm: str) -> dict[str, Any]:
    reports = [json.loads(line.split(SPEED_MARK, 1)[1])
               for line in (out/'smoke.log').read_text(encoding='utf-8', errors='replace').splitlines()
               if SPEED_MARK in line]
    if arm != 'C2':
        return dict(reports=len(reports), passed=not reports)
    report = reports[-1] if reports else {}
    counts = {key: report.get('stats', {}).get(key, 0) for key in REQUIRED_COUNTS}
    native = report.get('board_stats', {}).get('first_pass_native', 0)
    rows = terminal_rows(out/'output/spool/terminal_observations.pickle')
    endings = {(row['match_end'].get('observed_t_sec'), row['match_end'].get('winner'))
               for row in rows if (row.get('match_end') or {}).get('detected')}
    passed = all(value > 0 for value in counts.values()) and native > 0 and bool(endings)
    return dict(reports=len(reports), counts=counts, native_first_pass=native,
                terminal_rows=len(rows), detected_observations=len(endings), passed=passed)


def check(out: Path, arm: str) -> dict[str, Any]:
    """予測時間の増分は測定として報告し、見込みを合格閾値には使わない。"""
    updates = update_summary(out/'output')
    recognition = recognition_summary(out, arm)
    baseline = update_summary(BASELINE_RUN)
    delta = updates['inference_p50_ms']-baseline['inference_p50_ms']
    result = dict(arm=arm, updates=updates, recognition=recognition, baseline=baseline,
                  inference_p50_delta_ms=delta, passed=updates['passed'] and recognition['passed'])
    (out/'evidence_24f.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result