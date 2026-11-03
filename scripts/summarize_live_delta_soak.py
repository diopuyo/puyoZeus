"""55分の実測を既存と同じ評価終了時刻で集計する。品質一致には使わない。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import json
from pathlib import Path
import pickle
import sys
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_live_speed3 as existing
from scripts.bench_live_delta import check_resources

OUT = Path(_public_dev_text('<work>/delta'))
TRUNCATED: list[str] = []
MS = 1000.0


def complete_rows(path: Path) -> Iterator[dict]:
    """停止時の最後の未完了レコードだけを除外し、欠落を記録する。"""
    if path.suffix == '.pickle':
        with path.open('rb') as stream:
            while True:
                start = stream.tell()
                try:
                    yield pickle.load(stream)
                except EOFError:
                    if start != path.stat().st_size:
                        TRUNCATED.append(str(path))
                    return
                except pickle.UnpicklingError:
                    if stream.read(1):
                        raise  # 途中の破損を末尾欠損として隠さない。
                    TRUNCATED.append(str(path))
                    return
    else:
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.endswith('\n'):
                    TRUNCATED.append(str(path))
                    return
                yield json.loads(line)


def recognition(root: Path) -> dict:
    """認識を行ったコマと受理コマの母数を分け、実時間fpsを併記する。"""
    all_rows = complete_rows(root / 'output/spool/frame_timing_recognition.jsonl')
    reads = [r for r in all_rows if r['stages']['board_recognition']['calls'] > 0]
    accepted = [r for r in reads if not r.get('held', False)]
    span = accepted[-1]['captured_at'] - accepted[0]['captured_at'] if len(accepted) > 1 else 0.0
    return dict(read_frames=len(reads), accepted_frames=len(accepted), accepted_span_sec=span,
        accepted_fps=(len(accepted) - 1) / span if span else None,
        recognition_wall_ms=existing.stats([r['wall_ms'] for r in reads]),
        board_recognition_ms=existing.stats([r['stages']['board_recognition']['wall_ms'] for r in reads]),
        accepted_recognition_wall_ms=existing.stats([r['wall_ms'] for r in accepted]))


def environment(root: Path) -> dict:
    samples = list(complete_rows(root / 'samples.jsonl'))
    return dict(samples=len(samples),
        min_free_gib=min(r['free_kb'] / 1024**2 for r in samples if 'free_kb' in r),
        system_cpu_percent=existing.stats([r['system_cpu_percent'] for r in samples]),
        stop=json.loads((root / 'soak_status.json').read_text()),
        feed=json.loads((root / 'feed_status.json').read_text()))


def main() -> None:
    existing.ROOT, existing.rows = OUT, complete_rows
    result = dict(status='diagnostic_not_quality_pass', runs={})
    for mode in ('off', 'on'):
        check_resources()
        root = OUT / ('run_' + mode)
        assert json.loads((root / 'soak_status.json').read_text())['state'] == 'finished'
        result['runs'][mode] = dict(recognition=recognition(root),
            evaluation=existing.analyze(mode), environment=environment(root))
    result['truncated_tails'] = TRUNCATED
    result['note'] = '同一元動画だが受理画素列は不一致。表示勝率の品質比較へ代用しない。'
    (OUT / 'soak_comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for mode, run in result['runs'].items():
        print(mode, json.dumps(dict(recognition=run['recognition'],
            capture_to_evaluation_ms=run['evaluation']['all']['latency']), ensure_ascii=False))


if __name__ == '__main__':
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    main()
