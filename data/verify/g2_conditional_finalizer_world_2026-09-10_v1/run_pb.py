"""共有86の元保存18候補で一般PB結合を検査する。動画は再走しない。"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import world_api as A

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'g2_conditional_shared_runtime_candidate_2026-09-10_v1/prefix_cpu_v2'
NAMES = ('directional_history.jsonl', 'atomic_journal.jsonl', 'COMBINED_HIDDEN.json', 'POSTCOMMIT_CONSUMER_ROWS.json')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines()] if path.suffix == '.jsonl' else json.loads(text)


def write(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def inputs() -> dict[str, Any]:
    history, journal, hidden, outer = (read(SOURCE / name) for name in NAMES)
    return dict(history_rows=history, journal_rows=journal, hidden_events=hidden['events'], outer_rows=outer)


def main() -> int:
    output = ROOT / sys.argv[1]
    assert output.parent == ROOT and output.name.startswith('pb_cpu_v')
    output.mkdir(exist_ok=False)
    paths = [*ROOT.glob('*.py'), ROOT / 'PLAN.md', A.L.CURRENT, A.L.TYPED, A.L.SCOPE]
    paths += [SOURCE / name for name in NAMES]
    before = {str(path): sha(path) for path in paths}
    write(output / 'INPUTS.json', before)
    started, code = time.monotonic(), 0
    try:
        result = A.verify_probabilities(**inputs())
        assert len(result['conditional_frames']) == 18 and len(result['kinds']) == 3
        assert result['probability_content_verified'] and not result['runtime_finalization_allowed']
    except BaseException:
        result, code = dict(error=traceback.format_exc()), 1
    unchanged = all(sha(Path(path)) == digest for path, digest in before.items())
    result.update(pid=os.getpid(), seconds=time.monotonic()-started, exit_code=code,
        input_sha_unchanged=unchanged, input_files=len(before), original_updates_rerun=0)
    write(output / 'RESULT.json', result)
    write(output / 'INDEX.json', {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({k: v for k, v in result.items() if k != 'error'}), flush=True)
    return code or int(not unchanged)


if __name__ == '__main__':
    raise SystemExit(main())
