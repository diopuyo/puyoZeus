"""大きな原票を全読込みせず、測定の末尾だけを確認する。"""
from pathlib import Path
import argparse
import json

ROOT = Path('D:/puyo_analyzer/packaging/abba_20261021')
TAIL_BYTES = 256 * 1024


def last_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size-TAIL_BYTES))
        lines = stream.read().splitlines()
    for line in reversed(lines[1:] if size > TAIL_BYTES else lines):
        try:
            return json.loads(line.decode('utf-8-sig'))
        except (ValueError, UnicodeError):
            continue
    return {}


def main() -> None:
    environment = last_json(ROOT/'environment_r2.jsonl')
    name = environment.get('active_run', 'A1_f').strip()
    current = ROOT/name
    recognition = last_json(current/'output/spool/frame_timing_recognition.jsonl')
    poll = last_json(current/'polls.jsonl')
    evaluations = (poll.get('snapshot') or {}).get('evaluations') or {}
    print(json.dumps(dict(at=environment.get('at'), run=name,
        t_sec=recognition.get('t_sec'), frame=recognition.get('frame'),
        free_gib=environment.get('free_gib'),
        performance=[r for r in environment.get('clocks', []) if r['Name'] in
                     ('0,0', '0,4', '0,8', '0,12', '0,13', '0,14', '0,15', '_Total')],
        vm=[r for r in environment.get('processes', []) if r['Name'].startswith('vmmem')],
        reasons={k: (evaluations.get(k) or {}).get('reason') for k in ('v49f', 'ama_plain')},
        values={k: (evaluations.get(k) or {}).get(field) for k, field in
            (('practical', 'p1_win_probability'), ('v49f', 'p1'), ('ama_plain', 'p1'))},
        exits={p.name: json.loads(p.read_text(encoding='utf-8-sig'))
               for p in ROOT.glob('*.exit.json')},
        failures=[str(p.relative_to(current)) for folder in (current, current/'output')
                  for p in folder.glob('*failure*')]), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    ROOT = parser.parse_args().root
    main()
