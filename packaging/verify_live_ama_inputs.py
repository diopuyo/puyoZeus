"""認識原票から配信入力を再生し、研究v7.1の評価対象時刻を照合する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase_j.live_ama_inputs import AmaInputs
from src.ama_tsumo_seed_v5 import load_table
import puyo_core


def packets(video: int):
    path = Path('D:/puyo_analyzer/packaging/build_20261019a/packets/80_7.jsonl.gz') if video == 80 else Path(_public_dev_text('<work>/exch_review_render/ama_eval_v7/zenchi_g42/ed/out/inputs.jsonl.gz'))
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if video == 80:
                yield row
            elif row.get('kind') == 'update':
                args = row['args']['tuple']
                yield dict(id=index, t=float(args[3]), game=int(args[4]),
                           raw=[args[0]['namespace'][s]['namespace'] for s in ('p1', 'p2')])


def compare(video: int, game: int, table: object, out: Path) -> dict:
    stem = f'{video}_g{game}'
    targets = json.loads(Path(_public_dev_text(f'<work>/wt_ama_eval/logs/ama_eval_v71/inputs_{stem}.json')).read_text(encoding='utf-8'))
    targets = {r['request']['input_sec']: r for r in targets}
    inputs, matched, differences, evidence_times_differ = AmaInputs(puyo_core, table), 0, [], 0
    original = inputs.book
    def book(packet: dict) -> dict:
        value = original(packet)
        inputs.latest_book = value
        return value
    inputs.book = book
    for packet in packets(video):
        if packet['game'] != game:
            continue
        inputs.observe(packet)
        target = targets.get(packet['t'])
        if target is None:
            continue
        actual = json.loads(json.dumps(inputs.request(game, packet['t'], inputs.latest_book)))
        # 研究版は変化のない台帳原票を省略している。値と因果性を比較し、保存時刻の差は別記する。
        evidence_times_differ += target['request']['evidence_times'] != actual['request']['evidence_times']
        assert max(actual['request']['evidence_times']) <= packet['t']
        keys = ('boards', 'known', 'colors', 'pending', 'elapsed_sec', 'stable')
        found = {k: dict(expected=target['request'][k], actual=actual['request'][k])
                 for k in keys if target['request'][k] != actual['request'][k]}
        for key in ('carry', 'all_clear', 'missing'):
            if target.get(key) != actual.get(key):
                found[key] = dict(expected=target.get(key), actual=actual.get(key))
        matched += 1
        if found:
            differences.append(dict(t=packet['t'], differences=found))
        if matched % 100 == 0:
            (out/f'{stem}_progress.json').write_text(json.dumps(dict(matched=matched, mismatches=len(differences))))
            print(stem, matched, len(differences), flush=True)
    (out/f'{stem}_differences.json').write_text(json.dumps(differences, ensure_ascii=False, indent=2), encoding='utf-8')
    return dict(matched=matched, expected=len(targets), mismatches=len(differences),
                evidence_metadata_differences=evidence_times_differ)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(exist_ok=False)
    table = load_table(ROOT/'models/ama_plain/seed_table_cache.npz')
    receipt = {f'{v}_g{g}': compare(v, g, table, out) for v, g in ((80, 7), (1042, 1))}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(receipt, flush=True)
    assert all(r['matched'] == r['expected'] and r['mismatches'] == 0 for r in receipt.values())


if __name__ == '__main__':
    main()
