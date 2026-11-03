"""段1 直接公開の同値試験: 21c 保存入力 5,921 行を v4.9f 欄へ OFF と ON で流し、計時以外の一致を確かめる。

入力は 21c 同値検収 (docs/verification/live_obs_20261021c/recognition_parity.json) の再生出力
outputs.jsonl.gz。各行の B3 観測に入っている v4.9f 入力 (_v49f) を、本番と同じ LiveV49F・
AsyncV49F・作業 process で評価する。先に OFF が元の走行 (A2_f_fixed の v49f_updates.jsonl) の
出力を再現することを示し (測定台の健全性)、その後 OFF と ON を全行で比べる。
実行は配布 Python、作業場所は配布 app (models の相対パス)、src はこの worktree を先頭に置く。
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
from queue import Queue
import sys
import time
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
INPUTS = Path('D:/puyo_analyzer/packaging/build_20261021c_work/parity_final/replay_candidate/outputs.jsonl.gz')
SOURCE = Path('D:/puyo_analyzer/packaging/abba_20261021_default_r2/A2_f_fixed/output')
MODEL = Path('models/v49f/b3_v6.joblib')     # 21c は M3 試験構成 (live_b3.py:223-226)
TABLE = Path('models/b3/timing_table.json')
QUEUE_HIGH_WATER = 64      # 投入を待つ入力待ち行列の深さ (容量256の手前。値は FIFO 順を変えない)
POLL_SEC = .005
DRAIN_TIMEOUT_SEC = 600.
ENV_NAME = 'PUYO_LIVE_X_DIRECT_PUBLISH'
STATUS_NAME = 'v49f_display_status.json'
OFF_STATUS_KEYS = frozenset(('submitted', 'received', 'failure', 'pump_alive'))   # OFF の記録の欄 (増やさない)


def frames(path: Path = INPUTS) -> Iterator[tuple[dict, int]]:
    """B3 観測と、LiveB3.update と同じ規則の世代 (入力世代・採用試合の変化で +1) を返す。"""
    generation, epoch, adopted = 0, None, None
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row, _, _, current = json.loads(line)['b3']
            if current != epoch:
                generation, epoch, adopted = generation+1, current, None
            if row.get('adopted_game') is not None and row['adopted_game'] != adopted:
                generation, adopted = generation+1, row['adopted_game']
            yield row, generation


def apply_source_flags(source: Path = SOURCE) -> None:
    """元の走行と同じ bool 設定を spawn 子へ渡す (replay_live_recognition_20261021.py と同じ規則)。"""
    config = json.loads((source/'pipeline_config.json').read_text(encoding='utf-8'))
    for key, value in config.items():
        if isinstance(value, bool) or key in ('cpu_threads', 'evaluation_nice', 'b3_warning_workers'):
            os.environ['PUYO_'+key.upper()] = str(int(value))
    os.environ.update(PUYO_OBS_FAST_AUDIT='1', PUYO_OBS_EXACT_SEED='1', PUYO_B3_M3='1')


def run_arm(direct: bool, out: Path, limit: int | None) -> dict:
    """1 腕を流す。LiveV49F の出力は自分で読み取り、x_fresh と起床回数を数える。"""
    from src.phase_j.live_v49f import LiveV49F
    os.environ[ENV_NAME] = str(int(direct))
    out.mkdir(parents=True, exist_ok=False)
    owner = LiveV49F(MODEL, TABLE, out)
    # update() 自身も出力を読むので、読まれた結果は全部 queue 側で記録する。
    owner.outputs = RecordingQueue(owner.outputs.maxsize)
    results, counts, began = owner.outputs.seen, dict(wakes=0), time.perf_counter()
    try:
        for index, (row, generation) in enumerate(frames()):
            if (limit is not None and index >= limit) or owner.failure:
                break
            while owner.inputs.qsize() >= QUEUE_HIGH_WATER and not owner.failure:
                drain(owner, counts)
                time.sleep(POLL_SEC)
            owner.update(row, generation)
            drain(owner, counts)
        deadline = time.perf_counter()+DRAIN_TIMEOUT_SEC
        while owner.received < owner.submitted and time.perf_counter() < deadline and not owner.failure:
            drain(owner, counts)
            time.sleep(POLL_SEC)
        drain(owner, counts)
    finally:
        owner.close()
    from scripts.measure_live_x_effects import semantic
    shown = [semantic({k: v for k, v in r.items() if k != 'x_fresh'}) for r in results]
    status_path = out/STATUS_NAME
    return dict(direct=direct, submitted=owner.submitted, received=owner.received, failure=owner.failure,
                results=len(results), x_fresh=sum(bool(r.get('x_fresh')) for r in results),
                results_sha256=sha256(json.dumps(shown, sort_keys=True).encode()).hexdigest(),
                wakes=counts['wakes'], seconds=time.perf_counter()-began, out=str(out),
                status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else None)


class RecordingQueue(Queue):
    """LiveV49F の出力 queue。get_nowait で読まれた結果を順に記録する (値は変えない)。"""

    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize)
        self.seen: list[dict] = []

    def get_nowait(self) -> Any:
        value = super().get_nowait()
        self.seen.append(value)
        return value


def drain(owner: Any, counts: dict) -> None:
    """評価所有者と同じ receive() で読み切る (失敗の記録も本番どおり)。ON は起床 pipe も読む。"""
    wake = getattr(owner, 'wake', None)
    if wake is not None:
        counts['wakes'] += wake.drain()
    owner.receive()


def compare_audits(left: Path, right: Path, key: str = 'id') -> dict:
    """作業 process の原票を、計時の列だけ除いて全列で比べる (compare_wall_gate.py と同じ除外)。"""
    from scripts.measure_live_x_effects import semantic
    rows = [{(r['game'], r[key]): semantic(r) for r in map(json.loads, path.read_text(encoding='utf-8').splitlines())}
            for path in (left, right)]
    common = sorted(rows[0].keys() & rows[1].keys())
    different = [k for k in common if rows[0][k] != rows[1][k]]
    return dict(left=len(rows[0]), right=len(rows[1]), common=len(common), different=len(different),
                samples=[list(k) for k in different[:20]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    apply_source_flags()
    receipt = dict(inputs=str(INPUTS), input_sha256=sha256(INPUTS.read_bytes()).hexdigest(),
                   source=str(SOURCE), app=os.getcwd(), python=sys.executable, limit=args.limit)
    receipt['off'] = run_arm(False, args.out/'off', args.limit)
    receipt['on'] = run_arm(True, args.out/'on', args.limit)
    audits = {name: args.out/name/'v49f_updates.jsonl' for name in ('off', 'on')}
    receipt['off_vs_source'] = compare_audits(audits['off'], SOURCE/'v49f_updates.jsonl')
    receipt['on_vs_source'] = compare_audits(audits['on'], SOURCE/'v49f_updates.jsonl')
    receipt['off_vs_on'] = compare_audits(audits['off'], audits['on'])
    receipt['passed'] = passed(receipt)
    (args.out/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def record_matches(off: dict, on: dict) -> bool:
    """起床の記録 (v49f_display_status.json) の検査。受領票に 'status' 欄が無い古い形式は判定に入れない。

    OFF は従来の 4 欄のまま (x_* も direct_publish も出ない)。ON の x_fresh・x_wakes は、結果と起床 pipe を
    この台が独立に数えた値と一致し、x_fresh_fires (側×発火) は x_fresh 以上 (1 結果が両側なら 2 発火)。
    """
    if 'status' not in off or 'status' not in on:
        return True
    off_status, on_status = off['status'], on['status']
    if off_status is None or on_status is None:
        return False
    return (set(off_status) == OFF_STATUS_KEYS and on_status.get('direct_publish') is True
            and on_status.get('x_fresh') == on['x_fresh'] and on_status.get('x_wakes') == on['wakes']
            and on_status.get('x_fresh_fires', -1) >= on['x_fresh'])


def passed(receipt: dict) -> bool:
    """全行一致。OFF が旧予測時間門で止まった場合は、止まるまでの全行と ON の全長で判定する。"""
    off, on = receipt['off'], receipt['on']
    stopped = off['failure'] is not None and '予算超過' in off['failure'] or '予測時間超過' in (off['failure'] or '')
    complete = off['failure'] is None and off['results_sha256'] == on['results_sha256']
    return (on['failure'] is None and on['received'] == on['submitted']
            and all(receipt[k]['different'] == 0 for k in ('off_vs_source', 'on_vs_source', 'off_vs_on'))
            and receipt['off_vs_on']['common'] == off['received'] and (complete or stopped)
            and record_matches(off, on))


if __name__ == '__main__':
    main()
