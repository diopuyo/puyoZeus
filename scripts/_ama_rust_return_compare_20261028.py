"""PC2 の 200 件 (返せなかった 100・返された 100、+1 秒) を Rust の ama で探索し、返しの最大 C を出して比べる。
本物のツモを h 組 (最大 16) 渡し、幅 250・1 枝で探索、全ノードで撃った連鎖の得点の最大を C とする (citrus の C_ama と同じ定め)。
使い方: python rust_return.py <Rust の ama_reference> [スレッド数]"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_DEPTH = 16          # citrus の beam の深さと同じ上限
WIDTH = 250
NO_TRIGGER = 10**9      # 途中で打ち切らない
RATE = 70


def request(board: str, colors: str, pairs: str, h: int) -> dict:
    code = {c: i+1 for i, c in enumerate(colors)}
    cells = [0 if ch == '0' else 9 if ch == '9' else code[ch] for ch in board]
    queue = [[code[pairs[i]], code[pairs[i+1]]] for i in range(0, len(pairs)-1, 2)]
    depth = max(1, min(h, MAX_DEPTH, len(queue)))
    visible = queue[:max(depth, 2)]
    return dict(board=cells, visible=visible, op='ranked', branches=1, width=WIDTH, depth=depth, trigger=NO_TRIGGER)


def auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    wins = sum((p > n) + 0.5*(p == n) for p in pos for n in neg)
    return wins/(len(pos)*len(neg))


def main() -> None:
    exe, threads = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '1'
    rows = {r['id']: r for r in csv.DictReader((HERE/'ama_beam_cmp.csv').open(encoding='utf-8-sig'))}
    cases = [line.split() for line in (HERE/'ama_input.txt').read_text().splitlines()]
    proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                            env={'AMA_THREADS': threads})
    out, times = [], []
    for _, cid, board, colors, pairs, _eboard, _attack, _frame, _target, h in cases:
        began = time.perf_counter()
        proc.stdin.write(json.dumps(request(board, colors, pairs, int(h)))+'\n')
        proc.stdin.flush()
        result = json.loads(proc.stdout.readline())
        times.append(time.perf_counter()-began)
        best = max((c[1] for c in result.get('candidates', [])), default=0)
        row = rows[cid]
        c_rust = best//RATE
        out.append(dict(id=cid, kind=row['kind'], h=int(h), attack=float(row['attack']), C_beam=float(row['C_beam']),
                         C_ama=float(row['C_ama']), C_rust=c_rust, R_beam=float(row['R_beam']), R_ama=float(row['R_ama']),
                         R_rust=float(row['attack'])-c_rust, p=float(row['p'])))
    proc.stdin.close()
    labels = [int(r['kind'] == '返せなかった') for r in out]
    report = {name: round(auc([r[name] for r in out], labels), 3) for name in ('R_beam', 'R_ama', 'R_rust', 'p')}
    report['R_rust_both'] = round(auc([r['attack']-max(r['C_ama'], r['C_rust']) for r in out], labels), 3)
    for kind in ('返せなかった', '返された'):
        sub = [r for r in out if r['kind'] == kind]
        report[kind] = dict(n=len(sub), median_C_beam=sorted(r['C_beam'] for r in sub)[len(sub)//2],
                            median_C_ama=sorted(r['C_ama'] for r in sub)[len(sub)//2],
                            median_C_rust=sorted(r['C_rust'] for r in sub)[len(sub)//2],
                            rust_gt_ama=sum(r['C_rust'] > r['C_ama'] for r in sub),
                            rust_lt_ama=sum(r['C_rust'] < r['C_ama'] for r in sub),
                            beam_R_pos_but_rust_returns=sum(r['R_beam'] > 0 and r['R_rust'] <= 0 for r in sub))
    times.sort()
    report['ms_p50_p95_max'] = [round(1000*times[len(times)//2], 1), round(1000*times[int(len(times)*.95)], 1),
                                round(1000*times[-1], 1)]
    (HERE/'rust_return.json').write_text(json.dumps(dict(report=report, rows=out), ensure_ascii=False, indent=1),
                                         encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
