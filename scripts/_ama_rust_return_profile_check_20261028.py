"""return_profile (盤ごとに深さ 16 までを 1 回、手ごとの累積最大) から引いた C(h) が、
深さ h で探索し直した最大 (ranked、rust_return.py の C_rust) と一致するかを 200 件で確かめ、時間を測る。
使い方: python profile_check.py <Rust の ama_reference>"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_DEPTH = 16
RATE = 70


def main() -> None:
    previous = {r['id']: r for r in json.loads((HERE/'rust_return.json').read_text(encoding='utf-8'))['rows']}
    cases = [line.split() for line in (HERE/'ama_input.txt').read_text().splitlines()]
    proc = subprocess.Popen([sys.argv[1]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env={'AMA_THREADS': '1'})
    same, times, rows = 0, [], []
    for _, cid, board, colors, pairs, *_rest, h in cases:
        code = {c: i+1 for i, c in enumerate(colors)}
        cells = [0 if ch == '0' else 9 if ch == '9' else code[ch] for ch in board]
        queue = [[code[pairs[i]], code[pairs[i+1]]] for i in range(0, len(pairs)-1, 2)]
        depth = min(MAX_DEPTH, len(queue))
        began = time.perf_counter()
        proc.stdin.write(json.dumps(dict(board=cells, visible=queue[:max(depth, 2)], op='return_profile',
                                         width=250, depth=depth))+'\n')
        proc.stdin.flush()
        cummax = json.loads(proc.stdout.readline())['cummax']
        times.append(time.perf_counter()-began)
        k = max(1, min(int(h), MAX_DEPTH, len(queue)))
        c_profile = cummax[k-1]//RATE
        same += c_profile == previous[cid]['C_rust']
        rows.append(dict(id=cid, h=int(h), C_profile=c_profile, C_ranked=previous[cid]['C_rust']))
    proc.stdin.close()
    times.sort()
    report = dict(cases=len(rows), same=same, different=[r for r in rows if r['C_profile'] != r['C_ranked']][:5],
                  ms_p50=round(1000*times[len(times)//2], 1), ms_p95=round(1000*times[int(len(times)*.95)], 1),
                  ms_max=round(1000*times[-1], 1))
    (HERE/'profile_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
