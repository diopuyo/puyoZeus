"""chain/stable ちらつき分析スクリプト."""
import json
import sys

def count_puyos(board):
    if board is None:
        return -1
    n = 0
    for row in board:
        for v in row:
            if v not in (0, 10):
                n += 1
    return n

def boards_equal(a, b):
    if a is None or b is None:
        return False
    return a == b

def run(path, t_start=27.0, t_end=29.5):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"=== {path} ===")
    print(f"全フィールド: {sorted(records[0].keys())}\n")

    prev_state = None
    prev_cnn = None

    for r in records:
        t = r.get('t_sec', 0)
        if t_start <= t <= t_end:
            p1_state = r.get('p1_state', '')
            cnn = r.get('p1_raw_cnn_board', None)
            confirmed = r.get('p1_confirmed', None)
            n_cnn = count_puyos(cnn)
            n_conf = count_puyos(confirmed)
            same_as_prev = boards_equal(cnn, prev_cnn) if prev_cnn else False

            if p1_state != prev_state:
                fidx = r['frame_idx']
                print(f"STATE CHANGE t={t:.3f} f={fidx:5d}: {str(prev_state):12s} -> {p1_state}")
                print(f"  cnn_puyo={n_cnn}, conf_puyo={n_conf}, cnn_same_as_prev={same_as_prev}")
            prev_state = p1_state
            prev_cnn = cnn

if __name__ == '__main__':
    paths = [
        'data/verify/viz/v70_match01_t2_guardB_nocfill_2026-06-01.jsonl',
        'data/verify/viz/v89_match01_t2_guardB_nocfill_2026-06-01.jsonl',
    ]
    for p in paths:
        run(p, 0.0, 9999.0)
        print()
