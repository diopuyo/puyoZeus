"""guard実終了試験だけでGPU取得を固定票にする。実走には使わない。"""
from pathlib import Path
import sys
import signal
import time
import resource_guard as G

if __name__ == '__main__':
    output = Path(sys.argv[3])
    if output.name.startswith('hang'):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(20)
        raise SystemExit(0)
    if output.name.startswith('early'):
        raise RuntimeError('synthetic_guard_early_error')
    flag = 'Active' if output.name.startswith('thermal') else 'Not Active'
    raise SystemExit(G.monitor(int(sys.argv[1]), sys.argv[2], output, interval=.01,
                              read_gpu=lambda: G.thermal(f'0, 50, Not Active, {flag}')))
