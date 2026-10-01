"""提供側 FrozenQueue と exev 診断 Frozen(2) (R3 実測済み) の同値性をランダム列で確認する。"""
import importlib.util, json
import numpy as np
from src.next_queue_serving import FrozenQueue
spec = importlib.util.spec_from_file_location('diag', '/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/scripts/_diag_next_shift_align_20260930.py')
diag = importlib.util.module_from_spec(spec); spec.loader.exec_module(diag)
rng = np.random.default_rng(0); steps = 0; bad = 0
for trial in range(2000):
    a, b = FrozenQueue(), diag.Frozen(2)
    boards = [rng.integers(0, 3, size=(13, 6)) for _ in range(4)]
    for _ in range(60):
        if rng.random() < .1:
            a.leave(); b.leave(); continue
        g = boards[rng.integers(4)]
        q = tuple(int(v) for v in rng.integers(0, 4, size=4))
        steps += 1
        bad += int(a.update(g, q) != b.update(g, q))
print(json.dumps(dict(steps=steps, mismatches=bad)))
