import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rows = {}
count = 0
for p, _, files in os.walk(root):
    for f in files:
        fp = Path(p) / f
        rel = fp.relative_to(root).parts
        key = '/'.join(rel[:depth])
        rows[key] = rows.get(key, 0) + fp.stat().st_size
        count += 1
total = sum(rows.values())
print(f'total {total/1e6:.1f} MB, files {count}')
for k, v in sorted(rows.items(), key=lambda kv: -kv[1])[:int(sys.argv[3]) if len(sys.argv) > 3 else 25]:
    print(f'{v/1e6:9.1f} MB  {k}')
