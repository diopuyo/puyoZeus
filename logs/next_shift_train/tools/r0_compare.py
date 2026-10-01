"""台の確認: R0 再生出力と現本番 e36b_on のバイト比較 (display/events/diagnostics/status ほか全ファイル)。"""
import hashlib, json, sys
from pathlib import Path
base = Path('/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/pending_expiry/e36b_on/on')
new = Path('/mnt/d/puyo_analyzer/wt_nextfix/logs/next_shift_train/replay') / sys.argv[1] / 'on'
NAMES = ('display.npz', 'events.jsonl', 'events.diagnostics.json', 'status.json')
out = {}
for source in sys.argv[2:]:
    rel = Path(source) if source in ('zenchi', 'review') else Path('renders') / source / 'on'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    out[source] = {n: dict(same=sha(base / rel / n) == sha(new / rel / n), base=sha(base / rel / n) is not None)
                   for n in NAMES}
print(json.dumps(out, indent=1))
