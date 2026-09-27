"""B5の測定値と証跡ハッシュをコミット対象の記録へまとめる。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.run_live_pipeline_20260928 import save_json

DESTINATION = Path('docs/agent_coordination/PHASE_J_B5_2026-09-28.json')


def main() -> None:
    paths = dict(alignment='logs/live_b5_recognition/alignment.json',
                 unresolved='logs/live_b5_recognition/unresolved.json',
                 warmup='logs/live_b5_warmup/report.json', mc='logs/live_b5_mc/report.json')
    report = {name: json.loads(Path(path).read_text(encoding='utf-8')) for name, path in paths.items()}
    report['evidence_sha256'] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for pattern in ('logs/live_b5_recognition/*.npz', 'logs/live_b5_recognition/crf35_residual_*.png')
        for path in Path('.').glob(pattern)}
    report['base_commit'] = '8619006'
    save_json(DESTINATION, report)


if __name__ == '__main__':
    main()
