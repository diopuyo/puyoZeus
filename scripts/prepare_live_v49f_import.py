"""候補の本番未採用フラグを別モジュールへ隔離する。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT/'src').glob('prefire_*.py'):
    text = path.read_text(encoding='utf-8')
    text = text.replace('from src.production_config import PREFIRE_',
                        'from src.prefire_trial_defaults import PREFIRE_')
    path.write_text(text, encoding='utf-8', newline='\n')
