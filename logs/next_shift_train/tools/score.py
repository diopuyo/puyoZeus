"""report_e36b (現本番と同じ採点器) で再生結果を採点する。使い方: python score.py <variant>"""
import json, sys
from pathlib import Path
from scripts import report_e36b
v = sys.argv[1]
report_e36b.OUT = Path('/mnt/d/puyo_analyzer/wt_nextfix/logs/next_shift_train/replay') / v
s = report_e36b.report()
print(json.dumps({k: s.get(k) for k in ('q', 'zenchi', 'deaths', 'scene_first_sec', 'gates', 'passed', 'game14_false_times')}, default=str))
