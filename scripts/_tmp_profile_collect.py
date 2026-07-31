# collect の20秒窓を cProfile し、遅い関数を特定する(XII/認識CNN/背景の切り分け)
import cProfile
import pstats
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.collect_indicators_v2 import collect

pr = cProfile.Profile()
pr.enable()
n = collect(
    Path("/home/ryouj/frames/video_33.mp4"),
    Path("/tmp/prof_v33_20s.csv"),
    max_sec=20.0,
    start_sec=1200.0,  # mid窓(満杯盤面)で最悪ケースを見る
    board_npz_path=None,
)
pr.disable()
print(f"rows={n}")
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(25)
print(s.getvalue())
