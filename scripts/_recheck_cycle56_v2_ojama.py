"""cycle 56_v2 board_log を KB 追加後の評価ツールで再評価.

新 metric (= ojama_disappearance, ojama_global_scarcity) が cycle 56_v2 の
ojama 退行 (= -99.6%) を catch できるか検証。
"""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

videos = ["v29m2", "v40m7", "v51m2", "v57m2", "v70m2", "v89m3", "v95m15", "v97m11"]

print("=== cycle 56_v2 board_log 再評価 (= KB 後の評価ツール) ===")
print(f"{'video':<10} | {'critical':>8} | {'oj_disap':>9} | {'oj_scarce':>10}")
print("-" * 50)

total_critical = 0
total_oj_disap = 0
total_oj_scarce = 0

for v in videos:
    bl = ROOT / "logs" / "cycle56_v2_eval" / f"viz_{v}.jsonl"
    out = Path(f"/tmp/c56v2_recheck_{v}.json")
    if not bl.exists():
        continue
    subprocess.run(
        [str(ROOT / "venv/bin/python"), "-m", "scripts.evaluate_recognition",
         "--board-log", str(bl), "--report-out", str(out)],
        check=True, capture_output=True, cwd=str(ROOT),
        env={"PYTHONPATH": str(ROOT)},
    )
    d = json.loads(out.read_text(encoding="utf-8"))
    s = d["summary"]
    by_m = s.get("by_metric", {})
    c = s["critical"]
    od = by_m.get("ojama_disappearance", 0)
    og = by_m.get("ojama_global_scarcity", 0)
    total_critical += c
    total_oj_disap += od
    total_oj_scarce += og
    print(f"{v:<10} | {c:>8} | {od:>9} | {og:>10}")

print("-" * 50)
print(f"{'TOTAL':<10} | {total_critical:>8} | {total_oj_disap:>9} | {total_oj_scarce:>10}")
print()
print("= 旧評価ツール critical = 1111 (= ojama 退行 catch なし)")
print(f"= 新評価ツール critical = {total_critical} (= ojama 関連 +{total_oj_disap + total_oj_scarce} 件 追加 catch)")
