"""一時ヘルパー: phase_e 学習結果を整形表示."""
import json

import sys
path = sys.argv[1] if len(sys.argv) > 1 else (
    "data/verify/learned_weights_phase_e_phase_aware.json"
)
with open(path, "r", encoding="utf-8") as f:
    d = json.load(f)

print("feature_names:", d["feature_names"])
print("dropped:", d["dropped"])

for p in ("start", "mid", "end"):
    r = d["phases"][p]
    print(
        f"\n=== {p} (LOOV={r['loov_mean']:.3f}+/-"
        f"{r['loov_std']:.3f}, C={r['final_C']}) ==="
    )
    for n, v in r["weights"].items():
        print(f"  {n:30s}: {v:+.4f}")
