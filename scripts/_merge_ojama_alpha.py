"""KA3 系: cycle 56_v2 と baseline の FC1 (= 15 層) を α 線形補間.

KA  (FC2 のみ baseline) → ojama 75% / 5 色 -29.5% 改善
KA2 (FC1+FC2 baseline)  → ojama 102% / 5 色 +3.4% 悪化

FC1 復元度を α で調整:
- FC1.weight = α * baseline + (1-α) * cycle 56_v2
- FC1.bias   = α * baseline + (1-α) * cycle 56_v2
- FC2 (= 18) は完全 baseline 復元 (= KA と同じ)
"""
from __future__ import annotations
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "models" / "cnn_phase_b_large_v2.pt"
CYCLE56_V2 = ROOT / "models" / "cnn_cycle56_v2.pt"

ALPHAS = [0.3, 0.5, 0.7]


def main() -> None:
    state_base = torch.load(str(BASELINE), map_location="cpu", weights_only=True)
    state_c56 = torch.load(str(CYCLE56_V2), map_location="cpu", weights_only=True)

    for alpha in ALPHAS:
        merged = {k: v.clone() for k, v in state_c56.items()}
        # FC1 (15) = α × baseline + (1-α) × cycle 56_v2
        for key in ("15.weight", "15.bias"):
            merged[key] = (
                alpha * state_base[key] + (1.0 - alpha) * state_c56[key]
            )
        # FC2 (18) は完全 baseline (= KA と同じ、 ojama 出力強化)
        for key in ("18.weight", "18.bias"):
            merged[key] = state_base[key].clone()
        out = ROOT / "models" / f"cnn_cycle56_v3c_a{int(alpha*10)}.pt"
        torch.save(merged, str(out))
        print(f"  alpha={alpha:.1f} -> {out.name}")

    print("done")


if __name__ == "__main__":
    main()
