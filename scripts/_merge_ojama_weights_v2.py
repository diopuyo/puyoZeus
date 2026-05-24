"""KA2: cycle 56_v2 + baseline の最終 2 層 (= FC1=15、 FC2=18) を復元.

cycle 56_v3 (= KA、 最終層 18 のみ baseline 復元) は ojama 75% で水準外。
原因仮説: 中間層 (= FC1=15、 conv layers) も cycle 56_v2 で微調整され
ojama 特徴抽出が弱まっている。

KA2: 最終 2 層 = FC1 (= 15.weight, 15.bias) + FC2 (= 18.weight, 18.bias) を
全て baseline から復元する。 ojama 認識率向上期待 (= 90%+ 目標)。
5 色微改善は中間 conv 層 (= 0-11) で残る可能性。

CLASS_INDEX_TO_COLOR = (0:EMPTY, 1:RED, 2:BLUE, 3:GREEN, 4:YELLOW, 5:PURPLE, 6:OJAMA)
"""
from __future__ import annotations
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "models" / "cnn_phase_b_large_v2.pt"
CYCLE56_V2 = ROOT / "models" / "cnn_cycle56_v2.pt"
OUTPUT = ROOT / "models" / "cnn_cycle56_v3b.pt"


def main() -> None:
    state_base = torch.load(
        str(BASELINE), map_location="cpu", weights_only=True,
    )
    state_c56 = torch.load(
        str(CYCLE56_V2), map_location="cpu", weights_only=True,
    )

    # cycle 56_v2 を base に新 state_dict 構築
    merged = {k: v.clone() for k, v in state_c56.items()}

    # 最終 2 層を baseline から完全復元
    for key in ("15.weight", "15.bias", "18.weight", "18.bias"):
        merged[key] = state_base[key].clone()
        print(f"  restored {key}: {tuple(merged[key].shape)} from baseline")

    # 中間 conv 層は cycle 56_v2 維持 (= 5 色特徴抽出の微改善期待)
    for k in merged:
        if k.startswith(("15.", "18.")):
            continue
        assert torch.equal(merged[k], state_c56[k]), f"{k} 不一致"
    print("= conv 層 (0-11) all cycle 56_v2 と一致 (= 5 色微改善維持期待)")

    torch.save(merged, str(OUTPUT))
    print(f"\n=== saved to {OUTPUT} ===")


if __name__ == "__main__":
    main()
