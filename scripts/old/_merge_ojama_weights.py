"""KA: cycle 56_v2 の最終層 ojama 行/列を baseline から復元する.

cycle 56_v2 (= cnn_cycle56_v2.pt) は 5 色 critical -26.5% 改善したが、
ojama 認識が -99.6% 退行 (= board_log 158,950 → 600 cell)。
fine-tune 時に seed が ojama 0 件だったため、 ojama クラスの重みが
学習で削られた = ojama 認識能力消去。

対策: cycle 56_v2 の state_dict を base にコピーし、 最終層の
- 18.weight[6, :] (= ojama 出力 128 次元の重み)
- 18.bias[6] (= ojama 出力の bias)
を baseline (= cnn_phase_b_large_v2.pt) から復元する。

CLASS_INDEX_TO_COLOR = (0:EMPTY, 1:RED, 2:BLUE, 3:GREEN, 4:YELLOW, 5:PURPLE, 6:OJAMA)
index 6 = OJAMA (= COLOR_OJAMA = 9).

中間層 (= 0-15 層) は cycle 56_v2 のまま維持 = 5 色 + EMPTY 認識の改善を保持。
"""
from __future__ import annotations
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "models" / "cnn_phase_b_large_v2.pt"
CYCLE56_V2 = ROOT / "models" / "cnn_cycle56_v2.pt"
OUTPUT = ROOT / "models" / "cnn_cycle56_v3.pt"

# CLASS_INDEX_TO_COLOR = (0, 1, 2, 3, 4, 5, 9) → index 6 = OJAMA
OJAMA_INDEX: int = 6


def main() -> None:
    state_base = torch.load(
        str(BASELINE), map_location="cpu", weights_only=True,
    )
    state_c56 = torch.load(
        str(CYCLE56_V2), map_location="cpu", weights_only=True,
    )

    # cycle 56_v2 を base に新 state_dict 構築
    merged = {k: v.clone() for k, v in state_c56.items()}

    # 最終層の ojama 行/列を baseline から復元
    # 18.weight shape = (7, 128)、 ojama 行 = [6, :]
    # 18.bias shape = (7,)、 ojama 要素 = [6]
    before_w = merged["18.weight"][OJAMA_INDEX, :].clone()
    before_b = merged["18.bias"][OJAMA_INDEX].item()
    merged["18.weight"][OJAMA_INDEX, :] = state_base["18.weight"][OJAMA_INDEX, :]
    merged["18.bias"][OJAMA_INDEX] = state_base["18.bias"][OJAMA_INDEX]
    after_w = merged["18.weight"][OJAMA_INDEX, :].clone()
    after_b = merged["18.bias"][OJAMA_INDEX].item()

    # 差分確認
    w_diff = (after_w - before_w).abs().max().item()
    b_diff = abs(after_b - before_b)
    print("=== ojama 重み復元 ===")
    print(f"18.weight[{OJAMA_INDEX}, :] max abs diff: {w_diff:.6f}")
    print(f"18.bias[{OJAMA_INDEX}] diff: {b_diff:.6f}")
    print(f"= 0 でないことを確認 (= 0 なら復元失敗)")

    # 他の重み (= 5 色 + EMPTY + 中間層) は cycle 56_v2 と一致確認
    for k, v in merged.items():
        if k == "18.weight":
            # ojama 行以外は cycle 56_v2 と一致
            non_ojama_rows = list(range(7))
            non_ojama_rows.remove(OJAMA_INDEX)
            for i in non_ojama_rows:
                assert torch.equal(v[i, :], state_c56[k][i, :]), \
                    f"{k} row {i} 不一致"
        elif k == "18.bias":
            non_ojama_idx = [i for i in range(7) if i != OJAMA_INDEX]
            for i in non_ojama_idx:
                assert v[i].item() == state_c56[k][i].item(), \
                    f"{k}[{i}] 不一致"
        else:
            assert torch.equal(v, state_c56[k]), f"{k} 不一致"
    print("= 他重み all cycle 56_v2 と一致 (= 中間層 + 5 色 + EMPTY 維持)")

    # 保存
    torch.save(merged, str(OUTPUT))
    print(f"\n=== saved to {OUTPUT} ===")


if __name__ == "__main__":
    main()
