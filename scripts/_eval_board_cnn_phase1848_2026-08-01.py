"""学習済み board CNN を user 確定の位相定義 (18/48) で切り直して評価する。

train_board_cnn.py の位相は3分位近似 (暫定)。確定定義は:
  盤面のぷよ総量 (おじゃま含む) で 序盤<=18 / 中盤19-47 / 終盤>=48、
  1P/2P は遅い方 (max) を採用 (memory reference_phase_split_by_color_puyo_count)。

予測は学習スクリプトの _evaluate をそのまま使う (自前バッチ処理は
0.687→0.602 に化けるバグがあったため破棄。_evaluate はベストepoch の
val_loss/AUC を完全再現することを確認済み)。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.train_board_cnn import (  # noqa: E402
    DEFAULT_SEED, _evaluate, _load_pairs, _make_dataset, _split_video_ids,
)
from src.cnn_embedding.board_cnn import SiameseBoardCNN  # noqa: E402

PAIRS = Path("data/indicators_v2/board_pairs_regen_2026-08-01.npz")
MODEL = Path("models/board_cnn_regen_2026-08-01.pt")
EARLY_MAX, LATE_MIN = 18, 48


def phase_of(count: int) -> int:
    """ぷよ総量 (おじゃま含む) から位相 (0=序盤,1=中盤,2=終盤)。"""
    if count <= EARLY_MAX:
        return 0
    if count >= LATE_MIN:
        return 2
    return 1


def main() -> None:
    b1, b2, won, vid, pc = _load_pairs(PAIRS)
    _tr, va = _split_video_ids(vid, seed=DEFAULT_SEED)
    val_ds = _make_dataset(b1, b2, won, pc, va)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseBoardCNN(dropout=0.3, width_mult=0.5)
    sd = torch.load(str(MODEL), map_location=dev, weights_only=False)
    model.load_state_dict(
        sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd,
    )
    model = model.to(dev)
    vl, y_true, y_score = _evaluate(model, val_ds, dev)
    print(f"再現確認: val_loss={vl:.4f} 全体AUC={roc_auc_score(y_true, y_score):.4f}")

    # user 確定の位相: 側別ぷよ総量 → 遅い方 (max)
    b1v, b2v = b1[va], b2[va]
    c1 = (b1v.reshape(len(b1v), -1) != 0).sum(axis=1)
    c2 = (b2v.reshape(len(b2v), -1) != 0).sum(axis=1)
    ph = np.array(
        [max(phase_of(int(a)), phase_of(int(b))) for a, b in zip(c1, c2)],
    )
    for k, name in ((0, "序盤(<=18)"), (1, "中盤(19-47)"), (2, "終盤(>=48)")):
        m = ph == k
        if m.sum() > 50 and len(set(y_true[m])) > 1:
            print(f"  {name}: AUC={roc_auc_score(y_true[m], y_score[m]):.4f} "
                  f"(n={int(m.sum())})")


if __name__ == "__main__":
    main()
