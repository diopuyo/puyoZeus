"""seed 数に応じて epochs を算出する純粋関数群.

Phase L 反省 (= 2026-05-20):
    cycle 32c は 8 動画 60K patch を 5 epochs で baseline 達成。
    Phase L は 38 動画 280K patch を 5 epochs で実行 → 大幅悪化。
    1 epoch あたりの effective sample 数を一定に保つ必要。

設計思想:
    TARGET_EPOCH_SIZE = 60000 を anchor とする。
    epochs = max(MIN_EPOCHS, round(MIN_EPOCHS * n / TARGET_EPOCH_SIZE))
    上限 MAX_EPOCHS で clamp (= 過学習防止)。
"""
from __future__ import annotations

TARGET_EPOCH_SIZE: int = 60000  # cycle 32c baseline
MIN_EPOCHS: int = 5
MAX_EPOCHS: int = 30


def epochs_for_seed_count(n: int) -> int:
    """seed 数 n から推奨 epochs を算出.

    Args:
        n: seed 数 (= cell.jsonl の sample 数合計)

    Returns:
        推奨 epochs。 MIN <= epochs <= MAX。
    """
    if n <= 0:
        return MIN_EPOCHS
    ratio = n / TARGET_EPOCH_SIZE
    epochs = round(MIN_EPOCHS * ratio)
    if epochs < MIN_EPOCHS:
        return MIN_EPOCHS
    if epochs > MAX_EPOCHS:
        return MAX_EPOCHS
    return int(epochs)


def report_epochs_for_seed(n: int) -> dict:
    """epochs 算出の根拠を含む dict を返す (= log 用)."""
    return {
        "seed_count": n,
        "target_epoch_size": TARGET_EPOCH_SIZE,
        "recommended_epochs": epochs_for_seed_count(n),
        "min_epochs": MIN_EPOCHS,
        "max_epochs": MAX_EPOCHS,
        "ratio": round(n / TARGET_EPOCH_SIZE, 3),
    }
