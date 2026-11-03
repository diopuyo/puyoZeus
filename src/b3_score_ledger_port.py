"""得点ベース台帳 v2 (src/b3_score_ledger.py, 別エージェント作成) への接続口。

統括決定 2026-10-05: 予告・着弾・受け量 I の主入力は得点ベース台帳 v2。完成までは口だけ用意し、
ダミー値でなく欠測 mask で埋める。期待するインターフェース:

    class ScoreLedger:
        def update(self, frame: dict) -> dict | None   # 1フレームずつ (observations の kind=frame) 与える
    戻り値 dict のキー: available_at, pending_p1, pending_p2, generated_p1, generated_p2,
        offset_p1, offset_p2, dropped_p1, dropped_p2, carry_p1, carry_p2, rate

状態を持つが過去フレームだけに依存すること (prefix 不変)。試合境界の扱いは台帳側の責務だが、
GameState が試合切替で新しいインスタンスを作る。
"""
from __future__ import annotations

from typing import Optional

SL_KEYS: tuple[str, ...] = ("available_at", "pending_p1", "pending_p2", "generated_p1", "generated_p2",
                            "offset_p1", "offset_p2", "dropped_p1", "dropped_p2", "carry_p1", "carry_p2", "rate")


def new_ledger() -> Optional[object]:
    """台帳 v2 のインスタンス。モジュール未完成なら None (行は DEFERRED mask)。"""
    try:
        from src.b3_score_ledger import ScoreLedger  # type: ignore
    except ImportError:
        return None
    return ScoreLedger()


def validate_snapshot(snap: dict) -> dict:
    """出力契約の検査。キー不足は例外 (黙って0埋めしない)。"""
    missing = [k for k in SL_KEYS if k not in snap]
    if missing:
        raise ValueError(f"score ledger v2 の出力にキー不足: {missing}")
    return snap
