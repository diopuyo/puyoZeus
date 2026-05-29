"""B1 PiecePersistenceGuard: STABLE 中 cell 色の物理保護ガード。

設計原則:
- STABLE 中に確定した非 EMPTY cell を「物理事由なく消えない」 として保護
- NON-STABLE 中は保護一時停止 (= 連鎖 / 落下中の変化を許容)
- NON-STABLE → STABLE 復帰時は保護リセット (= 新しい盤面を受け入れ)
- 「ぷよを消す経路」 に絶対ならない (= EMPTY → 非 EMPTY 変換禁止)

用途: STABLE 中の散発色ブレ (= flicker) を構造的に削減する。
patch_fp の flicker 211 件削減が目的。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_UNKNOWN,
    Board,
)

# 保護対象として記録する最大 cell 数 (= 全 cell 数 BOARD_ROWS × BOARD_COLS)
_MAX_PROTECTED_CELLS: int = BOARD_ROWS * BOARD_COLS


@dataclass
class PiecePersistenceGuard:
    """STABLE 中 cell 色の物理保護ガード。

    STABLE 確定時の非 EMPTY cell を記録し、 STABLE 中に「色 → EMPTY/UNKNOWN」
    に変わる更新を block して元色を維持する。
    「ぷよを消す経路」には絶対にならない設計: EMPTY→非 EMPTY 変換は禁止。
    """

    _protected: dict[tuple[int, int], int] = field(default_factory=dict)
    _in_stable: bool = False

    def on_stable_confirmed(self, board: Board) -> None:
        """STABLE 確定時に非 EMPTY cell を保護登録。

        既存 _protected と統合する (= 新規 cell 追加、 既存 cell 値も最新化)。
        EMPTY / UNKNOWN cell は保護しない (= 「ぷよを消す経路」 防止)。
        """
        self._in_stable = True
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                v = int(board.get(r, c))
                if v not in (COLOR_EMPTY, COLOR_UNKNOWN):
                    # 非 EMPTY cell を保護 (= 既存値も最新化)
                    self._protected[(r, c)] = v

    def guard(self, candidate_board: Board) -> Board:
        """protected cell が EMPTY/UNKNOWN に変わる更新を block して返す。

        - candidate[r][c] == COLOR_EMPTY/UNKNOWN かつ _protected に登録あり
          → 元色 (= _protected 値) を維持
        - candidate[r][c] != EMPTY/UNKNOWN → そのまま (= 色変化は許容)
        - _protected に登録なし → そのまま
        「ぷよを消す経路」 に絶対ならない: EMPTY→非EMPTY 変換は行わない。
        """
        if not self._in_stable or not self._protected:
            return candidate_board
        result = candidate_board.copy()
        for (r, c), protected_color in self._protected.items():
            candidate_v = int(candidate_board.get(r, c))
            # EMPTY/UNKNOWN への変更を block → 元色維持
            if candidate_v in (COLOR_EMPTY, COLOR_UNKNOWN):
                result.set(r, c, protected_color)
        return result

    def on_non_stable_enter(self) -> None:
        """NON-STABLE 開始時に保護リセット (= 連鎖 / 落下中の変化を許容)。"""
        self._protected.clear()
        self._in_stable = False

    def on_non_stable_exit(self) -> None:
        """NON-STABLE 終了時 (= STABLE 復帰直前) の hook。通常は何もしない。"""
        # STABLE 復帰後の on_stable_confirmed で保護が再構築される
        pass

    def reset(self) -> None:
        """完全リセット (= 試合開始時等)。"""
        self._protected.clear()
        self._in_stable = False

    def to_dict(self) -> dict:
        """デバッグ用シリアライズ。"""
        return {
            "n_protected": len(self._protected),
            "in_stable": self._in_stable,
            "protected_positions": list(self._protected.keys()),
        }
