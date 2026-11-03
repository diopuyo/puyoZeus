"""公開盤の補正を評価・履歴へ戻さないための投影。"""
from __future__ import annotations

from dataclasses import replace
from typing import Any


def evaluation_result(result: Any) -> Any:
    """ガード介入前の盤を復元する。非介入時は同じオブジェクトを返す。"""
    sides = {}
    for name in ('p1', 'p2'):
        side = getattr(result, name, None)
        board = getattr(side, 'evaluation_confirmed_board', None)
        if board is not None:
            sides[name] = replace(side, confirmed_board=board, evaluation_confirmed_board=None)
    return replace(result, **sides) if sides else result


def publication_result(evaluated: Any, published: Any) -> Any:
    """評価側で追加された情報を残し、描画するセルだけ公開版に戻す。"""
    sides = {}
    for name in ('p1', 'p2'):
        original = getattr(published, name, None)
        if getattr(original, 'evaluation_confirmed_board', None) is not None:
            sides[name] = replace(getattr(evaluated, name),
                confirmed_board=original.confirmed_board,
                evaluation_confirmed_board=original.evaluation_confirmed_board)
    return replace(evaluated, **sides) if sides else evaluated
