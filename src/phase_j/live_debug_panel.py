"""試験用の情報パネルへ渡す追加欄 (debug_panel) を、既にある配列から詰めるだけで作る。

overlay.debug_panel が true のときだけ使う。勝率・判定・決着など既存の公開値には触れない。
連鎖数・置ける手数・予告おじゃま・色ぷよ数などは既存の evaluations.b3_contents を
表示側が読むため、ここでは作り直さない (盤面・状態・得点だけを追加する)。
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from .contracts import OverlaySnapshot

DEBUG_PANEL_KEY = 'debug_panel'
DEBUG_PANEL_VERSION = 1
BOARD_ROWS, BOARD_COLS = 13, 6
EMPTY_CODE, OJAMA_CODE, UNKNOWN_CODE = '.', 'o', '?'
OJAMA_VALUE, UNKNOWN_VALUE = 9, 10
COLOR_VALUES = (1, 2, 3, 4, 5)
SIDE_KEYS = ('p1', 'p2')
# 盤面の値 → 1文字。赤1 青2 緑3 黄4 紫5 おじゃま o 空 . 不明 ?
CODES = {0: EMPTY_CODE, OJAMA_VALUE: OJAMA_CODE, UNKNOWN_VALUE: UNKNOWN_CODE,
         **{value: str(value) for value in COLOR_VALUES}}


def encode_board(grid: Sequence[Sequence[int]] | None) -> list[str] | None:
    """13行×6列の整数配列を、行ごとの6文字の文字列13本へ詰める。形が違えば None。"""
    if grid is None or len(grid) != BOARD_ROWS or any(len(row) != BOARD_COLS for row in grid):
        return None
    return [''.join(CODES.get(int(cell), UNKNOWN_CODE) for cell in row) for row in grid]


def _formula(side: dict[str, Any]) -> dict[str, Any] | None:
    """画面の掛け算式を読めているときだけ (左, 右) を返す。"""
    formula = side.get('formula') or {}
    if not formula.get('valid'):
        return None
    return dict(left=formula.get('left'), right=formula.get('right'))


def side_from_frame(side: dict[str, Any]) -> dict[str, Any]:
    """B3 観測 frame の片側 (確定盤・読取盤・状態・得点) を試験用の欄へ写す。"""
    return dict(confirmed=encode_board(side.get('confirmed_board')),
                reading=encode_board(side.get('raw_cnn_board')),
                state=side.get('state'), provenance=side.get('board_provenance'),
                score=side.get('score'), tsumo_count=side.get('tsumo_count'),
                formula=_formula(side))


def side_from_result(side: Any) -> dict[str, Any]:
    """B3 観測が無い経路では、認識結果の確定盤だけを使う (読取盤は無し)。"""
    board = getattr(side, 'confirmed_board', None)
    grid = None if board is None else board._grid.astype(int).tolist()
    return dict(confirmed=encode_board(grid), reading=None, state=getattr(side.state, 'name', None),
                provenance=getattr(side, 'board_provenance', None), score=getattr(side, 'score', None),
                tsumo_count=None, formula=None)


def build_debug_panel(frame: dict[str, Any] | None, result: Any = None) -> dict[str, Any] | None:
    """frame (B3 観測) を優先し、無ければ result から作る。どちらも無ければ None。"""
    if frame is not None and len(frame.get('sides', ())) == len(SIDE_KEYS):
        sides = [side_from_frame(side) for side in frame['sides']]
        stamp = dict(frame=frame.get('frame'), t_sec=frame.get('t_sec'), match_active=frame.get('match_active'))
    elif result is not None:
        sides = [side_from_result(getattr(result, key)) for key in SIDE_KEYS]
        stamp = dict(frame=None, t_sec=None, match_active=getattr(result, 'is_match_active', None))
    else:
        return None
    return dict(version=DEBUG_PANEL_VERSION, **stamp, **dict(zip(SIDE_KEYS, sides)))


def build_debug_panel_safe(frame: dict[str, Any] | None, result: Any = None) -> dict[str, Any] | None:
    """試験表示の失敗で判定の公開を止めない。失敗は記録して追加欄なしにする。"""
    try:
        return build_debug_panel(frame, result)
    except Exception:  # 表示専用。例外の種類を問わず本線へ波及させない。
        logging.getLogger(__name__).exception('試験用パネルの追加欄の生成に失敗')
        return None


def panel_enabled(publisher: Any) -> bool:
    """公開側の overlay 設定で debug_panel が true のときだけ True。設定が無ければ False。"""
    overlay = getattr(getattr(publisher, 'server', None), 'overlay', None)
    return bool(isinstance(overlay, dict) and overlay.get('debug_panel', False))


def with_debug_panel(snapshot: OverlaySnapshot, row: dict[str, Any]) -> OverlaySnapshot:
    """row に debug_panel があるときだけ、追加欄として公開 DTO の未知 field へ載せる。"""
    panel = row.get(DEBUG_PANEL_KEY)
    if panel is None:
        return snapshot
    # 凍結済みmappingは再凍結できないため、JSON互換に戻してから追加欄を足す (ON のときだけ)。
    return OverlaySnapshot.from_mapping({**snapshot.to_mapping(), DEBUG_PANEL_KEY: panel})
