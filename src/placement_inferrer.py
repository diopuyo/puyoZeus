"""物理推論主軸の置き判定モジュール (Phase 1, cycle 71).

設計思想:
    TSUMO_FALL→STABLE 遷移時の着地確定で「CNN 差分位置」 に依存せず、
    物理的に妥当な配置パターンを全列挙 → NEXT 履歴色を割り当て →
    CNN 一致度で 1 つに絞る. CNN は答え合わせのヒントとして使う.

主 API:
    infer_placement(prev, cnn_after, next_pair) -> Board

    1. 物理パターン全列挙 (= 各列で縦置き 1 通り + 隣接列で横置き 1 通り)
    2. NEXT (top, bot) 色を縦置きは上下入替 2 通り、 横置きは左右入替 2 通りで配分
    3. CNN 差分との一致度 (= cell 単位一致数) で最適候補を選択
    4. 候補が複数同点なら decision (デフォルト着地パターン優先)

cycle 71l (β2', 2026-05-13): frame_bgr + region を渡せば着地 2 cell の HSV と
NEXT 色 2 種類の HSV center 距離で色順序を確定. 回転落下に対応、 CNN が
完全誤認しても NEXT 色 2 種類のどちらかに確定する.

仮説 A (= 置き誤認 → freeze) の根本対策として、 既存 _compute_landing_inferred
(= CNN 差分位置採用) を置き換える.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, COLOR_BLUE, COLOR_EMPTY, COLOR_GREEN, COLOR_OJAMA,
    COLOR_PURPLE, COLOR_RED, COLOR_UNKNOWN, COLOR_YELLOW, Board,
)


# cycle 71l (β2'): 各色の HSV center (= 近似中心). DEFAULT_COLOR_RANGES の範囲中央.
# H 環状 (= 0-180), S/V は 0-255 範囲. NEXT 色 2 種類への距離計算用.
COLOR_HSV_CENTERS: dict[int, tuple[int, int, int]] = {
    COLOR_RED:    (7, 220, 200),    # H=0-13
    COLOR_BLUE:   (115, 220, 180),  # H=100-130
    COLOR_GREEN:  (67, 220, 180),   # H=50-85
    COLOR_YELLOW: (26, 200, 220),   # H=14-38
    COLOR_PURPLE: (147, 200, 180),  # H=130-165
}


def _hsv_distance(
    h: int, s: int, v: int, target: tuple[int, int, int],
) -> float:
    """HSV 値から target color center までの加重距離.

    H は環状 (= 0-180)、 S/V は重み下げ (= 色相が支配的).
    """
    th, ts, tv = target
    dh = min(abs(h - th), 180 - abs(h - th))
    return float(dh * 2.0 + abs(s - ts) * 0.05 + abs(v - tv) * 0.05)


def _classify_next_pair_by_hsv(
    cell_a_patch: np.ndarray,
    cell_b_patch: np.ndarray,
    next_pair: tuple[int, int],
) -> tuple[int, int]:
    """着地 2 cell の HSV と NEXT 色 2 種類への距離で色順序を確定する.

    cycle 71l (β2'): ユーザー指摘 (= 回転多用、 整合性で物理推論) に対応.
    NEXT 色は 100% 信頼 (= memory project_next_detector_perfect_accuracy)
    なので、 「2 cell に NEXT 色 2 種類のどちらかが入る」 は確定. 順序のみを
    HSV 距離で判定.

    Args:
        cell_a_patch: 縦置きなら上 cell、 横置きなら左 cell の BGR patch.
        cell_b_patch: 縦置きなら下 cell、 横置きなら右 cell の BGR patch.
        next_pair: (NEXT[0], NEXT[1]) 色コード.

    Returns:
        (cell_a_color, cell_b_color) 確定色. 同色 NEXT or HSV center 不在は
        next_pair そのままで fallback.
    """
    c0, c1 = next_pair
    if c0 == c1:
        return (c0, c1)
    if c0 not in COLOR_HSV_CENTERS or c1 not in COLOR_HSV_CENTERS:
        return next_pair
    if cell_a_patch.size == 0 or cell_b_patch.size == 0:
        return next_pair

    hsv_a = cv2.cvtColor(cell_a_patch, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(cell_b_patch, cv2.COLOR_BGR2HSV)
    ha = int(np.median(hsv_a[:, :, 0]))
    sa = int(np.median(hsv_a[:, :, 1]))
    va = int(np.median(hsv_a[:, :, 2]))
    hb = int(np.median(hsv_b[:, :, 0]))
    sb = int(np.median(hsv_b[:, :, 1]))
    vb = int(np.median(hsv_b[:, :, 2]))

    center_0 = COLOR_HSV_CENTERS[c0]
    center_1 = COLOR_HSV_CENTERS[c1]
    # 候補 std: a=c0, b=c1
    d_std = (
        _hsv_distance(ha, sa, va, center_0)
        + _hsv_distance(hb, sb, vb, center_1)
    )
    # 候補 rev: a=c1, b=c0 (= 回転落下)
    d_rev = (
        _hsv_distance(ha, sa, va, center_1)
        + _hsv_distance(hb, sb, vb, center_0)
    )
    if d_rev < d_std:
        return (c1, c0)
    return (c0, c1)


def _extract_cell_patch_from_frame(
    frame_bgr: np.ndarray, region: object, row: int, col: int,
) -> np.ndarray | None:
    """1920x1080 frame から指定 cell の sample 領域 patch を抽出.

    region は BoardRegion 想定 (cell_sample_rect メソッドを持つ).
    """
    try:
        x1, y1, x2, y2 = region.cell_sample_rect(row, col)  # type: ignore[attr-defined]
    except Exception:
        return None
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame_bgr.shape[1], int(x2))
    y2 = min(frame_bgr.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


# 1 ツモは必ず 2 puyo. 配置の 2 cell を表す.
@dataclass(frozen=True)
class LandingPattern:
    """物理的に妥当な 1 ツモ着地パターン.

    Attributes:
        cells: ((row_a, col_a), (row_b, col_b))
            - vertical: cells[0] = 上 cell, cells[1] = 下 cell (= 同列、 row 隣接)
            - horizontal: cells[0] = 左 cell, cells[1] = 右 cell (= 同 row、 col 隣接)
        orientation: "vertical" or "horizontal"
    """
    cells: tuple[tuple[int, int], tuple[int, int]]
    orientation: str


def _top_empty_row(board: Board, col: int) -> int | None:
    """指定列で「最上空 row」 (= 次のぷよが落ちて停止する row) を返す.

    上から順に走査し、 最初の puyo (= EMPTY/UNKNOWN 以外) が見つかったら
    その 1 つ上 row を返す. puyo がなければ最下段 row=BOARD_ROWS-1.
    満杯なら None.
    """
    for r in range(BOARD_ROWS):
        v = int(board.get(r, col))
        if v not in (COLOR_EMPTY, COLOR_UNKNOWN):
            # row r に puyo あり → r-1 が落下停止 row
            return r - 1 if r > 0 else None
    return BOARD_ROWS - 1


def enumerate_landing_patterns(
    confirmed_before: Board,
) -> list[LandingPattern]:
    """既存盤面に「1 ツモを置ける」 物理パターンを全列挙.

    縦置き: 各列で最上空 row r、 (r-1, c) と (r, c) の 2 cell. r >= 1 必須.
    横置き: 隣接列 (c, c+1) で両列の最上空 row が同じ → (r, c), (r, c+1).
    """
    patterns: list[LandingPattern] = []
    # 各列の最上空 row を計算
    top_empty: list[int | None] = [
        _top_empty_row(confirmed_before, c) for c in range(BOARD_COLS)
    ]
    # 縦置き
    for c in range(BOARD_COLS):
        r = top_empty[c]
        if r is None or r < 1:
            continue  # 満杯 or 1 cell しか入らない (隠し段含めても 2 cell 不足)
        patterns.append(LandingPattern(
            cells=((r - 1, c), (r, c)),
            orientation="vertical",
        ))
    # 横置き
    for c in range(BOARD_COLS - 1):
        rl = top_empty[c]
        rr = top_empty[c + 1]
        if rl is None or rr is None or rl != rr:
            continue
        patterns.append(LandingPattern(
            cells=((rl, c), (rr, c + 1)),
            orientation="horizontal",
        ))
    return patterns


def materialize_pattern(
    base: Board, pattern: LandingPattern,
    color_first: int, color_second: int,
) -> Board:
    """base 盤面に pattern を color_first / color_second で書き込んだ新盤面を返す.

    cells[0] = color_first, cells[1] = color_second.
    """
    out = base.copy()
    (r1, c1), (r2, c2) = pattern.cells
    out.set(r1, c1, color_first)
    out.set(r2, c2, color_second)
    return out


def enumerate_color_assignments(
    pattern: LandingPattern, next_pair: tuple[int, int],
) -> list[tuple[int, int]]:
    """1 つの pattern に NEXT (top, bot) 色を配分する全候補を返す.

    縦置き: 2 通り ((top, bot), (bot, top))
    横置き: 2 通り ((top, bot), (bot, top))
    top == bot なら 1 通り.
    """
    top, bot = int(next_pair[0]), int(next_pair[1])
    if top == bot:
        return [(top, bot)]
    return [(top, bot), (bot, top)]


def _diff_match_score(
    candidate: Board, cnn_after: Board,
    pattern_cells: tuple[tuple[int, int], tuple[int, int]],
) -> int:
    """candidate と cnn_after の cell 単位一致数 (= 全盤面).

    特に pattern_cells の 2 cell が cnn_after と一致するかを重視する.
    一致 score = 全 cell の一致数 + pattern_cells で一致なら大ボーナス.

    cycle 71l β1: pattern_cells ボーナスを +5 → +50 に強化.
    着地位置 (= 物理パターン) を CNN 一致度より圧倒的に優先することで、
    CNN が着地後の cell 色を誤認した時でも正しい配置候補を選ばせる.
    結果として「着地後 5 秒誤認継続」 を抑止.
    """
    if cnn_after is None:
        return 0
    score = int(np.sum(candidate._grid == cnn_after._grid))
    for (r, c) in pattern_cells:
        cv = int(candidate.get(r, c))
        nv = int(cnn_after.get(r, c))
        if cv == nv:
            score += 50  # cycle 71l β1: 強化 (= 旧 5)
    return score


def _classify_diff_orientation(
    confirmed_before: Board, cnn_after: Board,
) -> str | None:
    """cnn_after の差分 cell 2 個の幾何配置から縦/横置きを判定する (案 B).

    confirmed_before で空だった cell が cnn_after で非空 (puyo 色) になっている
    cell を全列挙. 2 cell ちょうどあって幾何的に隣接していれば縦/横を確定.

    Returns:
        "vertical": 同列で row 隣接 (= 縦置き)
        "horizontal": 同 row で col 隣接 (= 横置き)
        None: 2 cell ちょうどでない or 隣接でない (= 確定不能)
    """
    diffs = _get_diff_cells(confirmed_before, cnn_after)
    if len(diffs) != 2:
        return None
    (r1, c1), (r2, c2) = diffs[0], diffs[1]
    if c1 == c2 and abs(r1 - r2) == 1:
        return "vertical"
    if r1 == r2 and abs(c1 - c2) == 1:
        return "horizontal"
    return None


def _get_diff_cells(
    confirmed_before: Board, cnn_after: Board,
) -> list[tuple[int, int]]:
    """confirmed_before で空、 cnn_after で非空 (= 色 puyo) となった cell 一覧."""
    diffs: list[tuple[int, int]] = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            base_v = int(confirmed_before.get(r, c))
            cur_v = int(cnn_after.get(r, c))
            if (
                base_v == COLOR_EMPTY
                and cur_v not in (COLOR_EMPTY, COLOR_UNKNOWN)
            ):
                diffs.append((r, c))
    return diffs


def _filter_bg_matched_cells(
    diff_cells: list[tuple[int, int]],
    frame_bgr: np.ndarray,
    region: "object",
    bg_fp: "object",
    bg_threshold: float | None,
) -> list[tuple[int, int]]:
    """diff cells のうち、 現 frame の HSV が背景 FP と一致する cells を除外.

    cycle 71v (2026-05-15): 背景を puyo 誤認した CNN の出力を物理推論段階で
    再フィルタする防御層。 image_reader 1st pass の bg_fp 早期 EMPTY 判定が
    cnn ぶれで通り抜けた場合の最後の砦。
    """
    try:
        import cv2 as _cv2
        from src.background_fingerprint import (
            CellFingerprint, DEFAULT_EMPTY_HSV_DISTANCE, is_empty_by_fp,
        )
    except Exception:
        return diff_cells
    threshold = (
        float(bg_threshold)
        if bg_threshold is not None
        else DEFAULT_EMPTY_HSV_DISTANCE
    )
    h, w = frame_bgr.shape[:2]
    hsv_full = _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2HSV)
    out: list[tuple[int, int]] = []
    for (r, c) in diff_cells:
        try:
            x1, y1, x2, y2 = region.cell_sample_rect(r, c)
        except Exception:
            out.append((r, c))
            continue
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(y1 + 1, min(int(y2), h))
        hsv_patch = hsv_full[y1:y2, x1:x2]
        if hsv_patch.size == 0:
            out.append((r, c))
            continue
        h_med = int(np.median(hsv_patch[:, :, 0]))
        s_med = int(np.median(hsv_patch[:, :, 1]))
        v_med = int(np.median(hsv_patch[:, :, 2]))
        cur_fp = CellFingerprint(h_med, s_med, v_med)
        # bg_fp cell_at は VISIBLE 行で受け取る (row 0=隠し段は HIDDEN_ROWS 引く)
        from src.board import HIDDEN_ROWS
        visible_row = r - HIDDEN_ROWS
        try:
            bg_cell = bg_fp.cell_at(visible_row, c)
        except Exception:
            out.append((r, c))
            continue
        if is_empty_by_fp(cur_fp, bg_cell, threshold=threshold):
            # 背景一致 → diff から除外
            continue
        out.append((r, c))
    return out


def _chain_count_of(
    candidate: Board, chain_sim: "object | None",
) -> int:
    """候補盤面の連鎖数を ChainSimulator で計算 (chain_sim=None なら 0)."""
    if chain_sim is None:
        return 0
    try:
        result = chain_sim.simulate(candidate)
        return int(result.chain_count)
    except Exception:
        return 0


def _apply_empty_hallucination_guard(
    base: Board,
    pattern: LandingPattern,
    diff_cells: list[tuple[int, int]],
    cnn_after: Board,
    color_first: int,
    color_second: int,
) -> Board:
    """guard_empty_hallucination 案 B: 観測セルのみ NEXT 色を書き、非観測 EMPTY セルは
    COLOR_UNKNOWN 留保した盤面を返す。

    設計変更 (案 B、2026-06-01):
    旧実装は「非 diff セルが明示 EMPTY な候補をスキップ → 全候補スキップ時 None 返却」
    だったが、正当な 2 ぷよ着地 (CNN が片側のみ観測) を丸ごと取りこぼし
    color→empty 破壊を増やす構造的副作用があった。

    新挙動:
    - diff_cells に含まれる (= CNN が観測した) セルには NEXT 色を書く。
    - 非 diff かつ cnn_after で明示 COLOR_EMPTY のセルには COLOR_UNKNOWN を書く
      (= hallucination 回避かつ commit refuse しない)。
    - 非 diff かつ cnn_after で COLOR_UNKNOWN のセルには NEXT 色で物理補完する
      (= CNN 不確実なので物理補完が妥当)。

    Args:
        base: 着地前の確定盤面。
        pattern: 着地パターン (2 cell 固定)。
        diff_cells: CNN 差分として検出されたセル座標リスト。
        cnn_after: 着地後の CNN 観測盤面。
        color_first: pattern.cells[0] へ書く予定の NEXT 色。
        color_second: pattern.cells[1] へ書く予定の NEXT 色。

    Returns:
        guard 適用後の盤面 (color_first/color_second を一部 COLOR_UNKNOWN で上書き)。
    """
    # まず通常通り materialize する
    result = materialize_pattern(base, pattern, color_first, color_second)
    diff_set = frozenset(diff_cells)
    planned = {pattern.cells[0]: color_first, pattern.cells[1]: color_second}
    for cell, _ in planned.items():
        if cell in diff_set:
            # diff セルは CNN が観測済 → NEXT 色をそのまま維持
            continue
        r, c = cell
        cnn_val = int(cnn_after.get(r, c))
        if cnn_val == COLOR_EMPTY:
            # 非観測かつ CNN が明示 EMPTY → COLOR_UNKNOWN 留保 (hallucination 防止)
            result.set(r, c, COLOR_UNKNOWN)
        # cnn_val == COLOR_UNKNOWN の場合は物理補完 (NEXT 色) をそのまま維持
    return result


def infer_placement(
    confirmed_before: Board,
    cnn_after: Board | None,
    next_pair: tuple[int, int] | None,
    chain_sim: "object | None" = None,
    score_delta_observed: int = 0,
    frame_bgr: np.ndarray | None = None,
    region: "object | None" = None,
    bg_fp: "object | None" = None,
    bg_threshold: float | None = None,
    guard_empty_hallucination: bool = False,
) -> Board | None:
    """物理推論主軸の置き判定 (= Phase 1 主 API, cycle 71 + 71b).

    cycle 71b: 案 A (= 連鎖整合性) + 案 B (= 縦/横幾何判定) を統合.
    cycle 71l (β2'): frame_bgr + region 渡しで着地 2 cell の HSV 距離で
    NEXT 色順序確定. 回転落下に対応、 CNN 完全誤認でも NEXT 色のどちらかに確定.

    Args:
        confirmed_before: TSUMO_FALL 開始前の確定盤面.
        cnn_after: 着地後の CNN 観測盤面. None 不可だが None なら物理パターン先頭.
        next_pair: 落下ツモ色 (top, bot).
        chain_sim: ChainSimulator. 渡されれば候補絞り込みに連鎖整合性を活用 (案 A).
        score_delta_observed: 着地直後の score 変化量. > 0 なら連鎖発生候補を優先
                              (案 A の連鎖整合性方向ヒント).
        frame_bgr: 着地後の 1920x1080 BGR frame. 渡されれば β2' で着地 2 cell の
                   HSV と NEXT 色 2 種類の距離で順序確定.
        region: BoardRegion. frame_bgr とセットで渡す.
        guard_empty_hallucination: True にすると、 案 B ガードを適用する。
            diff_cells に含まれる (= CNN が観測した) セルは NEXT 色を書く。
            非 diff かつ cnn_after で明示 COLOR_EMPTY のセルには COLOR_UNKNOWN を書き
            hallucination を防ぎつつ commit refuse しない。
            非 diff かつ COLOR_UNKNOWN セルは NEXT 色で物理補完 (従来通り)。
            default False = 従来挙動維持 (backwards compat)。

    Returns:
        着地後の確定盤面. 物理パターン 0 件 / next_pair 不明等で None.
    """
    if next_pair is None:
        return None
    if next_pair[0] in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN):
        return None
    if next_pair[1] in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN):
        return None
    patterns = enumerate_landing_patterns(confirmed_before)
    if not patterns:
        return None
    # diff_cells は cnn_after is not None の場合のみ設定される。
    # guard_empty_hallucination で参照するため事前に空リストで初期化。
    diff_cells: list[tuple[int, int]] = []
    # cycle 71v (2026-05-15): CNN diff の物理整合性ガードを厳格化.
    # 旧実装は CNN 観測が不完全 (= diff=0 or diff=1) でも infer を強制実行し、
    # arbitrary column の placement (= ゴースト) を生成する原因だった。
    # diff cells が valid pattern の cells subset と整合しない場合は None 返却 →
    # 呼び出し側で「commit せず TSUMO_FALL 維持」 を選択させる。
    if cnn_after is not None:
        diff_cells = _get_diff_cells(confirmed_before, cnn_after)
        # cycle 71v (2026-05-15): 背景 FP マッチ cells を diff から除外.
        # CNN が背景を puyo 誤認した cells は実際の placement の根拠にならない。
        # 設置時にも背景一致セルを reject する仕様 (= ユーザー要件)。
        if (
            bg_fp is not None
            and frame_bgr is not None
            and region is not None
            and diff_cells
        ):
            diff_cells = _filter_bg_matched_cells(
                diff_cells, frame_bgr, region, bg_fp, bg_threshold,
            )
        if len(diff_cells) == 0:
            # CNN が全く新規 cells を観測していない (or 全て背景) → commit refuse
            return None
        # diff cells が valid pattern の cells に subset として含まれる pattern のみ採用
        diff_set = frozenset(diff_cells)
        pos_filtered = [
            p for p in patterns if diff_set.issubset(frozenset(p.cells))
        ]
        if not pos_filtered:
            # CNN diff cells が valid pattern と整合しない → commit refuse
            return None
        patterns = pos_filtered
    # 全候補生成 (色配分の入替を含む)
    # cycle 71l (β2'): frame_bgr + region 渡し時、 各 pattern で着地 2 cell の HSV を
    # 直接見て NEXT 色順序を確定する. 色配分 2 通りではなく HSV 距離選択 1 通りに絞る.
    candidates: list[tuple[Board, LandingPattern]] = []
    use_hsv_classification = (
        frame_bgr is not None
        and region is not None
        and next_pair[0] != next_pair[1]
    )
    # guard_empty_hallucination: 候補生成前にパターン単位でスキップ判定。
    # cnn_after が None の場合は diff_cells が空リストのため guard は実質無効。
    # diff_cells は上記 if cnn_after is not None ブロックで設定済み。
    for p in patterns:
        if use_hsv_classification:
            (r1, c1_pos), (r2, c2_pos) = p.cells
            patch_a = _extract_cell_patch_from_frame(
                frame_bgr, region, r1, c1_pos,
            )
            patch_b = _extract_cell_patch_from_frame(
                frame_bgr, region, r2, c2_pos,
            )
            if patch_a is not None and patch_b is not None:
                color_a, color_b = _classify_next_pair_by_hsv(
                    patch_a, patch_b, next_pair,
                )
                # guard_empty_hallucination 案 B: 観測セルは NEXT 色、
                # 非観測 EMPTY セルは COLOR_UNKNOWN 留保にして materialize。
                if guard_empty_hallucination and cnn_after is not None:
                    board = _apply_empty_hallucination_guard(
                        confirmed_before, p, diff_cells,
                        cnn_after, color_a, color_b,
                    )
                else:
                    board = materialize_pattern(
                        confirmed_before, p, color_a, color_b,
                    )
                candidates.append((board, p))
                continue
            # HSV patch 取得失敗時は fallback (= 2 通り全 enumerate)
        for (c_a, c_b) in enumerate_color_assignments(p, next_pair):
            # guard_empty_hallucination 案 B: 観測セルは NEXT 色、
            # 非観測 EMPTY セルは COLOR_UNKNOWN 留保にして materialize。
            if guard_empty_hallucination and cnn_after is not None:
                board = _apply_empty_hallucination_guard(
                    confirmed_before, p, diff_cells,
                    cnn_after, c_a, c_b,
                )
            else:
                board = materialize_pattern(confirmed_before, p, c_a, c_b)
            candidates.append((board, p))
    # 全候補がガードでスキップされた場合は commit refuse (= TSUMO_FALL 維持)
    if not candidates:
        return None
    if cnn_after is None:
        return candidates[0][0]
    # CNN 一致度で score, 案 A の連鎖整合性で tie-break.
    # score_delta_observed > 0 → 連鎖発生候補を優先 (chain_count >= 1 にボーナス).
    # score_delta_observed == 0 → 連鎖なし候補を優先 (chain_count == 0 にボーナス).
    best_score = -10**9
    best_board: Board | None = None
    for (cand_board, cand_pattern) in candidates:
        s = _diff_match_score(cand_board, cnn_after, cand_pattern.cells)
        # 案 A tie-break: 連鎖整合性ボーナス.
        if chain_sim is not None:
            cc = _chain_count_of(cand_board, chain_sim)
            if score_delta_observed > 0:
                # 連鎖が起きている方向の証拠 → chain 候補を優先
                s += 3 * cc
            else:
                # 連鎖が起きていない方向の証拠 → chain なし候補を優先
                if cc == 0:
                    s += 3
        if s > best_score:
            best_score = s
            best_board = cand_board
    return best_board


# cycle 71c: prev_confirmed → new_confirmed で puyo cell 数がこれ以上増えていたら
# 「STABLE 認識欠落の物理推論補填」 とみなして連鎖判定をスキップ.
# A=hit 17 件分析で観測した急増幅 (+11〜+17 cells) を上限としてカバー.
# 通常の 1 ツモ着地は +2 cells (= 縦/横置き)、 隠し段含めても +4 cells 程度.
LARGE_ADD_GUARD_CELLS: int = 6


def _count_non_empty_cells(board: Board) -> int:
    """COLOR_EMPTY / COLOR_UNKNOWN 以外の cell 数を返す."""
    cnt = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            v = int(board.get(r, c))
            if v != COLOR_EMPTY and v != COLOR_UNKNOWN:
                cnt += 1
    return cnt


def resolve_after_placement(
    new_confirmed: Board,
    chain_sim: "object",
    prev_confirmed: Board | None = None,
    large_add_threshold: int = LARGE_ADD_GUARD_CELLS,
    score_delta_observed: int | None = None,
) -> tuple[Board, int]:
    """着地後盤面で即時連鎖判定 (= 仮説 B 解消、 cycle 71).

    ChainSimulator を即時呼び出し、 連鎖発生時は final_board に切替.
    score_delta や VideoChainTracker の puyo 数差分検出を**待たない**.

    cycle 71c: prev_confirmed を受け取り、 new_confirmed との puyo cell 数差分が
    large_add_threshold 超なら「STABLE 認識欠落の物理推論補填」 と判断し、
    連鎖判定を skip (= (new_confirmed, 0) 即時 return). A=hit (α) ケース対策.

    cycle 71n (案 ε, 2026-05-13): score_delta_observed == 0 なら chain_count >= 1
    でも final_board を採用せず new_confirmed のまま返す. 「ChainSimulator が連鎖
    発生と判定したが、 実際の score は動いていない = 偽陽性」 を抑止する.
    11 秒一瞬 EMPTY 現象 (= 連鎖偽判定で多数 cells EMPTY 化) への対策.

    Args:
        new_confirmed: 着地後の確定盤面 (= infer_placement の出力).
        chain_sim: src.chain.ChainSimulator インスタンス.
        prev_confirmed: 直前 STABLE の確定盤面. None なら従来通り chain_sim を呼ぶ.
        large_add_threshold: prev → new で許容する puyo cell 数増加幅.
        score_delta_observed: 直近 frame の score 増加量. 0 なら連鎖発生
            偽陽性として final_board 採用 skip.

    Returns:
        (final_board, chain_count).
        連鎖なし or guard 発動なら (new_confirmed, 0).
        連鎖ありなら (final_board, chain_count).
    """
    if prev_confirmed is not None:
        prev_cnt = _count_non_empty_cells(prev_confirmed)
        new_cnt = _count_non_empty_cells(new_confirmed)
        if new_cnt - prev_cnt > large_add_threshold:
            return new_confirmed, 0
    try:
        result = chain_sim.simulate(new_confirmed)
    except Exception:
        return new_confirmed, 0
    if result.chain_count < 1 or result.final_board is None:
        return new_confirmed, 0
    # cycle 71n (案 ε): score 動いていない時点で連鎖発生は偽陽性とみなす.
    # 連鎖 1 連鎖でも最低 +40 点入るので、 score_delta_observed が 0 なら明確な誤判定.
    if score_delta_observed == 0:
        return new_confirmed, 0
    # 浮きぷよ filter (= 物理整合性確認、 念のため)
    from src.board_state_machine import _apply_gravity_filter
    final = result.final_board.copy()
    _apply_gravity_filter(final)
    return final, int(result.chain_count)


__all__ = [
    "LandingPattern",
    "enumerate_landing_patterns",
    "materialize_pattern",
    "enumerate_color_assignments",
    "infer_placement",
    "resolve_after_placement",
]
