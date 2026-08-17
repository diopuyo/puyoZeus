"""W25根治 第3弾・最終 (2026-08-18): CNN観測入力段の会計整合フィルタ テスト。

docs/KNOWN_WEAKNESSES.md W25、
data/verify/diag_c13c22_recheck_2026-08-17/w25_guard_gap.md 参照。

検証観点:
- 非空色セルへの9書込み + クレジット不足 → 直近安定色へ差し替え (核心)
- 非空色セルへの9書込み + クレジット十分 → 素通し (会計が裏付ける場合は許容)
- 空セルへの9書込み (正規のおじゃま着弾経路) → クレジットに関わらず常に素通し
  (design要件(c)の固定テスト)
- 9以外の値への変化 → 常に素通し (フィルタ対象外)
- 盤面全体適用 (apply_ojama_write_accounting_filter) の入力非破壊・対象外セル維持
"""
from __future__ import annotations

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_RED,
    HIDDEN_ROWS,
    Board,
)
from src.ojama_write_accounting import (
    OJAMA_REJECT_TIMEOUT_SEC,
    apply_ojama_write_accounting_filter,
    filter_ojama_write_by_accounting,
)


# ---------------------------------------------------------------------------
# filter_ojama_write_by_accounting (1セル分の純関数)
# ---------------------------------------------------------------------------


def test_filter_blocks_color_to_ojama_when_credit_zero():
    """核心: 非空色セル + 9書込み + クレジット0 → 直近安定色へ差し替え。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_RED, new_cnn_value=COLOR_OJAMA,
        column_pending_ojama_credit=0,
    )
    assert out == COLOR_RED


def test_filter_blocks_color_to_ojama_when_credit_negative():
    """クレジットが負 (会計上あり得ないが防御的に) でも同様に差し替える。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_GREEN, new_cnn_value=COLOR_OJAMA,
        column_pending_ojama_credit=-1,
    )
    assert out == COLOR_GREEN


def test_filter_blocks_color_to_ojama_even_when_credit_positive():
    """実測に基づく設計修正 (2026-08-18、モジュール docstring参照): 当初は
    credit>0 で colored→9 を素通しする設計だったが、c13 実測で score OCR
    異常由来の巨大クレジット (floor(216/6)=36) がこの素通しを悪用し
    対象9セルが解消できないことが判明した。ぷよぷよのルール上おじゃまは
    空セルにのみ着弾するため、credit の大小に関わらず非空色セルへの
    9書込みは常に棄却するべき (物理的に説明不可能)。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_RED, new_cnn_value=COLOR_OJAMA,
        column_pending_ojama_credit=1,
    )
    assert out == COLOR_RED


def test_filter_blocks_color_to_ojama_even_when_credit_very_large():
    """credit が現実的にあり得ない大きさ (score OCR 異常由来の
    サニティ上限相当) でも棄却する (c13 実測の再現、回帰テスト)。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_RED, new_cnn_value=COLOR_OJAMA,
        column_pending_ojama_credit=36,
    )
    assert out == COLOR_RED


def test_filter_passthrough_empty_to_ojama_with_zero_credit():
    """設計要件(c)固定テスト: 空セルへの9書込み (正規のおじゃま着弾経路) は
    クレジット0でもフィルタ対象外 = 常に素通しする。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_EMPTY, new_cnn_value=COLOR_OJAMA,
        column_pending_ojama_credit=0,
    )
    assert out == COLOR_OJAMA


def test_filter_passthrough_empty_to_ojama_with_negative_credit():
    """空セル起点は負クレジットでも素通し (対象外の境界を厳密に確認)。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_EMPTY, new_cnn_value=COLOR_OJAMA,
        column_pending_ojama_credit=-5,
    )
    assert out == COLOR_OJAMA


def test_filter_passthrough_non_ojama_new_value():
    """9 以外への変化 (色→色、色→空 等) はクレジットに関わらず常に素通し。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_RED, new_cnn_value=COLOR_GREEN,
        column_pending_ojama_credit=0,
    )
    assert out == COLOR_GREEN


def test_filter_passthrough_color_to_empty():
    """色→空 (連鎖消去等の正当な物理事象) はフィルタ対象外。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_RED, new_cnn_value=COLOR_EMPTY,
        column_pending_ojama_credit=0,
    )
    assert out == COLOR_EMPTY


def test_filter_noop_when_already_ojama():
    """すでにおじゃまのセルへの再観測 (9→9) は差分なし、素通し。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_OJAMA, new_cnn_value=COLOR_OJAMA,
        column_pending_ojama_credit=0,
    )
    assert out == COLOR_OJAMA


# ---------------------------------------------------------------------------
# apply_ojama_write_accounting_filter (盤面全体適用)
# ---------------------------------------------------------------------------


def test_apply_filter_rejects_spurious_ojama_on_colored_cell():
    """盤面全体適用: memory に登録された色セルへの9書込みがクレジット0で
    棄却される (対象セル以外は無変化)。"""
    cnn = Board()
    cnn.set(9, 1, COLOR_OJAMA)  # 雲混入を模擬
    cnn.set(5, 3, COLOR_GREEN)  # 無関係セル (対象外)
    memory = {(9, 1): COLOR_RED, (5, 3): COLOR_GREEN}

    out = apply_ojama_write_accounting_filter(cnn, memory, column_pending_ojama_credit=0)

    assert int(out.get(9, 1)) == COLOR_RED, "雲混入セルは直近安定色 (赤) に差し替えられるべき"
    assert int(out.get(5, 3)) == COLOR_GREEN, "無関係セルは無変化のはず"


def test_apply_filter_does_not_mutate_input_board():
    """入力 cnn_board 自体は変更しない (純関数の非破壊性)。"""
    cnn = Board()
    cnn.set(9, 1, COLOR_OJAMA)
    memory = {(9, 1): COLOR_RED}

    out = apply_ojama_write_accounting_filter(cnn, memory, column_pending_ojama_credit=0)

    assert int(cnn.get(9, 1)) == COLOR_OJAMA, "入力盤面は不変であるべき"
    assert int(out.get(9, 1)) == COLOR_RED


def test_apply_filter_untracked_cell_defaults_to_empty_and_passes_through():
    """memory に未登録のセル (まだ安定観測が無い) は COLOR_EMPTY 扱いとなり、
    9書込みは (空セル起点として) 常に素通しされる。"""
    cnn = Board()
    cnn.set(3, 2, COLOR_OJAMA)
    memory: dict[tuple[int, int], int] = {}  # 空

    out = apply_ojama_write_accounting_filter(cnn, memory, column_pending_ojama_credit=0)

    assert int(out.get(3, 2)) == COLOR_OJAMA, (
        "未観測セルは EMPTY 扱い→フィルタ対象外で素通しされるべき"
    )


def test_apply_filter_rejects_colored_cell_ojama_even_with_positive_credit():
    """実測に基づく設計修正 (2026-08-18): 盤面全体適用でもクレジットの
    大小に関わらず colored→9 は棄却される (c13 実測の再現)。"""
    cnn = Board()
    cnn.set(9, 1, COLOR_OJAMA)
    memory = {(9, 1): COLOR_RED}

    out = apply_ojama_write_accounting_filter(cnn, memory, column_pending_ojama_credit=36)

    assert int(out.get(9, 1)) == COLOR_RED


def test_apply_filter_full_board_no_spurious_changes_when_no_ojama_present():
    """おじゃま書込みが一切無い盤面では出力が入力と完全一致する。"""
    cnn = Board()
    cnn.set(10, 0, COLOR_RED)
    cnn.set(11, 5, COLOR_GREEN)
    memory = {(10, 0): COLOR_RED, (11, 5): COLOR_GREEN}

    out = apply_ojama_write_accounting_filter(cnn, memory, column_pending_ojama_credit=0)

    for r in range(HIDDEN_ROWS, BOARD_ROWS):
        for c in range(BOARD_COLS):
            assert int(out.get(r, c)) == int(cnn.get(r, c))


# ---------------------------------------------------------------------------
# W25固着対策 (2026-08-18): 連続9観測タイムアウトによる棄却解除。
# data/verify/diag_w25_regression2_2026-08-18/ 参照 (c10/c109 永久固着の
# 直接対策)。
# ---------------------------------------------------------------------------


def test_filter_rejects_when_duration_below_timeout():
    """タイムアウト未到達 (duration < OJAMA_REJECT_TIMEOUT_SEC) では従来通り
    棄却する (固着対策導入前と bit-identical、回帰確認)。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_RED, new_cnn_value=COLOR_OJAMA,
        column_pending_ojama_credit=0,
        consecutive_raw9_duration_sec=OJAMA_REJECT_TIMEOUT_SEC - 0.01,
    )
    assert out == COLOR_RED


def test_filter_accepts_when_duration_reaches_timeout():
    """核心: duration が OJAMA_REJECT_TIMEOUT_SEC に達したら
    直近安定色メモリの陳腐化を疑い9を受理する (永久固着防止)。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_RED, new_cnn_value=COLOR_OJAMA,
        column_pending_ojama_credit=0,
        consecutive_raw9_duration_sec=OJAMA_REJECT_TIMEOUT_SEC,
    )
    assert out == COLOR_OJAMA


def test_filter_accepts_when_duration_exceeds_timeout():
    """duration がタイムアウトを大幅に超えても受理し続ける (タイムアウト後
    に再度棄却へ戻ることはない)。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_RED, new_cnn_value=COLOR_OJAMA,
        column_pending_ojama_credit=0,
        consecutive_raw9_duration_sec=OJAMA_REJECT_TIMEOUT_SEC + 10.0,
    )
    assert out == COLOR_OJAMA


def test_filter_duration_default_zero_is_backward_compatible():
    """consecutive_raw9_duration_sec を渡さない既存呼出しは既定 0.0 となり、
    タイムアウト機能が無効 (= 従来の棄却挙動と bit-identical)。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_RED, new_cnn_value=COLOR_OJAMA,
        column_pending_ojama_credit=0,
    )
    assert out == COLOR_RED


def test_filter_duration_does_not_affect_empty_cell_passthrough():
    """空セル起点は duration の大小に関わらず (そもそも判定に入る前に)
    常に素通しする (設計要件(c)はタイムアウト機構追加後も不変)。"""
    out = filter_ojama_write_by_accounting(
        prev_stable_color=COLOR_EMPTY, new_cnn_value=COLOR_OJAMA,
        column_pending_ojama_credit=0,
        consecutive_raw9_duration_sec=0.0,
    )
    assert out == COLOR_OJAMA


def test_apply_filter_with_duration_dict_below_timeout_rejects():
    """盤面全体適用: duration 辞書がタイムアウト未到達なら棄却される。"""
    cnn = Board()
    cnn.set(9, 1, COLOR_OJAMA)
    memory = {(9, 1): COLOR_RED}
    duration_by_cell = {(9, 1): OJAMA_REJECT_TIMEOUT_SEC - 0.5}

    out = apply_ojama_write_accounting_filter(
        cnn, memory, column_pending_ojama_credit=0,
        consecutive_raw9_duration_by_cell=duration_by_cell,
    )

    assert int(out.get(9, 1)) == COLOR_RED


def test_apply_filter_with_duration_dict_at_timeout_accepts():
    """盤面全体適用: duration 辞書がタイムアウトに達したセルのみ受理し、
    他のセルは無関係 (対象セル以外は無変化)。"""
    cnn = Board()
    cnn.set(9, 1, COLOR_OJAMA)  # タイムアウト到達 → 受理
    cnn.set(10, 2, COLOR_OJAMA)  # タイムアウト未到達 → 棄却
    memory = {(9, 1): COLOR_RED, (10, 2): COLOR_GREEN}
    duration_by_cell = {
        (9, 1): OJAMA_REJECT_TIMEOUT_SEC,
        (10, 2): 0.1,
    }

    out = apply_ojama_write_accounting_filter(
        cnn, memory, column_pending_ojama_credit=0,
        consecutive_raw9_duration_by_cell=duration_by_cell,
    )

    assert int(out.get(9, 1)) == COLOR_OJAMA, "タイムアウト到達セルは受理されるべき"
    assert int(out.get(10, 2)) == COLOR_GREEN, "タイムアウト未到達セルは棄却されたままのはず"


def test_apply_filter_duration_dict_none_is_backward_compatible():
    """consecutive_raw9_duration_by_cell を渡さない既存呼出し (None) は
    全セル duration=0.0 扱いとなり、タイムアウト機能が無効
    (= 従来の棄却挙動と bit-identical)。"""
    cnn = Board()
    cnn.set(9, 1, COLOR_OJAMA)
    memory = {(9, 1): COLOR_RED}

    out = apply_ojama_write_accounting_filter(cnn, memory, column_pending_ojama_credit=0)

    assert int(out.get(9, 1)) == COLOR_RED


def test_apply_filter_duration_dict_missing_cell_defaults_to_zero():
    """duration_by_cell に登録が無いセルは 0.0 扱い (= タイムアウト未到達、
    棄却ロジックのみ適用)。"""
    cnn = Board()
    cnn.set(9, 1, COLOR_OJAMA)
    memory = {(9, 1): COLOR_RED}

    out = apply_ojama_write_accounting_filter(
        cnn, memory, column_pending_ojama_credit=0,
        consecutive_raw9_duration_by_cell={},  # (9,1) 未登録
    )

    assert int(out.get(9, 1)) == COLOR_RED
