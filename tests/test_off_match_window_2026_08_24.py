"""試合外区間の判定テスト (2026-08-24、W37)。

判定規則は user 伝授 (`src/off_match_window.py` の docstring 参照):
    ネクストが動かない + スコアが動かない + スコアが 0 でない
    が 2 秒続いたら試合外。検知点から遡って 2 秒も試合外。
"""
from __future__ import annotations

from src.off_match_window import (
    OFF_MATCH_BACKTRACK_SEC,
    OFF_MATCH_FREEZE_SEC,
    SideObservation,
    find_off_match_spans,
    is_in_spans,
)

FPS = 30.0
FRAME = 1.0 / FPS


def _obs(rows: list[tuple[float, int | None, object]]) -> list[SideObservation]:
    return [SideObservation(t_sec=t, score=s, next_key=n) for t, s, n in rows]


def _frames(t0: float, n: int, score: int | None, next_key: object):
    return [(t0 + i * FRAME, score, next_key) for i in range(n)]


def test_frozen_over_two_seconds_is_off_match() -> None:
    """スコアとネクストが 2 秒以上凍結 → 試合外。"""
    rows = _frames(10.0, 120, 58503, "RG")  # 4 秒凍結
    spans = find_off_match_spans(_obs(rows))
    assert len(spans) == 1
    assert spans[0][1] - spans[0][0] >= OFF_MATCH_FREEZE_SEC


def test_backtrack_extends_span_before_detection() -> None:
    """判定成立時、遡って 2 秒ぶんも試合外に含める。

    凍結の検知には 2 秒かかるので、その 2 秒も盤面は消えている。
    物差しはオフライン処理なので先読み (遡り) が許される。
    """
    rows = _frames(10.0, 120, 58503, "RG")
    spans = find_off_match_spans(_obs(rows))
    assert spans[0][0] == 10.0 - OFF_MATCH_BACKTRACK_SEC


def test_chain_is_not_off_match_because_score_moves() -> None:
    """**連鎖中は試合外にしない。** ネクストは止まるがスコアが増え続ける。

    これがこの規則の要。「ネクストが止まる」だけでは連鎖と区別できない。
    """
    rows = []
    for i in range(120):  # 4 秒、ネクストは固定、スコアだけ増える
        rows.append((10.0 + i * FRAME, 1000 + i * 40, "RG"))
    assert find_off_match_spans(_obs(rows)) == []


def test_score_zero_is_not_off_match() -> None:
    """スコア 0 (試合開始前) は試合外にしない。"""
    rows = _frames(0.0, 150, 0, "RG")  # 5 秒、スコア 0 のまま
    assert find_off_match_spans(_obs(rows)) == []


def test_score_none_is_not_off_match() -> None:
    """スコアが読めていないフレームは判定に使わない。"""
    rows = _frames(0.0, 150, None, "RG")
    assert find_off_match_spans(_obs(rows)) == []


def test_short_freeze_under_two_seconds_is_ignored() -> None:
    """2 秒未満の凍結は試合外にしない (通常プレイの間合い)。"""
    rows = _frames(10.0, 30, 500, "RG")          # 1 秒凍結
    rows += _frames(11.0, 30, 540, "BY")         # 動いた
    assert find_off_match_spans(_obs(rows)) == []


def test_next_moving_breaks_the_freeze() -> None:
    """スコアが同じでもネクストが動いていれば試合中。"""
    rows = []
    for i in range(120):
        rows.append((10.0 + i * FRAME, 500, f"pair{i // 15}"))  # 0.5 秒ごとに変化
    assert find_off_match_spans(_obs(rows)) == []


def test_overlapping_spans_are_merged() -> None:
    """遡りで重なった区間は 1 つに結合する。"""
    rows = _frames(10.0, 90, 500, "RG")            # 3 秒凍結
    rows += [(13.0, 540, "BY")]                    # 一瞬動く
    rows += _frames(13.1, 90, 540, "BY")           # また 3 秒凍結
    spans = find_off_match_spans(_obs(rows))
    assert len(spans) == 1, f"結合されていない: {spans}"


def test_is_in_spans() -> None:
    spans = [(10.0, 15.0), (20.0, 25.0)]
    assert is_in_spans(12.0, spans)
    assert is_in_spans(10.0, spans)
    assert not is_in_spans(17.0, spans)


def test_empty_input() -> None:
    assert find_off_match_spans([]) == []


def test_v51_real_case_covers_contamination_window() -> None:
    """v51 の実測パターンを再現し、汚染窓を覆えることを確かめる。

    実測 (`logs/_probe_offmatch_rule_v51_2026-08-24.jsonl`):
    1P は t=56.33〜63.23 で凍結し、汚染窓は t=57.5〜63.5。
    死亡は t≈58.5、MENU 突入は t=63.3。
    """
    rows = []
    # 生きたプレイ (0.5 秒ごとにネクストが変わる)
    for i in range(int(6.0 * FPS)):
        t = 50.0 + i * FRAME
        rows.append((t, 40000 + i * 10, f"pair{i // 15}"))
    # t=56.33 以降は凍結 (スコアもネクストも止まる)
    for i in range(int(7.0 * FPS)):
        rows.append((56.33 + i * FRAME, 58503, "frozen"))
    spans = find_off_match_spans(_obs(rows))
    assert len(spans) == 1, f"区間が 1 つでない: {spans}"
    a, b = spans[0]
    assert a <= 57.5 and b >= 63.2, f"汚染窓 57.5-63.5 を覆えていない: {spans}"


def test_freeze_ending_with_score_reset_is_still_detected() -> None:
    """凍結がスコア 0 リセットで終わる場合も検出する (v51 で実際に踏んだ穴)。

    試合が終わるとスコアは 0 にリセットされる。実測 (v51 1P):

        t=56.333  score=58503 next=(4,1)  ← ここから凍結
        t=63.233  score=0                 ← 試合終了でリセット

    スコア 0 のフレームで蓄積中の凍結を閉じずに捨てる実装だと、
    **本命の区間を丸ごと取り逃す**。実際に初版で取り逃していた。
    """
    rows = _frames(56.333, int(6.9 * FPS), 58503, "(4, 1)")
    rows.append((63.233, 0, "(4, 1)"))
    spans = find_off_match_spans(_obs(rows))
    assert len(spans) == 1, f"スコア 0 で終わる凍結を検出できていない: {spans}"
    a, b = spans[0]
    assert a <= 57.5 and b >= 63.2, f"汚染窓 57.5-63.2 を覆えていない: {spans}"
