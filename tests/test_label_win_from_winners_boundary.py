"""
scripts/label_win_from_winners.py の find_winner_for_t 境界挙動 回帰テスト。

## 背景 (2026-07-28 監査)
study CSV はジョブ窓 (本体0-300s / _gap 300-900s / _mid 1200-1560s) 単位で
並列収集されており、実試合はこれらの窓境界をまたいで存在しうる。
find_winner_for_t() は winners JSON の絶対時刻区間 [start_sec, end_sec) の
半開区間マッチのみで勝者を決めており、game_idx には一切依存しない設計
(label_win_from_winners.py:71-80)。

この境界仕様 (半開区間・区間外はNone) を守るテストが存在しなかったため、
本ファイルで固定化する。
"""
from __future__ import annotations

from scripts.label_win_from_winners import find_winner_for_t


def _make_games() -> list[dict]:
    """隣接する2試合区間 (境界=300.0秒) のダミー games リストを作る。"""
    return [
        {"start_sec": 0.0, "end_sec": 300.0, "winner": "1P"},
        {"start_sec": 300.0, "end_sec": 600.0, "winner": "2P"},
    ]


# =============================================================================
# ジョブ窓の継ぎ目 (t_sec=300.0 付近) での境界挙動
# =============================================================================


def test_区間内の時刻は正しい試合のwinnerを返す() -> None:
    """区間境界の手前 (299.9秒) は前の試合(1P勝ち)に属すること。"""
    games = _make_games()
    assert find_winner_for_t(games, 299.9) == "1P"


def test_区間境界ちょうどの時刻は次の試合に属する_半開区間() -> None:
    """
    区間はstart_sec側を含みend_sec側を含まない半開区間 [s, e) であるため、
    t_sec=300.0 ちょうど (前試合のend_sec=次試合のstart_sec) は
    前試合(1P勝ち)ではなく次試合(2P勝ち)に属すること。
    """
    games = _make_games()
    assert find_winner_for_t(games, 300.0) == "2P"


def test_区間境界の直後は次の試合のwinnerを返す() -> None:
    """境界のわずかに後 (300.1秒) は次の試合(2P勝ち)に属すること。"""
    games = _make_games()
    assert find_winner_for_t(games, 300.1) == "2P"


def test_end_sec直前は含まれend_sec自体は含まれない() -> None:
    """
    半開区間の反対側の端点も確認する: end_sec直前(599.9秒)は含まれ、
    end_sec自体(600.0秒、どの試合にも属さない)はNoneになること。
    """
    games = _make_games()
    assert find_winner_for_t(games, 599.9) == "2P"
    assert find_winner_for_t(games, 600.0) is None


# =============================================================================
# どの区間にも属さない時刻・空リスト
# =============================================================================


def test_全区間の外側の時刻はNoneを返す() -> None:
    """開始前 (負値) や終了後の時刻は、どの試合にも属さずNoneになること。"""
    games = _make_games()
    assert find_winner_for_t(games, -1.0) is None
    assert find_winner_for_t(games, 700.0) is None


def test_試合区間の間の空白_gap窓相当は属する試合がなくNoneを返す() -> None:
    """
    実運用では本体窓(0-300s)とmid窓(1200-1560s)の間の_gap相当区間は
    実試合が記録されていないことがある。そのような「どの試合区間にも
    含まれない」時刻ではwinnerが決まらずNoneになること
    (=game_idxに基づく代替判定などにフォールバックしないこと)。
    """
    games = [
        {"start_sec": 0.0, "end_sec": 300.0, "winner": "1P"},
        {"start_sec": 1200.0, "end_sec": 1560.0, "winner": "2P"},
    ]
    assert find_winner_for_t(games, 500.0) is None


def test_games空リストならNoneを返す() -> None:
    """winners JSON に該当動画の試合情報が無い(空リスト)場合はNoneになること。"""
    assert find_winner_for_t([], 100.0) is None


def test_winnerがNoneの試合区間内ならNoneを返す() -> None:
    """区間には属するが winner が None (未確定) の場合、Noneがそのまま返ること。"""
    games = [{"start_sec": 0.0, "end_sec": 300.0, "winner": None}]
    assert find_winner_for_t(games, 100.0) is None
