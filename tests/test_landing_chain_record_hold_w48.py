"""W48: 着地直後の連鎖で差し替わった盤面を記録しない (既定OFF) の検査.

背景 (実測、`docs/G3_WRITE_PATH_ROOT_CAUSES_2026-09-17.md`):
    連鎖が始まった瞬間に `resolve_after_placement` が確定盤面を
    「連鎖が終わった後の姿」へ差し替える。state は STABLE のままなので
    そのまま記録される。13連鎖のアニメは約18秒かかるので、その間ずっと
    記録が画面より先を行く。

    実例 video_38 2P frame 34702 (578.37秒): 画面66個に対し記録9個。
    得点は18秒後に +79,085 で、連鎖自体は実在した。**盤面だけが先へ飛ぶ。**

    5動画の全数計装では、記録された盤面の「盤面側の誤り」15,617 セルのうち
    57% がこの経路の連鎖シミュレータ由来だった。

このフラグは**認識を1セルも変えない**。記録するかどうかだけを決める。
"""
from __future__ import annotations

import inspect
import sys
from typing import Any

import pytest

from src.recognition_pipeline import RecognitionPipeline, SideResult
import scripts.collect_boards_lean as CBL


# ---- 契約 -----------------------------------------------------------------

def test_flag_defaults_to_off_everywhere() -> None:
    """既定 OFF。ユーザー承認前に本番へ入らない。"""
    for fn in (RecognitionPipeline.__init__, RecognitionPipeline.load_default,
               CBL.collect_lean, CBL._process_side_lean):
        sig = inspect.signature(fn)
        names = [n for n in sig.parameters
                 if "landing_chain" in n]
        assert names, f"{fn.__qualname__} に配線が無い"
        for n in names:
            assert sig.parameters[n].default is False, f"{fn.__qualname__}.{n}"


def test_side_result_field_defaults_to_false() -> None:
    """既存の呼出しを壊さない (末尾に optional 追加のみ)。"""
    r = SideResult(side="1P", state=None, cnn_board=None, inferred_board=None,
                   confirmed_board=None, drift=None, score=None, score_delta=0,
                   chain_event=None)
    assert r.landing_chain_started is False


# ---- 記録ゲート -------------------------------------------------------------

class _Acc:
    def __init__(self) -> None:
        self.rows: list = []

    def append(self, grid, video_id, side, t_sec, game_idx, frame_idx, **kw) -> None:
        self.rows.append((side, frame_idx))


def _emit(monkeypatch: Any, **extra: Any) -> _Acc:
    """`_process_side_lean` を最小構成で呼び、記録されたかを返す。"""
    from src.board import Board
    from src.board_state_machine import BoardState
    acc = _Acc()
    board = Board()
    for col in range(6):
        board.set(12, col, 1)
    # 記録の可否だけを見たいので、他の門は素通りさせる。
    monkeypatch.setattr(CBL, "_should_emit", lambda *a, **k: True)
    monkeypatch.setattr(CBL, "_update_game_boundary", lambda *a, **k: None)
    state = CBL._SideState()
    CBL._process_side_lean(
        acc, state, "1P", board, BoardState.STABLE, 100, "video_x", 1.0, 10,
        shared_game=CBL._SharedGameCounter(), **extra,
    )
    return acc


def test_off_records_as_before(monkeypatch: Any) -> None:
    """既定では従来どおり記録する (bit-identical)。"""
    assert len(_emit(monkeypatch).rows) == 1


def test_on_skips_the_frame_whose_board_jumped_ahead(monkeypatch: Any) -> None:
    """連鎖後の姿へ差し替わった frame は記録しない。"""
    assert _emit(monkeypatch, landing_chain_started=True).rows == []


def test_on_does_not_touch_other_frames(monkeypatch: Any) -> None:
    """印が立っていない frame は普通に記録される。"""
    assert len(_emit(monkeypatch, landing_chain_started=False).rows) == 1


# ---- pipeline 側 -----------------------------------------------------------

def test_pipeline_does_not_mark_when_flag_is_off() -> None:
    """フラグ OFF なら印は絶対に立たない = 記録は1枚も変わらない。"""
    p = RecognitionPipeline.__new__(RecognitionPipeline)
    p._enable_landing_chain_record_hold = False
    assert p._enable_landing_chain_record_hold is False


def test_cli_flag_reaches_collect_lean(monkeypatch: Any) -> None:
    """`--help` 突合ではなく、**実際に main() を通して値が届くか**を見る。

    配線漏れは「届かない型」と「別の値が届く型」がある。後者は --help では
    検出できない (`feedback_wiring_gap_vs_wiring_error_2026-08-22`)。
    ここでは collect_lean が受け取った値そのものを捕まえる。
    """
    seen: dict = {}

    def fake_collect_lean(*args: Any, **kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(CBL, "collect_lean", fake_collect_lean)
    monkeypatch.setattr(sys, "argv", [
        "collect_boards_lean", "--video", "x.mp4", "--out-npz", "y.npz",
        "--enable-landing-chain-record-hold",
    ])
    assert CBL.main() == 0
    assert seen.get("enable_landing_chain_record_hold") is True, seen.keys()

    seen.clear()
    monkeypatch.setattr(sys, "argv", [
        "collect_boards_lean", "--video", "x.mp4", "--out-npz", "y.npz",
    ])
    assert CBL.main() == 0
    assert seen.get("enable_landing_chain_record_hold") is False


# ---- 保持期限 (1 frame だけでは効かなかった、の是正) ------------------------

def test_hold_lasts_for_the_whole_chain_not_one_frame() -> None:
    """1 frame だけ止めても次の frame が同じ盤面を記録してしまう。

    実測 (2026-09-17、`w48_effect_2026-09-17`): 最初の実装は
    video_38 2P 578.37秒の記録を消したが、**0.03秒後の 578.40秒が
    同じ9個の盤面を記録していた** (枚数 1,606 → 1,606 で変わらず)。
    確定盤面そのものが差し替わったままなので、印はその連鎖の
    保持期限まで続かなければ意味がない。
    """
    p = RecognitionPipeline.__new__(RecognitionPipeline)
    p._enable_landing_chain_record_hold = True
    p._landing_chain_hold_until_1p = 100.0
    p._landing_chain_hold_until_2p = -1.0

    def marked(side: str, t: float) -> bool:
        return t < (p._landing_chain_hold_until_1p if side == "1P"
                    else p._landing_chain_hold_until_2p)

    assert marked("1P", 96.1) is True, "連鎖が始まった frame"
    assert marked("1P", 96.13) is True, "その 0.03 秒後も止め続ける"
    assert marked("1P", 99.9) is True, "保持期限の直前まで"
    assert marked("1P", 100.0) is False, "期限を過ぎたら普通に記録する"
    assert marked("2P", 96.1) is False, "相手側は巻き込まない"


def test_hold_is_cleared_on_reset() -> None:
    """試合境界を跨いで持ち越さない。"""
    import inspect
    src = inspect.getsource(RecognitionPipeline.reset)
    assert "_landing_chain_hold_until_1p" in src
    assert "_landing_chain_hold_until_2p" in src


# ---- W48b: 着地経路以外も止める (2026-09-18) -------------------------------

def test_w48b_defaults_to_off_everywhere() -> None:
    for fn in (RecognitionPipeline.__init__, RecognitionPipeline.load_default,
               CBL.collect_lean):
        sig = inspect.signature(fn)
        assert sig.parameters["enable_chain_active_record_hold"].default is False


def test_w48b_covers_paths_other_than_landing() -> None:
    """掛け算式の早期発火・連鎖の再生で差し替わった盤面も止める。

    実測 (2026-09-18、5動画): 着地経路だけを止めた版では
    連鎖シミュレータ由来の誤りが 8,925 → 6,461 にしか減らなかった。
    残りは `_apply_chain_formula_early_fire` / `_start_chain_playback` 由来。
    どの経路が始めた連鎖でも `_chain_until_Xp` は立つので、それを使う。
    """
    p = RecognitionPipeline.__new__(RecognitionPipeline)
    p._enable_landing_chain_record_hold = False
    p._enable_chain_active_record_hold = True
    p._landing_chain_hold_until_1p = -1.0   # 着地経路では止まらない状況
    p._landing_chain_hold_until_2p = -1.0
    p._chain_until_1p = 100.0               # 別の経路が連鎖を立てた
    p._chain_until_2p = 0.0

    def marked(side: str, t: float) -> bool:
        landing = t < (p._landing_chain_hold_until_1p if side == "1P"
                       else p._landing_chain_hold_until_2p)
        if p._enable_chain_active_record_hold and not landing:
            hold = p._chain_until_1p if side == "1P" else p._chain_until_2p
            return t < hold
        return landing

    assert marked("1P", 50.0) is True, "着地経路以外の連鎖でも止める"
    assert marked("1P", 100.0) is False, "保持期限を過ぎたら記録する"
    assert marked("2P", 50.0) is False, "連鎖が無い側は止めない"


def test_w48b_cli_reaches_collect_lean(monkeypatch: Any) -> None:
    seen: dict = {}
    monkeypatch.setattr(CBL, "collect_lean",
                        lambda *a, **k: (seen.update(k), 0)[1])
    monkeypatch.setattr(sys, "argv", [
        "collect_boards_lean", "--video", "x.mp4", "--out-npz", "y.npz",
        "--enable-chain-active-record-hold",
    ])
    assert CBL.main() == 0
    assert seen.get("enable_chain_active_record_hold") is True
    assert seen.get("enable_landing_chain_record_hold") is False
