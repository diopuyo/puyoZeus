"""Gate 4 条件5のoverlay配線・sidecar・既定OFF回帰。"""
from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.board import Board  # noqa: E402
from src.chain_detector import CHAIN_MECHANISM_FORMULA, ChainEvent  # noqa: E402
from src.exchange_episode_tracker import ChainEventObservation  # noqa: E402
from src.exchange_ledger import PhysicalContext  # noqa: E402
from src.live_exchange_episode_tracker import LiveExchangeEpisodeTracker  # noqa: E402


def test_hold_capped_adv_recomputes_probability_from_displayed_value() -> None:
    """episode cap後も旧candidate勝率を残さず、表示advと同期する。"""
    source = inspect.getsource(vao.generate)
    assert source.count(
        "_sync_probability_after_episode_gate(") == 2
    assert vao.adv_to_winprob(
        vao.EPISODE_UNRESOLVED_ABS_CAP) == pytest.approx(0.90)
    assert source.count("_cap_unresolved_episode_display(") == 1


def test_partial_episode_gate_change_recomputes_probability() -> None:
    """±100候補でなくても表示advが変われば勝率を旧値のまま残さない。"""
    p1 = vao._sync_probability_after_episode_gate(-20.0, 0.4, -85.0)
    assert p1 == pytest.approx(vao.adv_to_winprob(-85.0))


def test_unchanged_episode_gate_keeps_existing_probability() -> None:
    """表示advが不変なら、既存の較正済み勝率を再変換しない。"""
    assert vao._sync_probability_after_episode_gate(20.0, 0.731, 20.0) == 0.731


def test_display_probability_direction_is_corrected_outside_even() -> None:
    """有利不利は2P方向なのに1P勝率が50%超、という表示矛盾を残さない。"""
    p1 = vao._ensure_display_probability_direction(-6.16, 0.563)
    assert p1 == pytest.approx(vao.adv_to_winprob(-6.16))
    assert p1 < 0.5


def test_display_probability_direction_keeps_even_band_history() -> None:
    """EVEN帯の微小な符号差では、較正済み確率を不要に再変換しない。"""
    assert vao._ensure_display_probability_direction(0.002, 0.4998) == 0.4998


@pytest.mark.parametrize("adv", [-100.0, -84.0, 84.0, 100.0])
def test_unresolved_episode_final_display_is_probability_bounded(adv: float) -> None:
    """旧hold/EMA由来でも未解決中の表示は勝率10〜90%を越えない。"""
    p1 = vao.adv_to_winprob(adv)
    capped_adv, capped_p1, applied = vao._cap_unresolved_episode_display(
        adv, p1, is_unresolved=True)
    expected = vao.EPISODE_UNRESOLVED_ABS_CAP * (1.0 if adv > 0.0 else -1.0)
    assert capped_adv == pytest.approx(expected)
    assert capped_p1 == pytest.approx(0.90 if adv > 0.0 else 0.10)
    assert applied is True


def test_resolved_episode_final_display_cap_is_noop() -> None:
    """交換が解決済みなら、確定した±100表示は弱めない。"""
    assert vao._cap_unresolved_episode_display(
        100.0, 0.993, is_unresolved=False) == (100.0, 0.993, False)


def test_unresolved_episode_final_display_keeps_value_inside_cap() -> None:
    """未解決でも通常評価の範囲内は較正済み確率を含めて変更しない。"""
    assert vao._cap_unresolved_episode_display(
        20.0, 0.731, is_unresolved=True) == (20.0, 0.731, False)


@pytest.mark.parametrize("after", [-100.0, 100.0])
def test_unresolved_episode_blocks_both_hard_override_directions(after: float) -> None:
    value, candidate, applied, reason = vao._apply_episode_hard_override_gate(
        12.5, after, False)
    expected = vao.EPISODE_UNRESOLVED_ABS_CAP * (1.0 if after > 0.0 else -1.0)
    assert value == pytest.approx(expected)
    assert candidate is True
    assert applied is False
    assert reason == "episode_unresolved_capped"


def test_resolved_episode_allows_hard_override() -> None:
    value, candidate, applied, reason = vao._apply_episode_hard_override_gate(
        -8.0, 100.0, True)
    assert value == 100.0
    assert candidate is True
    assert applied is True
    assert reason == ""


def test_unresolved_episode_keeps_partial_kill_evidence() -> None:
    """仕様が禁止するのは±100断定であり、向きを示す部分補正ではない。"""
    value, candidate, applied, reason = vao._apply_episode_hard_override_gate(
        12.5, 72.0, False, is_unresolved=True)
    assert value == 72.0
    assert candidate is False
    assert applied is False
    assert reason == ""


def test_unresolved_cap_does_not_weaken_existing_stronger_same_direction() -> None:
    """生モデルが既に+95なら、未解決capでモデル自身を弱めない。"""
    value, candidate, applied, reason = vao._apply_episode_hard_override_gate(
        95.0, 100.0, False, is_unresolved=True)
    assert value == 95.0
    assert candidate is True
    assert applied is False
    assert reason == "episode_unresolved_capped"


def test_decision_invariance_corrects_opposite_direction() -> None:
    """物理勝者方向を採用してもformal境界前は勝率90%までに留める。"""
    value, candidate, applied, reason = vao._apply_episode_hard_override_gate(
        12.5, -100.0, True, is_unresolved=True,
        hard_override_target=100.0)
    assert value == vao.EPISODE_UNRESOLVED_ABS_CAP
    assert candidate is True
    assert applied is False
    assert reason == "episode_physical_target_capped"


def test_decision_invariance_allows_matching_hard_direction() -> None:
    value, candidate, applied, reason = vao._apply_episode_hard_override_gate(
        12.5, 100.0, True, is_unresolved=True,
        hard_override_target=100.0)
    assert value == vao.EPISODE_UNRESOLVED_ABS_CAP
    assert candidate is True
    assert applied is False
    assert reason == "episode_physical_target_capped"


def test_decision_invariance_promotes_partial_value_to_physical_target() -> None:
    """勝者方向が得られてもformal境界前は勝率90%上限を維持する。"""
    value, candidate, applied, reason = vao._apply_episode_hard_override_gate(
        12.5, 72.0, True, is_unresolved=True,
        hard_override_target=100.0)
    assert value == vao.EPISODE_UNRESOLVED_ABS_CAP
    assert candidate is True
    assert applied is False
    assert reason == "episode_physical_target_capped"


def test_unresolved_hard_candidate_opposite_ledger_direction_is_rejected() -> None:
    """台帳が2P優勢なのに旧holdが1P致死勝ちを出しても採用しない。"""
    value, candidate, applied, reason = vao._apply_episode_hard_override_gate(
        27.0, 100.0, False, is_unresolved=True,
        physical_net_raw=-54.0)
    assert value == 27.0
    assert candidate is True
    assert applied is False
    assert reason == "episode_direction_conflict"


@pytest.mark.parametrize(
    ("net_raw", "expected"),
    [(553.0, (0.0, 553.0)), (-553.0, (553.0, 0.0)), (0.0, (0.0, 0.0))],
)
def test_episode_net_is_wired_to_receiver_kill_inputs(
    net_raw: float, expected: tuple[float, float],
) -> None:
    """1P視点の純残量を、攻撃側ではなく受け側のpendingへ渡す。"""
    assert vao._episode_kill_override_inputs(net_raw) == expected


def test_loop_uses_episode_net_as_accumulator_replacement() -> None:
    """条件5が旧累積器をOFFにするだけの未配線へ退行しない。"""
    source = Path(vao.__file__).read_text(encoding="utf-8")
    assert "kpending1, kpending2 = _episode_kill_override_inputs(" in source
    assert "episode_drive.snapshot.ledger.net_raw" in source


def test_unchanged_value_is_not_reported_as_candidate() -> None:
    value, candidate, applied, reason = vao._apply_episode_hard_override_gate(
        21.0, 21.0, False)
    assert value == 21.0
    assert candidate is False
    assert applied is False
    assert reason == ""


def _live_drive() -> vao._EpisodeDriveResult:
    tracker = LiveExchangeEpisodeTracker(enabled=True)
    snap = tracker.observe_frame(
        t_sec=1.0,
        context=PhysicalContext(p1_chaining=True),
        chain_observations=(ChainEventObservation(
            side="1P", t_sec=1.0, mechanism=CHAIN_MECHANISM_FORMULA,
            chain_count=1, total_score=700, ojama_sent=0,
            game_idx=0, elapsed_sec=1.0),),
    )
    return vao._EpisodeDriveResult(
        snapshot=snap, gross_inspected_sides=2,
        gross_residual_p1=0.25, gross_residual_p2=-0.5)


def test_episode_sidecar_columns_hold_nonzero_values(tmp_path: Path) -> None:
    row = vao._episode_timeline_row(
        _live_drive(), t_sec=1.0, game_idx=0,
        state1="CHAIN", state2="STABLE",
        hard_candidate=True, hard_applied=False,
        hard_path="live",
        hard_reason="episode_unresolved")
    path = tmp_path / "episode.npz"
    vao.save_episode_timeline(path, "video", [row])
    with np.load(str(path), allow_pickle=True) as data:
        assert set(vao.EpisodeTimelineRow.__dataclass_fields__) <= set(data.files)
        assert float(data["net_raw"][0]) == pytest.approx(10.0)
        assert int(data["active_chain_count"][0]) == 1
        assert str(data["state1"][0]) == "CHAIN"
        assert str(data["state2"][0]) == "STABLE"
        assert int(data["gross_inspected_sides"][0]) == 2
        assert float(data["gross_residual_p1"][0]) == pytest.approx(0.25)
        assert bool(data["hard_override_candidate"][0]) is True
        assert bool(data["hard_override_applied"][0]) is False
        assert float(data["hard_override_target"][0]) == 0.0
        assert str(data["hard_override_path"][0]) == "live"
        assert str(data["hard_override_hold_reason"][0]) == "episode_unresolved"


def test_hard_override_path_keeps_both_live_and_hold_sources() -> None:
    path = vao._append_episode_hard_path("", "live", True)
    path = vao._append_episode_hard_path(path, "hold_active", True)
    path = vao._append_episode_hard_path(path, "live", True)
    assert path == "live+hold_active"


def _event(score: int = 700) -> ChainEvent:
    return ChainEvent(
        trigger_sec=1.0, end_sec=2.0, before_board=Board(), chain_count=1,
        total_erased=4, total_score=score, base_score=score,
        all_clear_bonus_applied=0, ojama_sent=score // 70,
        leftover_score=0, is_all_clear=False,
        mechanism=CHAIN_MECHANISM_FORMULA)


def test_same_chain_after_none_gap_is_not_emitted_twice() -> None:
    adapter = vao._LiveEpisodeOverlayAdapter()
    result = types.SimpleNamespace(
        p1=types.SimpleNamespace(chain_event=_event()),
        p2=types.SimpleNamespace(chain_event=None))
    assert len(adapter._chain_observations(result, 1.0, 0, 1.0)) == 1
    result.p1.chain_event = None
    assert adapter._chain_observations(result, 1.1, 0, 1.1) == ()
    result.p1.chain_event = _event()
    assert adapter._chain_observations(result, 1.2, 0, 1.2) == ()
    result.p1.chain_event = _event(1400)
    assert len(adapter._chain_observations(result, 1.3, 0, 1.3)) == 1
