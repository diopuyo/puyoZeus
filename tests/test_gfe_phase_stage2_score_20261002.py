"""事前登録の不等号を境界で固定し、条件の取り違えを防ぐ。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.gfe_phase_stage2_score_20261002 import judge, MODES


def fixtures() -> tuple:
    """全門成立の最小115試合票を作る。"""
    sets = []
    for count in (57, 58):
        comparisons = {m: dict(m_time=dict(ci95=[-.003, .001]), agree=dict(ci95=[-.009, .01]))
                       for m in MODES[1:]}
        sets.append(dict(rows={m: [dict(m_time=.5)]*count for m in MODES}, comparisons=comparisons,
                         flips={m: 10 for m in MODES}, false_deaths={m: 0 for m in MODES}))
    old = {m: dict(false=0, scene_first_sec=2766., game14=[]) for m in MODES}
    speeds = {m: dict(p95_ms=20.) for m in MODES}
    metric = SimpleNamespace(paired=lambda *a: dict(ci95=[-.01, -.001], phase6=.4))
    return sets, old, speeds, metric


def test_all_gates_pass() -> None:
    """条件4の場面上限と条件6の20msは等号を許す。"""
    assert all(r["passed"] for r in judge(*fixtures()).values())


@pytest.mark.parametrize("condition", [1, 2, 3, 4, 6])
def test_reject_boundary(condition: int) -> None:
    """厳密不等号と上限超過を条件ごとに拒否する。"""
    sets, old, speeds, metric = fixtures()
    if condition == 1:
        metric = SimpleNamespace(paired=lambda *a: dict(ci95=[-.01, 0.], phase6=.4))
    elif condition == 2:
        sets[0]["comparisons"]["V3"]["m_time"]["ci95"][1] = .002
    elif condition == 3:
        sets[1]["comparisons"]["V3"]["agree"]["ci95"][0] = -.01
    elif condition == 4:
        old["V3"]["game14"] = [1.]
    else:
        speeds["V3"]["p95_ms"] = 20.00001
    result = judge(sets, old, speeds, metric)["V3"]
    assert not result["passed"] and not result["gates"][f"condition{condition}"]


def test_missing_game_rejected() -> None:
    """母数が欠けても採点済み扱いにしない。"""
    sets, old, speeds, metric = fixtures()
    sets[0]["rows"]["baseline"].pop()
    with pytest.raises(AssertionError):
        judge(sets, old, speeds, metric)


def test_fixed_proof_compatibility() -> None:
    """新しい既定OFF引数だけを吸収し、旧証明処理をそのまま継承する。"""
    from scripts.gfe_phase_stage2_replay_20261002 import bound_adapter
    class Original:
        def __init__(self) -> None:
            self.value = 123
        def prove(self) -> int:
            return self.value
    compatible = bound_adapter(Original)
    assert compatible(early_exit=False).prove() == Original().prove()
    assert compatible.prove is Original.prove
    with pytest.raises(ValueError):
        compatible(early_exit=True)


@pytest.mark.parametrize("options,version", [({}, "v3"), ({"count_sync": True}, "v4"),
                                           ({"e16": True}, "v4"), ({"e16": False}, "v3")])
def test_production_model_selection(options: dict, version: str) -> None:
    """案Cの構成選択を固定し、無条件に別版へ切り替えない。"""
    from scripts.gfe_phase_stage2_replay_20261002 import production_directory
    assert production_directory(options).name == "exchange_event_" + version


def test_notification_includes_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """一通知の複数呼出しを合計し、G_fe無し通知でP95を薄めない。"""
    from scripts.gfe_phase_stage2_replay_20261002 import install_notifications
    from src.exchange_event_overlay import ExchangeEventOverlay
    total, inference = [], []
    def update(overlay: object, values: list, model_values: list) -> None:
        total.extend(values)
        inference.extend(model_values)
    monkeypatch.setattr(ExchangeEventOverlay, "update", update)
    totals, inferences, counts = install_notifications(total, inference)
    ExchangeEventOverlay.update(None, [2., 3.], [.4, .5])
    ExchangeEventOverlay.update(None, [], [])
    assert totals == [5.] and inferences == [.9] and counts == [2]
