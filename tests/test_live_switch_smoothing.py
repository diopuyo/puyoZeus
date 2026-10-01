"""ライブ評価の切替平滑 (既定OFF): 状態の受け渡し・既定の不変・ランチャー設定。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.run_live_pipeline_20260928 import with_switch_smoothing
from src.exchange_display_smoothing import SwitchAwareDisplayEMA
from src.phase_j.live_notification_eval import NotificationExchangeOverlay

FRAME = 1 / 30
ROOT = Path(__file__).resolve().parents[1]


def tracker(source: str, game: int, landings: int = 0, probability: float | None = 0.5) -> SimpleNamespace:
    """評価器の最小の外形 (出どころ・試合・着地数・確率)。"""
    current = SimpleNamespace(exchange_id=1, chains=[], landings=[None] * landings)
    return SimpleNamespace(source=source, _game_idx=game, records=[None], current=current, probability=probability)


def sequence() -> list[tuple[SimpleNamespace, tuple[float, float] | None]]:
    """試合境界 (古い値から新試合へ) と非イベント切替を含む入力列。"""
    items = [(tracker("G_fe", 1), (-80.0, 0.05))] * 20
    items += [(tracker("S3_landing", 1, landings=1), (-60.0, 0.1))] * 20
    items += [(tracker("G_fe", 1, landings=1), (10.0, 0.55))] * 20
    items += [(tracker("waiting_confirmed", 2, probability=None), None)] * 5
    items += [(tracker("G_fe", 2), (5.0, 0.52))] * 20
    return items


def run(engine: SwitchAwareDisplayEMA, items: list, start: int = 0) -> list:
    return [engine.apply(t, target, (start + i) * FRAME) for i, (t, target) in enumerate(items)]


def test_state_roundtrip_matches_uninterrupted_run() -> None:
    """途中で状態を書き出して別インスタンスへ戻しても、以降の表示が完全に一致する (journal再実行の前提)。"""
    items = sequence()
    reference = run(SwitchAwareDisplayEMA(), items)
    for split in (7, 25, 45, 63):
        first = SwitchAwareDisplayEMA()
        run(first, items[:split])
        second = SwitchAwareDisplayEMA()
        second.restore_state(first.export_state())
        assert run(second, items[split:], split) == reference[split:]


def test_legacy_initial_state_is_ignored() -> None:
    """supervisor の初期値 (0., .5, None) は「状態なし」で、復元しても何も起きない。"""
    engine = SwitchAwareDisplayEMA()
    engine.restore_state((0.0, 0.5, None))
    assert engine.state is None and engine.shown is None


def test_game_boundary_does_not_resume_from_stale_ema() -> None:
    """新試合の最初の値は前試合末尾の古いEMA (-80付近) へ引きずられず、新しい値に合う。"""
    shown = run(SwitchAwareDisplayEMA(), sequence())
    first_new_game = shown[65]
    assert abs(first_new_game[0] - 5.0) < 1e-9


def test_overlay_default_is_off_and_state_is_legacy_tuple() -> None:
    """既定OFFは従来のEMA (3要素の状態)。ONは識別子つきの全状態。"""
    off = object.__new__(NotificationExchangeOverlay)
    off.switch_smoothing, off.smoothing = False, SimpleNamespace(adv=1.0, probability=0.6, last_sec=2.0)
    assert off.smoothing_state() == (1.0, 0.6, 2.0)
    on = object.__new__(NotificationExchangeOverlay)
    on.switch_smoothing, on.smoothing = True, SwitchAwareDisplayEMA()
    assert on.smoothing_state()[0] == "switch_aware_v1"
    on.restore_smoothing((0.0, 0.5, None))
    assert on.smoothing.state is None


def test_launcher_injection_only_when_enabled() -> None:
    """ランチャーの設定がOFFなら評価器をそのまま、ONなら switch_smoothing=True を足す。"""
    seen: dict = {}

    def evaluator(*args: object, **kwargs: object) -> str:
        seen.update(kwargs)
        return "ok"
    assert with_switch_smoothing(evaluator, SimpleNamespace(switch_smoothing=False)) is evaluator
    assert with_switch_smoothing(evaluator, SimpleNamespace())  is evaluator
    wrapped = with_switch_smoothing(evaluator, SimpleNamespace(switch_smoothing=True))
    assert wrapped(1, per_side_settled=True) == "ok" and seen == dict(switch_smoothing=True, per_side_settled=True)


def test_distribution_launcher_default_enables_it() -> None:
    """配布ランチャーの既定設定でON。コード側の既定は OFF (parse_args の既定)。"""
    defaults = json.loads((ROOT / "config/live_defaults.json").read_text(encoding="utf-8"))
    assert defaults["switch_smoothing"] is True
    import inspect
    from scripts import run_live_pipeline_20260928 as live
    assert "'--switch-smoothing', action=argparse.BooleanOptionalAction, default=False" in inspect.getsource(live.parse_args)
