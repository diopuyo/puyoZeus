"""切替対応の表示平滑 (既定OFF) の契約: (a) 旧評価器値もEMA (b) 古いEMAから再開しない (c) 非イベント切替のブレンド。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import exchange_display_smoothing as smoothing
from src.exchange_display_smoothing import SwitchAwareDisplayEMA, event_stamp

FRAME = 1 / 30


def tracker(source: str, game: int = 1, landings: int = 0, records: int = 0) -> SimpleNamespace:
    """評価器の最小の外形。着地数・記録数が物理イベントの観測を表す。"""
    chain = SimpleNamespace(end_signal_sec=None)
    current = SimpleNamespace(exchange_id=records, chains=[chain], landings=[None] * landings)
    return SimpleNamespace(source=source, _game_idx=game, records=[None] * records,
                           current=current if records else None)


def run(items: list[tuple[SimpleNamespace, tuple[float, float] | None]],
        engine: SwitchAwareDisplayEMA | None = None) -> list[tuple[float, float] | None]:
    """1/30 秒刻みで順に与え、各フレームの表示を返す。"""
    engine = engine or SwitchAwareDisplayEMA()
    return [engine.apply(t, target, i * FRAME) for i, (t, target) in enumerate(items)]


def test_ema_alpha_matches_production_display() -> None:
    """循環importを避けた再掲の時定数が、既存表示と同値であること。"""
    from scripts.visualize_advantage_overlay import EMA_ALPHA
    assert smoothing.EMA_ALPHA == EMA_ALPHA


def test_waiting_value_goes_through_ema_a() -> None:
    """(a) 旧評価器の値を直接ではなく、EMAを通して表示する。"""
    engine = SwitchAwareDisplayEMA()
    shown = run([(tracker("G_fe"), (0.0, 0.5)), (tracker("waiting_confirmed"), (40.0, 0.7))], engine)
    assert engine.state == pytest.approx((10.0, 0.55))  # 0.25*40 + 0.75*0
    assert shown[1][0] < 40.0  # 生の旧評価器値 (40) をそのまま出さない


def test_return_does_not_resume_from_stale_ema_b() -> None:
    """(b) waiting中もEMA状態が表示へ連続するので、戻った最初のフレームで古い値へ飛ばない。"""
    items = [(tracker("G_fe"), (-90.0, 0.04))] * 3 + [(tracker("waiting_confirmed"), (0.0, 0.5))] * 60
    items += [(tracker("G_fe", game=1), (5.0, 0.52))]
    shown = run(items)
    before, after = shown[-2], shown[-1]
    assert abs(after[0] - before[0]) < 5.0  # 古い -90 側の状態から再開していれば 20 以上飛ぶ


def test_confirmed_death_is_immediate_and_resyncs_state() -> None:
    """確定死亡は遅延させず即時表示し、EMA状態もその値へ合わせる。"""
    shown = run([(tracker("G_fe"), (0.0, 0.5)), (tracker("confirmed_death"), (100.0, 1.0)),
                 (tracker("G_fe"), (100.0, 1.0))])
    assert shown[1] == (100.0, 1.0) and shown[2] == pytest.approx((100.0, 1.0))


def test_game_boundary_snaps() -> None:
    """試合境界は新しい試合の値へ即時に合わせる (前の試合のEMAを持ち込まない)。"""
    shown = run([(tracker("G_fe", game=1), (80.0, 0.9)), (tracker("waiting_confirmed", game=2), (0.0, 0.5))])
    assert shown[1] == (0.0, 0.5)


def test_non_event_switch_is_blended() -> None:
    """(c) 物理イベントのない由来切替は、直前の表示から線形に移る。"""
    engine = SwitchAwareDisplayEMA()
    items = [(tracker("S3_landing", landings=1, records=1), (10.0, 0.55))] * 30
    items += [(tracker("G_fe", landings=1, records=1), (60.0, 0.8))] * 30
    shown = run(items, engine)
    first = shown[30]
    assert first == pytest.approx(shown[29])  # 切替フレームは直前値を保持
    assert engine.counters["blends"] == 1
    halfway = shown[30 + int(smoothing.BLEND_SEC * 30 / 2)]
    assert shown[29][0] < halfway[0] < shown[-1][0]
    assert shown[-1][0] == pytest.approx(engine.state[0])  # 終了後はEMA状態そのもの


def test_event_switch_is_not_blended() -> None:
    """発火 (記録数の増加) と同時の切替は正当な変化として平滑しない。"""
    engine = SwitchAwareDisplayEMA()
    items = [(tracker("G_fe"), (0.0, 0.5))] * 10 + [(tracker("S3_landing", records=1), (50.0, 0.7))]
    run(items, engine)
    assert engine.counters["blends"] == 0 and engine.counters["event_switches"] == 1


def test_death_switch_is_not_blended() -> None:
    """死亡を含む切替は、確定死亡でなくても平滑開始しない。"""
    engine = SwitchAwareDisplayEMA()
    items = [(tracker("S3_landing", landings=1, records=1), (0.0, 0.5))] * 40
    items += [(tracker("unavoidable_death", landings=1, records=1), (100.0, 1.0))]
    run(items, engine)
    assert engine.counters["blends"] == 0


def test_same_frame_reference_is_idempotent() -> None:
    """同一フレームの再参照では二重更新しない。"""
    engine = SwitchAwareDisplayEMA()
    first = engine.apply(tracker("G_fe"), (0.0, 0.5), 1.0)
    engine.apply(tracker("G_fe"), (40.0, 0.7), 2.0)
    again = engine.apply(tracker("G_fe"), (99.0, 0.9), 2.0)
    assert again == engine.shown and first == (0.0, 0.5)


def test_unknown_target_holds_without_marking_frame() -> None:
    """旧評価器の値も未知なら更新せず、同フレームの後続の実値呼出を阻害しない。"""
    engine = SwitchAwareDisplayEMA()
    engine.apply(tracker("G_fe"), (10.0, 0.55), 0.0)
    assert engine.apply(tracker("waiting_confirmed"), None, 1.0) == (10.0, 0.55)
    engine.apply(tracker("waiting_confirmed"), (50.0, 0.8), 1.0)
    assert engine.state[0] > 10.0  # 後続の実値で更新された (同フレームの空呼出に阻害されない)


def test_event_stamp_counts_records_chains_landings() -> None:
    """物理イベントの観測印は、記録数・連鎖数・終了信号・着地数の増加で変わる。"""
    assert event_stamp(tracker("G_fe")) != event_stamp(tracker("G_fe", records=1))
    assert event_stamp(tracker("G_fe", records=1)) != event_stamp(tracker("G_fe", records=1, landings=1))


def test_default_is_off_in_render_and_replay() -> None:
    """既定OFF: 描画・再生の関数と CLI の既定が False で、従来のEMAクラスを使う。"""
    import inspect
    from scripts import replay_exchange_event_20260926 as replay
    # Phase 6では本番描画CLIへ配線せず、再生器の任意指定だけを移植する。
    assert inspect.signature(replay.replay).parameters["switch_smoothing"].default is False


def test_raw_waiting_shows_old_value_and_syncs_state() -> None:
    """構成B: 評価器の値が無い間は旧評価器の生値を表示し、EMA状態をその値へ合わせる (戻り時に古い状態から再開しない)。"""
    engine = SwitchAwareDisplayEMA(raw_waiting=True)
    wait = tracker("waiting_confirmed")
    wait.probability = None
    ready = tracker("G_fe")
    ready.probability = 0.9
    items = [(ready, (-90.0, 0.04)), (wait, (40.0, 0.7)), (wait, (41.0, 0.71))]
    shown = run(items, engine)
    assert shown[1] == (40.0, 0.7) and shown[2] == (41.0, 0.71)
    assert engine.state == (41.0, 0.71)


def test_raw_waiting_default_off_keeps_config_a() -> None:
    """既定は構成A (旧評価器の値もEMAを通す)。"""
    assert SwitchAwareDisplayEMA().raw_waiting is False


def test_exchange_display_off_path_is_legacy() -> None:
    """平滑オブジェクトが従来のEMAなら、_exchange_display は従来式 (EMA、確定死亡だけ即時) のまま。"""
    from scripts.visualize_advantage_overlay import (
        EMA_ALPHA as alpha, _ExchangeDisplayEMA, _exchange_display, _winprob_to_adv)
    overlay = SimpleNamespace(tracker=SimpleNamespace(probability=0.8, source="G_fe"))
    ema = _ExchangeDisplayEMA()
    adv, prob = _exchange_display(overlay, 1.0, 0.5, ema, 1.0)
    assert prob == pytest.approx(alpha * 0.8 + (1 - alpha) * 0.5)
    assert adv == pytest.approx(alpha * _winprob_to_adv(0.8))
    overlay.tracker.source, overlay.tracker.probability = "confirmed_death", 1.0
    assert _exchange_display(overlay, 1.0, 0.5, ema, 2.0)[1] == 1.0
    overlay.tracker.probability = None  # 評価器の値なし: 旧評価器の値をそのまま返す (従来動作)
    assert _exchange_display(overlay, 7.0, 0.6, ema, 3.0) == (7.0, 0.6)
