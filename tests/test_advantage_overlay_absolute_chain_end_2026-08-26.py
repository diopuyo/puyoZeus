"""決着ホールドを「連鎖の終わりの絶対律」で解除する根治のテスト (2026-08-26)。

背景 (user決定、`docs/agent_coordination/DECISIONS.md` 2026-08-26 節):
`ResolvedExchangeTracker` のホールド解除条件は従来「両側の `chain_event` が None」
だった。ChainEvent は長い連鎖で 1.4秒ごとに断片化するため打ち合い中は長時間
成立せず、実測で**最大 45.07秒** settled 再計算が止まっていた
(ホールドが潰した settled 1880/3964 = 47.43%、評価行が旧比 −26%)。

根治として、user 伝授の絶対律
「連鎖の終わり = 連鎖している側のネクストが動いた瞬間 OR
  連鎖している側にお邪魔が落ちた瞬間」
(memory `reference_chain_end_absolute_signals_2026-08-21`) を観測する。
段間の一時的な終了候補は、次段の CHAIN 再突入で撤回する。

検査するもの:
- 既定 OFF での bit-identical (フラグを渡さなければ挙動が一切変わらない)
- ON でネクスト移動 / お邪魔着弾のそれぞれ単独で解除されること
- `next_pair=None` を「不動」と誤解しないこと
- 単発の点滅で早期解除しないこと (デバウンス)
- 安全弁が従来どおり効くこと
- フラグの配線 (シグネチャ / CLI / 構築2箇所 / `__init__` 既定)
"""
from __future__ import annotations

import dataclasses
import inspect
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.board import Board  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.chain_detector import ChainEvent  # noqa: E402
from src.ojama_accounting import OjamaAccountSnapshot  # noqa: E402


# ============================
# ローカルヘルパー (他テストファイルへ依存させない)
# ============================


def _ev(trigger_sec: float = 1.0, total_score: int = 500) -> ChainEvent:
    return ChainEvent(
        trigger_sec=trigger_sec, end_sec=trigger_sec + 1.0, before_board=Board(),
        chain_count=1, total_erased=0, total_score=total_score, base_score=0,
        all_clear_bonus_applied=0, ojama_sent=0, leftover_score=0,
        is_all_clear=False, score_estimated=False,
    )


def _snap(pending_p1: int = 0, pending_p2: int = 0,
          dropped1: int = 0, dropped2: int = 0) -> OjamaAccountSnapshot:
    """最小 snapshot。`total_dropped_to_pX` を指定できるのが既存ヘルパーとの差。"""
    return OjamaAccountSnapshot(
        t_sec=0.0, pending_p1=pending_p1, pending_p2=pending_p2,
        total_generated_by_p1=0, total_generated_by_p2=0,
        total_offset_by_p1=0, total_offset_by_p2=0,
        total_dropped_to_p1=dropped1, total_dropped_to_p2=dropped2,
        net_ojama_balance=pending_p2 - pending_p1,
        overflow_risk_p1=False, overflow_risk_p2=False, confidence=1.0,
        leftover_p1=0, leftover_p2=0,
        all_clear_pending_p1=False, all_clear_pending_p2=False,
        chain_end_triggered_p1=False, chain_end_triggered_p2=False,
        chain_total_score_p1=0, chain_total_score_p2=0,
    )


def _sig(chain_event, next_pair, score: int = 0,
         state: BoardState = BoardState.CHAIN,
         slide: bool = False) -> types.SimpleNamespace:
    """`SideResult` もどき。

    絶対律で読むのは `next_pair` (色ペア値)、`next_slide_motion` (物理スライド)、
    `state` (新ツモ落下の立ち上がり)。着弾量は snapshot 側から取る。
    """
    return types.SimpleNamespace(
        chain_event=chain_event, score=score, state=state,
        confirmed_board=Board(), next_pair=next_pair,
        next_slide_motion=slide,
    )


def _stub_score_advantage():
    calls: list[int] = []

    def _stub(model, b1, b2, snap, feature_cols=None, attribution_exclude=()) -> tuple:
        calls.append(1)
        n = len(calls)
        return float(n * 10), 0.5 + n * 0.05, []

    return _stub, calls


def _fire(tracker) -> None:
    """両側同時発火でホールドを開始させる (next は両側 (1,2))。"""
    tracker.update(_sig(_ev(total_score=500), (1, 2)),
                   _sig(_ev(total_score=300), (1, 2)), _snap(), 0.0)
    assert tracker._active is True


def _settle_abs_baseline(
    tracker, next1=(1, 2), next2=(1, 2), start_sec: float = 1.0,
) -> None:
    """開始設置のNEXT移動を除外し、連鎖中の安定値を基準化する。"""
    tracker.update(_sig(_ev(), next1), _sig(_ev(), next2), _snap(), start_sec)
    tracker.update(
        _sig(_ev(), next1), _sig(_ev(), next2), _snap(), start_sec + 0.033)
    assert tracker._abs_baseline_ready == [True, True]


# ============================
# 既定 OFF の bit-identical
# ============================


def test_off_does_not_release_on_next_change(monkeypatch) -> None:
    """既定 OFF では、ネクストが動いても chain_event が生きている限り解除しない
    (= 従来挙動そのまま)。これが 45秒ホールドの再現でもある。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object())
    _fire(tr)
    # 連鎖アニメは断片化して続いている (chain_event は非None) が、ネクストは動いた。
    # 末尾だけでなく毎フレーム検査する (解除→再発火で末尾が戻る罠を避ける)。
    for i in range(10):
        active, just = tr.update(_sig(_ev(), (3, 4)), _sig(_ev(), (3, 4)),
                                 _snap(), 1.0 + i * 0.033)
        assert active is True and just is False, f"OFF なのに frame {i} で解除された"


def test_off_state_untouched(monkeypatch) -> None:
    """既定 OFF では絶対律の追跡状態も統計も一切動かない (副作用なし)。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object())
    _fire(tr)
    tr.update(_sig(_ev(), (3, 4)), _sig(_ev(), (3, 4)), _snap(dropped1=9, dropped2=9), 1.0)
    assert tr.abs_end_stats["sessions"] == 0
    assert tr._abs_ended == [False, False]


def test_off_legacy_release_unchanged(monkeypatch) -> None:
    """既定 OFF の解除タイミングは従来どおり (両側 chain_event が None)。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object())
    _fire(tr)
    active, just = tr.update(_sig(None, (1, 2)), _sig(None, (1, 2)), _snap(), 1.0)
    assert active is False and just is True


# ============================
# ON: 絶対律の2信号
# ============================


def test_on_releases_on_next_move(monkeypatch) -> None:
    """【絶対律A】ネクストが動いたら、chain_event が生きていても解除する。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    assert tr.abs_end_stats["sessions"] == 1
    _settle_abs_baseline(tr)
    # 1フレーム目: 確認フレーム数に届かないのでまだ解除しない。
    active, just = tr.update(
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE), _snap(), 1.1)
    assert active is True and just is False
    # 2フレーム目で確定 → 着弾待ちへ入り、pending 0 なので即解放。
    active, just = tr.update(
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE), _snap(), 1.133)
    assert active is False and just is True
    assert tr.abs_end_stats["released_by_abs"] == 1
    assert tr.abs_end_stats["released_by_legacy"] == 0
    assert tr.abs_end_stats["end_by_next"] == [1, 1]


def test_on_releases_on_ojama_landing(monkeypatch) -> None:
    """【絶対律B】ネクストが動かなくても、お邪魔が落ちたら解除する。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    landed = _snap(dropped1=6, dropped2=6)
    # CHAIN表示中は直近物理段から1.5秒だけ継続を優先する。静穏後は
    # stateが残留していても着弾の絶対証拠を受理する。
    tr.update(_sig(_ev(), (1, 2)), _sig(_ev(), (1, 2)), landed, 1.6)
    active, just = tr.update(_sig(_ev(), (1, 2)), _sig(_ev(), (1, 2)), landed, 1.633)
    assert active is False and just is True
    assert tr.abs_end_stats["end_by_ojama"] == [1, 1]
    assert tr.abs_end_stats["end_by_next"] == [0, 0]


def test_on_releases_on_physical_slide_with_same_color_pair(monkeypatch) -> None:
    """【Codex 第27報レビュー】次ツモが**同じ色ペア**でも、NEXT の物理スライドで
    解除できる。

    色ペア値の比較だけだと、次ツモがたまたま同じ色 (4色で約6.25%) のとき
    連鎖の終わりを検出できない。物理信号ならその取りこぼしが無い。
    """
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)   # セッション開始時の next は (1, 2)
    _settle_abs_baseline(tr)
    # next_pair は (1, 2) のまま = 色ペア値では一切変化が見えない。
    tr.update(_sig(_ev(), (1, 2), state=BoardState.GRAVITY_SETTLE, slide=True),
              _sig(_ev(), (1, 2), state=BoardState.GRAVITY_SETTLE, slide=True),
              _snap(), 1.1)
    active, just = tr.update(
        _sig(_ev(), (1, 2), state=BoardState.GRAVITY_SETTLE, slide=True),
        _sig(_ev(), (1, 2), state=BoardState.GRAVITY_SETTLE, slide=True),
        _snap(), 1.133)
    assert active is False and just is True
    assert tr.abs_end_stats["end_by_slide"] == [1, 1]
    assert tr.abs_end_stats["end_by_next"] == [0, 0], "色ペア変化では検出していない"


def test_on_releases_on_new_tsumo_fall_with_same_color_pair(monkeypatch) -> None:
    """【Codex 第27報レビュー】次ツモが同じ色ペアでも、新しいツモの落下開始
    (`*→TSUMO_FALL`) で解除できる。置けた = ネクストが動いた、の物理的証拠。

    立ち上がりは1フレームだけのエッジなので、セッション内でラッチされて
    連続フレーム確認を通れることも同時に固定する。
    """
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    _settle_abs_baseline(tr)
    # 1フレーム目だけ TSUMO_FALL へ立ち上がり、以後は STABLE (エッジは消える)。
    tr.update(_sig(_ev(), (1, 2), state=BoardState.TSUMO_FALL),
              _sig(_ev(), (1, 2), state=BoardState.TSUMO_FALL), _snap(), 1.1)
    active, just = tr.update(_sig(_ev(), (1, 2), state=BoardState.STABLE),
                             _sig(_ev(), (1, 2), state=BoardState.STABLE),
                             _snap(), 1.133)
    assert active is False and just is True, "エッジがラッチされていない"
    assert tr.abs_end_stats["end_by_tsumo"] == [1, 1]


def test_on_requires_both_sides(monkeypatch) -> None:
    """片側だけ絶対律が成立しても解除しない (打ち合いは両側が終わって初めて終わる)。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    _settle_abs_baseline(tr)
    for i in range(6):
        active, just = tr.update(
            _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
            _sig(_ev(), (1, 2), state=BoardState.CHAIN),
            _snap(), 1.1 + i * 0.033)
    assert active is True and just is False
    assert tr._abs_ended == [True, False]


# ============================
# 誤検知への備え
# ============================


def test_on_next_none_is_not_evidence(monkeypatch) -> None:
    """`next_pair=None` は「未検知」であって「不動」でも「変化」でもない。
    None が続く間は絶対律Aを成立させない (第26報 指摘2 と同型の誤りを避ける)。

    **毎フレーム**検査するのが要点。最後のフレームだけ見ると、いったん解除された
    直後に同じ chain_event で新しい保持セッションが再発火して `active=True` に
    戻るため、誤りを見逃す (この検査自体が最初その罠を踏んでいた)。
    保持セッション数が1のままであることも併せて固定する。
    """
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    _settle_abs_baseline(tr)
    for i in range(10):
        active, just = tr.update(_sig(_ev(), None), _sig(_ev(), None),
                                 _snap(), 1.1 + i * 0.033)
        assert active is True and just is False, f"frame {i} で解除された"
        assert tr._abs_ended == [False, False], f"frame {i} で終了扱いになった"
    # 再発火していない = 一度も解除されていないことの独立な裏取り。
    assert tr.abs_end_stats["sessions"] == 1


def test_on_single_frame_flicker_does_not_release(monkeypatch) -> None:
    """単発フレームの点滅 (1フレームだけ next が変わって戻る) では解除しない。
    W30 の chain_event 点滅と同種の誤検知を、確認フレーム数で弾く。

    ここも**フレームごと**に検査する。1フレーム目で解除されてしまう実装でも、
    2フレーム目で再発火して `active=True` に戻るため、末尾だけ見ると通ってしまう。
    """
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    _settle_abs_baseline(tr)
    # 1フレーム目: next が変化。確認フレーム数に届かないので、まだ解除しない。
    active, just = tr.update(
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE), _snap(), 1.1)
    assert active is True and just is False, "1フレームの変化で解除された"
    assert tr._abs_pending_frames == [1, 1]
    # 2フレーム目: 元へ戻る (点滅だった)。確認カウンタは0へ戻る。
    active, just = tr.update(
        _sig(_ev(), (1, 2), state=BoardState.GRAVITY_SETTLE),
        _sig(_ev(), (1, 2), state=BoardState.GRAVITY_SETTLE), _snap(), 1.133)
    assert active is True and just is False
    assert tr._abs_ended == [False, False]
    assert tr._abs_pending_frames == [0, 0]
    # 3フレーム目: 再び変化。ここから数え直しなので、まだ解除しない。
    active, just = tr.update(
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE), _snap(), 1.166)
    assert active is True and just is False
    assert tr.abs_end_stats["sessions"] == 1


def test_on_safety_valve_still_applies(monkeypatch) -> None:
    """絶対律で着弾待ちへ入った後も、安全弁は従来どおり効く。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    _settle_abs_baseline(tr)
    stuck = _snap(pending_p1=999, pending_p2=999)
    tr.update(_sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
              _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE), stuck, 1.1)
    active, just = tr.update(
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE), stuck, 1.133)
    assert active is True and just is False          # 着弾未完了なので延長
    late = 1.133 + vao.RESOLVED_HOLD_LANDING_MAX_WAIT_SEC + 0.01
    active, just = tr.update(
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE), stuck, late)
    assert active is False and just is True          # 安全弁で強制解放


def test_on_legacy_condition_still_works(monkeypatch) -> None:
    """ON でも旧条件 (両側 chain_event が None) は生きている。OR なので、
    絶対律が観測されないケースで解除が遅くなることはない。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    tr.update(
        _sig(None, (1, 2), state=BoardState.GRAVITY_SETTLE),
        _sig(None, (1, 2), state=BoardState.GRAVITY_SETTLE), _snap(), 1.0)
    active, just = tr.update(
        _sig(None, (1, 2), state=BoardState.GRAVITY_SETTLE),
        _sig(None, (1, 2), state=BoardState.GRAVITY_SETTLE), _snap(), 1.033)
    assert active is False and just is True
    assert tr.abs_end_stats["released_by_legacy"] == 1
    assert tr.abs_end_stats["released_by_abs"] == 0
    assert tr._abs_rearm_blocked is True


def test_on_legacy_none_gap_does_not_release_while_chain_state(monkeypatch) -> None:
    """ON時は両event Noneでも、物理CHAIN中の断片化gapなら解除しない。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    for t in (1.0, 1.033, 1.066):
        active, just = tr.update(
            _sig(None, (1, 2), state=BoardState.CHAIN),
            _sig(None, (1, 2), state=BoardState.CHAIN), _snap(), t)
        assert active is True and just is False
    assert tr.abs_end_stats["released_by_legacy"] == 0


def test_starting_placement_next_move_becomes_baseline_not_end(monkeypatch) -> None:
    """発火させた設置に伴うNEXT移動は、連鎖終了信号として使わない。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    for t in (0.1, 0.2, 1.0, 1.033):
        active, just = tr.update(
            _sig(_ev(), (3, 4)), _sig(_ev(), (3, 4)), _snap(), t)
        assert active is True and just is False
    assert tr._abs_next_at_start == [(3, 4), (3, 4)]
    assert tr._abs_ended == [False, False]


def test_next_change_waits_until_own_chain_state_exits(monkeypatch) -> None:
    """NEXT変化後もCHAIN中なら待ち、同じ変化をCHAIN離脱後に回収する。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    _settle_abs_baseline(tr)
    for t in (1.1, 1.133):
        active, just = tr.update(
            _sig(_ev(), (3, 4), state=BoardState.CHAIN),
            _sig(_ev(), (3, 4), state=BoardState.CHAIN), _snap(), t)
        assert active is True and just is False
        assert tr._abs_ended == [False, False]
    tr.update(
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE), _snap(), 1.2)
    active, just = tr.update(
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE), _snap(), 1.233)
    assert active is False and just is True


def test_chain_reentry_revokes_step_gap_end_candidate(monkeypatch) -> None:
    """5→6連鎖の段間で立った終了候補は、次段CHAINで撤回する。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    _settle_abs_baseline(tr)
    # 1Pだけ前段末尾で終了候補になる。2PはCHAIN継続なので保持は続く。
    for t in (1.1, 1.133):
        active, just = tr.update(
            _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
            _sig(_ev(), (1, 2), state=BoardState.CHAIN), _snap(), t)
        assert active is True and just is False
    assert tr._abs_ended == [True, False]
    # 次段へ再突入したため、1Pの終了候補を取り消す。
    active, just = tr.update(
        _sig(_ev(), (3, 4), state=BoardState.CHAIN),
        _sig(_ev(), (1, 2), state=BoardState.CHAIN), _snap(), 1.2)
    assert active is True and just is False
    assert tr._abs_ended == [False, False]
    assert tr.abs_end_stats["reopened_by_chain"] == [1, 0]
    # 本当の末尾で両側が離脱した後に初めて解除する。
    for t in (1.3, 1.333):
        active, just = tr.update(
            _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
            _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE), _snap(), t)
    assert active is False and just is True
    assert tr.abs_end_stats["released_by_abs"] == 1


def test_ojama_end_is_not_revoked_by_lagging_chain_state(monkeypatch) -> None:
    """着弾は絶対証拠なので、表示stateのCHAIN残留では撤回しない。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    _settle_abs_baseline(tr)
    # 1Pだけ着弾で終了。2Pは未終了なので、次フレームまで状態を観測できる。
    for t in (1.6, 1.633, 1.666):
        active, just = tr.update(
            _sig(_ev(), (1, 2), state=BoardState.CHAIN),
            _sig(_ev(), (1, 2), state=BoardState.CHAIN),
            _snap(dropped1=6), t)
        assert active is True and just is False
    assert tr._abs_ended == [True, False]
    assert tr._abs_end_kind == ["ojama", None]
    assert tr.abs_end_stats["reopened_by_chain"] == [0, 0]


def test_score_step_revokes_end_and_blocks_mid_chain_release(monkeypatch) -> None:
    """終了候補後の40点以上の得点段は、同一交換の物理継続証拠になる。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    _settle_abs_baseline(tr)
    # 1Pだけ終了候補。2Pは継続中なのでまだ実解除しない。
    for t in (1.1, 1.133):
        tr.update(
            _sig(_ev(), (3, 4), score=0, state=BoardState.GRAVITY_SETTLE),
            _sig(_ev(), (1, 2), score=0, state=BoardState.CHAIN), _snap(), t)
    assert tr._abs_ended == [True, False]
    # stateがCHAINへ戻らなくても、実得点段で終了候補を撤回する。
    active, just = tr.update(
        _sig(_ev(), (3, 4), score=40, state=BoardState.GRAVITY_SETTLE),
        _sig(_ev(), (1, 2), score=0, state=BoardState.CHAIN), _snap(), 1.2)
    assert active is True and just is False
    assert tr._abs_ended == [False, False]
    assert tr.abs_end_stats["reopened_by_chain"] == [1, 0]


def test_active_session_is_closed_by_formal_game_boundary(monkeypatch) -> None:
    """試合終了まで続いた保持は未完了でなく、境界終端として母数を閉じる。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    tr.on_game_boundary()
    assert tr.abs_end_stats["sessions"] == 1
    assert tr.abs_end_stats["ended_by_boundary"] == 1
    assert tr.abs_end_stats["total_boundaries"] == 1
    assert "試合境界で終端 1/1" in tr.abs_end_summary()


def test_absolute_end_requires_observed_chain_per_side(monkeypatch) -> None:
    """chain_event があっても、実stateでCHAIN未観測のsideは絶対律対象外。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    tr.update(
        _sig(_ev(), (1, 2), state=BoardState.STABLE),
        _sig(_ev(), (1, 2), state=BoardState.STABLE), _snap(), 0.0)
    for t in (1.0, 1.033, 1.1, 1.133):
        active, just = tr.update(
            _sig(_ev(), (3, 4), state=BoardState.STABLE),
            _sig(_ev(), (3, 4), state=BoardState.STABLE), _snap(), t)
        assert active is True and just is False
    assert tr._abs_saw_chain == [False, False]


def test_absolute_release_blocks_same_event_rearm_until_neutral(monkeypatch) -> None:
    """絶対律解除後、同じ交換を新セッションとして数え直さない。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    _settle_abs_baseline(tr)
    tr.update(_sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
              _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE), _snap(), 1.1)
    active, just = tr.update(
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
        _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE), _snap(), 1.133)
    assert active is False and just is True
    active, just = tr.update(
        _sig(_ev(), (3, 4)), _sig(_ev(), (3, 4)), _snap(), 1.166)
    assert active is False and just is False
    assert tr.abs_end_stats["sessions"] == 1
    tr.update(_sig(None, (3, 4)), _sig(None, (3, 4)), _snap(), 1.2)
    assert tr._abs_rearm_blocked is True
    tr.update(
        _sig(None, (3, 4)), _sig(None, (3, 4)), _snap(),
        1.2 + vao.RESOLVED_ABS_REARM_NEUTRAL_SEC + 0.01)
    assert tr._abs_rearm_blocked is False
    tr.update(
        _sig(_ev(), (5, 6)), _sig(_ev(), (5, 6)), _snap(),
        1.3 + vao.RESOLVED_ABS_REARM_NEUTRAL_SEC)
    assert tr.abs_end_stats["sessions"] == 2


def test_transient_neutral_gap_does_not_allow_rearm(monkeypatch) -> None:
    """両event Noneが一瞬だけ出ても、同じ交換を再武装しない。"""
    stub, _ = _stub_score_advantage()
    monkeypatch.setattr(vao, "_score_advantage", stub)
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    _fire(tr)
    _settle_abs_baseline(tr)
    for t in (1.1, 1.133):
        active, just = tr.update(
            _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE),
            _sig(_ev(), (3, 4), state=BoardState.GRAVITY_SETTLE), _snap(), t)
    assert active is False and just is True
    tr.update(_sig(None, (3, 4)), _sig(None, (3, 4)), _snap(), 1.166)
    active, just = tr.update(
        _sig(_ev(), (3, 4)), _sig(_ev(), (3, 4)), _snap(), 1.2)
    assert active is False and just is False
    assert tr.abs_end_stats["sessions"] == 1


def test_abs_end_summary_has_denominator() -> None:
    """サマリは必ず母数と並べて出す (0 が「起きなかった」か「測っていない」かを
    取り違えないため、memory feedback_zero_needs_denominator_2026-08-25)。"""
    tr = vao.ResolvedExchangeTracker(model=object(), enable_absolute_chain_end=True)
    s = tr.abs_end_summary()
    assert "保持セッション 0" in s
    assert "/0" in s          # 母数が併記されている


# ============================
# フラグの配線
# ============================


def test_flag_default_false_in_tracker() -> None:
    tr = vao.ResolvedExchangeTracker(model=object())
    assert tr._enable_absolute_chain_end is False


def test_flag_default_false_in_generate_signature() -> None:
    sig = inspect.signature(vao.generate)
    prm = sig.parameters["enable_resolved_absolute_chain_end"]
    assert prm.default is False


def test_cli_flag_defined() -> None:
    src = inspect.getsource(vao.main)
    assert "--resolved-absolute-chain-end" in src
    assert 'dest="enable_resolved_absolute_chain_end"' in src


def test_wired_at_both_construction_sites() -> None:
    """`ResolvedExchangeTracker` は通常時と試合境界リセット時の**2箇所**で
    構築される。片方への配線漏れは過去4回起きているため静的に固定する
    (memory feedback_use_single_source_for_flags_2026-08-22)。"""
    src = inspect.getsource(vao.generate)
    assert src.count(
        "enable_absolute_chain_end=enable_resolved_absolute_chain_end") == 2
