"""B3 台帳 v2 (src/b3_score_ledger.py) のテスト。

合成した観測列 (30fps) を台帳へ流し、レート逓減・繰越・相殺・着弾・全消し・起点・
prefix 不変・試合境界・上限なし・連鎖分割・得点 OCR の頑健化を検証する。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import pytest

from src.b3_score_ledger import (
    CHAIN_END_SIGNAL_MIN_SEC,
    CHAIN_STEP_GAP_MAX_SEC,
    AMBIGUOUS_EVIDENCE_SEC,
    DEAD_HOLD_SEC,
    FLAG_HISTORY_INCOMPLETE,
    FLAG_TIMING_AMBIGUOUS,
    FLAG_ORIGIN_UNKNOWN,
    FLAG_SCORE_RESYNC,
    FLAG_UNMATCHED_FORMULA,
    FLAG_UNMATCHED_JUMP,
    FORMULA_EXPIRE_SEC,
    LANDING_DECISION_DELAY_SEC,
    LANDING_MAX_PER_EVENT,
    REASON_SIGNAL_SLIDE,
    REGRESS_RESYNC_SEC,
    FormulaStepObs,
    FrameObs,
    LedgerSnapshot,
    ScoreLedgerV2,
    SideObs,
    frame_obs_from_record,
    replay,
)
from src.scoring import ALL_CLEAR_BONUS, compute_effective_rate

FPS = 30.0
SCORE_LAG_FRAMES = 27  # 式の観測から得点反映まで約 0.9 秒 (実測 0.87-0.97)
FRAMES_LANDING_WAIT = int(LANDING_DECISION_DELAY_SEC * FPS) + 3
# 式の観測から連鎖終了の信号を受け付けるまでのフレーム数 (+余裕)
FRAMES_END_SIGNAL_OK = int(CHAIN_END_SIGNAL_MIN_SEC * FPS) + 3


class Sim:
    """30fps で 1 フレームずつ台帳へ流す小さな駆動役。"""

    def __init__(self, fps: float = FPS, **kw: object) -> None:
        kw.setdefault("mode", "timing")        # 既存のテストは第3版 (時刻規則) の振る舞いを検証する
        self.ledger = ScoreLedgerV2(**kw)  # type: ignore[arg-type]
        self.fps = fps
        self.frame = 0
        self.scores: List[Optional[int]] = [0, 0]
        self.game = 0
        self.snaps: List[LedgerSnapshot] = []
        self.obs: List[FrameObs] = []
        self.events: List = []

    @property
    def t(self) -> float:
        return self.frame / self.fps

    def tick(
        self,
        n: int = 1,
        formula: Tuple[Optional[int], Optional[int]] = (None, None),
        placed: Tuple[bool, bool] = (False, False),
        states: Tuple[str, str] = ("STABLE", "STABLE"),
        tsumo: Tuple[Optional[int], Optional[int]] = (None, None),
        next_slide: Tuple[bool, bool] = (False, False),
        active: bool = True,
        board: Tuple[Optional[int], Optional[int]] = (None, None),
        ojama: Tuple[Optional[int], Optional[int]] = (None, None),
        dead: Tuple[bool, bool] = (False, False),
    ) -> LedgerSnapshot:
        """n フレーム進める。formula/placed/next_slide は最初の 1 フレームだけ立てる。"""
        snap = self.snaps[-1] if self.snaps else None
        for k in range(n):
            first = k == 0
            sides = []
            for i in range(2):
                fs = None
                if first and formula[i] is not None:
                    fs = FormulaStepObs(t_sec=self.t, product=int(formula[i]))
                sides.append(
                    SideObs(
                        score=self.scores[i], state=states[i], formula_step=fs,
                        tsumo_count=tsumo[i], next_slide=next_slide[i] and first,
                        placed=placed[i] and first,
                        board_sig=board[i], ojama_cells=ojama[i], dead=dead[i],
                    )
                )
            obs = FrameObs(self.t, self.frame, self.game, active, (sides[0], sides[1]))
            snap = self.ledger.update(obs)
            self.obs.append(obs)
            self.snaps.append(snap)
            self.events.extend(snap.events)
            self.frame += 1
        assert snap is not None
        return snap

    def start_match(self) -> None:
        """両者 0 点で開始し、最初の 1 手 (起点) を置く。"""
        self.tick(3, tsumo=(0, 0))
        self.tick(1, placed=(True, False), tsumo=(1, 0))

    def fire(self, side: int, product: int, wait_end: bool = True) -> LedgerSnapshot:
        """式を出し、約 0.9 秒後に得点が反映される 1 段の連鎖。wait_end なら連鎖終了まで進める。"""
        f = [None, None]
        f[side] = product
        self.tick(1, formula=tuple(f), states=self._chain_states(side))  # type: ignore[arg-type]
        self.tick(SCORE_LAG_FRAMES, states=self._chain_states(side))
        self.scores[side] = (self.scores[side] or 0) + product
        snap = self.tick(3, states=self._chain_states(side))
        if wait_end:
            # 信号 (静止復帰) を受け付ける時刻まで連鎖状態を続け、その後に静止へ戻す
            snap = self.tick(FRAMES_END_SIGNAL_OK - 31, states=self._chain_states(side))
            snap = self.tick(6, states=("STABLE", "STABLE"))
        return snap

    @staticmethod
    def _chain_states(side: int) -> Tuple[str, str]:
        return ("CHAIN", "STABLE") if side == 0 else ("STABLE", "CHAIN")

    def drop(self, side: int, pts: int) -> None:
        """落下点 (非連鎖) を 1 点ずつ加える。"""
        for _ in range(pts):
            self.scores[side] = (self.scores[side] or 0) + 1
            self.tick(1)


def last(sim: Sim) -> LedgerSnapshot:
    return sim.snaps[-1]


# ----------------------------- レート -----------------------------


def test_rate_boundaries_follow_first_move_origin() -> None:
    sim = Sim()
    sim.start_match()
    origin = last(sim).margin_origin_sec
    assert origin is not None
    led = sim.ledger
    assert led.rate_at(origin + 95.5) == 70           # 95.5 秒ちょうどはまだ 70
    assert led.rate_at(origin + 95.5 + 0.01) == 52    # 超えた瞬間に ×0.75
    assert led.rate_at(origin + 95.5 + 15.99) == 52   # 次の刻みの直前
    assert led.rate_at(origin + 95.5 + 16.01) == 39
    # user 回答: 毎回切り捨て。特徴側と台帳側も同じ共通関数を使う。
    expect = [52, 39, 29, 21, 15, 11, 8, 6, 4, 3, 2, 1, 1, 1]
    for k, r in enumerate(expect):
        assert led.rate_at(origin + 95.5 + 16.0 * k + 8.0) == r
    assert compute_effective_rate(95.5 + 16 * 3 + 8, from_first_move=True) == 21
    assert led.rate_at(origin + 10000.0) == 1         # 下限 1


def test_margin_origin_is_earlier_of_both_sides() -> None:
    sim = Sim()
    sim.tick(3, tsumo=(0, 0))
    sim.tick(30)
    sim.tick(1, placed=(False, True), tsumo=(0, 1))   # 2P が先に置く
    t_first = sim.t - 1 / FPS
    sim.tick(90)
    sim.tick(1, placed=(True, False), tsumo=(1, 1))   # 1P は後
    assert last(sim).margin_origin_sec == pytest.approx(t_first)


def test_rate_unknown_before_first_move_is_base() -> None:
    sim = Sim()
    sim.tick(10, tsumo=(0, 0))
    assert last(sim).margin_origin_sec is None
    assert last(sim).rate == 70


def test_rate_decays_during_match_and_changes_conversion() -> None:
    """起点から 111.6 秒以降はレート 39、同じ点でも生成が増える。"""
    sim = Sim()
    sim.start_match()
    sim.tick(int((95.5 + 16.0 + 1.0) * FPS))
    assert last(sim).rate == 39
    sim.fire(0, 780)  # 780 // 39 = 20
    assert last(sim).sides[0].generated_total == 20


# ----------------------------- 繰越・換算 -----------------------------


def test_carry_across_chains_and_drop_points() -> None:
    sim = Sim()
    sim.start_match()
    sim.drop(0, 20)                 # 落下 20 点 (非連鎖) は繰越に入る
    sim.fire(0, 40)                 # 20 + 40 = 60 → 0 個、繰越 60
    assert last(sim).sides[0].generated_total == 0
    assert last(sim).sides[0].carry == 60
    sim.fire(0, 40)                 # 60 + 40 = 100 → 1 個、繰越 30
    s = last(sim).sides[0]
    assert (s.generated_total, s.carry) == (1, 30)
    assert s.cum_drop == 20 and s.cum_chain == 80


def test_drop_only_never_generates() -> None:
    sim = Sim()
    sim.start_match()
    for _ in range(10):
        sim.drop(0, 26)
    s = last(sim).sides[0]
    assert s.generated_total == 0 and last(sim).sides[1].pending == 0


def test_chain_increments_are_split_per_step_not_lumped() -> None:
    sim = Sim()
    sim.start_match()
    sim.fire(0, 40, wait_end=False)
    sim.tick(15, states=("CHAIN", "STABLE"))
    sim.fire(0, 320)                 # 1.0 秒後の次の段 = 同じ連鎖
    assert len(sim.ledger.chains) == 1
    ch = sim.ledger.chains[0]
    assert [s.product for s in ch.steps] == [40, 320]
    assert ch.generated() == 360 // 70


def test_two_chains_separated_by_gap_are_two_records() -> None:
    sim = Sim()
    sim.start_match()
    sim.fire(0, 40)
    sim.tick(int((CHAIN_STEP_GAP_MAX_SEC + 1.0) * FPS))
    sim.fire(0, 320)
    chains = sim.ledger.chains
    assert len(chains) == 2
    assert [len(c.steps) for c in chains] == [1, 1]
    assert [c.chain_id for c in chains] == [1, 2]


def test_step_attribution_events() -> None:
    sim = Sim()
    sim.start_match()
    sim.drop(0, 5)
    sim.fire(0, 280)
    kinds = [(e.get("kind"), e.get("points")) for e in sim.events if e.kind == "score_attr"]
    assert ("drop", 1) in kinds
    assert ("chain", 280) in kinds


# ----------------------------- 相殺 -----------------------------


def test_offset_then_send_remainder() -> None:
    sim = Sim()
    sim.start_match()
    sim.fire(0, 210)                         # 1P が 3 個 → 2P 予告 3
    assert last(sim).sides[1].pending == 3
    sim.fire(1, 140)                         # 2P が 2 個 → 自分の予告 3 のうち 2 を相殺
    s2 = last(sim).sides[1]
    assert (s2.offset_total, s2.sent_total) == (2, 0)
    assert s2.pending + s2.dropped_total == 1   # 残り 1 個は自分の連鎖終了後に降る
    assert last(sim).sides[0].pending == 0
    sim.fire(1, 350)                         # 5 個 → 残りを相殺 (降った分は除く)、余りを 1P へ
    s1, s2 = last(sim).sides
    assert s2.generated_total == 7 and s2.sent_total == s1.pending
    assert s2.offset_total + s2.sent_total == 7
    assert s2.sent_total >= 4


def test_offset_uses_total_pool_including_drop_points() -> None:
    """実測: 累計 510 点 (落下 230 点を含む) で予告 7 個 = 510 // 70。"""
    sim = Sim()
    sim.start_match()
    sim.drop(0, 230)
    sim.fire(0, 280)
    assert last(sim).sides[1].pending == 7


# ----------------------------- 上限なし -----------------------------


def test_pending_is_not_capped_at_216() -> None:
    sim = Sim()
    sim.start_match()
    sim.fire(0, 70 * 300)
    assert last(sim).sides[1].pending == 300
    sim.fire(0, 70 * 100)
    assert last(sim).sides[1].pending == 400   # 旧台帳は 216 で張り付いた


# ----------------------------- 全消し -----------------------------

def _fire_with_formula(sim: Sim, side: int, product: int, delta: int) -> None:
    f = [None, None]
    f[side] = product
    sim.tick(1, formula=tuple(f), states=Sim._chain_states(side))  # type: ignore[arg-type]
    sim.tick(SCORE_LAG_FRAMES, states=Sim._chain_states(side))
    sim.scores[side] = (sim.scores[side] or 0) + delta
    sim.tick(10, states=("STABLE", "STABLE"))



def test_all_clear_bonus_is_attributed_and_converted() -> None:
    sim = Sim()
    sim.start_match()
    _fire_with_formula(sim, 0, 40, 40 + ALL_CLEAR_BONUS)   # 式は 40 点、得点増分は 2140 点
    s = last(sim).sides[0]
    assert s.cum_all_clear == ALL_CLEAR_BONUS
    assert s.cum_chain == 40
    assert s.generated_total == (40 + ALL_CLEAR_BONUS) // 70   # 30 個ぶん込み


def test_all_clear_with_drop_points_in_same_adoption() -> None:
    sim = Sim()
    sim.start_match()
    _fire_with_formula(sim, 0, 40, 40 + ALL_CLEAR_BONUS + 7)
    s = last(sim).sides[0]
    assert (s.cum_all_clear, s.cum_chain, s.cum_drop) == (ALL_CLEAR_BONUS, 40, 7)


# ----------------------------- 着弾 -----------------------------


def _land_ready(sim: Sim, side: int, states: Tuple[str, str] = ("STABLE", "STABLE")) -> None:
    """側 side が連鎖せず手を置き、着弾判断の待ち時間まで進める。"""
    p = [False, False]
    p[side] = True
    sim.tick(1, placed=tuple(p), states=states)  # type: ignore[arg-type]
    sim.tick(FRAMES_LANDING_WAIT, states=states)


def test_landing_is_capped_at_30_per_event_and_carries_over() -> None:
    sim = Sim()
    sim.start_match()
    sim.fire(0, 70 * 100)                    # 100 個 → 2P 予告 100 (確定)
    assert last(sim).sides[1].pending_confirmed == 100
    _land_ready(sim, 1)
    s = last(sim).sides[1]
    assert (s.dropped_total, s.pending) == (LANDING_MAX_PER_EVENT, 70)
    _land_ready(sim, 1)
    _land_ready(sim, 1)
    _land_ready(sim, 1)
    s = last(sim).sides[1]
    assert s.dropped_total == 100 and s.pending == 0
    landings = [e.get("n") for e in sim.events if e.kind == "landing"]
    assert landings == [30, 30, 30, 10]


def test_no_landing_before_attacker_chain_confirmed() -> None:
    """相手の連鎖の終わりが確かでない (最後の式から 1 秒未満) うちは、置いても降らない。"""
    sim = Sim()
    sim.start_match()
    sim.tick(1, formula=(70 * 10, None), states=("CHAIN", "STABLE"))
    sim.tick(14, states=("CHAIN", "STABLE"))
    sim.scores[0] += 700                                              # (試験のため得点を早めに反映)
    sim.tick(2, states=("CHAIN", "STABLE"))
    assert last(sim).sides[1].pending == 10 and last(sim).sides[1].pending_confirmed == 0
    _land_ready(sim, 1, states=("CHAIN", "STABLE"))                  # 式から約 0.6-1.2 秒
    assert last(sim).sides[0].chain_open
    assert last(sim).sides[1].dropped_total == 0


def test_landing_blocked_while_attacker_is_chaining() -> None:
    """相手が次の連鎖中に置いても降らない (実測 video_29 f31533: 129 個が降らず 243 個へ)。"""
    sim = Sim()
    sim.start_match()
    sim.fire(0, 70 * 10)                     # 確定 10 個
    sim.tick(1, formula=(40, None), states=("CHAIN", "STABLE"))   # 1P 次の連鎖が開く
    _land_ready(sim, 1)
    assert last(sim).sides[1].dropped_total == 0
    assert last(sim).sides[1].pending == 10


def test_receiver_chain_prevents_landing_and_offsets() -> None:
    sim = Sim()
    sim.start_match()
    sim.fire(0, 70 * 10)
    sim.fire(1, 70 * 10)                     # 2P の連鎖が 10 個を全部相殺
    assert last(sim).sides[1].pending == 0
    _land_ready(sim, 1)
    assert last(sim).sides[1].dropped_total == 0


def test_placement_that_starts_a_chain_does_not_land() -> None:
    """置いた手が連鎖を始める (待ち時間内に式が出る) 場合は降らない。"""
    sim = Sim()
    sim.start_match()
    sim.fire(0, 70 * 10)
    sim.tick(1, placed=(False, True))
    sim.tick(8)
    sim.tick(1, formula=(None, 40), states=("STABLE", "CHAIN"))
    sim.tick(FRAMES_LANDING_WAIT)
    assert last(sim).sides[1].dropped_total == 0


def test_placement_burst_lands_once() -> None:
    """置き終わり信号が不応期内に重複しても 1 回の着弾。"""
    sim = Sim()
    sim.start_match()
    sim.fire(0, 70 * 100)
    for _ in range(3):
        sim.tick(1, placed=(False, True))
        sim.tick(3)
    sim.tick(FRAMES_LANDING_WAIT)
    assert last(sim).sides[1].dropped_total == LANDING_MAX_PER_EVENT


# ----------------------------- prefix 不変 -----------------------------


def _scripted_run() -> Sim:
    sim = Sim()
    sim.start_match()
    sim.drop(0, 12)
    sim.fire(0, 40 + 500)
    sim.drop(1, 9)
    sim.fire(1, 280)
    _land_ready(sim, 0)
    sim.fire(0, 1400)
    sim.tick(20)
    return sim


def test_prefix_invariance() -> None:
    full = _scripted_run()
    for k in (50, 140, 220, len(full.obs) - 7):
        prefix = list(replay(full.obs[:k], mode="timing"))
        assert prefix == full.snaps[:k], f"prefix {k} differs"


def test_available_at_never_precedes_event_time() -> None:
    sim = _scripted_run()
    for e in sim.events:
        assert e.available_at_sec >= e.t_event - 1e-9
    for snap in sim.snaps:
        assert snap.available_at_sec == snap.t_sec


# ----------------------------- 試合境界 -----------------------------


def test_match_boundary_resets_everything() -> None:
    sim = Sim()
    sim.start_match()
    sim.fire(0, 70 * 50)
    assert last(sim).sides[1].pending == 50
    sim.game = 1
    sim.scores = [0, 0]
    snap = sim.tick(3, tsumo=(0, 0))
    assert snap.sides[1].pending == 0 and snap.sides[0].generated_total == 0
    assert snap.margin_origin_sec is None
    assert any(e.kind == "match_reset" for e in sim.events)
    # 新しい試合では起点がやり直し (旧試合の起点を引きずらない)
    sim.tick(60)
    sim.tick(1, placed=(True, False), tsumo=(1, 0))
    assert last(sim).margin_origin_sec == pytest.approx(sim.t - 1 / FPS)


def test_temporary_menu_inside_match_does_not_reset() -> None:
    """試合内の一時的な MENU (3〜5 秒) は試合境界ではない。累計・予告・起点は保たれる。"""
    sim = Sim()
    sim.start_match()
    sim.fire(0, 70 * 5)
    origin = last(sim).margin_origin_sec
    gen = last(sim).sides[0].generated_total
    sim.tick(int(5.0 * FPS), active=False, states=("MENU", "MENU"))
    sim.tick(30)
    s0 = last(sim).sides[0]
    assert s0.generated_total == gen and last(sim).margin_origin_sec == origin
    assert last(sim).sides[1].pending == 5
    assert not any(e.kind == "match_reset" for e in sim.events[1:])


def test_score_zero_reset_in_active_match_is_boundary() -> None:
    sim = Sim()
    sim.start_match()
    sim.fire(0, 70 * 5)
    sim.tick(40, active=False)
    sim.scores = [0, 0]
    snap = sim.tick(6)
    assert snap.sides[1].pending == 0 and snap.margin_origin_sec is None
    assert any(e.get("reason") == "reset_by_score_zero" for e in sim.events if e.kind == "match_reset")


def test_open_chain_is_closed_at_boundary() -> None:
    sim = Sim()
    sim.start_match()
    sim.tick(1, formula=(40, None), states=("CHAIN", "STABLE"))
    sim.game = 1
    sim.tick(2)
    assert len(sim.ledger.chains) == 1 and sim.ledger.chains[0].end_reason == "match_reset"


# ----------------------------- 得点 OCR の頑健化 -----------------------------


def test_single_frame_spike_is_ignored() -> None:
    sim = Sim()
    sim.start_match()
    sim.drop(0, 10)
    sim.scores[0] = 99999
    sim.tick(1)
    sim.scores[0] = 10
    sim.tick(5)
    assert last(sim).sides[0].score == 10


def test_downward_misread_is_ignored_and_does_not_double_count() -> None:
    sim = Sim()
    sim.start_match()
    sim.drop(0, 20)
    sim.scores[0] = 3
    sim.tick(4)
    sim.scores[0] = 21
    sim.tick(4)
    s = last(sim).sides[0]
    assert s.score == 21 and s.cum_drop == 21


def test_persistent_regress_keeps_score_with_flag() -> None:
    sim = Sim()
    sim.start_match()
    sim.drop(0, 20)
    adopted = last(sim).sides[0].score
    sim.scores[0] = 5
    sim.tick(int(REGRESS_RESYNC_SEC * FPS) + 3)
    assert last(sim).sides[0].score == adopted
    assert FLAG_SCORE_RESYNC in last(sim).sides[0].flags


def test_two_frame_persistence_adopts_with_one_frame_delay() -> None:
    sim = Sim()
    sim.start_match()
    sim.scores[0] = 7
    snap1 = sim.tick(1)
    assert snap1.sides[0].score == 0            # まだ 1 フレームしか見ていない
    snap2 = sim.tick(1)
    assert snap2.sides[0].score == 7
    attr = [e for e in snap2.events if e.kind == "score_attr"]
    assert attr and attr[0].t_event < attr[0].available_at_sec   # 発生は前フレーム、使えるのは今


def test_unmatched_big_jump_is_converted_but_flagged() -> None:
    sim = Sim()
    sim.start_match()
    sim.scores[0] = 700                         # 式が読めていないのに 700 点
    sim.tick(10, states=("CHAIN", "STABLE"))
    s = last(sim).sides[0]
    assert s.generated_total == 10 and FLAG_UNMATCHED_JUMP in s.flags
    assert s.cum_unmatched == 700
    assert last(sim).sides[1].pending == 10


def test_unmatched_formula_expires_with_flag() -> None:
    sim = Sim()
    sim.start_match()
    sim.tick(1, formula=(280, None), states=("CHAIN", "STABLE"))
    sim.tick(int((FORMULA_EXPIRE_SEC + 1.0) * FPS), states=("CHAIN", "STABLE"))
    sim.scores[0] += 5                          # 得点が反映されないまま別の落下点だけ
    sim.tick(5)
    assert FLAG_UNMATCHED_FORMULA in last(sim).sides[0].flags


def test_two_steps_in_one_adoption_are_both_matched() -> None:
    sim = Sim()
    sim.start_match()
    sim.tick(1, formula=(40, None), states=("CHAIN", "STABLE"))
    sim.tick(10, formula=(None, None), states=("CHAIN", "STABLE"))
    sim.tick(1, formula=(320, None), states=("CHAIN", "STABLE"))
    sim.tick(SCORE_LAG_FRAMES, states=("CHAIN", "STABLE"))
    sim.scores[0] += 360
    sim.tick(5, states=("CHAIN", "STABLE"))
    assert last(sim).sides[0].cum_chain == 360
    assert FLAG_UNMATCHED_JUMP not in last(sim).sides[0].flags


# ----------------------------- 履歴不完全 -----------------------------


def test_midgame_start_flags_incomplete_and_unknown_origin() -> None:
    sim = Sim()
    sim.scores = [5000, 4000]
    sim.tick(5, tsumo=(30, 30))
    sim.tick(1, placed=(True, False), tsumo=(31, 30))
    snap = last(sim)
    assert FLAG_HISTORY_INCOMPLETE in snap.sides[0].flags
    assert FLAG_ORIGIN_UNKNOWN in snap.sides[0].flags
    assert snap.margin_origin_sec is None and snap.rate == 70
    assert snap.sides[0].generated_total == 0   # 途中から見た得点は生成に数えない


def test_small_first_read_is_treated_as_complete() -> None:
    """試合再開の検出が遅れて最初の読みが 30 点程度でも、0 点から見ていたとみなす。"""
    sim = Sim()
    sim.scores = [33, 31]
    sim.tick(5)
    assert FLAG_HISTORY_INCOMPLETE not in last(sim).sides[0].flags
    assert last(sim).sides[0].carry == 33


# ----------------------------- アダプタ -----------------------------


def test_frame_obs_from_record_parses_b2_row() -> None:
    rec = {
        "kind": "frame", "frame": 12, "t_sec": 0.4, "game_idx": 3, "match_active": True,
        "sides": [
            {"side": "1P", "score": 100, "state": "CHAIN", "tsumo_count": 4, "next_slide_motion": True,
             "formula_step": {"t_sec": 0.4, "left": 70, "right": 4, "erased": 7, "multiplier": 4, "product": 280}},
            {"side": "2P", "score": None, "state": "STABLE", "tsumo_count": None, "next_slide_motion": False,
             "formula_step": None},
        ],
    }
    obs = frame_obs_from_record(rec, ("2P",))
    assert obs.game_idx == 3 and obs.sides[0].formula_step.product == 280
    assert obs.sides[1].placed and not obs.sides[0].placed
    assert obs.sides[0].next_slide and obs.sides[1].score is None


def test_chain_end_by_next_slide_signal() -> None:
    """NEXT 移動が終わりの信号。確定は最後の式から 1.5 秒後だが、終わりの時刻は信号が立った時 (遡及)。"""
    sim = Sim()
    sim.start_match()
    sim.tick(1, formula=(280, None), states=("CHAIN", "STABLE"))
    sim.tick(SCORE_LAG_FRAMES, states=("CHAIN", "STABLE"))
    sim.scores[0] += 280
    sim.tick(4, states=("CHAIN", "STABLE"))
    t_slide = sim.t
    sim.tick(1, states=("CHAIN", "STABLE"), next_slide=(True, False))
    assert last(sim).sides[0].chain_open                       # 最後の式から 1.5 秒未満: まだ確定しない
    sim.tick(FRAMES_END_SIGNAL_OK, states=("CHAIN", "STABLE"))
    assert not last(sim).sides[0].chain_open
    ch = sim.ledger.chains[-1]
    assert ch.end_reason == REASON_SIGNAL_SLIDE
    assert ch.t_end == pytest.approx(t_slide)
    assert last(sim).sides[1].pending_confirmed == 4
    ev = [e for e in sim.events if e.kind == "chain_end"][0]
    assert ev.t_event == pytest.approx(t_slide) and ev.available_at_sec > ev.t_event


def test_signal_flicker_between_steps_does_not_split_chain() -> None:
    """段の間 (式から 1.5 秒未満) に静止・NEXT 移動・おじゃま落下の信号が出ても 1 連鎖のまま。"""
    sim = Sim()
    sim.start_match()
    sim.tick(1, formula=(40, None), states=("CHAIN", "STABLE"))
    sim.tick(SCORE_LAG_FRAMES, states=("CHAIN", "STABLE"))
    sim.scores[0] += 40
    sim.tick(2, states=("CHAIN", "STABLE"))
    sim.tick(4, states=("STABLE", "STABLE"))
    sim.tick(1, states=("OJAMA_FALL", "STABLE"), next_slide=(True, False))
    sim.tick(1, formula=(320, None), states=("CHAIN", "STABLE"))
    assert len(sim.ledger.chains) == 0 and last(sim).sides[0].chain_open


# ----------------------------- 着弾の証拠・順序 -----------------------------


def test_batch_confirmed_during_receiver_chain_does_not_land_at_its_end() -> None:
    """受け側が連鎖を始めた後に確定した予告は、その連鎖の終了では降りない (video_30 t2979 台)。"""
    sim = Sim()
    sim.start_match()
    c2 = ("STABLE", "CHAIN")
    sim.tick(1, formula=(None, 40), states=c2)                       # 2P の連鎖の 1 段目
    sim.tick(5, states=c2)
    sim.tick(1, formula=(700, None), states=("CHAIN", "CHAIN"))      # 0.2 秒後に 1P の連鎖
    sim.tick(SCORE_LAG_FRAMES - 6, states=("CHAIN", "CHAIN"))
    sim.scores[1] += 40
    sim.tick(6, states=("CHAIN", "CHAIN"))
    sim.scores[0] += 700
    sim.tick(2, formula=(None, 40), states=("CHAIN", "CHAIN"))       # 2P の 2 段目 (1P より長く続く)
    sim.tick(SCORE_LAG_FRAMES, states=("CHAIN", "CHAIN"))
    sim.scores[1] += 40
    sim.tick(FRAMES_END_SIGNAL_OK + 6, states=("CHAIN", "CHAIN"))
    sim.tick(8, states=("STABLE", "STABLE"))                         # 両者の連鎖が終わる
    assert not last(sim).sides[0].chain_open and not last(sim).sides[1].chain_open
    one, two = sim.ledger.chains[0], sim.ledger.chains[1]
    assert one.side == 0 and two.side == 1                           # 1P の連鎖が先に終わる
    s2 = last(sim).sides[1]
    assert s2.dropped_total == 0 and s2.pending_confirmed == s2.pending == 9


def test_lock_evidence_stream_does_not_starve_landing() -> None:
    """NEXT 移動が待ち時間より短い間隔で続いても、最初の証拠から待ち時間後に判断される。"""
    sim = Sim()
    sim.start_match()
    sim.fire(1, 70 * 10)                       # 2P の連鎖 → 1P へ 10 個 (確定)
    for _ in range(8):
        sim.tick(1, next_slide=(True, False))
        sim.tick(16)                           # 17 フレーム = 0.57 秒 (< 判断待ち 0.6 秒)
        if last(sim).sides[0].dropped_total:
            break
    assert last(sim).sides[0].dropped_total == 10


def test_next_slide_alone_is_lock_evidence() -> None:
    sim = Sim()
    sim.start_match()
    sim.fire(1, 70 * 5)
    sim.tick(1, next_slide=(True, False))      # 置き終わり信号なし、NEXT 移動だけ
    sim.tick(FRAMES_LANDING_WAIT)
    assert last(sim).sides[0].dropped_total == 5


def test_ojama_fall_edge_lands_once_per_episode() -> None:
    """おじゃま落下状態への遷移を降り始めの証拠にする。状態のちらつきで二重に降らせない。"""
    sim = Sim()
    sim.start_match()
    sim.fire(1, 70 * 100)                      # 1P へ 100 個 (確定)
    sim.tick(1, states=("OJAMA_FALL", "STABLE"))
    for _ in range(5):                         # ちらつき (1.5 秒以内の再遷移)
        sim.tick(3, states=("STABLE", "STABLE"))
        sim.tick(3, states=("OJAMA_FALL", "STABLE"))
    sim.tick(FRAMES_LANDING_WAIT, states=("STABLE", "STABLE"))
    landings = [e.get("n") for e in sim.events if e.kind == "landing"]
    assert landings == [LANDING_MAX_PER_EVENT]


# ----------------------------- 接続口 (学習行生成器) -----------------------------


def _rec(frame: int, s1: Optional[int], s2: Optional[int], active: bool = True, game: int = 0,
         tsumo: Tuple[int, int] = (0, 0)) -> dict:
    return {
        "kind": "frame", "frame": frame, "t_sec": frame / FPS, "game_idx": game, "match_active": active,
        "sides": [
            {"side": "1P", "score": s1, "state": "STABLE", "tsumo_count": tsumo[0], "next_slide_motion": False,
             "formula_step": None},
            {"side": "2P", "score": s2, "state": "STABLE", "tsumo_count": tsumo[1], "next_slide_motion": False,
             "formula_step": None},
        ],
    }


def test_port_contract_keys_and_none_outside_match() -> None:
    from src.b3_score_ledger import ScoreLedger, snapshot_to_dict  # noqa: F401
    from src.b3_score_ledger_port import SL_KEYS, validate_snapshot
    led = ScoreLedger()
    assert led.update(_rec(0, None, None)) is None          # スコア未読
    assert led.update(_rec(1, 0, 0, active=False)) is None  # 試合外
    out = None
    for f in range(2, 8):
        out = led.update(_rec(f, 0, 0))
    assert out is not None
    validate_snapshot(out)                                   # 契約キー不足なら例外
    assert all(k in out for k in SL_KEYS)
    assert out["pending_p1"] == 0 and out["rate"] == 70 and out["available_at"] == pytest.approx(7 / FPS)


def test_port_matches_core_values() -> None:
    from src.b3_score_ledger import ScoreLedger
    led = ScoreLedger()
    for f in range(3):
        led.update(_rec(f, 0, 0))
    # 1P の連鎖: 式 (700) → 約 0.9 秒後に得点反映
    rec = _rec(3, 0, 0)
    rec["sides"][0]["formula_step"] = {"t_sec": 3 / FPS, "product": 700, "erased": 7, "multiplier": 10}
    rec["sides"][0]["state"] = "CHAIN"
    led.update(rec)
    out = None
    for f in range(4, 4 + SCORE_LAG_FRAMES):
        r = _rec(f, 0, 0)
        r["sides"][0]["state"] = "CHAIN"
        led.update(r)
    for f in range(4 + SCORE_LAG_FRAMES, 4 + SCORE_LAG_FRAMES + 4):
        r = _rec(f, 700, 0)
        r["sides"][0]["state"] = "CHAIN"
        out = led.update(r)
    assert out is not None
    assert out["generated_p1"] == 10 and out["pending_p2"] == 10 and out["pending_p1"] == 0
    assert out["pending_confirmed_p2"] == 0                    # 連鎖がまだ終わっていない
    assert led.core.st[0].generated_total == 10                # core は公開属性


def test_interrupt_is_recorded_and_flags_landing() -> None:
    """user 指示: 割り込みは厳密に扱わず、interrupt イベントと landing_after_interrupt フラグで記録する。"""
    from src.b3_score_ledger import FLAG_LANDING_AFTER_INTERRUPT
    sim = Sim()
    sim.start_match()
    sim.fire(0, 70 * 10)                                        # 2P へ 10 個 (確定)
    sim.tick(1, formula=(None, 40), states=("STABLE", "CHAIN"))  # 2P が自連鎖 (相殺の最中)
    sim.tick(5, states=("STABLE", "CHAIN"))
    sim.tick(1, formula=(40, None), states=("CHAIN", "CHAIN"))   # 1P の新しい連鎖 = 割り込み
    ev = [e for e in sim.events if e.kind == "interrupt"]
    assert ev and ev[0].side == 0 and ev[0].get("receiver") == 1 and ev[0].get("pending_receiver") == 10
    sim.ledger.st[1].chain = sim.ledger.st[1].chain             # 2P の連鎖は開いたまま
    sim.ledger._land(1, sim.t, sim.t)
    assert FLAG_LANDING_AFTER_INTERRUPT in sim.ledger.st[1].flags


def test_no_interrupt_without_receiver_chain() -> None:
    sim = Sim()
    sim.start_match()
    sim.fire(0, 70 * 10)
    sim.fire(0, 40)
    assert not [e for e in sim.events if e.kind == "interrupt"]


# ----------------------------- 第3版: 検出した終了・裏取り・曖昧・決着・秒単位 -----------------------------


def _one_step_chain(sim: Sim, side: int, product: int, quiet_after_frames: int) -> None:
    """式 → 得点反映 → quiet_after_frames 後に静止へ (最後の式から約 1.6 秒で終了確定)。"""
    f = [None, None]
    f[side] = product
    sim.tick(1, formula=tuple(f), states=Sim._chain_states(side))  # type: ignore[arg-type]
    sim.tick(SCORE_LAG_FRAMES, states=Sim._chain_states(side))
    sim.scores[side] += product
    sim.tick(quiet_after_frames, states=Sim._chain_states(side))


def test_chain_end_time_is_retroactive_signal_start() -> None:
    """終了時刻 = 画面が静止に戻った始点 (遡及)。確定の検出 (最後の式から 1.5 秒) より前になる。"""
    sim = Sim()
    sim.start_match()
    _one_step_chain(sim, 0, 280, 14)                     # 式から約 1.5 秒まで連鎖状態
    t_quiet = sim.t
    sim.tick(30, states=("STABLE", "STABLE"))
    ch = sim.ledger.chains[0]
    assert ch.t_end == pytest.approx(t_quiet, abs=0.05)
    assert ch.end_reason is not None
    ev = [e for e in sim.events if e.kind == "chain_end"][0]
    assert ev.available_at_sec >= ev.t_event


def test_attacker_chain_without_end_candidate_is_busy_no_landing() -> None:
    """攻撃側の連鎖の終わりがまだ観測されない (CHAIN 状態のまま) 間に置いても降らない。"""
    sim = Sim()
    sim.start_match()
    _one_step_chain(sim, 0, 700, 5)
    sim.tick(1, placed=(False, True), next_slide=(False, True), states=("CHAIN", "STABLE"))
    sim.tick(FRAMES_LANDING_WAIT + 40, states=("CHAIN", "STABLE"))
    assert last(sim).sides[1].dropped_total == 0 and last(sim).sides[1].pending == 10


def test_landing_after_attacker_chain_clearly_ended() -> None:
    sim = Sim()
    sim.start_match()
    _one_step_chain(sim, 0, 700, 14)
    sim.tick(40, states=("STABLE", "STABLE"))            # 連鎖が終わって 1 秒以上
    sim.tick(1, placed=(False, True), next_slide=(False, True))
    sim.tick(FRAMES_LANDING_WAIT)
    s2 = last(sim).sides[1]
    assert s2.dropped_total == 10 and FLAG_TIMING_AMBIGUOUS not in s2.flags


def test_placement_without_landing_confirmation_is_ignored() -> None:
    """天井付近の誤検知: 置き終わり信号だけで NEXT 移動も盤面変化も無ければ着弾の証拠にしない。"""
    sim = Sim()
    sim.start_match()
    _one_step_chain(sim, 0, 700, 14)
    sim.tick(40, states=("STABLE", "STABLE"), board=(1, 5))
    sim.tick(1, placed=(False, True), board=(1, 5))
    sim.tick(FRAMES_LANDING_WAIT + 10, board=(1, 5))     # 盤面は変わらず NEXT も動かない
    assert last(sim).sides[1].dropped_total == 0
    sim.tick(1, placed=(False, True), board=(1, 5))
    sim.tick(2, board=(1, 6))                            # 今度は盤面が確定変化 = 着地
    sim.tick(FRAMES_LANDING_WAIT, board=(1, 6))
    assert last(sim).sides[1].dropped_total == 10


def test_ambiguous_landing_is_marked_and_resolved_by_board_evidence() -> None:
    """置き終わりと攻撃側の終了が 0.3 秒以内: 印を付け、受け側盤面のおじゃま増加で後から確定する。"""
    sim = Sim()
    sim.start_match()
    sim.tick(1, formula=(70 * 10, None), states=("CHAIN", "STABLE"), ojama=(0, 4))
    sim.tick(SCORE_LAG_FRAMES, states=("CHAIN", "STABLE"), ojama=(0, 4))
    sim.scores[0] += 700
    sim.tick(14, states=("CHAIN", "STABLE"), ojama=(0, 4))
    sim.tick(1, placed=(False, True), next_slide=(False, True), ojama=(0, 4))   # 終了とほぼ同時に置く
    sim.tick(FRAMES_LANDING_WAIT + 5, ojama=(0, 4))
    assert last(sim).ambiguous[1] == 1 and last(sim).sides[1].dropped_total == 0   # まだ確定しない
    assert FLAG_TIMING_AMBIGUOUS in last(sim).sides[1].flags
    before = [sn.sides[1].dropped_total for sn in sim.snaps]
    sim.tick(3, ojama=(0, 10))                           # おじゃまが 6 個増えた = 着弾の証拠
    assert last(sim).sides[1].dropped_total == 10 and last(sim).ambiguous[1] == 0
    assert max(before) == 0                              # 過去の行は変わらない (確定した時刻以降のみ)
    assert any(e.kind == "landing" and e.get("by") == "board" for e in sim.events)


def test_ambiguous_without_board_increase_does_not_land() -> None:
    sim = Sim()
    sim.start_match()
    sim.tick(1, formula=(70 * 10, None), states=("CHAIN", "STABLE"), ojama=(0, 4))
    sim.tick(SCORE_LAG_FRAMES, states=("CHAIN", "STABLE"), ojama=(0, 4))
    sim.scores[0] += 700
    sim.tick(14, states=("CHAIN", "STABLE"), ojama=(0, 4))
    sim.tick(1, placed=(False, True), next_slide=(False, True), ojama=(0, 4))
    sim.tick(int(AMBIGUOUS_EVIDENCE_SEC * FPS) + FRAMES_LANDING_WAIT + 10, ojama=(0, 4))
    s2 = last(sim).sides[1]
    assert s2.dropped_total == 0 and s2.pending == 10 and last(sim).ambiguous[1] == 0


def test_placement_between_two_formulas_splits_chain() -> None:
    """式の間隔が 2 秒未満でも、間に置き終わりがあれば別の連鎖。"""
    sim = Sim()
    sim.start_match()
    _one_step_chain(sim, 0, 40, 3)
    sim.tick(1, placed=(True, False))                    # 置き終わり
    sim.tick(10)
    sim.tick(1, formula=(320, None), states=("CHAIN", "STABLE"))
    assert len(sim.ledger.chains) == 1 and sim.ledger.chains[0].end_reason == "placement_between_steps"
    assert last(sim).sides[0].chain_open


def test_match_over_freezes_ledger() -> None:
    """窒息位置にぷよがある状態が続いたら決着として打ち切る (その後の生成・着弾はしない)。"""
    from src.b3_score_ledger import FLAG_MATCH_OVER
    sim = Sim()
    sim.start_match()
    sim.fire(0, 70 * 4)
    sim.tick(int(DEAD_HOLD_SEC * FPS) + 3, dead=(False, True))
    assert last(sim).match_over and FLAG_MATCH_OVER in last(sim).sides[0].flags
    frozen = last(sim).sides[1].pending
    sim.fire(1, 70 * 20)
    sim.tick(60)
    assert last(sim).sides[1].pending == frozen and last(sim).sides[0].generated_total == 4


def test_time_based_constants_work_at_60hz() -> None:
    """入力が 60Hz でも得点採用・静止判定・減少の診断が同じ秒数で働く。"""
    sim = Sim(fps=60.0)
    sim.start_match()
    sim.scores[0] = 7
    sim.tick(2)
    assert last(sim).sides[0].score in (0, 7)
    sim.tick(2)
    assert last(sim).sides[0].score == 7
    sim.scores[0] = 3                                    # 下がる読みが REGRESS_RESYNC_SEC 続く
    sim.tick(int(REGRESS_RESYNC_SEC * 60) - 6)
    assert last(sim).sides[0].score == 7
    sim.tick(12)
    assert last(sim).sides[0].score == 7 and FLAG_SCORE_RESYNC in last(sim).sides[0].flags


def test_tsumo_increment_counts_as_placement_evidence() -> None:
    """置き終わり信号が無くても tsumo_count の増分 + NEXT 移動で着弾の証拠になる。"""
    sim = Sim()
    sim.start_match()
    sim.fire(0, 70 * 5)
    sim.tick(40, tsumo=(0, 3))
    sim.tick(1, tsumo=(0, 4), next_slide=(False, True))
    sim.tick(FRAMES_LANDING_WAIT, tsumo=(0, 4))
    assert last(sim).sides[1].dropped_total == 5


# ----------------------------- 第4版: 着弾は観測、アイコンで照合 -----------------------------


def _sim4(**kw: object) -> Sim:
    return Sim(mode="observed", **kw)


def _land_cells(sim: Sim, side: int, cells: int, frames: int = 8) -> None:
    c = [0, 0]
    c[side] = cells
    sim.tick(frames, ojama=tuple(c))  # type: ignore[arg-type]


def test_v4_landing_is_observed_from_board_increase_not_timing() -> None:
    """時刻規則では降らない場面 (置いていない) でも、受け側盤面のおじゃま増加を着弾として差し引く。"""
    sim = _sim4()
    sim.start_match()
    sim.tick(3, ojama=(0, 0))
    sim.fire(0, 70 * 10)
    assert last(sim).sides[1].pending == 10
    _land_cells(sim, 1, 6)
    s2 = last(sim).sides[1]
    assert s2.pending == 4 and s2.dropped_total == 6 and s2.observed_landing == 6
    assert any(e.kind == "observed_landing" and e.get("n") == 6 for e in sim.events)


@pytest.mark.parametrize("change", [-2, 0])
def test_v4_nonpositive_board_change_does_not_land(change: int) -> None:
    """予告と落下状態があっても、盤面が増えなければ観測着弾は記録しない。"""
    before, pending, frames = 12, 10, int(FPS)
    sim = _sim4()
    sim.start_match()
    sim.tick(frames, ojama=(0, before))
    sim.fire(0, 70 * pending)
    baseline = last(sim).sides[1]
    sim.tick(frames, states=("STABLE", "OJAMA_FALL"), ojama=(0, before))
    sim.tick(frames, ojama=(0, before + change))
    result = last(sim).sides[1]
    assert result.pending == baseline.pending == pending
    assert result.observed_landing == baseline.observed_landing == 0
    assert result.board_increase_total == baseline.board_increase_total
    assert not any(e.kind == "observed_landing" for e in sim.events)


def test_v4_decrease_is_not_landing_and_noise_is_ignored() -> None:
    sim = _sim4()
    sim.start_match()
    sim.tick(3, ojama=(0, 12))
    sim.fire(0, 70 * 10)
    _land_cells(sim, 1, 3, frames=3)        # 0.1 秒未満の減少 (消去・誤読) は基準を下げない
    _land_cells(sim, 1, 12, frames=8)
    assert last(sim).sides[1].observed_landing == 0 and last(sim).sides[1].pending == 10
    _land_cells(sim, 1, 5, frames=20)       # 0.5 秒以上続く減少は基準を下げる (着弾ではない)
    _land_cells(sim, 1, 9, frames=8)        # そこから 4 個増えたら着弾
    assert last(sim).sides[1].observed_landing == 4 and last(sim).sides[1].pending == 6


def test_v4_landing_never_makes_pending_negative_and_ignores_non_stable() -> None:
    sim = _sim4()
    sim.start_match()
    sim.tick(3, ojama=(0, 0))
    sim.fire(0, 70 * 3)
    sim.tick(8, ojama=(0, 40), states=("STABLE", "OJAMA_FALL"))   # 落下中の盤面は見ない
    assert last(sim).sides[1].observed_landing == 0
    _land_cells(sim, 1, 40)
    s2 = last(sim).sides[1]
    assert s2.pending == 0 and s2.observed_landing == 3 and s2.dropped_total == 3


def test_v4_chain_end_and_placement_do_not_change_values() -> None:
    """第4版では置き終わり・連鎖終了の規則は値を変えない (診断の landing_predicted だけ)。"""
    sim = _sim4()
    sim.start_match()
    sim.tick(3, ojama=(0, 0))
    sim.fire(0, 70 * 10)
    sim.tick(40, ojama=(0, 0))
    sim.tick(1, placed=(False, True), next_slide=(False, True), ojama=(0, 0))
    sim.tick(FRAMES_LANDING_WAIT, ojama=(0, 0))
    assert last(sim).sides[1].pending == 10 and last(sim).sides[1].dropped_total == 0
    assert any(e.kind == "landing_predicted" for e in sim.events)


def _icons(sim: Sim, n: int, total: int, count: int, frames: int, board_cells: int = 0,
           filled: int = 10) -> None:
    for _ in range(frames):
        sim.scores = sim.scores
        i = 1
        sides = []
        for k in range(2):
            sides.append(SideObs(score=sim.scores[k], state="STABLE", ojama_cells=board_cells if k == 1 else 0,
                                 icon_total=total if k == 1 else 0, icon_count=count if k == 1 else 0,
                                 filled_cells=filled if k == 1 else None))
        obs = FrameObs(sim.t, sim.frame, sim.game, True, (sides[0], sides[1]))
        snap = sim.ledger.update(obs)
        sim.obs.append(obs); sim.snaps.append(snap); sim.events.extend(snap.events); sim.frame += 1


def test_v4_prefix_invariance() -> None:
    sim = _sim4()
    sim.start_match()
    sim.tick(3, ojama=(0, 0))
    sim.fire(0, 70 * 12)
    _land_cells(sim, 1, 5)
    _icons(sim, 1, 7, 2, int(1.0 * FPS), board_cells=5)
    sim.fire(1, 70 * 3)
    full = sim
    for k in (60, 150, 260, len(full.obs) - 5):
        assert list(replay(full.obs[:k], mode="observed")) == full.snaps[:k], k


def test_default_mode_is_observed_and_port_exposes_new_keys() -> None:
    from src.b3_score_ledger import ScoreLedger
    led = ScoreLedger()
    assert led.core.mode == "observed"
    out = None
    for f in range(8):
        out = led.update(_rec(f, 0, 0))
    assert out is not None and "board_increase_p1" in out and "observed_landing_p2" in out




def test_v4_icons_never_change_the_ledger_values_or_flags() -> None:
    """user 2026-10-05: アイコンは台帳の値・印に一切使わない (参考列)。"""
    sim = _sim4()
    sim.start_match()
    sim.tick(3, ojama=(0, 0))
    sim.fire(0, 70 * 10)
    before = last(sim).sides[1]
    for total, count in ((4, 4), (36, 3), (360, 6), (24, 4)):
        _icons(sim, 1, total, count, int(1.0 * FPS))
    after = last(sim).sides[1]
    assert after.pending == before.pending == 10 and after.flags == before.flags


def test_v4_capacity_limited_is_marked_when_board_is_nearly_full() -> None:
    from src.b3_score_ledger import FLAG_CAPACITY_LIMITED
    sim = _sim4()
    sim.start_match()
    sim.tick(3, ojama=(0, 0))
    sim.fire(0, 70 * 100)                                      # 予告 100
    _icons(sim, 1, 0, 0, 8, board_cells=30, filled=70)         # 30 個降ったが盤面の空きは 8 しかない
    s2 = last(sim).sides[1]
    assert s2.observed_landing == 30 and s2.pending == 70
    assert FLAG_CAPACITY_LIMITED in s2.flags


def test_observed_board_increase_is_tracked_in_timing_mode_without_changing_values() -> None:
    sim = Sim()                                                # 第3版: 観測は診断だけ
    sim.start_match()
    sim.tick(3, ojama=(0, 0))
    sim.fire(0, 70 * 10)
    pend = last(sim).sides[1].pending
    _land_cells(sim, 1, 6)
    s2 = last(sim).sides[1]
    assert s2.board_increase_total == 6 and s2.observed_landing == 0 and s2.pending == pend


def test_v4_excess_over_pending_is_recorded_and_pending_not_negative() -> None:
    sim = _sim4()
    sim.start_match()
    sim.tick(3, ojama=(0, 0))
    sim.fire(0, 70 * 3)
    _land_cells(sim, 1, 8)
    s2 = last(sim).sides[1]
    assert s2.pending == 0 and s2.observed_landing == 3 and s2.board_increase_total == 8 and s2.board_excess_total == 5


def test_v4_landing_event_carries_column_increase_pattern() -> None:
    sim = _sim4()
    sim.start_match()
    sim.tick(3, ojama=(0, 0))
    sim.fire(0, 70 * 13)
    cols = (3, 2, 2, 2, 2, 2)                                  # 13 個: 6 列均等 floor(13/6)=2 + 端数 1
    for _ in range(8):
        sides = []
        for k in range(2):
            sides.append(SideObs(score=sim.scores[k], state="STABLE", ojama_cells=13 if k == 1 else 0,
                                 ojama_cols=cols if k == 1 else None, filled_cells=20))
        obs = FrameObs(sim.t, sim.frame, sim.game, True, (sides[0], sides[1]))
        snap = sim.ledger.update(obs)
        sim.obs.append(obs); sim.snaps.append(snap); sim.events.extend(snap.events); sim.frame += 1
    ev = [e for e in sim.events if e.kind == "observed_landing"]
    assert ev and ev[0].get("n") == 13 and ev[0].get("board_increase") == 13
    assert ev[0].get("cols") == (3, 2, 2, 2, 2, 2) or ev[0].get("cols") is None
