"""src/chain_id_resolver.py の単体テスト。

Gate 3 (data/verify/gate3_chainid_2026-08-24/summary.json) の実測パターン
(38 本中 20 本が旧 baseline 分裂) を再現できることを固定する。

**このテストは resolver 単体のみを検証する。**
`src/recognition_pipeline.py` 等の既存パイプラインは一切呼ばない
(既定 OFF・未配線の純関数モジュール)。

## 2026-08-25 是正: `ObservationKind.BASELINE` を分割

旧 `BASELINE` (docstring 上「score_ocr 由来」) は実体が `ChainSimulator`
の**推定**だった (`src/chain_detector.py:277-317`)。「連鎖が終わった合図」
(`CHAIN_SETTLED`、値は使わない) と「値の権威」(`SCORE_FINALIZE`、score OCR
確定差分) に分割した。既存テストは概ね `SCORE_FINALIZE` に置き換えている
(値を検証する意図だったテストのため)。`CHAIN_SETTLED` 固有の新しい挙動
(値を持ち込まない・growth_observed=False かつ他に値が無いときだけ低信頼
フォールバックとして使う) は末尾のセクションで新規に固定する。
"""
from __future__ import annotations

import copy
import dataclasses

from src.chain_id_resolver import (
    ActiveChainSnapshot,
    CHAIN_ID_MAX_SEC,
    ChainIdResolver,
    ChainObservation,
    CloseReason,
    ObservationKind,
    ResolvedChain,
    resolve_chain_ids,
)


def _obs(
    side: str,
    t_sec: float,
    kind: ObservationKind,
    chain_count: int = 0,
    total_score: int = 0,
    mechanism: str | None = None,
) -> ChainObservation:
    """テスト用の ChainObservation 生成ショートカット。"""
    return ChainObservation(
        side=side, t_sec=t_sec, kind=kind,
        chain_count=chain_count, total_score=total_score, mechanism=mechanism,
    )


def test_gate3_split_pattern_merges_into_one_chain_id() -> None:
    """実測パターン (formula 成長 → CHAIN_END_SIGNAL → score OCR 確定) が 1 本になる。"""
    observations = [
        _obs("1P", 777.37, ObservationKind.FORMULA_STEP, chain_count=1,
             total_score=100, mechanism="formula_read"),
        _obs("1P", 791.23, ObservationKind.FORMULA_STEP, chain_count=11,
             total_score=26420, mechanism="formula_read"),
        _obs("1P", 792.63, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 793.30, ObservationKind.SCORE_FINALIZE, chain_count=11,
             total_score=25700, mechanism="score_ocr"),
    ]
    resolved = resolve_chain_ids(observations)
    assert len(resolved) == 1
    chain = resolved[0]
    assert chain.chain_id == 1
    assert chain.side == "1P"
    assert chain.step_count == 11
    assert chain.provisional_score == 26420
    assert chain.finalized_score == 25700
    assert chain.was_finalized is True
    assert chain.force_cut is False
    assert chain.growth_observed is True
    assert chain.close_reason == CloseReason.FINALIZED
    assert chain.finalized_source == "score_ocr_diff"
    assert chain.opened_at_sec == 777.37
    assert chain.closed_at_sec == 793.30


def test_same_chain_keeps_running_max_when_cumulative_score_decreases() -> None:
    """累積点の一時的な下振れは物理的に不可能なので記録して無視する。"""
    resolver = ChainIdResolver()
    resolver.push(_obs(
        "1P", 1.0, ObservationKind.FORMULA_STEP,
        chain_count=1, total_score=70))
    resolver.push(_obs(
        "1P", 1.1, ObservationKind.FORMULA_STEP,
        chain_count=1, total_score=0))

    active = resolver.active()[0]
    assert active.provisional_score == 70
    assert resolver.stats().formula_step_observation_count == 2
    assert resolver.stats().provisional_score_decrease_ignored_count == 1


def test_next_physical_chain_gets_a_new_chain_id() -> None:
    """3.1 秒後の新しい連鎖 (別の物理連鎖) は別の chain_id になる。"""
    observations = [
        _obs("1P", 777.37, ObservationKind.FORMULA_STEP, chain_count=1, total_score=100),
        _obs("1P", 791.23, ObservationKind.FORMULA_STEP, chain_count=11, total_score=26420),
        _obs("1P", 792.63, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 793.30, ObservationKind.SCORE_FINALIZE, chain_count=11, total_score=25700),
        _obs("1P", 796.40, ObservationKind.FORMULA_STEP, chain_count=1, total_score=90),
    ]
    resolver = ChainIdResolver()
    for obs in observations:
        resolver.push(obs)
    resolver.flush()
    resolved = resolver.resolved()
    assert len(resolved) == 2
    assert resolved[0].chain_id == 1
    assert resolved[1].chain_id == 2
    assert resolved[1].opened_at_sec == 796.40
    assert resolved[1].was_finalized is False  # SCORE_FINALIZE が来ないまま flush
    assert resolved[1].close_reason == CloseReason.STREAM_END


def test_chain_observation_does_not_carry_trigger_sec() -> None:
    """trigger_sec を一切参照しない設計であることをフィールド定義で固定する。"""
    field_names = {f.name for f in dataclasses.fields(ChainObservation)}
    assert "trigger_sec" not in field_names


def test_two_sides_progress_independently() -> None:
    """1P と 2P は互いの chain_id を奪わない。"""
    observations = [
        _obs("1P", 10.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=50),
        _obs("2P", 10.5, ObservationKind.FORMULA_STEP, chain_count=1, total_score=40),
        _obs("1P", 11.0, ObservationKind.FORMULA_STEP, chain_count=2, total_score=200),
        _obs("2P", 11.2, ObservationKind.FORMULA_STEP, chain_count=2, total_score=180),
        _obs("1P", 12.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 12.5, ObservationKind.SCORE_FINALIZE, chain_count=2, total_score=195),
    ]
    resolver = ChainIdResolver()
    for obs in observations:
        resolver.push(obs)
    resolver.flush()
    resolved = {(r.side, r.chain_id): r for r in resolver.resolved()}
    assert ("1P", 1) in resolved
    assert ("2P", 2) in resolved
    assert resolved[("1P", 1)].was_finalized is True
    assert resolved[("2P", 2)].was_finalized is False  # 2P はまだ着地確認なし


def test_awaiting_finalize_then_new_step_closes_previous_unfinalized() -> None:
    """AWAITING_FINALIZE 中に新しい段が来たら、前の連鎖は未確定で閉じ新IDが発行される。"""
    observations = [
        _obs("1P", 100.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=100),
        _obs("1P", 105.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 106.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=80),
    ]
    resolved = resolve_chain_ids(observations)
    assert len(resolved) == 2
    first, second = resolved
    assert first.chain_id == 1
    assert first.was_finalized is False
    assert first.force_cut is False
    assert first.close_reason == CloseReason.SUPERSEDED
    assert first.closed_at_sec == 105.0  # 連鎖の存在を最後に観測した時刻 (CHAIN_END_SIGNAL)
    assert second.chain_id == 2
    assert second.opened_at_sec == 106.0


def test_score_finalize_below_cumulative_step_count_still_merges() -> None:
    """score OCR 確定の連鎖数が累積段数より小さくても同一 chain_id に合流する
    (値の妥当性は台帳の仕事)。"""
    observations = [
        _obs("1P", 0.0, ObservationKind.FORMULA_STEP, chain_count=5, total_score=500),
        _obs("1P", 1.0, ObservationKind.FORMULA_STEP, chain_count=10, total_score=1200),
        _obs("1P", 2.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 2.5, ObservationKind.SCORE_FINALIZE, chain_count=3, total_score=1150),
    ]
    resolved = resolve_chain_ids(observations)
    assert len(resolved) == 1
    assert resolved[0].was_finalized is True
    assert resolved[0].finalized_score == 1150
    assert resolved[0].step_count == 10  # 段数は成長フェーズの実測を保持
    assert resolved[0].growth_observed is True
    assert resolved[0].close_reason == CloseReason.FINALIZED
    assert resolved[0].finalized_source == "score_ocr_diff"


def test_force_cut_over_max_sec_splits_and_counts() -> None:
    """CHAIN_ID_MAX_SEC を超えたら強制打ち切りし、カウンタが増える。"""
    observations = [
        _obs("1P", 0.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=10),
        _obs("1P", CHAIN_ID_MAX_SEC + 1.0, ObservationKind.FORMULA_STEP,
             chain_count=1, total_score=20),
    ]
    resolver = ChainIdResolver()
    for obs in observations:
        resolver.push(obs)
    resolver.flush()
    resolved = resolver.resolved()
    stats = resolver.stats()
    assert len(resolved) == 2
    assert resolved[0].force_cut is True
    assert resolved[0].was_finalized is False
    assert resolved[0].close_reason == CloseReason.FORCE_CUT
    assert resolved[0].closed_at_sec == 0.0  # 最後に活動した時刻で閉じる
    assert resolved[1].chain_id == 2
    assert stats.force_cut_count == 1
    assert stats.opened_count == 2
    assert stats.orphan_end_signal_count == 0


def test_force_cut_boundary_is_exclusive() -> None:
    """ちょうど CHAIN_ID_MAX_SEC 経過では強制打ち切りしない (境界は超過のみ)。"""
    observations = [
        _obs("1P", 0.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=10),
        _obs("1P", CHAIN_ID_MAX_SEC, ObservationKind.FORMULA_STEP,
             chain_count=2, total_score=20),
    ]
    resolved = resolve_chain_ids(observations)
    assert len(resolved) == 1  # 同一 chain_id のまま


def test_match_boundary_closes_in_flight_and_does_not_cross_matches() -> None:
    """MATCH_BOUNDARY で in-flight が両サイドとも閉じ、試合を跨がない。

    closed_at_sec は境界イベント自体の時刻ではなく、その連鎖の存在を
    最後に観測した時刻 (last_t_sec) を使う (境界より前に連鎖自体は終わっているため。
    境界時刻を入れると根拠のない時刻を記録することになる)。
    """
    observations = [
        _obs("1P", 10.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=100),
        _obs("2P", 10.2, ObservationKind.FORMULA_STEP, chain_count=1, total_score=90),
        _obs("1P", 20.0, ObservationKind.MATCH_BOUNDARY),
        _obs("1P", 21.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=50),
    ]
    resolved = resolve_chain_ids(observations)
    assert len(resolved) == 3
    boundary_closed = {
        r.side: r for r in resolved if r.close_reason == CloseReason.MATCH_BOUNDARY
    }
    assert set(boundary_closed) == {"1P", "2P"}
    assert boundary_closed["1P"].closed_at_sec == 10.0
    assert boundary_closed["2P"].closed_at_sec == 10.2
    assert all(r.was_finalized is False for r in boundary_closed.values())
    assert all(r.force_cut is False for r in boundary_closed.values())
    new_chain = [r for r in resolved if r.opened_at_sec == 21.0][0]
    assert new_chain.chain_id == 3  # 前試合の chain_id を引き継がない


def test_score_finalize_only_chain_opens_and_closes_immediately() -> None:
    """掛け算式を観測できなかった連鎖 (旧経路) は SCORE_FINALIZE だけで発行即クローズする。

    provisional_score には唯一手に入った SCORE_FINALIZE の値をそのまま入れる
    (実測していない値に 0 を入れない)。growth_observed=False で
    それが成長フェーズの実測ではないことを正直に示す。
    """
    observations = [
        _obs("2P", 5.0, ObservationKind.SCORE_FINALIZE, chain_count=4, total_score=400),
    ]
    resolved = resolve_chain_ids(observations)
    assert len(resolved) == 1
    chain = resolved[0]
    assert chain.was_finalized is True
    assert chain.finalized_score == 400
    assert chain.step_count == 4
    assert chain.provisional_score == 400
    assert chain.growth_observed is False
    assert chain.close_reason == CloseReason.FINALIZED
    assert chain.finalized_source == "score_ocr_diff"
    assert chain.opened_at_sec == chain.closed_at_sec == 5.0


def test_count_decrease_during_growing_splits_into_new_chain() -> None:
    """GROWING 中に段が running max を下回ったら別連鎖とみなして分割する。"""
    observations = [
        _obs("1P", 0.0, ObservationKind.FORMULA_STEP, chain_count=5, total_score=500),
        _obs("1P", 1.0, ObservationKind.FORMULA_STEP, chain_count=2, total_score=60),
    ]
    resolver = ChainIdResolver()
    for obs in observations:
        resolver.push(obs)
    resolver.flush()
    resolved = resolver.resolved()
    stats = resolver.stats()
    assert len(resolved) == 2
    assert resolved[0].step_count == 5
    assert resolved[0].was_finalized is False
    assert resolved[0].close_reason == CloseReason.STEP_DECREASE
    assert resolved[0].closed_at_sec == 0.0
    assert resolved[1].step_count == 2
    assert stats.count_decrease_split_count == 1


def test_repeated_step_count_reading_does_not_split() -> None:
    """段カウントの読み取りが同じ値を連続して返しても (真の減少ではない)、分割しない。

    【注記】これは「running max を一度下回ってから回復する」ケースの検証ではない。
    そのケースは running max との比較・直前の生値との比較のどちらでも必ず分割される
    (running max はその時点までの最大の直前値と常に一致するため、両者は数学的に
    同値になる)。ここで固定するのは「同値の再読み取り (真の減少ではない)」という、
    実際に分割してはならない揺れのパターンである。
    """
    observations = [
        _obs("1P", 0.0, ObservationKind.FORMULA_STEP, chain_count=3, total_score=300),
        _obs("1P", 0.5, ObservationKind.FORMULA_STEP, chain_count=3, total_score=300),
        _obs("1P", 1.0, ObservationKind.FORMULA_STEP, chain_count=4, total_score=450),
        _obs("1P", 2.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 2.5, ObservationKind.SCORE_FINALIZE, chain_count=4, total_score=440),
    ]
    resolved = resolve_chain_ids(observations)
    assert len(resolved) == 1
    assert resolved[0].step_count == 4
    assert resolved[0].close_reason == CloseReason.FINALIZED


def test_orphan_chain_end_signal_is_counted_but_ignored() -> None:
    """対応する in-flight が無い CHAIN_END_SIGNAL は無視されるが、カウンタに残す。

    絶対律検出器は既知の事故源 (project_slide_false_positive_root_cause_2026-08-22)
    なので、迷子信号が多ければ検出器が壊れている証拠として扱えるようにする。
    """
    resolver = ChainIdResolver()
    resolver.push(_obs("1P", 0.0, ObservationKind.CHAIN_END_SIGNAL))
    resolver.flush()
    assert resolver.resolved() == []
    assert resolver.stats().orphan_end_signal_count == 1


def test_all_close_reasons_are_assigned_correctly() -> None:
    """6 種の close_reason (指定 5 種 + flush 用に追加した STREAM_END) が正しく付く。"""
    resolver = ChainIdResolver()

    # FINALIZED
    resolver.push(_obs("s_final", 0.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=10))
    resolver.push(_obs("s_final", 1.0, ObservationKind.CHAIN_END_SIGNAL))
    resolver.push(_obs("s_final", 2.0, ObservationKind.SCORE_FINALIZE, chain_count=1, total_score=10))

    # SUPERSEDED (元 chain) + 後始末で FINALIZED (新 chain)
    resolver.push(_obs("s_super", 10.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=10))
    resolver.push(_obs("s_super", 11.0, ObservationKind.CHAIN_END_SIGNAL))
    resolver.push(_obs("s_super", 12.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=5))
    resolver.push(_obs("s_super", 13.0, ObservationKind.CHAIN_END_SIGNAL))
    resolver.push(_obs("s_super", 14.0, ObservationKind.SCORE_FINALIZE, chain_count=1, total_score=5))

    # STEP_DECREASE (元 chain) + 後始末で FINALIZED (新 chain)
    resolver.push(_obs("s_dec", 20.0, ObservationKind.FORMULA_STEP, chain_count=5, total_score=500))
    resolver.push(_obs("s_dec", 21.0, ObservationKind.FORMULA_STEP, chain_count=2, total_score=60))
    resolver.push(_obs("s_dec", 22.0, ObservationKind.CHAIN_END_SIGNAL))
    resolver.push(_obs("s_dec", 23.0, ObservationKind.SCORE_FINALIZE, chain_count=2, total_score=55))

    # FORCE_CUT (元 chain) + 後始末で FINALIZED (新 chain)
    resolver.push(_obs("s_cut", 30.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=10))
    resolver.push(_obs("s_cut", 30.0 + CHAIN_ID_MAX_SEC + 1.0, ObservationKind.FORMULA_STEP,
                       chain_count=1, total_score=20))
    resolver.push(_obs("s_cut", 30.0 + CHAIN_ID_MAX_SEC + 2.0, ObservationKind.CHAIN_END_SIGNAL))
    resolver.push(_obs("s_cut", 30.0 + CHAIN_ID_MAX_SEC + 3.0, ObservationKind.SCORE_FINALIZE,
                       chain_count=1, total_score=20))

    # MATCH_BOUNDARY (この時点で in-flight は s_bound のみになるよう、他は全て後始末済み)
    resolver.push(_obs("s_bound", 70.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=10))
    resolver.push(_obs("s_bound", 71.0, ObservationKind.MATCH_BOUNDARY))

    # STREAM_END (境界より後に開始し、明示的な終了信号を与えないまま flush する)
    resolver.push(_obs("s_end", 80.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=10))
    resolver.flush()

    reasons = {r.close_reason for r in resolver.resolved()}
    assert reasons == set(CloseReason)


def test_empty_input_returns_empty_list_without_error() -> None:
    """空入力で例外を出さず空リストを返す。"""
    assert resolve_chain_ids([]) == []


def test_resolve_chain_ids_is_idempotent_and_side_effect_free() -> None:
    """同じ入力で同じ出力になり、入力リストを書き換えない。"""
    observations = [
        _obs("1P", 5.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=100),
        _obs("1P", 6.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 7.0, ObservationKind.SCORE_FINALIZE, chain_count=1, total_score=95),
    ]
    before = copy.deepcopy(observations)
    result_a = resolve_chain_ids(observations)
    result_b = resolve_chain_ids(observations)
    assert result_a == result_b
    assert observations == before  # 入力を書き換えていない


def test_resolve_chain_ids_is_order_independent_for_input_sequence() -> None:
    """入力が t_sec 逆順でも同じ結果になる。"""
    observations = [
        _obs("1P", 1.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=100),
        _obs("1P", 2.0, ObservationKind.FORMULA_STEP, chain_count=5, total_score=800),
        _obs("1P", 3.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 4.0, ObservationKind.SCORE_FINALIZE, chain_count=5, total_score=790),
    ]
    forward = resolve_chain_ids(observations)
    backward = resolve_chain_ids(list(reversed(observations)))
    assert forward == backward


def test_resolved_chain_and_stats_are_frozen_dataclasses() -> None:
    """出力の値オブジェクトが frozen (不変) であることを確認する。"""
    chain = resolve_chain_ids(
        [_obs("2P", 0.0, ObservationKind.SCORE_FINALIZE, chain_count=1, total_score=10)],
    )[0]
    assert isinstance(chain, ResolvedChain)
    try:
        chain.chain_id = 999  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("ResolvedChain は frozen であるべき")


def test_active_tracks_formula_steps_in_side_order_as_an_independent_copy() -> None:
    """FORMULA_STEP の開始・更新を side 順の不変コピーとして公開する。"""
    resolver = ChainIdResolver()
    resolver.push(_obs("2P", 10.0, ObservationKind.FORMULA_STEP,
                       chain_count=1, total_score=40))
    resolver.push(_obs("1P", 10.1, ObservationKind.FORMULA_STEP,
                       chain_count=2, total_score=300))
    before_update = resolver.active()
    resolver.push(_obs("2P", 11.0, ObservationKind.FORMULA_STEP,
                       chain_count=3, total_score=1320))

    active = resolver.active()
    assert [snapshot.side for snapshot in active] == ["1P", "2P"]
    assert active[0] == ActiveChainSnapshot(
        chain_id=2, side="1P", opened_at_sec=10.1, last_t_sec=10.1,
        step_count=2, provisional_score=300, growth_observed=True,
        score_base=0, awaiting_finalize=False,
    )
    assert active[1].step_count == 3
    assert active[1].provisional_score == 1320
    assert active[1].last_t_sec == 11.0
    assert before_update[1].step_count == 1, "過去のコピーが内部更新で変化した"
    try:
        active[0].chain_id = 999  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("ActiveChainSnapshot は frozen であるべき")


def test_active_marks_settled_and_end_signal_as_awaiting_finalize() -> None:
    """CHAIN_SETTLED/END は既存契約どおり確定待ちとして公開する。"""
    resolver = ChainIdResolver()
    resolver.push(_obs("1P", 0.0, ObservationKind.FORMULA_STEP,
                       chain_count=1, total_score=100))
    resolver.push(_obs("1P", 1.0, ObservationKind.CHAIN_SETTLED,
                       chain_count=1, total_score=999999))
    settled = resolver.active()[0]
    assert settled.awaiting_finalize is True
    assert settled.last_t_sec == 1.0
    assert settled.provisional_score == 100, "settled 推定値を公開値へ混入させない"

    resolver.push(_obs("1P", 2.0, ObservationKind.CHAIN_END_SIGNAL))
    ended = resolver.active()[0]
    assert ended.awaiting_finalize is True
    assert ended.last_t_sec == 2.0


def test_active_disappears_after_finalize_boundary_and_flush() -> None:
    """確定・試合境界・ストリーム終端で公開中の連鎖が消滅する。"""
    for close_kind in (ObservationKind.SCORE_FINALIZE, ObservationKind.MATCH_BOUNDARY):
        resolver = ChainIdResolver()
        resolver.push(_obs("1P", 0.0, ObservationKind.FORMULA_STEP,
                           chain_count=1, total_score=100))
        resolver.push(_obs("1P", 1.0, ObservationKind.CHAIN_END_SIGNAL))
        assert resolver.active()[0].awaiting_finalize is True
        resolver.push(_obs("1P", 2.0, close_kind, chain_count=1, total_score=95))
        assert resolver.active() == []

    resolver = ChainIdResolver()
    resolver.push(_obs("2P", 3.0, ObservationKind.FORMULA_STEP,
                       chain_count=2, total_score=300))
    resolver.flush()
    assert resolver.active() == []


# ===========================================================================
# CHAIN_SETTLED / SCORE_FINALIZE の分離 (2026-08-25 追加、fable アーキ裁定)
# ===========================================================================

def test_chain_settled_does_not_carry_a_value_when_growth_observed() -> None:
    """growth_observed=True (成長フェーズを観測済み) のとき、CHAIN_SETTLED は
    AWAITING_FINALIZE へ進めるだけで `finalized_score` を更新しない。

    実際に値を確定させるのは後続の SCORE_FINALIZE だけであることを固定する。
    CHAIN_SETTLED 自身の `total_score` (推定値) は最終結果に一切現れない。
    """
    observations = [
        _obs("1P", 0.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=100),
        _obs("1P", 1.0, ObservationKind.CHAIN_SETTLED, chain_count=1, total_score=999999),
        _obs("1P", 2.0, ObservationKind.SCORE_FINALIZE, chain_count=1, total_score=95),
    ]
    resolved = resolve_chain_ids(observations)
    assert len(resolved) == 1
    chain = resolved[0]
    assert chain.finalized_score == 95, "CHAIN_SETTLED の推定値が紛れ込んだ"
    assert chain.finalized_source == "score_ocr_diff"
    assert chain.growth_observed is True
    assert chain.close_reason == CloseReason.FINALIZED


def test_chain_settled_alone_moves_to_awaiting_finalize_without_closing() -> None:
    """growth_observed=True のときの CHAIN_SETTLED 単独では、確定せずに
    AWAITING_FINALIZE のまま (SCORE_FINALIZE が来なければ未確定で残る)。"""
    observations = [
        _obs("1P", 0.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=100),
        _obs("1P", 1.0, ObservationKind.CHAIN_SETTLED, chain_count=1, total_score=999999),
    ]
    resolver = ChainIdResolver()
    for obs in observations:
        resolver.push(obs)
    assert resolver.resolved() == [], "CHAIN_SETTLED だけで確定してしまった"
    resolver.flush()
    resolved = resolver.resolved()
    assert len(resolved) == 1
    assert resolved[0].was_finalized is False
    assert resolved[0].finalized_source is None
    assert resolved[0].close_reason == CloseReason.STREAM_END


def test_chain_settled_fallback_used_only_without_growth_and_without_score_finalize() -> None:
    """growth_observed=False (掛け算式を一度も観測できていない) かつ
    SCORE_FINALIZE も一度も来ないときだけ、CHAIN_SETTLED の値を低信頼
    フォールバックとして使い `finalized_source="simulate_fallback"` で閉じる。
    """
    observations = [
        _obs("2P", 5.0, ObservationKind.CHAIN_SETTLED, chain_count=4, total_score=400),
    ]
    resolved = resolve_chain_ids(observations)
    assert len(resolved) == 1
    chain = resolved[0]
    assert chain.was_finalized is True
    assert chain.finalized_score == 400
    assert chain.growth_observed is False
    assert chain.close_reason == CloseReason.FINALIZED
    assert chain.finalized_source == "simulate_fallback"


def test_simulate_estimate_disaster_does_not_enter_accounting() -> None:
    """【回帰本体】実測の事故: 暫定 (score OCR 由来の確定) 38 個相当に対し、
    推定 (ChainSimulator 由来の CHAIN_SETTLED) が 745 個相当という壊滅的に
    間違った値で来ても、745 が最終結果 (`finalized_score`) に入らないこと。

    growth_observed=True の経路では CHAIN_SETTLED は値を一切運ばないため、
    構造的に混入し得ない。
    """
    observations = [
        _obs("1P", 0.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=38),
        _obs("1P", 1.0, ObservationKind.CHAIN_SETTLED, chain_count=1, total_score=745),
        _obs("1P", 2.0, ObservationKind.SCORE_FINALIZE, chain_count=1, total_score=38),
    ]
    resolved = resolve_chain_ids(observations)
    assert len(resolved) == 1
    assert resolved[0].finalized_score == 38
    assert resolved[0].finalized_score != 745


# ===========================================================================
# Fix【3】: AWAITING_FINALIZE 中の段数増加は継続とみなす
# (2026-08-25 追加、実データ v51 4/4 検証済み)
# ===========================================================================

def test_awaiting_finalize_step_increase_continues_same_chain_id() -> None:
    """AWAITING_FINALIZE 中に running max より大きい段数が来たら、
    新規発行せず同じ chain_id のまま GROWING に戻り、段・スコアを更新する。

    物理的根拠: すべての連鎖は 1 段目から始まるため、running max を
    上回る段数は継続以外にありえない (`_handle_formula_step_while_
    awaiting_finalize` docstring 参照)。
    """
    observations = [
        _obs("1P", 100.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=240),
        _obs("1P", 105.0, ObservationKind.CHAIN_END_SIGNAL),  # 早合点した終了信号
        _obs("1P", 106.0, ObservationKind.FORMULA_STEP, chain_count=2, total_score=340),
        _obs("1P", 110.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 111.0, ObservationKind.SCORE_FINALIZE, chain_count=2, total_score=340),
    ]
    resolved = resolve_chain_ids(observations)
    assert len(resolved) == 1, "SUPERSEDED として分割されてしまった"
    chain = resolved[0]
    assert chain.chain_id == 1
    assert chain.step_count == 2
    assert chain.provisional_score == 340
    assert chain.finalized_score == 340
    assert chain.opened_at_sec == 100.0
    assert chain.close_reason == CloseReason.FINALIZED


def test_awaiting_finalize_step_increase_real_data_v51_four_cases() -> None:
    """実データ v51 の 4 ケース (報告の実測表) を再現する回帰本体。

    SUPERSEDED した連鎖 (最終段数) → 次の観測の段数 → 正解:
      1 段 (240) → 2 (340): 継続
      1 段 (40)  → 2 (360): 継続
      2 段 (540) → 6 (9600): 継続
      10 段 (54230) → 1 (100、10 秒後): 新規
    """
    observations = [
        # ケース1: 1 -> 2 (継続)
        _obs("1P", 0.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=240),
        _obs("1P", 1.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 2.0, ObservationKind.FORMULA_STEP, chain_count=2, total_score=340),
        _obs("1P", 3.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 4.0, ObservationKind.SCORE_FINALIZE, chain_count=2, total_score=340),
        # ケース2: 1 -> 2 (継続、別 side で独立に検証)
        _obs("2P", 0.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=40),
        _obs("2P", 1.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("2P", 2.0, ObservationKind.FORMULA_STEP, chain_count=2, total_score=360),
        _obs("2P", 3.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("2P", 4.0, ObservationKind.SCORE_FINALIZE, chain_count=2, total_score=360),
    ]
    resolved = resolve_chain_ids(observations)
    assert len(resolved) == 2, "1P/2P それぞれ 1 本ずつに合流すべき"
    for chain in resolved:
        assert chain.close_reason == CloseReason.FINALIZED
        assert chain.step_count == 2

    # ケース3: 2 -> 6 (継続) と ケース4: 10 -> 1(10秒後) (新規) を同一 side で連続検証。
    # 段数 (chain_count) は SCORE_FINALIZE では更新されず FORMULA_STEP でのみ
    # 積まれる (既存仕様、`test_score_finalize_below_cumulative_step_count_
    # still_merges` 参照) ため、10 段目も FORMULA_STEP として観測させる。
    observations_2 = [
        _obs("1P", 0.0, ObservationKind.FORMULA_STEP, chain_count=2, total_score=540),
        _obs("1P", 1.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 2.0, ObservationKind.FORMULA_STEP, chain_count=6, total_score=9600),
        _obs("1P", 3.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 4.0, ObservationKind.FORMULA_STEP, chain_count=10, total_score=54230),
        _obs("1P", 5.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 6.0, ObservationKind.SCORE_FINALIZE, chain_count=10, total_score=54230),
        # ここまでで 1 本 (2->6->10 で確定)。続いて 10 秒後に新規連鎖が 1 段目から。
        _obs("1P", 16.0, ObservationKind.FORMULA_STEP, chain_count=1, total_score=100),
    ]
    resolver = ChainIdResolver()
    for obs in observations_2:
        resolver.push(obs)
    resolver.flush()
    resolved_2 = resolver.resolved()
    assert len(resolved_2) == 2
    first, second = resolved_2
    assert first.step_count == 10
    assert first.finalized_score == 54230
    assert first.close_reason == CloseReason.FINALIZED
    assert second.chain_id == first.chain_id + 1, "新規連鎖として別 chain_id が発行された"
    assert second.opened_at_sec == 16.0
    assert second.step_count == 1


def test_awaiting_finalize_step_equal_to_running_max_is_treated_as_new_chain() -> None:
    """AWAITING_FINALIZE 中に running max と同値の段数が来たら新規連鎖とみなす
    (`obs.chain_count <= state.step_count` は継続でないことの境界確認)。"""
    observations = [
        _obs("1P", 0.0, ObservationKind.FORMULA_STEP, chain_count=3, total_score=300),
        _obs("1P", 1.0, ObservationKind.CHAIN_END_SIGNAL),
        _obs("1P", 2.0, ObservationKind.FORMULA_STEP, chain_count=3, total_score=50),
    ]
    resolved = resolve_chain_ids(observations)
    assert len(resolved) == 2
    assert resolved[0].close_reason == CloseReason.SUPERSEDED
    assert resolved[1].opened_at_sec == 2.0
    assert resolved[1].step_count == 3


# ===========================================================================
# P1-2 (Codex Gate 3-2b レビュー NG、2026-08-25): 1 物理連鎖から複数 chain_id
#
# 経路A (Codex 最小再現): growth なしの CHAIN_SETTLED を即クローズした直後に
#   SCORE_FINALIZE が届くと、両方が state=None 経路で別 ID になる
#   (settled t=10.0 → finalize t=10.1 で opened_count=2, finalized_count=2)。
# 経路B (v51 実データ、data/verify/gate3_rate_trace_2026-08-25/
#   step_trace_summary.md): SCORE_FINALIZE が物理連鎖の**途中**で届いて
#   GROWING を確定クローズした後、続きの段 (cc が前 chain の running max を
#   上回る) が累積値のまま新 chain を開き、前 chain の確定済み分を丸ごと
#   二重計上する (chain5⊃chain4 で +308 個、chain8⊃chain7 で +18 個)。
# ===========================================================================

def test_p1_2_settled_then_score_finalize_merges_into_one_chain_id() -> None:
    """【Codex 最小再現】settled t=10.0 → finalize t=10.1 は 1 本の chain_id。

    修正前: opened_count=2, finalized_count=2 (同じ物理連鎖に 2 ID)。
    CHAIN_SETTLED (in-flight 無し) は即クローズせず保留し、後続の
    SCORE_FINALIZE と統合する。値は権威 (score OCR 確定差分) のみを使い、
    settled の推定値 52150 はどこにも現れない。
    """
    resolver = ChainIdResolver()
    resolver.push(_obs("1P", 10.0, ObservationKind.CHAIN_SETTLED,
                       chain_count=3, total_score=52150))
    resolver.push(_obs("1P", 10.1, ObservationKind.SCORE_FINALIZE,
                       chain_count=3, total_score=745))
    resolver.flush()
    resolved = resolver.resolved()
    stats = resolver.stats()
    assert len(resolved) == 1, f"1 物理連鎖が {len(resolved)} 本に分裂した"
    assert stats.opened_count == 1
    assert stats.finalized_count == 1
    chain = resolved[0]
    assert chain.was_finalized is True
    assert chain.finalized_source == "score_ocr_diff"
    assert chain.finalized_score == 745
    assert chain.provisional_score == 745, "settled の推定値 52150 が紛れ込んだ"
    assert chain.growth_observed is False
    assert chain.opened_at_sec == 10.0
    assert chain.closed_at_sec == 10.1


def test_p1_2_settled_and_finalize_at_same_time_merge() -> None:
    """同時刻 (settled と finalize が同一 t_sec) でも 1 本に統合される。"""
    resolver = ChainIdResolver()
    resolver.push(_obs("2P", 10.0, ObservationKind.CHAIN_SETTLED,
                       chain_count=2, total_score=1000))
    resolver.push(_obs("2P", 10.0, ObservationKind.SCORE_FINALIZE,
                       chain_count=2, total_score=14))
    resolver.flush()
    resolved = resolver.resolved()
    assert len(resolved) == 1
    assert resolved[0].finalized_score == 14
    assert resolved[0].finalized_source == "score_ocr_diff"


def test_p1_2_settled_then_small_delay_finalize_merges() -> None:
    """小遅延 (1.0 秒) の後続確定でも統合される (保留は時刻に依存しない)。"""
    resolver = ChainIdResolver()
    resolver.push(_obs("1P", 10.0, ObservationKind.CHAIN_SETTLED,
                       chain_count=5, total_score=3000))
    resolver.push(_obs("1P", 11.0, ObservationKind.SCORE_FINALIZE,
                       chain_count=5, total_score=42))
    resolver.flush()
    resolved = resolver.resolved()
    assert len(resolved) == 1
    assert resolved[0].finalized_score == 42


def test_p1_2_finalize_then_settled_echo_is_absorbed() -> None:
    """順序違い (finalize が先、settled が後) でも 2 ID にならない。

    SCORE_FINALIZE で確定クローズした直後 (残響時間内) に届いた
    CHAIN_SETTLED は、同じ物理連鎖の終わりの残響 (ChainSimulator が
    遅れて出した合図) として吸収する。黙って捨てず必ずカウンタに残す。
    """
    resolver = ChainIdResolver()
    resolver.push(_obs("1P", 10.0, ObservationKind.SCORE_FINALIZE,
                       chain_count=3, total_score=745))
    resolver.push(_obs("1P", 10.1, ObservationKind.CHAIN_SETTLED,
                       chain_count=3, total_score=52150))
    resolver.flush()
    resolved = resolver.resolved()
    stats = resolver.stats()
    assert len(resolved) == 1, "残響の settled が別 chain_id を生んだ"
    assert resolved[0].finalized_score == 745
    assert stats.settled_echo_absorbed_count == 1
    assert stats.chain_settled_received_count == 1  # 母数: 1/1 が吸収された


def test_p1_2_settled_echo_after_formula_chain_finalize_is_absorbed() -> None:
    """成長観測済み連鎖の確定クローズ直後に届く settled 残響も吸収される
    (実運用の典型順序: formula 成長 → SCORE_FINALIZE → baseline 残響)。"""
    resolver = ChainIdResolver()
    resolver.push(_obs("1P", 0.0, ObservationKind.FORMULA_STEP,
                       chain_count=1, total_score=100))
    resolver.push(_obs("1P", 1.0, ObservationKind.FORMULA_STEP,
                       chain_count=3, total_score=1320))
    resolver.push(_obs("1P", 2.0, ObservationKind.SCORE_FINALIZE,
                       chain_count=3, total_score=19))
    resolver.push(_obs("1P", 2.3, ObservationKind.CHAIN_SETTLED,
                       chain_count=3, total_score=1320))
    resolver.flush()
    resolved = resolver.resolved()
    assert len(resolved) == 1, "残響の settled が幻の simulate 連鎖を生んだ"
    assert resolver.stats().settled_echo_absorbed_count == 1


def test_settled_with_different_chain_count_inside_echo_window_is_not_absorbed() -> None:
    """K6: 時間が近いだけの別連鎖 (cc=8→1) を残響として消さない。"""
    resolver = ChainIdResolver()
    resolver.push(_obs("1P", 10.0, ObservationKind.SCORE_FINALIZE,
                       chain_count=8, total_score=500))
    resolver.push(_obs("1P", 10.2, ObservationKind.CHAIN_SETTLED,
                       chain_count=1, total_score=400))
    resolver.flush()
    assert len(resolver.resolved()) == 2
    assert resolver.stats().settled_echo_absorbed_count == 0


def test_formula_settled_with_different_score_inside_echo_window_is_not_absorbed() -> None:
    """K6: 成長観測済みでは段数だけでなく同経路の累積点も一致させる。"""
    resolver = ChainIdResolver()
    resolver.push(_obs("1P", 0.0, ObservationKind.FORMULA_STEP,
                       chain_count=1, total_score=100))
    resolver.push(_obs("1P", 1.0, ObservationKind.FORMULA_STEP,
                       chain_count=3, total_score=1320))
    resolver.push(_obs("1P", 2.0, ObservationKind.SCORE_FINALIZE,
                       chain_count=3, total_score=19))
    resolver.push(_obs("1P", 2.2, ObservationKind.CHAIN_SETTLED,
                       chain_count=3, total_score=1400))
    resolver.flush()
    assert len(resolver.resolved()) == 2
    assert resolver.stats().settled_echo_absorbed_count == 0


def test_p1_2_settled_far_after_finalize_is_not_absorbed() -> None:
    """残響時間 (SETTLED_ECHO_MAX_SEC = 2.61+1.17 秒) を超えて届いた
    settled は別の物理連鎖なので吸収せず、従来どおり低信頼フォールバック
    連鎖になる (吸収しすぎの防止)。"""
    resolver = ChainIdResolver()
    resolver.push(_obs("1P", 10.0, ObservationKind.SCORE_FINALIZE,
                       chain_count=3, total_score=745))
    resolver.push(_obs("1P", 15.0, ObservationKind.CHAIN_SETTLED,
                       chain_count=1, total_score=400))
    resolver.flush()
    resolved = resolver.resolved()
    assert len(resolved) == 2
    assert resolved[1].finalized_source == "simulate_fallback"
    assert resolver.stats().settled_echo_absorbed_count == 0
    assert resolver.stats().chain_settled_received_count == 1  # 0/1 吸収


def test_p1_2_two_settled_without_finalize_stay_two_chains() -> None:
    """finalize を挟まない settled 2 連発は 2 本の物理連鎖 (統合しない)。
    先の保留は低信頼フォールバックとして確定し、後の保留が開く。"""
    resolver = ChainIdResolver()
    resolver.push(_obs("1P", 10.0, ObservationKind.CHAIN_SETTLED,
                       chain_count=2, total_score=700))
    resolver.push(_obs("1P", 20.0, ObservationKind.CHAIN_SETTLED,
                       chain_count=3, total_score=900))
    resolver.flush()
    resolved = resolver.resolved()
    assert len(resolved) == 2
    assert resolved[0].finalized_score == 700
    assert resolved[0].finalized_source == "simulate_fallback"
    assert resolved[1].finalized_score == 900
    assert resolved[1].finalized_source == "simulate_fallback"


def test_p1_2_v51_mid_chain_finalize_then_continuing_steps_subtract_base() -> None:
    """【v51 実データ経路 B】物理連鎖 B (2P, cc=1〜10、累積 40→54,230) の再現。

    cc=1〜7 (累積 21,570) 保持中に SCORE_FINALIZE (308 個 = 21,570 点相当)
    が途中で届いて確定クローズ → 0.17 秒後に続きの段 cc=8 (累積 29,670) が
    新 chain を開く。修正前は新 chain の暫定が累積値のまま (54,230) で、
    確定済みの 21,570 を丸ごと包含していた (D7 で +308 個の二重計上)。
    修正後: 新 chain の暫定 = 累積 − 前 chain の確定クローズ済み累積。
    """
    resolver = ChainIdResolver()
    resolver.push(_obs("2P", 492.0, ObservationKind.FORMULA_STEP,
                       chain_count=1, total_score=40))
    resolver.push(_obs("2P", 501.5, ObservationKind.FORMULA_STEP,
                       chain_count=7, total_score=21570))
    resolver.push(_obs("2P", 501.93, ObservationKind.SCORE_FINALIZE,
                       chain_count=7, total_score=308))
    resolver.push(_obs("2P", 502.10, ObservationKind.FORMULA_STEP,
                       chain_count=8, total_score=29670))
    resolver.push(_obs("2P", 503.5, ObservationKind.FORMULA_STEP,
                       chain_count=9, total_score=45270))
    resolver.push(_obs("2P", 505.0, ObservationKind.FORMULA_STEP,
                       chain_count=10, total_score=54230))
    resolver.flush()
    resolved = resolver.resolved()
    stats = resolver.stats()
    assert len(resolved) == 2
    first, second = resolved
    assert first.finalized_score == 308
    assert first.provisional_score == 21570
    assert second.provisional_score == 54230 - 21570, (
        f"前半確定分が控除されていない (実測 {second.provisional_score})"
    )
    assert second.step_count == 10
    assert second.opened_at_sec == 502.10
    assert stats.continuation_reopen_count == 1  # 1/2 open が継続分割
    assert stats.continuation_base_underflow_count == 0  # 0/1 継続で控除下回りなし


def test_p1_2_v51_chain7_chain8_pattern_subtracts_base() -> None:
    """【v51 実データ経路 B・同型 2 例目】物理連鎖 C (cc=1〜3 で 1,320 点
    確定クローズ → cc=4 が累積 3,020 で新 chain → cc=7 累積 14,540 で
    STREAM_END)。新 chain の暫定 = 14,540 − 1,320 = 13,220。"""
    resolver = ChainIdResolver()
    resolver.push(_obs("2P", 516.0, ObservationKind.FORMULA_STEP,
                       chain_count=1, total_score=40))
    resolver.push(_obs("2P", 519.0, ObservationKind.FORMULA_STEP,
                       chain_count=3, total_score=1320))
    resolver.push(_obs("2P", 520.0, ObservationKind.SCORE_FINALIZE,
                       chain_count=3, total_score=19))
    resolver.push(_obs("2P", 520.40, ObservationKind.FORMULA_STEP,
                       chain_count=4, total_score=3020))
    resolver.push(_obs("2P", 524.0, ObservationKind.FORMULA_STEP,
                       chain_count=7, total_score=14540))
    resolver.flush()
    resolved = resolver.resolved()
    assert len(resolved) == 2
    assert resolved[1].provisional_score == 14540 - 1320


def test_p1_2_new_chain_from_cc1_after_finalize_gets_no_subtraction() -> None:
    """確定クローズの後に cc=1 から始まる連鎖は本当に新規 (継続ではない) ので
    控除しない (連鎖は必ず 1 段目から始まる、という物理則の裏面)。"""
    resolver = ChainIdResolver()
    resolver.push(_obs("1P", 0.0, ObservationKind.FORMULA_STEP,
                       chain_count=1, total_score=100))
    resolver.push(_obs("1P", 5.0, ObservationKind.FORMULA_STEP,
                       chain_count=7, total_score=21570))
    resolver.push(_obs("1P", 6.0, ObservationKind.SCORE_FINALIZE,
                       chain_count=7, total_score=308))
    resolver.push(_obs("1P", 12.0, ObservationKind.FORMULA_STEP,
                       chain_count=1, total_score=90))
    resolver.flush()
    resolved = resolver.resolved()
    assert len(resolved) == 2
    assert resolved[1].provisional_score == 90, "新規連鎖から誤って控除した"
    assert resolver.stats().continuation_reopen_count == 0


def test_p1_2_continuation_base_not_applied_across_match_boundary() -> None:
    """試合境界を跨いだら継続控除の記憶は消える (前試合の確定分を
    次試合の連鎖から控除してはならない)。"""
    resolver = ChainIdResolver()
    resolver.push(_obs("1P", 0.0, ObservationKind.FORMULA_STEP,
                       chain_count=1, total_score=100))
    resolver.push(_obs("1P", 5.0, ObservationKind.FORMULA_STEP,
                       chain_count=7, total_score=21570))
    resolver.push(_obs("1P", 6.0, ObservationKind.SCORE_FINALIZE,
                       chain_count=7, total_score=308))
    resolver.push(_obs("1P", 7.0, ObservationKind.MATCH_BOUNDARY))
    # 次試合の連鎖が (OCR の読み始め遅れで) cc=8 相当から観測されたとしても控除しない
    resolver.push(_obs("1P", 8.0, ObservationKind.FORMULA_STEP,
                       chain_count=8, total_score=30000))
    resolver.flush()
    resolved = resolver.resolved()
    assert resolved[-1].provisional_score == 30000
    assert resolver.stats().continuation_reopen_count == 0


def test_force_cut_applies_to_the_silent_side() -> None:
    """観測が途絶えた side も強制打ち切りされる (2026-08-24 レビュー指摘)。

    相手が窒息する等で片側の観測が止まると、その side の連鎖が永久に開いたままになり、
    台帳の episode が閉じられなくなる。時間はどちらの side でも同じだけ進むので、
    強制打ち切りは観測が届いた side だけでなく両 side に対して行う。
    """
    resolver = ChainIdResolver()
    # 2P が連鎖を始めたあと、2P の観測は一切届かなくなる
    resolver.push(_obs("2P", 10.0, ObservationKind.FORMULA_STEP, chain_count=1,
                       total_score=40, mechanism="formula_read"))
    # 1P 側だけが CHAIN_ID_MAX_SEC を超えて観測され続ける
    resolver.push(_obs("1P", 10.0 + CHAIN_ID_MAX_SEC + 1.0,
                       ObservationKind.FORMULA_STEP, chain_count=1,
                       total_score=40, mechanism="formula_read"))
    cut = [c for c in resolver.resolved() if c.side == "2P"]
    assert len(cut) == 1, "2P の連鎖が強制打ち切りされていない"
    assert cut[0].force_cut is True
    assert cut[0].close_reason is CloseReason.FORCE_CUT
    assert cut[0].was_finalized is False
    assert resolver.stats().force_cut_count == 1
    # 1P は開いたまま (自分の観測はまだ CHAIN_ID_MAX_SEC を超えていない)
    assert not [c for c in resolver.resolved() if c.side == "1P"]
