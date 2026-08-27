"""src/exchange_episode_tracker.py の単体テスト (Gate 3-2a、観測のみ・既定 OFF)。

**実装より先に書いている。** 後から辻褄を合わせないため。

このテストは tracker 単体のみを検証する。`src/recognition_pipeline.py` 等の
既存パイプラインは一切呼ばない (既定 OFF・未配線の観測専用モジュール)。

対象仕様: `docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md` §13 (D1〜D7)。
D5 (直近の物理イベント種別+時刻) は本モジュールの責務外 (時系列列であり、
tracker は動画 1 本の集計を返す。Gate 3-2b で dump 行へ直接足す予定)。

## 単位の注意 (2026-08-24 コーディネーター指摘を受けて追加)

台帳 (`ExchangeLedger`) の `amount` は **おじゃまの個数**であり、
`ChainEventObservation.total_score` (スコア点) をそのまま渡してはいけない。
本ファイルの D2 系テストはこの単位換算 (`score_to_ojama`) が正しく効いて
いることを固定する。

## 2026-08-25 是正: CHAIN_MECHANISM_BASELINE は値の権威を失った

`mechanism='baseline'` (`ChainSimulator` 産の推定) はもう `finalized_score`
を運ばない (`ObservationKind.CHAIN_SETTLED` へマップされ、連鎖が終わった
合図のみ)。値の権威は `ExchangeEpisodeTracker.observe_generation()`
(`GenerationObservation`、`OjamaAccountingTracker.total_generated_by_p1/p2`
の増分。**単位はおじゃま個数、換算不要**) だけが持つ。BASELINE 観測だけで
`was_finalized=True` を期待していた既存テストは、`observe_generation()` の
呼び出しを追加して初めて確定するよう更新した (1本ずつ理由をコミット/報告参照)。
"""
from __future__ import annotations

import pytest

from src.chain_detector import (
    CHAIN_MECHANISM_BASELINE,
    CHAIN_MECHANISM_FORMULA_READ,
    CHAIN_MECHANISM_SCORE_JUMP,
)
from src.chain_id_resolver import CloseReason
from src.exchange_episode_tracker import (
    ChainEventObservation,
    ExchangeEpisodeDiagnostics,
    ExchangeEpisodeTracker,
    GenerationObservation,
    GrossCounterDeltaClassification,
    PendingUncappedFrame,
    SettlementObservation,
    classify_gross_counter_delta,
    classify_pending_uncapped_delta,
)
from src.exchange_ledger import (
    FINALIZE_DOWNWARD_TOLERANCE,
    EventKind,
    ExchangeEvent,
    PhysicalContext,
    Side,
)
from src.ojama_accounting import GrossOjamaCounters
from src.scoring import OJAMA_RATE_STANDARD, compute_effective_rate


def _obs(
    side: str,
    t_sec: float,
    mechanism: str,
    chain_count: int = 0,
    total_score: int = 0,
    ojama_sent: int = 0,
    game_idx: int = 0,
    elapsed_sec: float = 0.0,
) -> ChainEventObservation:
    """テスト用の ChainEventObservation 生成ショートカット。

    `authoritative_ojama` 引数は Fix【5】(2026-08-25) で削除した
    (自己検算そのものを廃止したため、クラス docstring 参照)。
    """
    return ChainEventObservation(
        side=side, t_sec=t_sec, mechanism=mechanism,
        chain_count=chain_count, total_score=total_score,
        ojama_sent=ojama_sent, game_idx=game_idx, elapsed_sec=elapsed_sec,
    )


# Gate 3-0 実測パターン (tests/test_chain_id_resolver.py の
# test_gate3_split_pattern_merges_into_one_chain_id と同一実測値):
# formula_read cc=1..11 (暫定 26420 点) -> baseline cc=11 (確定 25700 点)。
# elapsed_sec=0 (マージンタイム未発動) のレート 70 で換算すると
# 暫定 26420 // 70 = 377 個、確定 25700 // 70 = 367 個 (差 -10 個)。
# ojama_sent は W38 (2026-08-25 判明) の影響下にあるため参照専用として残すだけ
# (会計にも検算にも使わない。自己検算そのものは Fix【5】で廃止済み)。
_GATE3_0_OBSERVATIONS = [
    _obs("1P", 777.37, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=100),
    _obs("1P", 791.23, CHAIN_MECHANISM_FORMULA_READ, chain_count=11, total_score=26420),
    _obs("1P", 793.30, CHAIN_MECHANISM_BASELINE, chain_count=11, total_score=25700),
]


# ---------------------------------------------------------------------------
# 1. enabled=False の no-op 保証
# ---------------------------------------------------------------------------

def test_disabled_tracker_observe_is_a_noop() -> None:
    """enabled=False (既定) では observe を何回呼んでも diagnostics が全ゼロ。"""
    tracker = ExchangeEpisodeTracker()
    for obs in _GATE3_0_OBSERVATIONS:
        tracker.observe(obs)
        tracker.observe(obs)  # 何回呼んでも変化しないことを固定する
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d7.chain_id_count == 0
    assert diag.unknown_mechanism_count == 0
    assert diag.d1.total_generated == 0.0
    assert diag.d2.divergences == ()
    assert diag.d3.chain_id_force_cut_count == 0
    assert diag.d4.counts_by_reason == {}
    assert diag.d6.divergence_event_count == 0


# ---------------------------------------------------------------------------
# 2. Gate 3-0 実測パターン -> chain_id 数 1
# ---------------------------------------------------------------------------

def test_gate3_0_pattern_yields_single_chain_id() -> None:
    """formula_read cc=1..11 -> baseline cc=11 は 1 本の chain_id に統合される。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    for obs in _GATE3_0_OBSERVATIONS:
        tracker.observe(obs)
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d7.chain_id_count == 1
    assert diag.d7.step_counts == (11,)
    assert diag.d7.growth_observed_ratio == 1.0


# ---------------------------------------------------------------------------
# 3. game_idx が変わると chain/episode が試合を跨がない
# ---------------------------------------------------------------------------

def test_game_idx_boundary_force_closes_in_flight_chain() -> None:
    """境界を跨ぐ発火は前試合の連鎖を確定なしで閉じ、別 chain_id にする。

    2026-08-25 是正: BASELINE (`CHAIN_SETTLED`) はもう確定値を運ばないため、
    第 2 の連鎖を FINALIZED にするには `observe_generation()`
    (score OCR 確定差分の別チャネル) を追加で呼ぶ必要がある。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_FORMULA_READ,
        chain_count=1, total_score=100, game_idx=0,
    ))
    # CHAIN_SETTLED (連鎖終了の合図) が来る前に次の試合の発火が来る -> 境界で確定なしクローズ
    tracker.observe(_obs(
        "1P", 20.0, CHAIN_MECHANISM_FORMULA_READ,
        chain_count=1, total_score=50, game_idx=1,
    ))
    tracker.observe(_obs(
        "1P", 22.0, CHAIN_MECHANISM_BASELINE,
        chain_count=1, total_score=50, game_idx=1, ojama_sent=0,
    ))
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=22.0, game_idx=1, generated_delta=1),
    )
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d7.chain_id_count == 2
    assert diag.d4.counts_by_reason.get(CloseReason.MATCH_BOUNDARY.name) == 1
    assert diag.d4.counts_by_reason.get(CloseReason.FINALIZED.name) == 1


# ---------------------------------------------------------------------------
# 4. 未知の mechanism はカウンタに載り FORMULA_STEP に混ざらない
# ---------------------------------------------------------------------------

def test_unknown_mechanism_is_counted_not_mixed_into_formula_step() -> None:
    """未知の mechanism (score_jump 等) はカウンタに記録し無視する。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 1.0, CHAIN_MECHANISM_SCORE_JUMP, chain_count=1, total_score=10,
    ))
    for obs in _GATE3_0_OBSERVATIONS:
        tracker.observe(obs)
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.unknown_mechanism_count == 1
    # score_jump が formula_step に混ざっていれば Gate3-0 の期待値がずれる。
    assert diag.d7.chain_id_count == 1
    assert diag.d7.step_counts == (11,)


# ---------------------------------------------------------------------------
# 5. D2 finalize 乖離 (単位はおじゃま個数、スコア点ではない)
# ---------------------------------------------------------------------------

def test_finalize_divergence_uses_ojama_units_not_score_points() -> None:
    """実測 (暫定 26420 -> 確定 25700 点) はおじゃま個数換算で -10 個の乖離。

    2026-08-24 コーディネーター指摘: total_score (点) をそのまま台帳の
    amount (個数) に渡すと FINALIZE_DOWNWARD_TOLERANCE(=4 個) との比較が
    無意味になる欠陥があった (720 点 vs 4 個)。score_to_ojama (rate=70,
    elapsed_sec=0) で 26420->377 個、25700->367 個に換算してから差を取る。

    2026-08-25 是正: BASELINE (`CHAIN_SETTLED`) はもう確定値を運ばない。
    確定値は `observe_generation()` (score OCR 確定差分。**既に
    おじゃま個数、換算不要**) で供給する。367 は旧仕様と同じ
    `score_to_ojama(25700, ...)` の結果をそのまま使う (実運用では
    `OjamaAccountingTracker` がこの換算を内部で行う)。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    for obs in _GATE3_0_OBSERVATIONS:
        tracker.observe(obs)
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=793.30, game_idx=0, generated_delta=367),
    )
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d2.divergences == (-10.0,)
    assert -10.0 < -FINALIZE_DOWNWARD_TOLERANCE
    assert diag.d2.gate_held_count == 1


def test_margin_time_applies_same_rate_to_provisional_and_confirmed() -> None:
    """経過秒 300 秒 (マージンタイム発動中) でも暫定・確定が同じレートで換算される。

    §4.1.2「暫定も確定と同一の score_to_ojama・同一の経過時刻で換算する」の
    固定テスト。経過秒 0 と 300 の両方で、期待値 (compute_effective_rate を
    そのまま使って手計算した値) と一致することを確認する。

    **2026-08-25 判断 (勝手に決めず報告):** BASELINE (`CHAIN_SETTLED`) は
    もう確定値を運ばないため、この不変条件の担い手が変わった。暫定側は
    引き続き tracker 自身が `_to_ojama` で換算するが、確定側は
    `observe_generation()` で**既に換算済みの値**を受け取るだけになった
    (換算するのは呼び出し側=`OjamaAccountingTracker` の責務に移った)。
    このテストは「もし呼び出し側が同じレートで正しく換算していれば、
    tracker がその値を壊さずに通す」ことだけを検証する
    (tracker 自身が両方を換算する一枚岩の保証ではなくなった)。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 300.0, CHAIN_MECHANISM_FORMULA_READ,
        chain_count=1, total_score=26420, elapsed_sec=300.0,
    ))
    tracker.observe(_obs(
        "1P", 303.0, CHAIN_MECHANISM_BASELINE,
        chain_count=1, total_score=25700, elapsed_sec=300.0,
    ))
    rate = compute_effective_rate(300.0, OJAMA_RATE_STANDARD)
    expected_prov = 26420 // rate
    expected_conf = 25700 // rate
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=303.0, game_idx=0, generated_delta=expected_conf),
    )
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d2.divergences == (float(expected_conf - expected_prov),)


# ---------------------------------------------------------------------------
# 6. D1 保存則 — episode 単位で検査される
# ---------------------------------------------------------------------------

def test_conservation_violation_zero_for_normal_input() -> None:
    """Gate 3-0 パターン (episode が OPEN のまま終わる) は保存則違反として数えない。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    for obs in _GATE3_0_OBSERVATIONS:
        tracker.observe(obs)
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d1.conservation_violation_count == 0


def test_no_cancel_keeps_episode_open_after_e1_removal() -> None:
    """初版仕様の E1 (net_raw==0 で閉じる) を固定していたテスト。

    E1 は 2026-08-24 に削除された (相殺の記録なしに決着扱いする欠陥、
    fable アーキ裁定)。現在の期待値は『相殺が供給されるまで閉じない』
    である。

    両側が同額発火して net_raw=0 になっても、CANCEL/LAND が一切
    供給されていなければ episode は OPEN のまま残る。生成量だけが
    積まれ相殺が確定していないため `closed_episodes()` には何も積まれず、
    D1 は全ゼロ (episodes_without_settlement_input も 0 — そもそも
    CLOSED した episode 自体が無いので「無settlementで閉じた」件数にすら
    数えられない)。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=700,
    ))
    tracker.observe(_obs(
        "1P", 12.0, CHAIN_MECHANISM_BASELINE, chain_count=1, total_score=700,
        ojama_sent=10,
    ))
    tracker.observe(_obs(
        "2P", 20.0, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=700,
    ))
    tracker.observe(_obs(
        "2P", 22.0, CHAIN_MECHANISM_BASELINE, chain_count=1, total_score=700,
        ojama_sent=10,
    ))
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d1.conservation_violation_count == 0
    assert diag.d1.episodes_without_settlement_input == 0
    assert diag.d1.total_generated == 0.0


def test_conservation_violation_when_cancel_overshoots_amount() -> None:
    """CANCEL を生成量より多く供給すると (クリップにより) 保存則違反が検出される。

    2026-08-24 の E1 削除により「CANCEL/LAND が生成量ちょうどに達するまで
    episode は閉じない」ようになったため、**過小**な CANCEL では
    そもそも E2 (`_all_settled()`) が成立せず episode が閉じられない
    (前掲テストの通り)。したがって「settlement input はあるのに量が
    合わない」を作るには、生成量を超えて CANCEL を供給し
    `ChainRecord.outstanding` のクリップ (`max(0.0, amount-canceled-landed)`)
    で E2 を満たしてしまうケースを使うしかない。過小ではなく過大供給だが、
    検査したい内容 (CANCEL はあるのに保存則が壊れる) は同じ。

    `ChainEventObservation` には CANCEL/LAND に対応する概念が無い
    (Gate 3-2b で `src/ojama_accounting.py` から配線予定) ため、
    tracker が内部で保持する `ExchangeLedger` へ直接供給するホワイトボックス
    的セットアップにしたうえで、結果は tracker の public API で読む。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    ledger = tracker._ledger  # noqa: SLF001 -- Gate 3-2b で配線される経路の代替
    ctx = PhysicalContext(game_idx=0)
    ledger.push(ExchangeEvent(
        kind=EventKind.FIRE, side=Side.P1, t_sec=1.0, chain_id=101, amount=10.0,
    ), ctx)
    ledger.push(ExchangeEvent(
        kind=EventKind.FIRE, side=Side.P2, t_sec=1.0, chain_id=102, amount=10.0,
    ), ctx)
    ledger.push(ExchangeEvent(
        kind=EventKind.FINALIZE, side=Side.P1, t_sec=1.5, chain_id=101, amount=10.0,
    ), ctx)
    ledger.push(ExchangeEvent(
        kind=EventKind.FINALIZE, side=Side.P2, t_sec=1.5, chain_id=102, amount=10.0,
    ), ctx)
    # 生成量 (10 個) を超えて 15 個キャンセルする (現実にはあり得ない量だが、
    # 会計側のクリップ挙動を突いて保存則を壊す目的で意図的に供給する)。
    ledger.push(ExchangeEvent(
        kind=EventKind.CANCEL, side=Side.P1, t_sec=2.0, chain_id=101, amount=15.0,
    ), ctx)
    ledger.push(ExchangeEvent(
        kind=EventKind.CANCEL, side=Side.P2, t_sec=2.0, chain_id=102, amount=15.0,
    ), ctx)
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d1.episodes_without_settlement_input == 0
    assert diag.d1.conservation_violation_count == 1


def test_d1_detects_oversettlement_even_when_conservation_check_passes() -> None:
    """D1 で『保存則違反 0 件だが oversettled_total > 0』が検出できることを固定する。

    2026-08-24 コーディネーター指摘への回帰テスト (本体)。

    正常 CLOSED (E2) の episode では、`outstanding` のクリップにより
    `canceled+landed >= amount` が全 chain で成り立つため、数学的に必ず
    `|generated - settled| == oversettled_total` になる。つまり
    「正常 CLOSED かつ oversettled>0」を作ると**必ず保存則違反も同時に立つ**
    (これは前掲 `test_conservation_violation_when_cancel_overshoots_amount`
    が実際に示した通り)。

    保存則違反とは独立に oversettled だけを検出できることを示すには、
    `CLOSED_FORCED` (仕様 I7 により保存則検査そのものの対象外) の episode に
    過大供給の chain を混ぜる必要がある。片方の chain (201) は二重供給等で
    生成量を超えて CANCEL され、もう片方 (202) は一切未決着のまま残るので
    `_all_settled()` が成立せず、`EPISODE_MAX_SEC` の安全弁で強制終了する。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    ledger = tracker._ledger  # noqa: SLF001 -- Gate 3-2b で配線される経路の代替
    ctx = PhysicalContext(game_idx=0)
    ledger.push(ExchangeEvent(
        kind=EventKind.FIRE, side=Side.P1, t_sec=1.0, chain_id=201, amount=10.0,
    ), ctx)
    ledger.push(ExchangeEvent(
        kind=EventKind.FIRE, side=Side.P2, t_sec=1.0, chain_id=202, amount=10.0,
    ), ctx)
    ledger.push(ExchangeEvent(
        kind=EventKind.FINALIZE, side=Side.P1, t_sec=1.5, chain_id=201, amount=10.0,
    ), ctx)
    ledger.push(ExchangeEvent(
        kind=EventKind.FINALIZE, side=Side.P2, t_sec=1.5, chain_id=202, amount=10.0,
    ), ctx)
    # chain201 だけ二重供給等で 20 個キャンセルする (過大供給、oversettled=10)。
    # chain202 は相殺・着弾を一切受けないまま outstanding=10 で残す。
    ledger.push(ExchangeEvent(
        kind=EventKind.CANCEL, side=Side.P1, t_sec=2.0, chain_id=201, amount=20.0,
    ), ctx)
    # chain202 が残っているため E2 は成立しない。EPISODE_MAX_SEC (60 秒)
    # 経過を検知させて強制終了 (CLOSED_FORCED) させる。
    ledger.push(ExchangeEvent(
        kind=EventKind.TSUMO_PLACED, side=Side.P1, t_sec=100.0,
    ), ctx)
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d1.conservation_violation_count == 0
    assert diag.d1.oversettled_total == 10.0
    assert diag.d1.oversettled_chain_count == 1


# ---------------------------------------------------------------------------
# 7. diagnostics() の冪等性
# ---------------------------------------------------------------------------

def test_diagnostics_is_idempotent() -> None:
    """diagnostics() を 2 回呼んでも内部状態を変えず同じ値を返す。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    for obs in _GATE3_0_OBSERVATIONS:
        tracker.observe(obs)
    tracker.finish()
    first = tracker.diagnostics()
    second = tracker.diagnostics()
    assert first == second


# ---------------------------------------------------------------------------
# 8. 空入力で例外なし
# ---------------------------------------------------------------------------

def test_empty_input_does_not_raise() -> None:
    """observe を一度も呼ばずに finish/diagnostics を呼んでも例外なし。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d7.chain_id_count == 0
    assert isinstance(diag, ExchangeEpisodeDiagnostics)


# ---------------------------------------------------------------------------
# 補助: D3 素点検算・自己検算・D4 迷子信号
# ---------------------------------------------------------------------------

def test_d3_score_multiple_of_ten_not_implemented() -> None:
    """D3 の素点検算は ChainEventObservation に base_score が無く未実装 (None)。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    for obs in _GATE3_0_OBSERVATIONS:
        tracker.observe(obs)
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d3.score_multiple_of_ten_violation_count is None


def test_d3_no_longer_has_self_check_fields() -> None:
    """Fix【5】(2026-08-25): 自己検算フィールドを削除したことを固定する。

    `SCORE_FINALIZE` の値がそのまま会計の確定値になる新設計では、それを
    検算する独立な第二の権威値が存在しない (`authoritative_ojama` は
    `finalized_score` と同一の accumulator 由来で、常に差 0 の
    tautological な検算だった。2026-08-25 実測で `n_authoritative_ojama_
    present = 0/20, 0/7` も確認済み)。意味のない 0 を残さないため
    フィールドごと削除した。
    """
    import dataclasses

    from src.exchange_episode_tracker import D3ForcedCloseCounters

    field_names = {f.name for f in dataclasses.fields(D3ForcedCloseCounters)}
    assert "self_check_mismatch_count" not in field_names
    assert "self_check_max_abs_diff" not in field_names
    assert "self_check_skipped_count" not in field_names

    obs_field_names = {f.name for f in dataclasses.fields(ChainEventObservation)}
    assert "authoritative_ojama" not in obs_field_names


def test_d4_orphan_end_signal_count_is_zero_when_no_signal_sent() -> None:
    """tracker は CHAIN_END_SIGNAL を一切送らないため迷子信号数は常に 0。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    for obs in _GATE3_0_OBSERVATIONS:
        tracker.observe(obs)
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d4.orphan_end_signal_count == 0


# ---------------------------------------------------------------------------
# 相殺・着弾の観測 (Gate 3-2b 純粋部分、2026-08-24 追加)
# ---------------------------------------------------------------------------

def _settlement(
    t_sec: float,
    game_idx: int = 0,
    canceled_by_1p: float = 0.0,
    canceled_by_2p: float = 0.0,
    landed_on_1p: float = 0.0,
    landed_on_2p: float = 0.0,
) -> SettlementObservation:
    """テスト用の SettlementObservation 生成ショートカット。"""
    return SettlementObservation(
        t_sec=t_sec, game_idx=game_idx,
        canceled_by_1p=canceled_by_1p, canceled_by_2p=canceled_by_2p,
        landed_on_1p=landed_on_1p, landed_on_2p=landed_on_2p,
    )


def _fire_and_finalize(
    tracker: ExchangeEpisodeTracker, side: str,
    fire_t: float, finalize_t: float, total_score: int, ojama_sent: int = 0,
) -> None:
    """テスト用: FORMULA_READ -> BASELINE の 1 本の連鎖を作るショートカット。"""
    tracker.observe(_obs(side, fire_t, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=total_score))
    tracker.observe(_obs(
        side, finalize_t, CHAIN_MECHANISM_BASELINE, chain_count=1,
        total_score=total_score, ojama_sent=ojama_sent,
    ))


def test_settlement_attribution_1p_cancels_2p_chain() -> None:
    """1P が相殺したとき、2P の chain の canceled が増える (向きの検査)。

    帰属規則: 台帳の `ChainRecord.canceled` は「その連鎖が生成した量の
    うち打ち消された量」。1P が相殺した量は 2P の連鎖が生成した量なので、
    2P の chain へ帰属しなければならない。逆向きになれば符号が反転する
    (2026-08-10 の側取り違えバグの再発)。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    _fire_and_finalize(tracker, "1P", 10.0, 12.0, 700, ojama_sent=10)
    _fire_and_finalize(tracker, "2P", 11.0, 13.0, 700, ojama_sent=10)
    tracker.observe_settlement(_settlement(14.0, canceled_by_1p=4.0))
    tracker.finish()
    chains = tracker._ledger._chains  # noqa: SLF001
    p1_chain = next(c for c in chains.values() if c.side is Side.P1)
    p2_chain = next(c for c in chains.values() if c.side is Side.P2)
    assert p2_chain.canceled == 4.0, "相手 (2P) の chain が相殺されるべき"
    assert p1_chain.canceled == 0.0, "自分 (1P) の chain が相殺されてはいけない"


def test_settlement_attribution_is_symmetric_for_2p() -> None:
    """1P/2P を入れ替えても帰属の向きは対称: 2P が相殺すると 1P の chain が増える。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    _fire_and_finalize(tracker, "1P", 10.0, 12.0, 700, ojama_sent=10)
    _fire_and_finalize(tracker, "2P", 11.0, 13.0, 700, ojama_sent=10)
    tracker.observe_settlement(_settlement(14.0, canceled_by_2p=4.0))
    tracker.finish()
    chains = tracker._ledger._chains  # noqa: SLF001
    p1_chain = next(c for c in chains.values() if c.side is Side.P1)
    p2_chain = next(c for c in chains.values() if c.side is Side.P2)
    assert p1_chain.canceled == 4.0, "相手 (1P) の chain が相殺されるべき"
    assert p2_chain.canceled == 0.0, "自分 (2P) の chain が相殺されてはいけない"


def test_settlement_attribution_fifo_order() -> None:
    """帰属先が複数本あるとき、古い chain (chain_id が小さい方) から消化される。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    _fire_and_finalize(tracker, "2P", 10.0, 11.0, 350, ojama_sent=5)   # 350//70=5個
    _fire_and_finalize(tracker, "2P", 20.0, 21.0, 350, ojama_sent=5)
    # 1P が 3 個だけ相殺 -> 古い方 (先に開いた chain) から消化されるはず。
    tracker.observe_settlement(_settlement(22.0, canceled_by_1p=3.0))
    tracker.finish()
    chains = tracker._ledger._chains  # noqa: SLF001
    p2_chains = sorted((cid, c) for cid, c in chains.items() if c.side is Side.P2)
    assert len(p2_chains) == 2
    _, old_chain = p2_chains[0]
    _, new_chain = p2_chains[1]
    assert old_chain.canceled == 3.0, "古い chain から消化されるはず"
    assert new_chain.canceled == 0.0, "新しい chain はまだ手つかずのはず"


def test_settlement_unattributed_when_no_open_chain_exists() -> None:
    """帰属先の chain が無ければ unattributed_settlement_total に計上され、例外は出ない。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe_settlement(_settlement(5.0, canceled_by_1p=7.0))
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d1.unattributed_settlement_total == 7.0


def test_land_split_for_large_amount_does_not_raise() -> None:
    """差分 70 個の着弾が 30/30/10 の 3 イベントに分割され、例外が出ない。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    _fire_and_finalize(tracker, "1P", 10.0, 12.0, 4900, ojama_sent=70)  # 4900//70=70個
    tracker.observe_settlement(_settlement(13.0, landed_on_2p=70.0))
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d3.land_split_count == 1
    assert diag.d1.total_landed == 70.0
    assert diag.d1.unattributed_settlement_total == 0.0


def test_disabled_tracker_observe_settlement_is_a_noop() -> None:
    """enabled=False では observe_settlement を呼んでも何も起きない。"""
    tracker = ExchangeEpisodeTracker()
    tracker.observe_settlement(_settlement(1.0, canceled_by_1p=10.0))
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d1.unattributed_settlement_total == 0.0
    assert diag.d3.land_split_count == 0


def test_correct_settlement_supply_closes_episode_normally() -> None:
    """相殺・着弾を正しく供給すると episode が正常に CLOSED し保存則が満たされる。

    これが E1 削除後に『配線すれば正常に閉じる』ことの証明である。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    _fire_and_finalize(tracker, "1P", 10.0, 12.0, 700, ojama_sent=10)
    tracker.observe_settlement(_settlement(13.0, landed_on_2p=10.0))
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d1.conservation_violation_count == 0
    assert diag.d1.episodes_without_settlement_input == 0
    assert diag.d1.total_landed == 10.0
    assert diag.d1.unattributed_settlement_total == 0.0


def test_land_split_events_keep_the_real_observed_timestamp() -> None:
    """分割された着弾イベントは、全て同一の本物の観測時刻を持つ。

    2026-08-24 コーディネーター指摘への回帰テスト:
    時刻を偽装 (微小オフセット) して重複排除 (I4) をすり抜けるのは禁じ手。
    分割は `ExchangeEvent.seq` (0, 1, 2, ...) で区別し、`t_sec` は
    観測された値のまま変えないことを固定する。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    _fire_and_finalize(tracker, "1P", 10.0, 12.0, 4900, ojama_sent=70)  # 4900//70=70個
    # 2P 側にも未決着 chain を残し、episode が CLOSED せず events を読めるようにする。
    _fire_and_finalize(tracker, "2P", 11.0, 13.0, 350, ojama_sent=5)   # 350//70=5個
    tracker.observe_settlement(_settlement(14.0, landed_on_2p=70.0))
    tracker.finish()
    episode = tracker._ledger.current_episode()  # noqa: SLF001
    assert episode is not None, "2P 側が未決着のまま残るので episode はまだ OPEN のはず"
    land_events = [e for e in episode.events if e.kind is EventKind.LAND]
    assert len(land_events) == 3, "70 個は 30/30/10 の 3 イベントに分割されるはず"
    assert {e.t_sec for e in land_events} == {14.0}, "全イベントが同一の本物の時刻を持つべき"
    assert sorted(e.seq for e in land_events) == [0, 1, 2]


# ===========================================================================
# 2026-08-25 コーディネーター是正3点の回帰テスト
# ===========================================================================

def test_observe_generation_injects_match_boundary_without_prior_observe_call() -> None:
    """`observe_generation()` 単独でも試合境界を検出し、前試合の chain へ
    確定値が紛れ込まない。

    試合境界の直後、次の `observe()` (chain_detector 由来) が来る前に
    `observe_generation()` (score OCR 由来) だけが新しい試合として届く
    順序でも、境界を取り逃してはいけない
    (「試合を跨ぐ汚染」と同じ形の欠陥をこのチャネルでも防ぐ、
    コーディネーター指摘の回帰本体)。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_FORMULA_READ,
        chain_count=1, total_score=100, game_idx=0,
    ))
    # observe() を一切挟まず、observe_generation() だけで試合1に切り替わる。
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=20.0, game_idx=1, generated_delta=50),
    )
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d7.chain_id_count == 2, "前試合の chain と新試合の chain で 2 本のはず"
    assert diag.d4.counts_by_reason.get(CloseReason.MATCH_BOUNDARY.name) == 1, (
        "前試合の chain が境界で確定なしクローズされていない"
    )
    assert diag.d4.counts_by_reason.get(CloseReason.FINALIZED.name) == 1, (
        "確定値が新試合の chain_id ではなく前試合の chain に紛れ込んだ"
    )


def test_observe_generation_negative_delta_is_counted_and_ignored() -> None:
    """`generated_delta < 0` (累積カウンタの減少、物理的にありえない) は
    黙って捨てず `negative_generation_delta_count` に記録し、会計には
    反映しない。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_FORMULA_READ,
        chain_count=1, total_score=100, game_idx=0,
    ))
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=11.0, game_idx=0, generated_delta=-30),
    )
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.negative_generation_delta_count == 1
    assert diag.d4.counts_by_reason.get(CloseReason.FINALIZED.name) is None, (
        "負の増分が会計 (確定) に反映されてしまった"
    )


def test_observe_generation_zero_delta_is_not_counted_as_negative() -> None:
    """`generated_delta == 0` (本当に増分が無い) は負ではないので
    `negative_generation_delta_count` に数えない (無視するだけ)。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=1.0, game_idx=0, generated_delta=0),
    )
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.negative_generation_delta_count == 0


def test_d2_separates_accepted_from_rejected_divergence() -> None:
    """台帳が既定で拒否する `simulate_fallback` の乖離は D2 の分布
    (`divergences`) に混ぜず、件数だけ `rejected_divergence_count` に
    別枠で出す (拒否された値で閾値設計を誤らないため)。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    # 1P: 成長観測 + observe_generation() で score_ocr_diff 確定 (accepted 側)。
    tracker.observe(_obs(
        "1P", 777.37, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=100,
    ))
    tracker.observe(_obs(
        "1P", 791.23, CHAIN_MECHANISM_FORMULA_READ, chain_count=11, total_score=26420,
    ))
    tracker.observe(_obs(
        "1P", 793.30, CHAIN_MECHANISM_BASELINE, chain_count=11, total_score=25700,
    ))
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=793.30, game_idx=0, generated_delta=367),
    )
    # 2P: 掛け算式を一切観測できず baseline (CHAIN_SETTLED) だけで即クローズ
    # = simulate_fallback (rejected 側、observe_generation() を呼ばない)。
    tracker.observe(_obs(
        "2P", 5.0, CHAIN_MECHANISM_BASELINE, chain_count=4, total_score=400,
    ))
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d2.divergences == (-10.0,), "accepted 側の乖離だけが分布に残るべき"
    assert diag.d2.rejected_divergence_count == 1, "simulate_fallback 側が数えられていない"


# ===========================================================================
# Fix【1】: 観測経路を「上限なしの保留レベルの差分」から再構成する
# (2026-08-25 追加、gate3_episode_v3 実装タスク)
# ===========================================================================

def _frame(
    t_sec: float,
    p1: float,
    p2: float,
    *,
    game_idx: int = 0,
    p1_tsumo: bool = False,
    p2_tsumo: bool = False,
    p1_fin: bool = False,
    p2_fin: bool = False,
) -> PendingUncappedFrame:
    """テスト用の PendingUncappedFrame 生成ショートカット。"""
    return PendingUncappedFrame(
        t_sec=t_sec, game_idx=game_idx, p1_uncapped=p1, p2_uncapped=p2,
        p1_tsumo_placed=p1_tsumo, p2_tsumo_placed=p2_tsumo,
        p1_chain_finalized=p1_fin, p2_chain_finalized=p2_fin,
    )


def test_pending_delta_increase_is_ignored() -> None:
    """増加 (相手が生成して送ってきた) は決済事象ではないので無視する。"""
    prev = _frame(0.0, 0.0, 0.0)
    curr = _frame(1.0, 50.0, 0.0)
    result = classify_pending_uncapped_delta(prev, curr)
    assert result.settlement is None
    assert result.wiped_sides == ()
    assert result.unclassified_drop_p1 == 0.0
    assert result.unclassified_drop_p2 == 0.0


def test_pending_delta_decrease_with_chain_finalized_is_cancel() -> None:
    """自分の連鎖確定と同時の減少は CANCEL として判別する。"""
    prev = _frame(0.0, 500.0, 0.0)
    curr = _frame(1.0, 480.0, 0.0, p1_fin=True)
    result = classify_pending_uncapped_delta(prev, curr)
    assert result.settlement is not None
    assert result.settlement.canceled_by_1p == pytest.approx(20.0)
    assert result.settlement.landed_on_1p == 0.0
    assert result.wiped_sides == ()


def test_pending_delta_decrease_with_tsumo_placed_within_cap_is_land() -> None:
    """ツモ設置直後・30 個以下の減少は LAND として判別する。"""
    prev = _frame(0.0, 100.0, 0.0)
    curr = _frame(1.0, 75.0, 0.0, p1_tsumo=True)
    result = classify_pending_uncapped_delta(prev, curr)
    assert result.settlement is not None
    assert result.settlement.landed_on_1p == pytest.approx(25.0)
    assert result.settlement.canceled_by_1p == 0.0


def test_pending_delta_decrease_exceeding_turn_cap_without_finalize_is_unclassified() -> None:
    """ツモ設置直後でも 30 個を超える減少は LAND と判定しない (未分類として計上)。"""
    prev = _frame(0.0, 100.0, 0.0)
    curr = _frame(1.0, 50.0, 0.0, p1_tsumo=True)  # 50 個減、OJAMA_MAX_DROP_PER_TURN(30)超
    result = classify_pending_uncapped_delta(prev, curr)
    assert result.settlement is None
    assert result.unclassified_drop_p1 == pytest.approx(50.0)
    assert result.wiped_sides == ()


def test_pending_delta_drop_to_zero_without_finalize_or_tsumo_is_wipe() -> None:
    """相殺・着弾のどちらの条件にも一致せず 0 になった減少はワイプと判別する。"""
    prev = _frame(0.0, 40.0, 0.0)
    curr = _frame(1.0, 0.0, 0.0)
    result = classify_pending_uncapped_delta(prev, curr)
    assert result.settlement is None
    assert result.wiped_sides == (Side.P1,)
    assert result.unclassified_drop_p1 == 0.0


def test_pending_delta_drop_not_to_zero_without_signal_is_unclassified() -> None:
    """0 にならない・どの条件にも一致しない減少は黙って捨てず未分類に積む。"""
    prev = _frame(0.0, 40.0, 0.0)
    curr = _frame(1.0, 30.0, 0.0)
    result = classify_pending_uncapped_delta(prev, curr)
    assert result.settlement is None
    assert result.wiped_sides == ()
    assert result.unclassified_drop_p1 == pytest.approx(10.0)


def test_pending_delta_chain_finalized_takes_priority_over_tsumo_placed() -> None:
    """理論上の衝突 (同一フレームで連鎖確定とツモ設置が両方真) では
    連鎖確定 (CANCEL) を優先する (より直接的な観測のため)。"""
    prev = _frame(0.0, 100.0, 0.0)
    curr = _frame(1.0, 80.0, 0.0, p1_tsumo=True, p1_fin=True)
    result = classify_pending_uncapped_delta(prev, curr)
    assert result.settlement.canceled_by_1p == pytest.approx(20.0)
    assert result.settlement.landed_on_1p == 0.0


def test_pending_delta_both_sides_classified_independently() -> None:
    """1P/2P は互いに独立して判別される (片方 CANCEL・もう片方 LAND)。"""
    prev = _frame(0.0, 100.0, 60.0)
    curr = _frame(1.0, 80.0, 40.0, p1_fin=True, p2_tsumo=True)
    result = classify_pending_uncapped_delta(prev, curr)
    assert result.settlement.canceled_by_1p == pytest.approx(20.0)
    assert result.settlement.landed_on_2p == pytest.approx(20.0)
    assert result.settlement.canceled_by_2p == 0.0
    assert result.settlement.landed_on_1p == 0.0


def test_pending_delta_uncapped_survives_beyond_216_cap() -> None:
    """根因 (A) への対処: 上限 216 を超えても真の生成が観測から漏れない
    (uncapped 値をそのまま渡せば cap の影響を受けない)。"""
    prev = _frame(0.0, 0.0, 0.0)
    curr = _frame(1.0, 300.0, 0.0)  # PENDING_ABS_CAP(216) を明確に超える値
    result = classify_pending_uncapped_delta(prev, curr)
    assert result.settlement is None, "増加なので決済ではないが、値自体は 300 のまま扱える"

    prev2 = _frame(1.0, 300.0, 0.0)
    curr2 = _frame(2.0, 0.0, 0.0, p1_fin=True)
    result2 = classify_pending_uncapped_delta(prev2, curr2)
    assert result2.settlement.canceled_by_1p == pytest.approx(300.0), (
        "216 で切り捨てられていたら 216 にしかならないはずの量が満額観測できている"
    )


def _gross(t_sec: float, **overrides: int) -> GrossOjamaCounters:
    """gross 累積カウンタのテスト用生成関数。"""
    values = {
        "generated_p1": 0, "generated_p2": 0,
        "offset_uncapped_p1": 0, "offset_uncapped_p2": 0,
        "dropped_uncapped_p1": 0, "dropped_uncapped_p2": 0,
        "boundary_wiped_uncapped_p1": 0, "boundary_wiped_uncapped_p2": 0,
        "boundary_resets_p1": 0, "boundary_resets_p2": 0,
        "clamp_loss_p1": 0, "clamp_loss_p2": 0,
    }
    values.update(overrides)
    return GrossOjamaCounters(t_sec=t_sec, **values)


def test_gross_delta_keeps_simultaneous_incoming_and_cancel_separate() -> None:
    """pending 100→80でも、gross cancel=50 と incoming=30を別々に復元する。"""
    result = classify_gross_counter_delta(
        _gross(0.0),
        _gross(1.0, generated_p1=50, generated_p2=30, offset_uncapped_p1=50),
        prev_pending=(100.0, 0.0), curr_pending=(80.0, 0.0), game_idx=0,
    )
    assert isinstance(result, GrossCounterDeltaClassification)
    assert result.settlement is not None
    assert result.settlement.canceled_by_1p == 50
    assert result.generated_by_2p == 30
    assert result.conservation_residual_p1 == 0.0
    assert result.conservation_residual_p2 == 0.0
    assert result.inspected_side_count == 2


def test_gross_delta_reconstructs_land_plus_incoming_without_guessing() -> None:
    result = classify_gross_counter_delta(
        _gross(0.0),
        _gross(1.0, generated_p2=10, dropped_uncapped_p1=30),
        prev_pending=(100.0, 0.0), curr_pending=(80.0, 0.0), game_idx=0,
    )
    assert result.settlement is not None
    assert result.settlement.landed_on_1p == 30
    assert result.generated_by_2p == 10
    assert result.conservation_residual_p1 == 0.0
    assert result.conservation_residual_p2 == 0.0


def test_gross_delta_uses_cumulative_wipe_amount_after_frame_skip() -> None:
    result = classify_gross_counter_delta(
        _gross(0.0),
        _gross(2.0, boundary_wiped_uncapped_p1=40, boundary_resets_p1=1),
        prev_pending=(40.0, 0.0), curr_pending=(0.0, 0.0), game_idx=1,
    )
    assert result.wiped_sides == (Side.P1,)
    assert result.boundary_wiped_on_1p == 40
    assert result.conservation_residual_p1 == 0.0


def test_gross_delta_rejects_cross_reset_counter_decrease() -> None:
    with pytest.raises(ValueError, match="generated_p1"):
        classify_gross_counter_delta(
            _gross(0.0, generated_p1=10), _gross(1.0),
            prev_pending=(0.0, 0.0), curr_pending=(0.0, 0.0), game_idx=0,
        )


# ===========================================================================
# 実装1 (2026-08-25 追加): 測定器の是正
# `total_unreconciled` は CLOSED/CLOSED_FORCED した episode だけを合算する
# ため、窓の終わりにまだ OPEN な episode の残量が構造的に見えない。
# `open_episode_outstanding`/`ledger_residual_all` を併記させ、
# 「0 は測っていないだけ」を検出できることを固定する。
# ===========================================================================

def test_d1_reveals_open_episode_residual_even_when_total_unreconciled_is_zero() -> None:
    """窓を早く切って強制終了が起きない状況 (episode がまだ OPEN のまま
    終わる) では `total_unreconciled=0` になるが、`open_episode_outstanding`/
    `ledger_residual_all` は残量を正しく示すこと (実装1、回帰本体)。

    **これが「0 は測っていないだけ」を検出する仕掛けである。**
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=700,
    ))
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=10.5, game_idx=0, generated_delta=10),
    )
    # 相殺・着弾を一切供給しない (= 窓がこの撃ち合いの続きを見る前に終わる想定)。
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d1.total_unreconciled == 0.0, "前提: closed episode がまだ無い"
    assert diag.d1.open_episode_outstanding == pytest.approx(10.0), (
        "OPEN な episode の残量が見えていない"
    )
    assert diag.d1.ledger_residual_all == pytest.approx(10.0), (
        "台帳の生値 (ledger_residual_all) が 0 に見えてしまっている"
    )


# ===========================================================================
# 実装2 (2026-08-25 追加): 退役した chain の相殺・生成量の転記
# ===========================================================================

def test_retired_chain_cancel_and_generated_are_not_dropped_from_totals() -> None:
    """退役 (`observe_wipe`) で消えた chain の相殺・生成量が `retired_canceled`/
    `retired_generated` に転記され、黙って消えないこと (実装2、chain6 の
    相殺 18 個消失の回帰本体)。

    退役は episode が summarize される**前**に chain を削除するため、
    修正前は `total_canceled`/`total_generated` (closed_episodes 経由) に
    この chain の値が一切反映されなかった。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 0.0, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=400,
    ))
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=0.5, game_idx=0, generated_delta=40),
    )
    tracker.observe_settlement(_settlement(1.0, canceled_by_2p=15.0))
    tracker.observe_wipe(Side.P2, 2.0, 0)
    tracker.finish()
    diag = tracker.diagnostics()
    # 修正前の挙動: この chain は summarize 前に消えるので episode 側の
    # total_* には一切現れない。
    assert diag.d1.total_canceled == 0.0
    assert diag.d1.total_generated == 0.0
    # 実装2: 退役枠に転記され、黙って消えない。
    assert diag.d1.retired_canceled == pytest.approx(15.0)
    assert diag.d1.retired_generated == pytest.approx(40.0)
    assert diag.d1.retired_landed == pytest.approx(0.0)
    # 真の合計を見るときは retired_* を足すこと。
    assert diag.d1.total_canceled + diag.d1.retired_canceled == pytest.approx(15.0)
    assert diag.d1.total_generated + diag.d1.retired_generated == pytest.approx(40.0)


# ===========================================================================
# 実装3 (2026-08-25 追加): 自己相殺の根治
# `cancel_own_pending_then_send_surplus` (発火した側が自分の受け予定を
# 先に打ち消し、余りだけを相手に送る) を FIRE/FINALIZE の登録量に反映する。
# ===========================================================================

def test_self_cancel_nets_fire_amount_to_zero_when_generation_fully_self_canceled() -> None:
    """発火本人が自分の生成全額を自己相殺したとき (実測 v51 chain7: 生成19・
    自己相殺19・送るべき余り0)、chain の登録量が 0 になること
    (実装3、自己相殺根治の回帰本体)。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=1330,
    ))  # 1330 // 70 = 19 個
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=10.5, game_idx=0, generated_delta=19),
    )
    # 同一フレーム (finalize と同時) に 1P 自身の pending が 19 個ぶん
    # 減少 = 自己相殺 (cancel_own_pending_then_send_surplus)。
    tracker.observe_settlement(_settlement(10.5, canceled_by_1p=19.0))
    tracker.finish()
    diag = tracker.diagnostics()
    chains = tracker._ledger._chains  # noqa: SLF001
    chain = next(iter(chains.values()))
    assert chain.amount == pytest.approx(0.0), "自己相殺後の残量が 0 になっていない"
    assert diag.d7.self_canceled_total == pytest.approx(19.0)
    assert diag.d7.raw_generation_total == pytest.approx(19.0)


def test_self_cancel_uses_finalized_amount_when_provisional_conversion_is_near_zero() -> None:
    """provisional (掛け算式の成長フェーズ) 由来の ojama 換算が 0/小さいのに
    対し finalized (score OCR 確定差分) が大きい場合でも、自己相殺が正しく
    finalized 側へ適用されること (実測 v51 chain6/chain11 で発覚した
    クリップバグの回帰本体)。

    v51 実測: chain6 は provisional_score=0点 (ojama換算0個)・
    finalized=620個・自己相殺145個。修正前は自己相殺が `provisional_ojama`
    (=0) でクリップされて 0 に潰れ、FINALIZE 側にも自己相殺が反映されず
    620 が丸ごと残ってしまっていた (2026-08-25 実データ検証で発覚)。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=0,
    ))
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=10.5, game_idx=0, generated_delta=620),
    )
    tracker.observe_settlement(_settlement(10.5, canceled_by_1p=145.0))
    tracker.finish()
    diag = tracker.diagnostics()
    chains = tracker._ledger._chains  # noqa: SLF001
    chain = next(iter(chains.values()))
    assert chain.amount == pytest.approx(620.0 - 145.0), "自己相殺が finalized 側に反映されていない"
    assert diag.d7.self_canceled_total == pytest.approx(145.0)


def test_self_cancel_clipped_when_raw_exceeds_both_provisional_and_finalized() -> None:
    """自己相殺の生値が provisional/finalized の両方を超える壊れたデータが
    来たら、黙って丸めず `self_cancel_clipped_count`/`_amount` に計上される
    こと (コーディネーター指摘、実測では未発生・母数つきで固定する)。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=700,
    ))  # 700 // 70 = 10 個
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=10.5, game_idx=0, generated_delta=10),
    )
    # 生成 10 個に対し自己相殺 100 個という物理的にありえない過大な観測。
    tracker.observe_settlement(_settlement(10.5, canceled_by_1p=100.0))
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d7.self_cancel_eligible_count == 1
    assert diag.d7.self_cancel_clipped_count == 1, "1/1 (クリップが発生)"
    assert diag.d7.self_cancel_clipped_amount == pytest.approx(90.0)
    assert diag.d7.self_canceled_total == pytest.approx(10.0), "クリップ後の量が総和に入るべき"


def test_self_cancel_not_clipped_in_normal_case() -> None:
    """通常ケース (生値が上限以下) では `self_cancel_clipped_count` は
    0 のまま (母数とセットで 0/N を確認する)。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=1330,
    ))
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=10.5, game_idx=0, generated_delta=19),
    )
    tracker.observe_settlement(_settlement(10.5, canceled_by_1p=19.0))
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d7.self_cancel_eligible_count == 1
    assert diag.d7.self_cancel_clipped_count == 0, "0/1 になるはず"
    assert diag.d7.self_cancel_clipped_amount == pytest.approx(0.0)


def test_original_generation_equals_net_generation_plus_self_canceled() -> None:
    """元の生成 (`raw_generation_total`) = 送付分 (台帳へ登録された net 生成量)
    + 自己相殺 (`self_canceled_total`) が成り立つこと (実装3、保存則テスト)。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=1330,
    ))  # 19 個
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=10.5, game_idx=0, generated_delta=19),
    )
    tracker.observe_settlement(_settlement(10.5, canceled_by_1p=7.0))  # 一部だけ自己相殺
    tracker.finish()
    diag = tracker.diagnostics()
    chains = tracker._ledger._chains  # noqa: SLF001
    net_registered_total = sum(c.amount for c in chains.values())
    assert diag.d7.raw_generation_total == pytest.approx(19.0)
    assert diag.d7.self_canceled_total == pytest.approx(7.0)
    assert net_registered_total == pytest.approx(19.0 - 7.0)
    assert diag.d7.raw_generation_total == pytest.approx(
        net_registered_total + diag.d7.self_canceled_total,
    )


def test_observe_wipe_retires_side_and_reports_via_ledger_extra() -> None:
    """`observe_wipe` が `finish()` のタイムラインで正しく反映され、
    台帳の retire カウンタに現れることを end-to-end で確認する。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 0.0, CHAIN_MECHANISM_FORMULA_READ, chain_count=4, total_score=400,
    ))
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=0.5, game_idx=0, generated_delta=40),
    )
    tracker.observe_wipe(Side.P2, 5.0, 0)
    tracker.finish()
    snap = tracker._ledger.snapshot()  # noqa: SLF001
    assert snap.retired_chain_count == 1
    assert snap.retired_unreconciled == pytest.approx(40.0)


# ===========================================================================
# P1-1 (Codex Gate 3-2b レビュー NG、2026-08-25): I16 の迂回
#
# `simulate_fallback` の FINALIZE は台帳 (I16) が拒否するが、同じ値が先に
# provisional FIRE として登録されて会計に入っていた。Codex 最小再現:
# baseline-only total_score=52150 で finalize_rejected_count=1 /
# finalize_rejected_amount=745 なのに net_raw=745 / total_generated=745。
# 既定 (allow_simulate_fallback=False) では simulate 由来量が
# **どのイベント種別からも**会計へ入らないことを固定する。
# ===========================================================================

def test_p1_1_simulate_value_does_not_enter_accounting_via_fire() -> None:
    """【Codex 最小再現】baseline-only 52150 点 (=745 個相当) の推定値が、
    FINALIZE 拒否をすり抜けて FIRE (provisional) から会計に入らないこと。

    修正前: net_raw=745 / total_generated=745 (I16 の迂回)。
    修正後: simulate 由来 chain はイベント化そのものを行わず、除外の
    件数・量をカウンタ (母数 = chain_id_count) に出す。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_BASELINE, chain_count=5, total_score=52150,
    ))
    tracker.finish()
    snap = tracker._ledger.snapshot()  # noqa: SLF001
    assert snap.net_raw == pytest.approx(0.0), (
        f"simulate 由来量が FIRE 経由で会計に入った (net_raw={snap.net_raw})"
    )
    assert snap.total_generated == pytest.approx(0.0)
    diag = tracker.diagnostics()
    assert diag.d7.chain_id_count == 1  # 母数: 除外は 1/1
    assert diag.d7.simulate_excluded_chain_count == 1
    assert diag.d7.simulate_excluded_amount == pytest.approx(745.0)
    # 独立検算と比較する raw 生成総和にも simulate 由来量を混ぜない
    # (docstring: raw_generation_total は「score OCR 由来の値そのまま」の総和)。
    assert diag.d7.raw_generation_total == pytest.approx(0.0)
    # 可視性は D2 の rejected 側に残る (黙って消さない)。
    assert diag.d2.rejected_divergence_count == 1


def test_p1_1_allow_simulate_fallback_true_lets_simulate_enter() -> None:
    """`allow_simulate_fallback=True` を明示した場合だけ、従来どおり
    simulate 由来 chain も会計に入る (低信頼と知って使う経路の維持)。"""
    tracker = ExchangeEpisodeTracker(enabled=True, allow_simulate_fallback=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_BASELINE, chain_count=5, total_score=52150,
    ))
    tracker.finish()
    snap = tracker._ledger.snapshot()  # noqa: SLF001
    assert snap.net_raw == pytest.approx(745.0)
    assert snap.finalize_rejected_count == 0, "許容時は台帳も拒否しない"
    diag = tracker.diagnostics()
    assert diag.d7.simulate_excluded_chain_count == 0  # 0/1 (除外なし)
    assert diag.d7.chain_id_count == 1
    assert diag.d7.raw_generation_total == pytest.approx(745.0)


def test_p1_1_formula_provisional_chain_is_not_excluded() -> None:
    """除外対象は simulate 由来 (finalized_source='simulate_fallback') のみ。
    掛け算式 (formula OCR) 由来の未確定暫定は従来どおり会計に入る
    (除外条件の過剰適用の防止)。"""
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=700,
    ))  # 700 // 70 = 10 個 (未確定のまま flush)
    tracker.finish()
    snap = tracker._ledger.snapshot()  # noqa: SLF001
    assert snap.net_raw == pytest.approx(10.0)
    diag = tracker.diagnostics()
    assert diag.d7.simulate_excluded_chain_count == 0  # 0/1
    assert diag.d7.raw_generation_total == pytest.approx(10.0)


# ===========================================================================
# P1-2 の tracker 側 (D7 二重計上): v51 実データ経路 B の end-to-end 回帰
# ===========================================================================

def test_p1_2_mid_chain_finalize_does_not_double_count_in_d7() -> None:
    """途中確定 → 段継続で新 chain が開くとき、D7 の raw_generation_total が
    前半確定分を二重計上しない (v51 実測: chain5⊃chain4 で +308 個)。

    cc=1〜7 (累積 21,570 点 → 確定 308 個) + 続き cc=8 (累積 29,670 点)。
    修正前: 308 + 29670//70 = 308 + 423 = 731 個 (21,570 点分が二重)。
    修正後: 308 + (29670-21570)//70 = 308 + 115 = 423 個。
    """
    tracker = ExchangeEpisodeTracker(enabled=True)
    tracker.observe(_obs(
        "1P", 10.0, CHAIN_MECHANISM_FORMULA_READ, chain_count=1, total_score=40,
    ))
    tracker.observe(_obs(
        "1P", 11.0, CHAIN_MECHANISM_FORMULA_READ, chain_count=7, total_score=21570,
    ))
    tracker.observe_generation(
        GenerationObservation(side="1P", t_sec=11.5, game_idx=0, generated_delta=308),
    )
    tracker.observe(_obs(
        "1P", 11.7, CHAIN_MECHANISM_FORMULA_READ, chain_count=8, total_score=29670,
    ))
    tracker.finish()
    diag = tracker.diagnostics()
    assert diag.d7.chain_id_count == 2
    assert diag.d7.raw_generation_total == pytest.approx(308.0 + 115.0), (
        f"前半確定分が二重計上されている (実測 {diag.d7.raw_generation_total})"
    )
