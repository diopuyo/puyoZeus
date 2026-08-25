"""交換エピソード会計の不変条件テスト (2026-08-24、Gate 2)。

仕様: `docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md` の §8 (不変条件) と §14 (テスト一覧)。

**実装より先に書いている。** 後から辻褄を合わせないため。

## なぜこの会計が要るのか

user 指摘「99% の勝率から 1% に急降下する」の根因は、
**一つの撃ち合いを一つの出来事として数えていない**こと。実測 (seg01 game2):

    t=176.30  1P が 525 個ぶん生成
    t=186.67  2P が 720 個で撃ち返す。**同時に 1P の 525 が計算から消える**
    t=197.53  会計が 11.5 秒遅れで追いつく。しかも cap 216 で丸められる
    t=211.43  断片化で生成量が 442 → 101。画面では何も起きていないのに符号反転

ここで固定する不変条件は、この 3 つの事故を**構造的に起こせなくする**ためのもの。
"""
from __future__ import annotations

import itertools

import pytest

from src.exchange_ledger import (
    FINALIZE_SOURCE_SCORE_OCR_DIFF,
    OJAMA_MAX_DROP_PER_TURN,
    ChainRecord,
    EpisodeStatus,
    EventKind,
    ExchangeEvent,
    PhysicalContext,
    Side,
)
from src.exchange_ledger import ExchangeLedger


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------

def _fire(side: Side, t: float, chain_id: int, amount: float = 0.0):
    return ExchangeEvent(
        kind=EventKind.FIRE, side=side, t_sec=t, chain_id=chain_id,
        amount=amount, source="formula_read",
    )


def _step(side: Side, t: float, chain_id: int, amount: float, cc: int = 1):
    return ExchangeEvent(
        kind=EventKind.STEP, side=side, t_sec=t, chain_id=chain_id,
        amount=amount, chain_count=cc, source="formula_read",
    )


def _finalize(side: Side, t: float, chain_id: int, amount: float):
    # 2026-08-25 是正 (I16): FINALIZE の値供給源は `FINALIZE_SOURCE_SCORE_OCR_DIFF`
    # に限定される。既存ヘルパーは旧文字列 "score_ocr" を使っていたため、
    # I16 導入後は台帳が黙って拒否していた (=このヘルパーを使う既存テストの
    # 半数弱で FINALIZE が反映されなくなる)。テスト用ヘルパーが実際に
    # 意図する「score OCR 確定差分」を正しく名乗るよう修正する。
    return ExchangeEvent(
        kind=EventKind.FINALIZE, side=side, t_sec=t, chain_id=chain_id,
        amount=amount, source=FINALIZE_SOURCE_SCORE_OCR_DIFF,
    )


def _cancel(side: Side, t: float, chain_id: int, amount: float):
    return ExchangeEvent(
        kind=EventKind.CANCEL, side=side, t_sec=t, chain_id=chain_id,
        amount=amount, source="ledger",
    )


def _land(side: Side, t: float, chain_id: int, amount: float):
    """side = **受けた側**。着弾は受け側のフィールドで観測する。"""
    return ExchangeEvent(
        kind=EventKind.LAND, side=side, t_sec=t, chain_id=chain_id,
        amount=amount, source="drain",
    )


def _tsumo(side: Side, t: float):
    return ExchangeEvent(
        kind=EventKind.TSUMO_PLACED, side=side, t_sec=t, source="drain",
    )


def _ctx(**kw) -> PhysicalContext:
    return PhysicalContext(**kw)


def _push_all(ledger: "ExchangeLedger", events, ctx: PhysicalContext | None = None):
    ctx = ctx or _ctx()
    for e in events:
        ledger.push(e, ctx)


# ===========================================================================
# I1: 保存則
# ===========================================================================

def test_i1_conservation_one_sided() -> None:
    """§9.2 片側だけが連鎖する。生成 60 = 着弾 30 + 30。

    現行 `ResolvedExchangeTracker` は「両側同時に chain_event がある」ことを
    要求するため、この場面で**一度も起動しない**
    (`scripts/visualize_advantage_overlay.py:1932-1934`)。
    本会計は片側でも episode を開く。
    """
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 10.0, 1),
        _step(Side.P1, 10.0, 1, 60.0, cc=5),
        _finalize(Side.P1, 16.0, 1, 60.0),
        _tsumo(Side.P2, 16.5),
        _land(Side.P2, 16.6, 1, 30.0),
        _tsumo(Side.P2, 17.3),
        _land(Side.P2, 17.4, 1, 30.0),
    ])
    snap = led.snapshot()
    assert snap.total_generated == pytest.approx(60.0)
    assert snap.total_landed == pytest.approx(60.0)
    assert snap.total_canceled == pytest.approx(0.0)
    assert snap.unreconciled == pytest.approx(0.0)
    assert snap.net_raw == pytest.approx(0.0)


def test_i1_conservation_with_cancel() -> None:
    """§9.3 相殺しきれず差分だけが降る。生成 140 = 相殺 80 + 着弾 60。

    **2P が生成した 40 個は 1 個も 1P へ届かない。**
    相殺で消えた分は CANCELED であって LANDED ではない
    (user 伝授 `reference_ojama_forecast_landing_spec_2026-08-21`:
     「相殺により予告お邪魔が無くなれば降りません」)。
    """
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 30.0, 1), _step(Side.P1, 30.0, 1, 100.0, cc=6),
        _fire(Side.P2, 33.0, 2), _step(Side.P2, 33.0, 2, 40.0, cc=3),
        _cancel(Side.P1, 35.0, 1, 40.0),
        _cancel(Side.P2, 35.0, 2, 40.0),
        _tsumo(Side.P2, 36.0), _land(Side.P2, 36.1, 1, 30.0),
        _tsumo(Side.P2, 37.0), _land(Side.P2, 37.1, 1, 30.0),
    ])
    snap = led.snapshot()
    assert snap.total_generated == pytest.approx(140.0)
    assert snap.total_canceled == pytest.approx(80.0)
    assert snap.total_landed == pytest.approx(60.0)
    assert snap.unreconciled == pytest.approx(0.0)


def test_i13_all_clear_bonus_keeps_conservation() -> None:
    """全消しボーナス (+2100 点 ≈ 30 個) を含む確定で置換しても保存則が成立する。

    掛け算式の暫定和は純連鎖得点しか含まないので、確定は暫定より大きくなるのが
    正常。**置換規約でなければここが破れる** (加算だと二重計上、
    max だと相殺・着弾の実測合計と永久に合わない)。
    """
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 70.0, 1), _step(Side.P1, 70.0, 1, 500.0, cc=8),
        _finalize(Side.P1, 78.0, 1, 530.0),
    ])
    snap = led.snapshot()
    assert snap.total_generated == pytest.approx(530.0)
    assert snap.net_raw == pytest.approx(530.0)


# ===========================================================================
# 超過決済 (over-settlement、2026-08-24 追加)
# ===========================================================================

def test_oversettled_zero_for_normal_settlement() -> None:
    """正常な供給 (相殺+着弾が生成量ちょうど) では oversettled は 0。"""
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 10.0, 1), _step(Side.P1, 10.0, 1, 60.0, cc=5),
        _finalize(Side.P1, 16.0, 1, 60.0),
        _tsumo(Side.P2, 16.5), _land(Side.P2, 16.6, 1, 30.0),
        _tsumo(Side.P2, 17.3), _land(Side.P2, 17.4, 1, 30.0),
    ])
    assert led.snapshot().oversettled_total == pytest.approx(0.0)


def test_oversettled_captures_excess_cancel_while_outstanding_clips_to_zero() -> None:
    """相殺を生成量より多く供給すると oversettled が超過分と一致し、
    outstanding はクリップされて 0 のままであること。

    2026-08-24 コーディネーター指摘: `ChainRecord.outstanding` の下限
    クリップ (`max(0.0, ...)`) により、超過決済は黙って消える。
    `oversettled` はその見落としを可視化するために追加した。
    """
    led = ExchangeLedger()
    ctx = _ctx()
    led.push(_fire(Side.P1, 10.0, 1), ctx)
    led.push(_step(Side.P1, 10.0, 1, 10.0, cc=1), ctx)
    led.push(_finalize(Side.P1, 10.5, 1, 10.0), ctx)
    # 生成 10 個に対し 15 個キャンセルする (二重供給等を想定した意図的な過大供給)。
    led.push(_cancel(Side.P1, 11.0, 1, 15.0), ctx)
    rec = led._chains[1]  # noqa: SLF001 -- ChainRecord 単体の値を直接確認する
    assert rec.outstanding == pytest.approx(0.0)
    assert rec.oversettled == pytest.approx(5.0)
    assert led.snapshot().oversettled_total == pytest.approx(5.0)


# ===========================================================================
# I2 / I10: 順序
# ===========================================================================

def test_i2_same_tsec_permutation_invariant() -> None:
    """同一時刻の同着イベントを**どの順で入れても** net_raw が一致する。

    フレーム内の並び順で結果が変わってはいけない。
    """
    base = [
        _step(Side.P1, 50.0, 1, 100.0, cc=4),
        _step(Side.P2, 50.0, 2, 40.0, cc=2),
        _cancel(Side.P1, 50.0, 1, 40.0),
        _cancel(Side.P2, 50.0, 2, 40.0),
    ]
    prologue = [_fire(Side.P1, 49.0, 1), _fire(Side.P2, 49.5, 2)]
    results = set()
    for perm in itertools.permutations(base):
        led = ExchangeLedger()
        _push_all(led, [*prologue, *perm])
        results.add(round(led.snapshot().net_raw, 6))
    assert len(results) == 1, f"順序で結果が変わった: {results}"


def test_i10_decreasing_tsec_raises_but_same_tsec_allowed() -> None:
    """時刻が**減る** push は例外。**同一時刻は許可**する。

    同一時刻を弾くと I2 (同着の順序不変) が成立しなくなるため、
    「単調増加」ではなく「非減少」が正しい。
    """
    led = ExchangeLedger()
    ctx = _ctx()
    led.push(_fire(Side.P1, 10.0, 1), ctx)
    led.push(_step(Side.P1, 10.0, 1, 40.0), ctx)  # 同一時刻は OK
    with pytest.raises(ValueError):
        led.push(_step(Side.P1, 9.9, 1, 40.0), ctx)


# ===========================================================================
# I3 / §4.1.1: 暫定値の置換
# ===========================================================================

def test_i3_finalize_is_idempotent() -> None:
    """同じ chain_id に finalize を何度呼んでも結果が変わらない。

    **加算していたら 517 + 517 = 1034 になる。** そうならないこと。
    """
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 10.0, 1), _step(Side.P1, 10.0, 1, 500.0, cc=9),
        _finalize(Side.P1, 20.0, 1, 517.0),
        _finalize(Side.P1, 21.0, 1, 517.0),
        _finalize(Side.P1, 22.0, 1, 517.0),
    ])
    assert led.snapshot().net_raw == pytest.approx(517.0)


def test_finalize_upward_replaces() -> None:
    """§9.6(c) 確定 ≥ 暫定は無条件で置換する (正常系)。"""
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 70.0, 1), _step(Side.P1, 70.0, 1, 500.0, cc=8),
        _finalize(Side.P1, 78.0, 1, 530.0),
    ])
    assert led.snapshot().net_raw == pytest.approx(530.0)


def test_finalize_small_downward_replaces() -> None:
    """§9.6(a) 小さい下げは置換する (落下ボーナス等で説明できる差)。

    「小さい」の基準は物理から来る。確定スコア差分は落下ボーナス
    (最大 約 250 点 = おじゃま約 3.6 個) を含みうるので、その幅までは
    正常な差として置換する。

    **仕様初版はここに「500 → 420 (差 80)」と書いていたが誤りだった。**
    80 はおじゃま個数で、点数に直すと 5,600 点。落下ボーナスでは説明できない。
    単位を取り違えた例示だった (2026-08-24 訂正)。
    """
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 70.0, 1), _step(Side.P1, 70.0, 1, 500.0, cc=8),
        _finalize(Side.P1, 78.0, 1, 497.0),
    ])
    snap = led.snapshot()
    assert snap.net_raw == pytest.approx(497.0)
    assert snap.finalize_gate_held is False


def test_finalize_large_downward_is_held_and_counted() -> None:
    """§9.6(b) 大きい下げは置換せず保留し、差を unreconciled に計上する。

    確定側が桁誤読 (W2) で極端に小さい場合に、画面実測の暫定値を
    黙って捨てないため。**黙って下げるのも黙って max で固定するのも危険**なので、
    保留してカウンタに出す。
    """
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 70.0, 1), _step(Side.P1, 70.0, 1, 500.0, cc=8),
        _finalize(Side.P1, 78.0, 1, 42.0),
    ])
    snap = led.snapshot()
    assert snap.net_raw == pytest.approx(500.0), "暫定値が下げ置換で潰された"
    assert snap.finalize_gate_held is True
    assert snap.unreconciled > 0.0


# ===========================================================================
# I4: 重複排除
# ===========================================================================

def test_i4_duplicate_push_counts_once() -> None:
    """同一キー (kind, side, chain_id, t_sec) を 2 回 push しても 1 回分。"""
    led = ExchangeLedger()
    ctx = _ctx()
    led.push(_fire(Side.P1, 10.0, 1), ctx)
    ev = _step(Side.P1, 10.0, 1, 60.0, cc=5)
    led.push(ev, ctx)
    led.push(ev, ctx)
    assert led.snapshot().net_raw == pytest.approx(60.0)


def test_i4_duplicate_land_with_same_seq_counts_once() -> None:
    """同一 (kind, side, chain_id, t_sec, seq) の LAND は 1 回分。

    2026-08-24 追加: `seq` (同一時刻・同一 chain の正当な複数イベントを
    区別する連番) は本物の重複計上を止める機構であり、同一 seq の重複は
    従来どおり 1 件に潰れることを固定する。
    """
    led = ExchangeLedger()
    ctx = _ctx()
    led.push(_fire(Side.P1, 10.0, 1), ctx)
    led.push(_step(Side.P1, 10.0, 1, 60.0, cc=5), ctx)
    ev = ExchangeEvent(
        kind=EventKind.LAND, side=Side.P2, t_sec=16.6, chain_id=1, amount=30.0, seq=0,
    )
    led.push(ev, ctx)
    led.push(ev, ctx)  # 同一 seq の重複 (断片化等による二重供給を模擬)
    assert led.snapshot().total_landed == pytest.approx(30.0)


def test_i4_different_seq_is_not_a_duplicate() -> None:
    """seq が異なれば同一 (kind, side, chain_id, t_sec) でも別イベントとして数える。

    LAND の分割 (1 ターン上限超過時) は同一時刻・同一 chain に複数件
    積まれる正当なケースであり、`seq` で区別されるので合算される。
    """
    led = ExchangeLedger()
    ctx = _ctx()
    led.push(_fire(Side.P1, 10.0, 1), ctx)
    led.push(_step(Side.P1, 10.0, 1, 70.0, cc=5), ctx)
    led.push(ExchangeEvent(
        kind=EventKind.LAND, side=Side.P2, t_sec=16.6, chain_id=1, amount=30.0, seq=0,
    ), ctx)
    led.push(ExchangeEvent(
        kind=EventKind.LAND, side=Side.P2, t_sec=16.6, chain_id=1, amount=30.0, seq=1,
    ), ctx)
    led.push(ExchangeEvent(
        kind=EventKind.LAND, side=Side.P2, t_sec=16.6, chain_id=1, amount=10.0, seq=2,
    ), ctx)
    assert led.snapshot().total_landed == pytest.approx(70.0)


# ===========================================================================
# open_chain_ids (2026-08-24 追加)
# ===========================================================================

def test_open_chain_ids_returns_fifo_order() -> None:
    """open_chain_ids は side の未決着 chain_id を開いた順 (FIFO) で返す。"""
    led = ExchangeLedger()
    ctx = _ctx()
    led.push(_fire(Side.P2, 10.0, 1), ctx)
    led.push(_step(Side.P2, 10.0, 1, 60.0, cc=5), ctx)
    led.push(_fire(Side.P2, 20.0, 2), ctx)
    led.push(_step(Side.P2, 20.0, 2, 60.0, cc=5), ctx)
    assert led.open_chain_ids(Side.P2) == [1, 2]
    assert led.open_chain_ids(Side.P1) == []


def test_open_chain_ids_excludes_settled_chains() -> None:
    """outstanding が 0 になった chain は open_chain_ids に含まれない。"""
    led = ExchangeLedger()
    ctx = _ctx()
    led.push(_fire(Side.P2, 10.0, 1), ctx)
    led.push(_step(Side.P2, 10.0, 1, 60.0, cc=5), ctx)
    led.push(_fire(Side.P2, 20.0, 2), ctx)
    led.push(_step(Side.P2, 20.0, 2, 60.0, cc=5), ctx)
    led.push(_cancel(Side.P2, 25.0, 1, 60.0), ctx)  # chain 1 を完全相殺
    assert led.open_chain_ids(Side.P2) == [2]


# ===========================================================================
# I6: cap の非汚染
# ===========================================================================

def test_i6_net_raw_is_not_capped_but_display_is() -> None:
    """§6 判定は cap 前、表示だけ cap を通す。

    実測値で確認する。1P 525 / 2P 720 のとき純残量は −195。
    **cap 216 を通した値どうしを引き算すると架空の攻撃が生まれる**
    (project_pm100_display_flip_2026-08-24 の根因②)。
    """
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 176.3, 1), _step(Side.P1, 176.3, 1, 525.0, cc=9),
        _fire(Side.P2, 186.7, 2), _step(Side.P2, 186.7, 2, 720.0, cc=11),
    ])
    snap = led.snapshot()
    # 真の残量は -195。**先に cap してから引くと -195 にならない。**
    # 525 も 720 も cap 216 に丸められるので 216 - 216 = 0 になり、
    # 「撃ち合いは互角」という嘘の結論が出る。これが根因②。
    assert snap.net_raw == pytest.approx(-195.0)
    assert abs(snap.net_display) <= 216.0

    # cap が実際に効く場面 (片側だけが 525 送っている) では表示だけが丸まる
    led2 = ExchangeLedger()
    _push_all(led2, [
        _fire(Side.P1, 176.3, 1), _step(Side.P1, 176.3, 1, 525.0, cc=9),
    ])
    s2 = led2.snapshot()
    assert s2.net_raw == pytest.approx(525.0), "判定側が cap を通してしまっている"
    assert s2.net_display == pytest.approx(216.0)


# ===========================================================================
# I9: 符号
# ===========================================================================

def test_i9_side_swap_flips_sign_only() -> None:
    """1P/2P を入れ替えたイベント列で net_raw の符号だけが反転する。

    側の取り違えは過去に事故を起こしている
    (project_game_idx_desync_bug_2026-07-29)。
    """
    def run(a: Side, b: Side) -> float:
        led = ExchangeLedger()
        _push_all(led, [
            _fire(a, 10.0, 1), _step(a, 10.0, 1, 100.0, cc=5),
            _fire(b, 12.0, 2), _step(b, 12.0, 2, 40.0, cc=3),
        ])
        return led.snapshot().net_raw

    assert run(Side.P1, Side.P2) == pytest.approx(-run(Side.P2, Side.P1))


# ===========================================================================
# I12: 量の健全性
# ===========================================================================

def test_i12_land_per_event_cannot_exceed_turn_cap() -> None:
    """1 回の着弾が 1 ターン上限 (30 個) を超えたら例外。

    実ゲームでは 1 手で 30 個までしか降らない。超える値が来たら
    着弾の一括計上バグなので、黙って受け取らない。
    """
    led = ExchangeLedger()
    ctx = _ctx()
    led.push(_fire(Side.P1, 10.0, 1), ctx)
    led.push(_step(Side.P1, 10.0, 1, 100.0, cc=5), ctx)
    with pytest.raises(ValueError):
        led.push(_land(Side.P2, 12.0, 1, OJAMA_MAX_DROP_PER_TURN + 1), ctx)


def test_i12_negative_amount_raises() -> None:
    """負の量は受け取らない。"""
    led = ExchangeLedger()
    with pytest.raises(ValueError):
        led.push(_step(Side.P1, 10.0, 1, -5.0), _ctx())


# ===========================================================================
# episode の開始・参加・終了
# ===========================================================================

def test_episode_opens_on_single_side_fire() -> None:
    """片側発火だけで episode が開く (現行の両側同時要求の欠陥の回帰防止)。"""
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 10.0, 1), _step(Side.P1, 10.0, 1, 60.0, cc=5),
    ])
    assert led.current_episode() is not None


def test_symmetric_fire_without_cancel_does_not_close_episode() -> None:
    """【2026-08-24 追加】E1 (net_raw==0 で閉じる) 削除の回帰テスト。

    fable アーキ裁定で判明した仕様欠陥: 初版の終了条件 E1
    (`net_raw() == 0`) は「相殺が起きる運命にある」ことを示すだけで、
    相殺という取引の**記録ではなかった**。相殺は
    `cancel_own_pending_then_send_surplus` (`src/ojama_accounting.py:742`)
    が計算する実イベント (`CANCEL`) であり、それが供給されないまま
    両側が同額発火しただけで `net_raw()` が偶然 0 になっても、
    それは「撃ち合いが会計に載らないまま決着扱いになる」ことに他ならない
    (`project_pm100_display_flip_2026-08-24` の根因そのもの)。

    10 個 vs 10 個の対称な撃ち合いで CANCEL を一切供給しなければ、
    `net_raw()` は 0 になるが episode は OPEN のままであることを固定する。
    """
    led = ExchangeLedger()
    ctx = _ctx()
    _push_all(led, [
        _fire(Side.P1, 10.0, 1), _step(Side.P1, 10.0, 1, 10.0, cc=1),
        _fire(Side.P2, 12.0, 2), _step(Side.P2, 12.0, 2, 10.0, cc=1),
    ], ctx)
    snap = led.snapshot(ctx)
    assert snap.net_raw == pytest.approx(0.0), "対称発火で net_raw は 0 になるはず"
    assert led.current_episode() is not None, "CANCEL が無いのに閉じてしまった (E1 の再発)"
    assert snap.status is EpisodeStatus.OPEN, "CLOSED になってはいけない (E1 の再発)"


def test_seichi_does_not_open_episode() -> None:
    """§9.4 整地 (送りおじゃま ≤ 4 個) では episode を開かない。

    序盤の掘り作業まで「未解決の撃ち合い」にすると、
    hard override が永久に禁止されてしまう。
    """
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 50.0, 1), _step(Side.P1, 50.0, 1, 3.0, cc=1),
    ])
    assert led.current_episode() is None
    assert led.snapshot().is_unresolved is False


def test_seichi_lazy_open_when_amount_grows_past_threshold() -> None:
    """§9.4.1 暫定生成量が 4 個を超えた瞬間に開く (lazy open)。

    発火の瞬間には総送付量が未知なので「整地だから開かない」を
    その場では判定できない (因果律)。量が育った時点で開く。
    """
    led = ExchangeLedger()
    ctx = _ctx()
    led.push(_fire(Side.P1, 50.0, 1), ctx)
    led.push(_step(Side.P1, 50.0, 1, 3.0, cc=1), ctx)
    assert led.current_episode() is None
    led.push(_step(Side.P1, 51.4, 1, 40.0, cc=2), ctx)
    assert led.current_episode() is not None


def test_episode_participation_across_nine_second_gap() -> None:
    """§9.1 未解決残量があれば 9.2 秒あいた応射も同じ episode に入る。

    **時間窓では絶対に繋げられない隔たり。** 固定 0.96 秒の
    ヒステリシス (現行案 B) が全域で悪化した直接の理由がこれ。
    """
    led = ExchangeLedger()
    ctx = _ctx()
    led.push(_fire(Side.P1, 176.3, 1), ctx)
    led.push(_step(Side.P1, 176.3, 1, 525.0, cc=9), ctx)
    first = led.current_episode()
    led.push(_fire(Side.P2, 186.7, 2), ctx)
    led.push(_step(Side.P2, 186.7, 2, 720.0, cc=11), ctx)
    assert led.current_episode() is first, "別 episode に分かれた"


def test_i7_closed_requires_zero_residual() -> None:
    """CLOSED は残差 0 が必須。CLOSED_FORCED は除外する。"""
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 10.0, 1), _step(Side.P1, 10.0, 1, 60.0, cc=5),
        _finalize(Side.P1, 16.0, 1, 60.0),
        _tsumo(Side.P2, 16.5), _land(Side.P2, 16.6, 1, 30.0),
        _tsumo(Side.P2, 17.3), _land(Side.P2, 17.4, 1, 30.0),
    ])
    snap = led.snapshot()
    if snap.status is EpisodeStatus.CLOSED:
        assert snap.provisional_residual == pytest.approx(0.0)
        assert snap.unreconciled == pytest.approx(0.0)


def test_no_two_open_episodes_at_once() -> None:
    """§2.4.1 OPEN な episode は高々 1 つ。"""
    led = ExchangeLedger()
    ctx = _ctx()
    led.push(_fire(Side.P1, 10.0, 1), ctx)
    led.push(_step(Side.P1, 10.0, 1, 60.0, cc=5), ctx)
    first = led.current_episode()
    # 残量が解消してから、参加条件を満たさない新しい発火
    led.push(_finalize(Side.P1, 16.0, 1, 60.0), ctx)
    led.push(_tsumo(Side.P2, 16.5), ctx)
    led.push(_land(Side.P2, 16.6, 1, 30.0), ctx)
    led.push(_tsumo(Side.P2, 17.3), ctx)
    led.push(_land(Side.P2, 17.4, 1, 30.0), ctx)
    led.push(_fire(Side.P2, 40.0, 9), ctx)
    led.push(_step(Side.P2, 40.0, 9, 60.0, cc=5), ctx)
    assert led.current_episode() is not first
    assert led.open_episode_count() == 1


def test_i11_match_boundary_forces_close() -> None:
    """§I11 試合を跨いだら強制終了し、残量を unreconciled として記録する。"""
    led = ExchangeLedger()
    ctx0 = _ctx(game_idx=0)
    led.push(_fire(Side.P1, 10.0, 1), ctx0)
    led.push(_step(Side.P1, 10.0, 1, 100.0, cc=6), ctx0)
    led.push(_tsumo(Side.P2, 12.0), _ctx(game_idx=1))
    snap = led.snapshot()
    assert snap.forced_close_count >= 1


def test_episode_max_sec_forces_close_and_counts() -> None:
    """§2.5 上限秒数で強制終了し、件数をカウンタに出す (黙って切らない)。"""
    led = ExchangeLedger()
    ctx = _ctx()
    led.push(_fire(Side.P1, 0.0, 1), ctx)
    led.push(_step(Side.P1, 0.0, 1, 100.0, cc=6), ctx)
    led.push(_tsumo(Side.P2, 200.0), ctx)
    assert led.snapshot().forced_close_count >= 1


# ===========================================================================
# I15 / I8 / §7.4: 未解決ゲートと早期解除
# ===========================================================================

def test_i15_no_episode_means_resolved() -> None:
    """OPEN な episode が無ければ未解決ではない。"""
    assert ExchangeLedger().snapshot().is_unresolved is False


def test_is_unresolved_true_while_open() -> None:
    """撃ち合いが続いている間は未解決。"""
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 10.0, 1), _step(Side.P1, 10.0, 1, 100.0, cc=6),
    ])
    assert led.snapshot().is_unresolved is True


def test_hard_override_forbidden_while_outcome_depends_on_provisional() -> None:
    """§7.4 どちらに転ぶか分からないうちは ±100 の完全上書きを許さない。

    seg01 game2 の t=177〜211 がまさにこれ。向きは出してよいが
    99%/1% と断定してはいけない場面だった。
    """
    led = ExchangeLedger()
    ctx = _ctx(p1_chaining=True, p1_room=65, p2_room=40)
    _push_all(led, [
        _fire(Side.P1, 176.3, 1), _step(Side.P1, 176.3, 1, 525.0, cc=9),
    ], ctx)
    assert led.snapshot(ctx).allows_hard_override is False


def test_hard_override_allowed_when_receiver_dies_regardless() -> None:
    """§7.4 未確定量を受け側に最も有利に倒しても死ぬなら断定を許す。

    「未解決中は断定しない」と「真の致死を弱めない」を両立させる要。
    確率推定 (W15、応手確率は実際に成功した応手を 25〜40% としか見積もれない)
    を使わないので、構造的に両立する。
    """
    led = ExchangeLedger()
    # 2P は実際に死んでいる (STABLE 確定盤面で成立)
    ctx = _ctx(p2_dead=True, p2_room=0)
    _push_all(led, [
        _fire(Side.P1, 176.3, 1), _step(Side.P1, 176.3, 1, 525.0, cc=9),
    ], ctx)
    assert led.snapshot(ctx).allows_hard_override is True


# ===========================================================================
# 性質テスト
# ===========================================================================

@pytest.mark.parametrize("seed_amounts", [
    (60.0, 0.0), (100.0, 40.0), (525.0, 720.0), (12.0, 12.0), (200.0, 5.0),
])
def test_property_net_raw_matches_generated_minus_settled(seed_amounts) -> None:
    """net_raw が「生成 − 相殺 − 着弾」の符号付き合計と一致する。

    どの組み合わせでも会計の等式が崩れないこと。
    """
    a1, a2 = seed_amounts
    led = ExchangeLedger()
    ctx = _ctx()
    events = [_fire(Side.P1, 10.0, 1), _step(Side.P1, 10.0, 1, a1, cc=5)]
    if a2 > 0:
        events += [_fire(Side.P2, 12.0, 2), _step(Side.P2, 12.0, 2, a2, cc=4)]
    _push_all(led, events, ctx)
    snap = led.snapshot(ctx)
    expected = a1 - a2
    assert snap.net_raw == pytest.approx(expected)


def test_turn_cap_constant_matches_scoring_module() -> None:
    """1 ターン着弾上限が `src/scoring.py` と一致していること。

    純粋コアの依存を増やさないため定数で持っているので、
    ずれていないかをここで固定する。
    """
    from src.scoring import OJAMA_MAX_DROP_PER_TURN as SCORING_CAP

    assert OJAMA_MAX_DROP_PER_TURN == SCORING_CAP


# ===========================================================================
# 試合境界での chain 退役 (2026-08-25 追加、回帰本体)
#
# 実測 `data/verify/gate3_breakdown_2026-08-25/episode_dedup.json`:
# 5 episode が連続で match_boundary 強制終了し、total_generated が
# 5,025 → 8,965 → 9,771 → 9,825 → 11,450 と単調増加していた
# (= 前 episode の未決着 chain が次 episode に繰り越されて再計上されていた)。
# 大域重複排除後の真の合計は 11,494 だが、episode 単純合計は 45,036
# (差 33,542 = 74.5% が二重計上)。以下は同じ機構を最小構成で再現する。
# ===========================================================================

def test_match_boundary_retired_unreconciled_does_not_double_count_episode_own_chain() -> None:
    """episode 自身の chain は `unreconciled` へ既に計上済みなので、
    `retired_unreconciled` へは二重に積まない (2026-08-25 是正)。

    修正前の初版は、force_close した episode 自身の chain の残量を
    `unreconciled` と `retired_unreconciled` の**両方**に積んでいた。
    2 つの数字は独立に加算できる値でなければならず (合算すると水増しに
    なる数字を並べてはいけない、コーディネーター指摘)、この場合は
    `retired_*` 側が 0 になるのが正しい。
    """
    led = ExchangeLedger()
    ctx0 = _ctx(game_idx=0)
    led.push(_fire(Side.P1, 10.0, 1), ctx0)
    led.push(_step(Side.P1, 10.0, 1, 100.0, cc=6), ctx0)
    led.push(_tsumo(Side.P2, 12.0), _ctx(game_idx=1))
    snap = led.snapshot()
    assert snap.unreconciled == pytest.approx(100.0), "episode 側の unreconciled は従来どおり"
    assert snap.retired_chain_count == 0, "episode 自身の chain を二重に退役計上した"
    assert snap.retired_unreconciled == pytest.approx(0.0), "unreconciled と二重計上した"


def test_match_boundary_retired_unreconciled_captures_chain_outside_any_episode() -> None:
    """episode を一度も開かなかった chain (整地等) の残量は、
    `unreconciled` には計上されない代わりに `retired_unreconciled` に積まれ、
    黙って消えない。

    整地 chain (2 個、SEICHI_OJAMA_MAX_COUNT=4 以下) は、その量だけでは
    episode を一度も開かせない (lazy open の対象は amount>4 の chain のみ)
    ので、`_force_close` の episode 単位の集計には現れない。
    """
    led = ExchangeLedger()
    ctx0 = _ctx(game_idx=0)
    led.push(_fire(Side.P1, 5.0, 9), ctx0)
    led.push(_step(Side.P1, 5.0, 9, 2.0, cc=1), ctx0)  # 整地: episode を開かない
    assert led.current_episode() is None, "前提: 整地だけでは episode を開かない"
    led.push(_tsumo(Side.P2, 12.0), _ctx(game_idx=1))
    snap = led.snapshot()
    assert snap.unreconciled == pytest.approx(0.0), "episode が無いので unreconciled は増えない"
    assert snap.retired_chain_count == 1
    assert snap.retired_unreconciled == pytest.approx(2.0)


def test_match_boundary_clears_open_chain_ids_from_previous_game() -> None:
    """試合境界のあと、前の試合の未決着 chain が open_chain_ids に残らない。"""
    led = ExchangeLedger()
    ctx0 = _ctx(game_idx=0)
    led.push(_fire(Side.P1, 10.0, 1), ctx0)
    led.push(_step(Side.P1, 10.0, 1, 100.0, cc=6), ctx0)
    assert led.open_chain_ids(Side.P1) == [1], "前提: 試合境界前は未決着"
    led.push(_tsumo(Side.P2, 12.0), _ctx(game_idx=1))
    assert led.open_chain_ids(Side.P1) == [], (
        "退役後も前の試合の chain_id が残っている (相殺・着弾が誤って"
        "前の試合の chain に帰属する経路)"
    )


def test_match_boundary_allows_normal_close_previously_blocked_forever() -> None:
    """試合境界のあと、前の試合の未決着 chain に永久にブロックされず
    次の試合の episode が正常 CLOSED できる。

    修正前は `_all_settled()` が台帳全体の `self._chains` を見るため、
    前の試合の chain が outstanding>0 のまま残っていると、以後**永久に**
    どの episode も正常 CLOSED できなくなっていた。
    """
    led = ExchangeLedger()
    ctx0 = _ctx(game_idx=0)
    led.push(_fire(Side.P1, 10.0, 1), ctx0)
    led.push(_step(Side.P1, 10.0, 1, 100.0, cc=6), ctx0)  # 試合 0: 未決着のまま放置

    ctx1 = _ctx(game_idx=1)
    led.push(_fire(Side.P1, 20.0, 2), ctx1)  # 試合 1 への遷移で試合 0 を退役させる
    led.push(_step(Side.P1, 20.0, 2, 60.0, cc=5), ctx1)
    led.push(_finalize(Side.P1, 26.0, 2, 60.0), ctx1)
    led.push(_tsumo(Side.P2, 26.5), ctx1)
    led.push(_land(Side.P2, 26.6, 2, 30.0), ctx1)
    led.push(_tsumo(Side.P2, 27.3), ctx1)
    led.push(_land(Side.P2, 27.4, 2, 30.0), ctx1)

    # episode を閉じた瞬間に `self._episode` は破棄される (`current_episode()`
    # は None を返す) ため、`closed_episodes()` の要約で判定する
    # (`test_i7_closed_requires_zero_residual` と同じ確認方法)。
    last = led.closed_episodes()[-1]
    assert last.status is EpisodeStatus.CLOSED, (
        "前試合の残骸に阻まれて正常 CLOSED できていない"
    )
    assert last.unreconciled == pytest.approx(0.0)


def test_match_boundary_series_generated_totals_do_not_leak_across_games() -> None:
    """5 試合連続で match_boundary 強制終了しても total_generated が
    試合を跨いで再計上されない (実測データの再現、回帰本体)。

    修正前はこの列を流すと、後の episode ほど前の episode の chain の
    amount を合成 FIRE 経由で余分に抱え込み、単純合計が真の合計より
    大きくなり続けた。
    """
    led = ExchangeLedger()
    amounts = [100.0, 150.0, 80.0, 200.0, 50.0]
    for game_idx, amount in enumerate(amounts):
        ctx = _ctx(game_idx=game_idx)
        led.push(_fire(Side.P1, 10.0 + game_idx, game_idx + 1), ctx)
        led.push(_step(Side.P1, 10.0 + game_idx, game_idx + 1, amount, cc=5), ctx)
    # 最後の試合境界を締めるための次試合ダミーイベント。
    led.push(_tsumo(Side.P2, 999.0), _ctx(game_idx=len(amounts)))

    forced = [
        s for s in led.closed_episodes() if s.close_reason == "match_boundary"
    ]
    assert len(forced) == len(amounts)
    for summary, amount in zip(forced, amounts):
        assert summary.total_generated == pytest.approx(amount), (
            "前試合からの繰越が混入した (単調増加バグの再発)"
        )
    assert sum(s.total_generated for s in forced) == pytest.approx(sum(amounts))
    assert led.snapshot().duplicate_generated_suppressed_count == 0


def test_generation_dedup_safety_net_prevents_double_count_after_max_sec_leak() -> None:
    """試合境界以外 (`max_sec` 強制終了) で chain が台帳に残り、直後の
    lazy open に紛れ込んでも、大域重複排除で total_generated が
    二重計上されない (§ 修正3の安全網が実際に機能することの確認)。

    `max_sec` 強制終了は試合境界ではないため chain を退役させない
    (同じ試合内での正当な繰越)。しかしその chain が次の lazy open に
    合成 FIRE として引き継がれると、素朴には二重計上が起こる。
    """
    led = ExchangeLedger()
    ctx0 = _ctx(game_idx=0)
    led.push(_fire(Side.P1, 0.0, 1), ctx0)
    led.push(_step(Side.P1, 0.0, 1, 50.0, cc=5), ctx0)
    # EPISODE_MAX_SEC (60.0 秒) を超えて同じ試合内で max_sec 強制終了させる。
    led.push(_tsumo(Side.P2, 61.0), ctx0)
    # 直後の lazy open が同じ chain_id=1 を拾い、別 episode として開いてしまう。
    # そのまま試合境界で締めて要約を確定させる。
    led.push(_tsumo(Side.P2, 62.0), _ctx(game_idx=1))

    summaries = led.closed_episodes()
    assert [s.close_reason for s in summaries] == ["max_sec", "match_boundary"]
    assert summaries[0].total_generated == pytest.approx(50.0)
    assert summaries[1].total_generated == pytest.approx(0.0), (
        "同じ chain_id が 2 episode 目でも計上され二重計上した"
    )
    assert sum(s.total_generated for s in summaries) == pytest.approx(50.0)

    snap = led.snapshot()
    assert snap.duplicate_generated_suppressed_count == 1
    assert snap.duplicate_generated_suppressed_amount == pytest.approx(50.0)


# ===========================================================================
# I16: FINALIZE の値供給源の限定 (2026-08-25 追加、fable アーキ裁定)
#
# `mechanism='baseline'` (ChainSimulator 産の推定) を FINALIZE の値として
# 使っていたのは配線の間違いだった。値の権威は score OCR 確定差分
# (`source=FINALIZE_SOURCE_SCORE_OCR_DIFF`) だけに限定する。
# ===========================================================================

def _finalize_with_source(
    side: Side, t: float, chain_id: int, amount: float, source: str,
) -> ExchangeEvent:
    """`source` を明示指定できる FINALIZE イベント (I16 のゲート挙動を試すため)。"""
    return ExchangeEvent(
        kind=EventKind.FINALIZE, side=side, t_sec=t, chain_id=chain_id,
        amount=amount, source=source,
    )


def test_i16_rejects_finalize_with_unauthoritative_source_by_default() -> None:
    """既定 (`allow_simulate_fallback=False`) では、score OCR 以外を出所とする
    FINALIZE を黙って拒否し、暫定値のまま残る。例外は投げない。"""
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 10.0, 1), _step(Side.P1, 10.0, 1, 500.0, cc=8),
        _finalize_with_source(Side.P1, 20.0, 1, 999999.0, source="simulate_fallback"),
    ])
    snap = led.snapshot()
    assert snap.net_raw == pytest.approx(500.0), "推定値が確定として紛れ込んだ"
    assert snap.finalize_rejected_count == 1
    assert snap.finalize_rejected_amount == pytest.approx(999999.0)


def test_i16_allow_simulate_fallback_true_accepts_it() -> None:
    """`allow_simulate_fallback=True` を明示すれば、低信頼フォールバックの
    FINALIZE も受け入れる (既定 OFF の optional 引数)。"""
    led = ExchangeLedger(allow_simulate_fallback=True)
    _push_all(led, [
        _fire(Side.P1, 10.0, 1), _step(Side.P1, 10.0, 1, 500.0, cc=8),
        # 上げ置換 (§9.6(c)) を使う。下げ幅は別ゲート
        # (FINALIZE_DOWNWARD_TOLERANCE) の対象になり I16 の検証が濁るため避ける。
        _finalize_with_source(Side.P1, 20.0, 1, 520.0, source="simulate_fallback"),
    ])
    snap = led.snapshot()
    assert snap.net_raw == pytest.approx(520.0)
    assert snap.finalize_rejected_count == 0


def test_i16_simulate_disaster_745_does_not_enter_accounting() -> None:
    """【回帰本体】実測の事故: 暫定 (score OCR 確定差分) 38 個相当に対し、
    推定 (ChainSimulator 由来) が 745 個相当という壊滅的に間違った値で
    FINALIZE として届いても、745 が会計 (`net_raw`) に入らないこと。
    """
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 0.0, 1), _step(Side.P1, 0.0, 1, 38.0, cc=1),
        _finalize_with_source(Side.P1, 1.0, 1, 745.0, source="simulate_fallback"),
    ])
    snap = led.snapshot()
    assert snap.net_raw == pytest.approx(38.0)
    assert snap.net_raw != pytest.approx(745.0)
    assert snap.finalize_rejected_count == 1
    assert snap.finalize_rejected_amount == pytest.approx(745.0)


def test_i16_does_not_raise_on_rejection() -> None:
    """低信頼フォールバックの FINALIZE を拒否しても例外を投げない。"""
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 10.0, 1), _step(Side.P1, 10.0, 1, 50.0, cc=3),
    ])
    # 例外が飛べばこの push 自体でテストが FAIL するので、通過すること自体が確認。
    led.push(
        _finalize_with_source(Side.P1, 12.0, 1, 12345.0, source="simulate_fallback"),
        _ctx(),
    )


def test_finalize_source_constant_matches_chain_id_resolver() -> None:
    """`FINALIZE_SOURCE_SCORE_OCR_DIFF` が `src/chain_id_resolver.py` の
    `FINALIZED_SOURCE_SCORE_OCR_DIFF` と文字列として一致していること。

    2 つの純粋コアが互いに import しない設計 (依存を増やさない方針) のため
    定数が重複しているので、ずれていないかをここで固定する
    (`test_turn_cap_constant_matches_scoring_module` と同じパターン)。
    """
    from src.chain_id_resolver import FINALIZED_SOURCE_SCORE_OCR_DIFF

    assert FINALIZE_SOURCE_SCORE_OCR_DIFF == FINALIZED_SOURCE_SCORE_OCR_DIFF


# ===========================================================================
# Fix【4】: `_all_settled`/`_provisional_residual` の episode 限定
# (2026-08-25 追加、gate3_episode_v3 実装タスク)
# ===========================================================================

def test_all_settled_and_provisional_residual_use_chains_for_episode() -> None:
    """`_all_settled`/`_provisional_residual` が `_chains_for_episode(ep)` に
    限定されていること (`_summarize_episode`/`_force_close` と同じ流儀) を、
    このepisode に一度も touch されていない chain を使って確認する。

    **報告 (2026-08-25 実装時の発見、重要)**: 通常の `push` 経路では
    この状況を作れない。`_apply` は OPEN な episode がある間に来た
    **あらゆる chain_id** のイベントを無条件で `Episode.touch` する
    (side/chain_id を問わない)。さらに episode の新規開始時
    (`_fire_events_of_open_chains`) も、その時点で outstanding>0 の
    **全 chain** を合成 FIRE として無条件で episode へ引き継ぐ。
    そのため、outstanding>0 の chain が「どの episode の events にも
    一度も現れない」状態は、通常の push 経路では実質的に発生しない。
    本テストは `ExchangeLedger._chains` へ直接 chain を注入すること
    でのみこの状況を再現できている (詳細は報告参照。この巻き込みの
    仕様自体が Fix【4】単体の実効果を制限する可能性がある、別の
    構造的な論点として報告済み)。
    """
    led = ExchangeLedger()
    ctx = _ctx(game_idx=0)
    led.push(_fire(Side.P2, 10.0, 2), ctx)
    led.push(_step(Side.P2, 10.0, 2, 50.0, cc=5), ctx)
    assert led.current_episode() is not None, "前提: episode が開いている"
    # このepisode の events には一度も現れない無関係な chain を直接注入する
    # (public API 経由では上記理由により再現できないための直接操作)。
    led._chains[999] = ChainRecord(  # noqa: SLF001
        chain_id=999, side=Side.P1, opened_at_sec=0.0, provisional_amount=2.0,
    )
    led.push(_cancel(Side.P2, 11.0, 2, 50.0), ctx)

    assert led.current_episode() is None, (
        "episode の events に一度も現れない chain に阻まれて閉じられていない"
    )
    closed = led.closed_episodes()
    assert len(closed) == 1
    assert closed[0].status is EpisodeStatus.CLOSED
    assert closed[0].unreconciled == pytest.approx(0.0)
    assert led.outstanding_of(999) == pytest.approx(2.0), (
        "無関係 chain 自体の残量は変えていない (episode の判定からだけ除外)"
    )


# ===========================================================================
# Fix【2】: `retire_side_chains` (ワイプの side 単位退役、2026-08-25 追加)
# ===========================================================================

def test_retire_side_chains_removes_only_chains_generated_by_the_other_side() -> None:
    """`retire_side_chains(Side.P2, ...)` は P1 が生成した未決着 chain だけを
    退役させ、P2 が生成した未決着 chain には触れない。"""
    led = ExchangeLedger()
    ctx = _ctx(game_idx=0)
    led.push(_fire(Side.P1, 0.0, 1), ctx)   # P2 に向かう分 (ワイプ対象)
    led.push(_step(Side.P1, 0.0, 1, 40.0, cc=4), ctx)
    led.push(_fire(Side.P2, 1.0, 2), ctx)   # P1 に向かう分 (ワイプ対象外)
    led.push(_step(Side.P2, 1.0, 2, 30.0, cc=3), ctx)

    led.retire_side_chains(Side.P2, 5.0, ctx)

    assert led.outstanding_of(1) == pytest.approx(0.0), "P1 生成分が退役していない"
    assert led.outstanding_of(2) == pytest.approx(30.0), "無関係な P2 生成分まで消えた"
    snap = led.snapshot()
    assert snap.retired_chain_count == 1
    assert snap.retired_unreconciled == pytest.approx(40.0)


def test_retire_side_chains_closes_episode_once_it_becomes_settled() -> None:
    """ワイプで残りの未決着分が消えた結果 episode が決着すれば、
    遅い試合境界を待たずにその場で CLOSED になる。"""
    led = ExchangeLedger()
    ctx = _ctx(game_idx=0)
    led.push(_fire(Side.P1, 0.0, 1), ctx)
    led.push(_step(Side.P1, 0.0, 1, 40.0, cc=4), ctx)
    assert led.current_episode() is not None, "前提: episode が開いている"

    led.retire_side_chains(Side.P2, 5.0, ctx)

    assert led.current_episode() is None, "ワイプ後も episode が開いたまま"
    closed = led.closed_episodes()
    assert len(closed) == 1
    assert closed[0].status is EpisodeStatus.CLOSED
    assert closed[0].close_reason == "side_wipe"


# ===========================================================================
# 実装1 (2026-08-25 追加): 測定器の是正
# `total_unreconciled` (closed episode だけの合算) は「まだ数えていない」を
# 「0=決着した」と誤読させる。`open_episode_outstanding`/
# `total_outstanding_all_chains` を別枠で出し、窓の切り方に依存しない
# 台帳の生値を必ず併記できるようにする。
# ===========================================================================

def test_open_episode_outstanding_reflects_only_open_episode_chains() -> None:
    """`open_episode_outstanding` は OPEN な episode に属する chain の
    outstanding だけを合算する。"""
    led = ExchangeLedger()
    ctx = _ctx(game_idx=0)
    led.push(_fire(Side.P1, 10.0, 1), ctx)
    led.push(_step(Side.P1, 10.0, 1, 60.0, cc=5), ctx)
    assert led.open_episode_outstanding() == pytest.approx(60.0)


def test_open_episode_outstanding_is_zero_when_no_episode_open() -> None:
    """OPEN な episode が無ければ 0.0 (窓を切っても架空の残量を作らない)。"""
    assert ExchangeLedger().open_episode_outstanding() == pytest.approx(0.0)


def test_total_outstanding_all_chains_includes_chain_absent_from_open_episode_events() -> None:
    """`total_outstanding_all_chains` は open episode の events に一度も
    現れない chain も含めて台帳全体の生値を返すが、`open_episode_outstanding`
    はその chain を含まない。

    **報告**: 通常の push 経路では、OPEN な episode がある間に来た
    あらゆる chain_id のイベントが無条件で `Episode.touch` されるため
    (`_apply` 参照)、「episode の events に一度も現れない outstanding>0 の
    chain」を通常経路で作ることはできない
    (`test_all_settled_and_provisional_residual_use_chains_for_episode` の
    報告と同じ制約)。そのため同じ手法 (`self._chains` への直接注入) で
    再現する。
    """
    led = ExchangeLedger()
    ctx = _ctx(game_idx=0)
    led.push(_fire(Side.P2, 10.0, 1), ctx)
    led.push(_step(Side.P2, 10.0, 1, 60.0, cc=5), ctx)
    assert led.current_episode() is not None
    led._chains[999] = ChainRecord(  # noqa: SLF001
        chain_id=999, side=Side.P1, opened_at_sec=0.0, provisional_amount=2.0,
    )
    assert led.total_outstanding_all_chains() == pytest.approx(62.0)
    assert led.open_episode_outstanding() == pytest.approx(60.0), "無関係 chain を含めてはいけない"


# ===========================================================================
# 実装2 (2026-08-25 追加): 退役した chain の相殺・着弾・生成量の転記
# `_chains_for_episode` の `if cid in self._chains` ガードは、chain が
# 削除された瞬間にその chain の相殺・着弾・生成量ごと episode の集計から
# 弾く (v51 chain6 の相殺 18 個消失、327 -> 309 の実測)。
# ===========================================================================

def test_retire_side_chains_captures_canceled_and_generated_before_deletion() -> None:
    """`retire_side_chains` で削除される chain の相殺・生成量が
    `retired_canceled`/`retired_generated` へ転記され、黙って消えないこと
    (実装2、chain6 の相殺 18 個消失の回帰本体、台帳単体)。"""
    led = ExchangeLedger()
    ctx = _ctx(game_idx=0)
    led.push(_fire(Side.P1, 0.0, 1), ctx)
    led.push(_step(Side.P1, 0.0, 1, 40.0, cc=4), ctx)
    led.push(_cancel(Side.P1, 1.0, 1, 15.0), ctx)  # 一部だけ相殺 (残 25)

    led.retire_side_chains(Side.P2, 2.0, ctx)

    snap = led.snapshot()
    assert snap.retired_unreconciled == pytest.approx(25.0)
    assert snap.retired_canceled == pytest.approx(15.0)
    assert snap.retired_generated == pytest.approx(40.0)
    assert snap.retired_landed == pytest.approx(0.0)


def test_match_boundary_retirement_captures_totals_of_chain_never_summarized() -> None:
    """episode を一度も開かなかった chain (整地等) が試合境界で退役するとき、
    その生成・相殺も `retired_generated`/`retired_canceled` に転記される
    こと (実装2)。"""
    led = ExchangeLedger()
    ctx0 = _ctx(game_idx=0)
    led.push(_fire(Side.P1, 5.0, 9), ctx0)
    led.push(_step(Side.P1, 5.0, 9, 2.0, cc=1), ctx0)  # 整地: episode を開かない
    led.push(_cancel(Side.P1, 6.0, 9, 1.0), ctx0)  # 一部だけ相殺 (残 1)
    assert led.current_episode() is None, "前提: 整地だけでは episode を開かない"

    led.push(_tsumo(Side.P2, 12.0), _ctx(game_idx=1))

    snap = led.snapshot()
    assert snap.retired_chain_count == 1
    assert snap.retired_unreconciled == pytest.approx(1.0)
    assert snap.retired_canceled == pytest.approx(1.0)
    assert snap.retired_generated == pytest.approx(2.0)


def test_match_boundary_retirement_does_not_double_count_episode_own_chain_totals() -> None:
    """試合境界で force_close した episode 自身の chain の生成・相殺量は、
    その episode の summary に既に計上済みなので `retired_generated`/
    `retired_canceled` へ二重に転記されないこと (実装2)。"""
    led = ExchangeLedger()
    ctx0 = _ctx(game_idx=0)
    led.push(_fire(Side.P1, 10.0, 1), ctx0)
    led.push(_step(Side.P1, 10.0, 1, 100.0, cc=6), ctx0)
    led.push(_cancel(Side.P1, 11.0, 1, 30.0), ctx0)
    led.push(_tsumo(Side.P2, 12.0), _ctx(game_idx=1))

    snap = led.snapshot()
    last = led.closed_episodes()[-1]
    assert last.total_generated == pytest.approx(100.0)
    assert last.total_canceled == pytest.approx(30.0)
    assert snap.retired_generated == pytest.approx(0.0), "既に summarize 済みの chain を二重計上した"
    assert snap.retired_canceled == pytest.approx(0.0)


# ===========================================================================
# コーディネーター指摘 (2026-08-25 追加): close 済み chain への相殺・着弾を
# 黙って落とさず数える (根治は未着手、可視化のみ)。
# ===========================================================================

def test_post_close_settlement_dropped_counts_cancel_arriving_after_max_sec_close() -> None:
    """max_sec で強制終了した chain に、close 後さらに CANCEL が届くと
    `post_close_settlement_dropped_count`/`_amount` に計上されること。

    CLOSED_FORCED (max_sec) は `retire_all_chains_at_match_boundary` を
    呼ばないため chain が `self._chains` に残り続け、close 後も届いた
    CANCEL/LAND が (どの episode 要約にも現れないまま) chain 自体には
    反映され続ける。この「黙って消える」経路を可視化するための回帰。
    """
    led = ExchangeLedger()
    ctx = _ctx(game_idx=0)
    led.push(_fire(Side.P1, 0.0, 1), ctx)
    led.push(_step(Side.P1, 0.0, 1, 100.0, cc=6), ctx)
    led.push(_tsumo(Side.P2, 61.0), ctx)  # EPISODE_MAX_SEC(60秒)超過でmax_sec強制終了
    closed = led.closed_episodes()
    assert closed[-1].status is EpisodeStatus.CLOSED_FORCED, "前提: max_sec で強制終了"
    assert closed[-1].total_canceled == pytest.approx(0.0), "前提: close時点で相殺なし"

    led.push(_cancel(Side.P1, 62.0, 1, 30.0), ctx)  # close 後にさらに CANCEL が届く

    snap = led.snapshot()
    assert snap.post_close_settlement_dropped_count == 1
    assert snap.post_close_settlement_dropped_amount == pytest.approx(30.0)


def test_post_close_settlement_dropped_is_zero_for_normal_flow() -> None:
    """通常の (close 前に相殺・着弾が完了する) 流れでは 0 のまま
    (母数を明示するため、正常系との対比を固定する)。"""
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 10.0, 1), _step(Side.P1, 10.0, 1, 60.0, cc=5),
        _finalize(Side.P1, 16.0, 1, 60.0),
        _tsumo(Side.P2, 16.5), _land(Side.P2, 16.6, 1, 30.0),
        _tsumo(Side.P2, 17.3), _land(Side.P2, 17.4, 1, 30.0),
    ])
    snap = led.snapshot()
    assert snap.post_close_settlement_dropped_count == 0
    assert snap.post_close_settlement_dropped_amount == pytest.approx(0.0)


def test_retire_side_chains_does_not_close_episode_while_still_chaining() -> None:
    """退役自体は起きても、`ctx` が連鎖中を示していれば episode は閉じない
    (`_should_close` の E2 判定「p1_chaining/p2_chaining なら閉じない」を
    `retire_side_chains` 経由でも尊重することの確認)。"""
    led = ExchangeLedger()
    ctx = _ctx(game_idx=0, p1_chaining=True)
    led.push(_fire(Side.P1, 0.0, 1), ctx)
    led.push(_step(Side.P1, 0.0, 1, 40.0, cc=4), ctx)

    led.retire_side_chains(Side.P2, 5.0, ctx)

    snap = led.snapshot()
    assert snap.retired_chain_count == 1, "退役自体は行われるべき"
    assert snap.retired_unreconciled == pytest.approx(40.0)
    assert led.current_episode() is not None, "chaining 中に誤って episode を閉じた"


# ===========================================================================
# P2-1 (Codex Gate 3-2b レビュー NG、2026-08-25): 下方 FINALIZE の冪等性
#
# 既存の `test_i3_finalize_is_idempotent` は「同値 517 を 3 回」しか
# 検査しておらず、下げ置換ゲート (§4.1.1) に保留された差分
# (`held_divergence`) の再入力・値更新・後続正常確定を覆っていなかった。
# Codex 最小再現: provisional=500 へ confirmed=42 を異なる時刻で 2 回
# 送ると、正しい保留差 458 ではなく unreconciled=916 になる。
# ===========================================================================

def test_p2_1_downward_finalize_reentry_at_different_times_is_idempotent() -> None:
    """【Codex 最小再現】下方 FINALIZE (42 < 500-4) を**異なる時刻**で 2 回
    送っても、保留差は 458 のまま (加算して 916 にならない)。

    異なる時刻 = dedup_key (I4) が異なるため重複排除では防げない。
    `_finalize` 自身が chain 単位で冪等でなければならない。
    """
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 70.0, 1), _step(Side.P1, 70.0, 1, 500.0, cc=8),
        _finalize(Side.P1, 78.0, 1, 42.0),
        _finalize(Side.P1, 79.0, 1, 42.0),  # 同値の再入力 (時刻だけ違う)
    ])
    snap = led.snapshot()
    assert snap.net_raw == pytest.approx(500.0), "暫定値は保持されるべき"
    assert snap.unreconciled == pytest.approx(458.0), (
        f"保留差 458 が二重計上された (実測 {snap.unreconciled})"
    )


def test_p2_1_downward_finalize_value_update_replaces_held_divergence() -> None:
    """下方 FINALIZE の**値が更新**されたら、保留差は最新の値で置換される
    (458 + 470 = 928 のような累積をしない)。"""
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 70.0, 1), _step(Side.P1, 70.0, 1, 500.0, cc=8),
        _finalize(Side.P1, 78.0, 1, 42.0),   # 保留差 458
        _finalize(Side.P1, 79.0, 1, 30.0),   # 保留差 470 へ更新 (置換)
    ])
    snap = led.snapshot()
    assert snap.net_raw == pytest.approx(500.0)
    assert snap.unreconciled == pytest.approx(470.0), (
        f"保留差が置換でなく加算された (実測 {snap.unreconciled})"
    )


def test_p2_1_downward_hold_is_cleared_by_subsequent_normal_finalize() -> None:
    """保留 (下方 FINALIZE) の後に許容帯内の正常な確定が届いたら、
    保留差は解消され unreconciled が 0 に戻る (偽陽性を残さない)。"""
    led = ExchangeLedger()
    _push_all(led, [
        _fire(Side.P1, 70.0, 1), _step(Side.P1, 70.0, 1, 500.0, cc=8),
        _finalize(Side.P1, 78.0, 1, 42.0),    # 桁誤読等で一度保留 (差 458)
        _finalize(Side.P1, 79.0, 1, 497.0),   # 正常な確定 (差 -3、許容帯内)
    ])
    snap = led.snapshot()
    assert snap.net_raw == pytest.approx(497.0), "正常な確定で置換されるべき"
    assert snap.unreconciled == pytest.approx(0.0), (
        f"解消済みの保留差が unreconciled に残った (実測 {snap.unreconciled})"
    )
