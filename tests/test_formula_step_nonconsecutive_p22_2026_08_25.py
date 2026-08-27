"""P2-2 回帰テスト: 掛け算式の段確定は「連続」した valid 観測に限る (2026-08-25)。

## 何を守るテストか (Codex 独立レビュー P2-2、CODEX_TO_CLAUDE.md 2026-08-25 08:10 節)

`FormulaStepAccumulator.update` の確定規則は
「同一値が FORMULA_STEP_CONFIRM_FRAMES **連続**で読めたら 1 段確定」
(`src/score_ocr.py` docstring) である。ところが修正前の実装は、

- 幕間分岐 (`score_displayed=True`)
- 無効読取り分岐 (`result.valid == False`)

のどちらでも `_pending` / `_pending_count` を破棄せずに return していた。
そのため `valid → invalid/幕間 → 同じ valid` という**非連続の 2 観測**が
「連続 2 フレーム」として段に確定してしまう。

## なぜ本番に影響しうるか

掛け算式 3 フラグ (`--enable-chain-formula-read-verify` /
`--enable-formula-chain-count-update` / `--enable-formula-step-interlude`)
は `RECOGNITION_ADOPTED` 登録済み = `recognition_load_default_kwargs()` で
本番認識構成 ON。幕間通知は本番で毎フレーム届くため、
フェード中の部分読み 1 フレーム + スコア誤読 1 フレーム + 部分読み 1 フレーム
の並びで**存在しない段を積む** (過大方向の事故) 余地があった。

confirm_frames=2 という値の意図は「出現/消滅アニメ中の部分読み 1.4% を除去」
であり、部分読みは孤立フレームで出る。連続確認を非連続観測で満たしてしまうと
このノイズ除去が機能しない。

## 修正内容 (このテストが固定する挙動)

valid 観測が途切れたら (幕間 or 無効読取り)、pending の連続確認を破棄する。
セッション (確定済みの段) は破棄しない — セッション破棄は従来どおり
時間経過 (FORMULA_SESSION_RESET_SEC) でのみ判定する。
"""
from __future__ import annotations

from src.score_ocr import (
    FORMULA_STEP_CONFIRM_FRAMES,
    FormulaReadResult,
    FormulaStep,
    FormulaStepAccumulator,
)

# 実効 30fps (--normalize-fps-30 後) の 1 フレーム。
FRAME_SEC: float = 1.0 / 30.0


def _valid(left: int, right: int) -> FormulaReadResult:
    """有効な掛け算式読取り 1 件。"""
    return FormulaReadResult(
        valid=True, left=left, right=right, product=left * right, mult_ncc=0.9,
    )


def _invalid() -> FormulaReadResult:
    """無効読取り 1 件 (桁が読めない等。幕間とは区別される)。"""
    return FormulaReadResult(
        valid=False, left=None, right=None, product=None,
        mult_ncc=0.0, reject_reason="no_left",
    )


def _absent() -> FormulaReadResult:
    """掛け算式が表示されていないフレーム (幕間 = 通常スコア表示中)。"""
    return FormulaReadResult(
        valid=False, left=None, right=None, product=None,
        mult_ncc=0.0, reject_reason="score_displayed",
    )


# ===========================================================================
# 1. Codex 再現ケース: valid → invalid → 同じ valid は確定しない
# ===========================================================================


def test_valid_invalid_valid_does_not_confirm() -> None:
    """無効読取りを挟んだ 2 観測は「連続 2 フレーム」ではない (P2-2 本体)。

    修正前: invalid 分岐が pending を破棄しないため、3 フレーム目で
    confirm_frames=2 の段が確定していた。
    """
    assert FORMULA_STEP_CONFIRM_FRAMES == 2, (
        "本テストは confirm_frames=2 前提で書かれている。定数変更時は見直すこと"
    )
    acc = FormulaStepAccumulator()
    assert acc.update(0 * FRAME_SEC, _valid(50, 162)) is None
    assert acc.update(1 * FRAME_SEC, _invalid()) is None
    step = acc.update(2 * FRAME_SEC, _valid(50, 162))

    assert step is None, (
        f"非連続の 2 観測で段が確定してはいけない。確定した段: {step}"
    )
    assert acc.step_count == 0


def test_valid_interlude_valid_does_not_confirm() -> None:
    """幕間 (通常スコア誤読) を挟んだ 2 観測も「連続」ではない (本番経路)。

    幕間通知は本番 3 フラグ ON で毎フレーム届くため、こちらが本番で
    実際に通る経路。修正前は幕間分岐も pending を破棄していなかった。
    """
    acc = FormulaStepAccumulator()
    assert acc.update(0 * FRAME_SEC, _valid(50, 162)) is None
    assert acc.update(
        1 * FRAME_SEC, _absent(), score_displayed=True,
    ) is None
    step = acc.update(2 * FRAME_SEC, _valid(50, 162))

    assert step is None, (
        f"幕間を挟んだ 2 観測で段が確定してはいけない。確定した段: {step}"
    )
    assert acc.step_count == 0


def test_isolated_partial_reads_around_interlude_never_confirm() -> None:
    """孤立した部分読みが幕間を挟んで同値でも、段を積まない (過大方向の事故防止)。

    実測のフェード部分読み (「50×386」→「50× 86」) は孤立フレームで出る。
    幕間を挟んで同じ誤値が 2 回出ても、連続でなければノイズのまま。
    """
    acc = FormulaStepAccumulator()
    acc.update(0 * FRAME_SEC, _valid(50, 86))   # フェード部分読み (孤立)
    for i in range(1, 14):                       # 幕間 13 フレーム (実測下限)
        acc.update(i * FRAME_SEC, _absent(), score_displayed=True)
    acc.update(14 * FRAME_SEC, _valid(50, 86))  # 出現アニメ中の部分読み (孤立)
    acc.update(15 * FRAME_SEC, _invalid())

    assert acc.step_count == 0, (
        f"孤立部分読み 2 回で段を積んではいけない。実際 {acc.step_count} 段"
    )
    assert acc.total_power == 0


# ===========================================================================
# 2. 破棄するのは pending だけ。確定済みの段 (セッション) は失わない
# ===========================================================================


def test_invalid_discards_pending_but_keeps_confirmed_steps() -> None:
    """無効読取りは連続確認だけを破棄し、確定済みの段は保持する。

    セッション破棄は従来どおり時間経過 (2.0 秒) でのみ判定する
    (「無効読取り: セッション破棄は時間経過でのみ判定する」の既存契約)。
    """
    acc = FormulaStepAccumulator()
    t = 0.0
    for _ in range(FORMULA_STEP_CONFIRM_FRAMES):
        acc.update(t, _valid(40, 1))
        t += FRAME_SEC
    assert acc.step_count == 1

    acc.update(t, _invalid())  # 煙等で 1 フレーム読めない
    t += FRAME_SEC

    assert acc.step_count == 1, "無効読取りで確定済みの段を失ってはいけない"
    assert acc.total_power == 40 * 1


def test_confirm_recovers_with_truly_consecutive_frames() -> None:
    """途切れの後でも、本当に連続した confirm_frames 観測なら確定する (取り逃し防止)。

    段は実測 28 フレーム表示されるので、途中 1 フレームの誤読があっても
    残りの連続表示で確定できる。過小方向 (段の取り逃し) の退行を防ぐ。
    """
    acc = FormulaStepAccumulator()
    acc.update(0 * FRAME_SEC, _valid(50, 162))
    acc.update(1 * FRAME_SEC, _invalid())        # 1 フレームだけ読めない
    step: FormulaStep | None = None
    t_frame = 2
    for _ in range(FORMULA_STEP_CONFIRM_FRAMES):  # ここから連続で読める
        got = acc.update(t_frame * FRAME_SEC, _valid(50, 162))
        if got is not None:
            step = got
        t_frame += 1

    assert step is not None, "連続 confirm_frames 観測が揃ったのに確定しない"
    assert acc.step_count == 1
    assert acc.total_power == 50 * 162


def test_step_display_with_sporadic_dropouts_still_confirms() -> None:
    """段の表示 28 フレーム中に散発的な読取り失敗があっても段は確定する。

    修正 (途切れで pending 破棄) が過剰に働いて本物の段を取り逃さないこと。
    4 フレームごとに 1 回失敗という厳しめの想定でも、間の 3 連続で確定できる。
    """
    acc = FormulaStepAccumulator()
    step: FormulaStep | None = None
    for i in range(28):  # 実測の段表示長
        if i % 4 == 3:
            r: FormulaReadResult = _invalid()
        else:
            r = _valid(110, 10)
        got = acc.update(i * FRAME_SEC, r)
        if got is not None:
            step = got

    assert step is not None, "散発的な読取り失敗で段を取り逃してはいけない"
    assert acc.step_count == 1
    assert acc.total_power == 110 * 10


# ===========================================================================
# 3. 幕間の正規動作は壊さない (退行防止)
# ===========================================================================


def test_interlude_split_still_counts_two_steps() -> None:
    """幕間で区切られた同値 2 段は引き続き 2 段と数える (Q-01 挙動の維持)。"""
    acc = FormulaStepAccumulator()
    t_frame = 0

    def _run(result: FormulaReadResult, n: int, *, interlude: bool) -> None:
        nonlocal t_frame
        for _ in range(n):
            acc.update(t_frame * FRAME_SEC, result, score_displayed=interlude)
            t_frame += 1

    _run(_valid(200, 28), 28, interlude=False)
    _run(_absent(), 13, interlude=True)
    _run(_valid(200, 28), 28, interlude=False)

    assert acc.step_count == 2
    assert acc.total_power == 2 * 200 * 28
