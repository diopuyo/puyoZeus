"""Q-01 回帰テスト: 掛け算式の「段の同一性」判定 (2026-08-24 Codex 品質精査)。

## 何を守るテストか

`FormulaStepAccumulator` は連鎖の段を数える中核である。段を 1 つ落とすと
連鎖数・火力・CHAIN 保持時間・安全弁の入力がまとめて過小になり、
99%→1% 表示問題の修正入力そのものが壊れる。

現行実装は「右辺 (ボーナス倍率) は同一連鎖内で単調増加する」という前提を
置いているが、**この前提は公式得点式に照らして誤り**である。

    段の素点 = (消したぷよ数) × 10 × max(1, 連鎖ボーナス + 連結ボーナス + 色数ボーナス)
                                        ^^^^^^^^^^  ^^^^^^^^^^^^  ^^^^^^^^^^^^
                                        単調増加     段ごとに変動   段ごとに変動

`src/scoring.py` のボーナステーブル (2026-04-25 ウェブ調査済み) より:

- 連鎖ボーナス: [0, 8, 16, 32, 64, 96, ...] — 1→2 段の増分は **+8** しかない
- 連結ボーナス: 4個→0, 5→2, ..., 11個以上→10 (グループごとの和)
- 色数ボーナス: 1色→0, 2色→3, 3色→6, 4色→12

連結・色数の合計は最大 22 程度まで動くので、**連鎖ボーナスの増分 (+8/+16) を
上回って減ることがある**。したがって右辺は同一連鎖内で減少も同値もし得る。

## 現行実装が落とすもの (src/score_ocr.py:1065-1076)

1. 直前確定段と (左辺, 右辺) が完全一致する読取りを、**経過時間を問わず**
   「同じ段の再読」として棄却する。
2. 右辺が減少 or 同値の読取りを、消失ギャップ 0.5 秒以上なら
   **セッション破棄 (それまでの全段を消去)**、未満なら棄却する。

## 段の区切りの正しい信号 = 「幕間」(2026-08-24 実測で確定)

`src/score_ocr.py:113` の「同一連鎖中は掛け算式が連続表示される (402 フレーム実測)」は
**測定結果の誤引用**だった。402 フレームの実測は
memory `reference_chain_formula_layout_2026-08-24.md:41` の
「位置・サイズが不変 (y=30±1、倍率 1.0)」という**レイアウト不変性**の測定であり、
時間的連続性は測っていない。

実際には `src/recognition_pipeline.py:6272-6273` が

    if cached_score_val is not None:
        return None  # 通常スコア表示中 (掛け算式と排他)

としており、**掛け算式と通常スコアは排他**である。
`logs/_diag_formula_fix_e2e_2026-08-24/trace_on.jsonl` の 15 連鎖・13 境界を
フレーム単位で数えた実測:

- 1 段の表示 = 各 28 フレーム (0.933 秒) で一定
- **段間の幕間 = 13〜19 フレーム (0.433〜0.634 秒)、通常スコアが表示される**
- 幕間中のスコア増分 = 直前段の「左×右」と完全一致 (12/12)

実画面でも確認済み:
`logs/_diag_formula_fix_e2e_2026-08-24/screen_1p_t6697.6.jpg` はスコア欄が「40× 1」、
`同 screen_1p_t6698.9.jpg` は同じ位置が「00000304」(通常スコア)。

→ `FORMULA_NEW_SESSION_MIN_GAP_SEC = 0.5秒` は幕間分布 0.433〜0.634 秒の
**ど真ん中に刺さっている**。右辺が減れば、0.5 秒を超えた側では全段消去、
下回った側では段の棄却。どちらに転んでも壊れる。

## したがって段の同一性は「幕間の有無」で決める

- **幕間 (掛け算式が読めないフレーム) を挟んだ値の変化 = 新しい段。**
  値が直前段と同一でも、右辺が減っていても受理する。
- **幕間なしの値の変化 = フェード中の部分読み。棄却する。**
  (実測: 「50×386」→ 消滅アニメ中に先頭桁が落ちて「50× 86」)

新しい時間定数を導入しない。区切りがイベント駆動になるため不要。

## このファイルの位置づけ

「あるべき挙動」を先に固定する回帰テスト。
`test_*_with_interlude` 系が修正前は失敗する。

なお本ファイルは **accumulator 単体 (純ロジック)** を検証する。
「幕間のフレームで `update()` が呼ばれること」自体は配線の問題であり
(現状 `recognition_pipeline` は読取り None のフレームで `update()` を呼んでいない)、
**別途、非汎用スクリプトによる配線確認が要る**
(`feedback_wiring_check_needs_nongeneric_scripts_2026-08-18`)。

既存の `tests/test_formula_value_read.py` には旧前提を期待値として書いた項目
(`test_accumulator_new_session_on_decrease_with_gap` 等) があり、修正時には
そちらの見直しも要る。**根拠は物理則 (公式得点式) と実測であって、
テストを通すための都合ではない。**
"""
from __future__ import annotations

import pytest

from src.score_ocr import (
    FORMULA_STEP_CONFIRM_FRAMES,
    FormulaReadResult,
    FormulaStepAccumulator,
)

# 段の表示周期の実測値 (trace_on.jsonl、15連鎖13境界)。
STEP_PERIOD_SEC: float = 1.4

# 1 段が表示され続けるフレーム数の実測値 (各段 28 フレーム = 0.933 秒で一定)。
STEP_VISIBLE_FRAMES: int = 28

# 段間の幕間 (通常スコア表示 = 掛け算式が読めない区間) の実測フレーム数。
# 実測 13〜19 フレーム (0.433〜0.634 秒)。テストには下限側を使う。
INTERLUDE_FRAMES: int = 13

# 実効 30fps (--normalize-fps-30 後) の 1 フレーム。
FRAME_SEC: float = 1.0 / 30.0


def _valid(left: int, right: int) -> FormulaReadResult:
    """有効な掛け算式読取り 1 件。"""
    return FormulaReadResult(
        valid=True, left=left, right=right, product=left * right, mult_ncc=0.9,
    )


def _absent() -> FormulaReadResult:
    """掛け算式が表示されていないフレーム (幕間 = 通常スコア表示中)。

    `RecognitionPipeline._read_formula_value` は通常スコアが読めたフレームで
    None を返す (`src/recognition_pipeline.py:6272-6273`)。累積器から見ると
    「無効読取り」と同じ扱いになる。
    """
    return FormulaReadResult(
        valid=False, left=None, right=None, product=None,
        mult_ncc=0.0, reject_reason="score_displayed",
    )


class _Clock:
    """フレーム単位で進む単調時計 (テストの時刻計算を実態に合わせる)。"""

    def __init__(self) -> None:
        self.t: float = 0.0

    def advance(self, frames: int) -> None:
        self.t += frames * FRAME_SEC


def _show_step(
    acc: FormulaStepAccumulator,
    clock: _Clock,
    left: int,
    right: int,
    frames: int = STEP_VISIBLE_FRAMES,
) -> None:
    """1 段を frames フレーム表示する。"""
    for _ in range(frames):
        acc.update(clock.t, _valid(left, right))
        clock.advance(1)


def _show_interlude(
    acc: FormulaStepAccumulator,
    clock: _Clock,
    frames: int = INTERLUDE_FRAMES,
) -> None:
    """段間の幕間 (通常スコア表示) を frames フレーム流す。

    `score_displayed=True` は「通常スコアが読めた = 掛け算式は出ていない」
    という肯定的な観測。読取り失敗とは区別する必要がある
    (`FormulaStepAccumulator.update` の docstring 参照)。
    """
    for _ in range(frames):
        acc.update(clock.t, _absent(), score_displayed=True)
        clock.advance(1)


# ===========================================================================
# 1. 同じ掛け算式が幕間をはさんで再出現したら 2 段として数える
# ===========================================================================


def test_same_formula_with_interlude_counts_as_two_steps() -> None:
    """同一の (左辺, 右辺) でも、幕間をはさんで再出現したら別の段として数える。

    物理的な実現例 (公式ボーナステーブルより、いずれも 20 個消し = 左辺 200):
      - 2 連鎖目: 5+5+5+5 の 4 色同時
            連鎖 8 + 連結 (2×4=8) + 色数 12 = 28
      - 3 連鎖目: 4+4+4+4+4 の 4 色 (1 色が 2 グループ)
            連鎖 16 + 連結 0 + 色数 12 = 28
    右辺がたまたま一致するが、これは紛れもなく別の段である。
    """
    acc = FormulaStepAccumulator()
    clock = _Clock()
    _show_step(acc, clock, 200, 28)
    _show_interlude(acc, clock)
    _show_step(acc, clock, 200, 28)

    assert acc.step_count == 2, (
        "同一値でも幕間をはさんで再出現したら別の段。"
        f" 実際は {acc.step_count} 段"
    )
    assert acc.total_power == 2 * 200 * 28


def test_same_formula_without_interlude_is_not_double_counted() -> None:
    """同一段の連続表示 (幕間なし) は二重計上しない。

    上の修正で「同一値なら常に新しい段」にしてしまうと、表示が続いている
    だけの同一段を毎フレーム数えてしまう。その退行を防ぐ。
    """
    acc = FormulaStepAccumulator()
    clock = _Clock()
    _show_step(acc, clock, 50, 162, frames=90)  # 3 秒ぶん連続表示、幕間なし

    assert acc.step_count == 1, (
        f"幕間なしの連続表示は 1 段。実際は {acc.step_count} 段"
    )
    assert acc.total_power == 50 * 162


# ===========================================================================
# 2. 右辺倍率が下がる正当な連鎖を、同じセッションの 2 段として数える
# ===========================================================================


@pytest.mark.parametrize("interlude_frames", [13, 16, 19])
def test_decreasing_multiplier_with_interlude_keeps_both_steps(
    interlude_frames: int,
) -> None:
    """1 段目より 2 段目の右辺が低い正当な連鎖を、同一連鎖の 2 段として数える。

    物理的な実現例:
      - 1 連鎖目: 11 個の 1 色連結
            連鎖 0 + 連結 10 + 色数 0 = 10   (左辺 110)
      - 2 連鎖目: 4 個消し 1 色
            連鎖 8 + 連結 0 + 色数 0 = 8    (左辺 40)
    右辺は 10 → 8 と減るが、同じ連鎖の第 1 段と第 2 段である。

    ## 幕間の長さを 13 / 16 / 19 フレームで振る理由

    実測の幕間は 13〜19 フレーム (0.433〜0.634 秒) に分布し、
    現行の `FORMULA_NEW_SESSION_MIN_GAP_SEC = 0.5 秒` (= 15 フレーム) が
    **その分布のど真ん中に刺さっている**。修正前の実測値 (本ケースで計測):

      | 幕間 | 動作 | 残る段 | total_power |
      |---|---|---|---|
      | 13f (0.433秒) | 2 段目を**棄却** | (110, 10) | 1,100 |
      | 15f (0.500秒) | **セッション破棄** | (40, 8) | 320 |
      | 16f (0.533秒) | **セッション破棄** | (40, 8) | 320 |
      | 19f (0.633秒) | **セッション破棄** | (40, 8) | 320 |

    どちらも段数は 1 だが、**残る中身が真逆**になる。
    同じ物理現象が幕間の揺らぎ (0.1 秒) だけで 2 通りに壊れ、
    しかも火力が 1,100 と 320 で 3.4 倍ずれる。
    正しい答えはどちらでもなく **2 段・1,420** である。
    """
    acc = FormulaStepAccumulator()
    clock = _Clock()
    _show_step(acc, clock, 110, 10)
    _show_interlude(acc, clock, frames=interlude_frames)
    _show_step(acc, clock, 40, 8)

    assert acc.step_count == 2, (
        f"右辺が減っても同一連鎖の 2 段 (幕間 {interlude_frames} フレーム)。"
        f" 実際は {acc.step_count} 段"
    )
    assert acc.total_power == 110 * 10 + 40 * 8


def test_decreasing_multiplier_mid_chain_keeps_earlier_steps() -> None:
    """連鎖の途中で右辺が下がっても、それ以前の段を失わない。

    4 色同時消しが挟まると連鎖ボーナスの増分 (+8/+16) を
    色数ボーナスの変動が上回り、右辺が一時的に下がる。
    """
    seq = [(40, 1), (160, 20), (40, 16), (40, 32)]  # 3 段目で右辺が下がる
    acc = FormulaStepAccumulator()
    clock = _Clock()
    for i, (left, right) in enumerate(seq):
        if i > 0:
            _show_interlude(acc, clock)
        _show_step(acc, clock, left, right)

    assert acc.step_count == 4, (
        f"右辺の一時的な低下で段を失ってはいけない。実際は {acc.step_count} 段"
    )
    assert acc.total_power == sum(left * right for left, right in seq)


# ===========================================================================
# 3. フェード中の部分読みは棄却する (退行防止)
# ===========================================================================


def test_fadeout_partial_read_without_interlude_is_rejected() -> None:
    """消滅アニメ中の桁欠け (幕間をはさまず値が変わる) は段にしない。

    実測 c04/c05: 「50×386」→ 消滅アニメ中に先頭桁が落ちて「50× 86」。
    幕間をはさまないので、正しい段の切り替わりとは区別できる。
    """
    acc = FormulaStepAccumulator()
    clock = _Clock()
    _show_step(acc, clock, 50, 386)
    _show_step(acc, clock, 50, 86, frames=3)  # 幕間なしで値が変化 = 部分読み

    assert acc.step_count == 1, (
        f"幕間なしの値変化は段にしない。実際は {acc.step_count} 段"
    )
    assert acc.total_power == 50 * 386


def test_single_frame_noise_is_rejected() -> None:
    """1 フレームだけのノイズ読取りは段にしない (既存挙動の維持)。"""
    acc = FormulaStepAccumulator()
    clock = _Clock()
    _show_step(acc, clock, 40, 160)
    acc.update(clock.t, _valid(40, 999))  # 単発ノイズ (幕間なし)
    clock.advance(1)
    _show_interlude(acc, clock)
    _show_step(acc, clock, 40, 192)

    assert acc.step_count == 2, (
        f"単発ノイズは段にしない。実際は {acc.step_count} 段"
    )
    assert acc.total_power == 40 * 160 + 40 * 192


# ===========================================================================
# 4. 実測の 15 連鎖を丸ごと再現する
# ===========================================================================


def test_real_15_chain_sequence_from_trace() -> None:
    """実測 (trace_on.jsonl、1P の 15 連鎖) の値系列を幕間つきで再現する。

    段の表示 28 フレーム / 幕間 13 フレームという実測のリズムで流し、
    全段が数えられることを確認する。
    """
    seq = [
        (40, 1), (50, 10), (40, 16), (40, 32), (40, 64),
        (40, 96), (40, 128), (50, 162), (40, 192), (60, 227),
        (40, 256), (40, 288), (40, 320), (40, 352),
    ]
    acc = FormulaStepAccumulator()
    clock = _Clock()
    for i, (left, right) in enumerate(seq):
        if i > 0:
            _show_interlude(acc, clock)
        _show_step(acc, clock, left, right)

    assert acc.step_count == len(seq), (
        f"実測 15 連鎖の段数が合わない。期待 {len(seq)} / 実際 {acc.step_count}"
    )
    assert acc.total_power == sum(left * right for left, right in seq)


# ===========================================================================
# 5. 別の連鎖はきちんと分離する (退行防止)
# ===========================================================================


def test_long_silence_starts_new_chain() -> None:
    """有効読取りが十分長く途絶えたら、別の連鎖として数え直す。

    連鎖と連鎖の間には、段間の幕間 (最大 0.634 秒) よりはるかに長い
    無表示区間がある。フェイルセーフとして機能すること。
    """
    acc = FormulaStepAccumulator()
    clock = _Clock()
    _show_step(acc, clock, 40, 8)
    _show_interlude(acc, clock, frames=150)  # 5 秒の無表示 = 別の連鎖
    _show_step(acc, clock, 40, 16)

    assert acc.step_count == 1, (
        f"別の連鎖は分離する。実際は {acc.step_count} 段"
    )
    assert acc.total_power == 40 * 16


def test_reset_clears_everything() -> None:
    """試合切替等の明示リセットで完全に初期化される。"""
    acc = FormulaStepAccumulator()
    clock = _Clock()
    _show_step(acc, clock, 50, 162)
    acc.reset()

    assert acc.step_count == 0
    assert acc.total_power == 0


# ===========================================================================
# 6. 物理則の検算 — 段の素点は必ず 10 の倍数
# ===========================================================================


@pytest.mark.parametrize(
    "left,right",
    [(40, 1), (110, 10), (200, 28), (50, 386), (160, 20)],
)
def test_step_product_is_multiple_of_ten(left: int, right: int) -> None:
    """左辺は消去数×10 なので、段の素点は必ず 10 の倍数になる。

    docs/KNOWN_WEAKNESSES.md W2 の物理制約。読取り値の検算に使える。
    """
    acc = FormulaStepAccumulator()
    clock = _Clock()
    _show_step(acc, clock, left, right)

    assert acc.step_count == 1
    assert acc.total_power % 10 == 0


def test_confirm_frames_constant_is_below_observed_step_length() -> None:
    """確認フレーム数が実測の段表示長より十分短いこと (取り逃し防止)。

    段は実測 28 フレーム表示されるので、確認閾値がそれを超えると
    正しい段を取り逃す。物理量による上限の固定。
    """
    assert FORMULA_STEP_CONFIRM_FRAMES < STEP_VISIBLE_FRAMES


# ===========================================================================
# 7. 幕間を渡さない呼出しは旧挙動のまま (backwards compat)
# ===========================================================================


def test_single_frame_score_misread_does_not_open_a_step() -> None:
    """1 フレームだけの通常スコア誤読を「幕間」と認めない。

    幕間フラグが 1 フレームで立つと、直後のフェード部分読み
    (「50×386」→「50× 86」) を新しい段として拾ってしまう
    (**過大方向の事故** = 存在しない火力を作る)。
    実測の幕間は 13〜19 フレームなので、連続確認を要求しても
    本物の幕間は取り逃さない。
    """
    acc = FormulaStepAccumulator()
    clock = _Clock()
    _show_step(acc, clock, 50, 386)
    _show_interlude(acc, clock, frames=1)      # 単発のスコア誤読
    _show_step(acc, clock, 50, 86, frames=3)   # フェード部分読み

    assert acc.step_count == 1, (
        f"単発のスコア誤読で段を開いてはいけない。実際は {acc.step_count} 段"
    )
    assert acc.total_power == 50 * 386


def test_two_frame_interlude_is_accepted() -> None:
    """確認フレーム数ぶんの幕間があれば段の区切りとして認める (取り逃し防止)。"""
    acc = FormulaStepAccumulator()
    clock = _Clock()
    _show_step(acc, clock, 110, 10)
    _show_interlude(acc, clock, frames=FORMULA_STEP_CONFIRM_FRAMES)
    _show_step(acc, clock, 40, 8)

    assert acc.step_count == 2
    assert acc.total_power == 110 * 10 + 40 * 8


def test_without_interlude_signal_legacy_rules_still_apply() -> None:
    """`score_displayed` を渡さない呼出しは旧規則で動く。

    幕間を観測できない構成 (フラグ OFF、あるいは通常スコアも読めない区間) では、
    右辺の単調増加を使う旧規則にフォールバックする。
    これがあるので既定 OFF で bit-identical が成立する。
    """
    acc = FormulaStepAccumulator()
    clock = _Clock()
    _show_step(acc, clock, 110, 10)
    # 幕間を「渡さずに」時間だけ進める (掛け算式も通常スコアも読めない区間)
    clock.advance(INTERLUDE_FRAMES)
    _show_step(acc, clock, 40, 8)

    # 旧規則: 右辺減少 + gap 0.433秒 < 0.5秒 → 2 段目を棄却
    assert acc.step_count == 1
    assert acc.total_power == 110 * 10


# ===========================================================================
# 8. 配線テスト — 幕間が実際に累積器へ届くか
#    (feedback_wiring_check_needs_nongeneric_scripts_2026-08-18:
#     --help 突合や署名確認では配線漏れを検出できない)
# ===========================================================================


class _AccumSpy:
    """`update` の呼び出しを記録するだけのスパイ。"""

    def __init__(self) -> None:
        self.calls: list[tuple[float, bool]] = []

    def update(self, t_sec: float, result: object, *,
               score_displayed: bool = False) -> None:
        self.calls.append((t_sec, score_displayed))
        return None


def _make_pipeline_stub(*, flag: bool) -> object:
    """`_notify_formula_interlude` を呼べる最小のスタブを作る。

    `RecognitionPipeline` の実インスタンス生成は CNN 等の重い依存を伴うため、
    メソッドを未束縛のまま最小の属性セットに適用する。
    **検査対象は「フラグと cached_score_val の分岐が実際に効くか」**であり、
    この分岐は `_notify_formula_interlude` の中に閉じている。
    """
    from src.recognition_pipeline import RecognitionPipeline

    class _Stub:
        pass

    stub = _Stub()
    stub._enable_formula_step_interlude = flag  # type: ignore[attr-defined]
    stub._formula_accum_1p = _AccumSpy()        # type: ignore[attr-defined]
    stub._formula_accum_2p = _AccumSpy()        # type: ignore[attr-defined]
    stub._notify = (                            # type: ignore[attr-defined]
        RecognitionPipeline._notify_formula_interlude.__get__(stub)
    )
    return stub


def test_wiring_interlude_reaches_accumulator_when_flag_on() -> None:
    """フラグ ON かつ通常スコアが読めたフレームで、累積器へ幕間が届く。"""
    stub = _make_pipeline_stub(flag=True)
    stub._notify("1P", 12.5, 4820)  # type: ignore[attr-defined]

    calls = stub._formula_accum_1p.calls  # type: ignore[attr-defined]
    assert calls == [(12.5, True)], f"幕間が届いていない: {calls}"
    assert stub._formula_accum_2p.calls == []  # type: ignore[attr-defined]


def test_wiring_interlude_not_sent_when_flag_off() -> None:
    """フラグ OFF では一切通知しない (既定 OFF で bit-identical)。"""
    stub = _make_pipeline_stub(flag=False)
    stub._notify("1P", 12.5, 4820)  # type: ignore[attr-defined]

    assert stub._formula_accum_1p.calls == []  # type: ignore[attr-defined]


def test_wiring_read_failure_is_not_treated_as_interlude() -> None:
    """通常スコアが読めなかったフレームを幕間扱いしない。

    「読めなかった」と「表示されていないことが分かった」は別物。
    前者を幕間と誤認すると**存在しない段を積む** (過大方向の事故)。
    """
    stub = _make_pipeline_stub(flag=True)
    stub._notify("1P", 12.5, None)  # type: ignore[attr-defined]

    assert stub._formula_accum_1p.calls == []  # type: ignore[attr-defined]


def test_wiring_side_is_not_swapped() -> None:
    """1P/2P の取り違えが無いこと (側の取り違えは過去に事故を起こしている)。"""
    stub = _make_pipeline_stub(flag=True)
    stub._notify("2P", 30.0, 1234)  # type: ignore[attr-defined]

    assert stub._formula_accum_1p.calls == []  # type: ignore[attr-defined]
    assert stub._formula_accum_2p.calls == [(30.0, True)]  # type: ignore[attr-defined]


def test_flag_forces_formula_value_read_on() -> None:
    """⑤ を ON にすると ① (実読) が強制 ON になる。

    累積器そのものが `enable_formula_value_read` の下でしか生成されないため、
    ここが繋がっていないと幕間を届ける先が存在しない (配線漏れ)。
    """
    import inspect

    from src.recognition_pipeline import RecognitionPipeline

    src = inspect.getsource(RecognitionPipeline.__init__)
    assert "or self._enable_formula_step_interlude" in src, (
        "enable_formula_step_interlude が enable_formula_value_read を"
        "強制 ON にする配線が見当たらない"
    )
