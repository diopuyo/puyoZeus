"""STABLE凍結デッドロック根治 4フラグのテスト (2026-08-24)。

対象:
    - src/score_ocr.py: parse_formula_cells / FormulaReadResult /
      FormulaStepAccumulator / ScoreOcr.read_formula_side / read_side_detail
    - src/recognition_pipeline.py: enable_formula_value_read /
      enable_chain_formula_read_verify / enable_formula_chain_count_update /
      enable_slide_exit_no_min_display
    - src/state_detectors.py: ChainPhaseDetector.enable_formula_read_gate_bypass

方針: 全フラグ既定 OFF で既存挙動と bit-identical であることを静的に担保し
(既定値検査 + 既定経路の等価性)、ON 時の各機構を単体で検証する。
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.board import Board
from src.chain_detector import (
    CHAIN_MECHANISM_BASELINE,
    CHAIN_MECHANISM_FORMULA_READ,
    ChainEvent,
)
from src.recognition_pipeline import (
    RecognitionPipeline,
    _should_suppress_slide_exit,
)
from src.score_ocr import (
    FORMULA_MULT_NCC_MIN,
    FormulaReadResult,
    FormulaStep,
    FormulaStepAccumulator,
    ScoreOcr,
    parse_formula_cells,
)
from src.state_detectors import ChainPhaseDetector


# ===========================================================================
# parse_formula_cells (stateless 純関数)
# ===========================================================================

_HI = 0.95  # 数字セルの高 NCC
_LO = 0.30  # 空白セルの低 NCC


def _confs(*idx_high: int) -> tuple[float, ...]:
    return tuple(_HI if i in idx_high else _LO for i in range(8))


def test_parse_two_digit_left_three_digit_right() -> None:
    """「 50×162」(実測 c01 t=6707.3) が正読される。"""
    r = parse_formula_cells(
        (None, None, 5, 0, None, 1, 6, 2), _confs(2, 3, 5, 6, 7), 0.87,
    )
    assert r.valid and r.left == 50 and r.right == 162 and r.product == 8100


def test_parse_three_digit_left() -> None:
    """左辺3桁「100×294」(実測 c01 t=6712.9、user訂正 2026-08-24 の実例)。

    レイアウト: 左辺は cell1-3 右詰め、× は cell4 固定のまま。
    """
    r = parse_formula_cells(
        (None, 1, 0, 0, None, 2, 9, 4), _confs(1, 2, 3, 5, 6, 7), 0.87,
    )
    assert r.valid and r.left == 100 and r.right == 294 and r.product == 29400


def test_parse_one_digit_right() -> None:
    """右辺1桁「 40×  1」(実測 c03 t=4250.4、連鎖1段目のボーナス=1)。"""
    r = parse_formula_cells(
        (None, None, 4, 0, None, None, None, 1), _confs(2, 3, 7), 0.87,
    )
    assert r.valid and r.left == 40 and r.right == 1


def test_parse_rejects_low_mult_ncc() -> None:
    """×セルの NCC が閾値未満なら棄却 (通常スコア表示の最大実測 0.312)。"""
    r = parse_formula_cells(
        (None, None, 5, 0, None, 1, 6, 2), _confs(2, 3, 5, 6, 7),
        FORMULA_MULT_NCC_MIN - 0.01,
    )
    assert not r.valid and r.reject_reason == "mult"


def test_parse_rejects_lead_not_blank() -> None:
    """cell0 に数字がある (通常スコア残像等) 場合は棄却。"""
    r = parse_formula_cells(
        (9, 1, 0, 0, None, 2, 9, 4), _confs(0, 1, 2, 3, 5, 6, 7), 0.87,
    )
    assert not r.valid and r.reject_reason == "lead_not_blank"


def test_parse_rejects_left_gap_and_right_gap() -> None:
    """桁の非連続 (部分読み) は棄却。"""
    r1 = parse_formula_cells(
        (None, 1, None, 0, None, None, None, 1), _confs(1, 3, 7), 0.87,
    )
    assert not r1.valid and r1.reject_reason == "left_gap"
    r2 = parse_formula_cells(
        (None, None, 4, 0, None, 2, None, 4), _confs(2, 3, 5, 7), 0.87,
    )
    assert not r2.valid and r2.reject_reason == "right_gap"


def test_parse_rejects_implausible_left() -> None:
    """左辺の物理制約: 40 未満 (消去4個未満) / 10 の倍数でない値は棄却。"""
    # 「 0×224」 (実測 c01 t=6710.7 の遷移アーティファクト)
    r0 = parse_formula_cells(
        (None, None, None, 0, None, 2, 2, 4), _confs(3, 5, 6, 7), 0.87,
    )
    assert not r0.valid and r0.reject_reason == "left_implausible"
    # 45 (10の倍数でない = 桁欠け)
    r45 = parse_formula_cells(
        (None, None, 4, 5, None, 2, 2, 4), _confs(2, 3, 5, 6, 7), 0.87,
    )
    assert not r45.valid and r45.reject_reason == "left_implausible"


def test_parse_rejects_missing_required_digits() -> None:
    """左辺最下位 (cell3) / 右辺最下位 (cell7) の欠落は棄却。"""
    r1 = parse_formula_cells(
        (None, None, None, None, None, 1, 6, 2), _confs(5, 6, 7), 0.87,
    )
    assert not r1.valid and r1.reject_reason == "no_left"
    r2 = parse_formula_cells(
        (None, None, 5, 0, None, None, None, None), _confs(2, 3), 0.87,
    )
    assert not r2.valid and r2.reject_reason == "no_right"


# ===========================================================================
# FormulaStepAccumulator
# ===========================================================================


def _vr(left: int, right: int) -> FormulaReadResult:
    return FormulaReadResult(
        valid=True, left=left, right=right, product=left * right, mult_ncc=0.9,
    )


def _feed(acc: FormulaStepAccumulator, t: float, left: int, right: int,
          n: int = 2, dt: float = 1.0 / 30.0) -> "FormulaStep | None":
    """同一値を n フレーム連続で与え、最後の戻り値を返す。"""
    out = None
    for i in range(n):
        out = acc.update(t + i * dt, _vr(left, right))
    return out


def test_accumulator_confirms_after_two_frames() -> None:
    acc = FormulaStepAccumulator()
    assert acc.update(0.0, _vr(50, 162)) is None, "1フレーム目では未確定"
    step = acc.update(1.0 / 30.0, _vr(50, 162))
    assert step is not None and step.left == 50 and step.right == 162
    assert acc.step_count == 1 and acc.total_power == 8100


def test_accumulator_single_frame_partial_not_confirmed() -> None:
    """1フレームだけの読取り (部分読み) は段にならない。"""
    acc = FormulaStepAccumulator()
    _feed(acc, 0.0, 40, 160)
    acc.update(0.10, _vr(40, 999))  # 単発ノイズ
    _feed(acc, 1.4, 40, 192)
    assert acc.step_count == 2
    assert acc.total_power == 40 * 160 + 40 * 192


def test_accumulator_counts_steps_and_power() -> None:
    """実測 c03 (10連鎖) の値系列で 10 段 / 素点合計が一致する。"""
    seq = [(40, 1), (60, 11), (40, 16), (40, 32), (40, 64), (40, 96),
           (40, 128), (50, 162), (40, 192), (60, 227)]
    acc = FormulaStepAccumulator()
    t = 0.0
    for left, right in seq:
        _feed(acc, t, left, right, n=3)
        t += 1.4  # 段周期の実測値
    assert acc.step_count == 10
    assert acc.total_power == sum(l * r for l, r in seq)


def test_accumulator_drops_fadeout_partial_without_gap() -> None:
    """右辺減少 + 表示連続 (gap<0.5s) = フェード部分読みは棄却する。

    実測 c04/c05: 「50×386」→ 消滅アニメ中「50× 86」。
    """
    acc = FormulaStepAccumulator()
    _feed(acc, 0.0, 50, 386)
    _feed(acc, 0.2, 50, 86, n=3)  # 減少 + gap 0.2s < 0.5s
    assert acc.step_count == 1
    assert acc.total_power == 50 * 386


def test_accumulator_new_session_on_decrease_with_gap() -> None:
    """右辺減少 + 0.5s 以上の消失 = 新しい連鎖としてセッションを切り替える。"""
    acc = FormulaStepAccumulator()
    _feed(acc, 0.0, 50, 386)
    _feed(acc, 1.0, 40, 8, n=3)  # 減少 + gap 1.0s ≥ 0.5s → 新連鎖
    assert acc.step_count == 1
    assert acc.total_power == 40 * 8


def test_accumulator_session_reset_after_timeout() -> None:
    """有効読取りが 2.0s 超途絶えたらセッション破棄 (右辺増加でも別連鎖)。"""
    acc = FormulaStepAccumulator()
    _feed(acc, 0.0, 40, 8)
    _feed(acc, 3.0, 40, 16, n=3)  # 3.0s 途絶 → 別セッション
    assert acc.step_count == 1
    assert acc.total_power == 40 * 16


def test_accumulator_same_value_no_double_count() -> None:
    """同一段の再読 (値不変) は二重計上しない。"""
    acc = FormulaStepAccumulator()
    _feed(acc, 0.0, 50, 162, n=10)
    assert acc.step_count == 1


def test_accumulator_reset() -> None:
    acc = FormulaStepAccumulator()
    _feed(acc, 0.0, 50, 162)
    acc.reset()
    assert acc.step_count == 0 and acc.total_power == 0
    assert acc.last_valid_t is None


# ===========================================================================
# ScoreOcr: read_side_detail / read_formula_side (合成フレーム)
# ===========================================================================


def _synthetic_formula_frame(
    ocr: ScoreOcr, side: str, cells: dict[int, "np.ndarray"],
) -> np.ndarray:
    """テンプレ画像を score グリッドへ貼った 1080p 合成フレームを作る。"""
    from src.score_ocr import (
        DIGIT_HEIGHT, DIGIT_LEFTS_1P, DIGIT_LEFTS_2P, DIGIT_TOP,
        SCORE_1P_REGION, SCORE_2P_REGION,
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    region = SCORE_1P_REGION if side == "1P" else SCORE_2P_REGION
    lefts = DIGIT_LEFTS_1P if side == "1P" else DIGIT_LEFTS_2P
    y1 = region[0] + DIGIT_TOP
    for idx, img in cells.items():
        x1 = region[2] + lefts[idx]
        h, w = img.shape[:2]
        h = min(h, DIGIT_HEIGHT)
        patch = img[:h] if img.ndim == 3 else np.stack([img[:h]] * 3, axis=-1)
        frame[y1:y1 + h, x1:x1 + patch.shape[1]] = patch
    return frame


@pytest.fixture(scope="module")
def default_ocr() -> ScoreOcr:
    return ScoreOcr.load_default()


def test_read_formula_side_synthetic(default_ocr: ScoreOcr) -> None:
    """digit テンプレ + ×テンプレの合成フレームで「 50×162」を正読する。"""
    if default_ocr._mult_template_gray is None:
        pytest.skip("formula_mult.png 未整備")
    tpl = default_ocr._templates_gray
    if not all(k in tpl for k in (0, 1, 2, 5, 6)):
        pytest.skip("digit テンプレ未整備")
    mult = default_ocr._mult_template_gray
    cells = {2: tpl[5], 3: tpl[0], 4: mult, 5: tpl[1], 6: tpl[6], 7: tpl[2]}
    frame = _synthetic_formula_frame(default_ocr, "1P", cells)
    r = default_ocr.read_formula_side(frame, "1P")
    assert r.valid, f"棄却された: {r}"
    assert r.left == 50 and r.right == 162 and r.product == 8100


def test_read_formula_side_normal_score_is_invalid(default_ocr: ScoreOcr) -> None:
    """通常スコア表示 (8桁数字、×なし) では invalid になる。"""
    if default_ocr._mult_template_gray is None:
        pytest.skip("formula_mult.png 未整備")
    tpl = default_ocr._templates_gray
    if 0 not in tpl:
        pytest.skip("digit テンプレ未整備")
    cells = {i: tpl[0] for i in range(8)}
    frame = _synthetic_formula_frame(default_ocr, "1P", cells)
    r = default_ocr.read_formula_side(frame, "1P")
    assert not r.valid


def test_read_side_detail_matches_read_side(default_ocr: ScoreOcr) -> None:
    """read_side_detail の score/conf は read_side と完全一致 (同一実装委譲)。"""
    tpl = default_ocr._templates_gray
    if 0 not in tpl:
        pytest.skip("digit テンプレ未整備")
    cells = {i: tpl[0] for i in range(8)}
    frame = _synthetic_formula_frame(default_ocr, "2P", cells)
    s1, c1 = default_ocr.read_side(frame, "2P")
    s2, c2, labels, confs = default_ocr.read_side_detail(frame, "2P")
    assert s1 == s2 and c1 == c2
    assert len(labels) == 8 and len(confs) == 8


def test_read_formula_side_without_template_is_noop() -> None:
    """×テンプレ未登録の ScoreOcr では常に invalid (既存挙動への影響ゼロ)。"""
    ocr = ScoreOcr(templates={})
    r = ocr.read_formula_side(np.zeros((1080, 1920, 3), np.uint8), "1P")
    assert not r.valid and r.reject_reason == "no_template"


# ===========================================================================
# ChainPhaseDetector: 4連結ゲートのバイパス (根治②)
# ===========================================================================


def _frozen_ctx_without_erasable() -> object:
    """消せる4連結が無い凍結盤面の StateContext を作る。"""
    from src.board_state_machine import BoardState, StateContext
    ctx = StateContext()
    ctx.state = BoardState.STABLE
    b = Board()
    # 4連結が成立しない散在配置
    b.set(12, 0, 1)
    b.set(12, 2, 2)
    b.set(12, 4, 3)
    ctx.confirmed_board = b
    return ctx


def _formula_read_event() -> ChainEvent:
    return ChainEvent(
        trigger_sec=1.0, end_sec=2.0, before_board=Board(), chain_count=1,
        total_erased=0, total_score=0, base_score=0,
        all_clear_bonus_applied=0, ojama_sent=0, leftover_score=0,
        is_all_clear=False, mechanism=CHAIN_MECHANISM_FORMULA_READ,
    )


def _make_signals(ev: ChainEvent) -> object:
    from src.board_state_machine import DetectorSignals
    return DetectorSignals(
        time_sec=1.0, cnn_board=Board(), is_match_active=True, chain_event=ev,
    )


def test_gate_bypass_default_off_rejects_on_frozen_board() -> None:
    """既定 OFF: 凍結盤面に4連結が無ければ formula_read でも従来通り棄却。"""
    from src.chain import ChainSimulator
    det = ChainPhaseDetector(chain_sim=ChainSimulator())
    assert det.enable_formula_read_gate_bypass is False, "既定は OFF であること"
    ok = det._passes_erasable_gate(
        _frozen_ctx_without_erasable(), _make_signals(_formula_read_event()),
    )
    assert ok is False


def test_gate_bypass_on_allows_formula_read_only() -> None:
    """ON: formula_read イベントのみゲートをバイパスし、baseline は従来判定。"""
    from dataclasses import replace
    from src.chain import ChainSimulator
    det = ChainPhaseDetector(
        chain_sim=ChainSimulator(), enable_formula_read_gate_bypass=True,
    )
    ctx = _frozen_ctx_without_erasable()
    ok = det._passes_erasable_gate(ctx, _make_signals(_formula_read_event()))
    assert ok is True, "formula_read はバイパスされること"
    baseline_ev = replace(_formula_read_event(), mechanism=CHAIN_MECHANISM_BASELINE)
    ok2 = det._passes_erasable_gate(ctx, _make_signals(baseline_ev))
    assert ok2 is False, "baseline は従来通り凍結盤面ゲートで判定されること"


# ===========================================================================
# _should_suppress_slide_exit: X1 除外 (根治④)
# ===========================================================================


def test_slide_exit_guard_default_includes_x1() -> None:
    """既定 (include_min_display_guard=True): X1 で抑止される (従来挙動)。"""
    assert _should_suppress_slide_exit(
        time_sec=1.0, chain_entry_t=0.5, chain_count=5,
        chain_min_display_sec=0.8, chain_game_event_min_count=2,
        current_next=(1, 2), start_next=(3, 4),  # NEXT は変化済み
    ) is True, "entry から 0.5s < 0.8s なので X1 抑止 (従来挙動)"


def test_slide_exit_guard_no_min_display_drops_x1_keeps_x4() -> None:
    """include_min_display_guard=False: X1 は外れ、X4 と NEXT 裏取りは残る。"""
    common = dict(
        time_sec=1.0, chain_entry_t=0.5,
        chain_min_display_sec=0.8, chain_game_event_min_count=2,
    )
    # X1 相当の状況 + NEXT 変化済み + 連鎖数 X4 以上 → 抑止しない (即終了許可)
    assert _should_suppress_slide_exit(
        chain_count=5, current_next=(1, 2), start_next=(3, 4),
        include_min_display_guard=False, **common,
    ) is False
    # X4 (短連鎖) は維持される
    assert _should_suppress_slide_exit(
        chain_count=1, current_next=(1, 2), start_next=(3, 4),
        include_min_display_guard=False, **common,
    ) is True
    # NEXT 未変化の corroboration も維持される
    assert _should_suppress_slide_exit(
        chain_count=5, current_next=(3, 4), start_next=(3, 4),
        include_min_display_guard=False, **common,
    ) is True


# ===========================================================================
# RecognitionPipeline: フラグ既定値 (静的回帰) + 発火/更新経路
# ===========================================================================

_NEW_FLAGS = (
    "enable_formula_value_read",
    "enable_chain_formula_read_verify",
    "enable_formula_chain_count_update",
    "enable_slide_exit_no_min_display",
)


def test_new_flags_default_false_in_init_and_load_default() -> None:
    """新規 4 フラグは __init__ / load_default とも既定 False (bit-identical 保証)。"""
    for fn in (RecognitionPipeline.__init__, RecognitionPipeline.load_default):
        sig = inspect.signature(fn)
        for name in _NEW_FLAGS:
            assert name in sig.parameters, f"{fn.__qualname__} に {name} がない"
            assert sig.parameters[name].default is False, (
                f"{fn.__qualname__} の {name} 既定値が False でない"
            )


def _make_pipeline(**flags: bool) -> RecognitionPipeline:
    """最小構成 pipeline (CNN 不要、OCR モック)。"""
    from src.image_reader import ImageReader
    from src.match_state import MatchState, MatchStateDetector

    reader = MagicMock(spec=ImageReader)
    reader.read_both_boards.return_value = (Board(), Board())
    reader._classifier = None
    match_det = MagicMock(spec=MatchStateDetector)
    match_det.detect.return_value = MatchState.IN_MATCH
    ocr = MagicMock(spec=ScoreOcr)
    return RecognitionPipeline(
        image_reader=reader,
        match_state_detector=match_det,
        score_ocr=ocr,
        force_in_match=True,
        **flags,
    )


def test_flags_off_no_accumulators() -> None:
    """フラグ OFF では段累積器を作らない (既存経路への構造的影響ゼロ)。"""
    pipe = _make_pipeline()
    assert pipe._formula_accum_1p is None and pipe._formula_accum_2p is None
    assert pipe._enable_formula_value_read is False


def test_verify_or_update_flag_forces_value_read() -> None:
    """②/③ は ① (実読) を内部で強制 ON にする。"""
    p1 = _make_pipeline(enable_chain_formula_read_verify=True)
    assert p1._enable_formula_value_read is True
    assert p1._formula_accum_1p is not None
    p2 = _make_pipeline(enable_formula_chain_count_update=True)
    assert p2._enable_formula_value_read is True


def test_read_verify_fire_uses_formula_read_mechanism() -> None:
    """② ON + 実読 valid: 凍結盤面 simulate を通さず formula_read で発火する。

    盤面は空 (simulate 検証なら連鎖数0で棄却される状況) を渡し、
    実読証拠だけで発火することを確認する = デッドロックの直接解消。
    """
    pipe = _make_pipeline(enable_chain_formula_read_verify=True)
    pipe._formula_last_read_1p = _vr(50, 162)
    assert pipe._formula_accum_1p is not None
    _feed(pipe._formula_accum_1p, 10.0, 50, 162)
    pipe._apply_chain_formula_early_fire(
        side="1P", time_sec=10.1, prev_confirmed=Board(),  # 空盤面 = 4連結なし
    )
    ev = pipe._active_chain_1p
    assert ev is not None, "実読証拠で発火すること"
    assert ev.mechanism == CHAIN_MECHANISM_FORMULA_READ
    assert ev.chain_count == 1
    assert ev.total_score == 8100 and ev.score_estimated is False


def test_read_verify_falls_back_to_simulate_when_no_read() -> None:
    """② ON でも実読が無いフレームは従来の simulate 検証へフォールバック。

    空盤面 (連鎖数0) なので従来通り発火が抑制される = W7 対策の網は維持。
    """
    pipe = _make_pipeline(enable_chain_formula_read_verify=True)
    pipe._formula_last_read_1p = None
    pipe._apply_chain_formula_early_fire(
        side="1P", time_sec=10.0, prev_confirmed=Board(),
    )
    assert pipe._active_chain_1p is None, "simulate 検証で抑制されること"


def test_step_update_raises_count_and_score() -> None:
    """③: 段確定で formula 系イベントの count/score が引き上がり hold 延長。"""
    pipe = _make_pipeline(
        enable_chain_formula_read_verify=True,
        enable_formula_chain_count_update=True,
    )
    pipe._formula_last_read_1p = _vr(40, 1)
    acc = pipe._formula_accum_1p
    assert acc is not None
    _feed(acc, 10.0, 40, 1)
    pipe._apply_chain_formula_early_fire(
        side="1P", time_sec=10.1, prev_confirmed=Board(),
    )
    assert pipe._active_chain_1p is not None
    until_before = pipe._chain_until_1p
    # 2 段目・3 段目を確定させて更新を反映
    _feed(acc, 11.4, 40, 8)
    updated = pipe._apply_formula_step_update("1P", 11.45)
    _feed(acc, 12.8, 40, 16)
    updated = pipe._apply_formula_step_update("1P", 12.85)
    assert updated is not None
    ev = pipe._active_chain_1p
    assert ev.chain_count == 3
    assert ev.total_score == 40 * 1 + 40 * 8 + 40 * 16
    assert ev.score_estimated is False
    assert pipe._chain_until_1p > until_before, "hold が延長されること"


def test_step_update_does_not_touch_baseline_event() -> None:
    """③: baseline (実測) イベントには触れない。"""
    from dataclasses import replace
    pipe = _make_pipeline(enable_formula_chain_count_update=True)
    baseline = replace(
        _formula_read_event(), mechanism=CHAIN_MECHANISM_BASELINE,
        chain_count=7, total_score=12345,
    )
    pipe._active_chain_1p = baseline
    acc = pipe._formula_accum_1p
    assert acc is not None
    _feed(acc, 10.0, 40, 1)
    assert pipe._apply_formula_step_update("1P", 10.1) is None
    assert pipe._active_chain_1p is baseline
    assert pipe._active_chain_1p.chain_count == 7


def test_reset_clears_formula_state() -> None:
    """reset() で段累積器・実読キャッシュがクリアされる。"""
    pipe = _make_pipeline(enable_formula_value_read=True)
    acc = pipe._formula_accum_1p
    assert acc is not None
    _feed(acc, 10.0, 50, 162)
    pipe._formula_last_read_1p = _vr(50, 162)
    pipe.reset()
    assert acc.step_count == 0
    assert pipe._formula_last_read_1p is None
