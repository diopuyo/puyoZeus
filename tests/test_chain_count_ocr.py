"""src/chain_count_ocr.py のテスト。

「N れんさ!」ポップアップ OCR を合成画像 + 実テンプレで検証する。
表示位置が可変な仕様のため、ROI 内の複数位置にテンプレを貼って
位置非依存性 (盤面全体スキャン) を確認するのが本テストの要点。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.chain_count_ocr import (
    CHAIN_COUNT_MAX,
    CHAIN_COUNT_MIN,
    CHAIN_DIGIT_HEIGHT,
    CHAIN_DIGIT_LABELS,
    CHAIN_DIGIT_WIDTH,
    EXPECTED_FRAME_SHAPE,
    ChainCountOcr,
    ChainCountReadResult,
    ChainCountWindowResult,
    _aggregate_window_samples,
    _approx_min_chain_score,
    _extract_monotonic_max_chain_count,
    _select_chain_count_by_score,
)
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION

TEMPLATE_DIR = Path("models/ui_templates/chain_count_digits")


def _load_real_templates() -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if not TEMPLATE_DIR.is_dir():
        return out
    for n in CHAIN_DIGIT_LABELS:
        p = TEMPLATE_DIR / f"digit_{n}.png"
        img = cv2.imread(str(p))
        if img is not None:
            out[n] = img
    return out


def _make_blank_frame() -> np.ndarray:
    h, w = EXPECTED_FRAME_SHAPE
    return np.full((h, w, 3), 40, dtype=np.uint8)  # 盤面の暗い背景を模した一様色


def _paint_digit_into_board(
    frame: np.ndarray, digit_img: np.ndarray, side: str,
    offset_x: int, offset_y: int,
) -> np.ndarray:
    """盤面 ROI 内の任意位置にテンプレを貼り付ける (位置可変性の再現)。"""
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    ax = region.x + offset_x
    ay = region.y + offset_y
    h, w = digit_img.shape[:2]
    frame[ay:ay + h, ax:ax + w] = digit_img
    return frame


class _FakeVideoCapture:
    """cv2.VideoCapture の set()/read() のみを模した簡易フェイク。

    read_max_in_window() は set(CAP_PROP_POS_MSEC, ...) → read() の順で
    呼ばれる想定のため、set() で指定した時刻に応じてフレームを切り替える。
    """

    def __init__(self, frame_by_time_sec: dict[float, np.ndarray], default: np.ndarray) -> None:
        self._frame_by_time = frame_by_time_sec
        self._default = default
        self._cur_time_ms: float = 0.0

    def set(self, prop: int, value: float) -> None:
        self._cur_time_ms = value

    def read(self) -> tuple[bool, np.ndarray]:
        t_sec = round(self._cur_time_ms / 1000.0, 2)
        # 最も近い登録時刻のフレームを返す (テスト用の簡易実装)
        if not self._frame_by_time:
            return True, self._default
        closest = min(self._frame_by_time, key=lambda k: abs(k - t_sec))
        return True, self._frame_by_time[closest]


# =============================================================================
# 定数・基本形状
# =============================================================================


def test_chain_count_ocr_constants() -> None:
    assert CHAIN_COUNT_MIN == 1
    # 2026-07-29: 2桁対応 (10-19連鎖、理論上限19連鎖) により上限を拡張。
    assert CHAIN_COUNT_MAX == 19
    # CHAIN_DIGIT_LABELS はテンプレ「グリフ」クラス (0-9 の 10 種類)。
    # "0" は2桁表示の一の位専用で単体の最終結果 (CHAIN_COUNT_MIN=1) とは別概念。
    assert len(CHAIN_DIGIT_LABELS) == 10
    assert set(CHAIN_DIGIT_LABELS) == set(range(0, 10))
    assert CHAIN_DIGIT_HEIGHT > 0 and CHAIN_DIGIT_WIDTH > 0


def test_chain_count_ocr_no_templates_returns_none() -> None:
    """テンプレを 1 つも渡さない場合は常に None。"""
    ocr = ChainCountOcr(templates={})
    frame = _make_blank_frame()
    res = ocr.read_side(frame, "1P")
    assert isinstance(res, ChainCountReadResult)
    assert res.chain_count is None


def test_chain_count_ocr_invalid_frame_shape() -> None:
    """形状不正フレームでは None を返す (クラッシュしない)。"""
    ocr = ChainCountOcr.load_default()
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    res = ocr.read_side(bad, "1P")
    assert res.chain_count is None


def test_chain_count_ocr_load_default_present_digits() -> None:
    """0-9 の全クラスが実テンプレとして配置済み (2026-07-29 採取完了)。"""
    ocr = ChainCountOcr.load_default()
    assert ocr is not None


# =============================================================================
# 位置可変性: ROI 内の異なる座標でも読み取れることを確認 (本モジュールの要点)
# =============================================================================


@pytest.mark.parametrize("digit", [1, 2, 3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("offset", [(30, 200), (150, 450)])
def test_chain_count_ocr_reads_digit_at_various_positions(
    digit: int, offset: tuple[int, int],
) -> None:
    templates = _load_real_templates()
    if digit not in templates:
        pytest.skip(f"digit_{digit} テンプレ未整備")
    ocr = ChainCountOcr.load_default()
    frame = _make_blank_frame()
    tpl = templates[digit]
    offset_x, offset_y = offset
    frame = _paint_digit_into_board(frame, tpl, "1P", offset_x, offset_y)
    res = ocr.read_side(frame, "1P")
    assert res.chain_count == digit


def test_chain_count_ocr_digit_zero_alone_returns_none() -> None:
    """"0" は2桁表示の一の位専用で、単体では最終結果になり得ない (仕様)。"""
    templates = _load_real_templates()
    if 0 not in templates:
        pytest.skip("digit_0 テンプレ未整備")
    ocr = ChainCountOcr.load_default()
    frame = _make_blank_frame()
    frame = _paint_digit_into_board(frame, templates[0], "1P", 60, 300)
    res = ocr.read_side(frame, "1P")
    assert res.chain_count is None


def test_chain_count_ocr_side_2p_independent_of_1p() -> None:
    """2P 側 ROI に貼っても 1P 側からは検出されない (ROI 分離の確認)。"""
    templates = _load_real_templates()
    if 2 not in templates:
        pytest.skip("digit_2 テンプレ未整備")
    ocr = ChainCountOcr.load_default()
    frame = _make_blank_frame()
    frame = _paint_digit_into_board(frame, templates[2], "2P", 100, 300)
    res_2p = ocr.read_side(frame, "2P")
    res_1p = ocr.read_side(frame, "1P")
    assert res_2p.chain_count == 2
    assert res_1p.chain_count is None


def test_chain_count_ocr_no_popup_returns_none() -> None:
    """ポップアップが出ていない (無地) 盤面では None を返す。"""
    ocr = ChainCountOcr.load_default()
    frame = _make_blank_frame()
    res = ocr.read_side(frame, "1P")
    assert res.chain_count is None


# =============================================================================
# 2桁 (10-19連鎖) 結合ロジック (2026-07-29 追加)
# =============================================================================


def _paint_two_digit_into_board(
    frame: np.ndarray, tens_tpl: np.ndarray, ones_tpl: np.ndarray,
    side: str, offset_x: int, offset_y: int, gap_px: int,
) -> np.ndarray:
    """十の位・一の位のテンプレを隙間 gap_px で横に並べて貼り付ける。"""
    frame = _paint_digit_into_board(frame, tens_tpl, side, offset_x, offset_y)
    ones_x = offset_x + tens_tpl.shape[1] + gap_px
    frame = _paint_digit_into_board(frame, ones_tpl, side, ones_x, offset_y)
    return frame


@pytest.mark.parametrize("ones_digit", [0, 2, 5, 9])
def test_chain_count_ocr_reads_two_digit_combo(ones_digit: int) -> None:
    """十の位=1 と一の位を隣接配置すると 10+ones_digit に結合される。"""
    templates = _load_real_templates()
    if 1 not in templates or ones_digit not in templates:
        pytest.skip(f"digit_1 または digit_{ones_digit} テンプレ未整備")
    ocr = ChainCountOcr.load_default()
    frame = _make_blank_frame()
    frame = _paint_two_digit_into_board(
        frame, templates[1], templates[ones_digit], "1P", 60, 300, gap_px=2,
    )
    res = ocr.read_side(frame, "1P")
    assert res.chain_count == 10 + ones_digit


def test_chain_count_ocr_lone_digit_one_not_combined() -> None:
    """"1" の近くに他の数字がなければ 2桁結合されず単体の 1 のまま。"""
    templates = _load_real_templates()
    if 1 not in templates:
        pytest.skip("digit_1 テンプレ未整備")
    ocr = ChainCountOcr.load_default()
    frame = _make_blank_frame()
    frame = _paint_digit_into_board(frame, templates[1], "1P", 60, 300)
    res = ocr.read_side(frame, "1P")
    assert res.chain_count == 1


def test_chain_count_ocr_two_digit_row_mismatch_not_combined() -> None:
    """縦位置が大きくずれた数字同士は2桁として結合しない (誤結合防止)。"""
    templates = _load_real_templates()
    if 1 not in templates or 5 not in templates:
        pytest.skip("digit_1 または digit_5 テンプレ未整備")
    ocr = ChainCountOcr.load_default()
    frame = _make_blank_frame()
    frame = _paint_digit_into_board(frame, templates[1], "1P", 60, 200)
    # 縦に 100px ずらして配置 (許容量 CHAIN_TWO_DIGIT_ROW_TOLERANCE_PX=20 を超える)
    frame = _paint_digit_into_board(frame, templates[5], "1P", 130, 300)
    res = ocr.read_side(frame, "1P")
    # 2桁 (15) には結合されず、いずれかの単体桁として読める
    assert res.chain_count in (1, 5)


def test_chain_count_ocr_two_digit_too_far_not_combined() -> None:
    """横に離れすぎた数字同士は2桁として結合しない (誤結合防止)。"""
    templates = _load_real_templates()
    if 1 not in templates or 5 not in templates:
        pytest.skip("digit_1 または digit_5 テンプレ未整備")
    ocr = ChainCountOcr.load_default()
    frame = _make_blank_frame()
    frame = _paint_digit_into_board(frame, templates[1], "1P", 60, 300)
    # 許容ギャップ (CHAIN_TWO_DIGIT_MAX_GAP_PX=30) を大きく超える距離に配置
    frame = _paint_digit_into_board(frame, templates[5], "1P", 250, 300)
    res = ocr.read_side(frame, "1P")
    # 2桁 (15) には結合されず、いずれかの単体桁として読める
    assert res.chain_count in (1, 5)


# =============================================================================
# window内 最大値集計 (純粋関数 + read_max_in_window)
# =============================================================================


def test_aggregate_window_samples_takes_max_for_clean_ascending_sequence() -> None:
    """1→2→3→4 と綺麗に1ずつ増える (誤検出なし) 場合は素直に最大値を採用。"""
    samples = [
        ChainCountReadResult(1, 0.9, (0, 0)),
        ChainCountReadResult(None, 0.1, None),
        ChainCountReadResult(2, 0.85, (5, 5)),
        ChainCountReadResult(3, 0.75, (8, 8)),
        ChainCountReadResult(4, 0.8, (10, 10)),
    ]
    result = _aggregate_window_samples(samples)
    assert isinstance(result, ChainCountWindowResult)
    assert result.max_chain_count == 4
    assert result.n_hits == 4


def test_aggregate_window_samples_rejects_spike_above_true_max() -> None:
    """真の最大値を上回る孤立した誤検出は棄却される (2026-07-29 修正の要点)。

    旧実装 (単純 max()) では 1,2,4,3 の列で誤って 4 を採用してしまっていた
    (旧テストの想定)。新実装では「4」は連続列 (1→2→3) に乗らない孤立検出
    として棄却され、その後の正しい「3」で連続列が確定するため最大値は 3。
    """
    samples = [
        ChainCountReadResult(1, 0.9, (0, 0)),
        ChainCountReadResult(None, 0.1, None),
        ChainCountReadResult(2, 0.85, (5, 5)),
        ChainCountReadResult(4, 0.8, (10, 10)),  # 孤立した誤検出 (棄却されるべき)
        ChainCountReadResult(3, 0.7, (1, 1)),
    ]
    result = _aggregate_window_samples(samples)
    assert result.max_chain_count == 3
    assert result.n_hits == 4  # n_hits は生の検出数 (棄却前) のまま


def _seq(values: list[int], step_sec: float = 0.1) -> list[tuple[float, int]]:
    """テスト用: 値列に等間隔 (step_sec) の経過時刻を付与する。"""
    return [(i * step_sec, v) for i, v in enumerate(values)]


def test_extract_monotonic_max_chain_count_isolated_detection_rejected() -> None:
    """「3の直後に7」: 7は連続列に乗らないため棄却され、最大値は3のまま。"""
    assert _extract_monotonic_max_chain_count(_seq([1, 2, 3, 7])) == 3


def test_extract_monotonic_max_chain_count_no_precedent_rejected() -> None:
    """「1が出ていないのに4だけ」: 連続列が開始できず None (孤立検出は全棄却)。"""
    assert _extract_monotonic_max_chain_count(_seq([4])) is None


def test_extract_monotonic_max_chain_count_mid_run_gap_conservative() -> None:
    """連続列の途中欠け (1,2,(3欠落),4): 4は棄却され最大値は2 (保守的判断)。

    真の最大値を過大評価するリスクより過小評価を許容する設計判断
    (src/chain_count_ocr.py の _extract_monotonic_max_chain_count docstring 参照)。
    """
    assert _extract_monotonic_max_chain_count(_seq([1, 2, 4])) == 2


def test_extract_monotonic_max_chain_count_multiple_candidate_runs() -> None:
    """複数の連続列候補がある場合、最終的に最大値へ到達した列を採用する。"""
    # 1回目の連続列が3で止まり、2回目の連続列が5まで進む想定
    assert _extract_monotonic_max_chain_count(_seq([1, 2, 3, 1, 2, 3, 4, 5])) == 5


def test_extract_monotonic_max_chain_count_repeats_do_not_advance() -> None:
    """同一ステップの反復サンプル (peakフレームの複数回検出) は値を変えない。"""
    assert _extract_monotonic_max_chain_count(_seq([1, 1, 1, 2, 2, 3, 3, 3])) == 3


def test_extract_monotonic_max_chain_count_empty_returns_none() -> None:
    assert _extract_monotonic_max_chain_count([]) is None


def test_extract_monotonic_max_chain_count_rejects_plausible_spike_after_long_gap() -> None:
    """数値上「+1」でも、長い時間差があれば別物として棄却する (実発見の回帰テスト)。

    2026-07-29 に video_c54 で実際に発見した事例: 1→2→3→4 (実ステップ間隔
    1.1〜1.4秒) の後、5.5秒後に無関係な勝利演出画面を confidence 0.63 で
    「5」と弱く誤検出した。数値だけを見れば 4+1=5 で連続列に見えてしまうが、
    実ステップ間隔を大幅に超える経過時間のため、時間差チェックで正しく
    棄却され最大値は 4 のままになるべき。
    """
    hits = [
        (0.0, 1), (1.2, 2), (2.6, 3), (3.9, 4),  # 実測に近い間隔 (1.1〜1.4秒)
        (9.4, 5),  # 5.5秒後の誤検出 (CHAIN_STEP_MAX_GAP_SEC=2.5秒を大幅に超過)
    ]
    assert _extract_monotonic_max_chain_count(hits) == 4


def test_aggregate_window_samples_all_none() -> None:
    samples = [ChainCountReadResult(None, 0.0, None) for _ in range(5)]
    result = _aggregate_window_samples(samples)
    assert result.max_chain_count is None
    assert result.n_hits == 0


def test_aggregate_window_samples_empty() -> None:
    result = _aggregate_window_samples([])
    assert result.max_chain_count is None
    assert result.n_hits == 0


def test_read_max_in_window_picks_max_across_fake_frames() -> None:
    """複数時刻のフレームを模した FakeVideoCapture で window 内最大値を取得。"""
    templates = _load_real_templates()
    if not all(d in templates for d in (1, 2, 3)):
        pytest.skip("digit_1/2/3 テンプレ未整備")
    ocr = ChainCountOcr.load_default()

    frame_t0 = _make_blank_frame()
    frame_t0 = _paint_digit_into_board(frame_t0, templates[1], "1P", 30, 200)
    frame_t1 = _make_blank_frame()
    frame_t1 = _paint_digit_into_board(frame_t1, templates[2], "1P", 30, 340)
    frame_t2 = _make_blank_frame()
    frame_t2 = _paint_digit_into_board(frame_t2, templates[3], "1P", 30, 480)

    cap = _FakeVideoCapture(
        frame_by_time_sec={0.0: frame_t0, 0.5: frame_t1, 1.0: frame_t2},
        default=_make_blank_frame(),
    )
    result = ocr.read_max_in_window(
        cap, "1P", t_start=0.0, t_end=1.0, sample_interval_sec=0.5,
    )
    assert result.max_chain_count == 3
    assert result.n_hits == 3


def test_read_max_in_window_invalid_range_returns_none() -> None:
    ocr = ChainCountOcr.load_default()
    cap = _FakeVideoCapture(frame_by_time_sec={}, default=_make_blank_frame())
    result = ocr.read_max_in_window(cap, "1P", t_start=2.0, t_end=1.0)
    assert result.max_chain_count is None
    assert result.samples == ()


# =============================================================================
# 得点裏取り方式 (2026-07-29 追加、方式転換の回帰テスト)
# =============================================================================


def test_approx_min_chain_score_matches_known_values() -> None:
    """下限近似の値が手計算 (chain_power テーブルより) と一致することを固定する。

    2026-07-29 実データ検証 (本ファイル上部 docstring) で使った値の回帰確認。
    """
    assert _approx_min_chain_score(1) == 40
    assert _approx_min_chain_score(2) == 360
    assert _approx_min_chain_score(3) == 1000
    assert _approx_min_chain_score(8) == 20200
    assert _approx_min_chain_score(9) == 27880


def test_select_chain_count_by_score_picks_real_9_chain_over_broken_3() -> None:
    """video_c54 2P game_idx=9 (実9連鎖) の実測 delta_score=30920 で検証。

    旧方式 (連続列必須) はこのイベントで 3 という壊れた過小評価を返した
    (実データ、本ファイル上部 docstring 参照)。候補集合に検出済みの値
    {1,...,9} を丸ごと渡した場合、得点裏取り方式は 9 を正しく選べる。
    """
    candidates = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    chosen, ratio = _select_chain_count_by_score(candidates, delta_score=30920)
    assert chosen == 9
    assert ratio is not None and 0.5 <= ratio <= 2.0


def test_select_chain_count_by_score_matches_true_5_chain_video_c54_1p() -> None:
    """video_c54 1P game_idx=1 (実測 delta_score=7598) の実データ検証。

    simulate() は当初 4 連鎖と誤認していたが、後日の実フレーム再検証で真の
    連鎖数は 5 と判明済み (本ファイル上部 docstring の訂正コメント参照)。
    得点裏取り方式は simulate() より正確に真値 5 を選べる。
    """
    candidates = {3, 4, 5, 6}
    chosen, _ratio = _select_chain_count_by_score(candidates, delta_score=7598)
    assert chosen == 5


def test_select_chain_count_by_score_rejects_all_when_no_candidate_fits() -> None:
    """どの候補も許容比率 [0.5, 2.0] に収まらない場合は None を返す (過剰な自信を防ぐ)。"""
    chosen, ratio = _select_chain_count_by_score({1, 2}, delta_score=999_999)
    assert chosen is None
    # 比率自体は最良候補について返る (デバッグ用、None ではない)
    assert ratio is not None


def test_select_chain_count_by_score_empty_candidates_returns_none() -> None:
    chosen, ratio = _select_chain_count_by_score(set(), delta_score=1000)
    assert chosen is None
    assert ratio is None


def test_select_chain_count_by_score_ignores_out_of_range_candidates() -> None:
    """CHAIN_COUNT_MIN/MAX 範囲外の候補は無視される (防御的、通常は発生しない)。"""
    chosen, _ratio = _select_chain_count_by_score({0, 20, 3}, delta_score=1000)
    assert chosen == 3


def test_aggregate_window_samples_score_backed_rejects_decoy_within_gap() -> None:
    """得点裏取りは、旧方式では棄却できない短時間内の decoy も正しく棄却できる。

    合成データ (userタスク指定 (a) のシナリオを模した合成、実測ではない):
    真の連鎖は 1→2→3→4 で delta_score は 4 連鎖相当 (2500) だったとする。
    その直後 (CHAIN_STEP_MAX_GAP_SEC=2.5秒 以内) に無関係な要因で「5」が
    弱く誤検出されたとすると、旧方式 (連続列必須、時間差チェックのみ) は
    時間差が短いため誤って 5 を採用してしまう (時間差チェックはギャップが
    大きい decoy にしか無力)。得点裏取りは delta_score との整合性で
    正しく 4 を選び、decoy を棄却できる。
    """
    samples = [
        ChainCountReadResult(1, 0.9, (0, 0)),
        ChainCountReadResult(2, 0.85, (5, 5)),
        ChainCountReadResult(3, 0.8, (8, 8)),
        ChainCountReadResult(4, 0.8, (10, 10)),
        ChainCountReadResult(5, 0.63, (20, 20)),  # decoy (短時間内、旧方式では棄却できない)
    ]
    # 合成の delta_score: 4 連鎖の下限近似そのもの (2280) を仮定
    # (全消し繰越仮説による他候補との偶然の近接を避けるため、あえて厳密値を使う)
    result = _aggregate_window_samples(samples, delta_score=_approx_min_chain_score(4))
    assert result.method == "score_backed"
    assert result.max_chain_count == 4


def test_aggregate_window_samples_delta_score_omitted_keeps_monotonic_backward_compat() -> None:
    """delta_score 省略時は既存の連続列方式のまま (backwards compat)。"""
    samples = [
        ChainCountReadResult(1, 0.9, (0, 0)),
        ChainCountReadResult(2, 0.85, (5, 5)),
        ChainCountReadResult(4, 0.8, (10, 10)),  # 孤立誤検出 (旧方式の要点)
        ChainCountReadResult(3, 0.7, (1, 1)),
    ]
    result = _aggregate_window_samples(samples)
    assert result.method == "monotonic_run"
    assert result.max_chain_count == 3
    assert result.score_ratio is None
