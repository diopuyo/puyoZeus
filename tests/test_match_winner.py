"""src/match_winner.py のテスト。"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import numpy as np

from src.match_winner import (
    DIGIT_DIFF_HAMMING,
    DIGIT_SAME_HAMMING,
    SIGNATURE_SIZE,
    MatchWinnerDetector,
    compare_digit_pairs,
    digit_ncc,
    digit_signature,
    extract_digit_patches,
    hamming_distance,
    last_winner_by_ncc,
)
from src.win_panel import NUMBER_LEFT_X, NUMBER_RIGHT_X, NUMBER_Y


def _solid_patch(value: int, size: int = 45) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _patch_with_pattern(seed: int, size: int = 45) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_digit_signature_shape() -> None:
    sig = digit_signature(_solid_patch(120))
    assert sig.shape == (SIGNATURE_SIZE * SIGNATURE_SIZE,)
    assert sig.dtype == np.uint8


def test_digit_signature_empty() -> None:
    sig = digit_signature(np.zeros((0, 0, 3), dtype=np.uint8))
    assert sig.shape == (SIGNATURE_SIZE * SIGNATURE_SIZE,)


def test_hamming_distance_identical() -> None:
    p = _patch_with_pattern(42)
    sig = digit_signature(p)
    assert hamming_distance(sig, sig) == 0


def test_hamming_distance_different() -> None:
    p1 = _patch_with_pattern(1)
    p2 = _patch_with_pattern(99)
    d = hamming_distance(digit_signature(p1), digit_signature(p2))
    assert d > DIGIT_DIFF_HAMMING


def test_compare_digit_pairs_left_won() -> None:
    """左 (1P) だけ変わった場合 → 1P 勝利。"""
    left_a = _patch_with_pattern(1)
    right_a = _patch_with_pattern(2)
    left_b = _patch_with_pattern(99)        # 左変化大
    right_b = right_a.copy()                # 右変化なし
    result = compare_digit_pairs(left_a, right_a, left_b, right_b)
    assert result.winner == "1P"
    assert result.left_changed is True
    assert result.right_changed is False


def test_compare_digit_pairs_right_won() -> None:
    """右 (2P) だけ変わった場合 → 2P 勝利。"""
    left_a = _patch_with_pattern(1)
    right_a = _patch_with_pattern(2)
    left_b = left_a.copy()
    right_b = _patch_with_pattern(99)
    result = compare_digit_pairs(left_a, right_a, left_b, right_b)
    assert result.winner == "2P"
    assert result.right_changed is True
    assert result.left_changed is False


def test_compare_digit_pairs_both_unchanged() -> None:
    """両方変化なし → 判定不能。"""
    left_a = _patch_with_pattern(1)
    right_a = _patch_with_pattern(2)
    result = compare_digit_pairs(left_a, right_a, left_a.copy(), right_a.copy())
    assert result.winner is None
    assert result.left_hamming == 0
    assert result.right_hamming == 0


def test_compare_digit_pairs_both_changed() -> None:
    """両方変化 → 判定不能（同時変化はあり得ないので）。"""
    left_a = _patch_with_pattern(1)
    right_a = _patch_with_pattern(2)
    left_b = _patch_with_pattern(99)
    right_b = _patch_with_pattern(77)
    result = compare_digit_pairs(left_a, right_a, left_b, right_b)
    assert result.winner is None


def test_compare_digit_pairs_none_inputs() -> None:
    result = compare_digit_pairs(None, None, None, None)
    assert result.winner is None


def test_extract_digit_patches_correct_size() -> None:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    left, right = extract_digit_patches(frame)
    assert left is not None and right is not None
    # NUMBER_Y=(965, 1010), heights=45; NUMBER_LEFT_X width=60, NUMBER_RIGHT_X width=60
    assert left.shape == (45, 60, 3)
    assert right.shape == (45, 60, 3)


def test_extract_digit_patches_wrong_resolution() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    left, right = extract_digit_patches(frame)
    assert left is None and right is None


# ============================
# 端点修正 (2026-08-19): 最初/最後の試合の勝敗ラベル系統的欠損の修正
# ============================


def _frame_with_digits(
    left_seed: int, right_seed: int, panel_present: bool,
) -> np.ndarray:
    """左右の数値領域に seed 由来パターンを描いた 1080p フレームを作る。

    パネル可視性は frame[0,0,0]==255 のマーカーで表現し、
    _MarkerPanelDetector がそれを読む (テンプレートマッチの代替)。
    """
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for seed, (x0, x1) in ((left_seed, NUMBER_LEFT_X), (right_seed, NUMBER_RIGHT_X)):
        rng = np.random.default_rng(seed)
        frame[NUMBER_Y[0]:NUMBER_Y[1], x0:x1] = rng.integers(
            0, 256, size=(NUMBER_Y[1] - NUMBER_Y[0], x1 - x0, 3), dtype=np.uint8,
        )
    if panel_present:
        frame[0, 0] = 255
    return frame


class _MarkerPanelDetector:
    """frame[0,0,0]==255 をパネル可視とみなす WinPanelDetector フェイク。"""

    def detect(self, frame: np.ndarray) -> SimpleNamespace:
        present = frame is not None and bool(frame[0, 0, 0] == 255)
        return SimpleNamespace(present=present, score=1.0 if present else 0.0)


class _TimelineCap:
    """時刻→フレームの関数で駆動する cv2.VideoCapture フェイク。"""

    def __init__(self, frame_fn: Callable[[float], np.ndarray | None]) -> None:
        self._fn = frame_fn
        self._t = 0.0
        self.read_times: list[float] = []

    def set(self, prop: int, value: float) -> None:
        self._t = value / 1000.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_times.append(self._t)
        frame = self._fn(self._t)
        return (frame is not None), frame


def _make_detector() -> MatchWinnerDetector:
    return MatchWinnerDetector(panel_detector=_MarkerPanelDetector())


def test_detect_all_winners_game0_snaps_forward_over_intro() -> None:
    """動画冒頭のイントロ (パネル不可視) を飛ばして game 0 が判定できること。

    実測 50/50 動画で t=start+1 秒はイントロ映像だったため game 0 の
    ラベルが全滅していた (2026-08-19 修正)。
    """
    # 0-50s: イントロ / 50-100s: game 0 (数値 1,2) / 100-160s: game 1 (99,2)
    # / 160-200s: game 1 リザルト (99,77)
    def frame_fn(t: float) -> np.ndarray:
        if t < 50.0:
            return _frame_with_digits(0, 0, panel_present=False)
        if t < 100.0:
            return _frame_with_digits(1, 2, panel_present=True)
        if t < 160.0:
            return _frame_with_digits(99, 2, panel_present=True)
        return _frame_with_digits(99, 77, panel_present=True)

    det = _make_detector()
    results = det.detect_all_winners(
        _TimelineCap(frame_fn), match_starts=[0.0, 100.0],
        last_observable_sec=200.0,
    )
    assert results[0].winner == "1P"          # 旧実装ではイントロ画像比較で None
    assert results[0].panel_unavailable is False
    assert results[1].winner == "2P"
    assert results[1].panel_unavailable is False


def test_detect_all_winners_game0_unchanged_when_panel_visible_at_start() -> None:
    """開始時点でパネルが映っている動画は従来と同じ読取時刻を使うこと (挙動不変)。"""
    def frame_fn(t: float) -> np.ndarray:
        if t < 100.0:
            return _frame_with_digits(1, 2, panel_present=True)
        return _frame_with_digits(99, 2, panel_present=True)

    cap = _TimelineCap(frame_fn)
    det = _make_detector()
    results = det.detect_all_winners(
        cap, match_starts=[0.0, 100.0], last_observable_sec=150.0,
    )
    # 最初の読取が従来通り match_starts[0]+offset_before=1.0 であること
    assert cap.read_times[0] == 1.0
    assert results[0].winner == "1P"


def test_detect_all_winners_last_game_scanback_beyond_30s() -> None:
    """最終試合終了後のリザルト画面が 30 秒より前でも遡って届くこと。

    旧実装は遡り上限 30 秒でリザルト画面 (勝敗数値の最終値) に届かず、
    実測 47/50 動画で最終試合のラベルが欠損していた (2026-08-19 修正)。
    """
    # 0-140s: game 0 (数値 1,2) / 140-146s: リザルト (55,2) / 146-300s: アウトロ
    def frame_fn(t: float) -> np.ndarray:
        if t < 140.0:
            return _frame_with_digits(1, 2, panel_present=True)
        if t < 146.0:
            return _frame_with_digits(55, 2, panel_present=True)
        return _frame_with_digits(0, 0, panel_present=False)

    det = _make_detector()
    results = det.detect_all_winners(
        _TimelineCap(frame_fn), match_starts=[0.0], last_observable_sec=300.0,
    )
    assert results[0].winner == "1P"
    assert results[0].panel_unavailable is False


def test_detect_all_winners_panel_never_visible_marks_unavailable() -> None:
    """試合開始以降パネルが一度も映らない場合は panel_unavailable=True で
    winner=None (不可視フレームの切り出し画像同士を比較しない)。"""
    blank = _frame_with_digits(0, 0, panel_present=False)

    det = _make_detector()
    results = det.detect_all_winners(
        _TimelineCap(lambda t: blank), match_starts=[100.0],
        last_observable_sec=200.0,
    )
    assert results[0].winner is None
    assert results[0].panel_unavailable is True


def test_detect_all_winners_forward_snap_clamped_by_next_boundary() -> None:
    """game 0 の前方探索は game 1 の読取時刻手前で打ち切ること。

    game 1 の区間まで踏み込んで比較基準にすると誤った勝者が出るため、
    見つからない場合は panel_unavailable=True とする。
    """
    # パネルは t>=30 (game 1 中) から可視。game 0 (0-20s) 中は一度も可視でない
    def frame_fn(t: float) -> np.ndarray:
        if t < 30.0:
            return _frame_with_digits(0, 0, panel_present=False)
        return _frame_with_digits(9, 2, panel_present=True)

    det = _make_detector()
    results = det.detect_all_winners(
        _TimelineCap(frame_fn), match_starts=[0.0, 20.0],
        last_observable_sec=60.0,
    )
    assert results[0].winner is None
    assert results[0].panel_unavailable is True


def test_find_panel_visible_time_respects_not_before() -> None:
    """not_before_sec より過去には遡らないこと (最終試合開始前の数値と
    比較して誤った勝者を出すことの防止)。"""
    # パネルは t<50 でのみ可視 (前試合の数値)
    def frame_fn(t: float) -> np.ndarray:
        return _frame_with_digits(1, 2, panel_present=t < 50.0)

    det = _make_detector()
    t = det._find_panel_visible_time(
        _TimelineCap(frame_fn), around_sec=200.0, not_before_sec=100.0,
    )
    assert t is None


def test_median_digit_patches_removes_moving_effect_noise() -> None:
    """勝利演出の移動エフェクト (フレームごとに位置が変わるノイズ帯) が
    ピクセル中央値合成で除去され、静止した数字が復元されること。

    実測 8/11 動画で、最終試合の終点=勝利演出画面の星エフェクトが左右の
    数字に重なり単一フレーム比較が「両側変化」で判定不能になっていた
    (2026-08-19 追加)。
    """
    clean = _frame_with_digits(55, 2, panel_present=True)

    def frame_fn(t: float) -> np.ndarray:
        # t=143.0..145.0 (0.5 刻み 5 枚) で、数値領域の 1/5 幅ずつ位置の
        # 異なるノイズ帯を重ねる (各ピクセルは 5 枚中 1 枚だけノイズ)
        frame = clean.copy()
        j = int(round((t - 143.0) * 2)) % 5
        rng = np.random.default_rng(1000 + j)
        for x0, x1 in (NUMBER_LEFT_X, NUMBER_RIGHT_X):
            w = (x1 - x0) // 5
            bx0 = x0 + j * w
            frame[NUMBER_Y[0]:NUMBER_Y[1], bx0:bx0 + w] = rng.integers(
                0, 256, size=(NUMBER_Y[1] - NUMBER_Y[0], w, 3), dtype=np.uint8,
            )
        return frame

    det = _make_detector()
    left_med, right_med = det._median_digit_patches(
        _TimelineCap(frame_fn), t_center=145.0, floor_sec=100.0,
    )
    assert left_med is not None and right_med is not None
    clean_left, clean_right = extract_digit_patches(clean)
    dl = hamming_distance(digit_signature(left_med), digit_signature(clean_left))
    dr = hamming_distance(digit_signature(right_med), digit_signature(clean_right))
    # 中央値合成後はノイズなし画像とほぼ同一指紋 (=「変化なし」域)
    assert dl <= DIGIT_SAME_HAMMING
    assert dr <= DIGIT_SAME_HAMMING


def test_detect_all_winners_last_game_with_effect_overlay() -> None:
    """最終試合の終点が勝利演出 (エフェクト重畳) でも勝者判定できること。"""
    game_frame = _frame_with_digits(1, 2, panel_present=True)
    celebration = _frame_with_digits(55, 2, panel_present=True)

    def frame_fn(t: float) -> np.ndarray:
        if t < 140.0:
            return game_frame
        if t < 146.0:
            # 勝利演出: 位置が動くノイズ帯を左右の数字に重ねる
            frame = celebration.copy()
            j = int(round(t * 2)) % 5
            rng = np.random.default_rng(2000 + j)
            for x0, x1 in (NUMBER_LEFT_X, NUMBER_RIGHT_X):
                w = (x1 - x0) // 5
                bx0 = x0 + j * w
                frame[NUMBER_Y[0]:NUMBER_Y[1], bx0:bx0 + w] = rng.integers(
                    0, 256, size=(NUMBER_Y[1] - NUMBER_Y[0], w, 3),
                    dtype=np.uint8,
                )
            return frame
        return _frame_with_digits(0, 0, panel_present=False)

    det = _make_detector()
    results = det.detect_all_winners(
        _TimelineCap(frame_fn), match_starts=[0.0], last_observable_sec=300.0,
    )
    assert results[0].winner == "1P"
    assert results[0].panel_unavailable is False


def test_winner_detection_result_panel_unavailable_default_false() -> None:
    """既存経路 (compare_digit_pairs) の結果は panel_unavailable=False (後方互換)。"""
    result = compare_digit_pairs(None, None, None, None)
    assert result.panel_unavailable is False


# ============================
# NCC 二次判別 (勝利演出のパネル発光対策、2026-08-19)
# ============================


def _brighten(patch: np.ndarray) -> np.ndarray:
    """勝利演出のパネル発光を模した一様な輝度・コントラストシフト。"""
    return np.clip(patch.astype(np.float32) * 0.7 + 60.0, 0, 255).astype(np.uint8)


def _structured_digit(seed: int, size: tuple[int, int] = (45, 60)) -> np.ndarray:
    """数字らしい構造 (粗いブロックパターン) を持つパッチを作る。

    一様乱数だと NCC の分母が画素ノイズに支配されるため、8x8 ブロックの
    構造パターンで「数字の字形」を模す。
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    return cv2_resize_nearest(coarse, size)


def cv2_resize_nearest(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2 as _cv2
    return _cv2.resize(img, (size[1], size[0]), interpolation=_cv2.INTER_NEAREST)


def test_digit_ncc_invariant_to_brightness_shift() -> None:
    """同じ数字は発光 (輝度シフト) 下でも高 NCC を保つこと。"""
    digit = _structured_digit(7)
    assert digit_ncc(digit, _brighten(digit)) > 0.95


def test_last_winner_by_ncc_static_loser_identifies_winner() -> None:
    """敗者側=静止 (発光のみ)、勝者側=別数字 → 勝者を特定できること。"""
    left_a = _structured_digit(1)
    right_a = _structured_digit(2)
    left_b = _brighten(_structured_digit(99))   # 1P 側は数字が変わった (勝者)
    right_b = _brighten(right_a)                # 2P 側は静止 (敗者)
    assert last_winner_by_ncc(left_a, right_a, left_b, right_b) == "1P"


def test_last_winner_by_ncc_both_changed_returns_none() -> None:
    """両側とも数字が変わった (多試合スパン) → None (誤発火しない)。"""
    left_a = _structured_digit(1)
    right_a = _structured_digit(2)
    left_b = _brighten(_structured_digit(99))
    right_b = _brighten(_structured_digit(77))
    assert last_winner_by_ncc(left_a, right_a, left_b, right_b) is None


def test_last_winner_by_ncc_both_static_returns_none() -> None:
    """両側とも静止 (試合が終わっていない) → None (差が小さく不発)。"""
    left_a = _structured_digit(1)
    right_a = _structured_digit(2)
    assert last_winner_by_ncc(
        left_a, right_a, _brighten(left_a), _brighten(right_a),
    ) is None


def test_detect_all_winners_last_game_glow_overlay_rescued_by_ncc() -> None:
    """勝利演出のパネル発光 (両側の指紋ハミングが増大) でも、NCC 二次判別で
    最終試合の勝者を特定できること。"""
    left0, right0 = _structured_digit(11), _structured_digit(12)
    game_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    game_frame[NUMBER_Y[0]:NUMBER_Y[1], NUMBER_LEFT_X[0]:NUMBER_LEFT_X[1]] = left0
    game_frame[NUMBER_Y[0]:NUMBER_Y[1], NUMBER_RIGHT_X[0]:NUMBER_RIGHT_X[1]] = right0
    game_frame[0, 0] = 255
    # 勝利演出: 1P 側は別数字、2P 側は同じ数字が発光 (輝度シフト)
    celeb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    celeb[NUMBER_Y[0]:NUMBER_Y[1], NUMBER_LEFT_X[0]:NUMBER_LEFT_X[1]] = (
        _brighten(_structured_digit(88))
    )
    celeb[NUMBER_Y[0]:NUMBER_Y[1], NUMBER_RIGHT_X[0]:NUMBER_RIGHT_X[1]] = (
        _brighten(right0)
    )
    celeb[0, 0] = 255
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def frame_fn(t: float) -> np.ndarray:
        if t < 140.0:
            return game_frame
        if t < 146.0:
            return celeb
        return blank

    det = _make_detector()
    results = det.detect_all_winners(
        _TimelineCap(frame_fn), match_starts=[0.0], last_observable_sec=300.0,
    )
    assert results[0].winner == "1P"
