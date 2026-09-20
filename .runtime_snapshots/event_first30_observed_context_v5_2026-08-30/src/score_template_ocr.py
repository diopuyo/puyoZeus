"""ぷよぷよ score 用 pixel-level digit テンプレマッチング OCR.

設計思想:
    先行研究 (NES OCR 議論、leshokunin/Video-Game-OCR) の共通報告 = Tesseract は
    game font 精度フラストレーション。**ピクセル単位 digit テンプレートマッチ**
    が静的フォントの数字には最も安定。

    本モジュールは ``src/score_ocr.py`` の NCC 実装をベースに、
    ``recognize_digits(score_roi_bgr, templates) -> (int, list[float])`` という
    薄い public API を提供する。Tesseract 等の OCR engine を呼ぶ既存経路は無いが、
    将来 fallback を差し込めるよう、低 confidence 時に ``fallback_fn`` を呼ぶ
    callable hook 形式にしてある。

主な使い方::

    from src.score_template_ocr import (
        load_default_templates,
        recognize_digits,
    )
    templates = load_default_templates()
    score, per_digit_conf = recognize_digits(score_roi_bgr, templates)

ROI 仕様 (1920x1080 想定):
    - 高さ 65 (DIGIT_TOP+DIGIT_HEIGHT より広い), 幅 320
    - 内部 8 桁が pitch=40px、左上 x=0,40,...,280
    - 桁画像 50x40 (h x w)

戻り値:
    - score: 8 桁全て classify 成功なら ``int`` (0..99_999_999)、失敗時 ``None``
    - per_digit_confidence: 長さ 8 の list[float]、各桁の NCC 最大値 (0..1)
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.score_ocr import (
    DEFAULT_TEMPLATE_DIR,
    DIGIT_COUNT,
    DIGIT_HEIGHT,
    DIGIT_LABELS,
    DIGIT_LEFTS_1P,
    DIGIT_LEFTS_2P,
    DIGIT_TOP,
    DIGIT_WIDTH,
    NCC_MARGIN_MIN,
    NCC_MIN_CONFIDENCE,
    ScoreOcr,
)

# ======================================================================
# 定数
# ======================================================================

# fallback を起動する閾値 (per_digit_confidence の最低値がこれを下回ったら
# fallback_fn を呼ぶ)。``recognize_digits`` の引数から override 可。
DEFAULT_FALLBACK_THRESHOLD: float = 0.70

# ======================================================================
# テンプレロード helper
# ======================================================================


def load_default_templates(
    template_dir: Path = DEFAULT_TEMPLATE_DIR,
) -> dict[int, np.ndarray]:
    """``models/ui_templates/score_digits/digit_N.png`` を読み込む。

    Args:
        template_dir: テンプレ格納ディレクトリ。

    Returns:
        ``{digit: BGR ndarray (50x40)}`` の dict。1 枚も無ければ空 dict。
    """
    templates: dict[int, np.ndarray] = {}
    if not template_dir.is_dir():
        return templates
    for label in DIGIT_LABELS:
        path = template_dir / f"digit_{label}.png"
        if not path.is_file():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        templates[label] = img
    return templates


# ======================================================================
# 桁セグメント
# ======================================================================


def segment_digits(
    score_roi: np.ndarray,
    side: str = "1P",
    digit_count: int = DIGIT_COUNT,
) -> list[tuple[int, int, int, int]]:
    """ROI 内の各桁矩形 ``(x, y, w, h)`` を返す.

    現状は固定座標 (DIGIT_LEFTS_1P / DIGIT_LEFTS_2P, DIGIT_TOP/HEIGHT/WIDTH) を
    そのまま返す stateless 実装。動的検出は将来差し替え対象。

    Args:
        score_roi: 65x320 (or それ以上) の BGR ROI.
        side: ``"1P"`` または ``"2P"``。1P/2P で同じ pitch だが将来差分対応用。
        digit_count: 切り出す桁数 (デフォルト 8)。``len(DIGIT_LEFTS_*)`` を超えると
            ValueError。

    Returns:
        ``[(x, y, w, h), ...]``。座標は ROI 左上を原点とする整数。
    """
    lefts = DIGIT_LEFTS_1P if side == "1P" else DIGIT_LEFTS_2P
    if digit_count < 0 or digit_count > len(lefts):
        raise ValueError(
            f"digit_count must be in [0, {len(lefts)}] (got {digit_count})"
        )
    return [
        (int(lefts[i]), int(DIGIT_TOP), int(DIGIT_WIDTH), int(DIGIT_HEIGHT))
        for i in range(digit_count)
    ]


# ======================================================================
# 桁分類 (NCC)
# ======================================================================


def _to_gray(img: np.ndarray) -> np.ndarray:
    """BGR → gray 変換。grayscale 入力はそのまま。"""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _normalize_template_set(
    templates: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """テンプレを (DIGIT_HEIGHT, DIGIT_WIDTH) gray に揃える."""
    out: dict[int, np.ndarray] = {}
    for label, tpl in templates.items():
        if tpl is None:
            continue
        gray = _to_gray(tpl)
        if gray.shape != (DIGIT_HEIGHT, DIGIT_WIDTH):
            gray = cv2.resize(
                gray,
                (DIGIT_WIDTH, DIGIT_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
        out[int(label)] = gray
    return out


def classify_digit_cell(
    cell: np.ndarray,
    templates_gray: dict[int, np.ndarray],
    min_confidence: float = NCC_MIN_CONFIDENCE,
    margin_min: float = NCC_MARGIN_MIN,
) -> tuple[int | None, float]:
    """1 桁分のセル画像 (50x40) を 0..9 に分類.

    Args:
        cell: 50x40 (or 任意 → resize) の BGR/grayscale.
        templates_gray: ``_normalize_template_set`` 済みの dict.
        min_confidence: NCC 最大値の下限。これ未満なら ``None``.
        margin_min: 1 位 - 2 位 の差。これ未満なら ``None`` (曖昧)。

    Returns:
        ``(label or None, best_ncc)``.
    """
    if not templates_gray:
        return None, 0.0
    gray = _to_gray(cell)
    if gray.shape != (DIGIT_HEIGHT, DIGIT_WIDTH):
        gray = cv2.resize(
            gray, (DIGIT_WIDTH, DIGIT_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
    scores: dict[int, float] = {}
    for label, tpl in templates_gray.items():
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        scores[label] = float(res.max())
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    if best_score < min_confidence:
        return None, best_score
    if best_score - second_score < margin_min:
        return None, best_score
    return best_label, best_score


# ======================================================================
# 公開 API: recognize_digits
# ======================================================================


# fallback_fn 型: (score_roi_bgr, segments, per_digit_conf) ->
#     (score, per_digit_confidence) または None で失敗扱い
FallbackFn = Callable[
    [np.ndarray, list[tuple[int, int, int, int]], list[float]],
    "tuple[int | None, list[float]] | None",
]


def recognize_digits(
    score_roi_bgr: np.ndarray,
    templates: dict[int, np.ndarray],
    side: str = "1P",
    digit_count: int = DIGIT_COUNT,
    min_confidence: float = NCC_MIN_CONFIDENCE,
    margin_min: float = NCC_MARGIN_MIN,
    fallback_threshold: float = DEFAULT_FALLBACK_THRESHOLD,
    fallback_fn: FallbackFn | None = None,
) -> tuple[int | None, list[float]]:
    """score ROI 全体を pixel-level template matching で読み取る.

    手順:
        1. ``segment_digits`` で 8 桁矩形を取得
        2. 各セルを ``classify_digit_cell`` で 0..9 (or None) 判定
        3. 全桁成功 + 最低 confidence >= fallback_threshold なら確定
        4. それ以外は ``fallback_fn`` (与えられていれば) を呼び、その結果を返す

    Args:
        score_roi_bgr: 8 桁数字の含まれる BGR ROI (高さ >= DIGIT_TOP+DIGIT_HEIGHT,
            幅 >= 各桁配置に十分)。
        templates: ``{0..9: BGR ndarray}`` 形式のテンプレ辞書。
            ``load_default_templates()`` で取得可。
        side: 桁配置決定 (``"1P"`` / ``"2P"``)。
        digit_count: 読み取る桁数 (default 8)。
        min_confidence: 各桁 NCC 下限。
        margin_min: 1 位と 2 位の差の下限。
        fallback_threshold: 全桁 NCC の最小値がこれを下回ったら fallback 起動。
        fallback_fn: ``(roi, segments, per_digit_conf)`` を取って結果 or None を返す
            callable。``None`` なら fallback せず ``(None, per_digit_conf)``.

    Returns:
        ``(score, per_digit_confidence)``.
        - 成功: ``score`` は ``0..10**digit_count - 1`` の int.
        - 失敗: ``score`` は ``None``、``per_digit_confidence`` は長さ
          ``digit_count`` の float リスト (各桁 NCC 最大値、未試行時 0.0)。
    """
    empty_conf = [0.0] * digit_count
    if score_roi_bgr is None or score_roi_bgr.size == 0:
        return None, empty_conf
    segments = segment_digits(
        score_roi_bgr, side=side, digit_count=digit_count,
    )
    templates_gray = _normalize_template_set(templates)
    if not templates_gray:
        return None, empty_conf

    digits: list[int | None] = []
    confidences: list[float] = []
    for (x, y, w, h) in segments:
        cell = score_roi_bgr[y:y + h, x:x + w]
        label, conf = classify_digit_cell(
            cell, templates_gray,
            min_confidence=min_confidence,
            margin_min=margin_min,
        )
        digits.append(label)
        confidences.append(conf)

    all_classified = all(d is not None for d in digits)
    min_conf = min(confidences) if confidences else 0.0

    if all_classified and min_conf >= fallback_threshold:
        score = 0
        for d in digits:
            assert d is not None
            score = score * 10 + d
        return score, confidences

    # fallback ルート
    if fallback_fn is not None:
        result = fallback_fn(score_roi_bgr, segments, list(confidences))
        if result is not None:
            return result
    # fallback なし or fallback も失敗 → None
    if all_classified:
        # 全桁 classify 成功だが confidence 不足 → score は捨てる
        return None, confidences
    return None, confidences


# ======================================================================
# 既存 ScoreOcr とのブリッジ (推奨経路)
# ======================================================================


def make_score_ocr(
    template_dir: Path = DEFAULT_TEMPLATE_DIR,
    use_template_ocr: bool = True,
) -> ScoreOcr:
    """既存 ``ScoreOcr`` インスタンスを作る薄い wrapper.

    現在の ``ScoreOcr`` は既に NCC テンプレマッチング主軸 (Tesseract 不使用)。
    ``use_template_ocr`` flag は将来 OCR engine を切替可能にした際の hook。
    現状 ``False`` 指定時は ``NotImplementedError``.

    Args:
        template_dir: digit テンプレ格納先。
        use_template_ocr: 現状 True 必須。

    Returns:
        初期化済 ``ScoreOcr``.
    """
    if not use_template_ocr:
        raise NotImplementedError(
            "Non-template OCR engine (Tesseract 等) は本プロジェクトでは未実装。"
            " 将来 fallback として追加予定。"
        )
    return ScoreOcr.load_default(template_dir=template_dir)


__all__ = [
    "DEFAULT_FALLBACK_THRESHOLD",
    "FallbackFn",
    "classify_digit_cell",
    "load_default_templates",
    "make_score_ocr",
    "recognize_digits",
    "segment_digits",
]
