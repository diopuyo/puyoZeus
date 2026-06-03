"""
ぷよぷよ score 数値 OCR モジュール (NCC テンプレマッチ).

原理:
    画面下部の 8 桁ゼロ埋め score 表示 (1P / 2P) を、各桁ごとに 0-9 の
    テンプレートと NCC マッチングで読み取る。連鎖前後の score 差分を
    取れば、落下ボーナス・全消しボーナス込みで「実際にプレイヤーが
    獲得した score」が分かるため、得点 → おじゃまぷよ換算が直接可能。

ROI:
    - 1P: y=890..955, x=200..680 (1920x1080 想定)
    - 2P: y=890..955, x=1260..1740
    - 各 ROI は 65x480、内部に 8 桁の数字が pitch=40px で並ぶ
    - 桁の左上 x 座標 (ROI 内): [135,175,215,255,295,335,375,415]
    - 桁画像サイズ: 50x40 (h x w)、ROI 内 y=8..58

テンプレ:
    - models/ui_templates/score_digits/digit_0.png .. digit_9.png
    - 各テンプレは 50x40 (h x w) の BGR 画像
    - 不在クラスは OCR 失敗 (None) として扱う

使い方:
    ocr = ScoreOcr.load_default()
    res = ocr.read(frame_1080p)
    print(res.score_1p, res.score_2p)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

# ============================
# 定数 (1920x1080 前提)
# ============================

DIGIT_COUNT: int = 8

# score ROI (y1, y2, x1, x2)
# 1920x1080 想定。8 桁の 0 が確実に並ぶ範囲を採用。
# 注意: 2026-04-27 にレビューで 0.5 文字分 (20px) 右にシフト。
# 旧: 1P=(335,655), 2P=(1233,1553) → 新: +20px
# 1P: 数字左上 x=355..635 (8桁、ピッチ 40)、ROI は数字 8 桁分のみ
# 2P: 数字左上 x=1253..1533 (8桁、ピッチ 40)
SCORE_1P_REGION: tuple[int, int, int, int] = (890, 955, 355, 675)
SCORE_2P_REGION: tuple[int, int, int, int] = (890, 955, 1253, 1573)

# ROI 内の各桁切り出し範囲 (左上原点)
DIGIT_TOP: int = 8
DIGIT_HEIGHT: int = 50
DIGIT_WIDTH: int = 40
# ピッチ 40px、8 桁分、ROI 内座標で 0 から始まる
DIGIT_LEFTS_1P: tuple[int, ...] = (0, 40, 80, 120, 160, 200, 240, 280)
DIGIT_LEFTS_2P: tuple[int, ...] = (0, 40, 80, 120, 160, 200, 240, 280)
# 互換用 (旧コード対応)
DIGIT_LEFTS: tuple[int, ...] = DIGIT_LEFTS_1P

# テンプレ格納先
DEFAULT_TEMPLATE_DIR: Path = Path("models/ui_templates/score_digits")

# OCR 信頼度閾値: NCC が この値未満なら 「読めない」と判定
# 2026-04-27: 連鎖中の計算式表示で偶然 8 桁読めてしまうケースを排除するため
# 0.45 → 0.55 に引き上げ。
NCC_MIN_CONFIDENCE: float = 0.55

# 全桁平均 NCC の下限。これ未満なら 8 桁揃っていても無効と判定。
# 連鎖中の計算式表示は「+1240」等で各桁の NCC が下がるため、平均で弾く。
NCC_AVG_MIN_CONFIDENCE: float = 0.65

# 信頼度判定: 1 位と 2 位の NCC スコア差。これより小さいと曖昧と判定
NCC_MARGIN_MIN: float = 0.04

# テンプレ内の「数字部分」を抽出するための輝度しきい値
DIGIT_MASK_THRESHOLD: int = 180

# 入力フレームの想定サイズ
EXPECTED_FRAME_SHAPE: tuple[int, int] = (1080, 1920)

# 桁ラベル
DIGIT_LABELS: tuple[int, ...] = tuple(range(10))

# サイド指定
Side = Literal["1P", "2P"]


# ============================
# 結果データクラス
# ============================


@dataclass(frozen=True)
class ScoreReadResult:
    """ScoreOcr.read() の結果。

    Attributes:
        score_1p: 1P スコア値 (8 桁全て読めた時のみ非 None)。
        score_2p: 2P スコア値。
        confidence_1p: 1P 全桁の最低 NCC 信頼度 (0.0..1.0)、失敗時 0.0。
        confidence_2p: 2P 全桁の最低 NCC 信頼度。
        digits_1p: 1P 各桁の読取値 (None=失敗)。長さ 8。
        digits_2p: 2P 各桁の読取値。長さ 8。
    """

    score_1p: int | None
    score_2p: int | None
    confidence_1p: float
    confidence_2p: float
    digits_1p: tuple[int | None, ...]
    digits_2p: tuple[int | None, ...]


# ============================
# 内部ヘルパ
# ============================


def _ensure_1080p(frame: np.ndarray) -> np.ndarray | None:
    """1080p 以外なら resize、サイズ 0 や ndim 不正は None。"""
    if frame is None or frame.ndim != 3:
        return None
    h, w = frame.shape[:2]
    if (h, w) == EXPECTED_FRAME_SHAPE:
        return frame
    if h < 100 or w < 100:
        return None
    return cv2.resize(frame, (EXPECTED_FRAME_SHAPE[1], EXPECTED_FRAME_SHAPE[0]),
                      interpolation=cv2.INTER_AREA)


def _crop_score_roi(frame: np.ndarray, side: Side) -> np.ndarray | None:
    """フレームから 1P/2P score ROI (65x480) を切り出す。"""
    region = SCORE_1P_REGION if side == "1P" else SCORE_2P_REGION
    y1, y2, x1, x2 = region
    if y2 > frame.shape[0] or x2 > frame.shape[1]:
        return None
    return frame[y1:y2, x1:x2].copy()


def _crop_digit_cell(roi: np.ndarray, idx: int, side: Side = "1P") -> np.ndarray:
    """ROI 内の i 番目の桁セル (50x40) を切り出す。"""
    lefts = DIGIT_LEFTS_1P if side == "1P" else DIGIT_LEFTS_2P
    x = lefts[idx]
    return roi[DIGIT_TOP:DIGIT_TOP + DIGIT_HEIGHT, x:x + DIGIT_WIDTH]


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ============================
# ScoreOcr 本体
# ============================


class ScoreOcr:
    """8 桁 score 数値の OCR 器。

    各桁セル (50x40) に対し 0-9 のテンプレを NCC でマッチし、最大スコアの
    クラスを採用する。テンプレ未整備のクラスは欠損扱い (None)。
    """

    DIGIT_COUNT: int = DIGIT_COUNT

    def __init__(
        self,
        templates: dict[int, np.ndarray] | None = None,
        min_confidence: float = NCC_MIN_CONFIDENCE,
        margin_min: float = NCC_MARGIN_MIN,
        avg_min_confidence: float = NCC_AVG_MIN_CONFIDENCE,
    ) -> None:
        """Args:
            templates: 0-9 → テンプレ画像 (50x40 BGR or grayscale) の辞書。
                未指定 (None) クラスは OCR で None 扱い。
            min_confidence: 各桁の NCC 最低スコア。これ未満なら None。
            margin_min: 1 位と 2 位のスコア差。これ未満なら曖昧と見て None。
        """
        self._templates_gray: dict[int, np.ndarray] = {}
        # 数字部分のマスク (背景を NCC から除外して識別力を上げるため)
        self._templates_mask: dict[int, np.ndarray] = {}
        if templates:
            for label, tpl in templates.items():
                if tpl is None:
                    continue
                gray = _to_gray(tpl)
                # 期待サイズに合わせる
                if gray.shape != (DIGIT_HEIGHT, DIGIT_WIDTH):
                    gray = cv2.resize(gray, (DIGIT_WIDTH, DIGIT_HEIGHT),
                                       interpolation=cv2.INTER_AREA)
                self._templates_gray[int(label)] = gray
                # 数字 (白っぽい) 部分を 255、背景を 0
                mask = (gray > DIGIT_MASK_THRESHOLD).astype(np.uint8) * 255
                self._templates_mask[int(label)] = mask
        self._min_confidence = float(min_confidence)
        self._margin_min = float(margin_min)
        self._avg_min_confidence = float(avg_min_confidence)
        # 警告は 1 度だけ出す
        self._warned_missing = False
        if not self._templates_gray:
            print("[score_ocr] WARNING: テンプレが 1 つも登録されていない。OCR は常に None を返す。")
            self._warned_missing = True
        else:
            missing = sorted(set(DIGIT_LABELS) - set(self._templates_gray.keys()))
            if missing and not self._warned_missing:
                print(f"[score_ocr] WARNING: 未整備クラス {missing} は OCR で None 扱いになる。")
                self._warned_missing = True

    # ============================
    # クラス生成
    # ============================

    @classmethod
    def load_default(
        cls,
        template_dir: Path = DEFAULT_TEMPLATE_DIR,
        min_confidence: float = NCC_MIN_CONFIDENCE,
        margin_min: float = NCC_MARGIN_MIN,
        avg_min_confidence: float = NCC_AVG_MIN_CONFIDENCE,
    ) -> "ScoreOcr":
        """models/ui_templates/score_digits/ から digit_N.png を読み込む。"""
        templates = cls._load_templates_from_dir(template_dir)
        return cls(
            templates=templates,
            min_confidence=min_confidence,
            margin_min=margin_min,
            avg_min_confidence=avg_min_confidence,
        )

    @staticmethod
    def _load_templates_from_dir(template_dir: Path) -> dict[int, np.ndarray]:
        """digit_N.png を全部スキャンする (N=0..9)。"""
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

    # ============================
    # 公開: 読取り
    # ============================

    def read(self, frame: np.ndarray) -> ScoreReadResult:
        """1920x1080 BGR フレームから 1P/2P スコアを読み取る。"""
        f = _ensure_1080p(frame)
        if f is None:
            empty: tuple[int | None, ...] = (None,) * DIGIT_COUNT
            return ScoreReadResult(None, None, 0.0, 0.0, empty, empty)

        score_1p, conf_1p, digits_1p = self._read_one_side(f, "1P")
        score_2p, conf_2p, digits_2p = self._read_one_side(f, "2P")
        return ScoreReadResult(
            score_1p=score_1p,
            score_2p=score_2p,
            confidence_1p=conf_1p,
            confidence_2p=conf_2p,
            digits_1p=digits_1p,
            digits_2p=digits_2p,
        )

    def read_side(
        self, frame: np.ndarray, side: Side
    ) -> tuple[int | None, float]:
        """指定サイドのみ読み取る簡易版。"""
        f = _ensure_1080p(frame)
        if f is None:
            return None, 0.0
        score, conf, _ = self._read_one_side(f, side)
        return score, conf

    def read_with_neighbor_search(
        self,
        cap: "cv2.VideoCapture",
        t_sec: float,
        search_radius_sec: float = 0.3,
        n_samples: int = 5,
    ) -> ScoreReadResult:
        """指定時刻 t_sec の周辺 ±search_radius_sec で複数フレーム探索し、
        最高 confidence の結果を返す。

        連鎖中アニメや一瞬の表示崩れで t_sec ピンポイントで読めない場合に、
        readable 率を向上させる。

        Args:
            cap: cv2.VideoCapture
            t_sec: 中心時刻
            search_radius_sec: 探索半径 (左右に同じ距離)
            n_samples: 探索フレーム数 (奇数推奨、中心 + 左右)

        Returns:
            最良の ScoreReadResult (1P/2P 両方読めたフレームが最優先)。
        """
        if n_samples < 1:
            n_samples = 1
        # 探索時刻: t-radius, ..., t, ..., t+radius を等間隔
        if n_samples == 1:
            offsets = [0.0]
        else:
            step = (2.0 * search_radius_sec) / (n_samples - 1)
            offsets = [-search_radius_sec + i * step for i in range(n_samples)]
        # 中心を最初に評価して、すぐ高 conf 取れたら早期 return
        offsets.sort(key=lambda o: abs(o))

        best: ScoreReadResult | None = None
        best_score: float = -1.0
        empty: tuple[int | None, ...] = (None,) * DIGIT_COUNT
        for off in offsets:
            cap.set(cv2.CAP_PROP_POS_MSEC, (t_sec + off) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            res = self.read(frame)
            # 両側読めたら優先 (合算 conf)
            both_ok = (res.score_1p is not None
                       and res.score_2p is not None)
            single_ok = (res.score_1p is not None
                         or res.score_2p is not None)
            # スコア (両側読み = +1.0、片側読み = +0.5、conf 平均加算)
            rank = (
                (1.0 if both_ok else (0.5 if single_ok else 0.0))
                + (res.confidence_1p + res.confidence_2p) / 2.0
            )
            if rank > best_score:
                best_score = rank
                best = res
            # 中心で両側 conf 高いなら早期終了 (高速化)
            if off == 0.0 and both_ok and res.confidence_1p > 0.85 \
                    and res.confidence_2p > 0.85:
                return res
        if best is None:
            return ScoreReadResult(None, None, 0.0, 0.0, empty, empty)
        return best

    # ============================
    # 内部: 1 サイドの読取
    # ============================

    def _read_one_side(
        self, frame: np.ndarray, side: Side
    ) -> tuple[int | None, float, tuple[int | None, ...]]:
        roi = _crop_score_roi(frame, side)
        if roi is None or roi.size == 0:
            empty: tuple[int | None, ...] = (None,) * DIGIT_COUNT
            return None, 0.0, empty
        digits: list[int | None] = []
        confidences: list[float] = []
        for i in range(DIGIT_COUNT):
            cell = _crop_digit_cell(roi, i, side)
            label, conf = self._classify_digit(cell)
            digits.append(label)
            confidences.append(conf)
        digits_t: tuple[int | None, ...] = tuple(digits)
        # 全桁が読めた時のみ score を確定
        if any(d is None for d in digits):
            min_conf = float(min(confidences)) if confidences else 0.0
            return None, min_conf, digits_t
        # 平均 confidence チェック: 連鎖中の計算式表示で偶然 8 桁読めるケースを排除
        avg_conf = float(sum(confidences) / len(confidences))
        if avg_conf < self._avg_min_confidence:
            return None, avg_conf, digits_t
        score = 0
        for d in digits:
            assert d is not None
            score = score * 10 + d
        min_conf = float(min(confidences))
        return score, min_conf, digits_t

    def _classify_digit(self, cell: np.ndarray) -> tuple[int | None, float]:
        """セル画像 (50x40) から 0-9 を NCC で判定。失敗時 (None, 信頼度)。

        2 段階判定:
            1. 通常 NCC で全クラスのスコアを取得 (安定性重視)
            2. 1 位と 2 位が接近している場合は数字マスク内 SAD で再判定 (識別力重視)
        """
        if not self._templates_gray:
            return None, 0.0
        gray = _to_gray(cell)
        if gray.shape != (DIGIT_HEIGHT, DIGIT_WIDTH):
            gray = cv2.resize(gray, (DIGIT_WIDTH, DIGIT_HEIGHT),
                               interpolation=cv2.INTER_AREA)
        scores: dict[int, float] = {}
        for label, tpl in self._templates_gray.items():
            res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            scores[label] = float(res.max())
        # 最大スコア
        sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_score = sorted_scores[0]
        second_label, second_score = (
            sorted_scores[1] if len(sorted_scores) > 1 else (None, 0.0)
        )

        # 信頼度フィルタ (最低スコア)
        if best_score < self._min_confidence:
            return None, best_score

        # 1 位と 2 位が拮抗 (margin 未満) ならマスク内 SAD で再判定
        if (
            second_label is not None
            and best_score - second_score < self._margin_min
        ):
            tiebreak = self._tiebreak_by_mask_sad(gray, best_label, second_label)
            if tiebreak is None:
                # タイブレイクしても決まらない場合は曖昧で None
                return None, best_score
            return tiebreak, best_score
        return best_label, best_score

    def _tiebreak_by_mask_sad(
        self, gray: np.ndarray, label_a: int, label_b: int
    ) -> int | None:
        """2 候補を「数字マスク内 SAD」 で比較してより近い方を返す。

        マスク = テンプレ内で輝度が高い (数字筆跡) 部分のみ。
        """
        tpl_a = self._templates_gray.get(label_a)
        tpl_b = self._templates_gray.get(label_b)
        mask_a = self._templates_mask.get(label_a)
        mask_b = self._templates_mask.get(label_b)
        if tpl_a is None or tpl_b is None or mask_a is None or mask_b is None:
            return None
        # マスク内の絶対差分平均 (低い方が一致)
        # gray と tpl が同サイズである前提
        if gray.shape != tpl_a.shape:
            return None
        diff_a = np.abs(gray.astype(np.int32) - tpl_a.astype(np.int32))
        diff_b = np.abs(gray.astype(np.int32) - tpl_b.astype(np.int32))
        # マスクは 255/0 の uint8、そのまま重みとして使う
        sad_a = float((diff_a * (mask_a > 0)).sum() / max(1, (mask_a > 0).sum()))
        sad_b = float((diff_b * (mask_b > 0)).sum() / max(1, (mask_b > 0).sum()))
        # 差が小さい (<= 2) なら判定不能
        if abs(sad_a - sad_b) < 2.0:
            return None
        return label_a if sad_a < sad_b else label_b


# ============================
# モジュール公開定数
# ============================

# ============================
# 機能D: 連鎖開始 掛け算式 検知ヘルパ
# ============================

# ink_ratio 輝度閾値: この値より高い輝度ピクセルを「描画あり」とみなす。
# メニュー/試合外の真黒 ROI (輝度≈0) を除外するための下限。
# 通常スコア表示・掛け算式表示はいずれも ink_ratio が高いため、
# ink_ratio は「黒 ROI 除外」専用で formula vs 通常数字の区別は score=None が担う。
# 実測: formula(ink=0.98-1.00), 通常(ink=1.00), 真黒(ink=0.00)。
# 閾値 10 で真黒(輝度≤5)のみ 0 になり、それ以外は全て 1.0 付近になる。
SCORE_ROI_INK_THRESHOLD: int = 10

# ink_ratio の最低値。これ未満 = ROI が真黒 = メニュー/試合外と判定し発火除外。
# 実データ: formula=0.975-1.000、通常=1.000、真黒=0.000。
# 0.1 は保守的な下限として設定 (十分なマージンを確保)。
SCORE_ROI_INK_RATIO_MIN: float = 0.1


def compute_score_roi_ink_ratio(roi: np.ndarray) -> float:
    """score ROI の ink_ratio (描画ピクセル割合) を計算する。

    ink_ratio = 輝度 > SCORE_ROI_INK_THRESHOLD のピクセル数 / 全ピクセル数。
    メニュー/試合外の真黒 ROI (ink_ratio≈0) を連鎖発火から除外するための
    ガード信号。formula 表示・通常スコア表示はいずれも高い値 (≈1.0) になる。

    Args:
        roi: score ROI の BGR or grayscale 画像。

    Returns:
        0.0〜1.0 の ink_ratio。ROI が None/空の場合は 0.0。
    """
    if roi is None or roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    return float((gray > SCORE_ROI_INK_THRESHOLD).sum()) / max(1, gray.size)


# ============================
# 連続 frame 用 Tracker (Phase B-4)
# ============================


@dataclass
class ScoreDelta:
    """連続 frame の score 差分情報 (1 サイド分).

    Attributes:
        side: "1P" or "2P"
        prev_score: 直前 frame で読めた score (None なら未確定)
        cur_score: 現 frame で読めた score (None なら未確定)
        delta: cur - prev、いずれか None なら 0
        is_valid: prev/cur 両方読めて差分が非負か
    """

    side: Side
    prev_score: int | None
    cur_score: int | None
    delta: int

    @property
    def is_valid(self) -> bool:
        return (
            self.prev_score is not None
            and self.cur_score is not None
            and self.delta >= 0
        )


class ScoreTracker:
    """1 サイド score の連続 frame 監視 + 差分計算.

    BoardStateMachine の DetectorSignals.score_delta を生成する用途。
    連鎖完了時に大きな delta が出るので、OjamaPhaseDetector の発火条件に使う。

    Usage:
        tr = ScoreTracker(side="1P", ocr=ScoreOcr.load_default())
        for frame in stream:
            d = tr.update(frame)
            if d.is_valid and d.delta > 0:
                # 連鎖完了 → おじゃま予告
                ...
    """

    def __init__(self, side: Side, ocr: ScoreOcr) -> None:
        if side not in ("1P", "2P"):
            raise ValueError(f"side must be '1P' or '2P' (got {side!r})")
        self._side: Side = side
        self._ocr = ocr
        self._last_score: int | None = None

    @property
    def side(self) -> Side:
        return self._side

    @property
    def last_score(self) -> int | None:
        return self._last_score

    def update(self, frame: np.ndarray) -> ScoreDelta:
        cur, _conf = self._ocr.read_side(frame, self._side)
        prev = self._last_score
        if cur is not None and prev is not None:
            delta = cur - prev
        else:
            delta = 0
        if cur is not None:
            self._last_score = cur
        return ScoreDelta(
            side=self._side,
            prev_score=prev,
            cur_score=cur,
            delta=delta,
        )

    def reset(self) -> None:
        """試合切替時など、最終 score をクリアする."""
        self._last_score = None


# ============================
# モジュール公開定数
# ============================

__all__ = [
    "DIGIT_COUNT",
    "DIGIT_HEIGHT",
    "DIGIT_LEFTS",
    "DIGIT_LEFTS_1P",
    "DIGIT_LEFTS_2P",
    "DIGIT_MASK_THRESHOLD",
    "DIGIT_TOP",
    "DIGIT_WIDTH",
    "EXPECTED_FRAME_SHAPE",
    "NCC_AVG_MIN_CONFIDENCE",
    "NCC_MARGIN_MIN",
    "NCC_MIN_CONFIDENCE",
    "SCORE_1P_REGION",
    "SCORE_2P_REGION",
    "SCORE_ROI_INK_RATIO_MIN",
    "SCORE_ROI_INK_THRESHOLD",
    "ScoreDelta",
    "ScoreOcr",
    "ScoreReadResult",
    "ScoreTracker",
    "compute_score_roi_ink_ratio",
]
