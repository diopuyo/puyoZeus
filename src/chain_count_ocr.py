"""ぷよぷよ 画面中央「N れんさ!」表示の連鎖数 OCR モジュール (NCC テンプレマッチ)。

## 背景 (userタスク指定 2026-07-29)

得点整合性チェック (src.scoring.score_consistency_ratio) を 514 イベントに
適用した結果、40.3% (207件) が不整合と判明した。不整合の主因は
`simulate(before_grid)` が使う「連鎖直前の静止盤面 (起点)」の認識誤りで、
連鎖数を大きく取り違える (例: 本当は 4 連鎖なのに simulate() は 1 連鎖と誤認)。

画面中央には連鎖ステップが進むたびに「1 れんさ!」→「2 れんさ!」→…と
ポップアップ表示が出る (掛け算表示とは別物、連鎖数そのものを直接表示する)。
この表示の最大値を読み取れば、simulate() の誤りに影響されない「真の連鎖数」が
得られる。

## 表示仕様 (実フレームで確認済み: video_c54.mp4, 1P側, t=252.6〜256.3秒)

- 連鎖 1 ステップごとに「N れんさ!」がポップアップし、出現から消滅まで
  約 0.6〜0.7 秒 (60fps 実測) 表示される。フェード/縮小アニメーションあり。
- **表示位置は固定ではない**。そのステップで消えたぷよ群の付近 (盤面内) に
  出現するため、ステップが進むごとに位置が変わる。
  実測 (1920x1080 絶対座標、1P盤面内、数字グリフのみの tight crop 基準):
    1連鎖目: y≈363-473, x≈340-398 (58x110)
    2連鎖目: y≈500-610, x≈385-455 (70x110)
    3連鎖目: y≈665-775, x≈450-520 (70x110)
    4連鎖目: y≈735-845, x≈400-473 (73x110)
  → 固定の小さな ROI (score_ocr の桁セルのような) では捕捉できない。
    盤面全体 (DEFAULT_P1_REGION / DEFAULT_P2_REGION) を検索範囲とし、
    数字テンプレを matchTemplate でスキャンして最良一致位置を探す。
- フォントは src/score_ocr.py の score 桁 (白ゴシック体・固定ピッチ) とは
  異なる (黄〜オレンジのグラデーション・黒縁取りの装飾数字)。専用テンプレが
  必要 (models/ui_templates/chain_count_digits/digit_N.png)。
- 掛け算表示 (score 直上に出る "40× 16" 等) とは別物。今回は扱わない。

## 流用元

src/score_ocr.py の NCC (cv2.matchTemplate + TM_CCOEFF_NORMED) 方式を流用。
score OCR は「固定セル」に対して数字クラスを分類するが、本モジュールは表示
位置が可変なため「盤面全体」に対してテンプレをスキャンし、最良一致位置と
スコアを採用する (matchTemplate 自体が 2D 相関マップを返す性質を利用)。

## 既知の制約 (2026-07-29 時点、正直に明記)

- **2 桁の連鎖数 (10連鎖以上) は未対応**。テンプレ・検出ロジックとも 1 桁前提。
  10連鎖以上の局面では検出できず None を返す (要将来拡張)。
- **digit_5〜digit_9 のテンプレ画像は本タスク時点で未採取** (video_c54 では
  1〜4連鎖しか実フレームで確認できなかったため)。未整備クラスは ScoreOcr と
  同様 OCR で None 扱いになる (欠損クラスは無視され警告ログを出す)。
  全動画への適用検証時に、より長い連鎖が映る動画から追加採取が必要。
- **2P側の表示位置・フォントは 1P側と同一という前提**。同一エンジン描画のため
  同一と推測されるが、実フレームでの確認はまだ行っていない。
- テンプレは手動crop (背景の隣接ぷよが一部写り込む) のため、score_ocr の
  digit テンプレほど背景ノイズを除去できていない。tight crop (数字グリフ
  中心、背景余白を最小化) により誤検出はある程度抑えたが、完全排除はできて
  いない (下記閾値の根拠を参照)。
- **色ベース mask + TM_CCORR_NORMED は不採用 (実験して悪化を確認済み)**。
  数字部分を HSV 色域 (H:5-35, S:80-255, V:80-255) でマスクし
  cv2.matchTemplate(..., mask=mask) で相関を取る手法を試したが、
  TM_CCORR_NORMED は平均を引かずに正規化するため全クラスのスコアが
  0.95〜1.0 に張り付いて識別力を失った (実験スクリプトで確認、本実装には
  含めていない)。現状は無地 (mask なし) TM_CCOEFF_NORMED を採用する。
- **誤検出耐性は限定的**: ポップアップ非表示フレームでの最大誤検出スコアが
  0.578 まで observed (n=2、video_c54 1P側のみ)。閾値 0.60 で当該 2 件は
  排除できるが、この閾値は極小サンプルに基づく暫定値であり、全動画検証で
  再検証が必須。閾値を上げた副作用として、遷移中の弱い信号フレーム
  (n=3, score 0.465〜0.505) は検出できなくなるが、1 ステップの表示継続時間
  (約0.65秒) 内に高信頼度な peak フレームが必ず含まれる前提で許容している。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.image_reader import BoardRegion, DEFAULT_P1_REGION, DEFAULT_P2_REGION

# ============================
# 定数 (1920x1080 前提)
# ============================

Side = Literal["1P", "2P"]

# テンプレ格納先
DEFAULT_CHAIN_TEMPLATE_DIR: Path = Path("models/ui_templates/chain_count_digits")

# テンプレ数字の標準の高さ (video_c54 実測 crop、全クラス共通)。
# 幅は数字ごとの自然な字幅が異なる (例: "1" は細く "4" は太い) ため、
# score_ocr の等ピッチ digit (50x40 固定) とは異なりリサイズ強制しない。
# ここでは代表値としてテンプレ読込時の目安チェックにのみ使う。
CHAIN_DIGIT_HEIGHT: int = 110
CHAIN_DIGIT_WIDTH: int = 70  # 代表値 (1=58px 〜 4=73px の中間、目安表示用)

# 検出対象の連鎖数範囲 (1桁のみ対応。10連鎖以上は本モジュール未対応)
CHAIN_COUNT_MIN: int = 1
CHAIN_COUNT_MAX: int = 9
CHAIN_DIGIT_LABELS: tuple[int, ...] = tuple(range(CHAIN_COUNT_MIN, CHAIN_COUNT_MAX + 1))

# NCC 信頼度閾値 (matchTemplate の TM_CCOEFF_NORMED 最大値)。
# 2026-07-29 実測 (video_c54, 1P側, 12サンプル: peak4/near-peak6/no-popup2):
#   ポップアップなしフレームでの最大誤検出スコア = 0.578
#   ポップアップありフレーム (peak〜遷移中) の最小スコア = 0.465〜1.000
# → 0.45 では誤検出 2/2 を弾けない。0.60 に引き上げることで誤検出 2/2 を
#   排除できる一方、遷移中の弱い信号 3 サンプルが未検出 (None) になる。
# window内サンプリングで peak フレーム (score≈0.9〜1.0) を捕捉できれば
# 十分なため、誤検出排除を優先し 0.60 を採用する。
# 注意: n=12・1動画1サイドのみの実測であり、全動画検証で要再検証。
CHAIN_NCC_MIN_CONFIDENCE: float = 0.60

# ポップアップ表示のおおよその表示継続時間 (秒、60fps 実測 0.6〜0.7s)。
# window探索のサンプリング間隔を決める目安として使う。
CHAIN_POPUP_DISPLAY_DURATION_SEC: float = 0.65

# window内サンプリングの既定間隔 (秒)。表示継続時間の 1/10 程度を確保すれば
# 取りこぼしのリスクは低い (60fps で 0.05s = 3 フレームおき)。
CHAIN_WINDOW_SAMPLE_INTERVAL_SEC: float = 0.05

# 入力フレームの想定サイズ (score_ocr と共通)
EXPECTED_FRAME_SHAPE: tuple[int, int] = (1080, 1920)


# ============================
# 結果データクラス
# ============================


@dataclass(frozen=True)
class ChainCountReadResult:
    """ChainCountOcr.read_side() の結果。

    Attributes:
        chain_count: 読み取った連鎖数 (1-9)。未検出/信頼度不足なら None。
        confidence: 採用したクラスの NCC 最大値 (0.0..1.0)。
        location: ROI 内でのテンプレ左上座標 (デバッグ用)。未検出時 None。
    """

    chain_count: int | None
    confidence: float
    location: tuple[int, int] | None


@dataclass(frozen=True)
class ChainCountWindowResult:
    """ChainCountOcr.read_max_in_window() の結果。

    Attributes:
        max_chain_count: window内で観測した連鎖数の最大値 (未検出のみなら None)。
        samples: 各サンプル時刻の生の読み取り結果 (デバッグ・検証用)。
        n_hits: chain_count が非 None だったサンプル数。
    """

    max_chain_count: int | None
    samples: tuple[ChainCountReadResult, ...]
    n_hits: int


# ============================
# 内部ヘルパ
# ============================


def _ensure_1080p(frame: np.ndarray) -> np.ndarray | None:
    """1080p 以外なら resize、サイズ 0 や ndim 不正は None (score_ocr と共通仕様)。"""
    if frame is None or frame.ndim != 3:
        return None
    h, w = frame.shape[:2]
    if (h, w) == EXPECTED_FRAME_SHAPE:
        return frame
    if h < 100 or w < 100:
        return None
    return cv2.resize(frame, (EXPECTED_FRAME_SHAPE[1], EXPECTED_FRAME_SHAPE[0]),
                      interpolation=cv2.INTER_AREA)


def _region_to_bounds(region: BoardRegion) -> tuple[int, int, int, int]:
    """BoardRegion を (y1, y2, x1, x2) の切り出し範囲に変換する。"""
    return region.y, region.y + region.height, region.x, region.x + region.width


def _crop_search_roi(frame: np.ndarray, side: Side) -> np.ndarray | None:
    """フレームから「N れんさ!」ポップアップの検索対象 ROI (盤面全体) を切り出す。"""
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    y1, y2, x1, x2 = _region_to_bounds(region)
    if y2 > frame.shape[0] or x2 > frame.shape[1]:
        return None
    return frame[y1:y2, x1:x2].copy()


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _match_digit_in_roi(
    gray_roi: np.ndarray, tpl_gray: np.ndarray,
) -> tuple[float, tuple[int, int]]:
    """1テンプレを ROI 全体でスキャンし、(最大 NCC スコア, 左上座標) を返す。"""
    if (gray_roi.shape[0] < tpl_gray.shape[0]
            or gray_roi.shape[1] < tpl_gray.shape[1]):
        return 0.0, (0, 0)
    res = cv2.matchTemplate(gray_roi, tpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return float(max_val), (int(max_loc[0]), int(max_loc[1]))


# ============================
# ChainCountOcr 本体
# ============================


class ChainCountOcr:
    """「N れんさ!」ポップアップの連鎖数 OCR 器。

    score_ocr.ScoreOcr と同様の NCC 分類だが、表示位置が可変なため盤面全体
    (DEFAULT_P1_REGION / DEFAULT_P2_REGION) をスキャンして最良一致を探す。
    """

    def __init__(
        self,
        templates: dict[int, np.ndarray] | None = None,
        min_confidence: float = CHAIN_NCC_MIN_CONFIDENCE,
    ) -> None:
        """Args:
            templates: 1-9 → テンプレ画像 (BGR or grayscale) の辞書。
                未指定 (None) クラスは OCR で None 扱いになる。
                数字ごとに自然な字幅が異なるため (例: "1"は細い/"4"は太い)、
                score_ocr と異なり共通サイズへのリサイズは行わない
                (各テンプレは採取時の実寸のまま保持する)。
            min_confidence: NCC 最低スコア。これ未満なら None。
        """
        self._templates_gray: dict[int, np.ndarray] = {}
        if templates:
            for label, tpl in templates.items():
                if tpl is None:
                    continue
                self._templates_gray[int(label)] = _to_gray(tpl)
        self._min_confidence = float(min_confidence)
        self._warned_missing = False
        if not self._templates_gray:
            print("[chain_count_ocr] WARNING: テンプレが 1 つも登録されていない。"
                  "OCR は常に None を返す。")
            self._warned_missing = True
        else:
            missing = sorted(set(CHAIN_DIGIT_LABELS) - set(self._templates_gray.keys()))
            if missing and not self._warned_missing:
                print(f"[chain_count_ocr] WARNING: 未整備クラス {missing} は "
                      "OCR で None 扱いになる。")
                self._warned_missing = True

    # ============================
    # クラス生成
    # ============================

    @classmethod
    def load_default(
        cls,
        template_dir: Path = DEFAULT_CHAIN_TEMPLATE_DIR,
        min_confidence: float = CHAIN_NCC_MIN_CONFIDENCE,
    ) -> "ChainCountOcr":
        """models/ui_templates/chain_count_digits/ から digit_N.png を読み込む。"""
        templates = cls._load_templates_from_dir(template_dir)
        return cls(templates=templates, min_confidence=min_confidence)

    @staticmethod
    def _load_templates_from_dir(template_dir: Path) -> dict[int, np.ndarray]:
        """digit_N.png を全部スキャンする (N=1..9)。"""
        templates: dict[int, np.ndarray] = {}
        if not template_dir.is_dir():
            return templates
        for label in CHAIN_DIGIT_LABELS:
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

    def read_side(self, frame: np.ndarray, side: Side) -> ChainCountReadResult:
        """1920x1080 BGR フレームから指定サイドの連鎖数ポップアップを読み取る。"""
        f = _ensure_1080p(frame)
        if f is None:
            return ChainCountReadResult(None, 0.0, None)
        roi = _crop_search_roi(f, side)
        if roi is None or roi.size == 0:
            return ChainCountReadResult(None, 0.0, None)
        gray_roi = _to_gray(roi)
        return self._classify(gray_roi)

    def _classify(self, gray_roi: np.ndarray) -> ChainCountReadResult:
        """ROI 全体をスキャンし、最良一致クラスを返す (0-9 分類ロジック相当)。"""
        if not self._templates_gray:
            return ChainCountReadResult(None, 0.0, None)
        best_label: int | None = None
        best_score: float = -1.0
        best_loc: tuple[int, int] = (0, 0)
        for label, tpl in self._templates_gray.items():
            score, loc = _match_digit_in_roi(gray_roi, tpl)
            if score > best_score:
                best_score = score
                best_label = label
                best_loc = loc
        if best_label is None or best_score < self._min_confidence:
            return ChainCountReadResult(None, max(0.0, best_score), None)
        return ChainCountReadResult(best_label, best_score, best_loc)

    # ============================
    # 公開: window内 最大値集計
    # ============================

    def read_max_in_window(
        self,
        cap: "cv2.VideoCapture",
        side: Side,
        t_start: float,
        t_end: float,
        sample_interval_sec: float = CHAIN_WINDOW_SAMPLE_INTERVAL_SEC,
    ) -> ChainCountWindowResult:
        """[t_start, t_end] を一定間隔でサンプリングし、連鎖数の最大値を返す。

        「N れんさ!」は連鎖ステップが進むたびに増えるため、1回の連鎖window内で
        観測した最大値が最終連鎖数になる (userタスク仕様)。

        Args:
            cap: cv2.VideoCapture (呼び出し側でオープン済み)。
            side: "1P" or "2P"。
            t_start: window開始時刻 (秒)。
            t_end: window終了時刻 (秒)。t_start 以上であること。
            sample_interval_sec: サンプリング間隔 (既定 0.05秒)。

        Returns:
            ChainCountWindowResult。
        """
        if t_end < t_start or sample_interval_sec <= 0.0:
            return ChainCountWindowResult(None, (), 0)
        samples: list[ChainCountReadResult] = []
        t = t_start
        while t <= t_end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                samples.append(self.read_side(frame, side))
            t += sample_interval_sec
        return _aggregate_window_samples(samples)


# ============================
# 純粋関数: window集計ロジック (stateless、CLAUDE.md 指標stateless原則)
# ============================


def _aggregate_window_samples(
    samples: list[ChainCountReadResult],
) -> ChainCountWindowResult:
    """サンプル列から最大連鎖数を集計する (video I/O を含まない純粋関数)。"""
    hits = [s for s in samples if s.chain_count is not None]
    max_count = max((s.chain_count for s in hits), default=None)
    return ChainCountWindowResult(
        max_chain_count=max_count,
        samples=tuple(samples),
        n_hits=len(hits),
    )


__all__ = [
    "CHAIN_COUNT_MAX",
    "CHAIN_COUNT_MIN",
    "CHAIN_DIGIT_HEIGHT",
    "CHAIN_DIGIT_LABELS",
    "CHAIN_DIGIT_WIDTH",
    "CHAIN_NCC_MIN_CONFIDENCE",
    "CHAIN_POPUP_DISPLAY_DURATION_SEC",
    "CHAIN_WINDOW_SAMPLE_INTERVAL_SEC",
    "DEFAULT_CHAIN_TEMPLATE_DIR",
    "ChainCountOcr",
    "ChainCountReadResult",
    "ChainCountWindowResult",
]
