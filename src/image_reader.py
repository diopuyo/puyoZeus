"""
フレーム→盤面変換モジュール

ぷよぷよeスポーツ (1920×1080) のスクリーンショット/フレーム画像から
各プレイヤーの盤面データを読み取る。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_OJAMA,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
    HIDDEN_ROWS,
    VISIBLE_ROWS,
    Board,
)

# ============================
# 定数定義
# ============================

# セルサンプリング範囲 (セルサイズに対する割合)
# puyo の中央部分のみサンプリング (目・縁・周辺背景の影響を排除し、
# 純粋なぷよ色の median を取得)。0.5 = セル中央 50% を使用。
CELL_SAMPLE_RATIO: float = 0.5

# 空セル判定の輝度閾値 (V < この値なら空)
EMPTY_V_THRESHOLD: int = 40

# おじゃま判定の彩度閾値 (S < この値かつV > EMPTY_V_THRESHOLD ならおじゃま候補)
# 2026-05-12 cycle 71f (提案 B): 30→20 に下げて、 灰色寄り黄ぷよ等を
# OJAMA に倒さない. 真の OJAMA は S=10-20 範囲が大半で 20 でも十分検出可能.
# 2026-05-11 サイクル70: 60→30 で 薄い黄色 puyo の OJAMA 誤分類を回避.
# 真の OJAMA puyo は S<20 (= ほぼグレー) なので 30 でも安全.
OJAMA_S_THRESHOLD: int = 20

# おじゃま判定の輝度下限 (V > この値でないとおじゃまとみなさない)
OJAMA_V_MIN: int = 100

# 2026-05-11 サイクル71: per-pixel 投票方式の閾値
# puyo 色票が全 cell ピクセル数 × VOTE_PUYO_MIN_RATIO 以上なら puyo 採用.
# 「半分埋まり」 「ハイライト残光」 cell でも本物の色を取れるよう 10% 程度に設定.
VOTE_PUYO_MIN_RATIO: float = 0.10
# おじゃま票がこの割合以上なら OJAMA. ぷよぷよ ojama は ほぼ cell 全体グレーなので
# 高めに設定 (40%).
VOTE_OJAMA_MIN_RATIO: float = 0.40
# 投票で red 拡張範囲 (h 11-18) を採用するかの BGR R-G ピクセル単位差閾値.
# median 方式と同じ意味だが、 個別ピクセル単位での適用.
VOTE_RED_EXTENDED_RG_DIFF: int = 80

# 赤と黄の区別: BGR の R-G 差がこれ以上なら赤、未満なら黄
# (赤=R突出、黄=R≈G の混合色、HSV では H/S 共に被るため BGR で区別)
RED_GREEN_DIFF_FOR_RED: int = 80
# 赤候補で V がこれ以下なら紫候補 (赤紫判定、暗い赤系は紫の方が真値多数)
PURPLE_V_MAX_FOR_RED_CANDIDATE: int = 170

# ============================
# 背景 FP tier 1 (EXTREME) threshold (cycle 33/37 確定値)
# ============================
# tier 1: 距離 < BG_EXTREME_THRESHOLD_DEFAULT → 無条件 EMPTY (= 全 cell 共通)
# cycle 37 sweep 結果: t=25 が「副作用最小 + v97m11 -32 件改善」 最適。
BG_EXTREME_THRESHOLD_DEFAULT: float = 25.0
# 軸 3-b (Phase L): 1P/2P 盤面左上エリア (visible_row >= 5, col <= 1) 用閾値。
# 2026-05-27 v40 col=1 EMPTY 症状により軸 3-b 撤回: +15.0 → +0.0 (= DEFAULT と同値)。
# _resolve_tier1_threshold のエリア別分岐ロジックは将来の per-region 調整のため残す。
BG_EXTREME_THRESHOLD_LEFT_UPPER: float = BG_EXTREME_THRESHOLD_DEFAULT + 0.0
# 左上エリアの定義: 表示行 (visible_row = row - HIDDEN_ROWS) のうち中盤 ~ 下部
BG_LEFT_UPPER_VISIBLE_ROW_MIN: int = 5  # 表示行 5 以上 (= 表示中盤下から最下段)
BG_LEFT_UPPER_COL_MAX: int = 1  # 列 0-1 (= 最左 2 列)

# bg_fp 採取前保護モード (= I1 対応 A):
# bg_fp が未採取の期間に tier 1 を 0.0 に設定して CNN 経路を無効化し
# HSV-only 経路に倒す。 bg_fp 採取完了後は DEFAULT に戻る。
BG_EXTREME_THRESHOLD_PRE_CAPTURE: float = 0.0

# ============================
# データクラス
# ============================


@dataclass
class HsvRange:
    """
    OpenCV HSV色空間の閾値範囲。
    H: 0–180 / S: 0–255 / V: 0–255
    赤は H が 0 付近と 170–180 で折り返すため h_max > h_min でない場合がある。
    """
    h_min: int
    h_max: int
    s_min: int = 80
    s_max: int = 255
    v_min: int = 80
    v_max: int = 255


@dataclass
class BoardRegion:
    """
    フレーム画像上での盤面の矩形領域。

    盤面は画面上 VISIBLE_ROWS 行 (=12) が見えており、隠し段 (row 0) は
    画面外で直接は読み取れない。本リージョンは可視 12 行を包含する矩形を
    表す (width/height は可視領域のみ)。

    NOTE: ぷよぷよeスポーツ 1920×1080 での座標は実際の映像で要キャリブレーション。
    """
    x: int       # 左端X座標 (px)
    y: int       # 上端Y座標 (px、可視領域上端)
    width: int   # 盤面幅 (px)
    height: int  # 可視領域の高さ (px、12行分)

    @property
    def cell_width(self) -> float:
        """1セルの幅 (px)。"""
        return self.width / BOARD_COLS

    @property
    def cell_height(self) -> float:
        """1セルの高さ (px)。 可視領域高さ / VISIBLE_ROWS。"""
        return self.height / VISIBLE_ROWS

    def cell_center(self, row: int, col: int) -> tuple[int, int]:
        """
        指定セルの中心座標 (x, y) を返す。

        row = 0..HIDDEN_ROWS-1 は画面外 (隠し段) → 画面上では領域の上方に
        推定位置を返すが、画像データとしては確定不能。
        row = HIDDEN_ROWS..BOARD_ROWS-1 は可視領域の行。
        """
        visible_row = row - HIDDEN_ROWS  # 可視領域での行番号 (-1 は隠し段)
        cx = int(self.x + (col + 0.5) * self.cell_width)
        cy = int(self.y + (visible_row + 0.5) * self.cell_height)
        return cx, cy

    def is_visible_row(self, row: int) -> bool:
        """指定行が画面に見えている(読み取れる)行か。"""
        return row >= HIDDEN_ROWS

    def cell_sample_rect(self, row: int, col: int) -> tuple[int, int, int, int]:
        """
        指定セルのサンプリング矩形 (x1, y1, x2, y2) を返す。
        セル中央の CELL_SAMPLE_RATIO 分の領域をサンプルする。

        cycle 71i (2026-05-12 ユーザー指摘): 上部 row の下寄せロジックを撤回し、
        全 row で cell 中央 sample に統一する. 旧 cycle 69-A/70 の下寄せは
        「上部 cropped」 仮定だったが、 ラベリング検証で「上部 row でも
        ぷよ全体が cell 内に収まっている / 下寄せすると上半分の色情報が
        学習データに含まれない」 ことが判明したため.

        Note: この変更で学習用 patch 領域が変わるため、 既存 CNN model は
        新しい sample 領域で再 fine-tune する必要がある.
        """
        cx, cy = self.cell_center(row, col)
        half_w = max(1, int(self.cell_width * CELL_SAMPLE_RATIO / 2))
        half_h = max(1, int(self.cell_height * CELL_SAMPLE_RATIO / 2))
        return cx - half_w, cy - half_h, cx + half_w, cy + half_h


# ============================
# デフォルト盤面領域 (1920×1080 要キャリブレーション)
# ============================

# 1P 盤面領域 (左側) — calibration_video01.json と同期
DEFAULT_P1_REGION: BoardRegion = BoardRegion(x=282, y=160, width=384, height=720)

# 2P 盤面領域 (右側) — calibration_video01.json と同期
DEFAULT_P2_REGION: BoardRegion = BoardRegion(x=1258, y=160, width=384, height=720)


# ============================
# HSV 色閾値テーブル
# ============================

# 各色のHSV範囲リスト (複数範囲の OR 判定に対応)
DEFAULT_COLOR_RANGES: dict[int, list[HsvRange]] = {
    # 2026-05-10 FIX-A2: 全色 S_min を背景誤認識回避レベルに引き上げ
    # 試合中の本物 puyo は S>=160 が大半なので影響少
    COLOR_RED: [
        # 2026-05-12 cycle 71f (提案 B): H_max 18→13 で YELLOW (= H=14-38) との
        # 重複解消. dim 赤の救済は R-G diff チェック (= 268-273 行) で代替.
        HsvRange(h_min=0,   h_max=13,  s_min=160, v_min=100),
        HsvRange(h_min=166, h_max=180, s_min=160, v_min=100),
    ],
    COLOR_BLUE: [
        HsvRange(h_min=100, h_max=130, s_min=160, v_min=80),
    ],
    COLOR_GREEN: [
        HsvRange(h_min=50,  h_max=85,  s_min=160, v_min=80),
    ],
    COLOR_YELLOW: [
        # 黄は S 低めでも観測されるが、 試合中の本物は S>=80 程度
        # 2026-05-11 サイクル68: V_min 180→120 で上部 dim cell 救済.
        # RED が dict 順で先に check されるため、 dim 赤との混同回避済.
        # 2026-05-12 cycle 71f (提案 B): S_min 80→100 で灰色寄り黄を除外し
        # OJAMA への誤判定を抑止. 真の黄ぷよは S>=120 が大半.
        HsvRange(h_min=14,  h_max=38,  s_min=100, v_min=120),
    ],
    COLOR_PURPLE: [
        HsvRange(h_min=130, h_max=165, s_min=130, v_min=80),
    ],
}


# ============================
# 色分類器
# ============================

class ColorClassifier:
    """
    BGR画像パッチからぷよの色を分類する。

    HSV中央値を計算し、閾値テーブルと照合する。
    """

    def __init__(
        self, color_ranges: dict[int, list[HsvRange]] | None = None,
        vote_mode: bool = False,
    ) -> None:
        """
        Args:
            color_ranges: 色コードからHsvRangeリストへのマッピング。
                          Noneの場合はDEFAULT_COLOR_RANGESを使用。
            vote_mode: True なら per-pixel 投票方式で分類 (サイクル71).
                       False (default) は HSV 中央値 + cycle 69-B サブ region vote
                       (= 後方互換). 投票方式は混合色 cell や半分埋まり cell に強い.
        """
        self._ranges: dict[int, list[HsvRange]] = (
            color_ranges if color_ranges is not None else DEFAULT_COLOR_RANGES
        )
        # 2026-05-11 サイクル63: 解像度依存 S_min スケール係数.
        # 1.0 = 720p+ (= default 通り), 0.7 = 360p (S 下限緩和).
        # 360p アップスケール時の色彩飽和度低下を補償.
        self._s_min_scale: float = 1.0
        # 2026-05-11 サイクル71: per-pixel 投票分類モード.
        self._vote_mode: bool = bool(vote_mode)

    def classify(self, bgr_patch: np.ndarray) -> int:
        """
        BGRパッチの色を分類して色コードを返す。

        vote_mode=True なら per-pixel 投票方式 (サイクル71).
        vote_mode=False なら HSV 中央値 + cycle 69-B サブ region vote (default).

        Args:
            bgr_patch: shape=(H, W, 3) のBGR画像パッチ。

        Returns:
            int: 色コード (COLOR_* 定数)。
        """
        if bgr_patch.size == 0:
            return COLOR_EMPTY

        if self._vote_mode:
            return self._classify_by_vote(bgr_patch)

        hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        h = int(np.median(hsv_patch[:, :, 0]))
        s = int(np.median(hsv_patch[:, :, 1]))
        v = int(np.median(hsv_patch[:, :, 2]))

        if v < EMPTY_V_THRESHOLD:
            return COLOR_EMPTY

        # 色閾値照合 (OJAMA より先に: 黄等の低彩度色を OJAMA に倒さない)
        # 赤の H 11-18 (黄と被る拡張範囲) は BGR の R-G 差で黄と区別する。
        # 2026-05-11: _s_min_scale を s_min に乗算 (低解像度時の S 緩和).
        red_skipped = False
        scale = self._s_min_scale
        for color_code, ranges in self._ranges.items():
            for rng in ranges:
                eff_s_min = int(rng.s_min * scale) if scale < 1.0 else rng.s_min
                if (
                    rng.h_min <= h <= rng.h_max
                    and eff_s_min <= s <= rng.s_max
                    and rng.v_min <= v <= rng.v_max
                ):
                    if color_code == COLOR_RED and 11 <= h <= 18:
                        # 拡張範囲 (黄との境界) → BGR で確認
                        g_med = int(np.median(bgr_patch[:, :, 1]))
                        r_med = int(np.median(bgr_patch[:, :, 2]))
                        if r_med - g_med >= RED_GREEN_DIFF_FOR_RED:
                            return COLOR_RED
                        # R-G 差不足 → 赤判定をスキップして黄等を試す
                        red_skipped = True
                        break
                    return color_code
            if red_skipped:
                red_skipped = False
                continue

        if s < OJAMA_S_THRESHOLD and v >= OJAMA_V_MIN:
            return COLOR_OJAMA

        # 2026-05-11 サイクル69-B: 中央 median が EMPTY 判定された場合、 4 sub-region
        # に分けてどれかが puyo 色を返せばそれを採用 (= 部分的に visible な puyo を救済).
        if bgr_patch.shape[0] >= 4 and bgr_patch.shape[1] >= 4:
            h2 = bgr_patch.shape[0] // 2
            w2 = bgr_patch.shape[1] // 2
            sub_patches = [
                bgr_patch[:h2, :w2],
                bgr_patch[:h2, w2:],
                bgr_patch[h2:, :w2],
                bgr_patch[h2:, w2:],
            ]
            sub_colors: list[int] = []
            for sp in sub_patches:
                if sp.size == 0:
                    continue
                c = self._classify_single_patch_no_subregion(sp)
                if c not in (COLOR_EMPTY, COLOR_UNKNOWN):
                    sub_colors.append(c)
            if sub_colors:
                # 最頻 puyo 色を返す
                from collections import Counter
                most = Counter(sub_colors).most_common(1)[0][0]
                return most

        return COLOR_EMPTY

    def _classify_single_patch_no_subregion(
        self, bgr_patch: np.ndarray,
    ) -> int:
        """サブ領域 vote 用、 純 median 分類 (再帰せず)."""
        if bgr_patch.size == 0:
            return COLOR_EMPTY
        hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        h = int(np.median(hsv_patch[:, :, 0]))
        s = int(np.median(hsv_patch[:, :, 1]))
        v = int(np.median(hsv_patch[:, :, 2]))
        if v < EMPTY_V_THRESHOLD:
            return COLOR_EMPTY
        scale = self._s_min_scale
        for color_code, ranges in self._ranges.items():
            for rng in ranges:
                eff_s_min = int(rng.s_min * scale) if scale < 1.0 else rng.s_min
                if (
                    rng.h_min <= h <= rng.h_max
                    and eff_s_min <= s <= rng.s_max
                    and rng.v_min <= v <= rng.v_max
                ):
                    if color_code == COLOR_RED and 11 <= h <= 18:
                        g_med = int(np.median(bgr_patch[:, :, 1]))
                        r_med = int(np.median(bgr_patch[:, :, 2]))
                        if r_med - g_med < RED_GREEN_DIFF_FOR_RED:
                            continue
                    return color_code
        if s < OJAMA_S_THRESHOLD and v >= OJAMA_V_MIN:
            return COLOR_OJAMA
        return COLOR_EMPTY

    def _classify_by_vote(self, bgr_patch: np.ndarray) -> int:
        """サイクル71: per-pixel 投票方式で分類する.

        各ピクセルを HSV 色レンジに照合し、 puyo 色 (1-5) 票が最多の色を採用する.
        median 方式と比較した利点:
            - cell の半分だけ puyo が visible でも本物の色を取れる
            - ハイライト残光 / 影に強い (= mean に引っ張られない)
            - 混合色 (= puyo + 背景) でも純色領域がある程度あれば正しく取れる

        判定ロジック:
            1. puyo 色 (1-5) のうち最多票 puyo_top_votes と ojama 票 ojama_votes を比較
            2. puyo_top_votes >= VOTE_PUYO_MIN_RATIO × 全ピクセル かつ puyo > ojama
               → puyo 色採用
            3. ojama_votes >= VOTE_OJAMA_MIN_RATIO × 全ピクセル → OJAMA
            4. それ以外 → EMPTY

        Args:
            bgr_patch: shape=(H, W, 3) BGR パッチ.

        Returns:
            int: 色コード (COLOR_*).
        """
        hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]
        total = int(h.size)
        if total == 0:
            return COLOR_EMPTY

        scale = self._s_min_scale
        votes: dict[int, int] = {}
        # 各 puyo 色 (1-5) のピクセル票を計算
        for color_code, ranges in self._ranges.items():
            color_mask = np.zeros_like(h, dtype=bool)
            for rng in ranges:
                eff_s_min = (
                    int(rng.s_min * scale) if scale < 1.0 else rng.s_min
                )
                m = (
                    (h >= rng.h_min) & (h <= rng.h_max)
                    & (s >= eff_s_min) & (s <= rng.s_max)
                    & (v >= rng.v_min) & (v <= rng.v_max)
                )
                color_mask = color_mask | m
            # red の拡張範囲 (h 11-18) は BGR R-G 差で黄と分離が必要.
            # 該当ピクセルだけ R-G 差を確認し、 不足分を mask から落とす.
            if color_code == COLOR_RED:
                extended = (h >= 11) & (h <= 18) & color_mask
                if bool(extended.any()):
                    rg_diff = (
                        bgr_patch[..., 2].astype(np.int16)
                        - bgr_patch[..., 1].astype(np.int16)
                    )
                    keep = rg_diff >= VOTE_RED_EXTENDED_RG_DIFF
                    color_mask = color_mask & (~extended | keep)
            votes[color_code] = int(color_mask.sum())

        # おじゃま票 (S 低 かつ V 高)
        ojama_mask = (s < OJAMA_S_THRESHOLD) & (v >= OJAMA_V_MIN)
        ojama_votes = int(ojama_mask.sum())

        # puyo 票最多 (1-5 のみ対象)
        puyo_top_color = COLOR_EMPTY
        puyo_top_votes = 0
        for color_code, n in votes.items():
            if n > puyo_top_votes:
                puyo_top_votes = n
                puyo_top_color = color_code

        puyo_min = max(1, int(total * VOTE_PUYO_MIN_RATIO))
        ojama_min = max(1, int(total * VOTE_OJAMA_MIN_RATIO))

        # 判定優先順位: puyo > ojama > empty
        if puyo_top_votes >= puyo_min and puyo_top_votes > ojama_votes:
            return puyo_top_color
        if ojama_votes >= ojama_min:
            return COLOR_OJAMA
        return COLOR_EMPTY

    def classify_batch(self, bgr_patches: list[np.ndarray]) -> list[int]:
        """Z-3C: 複数 patch をまとめて分類 (個別 classify を回す簡易版)。

        ColorClassifier 自体は per-patch ロジックなので、純粋な loop。
        ただし呼び出しオーバーヘッド削減と、API 統一のため提供。
        """
        return [self.classify(p) for p in bgr_patches]

    def set_color_ranges_from_simple(
        self,
        simple_ranges: dict[int, tuple[int, int, int, int, int, int]],
        append: bool = True,
    ) -> None:
        """Z-3I: OnlineHsvCalibrator から動画別 HSV 範囲を注入。

        Args:
            simple_ranges: color → (h_min, h_max, s_min, s_max, v_min, v_max)
            append: True なら default ranges に動画別 ranges を追加 (=広い OR 判定)、
                    False なら上書き (旧挙動、 学習が tight すぎると一部 cell を
                    捕捉できず empty に倒れる問題があるため default は True)。
        """
        new_ranges: dict[int, list[HsvRange]] = {
            k: list(v) for k, v in self._ranges.items()
        }
        for color, (h_min, h_max, s_min, s_max, v_min, v_max) in (
            simple_ranges.items()
        ):
            db_range = HsvRange(
                h_min=int(h_min), h_max=int(h_max),
                s_min=int(s_min), s_max=int(s_max),
                v_min=int(v_min), v_max=int(v_max),
            )
            if append and color in new_ranges:
                new_ranges[color] = list(new_ranges[color]) + [db_range]
            else:
                new_ranges[color] = [db_range]
        self._ranges = new_ranges

    def set_s_min_scale(self, scale: float) -> None:
        """解像度依存 S_min 緩和係数を設定 (1.0=既定、 0.7=360p 想定)."""
        self._s_min_scale = float(max(0.3, min(1.0, scale)))

    def classify_hsv(self, h: int, s: int, v: int) -> int:
        """
        HSV値を直接渡して色分類する (テスト・デバッグ用)。

        Args:
            h: Hue (0–180)
            s: Saturation (0–255)
            v: Value (0–255)

        Returns:
            int: 色コード。

        Note:
            ピクセルカウント分類器に合わせて 8×8 の均一パッチを生成。
        """
        dummy = np.full((8, 8, 3), [h, s, v], dtype=np.uint8)
        bgr = cv2.cvtColor(dummy, cv2.COLOR_HSV2BGR)
        return self.classify(bgr)


# ============================
# 案 P2: 白ハイライト blob override ヘルパー
# ============================

def _has_puyo_highlight(patch_hsv: np.ndarray) -> bool:
    """現フレームパッチに白ハイライト blob があれば True。

    背景 FP が tier1 EMPTY と判定したセルに対して、
    ぷよ固有の白ハイライト円を検出して「本物ぷよ」として救済する。
    """
    from src.background_fingerprint import detect_highlight_blob
    return detect_highlight_blob(patch_hsv)


# ============================
# 画像読み取り器
# ============================

class ImageReader:
    """
    フレーム画像からBoard (盤面データ) を読み取るクラス。

    Usage:
        reader = ImageReader()
        board_1p, board_2p = reader.read_both_boards(frame)
    """

    def __init__(
        self,
        classifier: ColorClassifier | None = None,
        p1_region: BoardRegion | None = None,
        p2_region: BoardRegion | None = None,
        bg_fingerprint_p1: "BackgroundFingerprint | None" = None,
        bg_fingerprint_p2: "BackgroundFingerprint | None" = None,
        bg_empty_threshold: float | None = None,
        apply_inference: bool = True,
        floating_min_gap: int = 2,
        use_ui_mask: bool = True,
        use_match_state: bool = False,
        use_telop_mask: bool = False,
        patch_ncc_threshold: float | None = None,
        use_highlight_override: bool = False,
        puyo_profile_db: "PuyoColorProfileDB | None" = None,
    ) -> None:
        """
        Args:
            classifier: 色分類器。Noneの場合はデフォルトを使用。
            p1_region: 1P盤面の領域。Noneの場合はデフォルトを使用。
            p2_region: 2P盤面の領域。Noneの場合はデフォルトを使用。
            bg_fingerprint_p1: 1P 試合開始時の背景 FP (Phase T サイクル 1)。
                指定すると各セルの「ぷよあり/なし」が背景差分で先に判定される。
            bg_fingerprint_p2: 2P 同上。
            bg_empty_threshold: 背景との HSV 距離が これ未満なら「空」と判定。
                None ならデフォルト (DEFAULT_EMPTY_HSV_DISTANCE)。
            apply_inference: 浮遊ぷよ削除・隠し段推論を行うか。
            floating_min_gap: 浮遊判定の最小ギャップ。
            patch_ncc_threshold: PatchBackgroundFingerprint NCC 空判定閾値の上書き値。
                None なら background_fingerprint.py の PATCH_NCC_EMPTY_THRESHOLD (= 0.92) を使用。
                NCC sweep 用 (case d 閾値探索)。
            use_highlight_override: 案 P2 白ハイライト blob 検出による tier1 EMPTY 却下。
                2026-05-28 案 R3 改: デフォルトを False に変更 (案 P2 同時撤回)。
                True の場合、tier1 が EMPTY 判定したセルでも白ハイライト blob が
                検出されれば「ぷよあり」として classify に進む (再評価可能性のため残置)。
                False (デフォルト) は override 無効 (= 従来挙動)。
            puyo_profile_db: 案 R3 改 per-video ぷよ色プロファイル DB。
                指定すると classify 後の色に対してプロファイル距離チェックを行い、
                不一致なら EMPTY 化 (= 幻ぷよ抑制)。None で無効 (既存挙動)。
        """
        self._classifier: ColorClassifier = classifier or ColorClassifier()
        self._p1_region: BoardRegion = p1_region or DEFAULT_P1_REGION
        self._p2_region: BoardRegion = p2_region or DEFAULT_P2_REGION
        self._bg_fp_p1 = bg_fingerprint_p1
        self._bg_fp_p2 = bg_fingerprint_p2
        if bg_empty_threshold is None:
            from src.background_fingerprint import DEFAULT_EMPTY_HSV_DISTANCE
            bg_empty_threshold = DEFAULT_EMPTY_HSV_DISTANCE
        self._bg_threshold: float = float(bg_empty_threshold)
        # cycle 33 (2026-05-20): tiered bg_fp empty 判定
        # tier 1: 距離 < EXTREME 閾値 → 無条件 EMPTY (= 確実な背景、 puyo 誤認リスク無視小)
        # tier 2: 距離 < _bg_threshold → AND 条件 (= cycle 19 既存、 HSV 単独も EMPTY 必要)
        # 既存挙動の互換性: tier 1 を 0 に設定すれば旧挙動と同じ
        # cycle 37 確定 (2026-05-20 深夜 sweep 結果): threshold 25.0 採用
        # sweep 結果: t=20 (101.7) > t=25 (92.3) ✅ > t=27 (99.7) 副作用大 > t=30 (90.0)
        # 25 が「副作用最小 + v97m11 -32 件改善」 の最適バランス。
        # 27 は非線形挙動で auto_correction +53 副作用、 30 は v89m3 副作用 +35。
        # 軸 3-b (Phase L): 定数参照に変更 (= マジックナンバー排除)
        self._bg_extreme_threshold: float = BG_EXTREME_THRESHOLD_DEFAULT
        # I1 対応 A: bg_fp 採取前は tier 1 threshold を 0 に倒して CNN 経路を無効化
        # (= HSV-only 経路に強制)。 RecognitionPipeline が採取前後で切り替える。
        self._pre_capture_mode: bool = False
        # NCC sweep 用: PatchBackgroundFingerprint の空判定閾値上書き。
        # None なら PATCH_NCC_EMPTY_THRESHOLD (= 0.92) をそのまま使う。
        self._patch_ncc_threshold: float | None = patch_ncc_threshold
        # 案 P2: 白ハイライト blob override (tier1 EMPTY 判定後に救済チェック)
        # 2026-05-28 案 R3 改: デフォルト False (= 案 P2 同時撤回)
        self._use_highlight_override: bool = bool(use_highlight_override)
        # 案 R3 改: per-video ぷよ色プロファイル DB (classify 後の下段 EMPTY 化用)
        self._puyo_profile_db: "PuyoColorProfileDB | None" = puyo_profile_db
        self._apply_inference: bool = bool(apply_inference)
        self._floating_min_gap: int = int(floating_min_gap)
        # UI Mask (X 印など UI オーバーレイの誤検出を排除)
        if use_ui_mask:
            from src.ui_mask import UiMaskMatcher
            self._ui_matcher: "UiMaskMatcher | None" = (
                UiMaskMatcher.load_default()
            )
        else:
            self._ui_matcher = None
        # 試合状態判定 (試合中以外は強制 EMPTY)
        if use_match_state:
            from src.match_state import MatchStateDetector
            self._match_state_detector: "MatchStateDetector | None" = (
                MatchStateDetector.load_default()
            )
        else:
            self._match_state_detector = None
        # テロップ検出 (V3.1): 検出時に被覆セルを COLOR_UNKNOWN に倒す。
        # 中央テロップ (チャレンジャーリーグ等) で読めないセルを構造的に「不明」化。
        if use_telop_mask:
            from src.telop_detector import TelopDetector
            self._telop_detector: "TelopDetector | None" = (
                TelopDetector.load_default()
            )
        else:
            self._telop_detector = None
        # フレーム単位で 1 度だけテロップ検出する用キャッシュ (read_both_boards で更新)
        self._cached_telop_bbox: tuple[int, int, int, int] | None = None
        # T4: 静的背景マスク (pixel-level diff による AND ガード)
        # RecognitionPipeline が bg_fp 採取と同タイミングで inject する。
        self._static_mask_p1: "StaticBoardMask | None" = None
        self._static_mask_p2: "StaticBoardMask | None" = None

    def set_resolution_aware_s_min(self, source_height: int) -> None:
        """source_height に応じて HSV/CNN を低解像度向けに調整.

        源解像度 → スケール:
            >= 720: 1.0 (既定、 CNN 主軸)
            540-720: S 0.85x、 CNN そのまま
            < 540: S 0.7x、 CNN override 閾値 1.01 (= 事実上無効、 HSV 主軸)
                  -- 低解像度では CNN mode collapse で誤分類 (BLUE→RED 等) のため
        """
        if source_height >= 720:
            scale = 1.0
            cnn_override = None
        elif source_height >= 540:
            scale = 0.85
            cnn_override = None
        else:
            scale = 0.7
            cnn_override = 1.01  # 低解像度は CNN を信頼しない
        # HybridClassifier 経由なら _hsv 配下、 ColorClassifier 直接ならそのまま
        target = getattr(self._classifier, "_hsv", self._classifier)
        if hasattr(target, "set_s_min_scale"):
            target.set_s_min_scale(scale)
        # CNN override 閾値も低解像度では引き上げ
        if cnn_override is not None and hasattr(self._classifier, "set_cnn_override_prob"):
            self._classifier.set_cnn_override_prob(cnn_override)

    def set_background_fingerprints(
        self,
        bg_fp_p1: "BackgroundFingerprint | PatchBackgroundFingerprint | None",
        bg_fp_p2: "BackgroundFingerprint | PatchBackgroundFingerprint | None",
    ) -> None:
        """試合開始時の背景 FP を設定する (動画/試合ごとに更新可能)。
        案 d: PatchBackgroundFingerprint も受け付ける (後退互換)。
        """
        self._bg_fp_p1 = bg_fp_p1
        self._bg_fp_p2 = bg_fp_p2

    def set_static_mask(
        self,
        mask_p1: "StaticBoardMask | None",
        mask_p2: "StaticBoardMask | None",
    ) -> None:
        """T4: 試合開始時の静的背景マスクを設定する (試合ごとに更新可)。

        Args:
            mask_p1: 1P 側 StaticBoardMask。None で無効化。
            mask_p2: 2P 側 StaticBoardMask。None で無効化。
        """
        self._static_mask_p1 = mask_p1
        self._static_mask_p2 = mask_p2

    def set_puyo_profile_db(
        self,
        db: "PuyoColorProfileDB | None",
    ) -> None:
        """案 R3 改: per-video ぷよ色プロファイル DB を設定 (試合ごとに更新可)。

        Args:
            db: PuyoColorProfileDB インスタンス、または None (無効化)
        """
        self._puyo_profile_db = db

    def set_pre_capture_mode(self, enabled: bool) -> None:
        """bg_fp 採取前保護モードを切り替える (I1 対応 A)。

        enabled=True のとき tier 1 threshold を BG_EXTREME_THRESHOLD_PRE_CAPTURE
        (= 0.0) に設定し、bg_fp が None の場合でも tier 1 スキップで HSV-only
        経路に倒す。bg_fp 採取完了後は RecognitionPipeline が False に戻す。

        Args:
            enabled: True = 採取前保護モード、 False = 通常モード (DEFAULT 閾値)。
        """
        self._pre_capture_mode = enabled

    def _bg_fp_for_region(
        self, region: BoardRegion,
    ) -> "BackgroundFingerprint | None":
        """指定 region に対応する背景 FP を返す (P1 / P2 を判別)。

        ROI 動的補正で region.x/y がシフトしても判別できるよう、
        画面中央 (x=960) より左なら P1、右なら P2 とする。
        """
        if region is self._p1_region:
            return self._bg_fp_p1
        if region is self._p2_region:
            return self._bg_fp_p2
        # シフト済 region: 中央線で判別
        if region.x + region.width / 2 < 960:
            return self._bg_fp_p1
        return self._bg_fp_p2

    @staticmethod
    def _shifted_region(
        base: BoardRegion, offset: tuple[float, float],
    ) -> BoardRegion:
        """ROI を (dx, dy) px シフトした BoardRegion を返す。

        T-v2-B: ShakeDetector の検出シフトで毎フレーム ROI を補正し、
        振動中も解析を継続できるようにする。
        """
        dx, dy = offset
        return BoardRegion(
            x=base.x + int(round(dx)),
            y=base.y + int(round(dy)),
            width=base.width,
            height=base.height,
        )

    def _resolve_tier1_threshold(
        self, visible_row: int, col: int,
    ) -> float:
        """tier 1 (EXTREME) threshold をセル位置に応じて返す。

        優先順位:
          1. pre_capture_mode = True → BG_EXTREME_THRESHOLD_PRE_CAPTURE (= 0.0)
             bg_fp 未採取期間は tier 1 をスキップして HSV-only 経路に倒す (I1 対応 A)。
          2. キャラ背景隣接エリア (軸 3-b, Phase L) → BG_EXTREME_THRESHOLD_LEFT_UPPER
             1P: col=0,1 (= 画面左端、 キャラ背景隣接)
          3. その他 → BG_EXTREME_THRESHOLD_DEFAULT (_bg_extreme_threshold)

        Args:
            visible_row: 表示行インデックス (0〜VISIBLE_ROWS-1)。
            col: 列インデックス (0〜BOARD_COLS-1)。
        """
        if self._pre_capture_mode:
            return BG_EXTREME_THRESHOLD_PRE_CAPTURE
        is_outer_edge = (
            visible_row >= BG_LEFT_UPPER_VISIBLE_ROW_MIN
            and col <= BG_LEFT_UPPER_COL_MAX
        )
        return (
            BG_EXTREME_THRESHOLD_LEFT_UPPER if is_outer_edge
            else self._bg_extreme_threshold
        )

    def _is_empty_tier1(
        self,
        bg_cell: "CellFingerprint | CellPatchFingerprint",
        cur_patch_hsv: np.ndarray,
        cur_fp: "CellFingerprint",
        visible_row: int,
        col: int,
    ) -> bool:
        """tier 1 (EXTREME) 空判定。案 d の NCC 経路と従来の距離経路を切り替える。

        PatchBackgroundFingerprint の場合は NCC 比較、
        BackgroundFingerprint の場合は従来の距離閾値比較を行う。

        Args:
            bg_cell: 背景セル FP (CellFingerprint or CellPatchFingerprint)。
            cur_patch_hsv: 現在フレームのセルパッチ HSV (float32 or uint8)。
            cur_fp: 現在フレームの CellFingerprint (median 3 値)。
            visible_row: 表示行インデックス (0〜VISIBLE_ROWS-1)。
            col: 列インデックス (0〜BOARD_COLS-1)。

        Returns:
            True = 空 (背景と同じ)、False = ぷよあり (次 tier に進む)。
        """
        from src.background_fingerprint import (
            BG_PATCH_VALID_V_MIN,
            CellPatchFingerprint,
            is_empty_by_patch_fp,
        )
        if isinstance(bg_cell, CellPatchFingerprint):
            # 第一層ガード: bg パッチが採取失敗ゼロパッチ (V median 極小) なら
            # NCC を実行せず False (= 非 EMPTY) を返す。
            # これにより「採取失敗パッチが FALLBACK=1.0 → 強制 EMPTY」を防ぐ。
            # 正当な均一 EMPTY セル (明るい平坦背景) は V median が
            # BG_PATCH_VALID_V_MIN (5.0) を超えるため従来通り NCC 経路に進む。
            bg_v_med = float(np.median(bg_cell.patch_hsv[:, :, 2]))
            if bg_v_med < BG_PATCH_VALID_V_MIN:
                return False
            cur_cell_patch = CellPatchFingerprint(
                patch_hsv=cur_patch_hsv.astype(np.float32),
            )
            # NCC sweep: None なら is_empty_by_patch_fp が PATCH_NCC_EMPTY_THRESHOLD を使用
            if self._patch_ncc_threshold is not None:
                return is_empty_by_patch_fp(
                    cur_cell_patch, bg_cell, threshold=self._patch_ncc_threshold,
                )
            return is_empty_by_patch_fp(cur_cell_patch, bg_cell)
        # 従来の距離閾値比較 (BackgroundFingerprint 経路)
        tier1_threshold = self._resolve_tier1_threshold(visible_row, col)
        dist = cur_fp.distance_to(bg_cell)
        return dist < tier1_threshold

    def _get_static_mask_for_region(
        self, region: BoardRegion,
    ) -> "StaticBoardMask | None":
        """region に対応する StaticBoardMask を返す (P1 / P2 を判別)。"""
        if region is self._p1_region:
            return self._static_mask_p1
        if region is self._p2_region:
            return self._static_mask_p2
        # シフト済 region: 中央線で判別
        if region.x + region.width / 2 < 960:
            return self._static_mask_p1
        return self._static_mask_p2

    def _is_empty_static_mask(
        self,
        frame: np.ndarray,
        region: BoardRegion,
        visible_row: int,
        col: int,
        cur_patch_hsv: np.ndarray,
    ) -> bool:
        """T4: StaticBoardMask + AND ガードによる空判定。

        設計:
          A = StaticBoardMask の diff < STATIC_BG_DIFF_THRESHOLD (= 背景と同じ)
          D = HSV 各色 range に hit (= 色あり signal)
          戻り値 = A AND NOT D

        「ぷよっぽい」 信号が 1 つでもあれば EMPTY 化しない (= fail-silent 禁止)。
        StaticBoardMask が未設定なら常に False を返す (= 判定スキップ)。

        Args:
            frame: 現在フレーム (BGR)。
            region: 盤面領域。
            visible_row: 可視行インデックス (0 〜 VISIBLE_ROWS-1)。
            col: 列インデックス (0 〜 BOARD_COLS-1)。
            cur_patch_hsv: 現フレームのセルパッチ HSV (float32 or uint8)。

        Returns:
            True = 「背景と同じかつ色なし」 → EMPTY 化してよい。
            False = 判定スキップ (従来経路に委ねる)。
        """
        from src.background_fingerprint import STATIC_BG_DIFF_THRESHOLD
        static_mask = self._get_static_mask_for_region(region)
        if static_mask is None:
            return False
        # A: pixel-level diff < 閾値
        # _cell_bgr_patch の bg は static_mask.bg_roi の座標から切り出す
        x1, y1, x2, y2 = region.cell_sample_rect(
            visible_row + HIDDEN_ROWS, col,
        )
        img_h, img_w = frame.shape[:2]
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(y1 + 1, min(y2, img_h))
        cur_bgr = frame[y1:y2, x1:x2].astype(np.float32)
        bg_roi = static_mask.bg_roi
        bg_h, bg_w = bg_roi.shape[:2]
        bx1 = max(0, min(x1, bg_w - 1))
        bx2 = max(bx1 + 1, min(x2, bg_w))
        by1 = max(0, min(y1, bg_h - 1))
        by2 = max(by1 + 1, min(y2, bg_h))
        bg_patch = bg_roi[by1:by2, bx1:bx2].astype(np.float32)
        if cur_bgr.size == 0 or bg_patch.size == 0:
            return False
        # shape 不一致はリサイズ (region ずれ対策)
        if cur_bgr.shape != bg_patch.shape:
            bg_patch = cv2.resize(
                bg_patch.astype(np.float32),
                (cur_bgr.shape[1], cur_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        diff_max = float(np.max(np.abs(cur_bgr - bg_patch)))
        if diff_max >= STATIC_BG_DIFF_THRESHOLD:
            return False  # A = False (差分大 = ぷよ存在可能性)
        # D: HSV 各色 range に hit するか (AND ガード)
        if cur_patch_hsv is None or cur_patch_hsv.size == 0:
            return True  # A=True, D=False → EMPTY 化
        h_med = int(np.median(cur_patch_hsv[:, :, 0]))
        s_med = int(np.median(cur_patch_hsv[:, :, 1]))
        v_med = int(np.median(cur_patch_hsv[:, :, 2]))
        scale = getattr(
            getattr(self._classifier, "_hsv", self._classifier),
            "_s_min_scale", 1.0,
        )
        for ranges in self._classifier._ranges.values() if hasattr(
            self._classifier, "_ranges",
        ) else []:
            for rng in ranges:
                eff_s = int(rng.s_min * scale) if scale < 1.0 else rng.s_min
                if (
                    rng.h_min <= h_med <= rng.h_max
                    and eff_s <= s_med <= rng.s_max
                    and rng.v_min <= v_med <= rng.v_max
                ):
                    return False  # D=True → EMPTY 化キャンセル
        return True  # A=True, D=False → EMPTY 化

    def read_board(
        self,
        frame: np.ndarray,
        region: BoardRegion,
        hsv_full: np.ndarray | None = None,
        skip_tier1: bool = False,
    ) -> Board:
        """
        フレームから指定領域の盤面を読み取る。

        隠し段 (row 0〜HIDDEN_ROWS-1) は画面外のため直接は見えないが、
        物理ルール (重力) による推論を適用:
          - 可視最上段 (row HIDDEN_ROWS) が空の列 → 隠し段も空 (確定)
          - 可視最上段に puyo がある列 → 隠し段は UNKNOWN (回し入れの可能性)

        Args:
            frame: BGR形式のフレーム画像 (H×W×3 のnumpy配列)。
            region: 読み取る盤面の領域 (可視領域のみ)。
            hsv_full: 事前計算済み HSV 全画像 (省略時は内部で変換)。
            skip_tier1: True のとき tier1 (bg_fp NCC / 距離による無条件 EMPTY 化)
                をスキップする。NON-STABLE → STABLE 遷移直後の N frame に使用し、
                ツモ着地直後の cell を tier1 が誤 EMPTY 化するのを防ぐ。
                HSV + CNN の通常判定は走るので背景誤認のリスクは小さい。

        Returns:
            Board: 読み取った盤面データ。
        """
        img_h, img_w = frame.shape[:2]
        board = Board()

        # 背景 FP があれば「空セル先判定」用に取得
        bg_fp = self._bg_fp_for_region(region)
        if bg_fp is not None:
            from src.background_fingerprint import (
                CellFingerprint,
                CellPatchFingerprint,
                is_empty_by_fp,
            )
            # Z-3C: hsv_full を呼び出し側から受け取れば cvtColor を回避
            if hsv_full is None:
                hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            hsv_full = None

        # 可視領域を画像分類 (Z-3C: バッチ化)
        has_position_api = hasattr(self._classifier, "classify_at")
        has_batch_api = (
            not has_position_api
            and hasattr(self._classifier, "classify_batch")
        )
        # 1st pass: patch 切り出し + 背景 FP 早期判定
        # cycle 34 (2026-05-20): bg_fp 距離も track して 2nd pass (= HybridClassifier
        # classify_batch) に渡す → CNN logit に soft prior 適用
        # 案 R3 改: hsv_patch も保持して 2nd pass のプロファイルチェックで再利用
        cells_to_classify: list[
            tuple[int, int, np.ndarray, float | None, np.ndarray | None]
        ] = []
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            visible_row = row - HIDDEN_ROWS
            for col in range(BOARD_COLS):
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                x1 = max(0, min(x1, img_w - 1))
                x2 = max(x1 + 1, min(x2, img_w))
                y1 = max(0, min(y1, img_h - 1))
                y2 = max(y1 + 1, min(y2, img_h))
                patch = frame[y1:y2, x1:x2]
                # 背景 FP 早期 EMPTY 判定 (cycle 33 tiered 化 + cycle 34 soft prior)
                # tier 1 (cycle 33): 距離 < EXTREME 閾値 → 無条件 EMPTY
                # tier 2 (cycle 19 既存): 距離 < _bg_threshold → AND 条件で early empty
                # tier 3 (cycle 34): それ以外 → CNN 経路 + bg_distance を soft prior に
                cell_bg_distance: float | None = None
                # 案 R3 改: 1st pass で hsv_patch を記録し 2nd pass でプロファイル検査に再利用
                cell_hsv_patch: np.ndarray | None = None
                if bg_fp is not None and hsv_full is not None and patch.size > 0:
                    hsv_patch = hsv_full[y1:y2, x1:x2]
                    if hsv_patch.size > 0:
                        cell_hsv_patch = hsv_patch  # 2nd pass 再利用用に保持
                        h_med = int(np.median(hsv_patch[:, :, 0]))
                        s_med = int(np.median(hsv_patch[:, :, 1]))
                        v_med = int(np.median(hsv_patch[:, :, 2]))
                        cur_fp = CellFingerprint(h_med, s_med, v_med)
                        bg_cell = bg_fp.cell_at(visible_row, col)
                        dist = cur_fp.distance_to(bg_cell)
                        cell_bg_distance = float(dist)
                        # T4: StaticBoardMask AND ガード (既存 tier 1 より先に評価)
                        # A (diff < 閾値) AND NOT D (HSV 色あり) の場合のみ EMPTY 化。
                        # 片方でも「ぷよっぽい」 なら skip して従来経路に流す。
                        if self._is_empty_static_mask(
                            frame, region, visible_row, col, hsv_patch,
                        ):
                            board.set(row, col, COLOR_EMPTY)
                            continue
                        # tier 1: extreme close = 確実な背景 (案 d: NCC or 距離)
                        # PatchBackgroundFingerprint の場合は _is_empty_tier1 が NCC 判定。
                        # BackgroundFingerprint の場合は従来の距離閾値比較。
                        # PatchBackgroundFingerprint では cell_at_patch を使う
                        # skip_tier1=True (NON-STABLE→STABLE 遷移直後) はスキップ:
                        # ツモ着地直後の cell を誤 EMPTY 化しない (= 失敗教訓遵守)。
                        # HSV + CNN の通常判定は続行するため背景誤認リスクは小さい。
                        from src.background_fingerprint import PatchBackgroundFingerprint
                        if isinstance(bg_fp, PatchBackgroundFingerprint):
                            bg_cell_for_tier1 = bg_fp.cell_at_patch(visible_row, col)
                        else:
                            bg_cell_for_tier1 = bg_cell
                        if not skip_tier1 and self._is_empty_tier1(
                            bg_cell_for_tier1, hsv_patch, cur_fp,
                            visible_row, col,
                        ):
                            # 案 P2: 白ハイライト blob override
                            # tier1 が EMPTY 判定しても、ぷよ固有の白ハイライト円が
                            # あれば「本物ぷよあり」として classify に進む
                            if (
                                self._use_highlight_override
                                and _has_puyo_highlight(hsv_patch)
                            ):
                                pass  # EMPTY 却下 → cells_to_classify に流れる
                            else:
                                board.set(row, col, COLOR_EMPTY)
                                continue
                        # tier 2: AND 条件 (= cycle 19 既存)
                        if is_empty_by_fp(
                            cur_fp, bg_cell, threshold=self._bg_threshold,
                        ):
                            hsv_target = getattr(
                                self._classifier, "_hsv", self._classifier,
                            )
                            hsv_only_color = COLOR_EMPTY
                            try:
                                hsv_only_color = int(
                                    hsv_target.classify(patch),
                                )
                            except Exception:
                                hsv_only_color = COLOR_EMPTY
                            if hsv_only_color in (
                                COLOR_EMPTY, COLOR_UNKNOWN,
                            ):
                                board.set(row, col, COLOR_EMPTY)
                                continue
                cells_to_classify.append(
                    (row, col, patch, cell_bg_distance, cell_hsv_patch),
                )

        # 2nd pass: バッチ classify (HybridClassifier で 5-20x 高速化)
        # cycle 34: bg_distance を classify_batch に渡して CNN logit soft prior
        if has_batch_api and cells_to_classify:
            patches = [p for _, _, p, _, _ in cells_to_classify]
            distances = [d for _, _, _, d, _ in cells_to_classify]
            try:
                colors = self._classifier.classify_batch(
                    patches, bg_distances=distances,
                )
            except TypeError:
                # backwards compat: 古い classify_batch は bg_distances 未対応
                colors = self._classifier.classify_batch(patches)
            for (row, col, patch, _, hsv_p), color in zip(
                cells_to_classify, colors,
            ):
                # 案 R3 改: プロファイル不一致の色を EMPTY 化 (下段方向のみ)
                color = self._apply_profile_filter(color, hsv_p, patch)
                # classify_batch は UI mask 適用済 (HybridClassifier 内で処理)
                board.set(row, col, color)
        else:
            # フォールバック: 個別 classify
            for row, col, patch, _, hsv_p in cells_to_classify:
                visible_row = row - HIDDEN_ROWS
                if has_position_api:
                    color = self._classifier.classify_at(
                        patch, visible_row, col,
                    )
                else:
                    color = self._classifier.classify(patch)
                if (
                    self._ui_matcher is not None
                    and color != COLOR_EMPTY
                    and patch.size > 0
                    and self._ui_matcher.is_ui(patch)
                ):
                    color = COLOR_EMPTY
                # 案 R3 改: プロファイル不一致の色を EMPTY 化 (下段方向のみ)
                color = self._apply_profile_filter(color, hsv_p, patch)
                board.set(row, col, color)

        # V3.1: テロップ被覆セルを COLOR_UNKNOWN に倒す (浮遊削除前)
        # キャッシュした bbox を使う (read_both_boards で 1 度だけ検出)
        if (
            self._telop_detector is not None
            and self._cached_telop_bbox is not None
        ):
            from src.telop_detector import TelopDetector
            covered = TelopDetector.cells_covered_by_bbox(
                self._cached_telop_bbox, region,
            )
            for row, col in covered:
                board.set(row, col, COLOR_UNKNOWN)

        # Phase T サイクル 5: 推論強化
        # 浮遊ぷよ (UI オーバーレイ・連鎖アニメ・落下中ぷよの誤検出) を除去
        if self._apply_inference:
            from src.board_rules import clear_floating_above_gap
            board = clear_floating_above_gap(
                board, min_gap=self._floating_min_gap, skip_hidden=True,
            )

        # 隠し段を物理推論で確定 or UNKNOWN にする
        self._infer_hidden_rows(board)

        return board

    def _apply_profile_filter(
        self,
        color: int,
        hsv_patch: np.ndarray | None,
        bgr_patch: np.ndarray,
    ) -> int:
        """案 R3 改: classify 結果がプロファイルに合致しない場合 EMPTY 化する。

        設計制約:
          - 下段方向のみ: classify が色を返した場合に EMPTY 化 (上段救済は禁止)
          - hsv_patch が None (= bg_fp 未採取期間) の場合は bgr_patch から計算
          - COLOR_EMPTY / COLOR_UNKNOWN は通過させる (変更しない)

        Args:
            color: classify が返した色コード
            hsv_patch: 1st pass で計算済の HSV パッチ (None なら再計算)
            bgr_patch: BGR パッチ (hsv_patch が None のとき変換元)

        Returns:
            int: 確認済み色コード (プロファイル不一致なら COLOR_EMPTY)
        """
        # EMPTY / UNKNOWN は変更しない
        if color in (COLOR_EMPTY, COLOR_UNKNOWN):
            return color
        # プロファイル DB なし → 無効 (既存挙動維持)
        if self._puyo_profile_db is None:
            return color
        # HSV 中央値を取得 (1st pass 再利用 or 再計算)
        if hsv_patch is not None and hsv_patch.size > 0:
            h_med = int(np.median(hsv_patch[:, :, 0]))
            s_med = int(np.median(hsv_patch[:, :, 1]))
            v_med = int(np.median(hsv_patch[:, :, 2]))
        elif bgr_patch.size > 0:
            _hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
            h_med = int(np.median(_hsv[:, :, 0]))
            s_med = int(np.median(_hsv[:, :, 1]))
            v_med = int(np.median(_hsv[:, :, 2]))
        else:
            # パッチが空 → 判定不能、保守的に通過
            return color
        # プロファイル距離チェック: 不一致なら EMPTY 化
        if not self._puyo_profile_db.is_puyo_like(color, h_med, s_med, v_med):
            return COLOR_EMPTY
        return color

    # T-v2 系融合判定は archive/legacy_phase_t_v2/ に移動 (Phase U で廃止)。

    @staticmethod
    def _infer_hidden_rows(board: Board) -> None:
        """
        重力ルールから隠し段の状態を推論する (in-place)。

        各列について、可視最上段 (row HIDDEN_ROWS) の状態で判定:
          - 空 → 隠し段の同列も空 (重力により落下しているはず)
          - 非空 → 隠し段は UNKNOWN (回し入れで puyo がある可能性)

        Args:
            board: 推論対象の盤面 (可視領域は既に読み取り済み)。
        """
        top_visible_row = HIDDEN_ROWS
        for col in range(BOARD_COLS):
            top_cell = board.get(top_visible_row, col)
            if top_cell == COLOR_EMPTY:
                # 重力より隠し段も空 (確定)
                for hidden_row in range(HIDDEN_ROWS):
                    board.set(hidden_row, col, COLOR_EMPTY)
            else:
                # 回し入れの可能性あり → UNKNOWN
                for hidden_row in range(HIDDEN_ROWS):
                    board.set(hidden_row, col, COLOR_UNKNOWN)

    def read_both_boards(
        self,
        frame: np.ndarray,
        p1_roi_offset: tuple[float, float] = (0.0, 0.0),
        p2_roi_offset: tuple[float, float] = (0.0, 0.0),
        skip_tier1_1p: bool = False,
        skip_tier1_2p: bool = False,
    ) -> tuple[Board, Board]:
        """
        フレームから1P・2P両方の盤面を読み取る。

        Args:
            frame: BGR形式のフレーム画像。
            p1_roi_offset: 1P 盤面の ROI 補正シフト (dx, dy) px (T-v2-B)。
                振動検出器が返した dx, dy を渡すと、毎フレーム ROI を補正
                できる。デフォルト (0, 0) で従来挙動。
            p2_roi_offset: 2P 盤面の ROI 補正シフト (dx, dy) px。
            skip_tier1_1p: True のとき 1P 側 tier1 をスキップ (NON-STABLE→STABLE 遷移直後用)。
            skip_tier1_2p: True のとき 2P 側 tier1 をスキップ (NON-STABLE→STABLE 遷移直後用)。

        Returns:
            tuple[Board, Board]: (1P盤面, 2P盤面) のタプル。
        """
        # キャリブレーションは 1920x1080 前提。異なる解像度は自動リサイズ。
        # C (2026-05-11): 拡大 (360p→1080p 等) は INTER_LANCZOS4 で puyo
        # 境界をシャープに保つ. 縮小 (例 4K→1080p) は INTER_AREA が最良.
        h, w = frame.shape[:2]
        if (h, w) != (1080, 1920):
            interp = cv2.INTER_LANCZOS4 if h < 1080 else cv2.INTER_AREA
            frame = cv2.resize(frame, (1920, 1080), interpolation=interp)
        # 試合状態判定 (試合中以外は両盤面 EMPTY)
        if self._match_state_detector is not None:
            from src.match_state import MatchState
            state = self._match_state_detector.detect(frame)
            if state.state != MatchState.IN_MATCH:
                return Board(), Board()
        # V3.1: テロップ検出 (フレーム単位で 1 度。read_board が cached bbox を使う)
        if self._telop_detector is not None:
            telop_res = self._telop_detector.detect(frame)
            self._cached_telop_bbox = telop_res.bbox if telop_res.is_visible else None
        else:
            self._cached_telop_bbox = None
        if p1_roi_offset == (0.0, 0.0):
            p1_region = self._p1_region
        else:
            p1_region = self._shifted_region(self._p1_region, p1_roi_offset)
        if p2_roi_offset == (0.0, 0.0):
            p2_region = self._p2_region
        else:
            p2_region = self._shifted_region(self._p2_region, p2_roi_offset)
        # Z-3C: BG FP が両 region 共通で必要 → hsv_full を 1 度計算して共有
        hsv_full: np.ndarray | None = None
        if (self._bg_fp_for_region(p1_region) is not None
                or self._bg_fp_for_region(p2_region) is not None):
            hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        board_1p = self.read_board(frame, p1_region, hsv_full=hsv_full, skip_tier1=skip_tier1_1p)
        board_2p = self.read_board(frame, p2_region, hsv_full=hsv_full, skip_tier1=skip_tier1_2p)
        return board_1p, board_2p

    def _get_hsv_classifier(self) -> "ColorClassifier | None":
        """内部の HSV-only 分類器を取得する。

        HybridClassifier を使用している場合は _hsv を返す。
        ColorClassifier の場合はそのまま返す。
        どちらでもなければ None を返す。
        """
        clf = self._classifier
        # HybridClassifier は _hsv 属性に ColorClassifier を保持する
        if hasattr(clf, "_hsv"):
            return clf._hsv  # type: ignore[return-value]
        if isinstance(clf, ColorClassifier):
            return clf
        return None

    def read_board_hsv_only(
        self,
        frame: np.ndarray,
        region: BoardRegion,
    ) -> Board:
        """HSV-only 分類器のみで盤面を読み取る (T2 CNN+HSV 合意 yield 用)。

        HybridClassifier の CNN を使わず HSV ColorClassifier だけで判定する。
        bg_fp / tier1 / telop マスク等の後処理は行わない簡易版。
        目的: T2 の「CNN と HSV が両方 prev_stable と異なる同色を支持」判定専用。
        パフォーマンス: CNN 推論を省略するため通常の read_board より高速。

        Returns:
            Board: HSV-only 判定盤面 (visible 行のみ。隠し段は EMPTY / UNKNOWN 推論)。
        """
        hsv_clf = self._get_hsv_classifier()
        if hsv_clf is None:
            # HSV 分類器を取得できなければ空 Board を返す (= yield 判定に使わない)
            return Board()
        h, w = frame.shape[:2]
        if (h, w) != (1080, 1920):
            interp = cv2.INTER_LANCZOS4 if h < 1080 else cv2.INTER_AREA
            frame = cv2.resize(frame, (1920, 1080), interpolation=interp)
        board = Board()
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                x1 = max(0, min(int(x1), w - 1))
                x2 = max(x1 + 1, min(int(x2), w))
                y1 = max(0, min(int(y1), h - 1))
                y2 = max(y1 + 1, min(int(y2), h))
                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    board.set(row, col, COLOR_EMPTY)
                    continue
                board.set(row, col, hsv_clf.classify(patch))
        # 隠し段を物理推論で確定 or UNKNOWN にする
        self._infer_hidden_rows(board)
        return board

    def read_both_boards_hsv(
        self,
        frame: np.ndarray,
    ) -> "tuple[Board, Board]":
        """1P/2P 両方の HSV-only 盤面を返す (T2 CNN+HSV 合意 yield 用)。

        Args:
            frame: BGR フレーム画像。

        Returns:
            tuple[Board, Board]: (1P HSV-only 盤面, 2P HSV-only 盤面)。
            HSV 分類器が取得できない場合は空 Board のタプルを返す。
        """
        if self._get_hsv_classifier() is None:
            return Board(), Board()
        board_1p = self.read_board_hsv_only(frame, self._p1_region)
        board_2p = self.read_board_hsv_only(frame, self._p2_region)
        return board_1p, board_2p

    def debug_frame(
        self, frame: np.ndarray, region: BoardRegion
    ) -> np.ndarray:
        """
        各セルのサンプリング位置をフレームに描画して返す (デバッグ用)。
        可視領域 (row=HIDDEN_ROWS〜) のみ描画。

        Args:
            frame: BGR形式のフレーム画像。
            region: デバッグ対象の盤面領域。

        Returns:
            np.ndarray: サンプリング矩形を描画した画像。
        """
        debug_img = frame.copy()
        for row in range(HIDDEN_ROWS, BOARD_ROWS):
            for col in range(BOARD_COLS):
                x1, y1, x2, y2 = region.cell_sample_rect(row, col)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return debug_img
