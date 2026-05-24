"""
予告お邪魔ぷよ検出モジュール

ぷよぷよeスポーツ画面で各盤面 (1P/2P) の上部に表示される
攻撃予告アイコン列を画像認識し、種類別 (small/line/rock/moon/crown/...)
の検出結果と総おじゃま個数を返す。

ROI:
    1920×1080 フレーム前提、盤面 BoardRegion(x,y) と整合した
    上部 55 px のストリップを 6 セル等分して個別判定する。

検出方式:
    1. ROI から 6 セル切り出し
    2. 各セルで「アイコンが存在するか」を中央パッチの HSV 統計で判定
    3. 存在時は HSV 色域 + 任意のテンプレ NCC で 7 種から最尤分類
       (テンプレが無い種類は HSV のみで best-effort)

注意:
    予告アイコン表示は実際には N 個並ぶ場合に左寄せで詰めて配置されるが、
    本モジュールは 6 セル等分で評価する簡易実装。少数アイコン時の
    精密な位置同定は将来の課題 (現状は「何個・合計いくつ」が分かれば十分)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# ============================
# ROI 定数 (1920×1080 前提)
# ============================
FRAME_WIDTH: int = 1920
FRAME_HEIGHT: int = 1080

# 1P / 2P 盤面の左端 X (calibration_video01.json と一致)
P1_BOARD_X: int = 282
P2_BOARD_X: int = 1258
BOARD_WIDTH: int = 384

# 予告ストリップの縦範囲
WARNING_TOP_Y: int = 105
WARNING_BOTTOM_Y: int = 160
WARNING_HEIGHT: int = WARNING_BOTTOM_Y - WARNING_TOP_Y

# 6 セル等分時のセル幅と中央サンプル幅
CELL_COUNT: int = 6
CELL_WIDTH: int = BOARD_WIDTH // CELL_COUNT  # 64
ICON_SAMPLE_HALF: int = 18                   # 中央 36×36 を判定パッチに
ICON_CROP_HALF: int = 30                     # 切り出し全体は 60×55

# ============================
# アイコン種類定義
# ============================
ICON_EMPTY: str = "empty"
ICON_SMALL: str = "small"
ICON_LINE: str = "line"
ICON_ROCK: str = "rock"
ICON_MOON: str = "moon"
ICON_CROWN: str = "crown"
ICON_BIG_CROWN: str = "big_crown"
ICON_SUPERCROWN: str = "supercrown"

# 各種類のおじゃま個数換算
COUNT_TABLE: dict[str, int] = {
    ICON_EMPTY: 0,
    ICON_SMALL: 1,
    ICON_LINE: 6,
    ICON_ROCK: 30,
    ICON_MOON: 60,
    ICON_CROWN: 180,
    ICON_BIG_CROWN: 360,
    ICON_SUPERCROWN: 720,
}

# ============================
# HSV 判定閾値
# ============================
# 中央パッチの「特徴ピクセル比率」でアイコン存在 + 種類を同時に判定する。
# 観測値 (data/verify/strips/ 各種フレーム):
#   空セル背景 (青/赤): gray=0.04~0.14, yellow=0.0~0.0, dark=0.0
#   rock (灰色岩): gray=0.65~0.72
#   moon (黄色): yellow=0.5~0.65 (Hue 18~45 で S>=100)
#   small (小さい黒玉): dark=0.4 以上, S 低めもあり
GRAY_S_MAX: int = 60                  # gray判定の最大彩度
GRAY_PIXEL_RATIO_MIN: float = 0.40    # rock 判定の最小灰色画素比率

YELLOW_H_MIN: int = 18
YELLOW_H_MAX: int = 45
YELLOW_S_MIN: int = 100
YELLOW_PIXEL_RATIO_MIN: float = 0.30  # moon 判定の最小黄色画素比率

DARK_V_MAX: int = 70                  # 暗ピクセル(=黒)の輝度上限
DARK_PIXEL_RATIO_MIN: float = 0.30    # small/line 判定の最小暗画素比率
LINE_DARK_PIXEL_RATIO: float = 0.55   # line とみなす暗画素比率

# アイコンの実体性チェック (cell 全体の単色背景を弾く)
# 完全に均一な単色フレーム (合成テストや非試合中) は v_std=0 で empty
# 実フレームのアイコンは境界エッジで std≥17 あるため 8.0 で十分判別可能
PRESENCE_V_STD_MIN: float = 12.0

# 高彩度 + 赤紫 = crown 系 (best-effort)
CROWN_S_MIN: int = 120
CROWN_PIXEL_RATIO_MIN: float = 0.30
CROWN_H_BANDS: tuple[tuple[int, int], ...] = (
    (0, 12), (160, 180), (130, 160),  # 赤・濃赤・紫帯
)

# ============================
# テンプレ照合パラメータ
# ============================
TEMPLATE_DIR_DEFAULT: Path = Path("models/ui_templates/ojama")
TEMPLATE_NCC_THRESHOLD: float = 0.55

# 種類名 → テンプレファイル名 (拡張子省略)
TEMPLATE_FILES: dict[str, str] = {
    ICON_ROCK: "rock.png",
    ICON_MOON: "moon.png",
    ICON_LINE: "line.png",
    ICON_CROWN: "crown.png",
    ICON_BIG_CROWN: "big_crown.png",
    ICON_SUPERCROWN: "supercrown.png",
    ICON_SMALL: "small.png",
}


# ============================
# データクラス
# ============================


@dataclass(frozen=True)
class OjamaIcon:
    """予告アイコン 1 個分の情報。

    Attributes:
        icon_type: ICON_* のいずれかの種類名。
        count_value: 個数換算値 (1, 6, 30, 60, 180, 360, 720)。
    """
    icon_type: str
    count_value: int


@dataclass(frozen=True)
class OjamaWarningResult:
    """1 サイドの予告お邪魔状態。

    Attributes:
        side: "1P" または "2P"。
        icons: 検出されたアイコン群 (空セルは含めない)。
        total_count: 合計お邪魔個数 (アイコン値の和)。
    """
    side: str
    icons: tuple[OjamaIcon, ...]
    total_count: int


# ============================
# テンプレ管理
# ============================


def _load_templates(
    template_dir: Path,
) -> dict[str, list[np.ndarray]]:
    """テンプレ画像 (BGR) を kind → サブテンプレリストで読み込む。

    2026-04-27: クラスタリングで生成されたサブテンプレ (例: rock_2.png) も
    同 kind の追加テンプレとして読む。NCC 照合時は全サブテンプレで最大スコアを採用。
    """
    templates: dict[str, list[np.ndarray]] = {}
    if not template_dir.exists():
        return templates
    for kind, fname in TEMPLATE_FILES.items():
        # メインテンプレ (例: rock.png)
        main_path = template_dir / fname
        sub_list: list[np.ndarray] = []
        if main_path.exists():
            img = cv2.imread(str(main_path))
            if img is not None:
                sub_list.append(img)
        # サブテンプレ (例: rock_2.png, rock_3.png ...)
        stem = main_path.stem
        for sub in template_dir.glob(f"{stem}_*.png"):
            sub_img = cv2.imread(str(sub))
            if sub_img is not None:
                sub_list.append(sub_img)
        if sub_list:
            templates[kind] = sub_list
    return templates


# ============================
# OjamaWarningDetector
# ============================


class OjamaWarningDetector:
    """予告お邪魔アイコンを検出するクラス。

    Usage:
        det = OjamaWarningDetector()
        p1, p2 = det.detect(frame)
        print(p1.total_count, [i.icon_type for i in p1.icons])
    """

    def __init__(
        self,
        template_dir: Path = TEMPLATE_DIR_DEFAULT,
        ncc_threshold: float = TEMPLATE_NCC_THRESHOLD,
        use_cnn: bool = True,
        cnn_confidence_min: float = 0.5,
    ) -> None:
        """テンプレ画像 (任意) を読み込む。

        Args:
            template_dir: テンプレ画像ディレクトリ
            ncc_threshold: NCC 採用閾値
            use_cnn: 訓練済み CNN がある場合は CNN を主軸に判定 (推奨)
            cnn_confidence_min: CNN 出力の confidence 下限。これ未満は
                テンプレ NCC + HSV にフォールバック
        """
        self._templates = _load_templates(template_dir)
        self._ncc_threshold = ncc_threshold
        self._cnn_confidence_min = float(cnn_confidence_min)
        self._cnn = None
        if use_cnn:
            try:
                from src.ojama_cnn import load_cnn
                self._cnn = load_cnn()
            except Exception:
                self._cnn = None

    # ---- 公開メソッド ---------------------------------------------

    def detect(
        self, frame: np.ndarray,
    ) -> tuple[OjamaWarningResult, OjamaWarningResult]:
        """1P / 2P の予告状態を返す。"""
        if frame is None or frame.ndim != 3:
            return self._empty_pair()
        if frame.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
            return self._empty_pair()
        p1 = self._detect_side(frame, P1_BOARD_X, "1P")
        p2 = self._detect_side(frame, P2_BOARD_X, "2P")
        return p1, p2

    # ---- 内部 -----------------------------------------------------

    def _detect_side(
        self, frame: np.ndarray, board_x: int, side: str,
    ) -> OjamaWarningResult:
        """指定盤面 1 つ分の予告アイコン列を検出する。"""
        icons: list[OjamaIcon] = []
        for i in range(CELL_COUNT):
            cell = self._extract_cell(frame, board_x, i)
            kind = self._classify_cell(cell)
            if kind == ICON_EMPTY:
                continue
            icons.append(OjamaIcon(
                icon_type=kind, count_value=COUNT_TABLE[kind],
            ))
        total = sum(ic.count_value for ic in icons)
        return OjamaWarningResult(
            side=side, icons=tuple(icons), total_count=total,
        )

    def _extract_cell(
        self, frame: np.ndarray, board_x: int, idx: int,
    ) -> np.ndarray:
        """6 セル中 idx 番目のアイコン候補領域を切り出す。"""
        cx = board_x + int((idx + 0.5) * CELL_WIDTH)
        x1 = max(0, cx - ICON_CROP_HALF)
        x2 = min(FRAME_WIDTH, cx + ICON_CROP_HALF)
        return frame[WARNING_TOP_Y:WARNING_BOTTOM_Y, x1:x2]

    def _classify_cell(self, cell: np.ndarray) -> str:
        """1 セルのアイコン種類を判定する。

        2026-04-27: CNN を主軸 (val_acc 0.981) に切り替え。CNN 未配置 or
        confidence 未達のとき テンプレ NCC + HSV フォールバックに倒す。

        手順:
            1. cell の V 標準偏差で empty 判定
            2. CNN モデルあれば中央 36×36 を CNN 推論、confidence ≥ 閾値で採用
            3. CNN 未配置 or confidence 不足 → テンプレ NCC
            4. テンプレ NCC 未達 → HSV 特徴量
        """
        if cell.size == 0:
            return ICON_EMPTY
        feats = self._patch_features(cell)
        if feats["v_std"] < PRESENCE_V_STD_MIN:
            return ICON_EMPTY
        # CNN 推論 (ある場合)
        if self._cnn is not None:
            patch = self._extract_center_patch(cell)
            if patch is not None:
                kind, conf = self._cnn.predict_class(patch)
                if conf >= self._cnn_confidence_min:
                    return kind
        # テンプレ NCC
        templ_kind = self._match_templates(cell)
        if templ_kind is not None:
            return templ_kind
        return self._classify_by_features(feats)

    @staticmethod
    def _extract_center_patch(cell: np.ndarray) -> np.ndarray | None:
        """cell から中央 36×36 を切り出す (CNN 入力用)。"""
        h, w = cell.shape[:2]
        if h < 36 or w < 36:
            return None
        cy = h // 2
        cx = w // 2
        patch = cell[cy - 18: cy + 18, cx - 18: cx + 18]
        if patch.shape[:2] != (36, 36):
            return None
        return patch

    def _patch_features(self, cell: np.ndarray) -> dict[str, float]:
        """中央パッチ + cell 全体から比率特徴を計算する。

        v_std は cell 全体で評価する。完全な単色背景 (合成画像など) でも
        cell 全体としては std≈0 だが、本物のアイコンが描かれていれば
        境界部の輝度ムラで std ≥ 数 になるためアイコン判定が成立する。
        """
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        # 中央パッチ (色種類判定用)
        h, w = hsv.shape[:2]
        cy, cx = h // 2, w // 2
        y1 = max(0, cy - ICON_SAMPLE_HALF)
        y2 = min(h, cy + ICON_SAMPLE_HALF)
        x1 = max(0, cx - ICON_SAMPLE_HALF)
        x2 = min(w, cx + ICON_SAMPLE_HALF)
        patch = hsv[y1:y2, x1:x2]
        if patch.size == 0:
            return {
                "gray": 0.0, "yellow": 0.0, "dark": 0.0,
                "crown": 0.0, "v_std": 0.0,
            }
        h_ch, s_ch, v_ch = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
        gray_ratio = float(np.mean(s_ch <= GRAY_S_MAX))
        yellow_mask = (
            (h_ch >= YELLOW_H_MIN) & (h_ch <= YELLOW_H_MAX)
            & (s_ch >= YELLOW_S_MIN)
        )
        dark_ratio = float(np.mean(v_ch <= DARK_V_MAX))
        crown_mask = np.zeros_like(h_ch, dtype=bool)
        for lo, hi in CROWN_H_BANDS:
            crown_mask |= (h_ch >= lo) & (h_ch <= hi) & (s_ch >= CROWN_S_MIN)
        # cell 全体の輝度標準偏差 (アイコン境界のムラを検出)
        v_std_full = float(np.std(hsv[:, :, 2]))
        return {
            "gray": gray_ratio,
            "yellow": float(np.mean(yellow_mask)),
            "dark": dark_ratio,
            "crown": float(np.mean(crown_mask)),
            "v_std": v_std_full,
        }

    @staticmethod
    def _classify_by_features(feats: dict[str, float]) -> str:
        """特徴比率から種類を投票する。

        判定順序:
            1. dark が極端に高ければ line/small (黒玉系が最優先)
            2. yellow が条件満たせば moon
            3. gray が条件満たせば rock
            4. dark が中程度なら small
            5. それ以外は empty

        crown 系 (crown/big_crown/supercrown) は背景の赤紫色と区別が
        困難なため、HSV 単独では判定せずテンプレ照合に頼る。
        """
        gray = feats["gray"]
        yellow = feats["yellow"]
        dark = feats["dark"]
        # dark がパッチの過半を占めるなら line (黒帯) — 灰色 rock より優先
        if dark >= LINE_DARK_PIXEL_RATIO:
            return ICON_LINE
        if yellow >= YELLOW_PIXEL_RATIO_MIN:
            return ICON_MOON
        if gray >= GRAY_PIXEL_RATIO_MIN:
            return ICON_ROCK
        if dark >= DARK_PIXEL_RATIO_MIN:
            return ICON_SMALL
        return ICON_EMPTY

    def _match_templates(self, cell: np.ndarray) -> str | None:
        """ロード済みテンプレと NCC 照合し、最大スコアの種類を返す。

        各 kind が複数のサブテンプレ (クラスタリング結果) を持つ場合、
        その中で最高 NCC スコアを採用してから kind 間で比較する。
        """
        if not self._templates:
            return None
        best_kind: str | None = None
        best_score: float = self._ncc_threshold
        for kind, templ_list in self._templates.items():
            # サブテンプレ全部で照合 → 最大値が kind の代表スコア
            kind_score = max(_ncc_score(cell, t) for t in templ_list)
            if kind_score > best_score:
                best_score = kind_score
                best_kind = kind
        return best_kind

    @staticmethod
    def _empty_pair() -> tuple[OjamaWarningResult, OjamaWarningResult]:
        """フレーム不正時に返す空ペア。"""
        return (
            OjamaWarningResult(side="1P", icons=(), total_count=0),
            OjamaWarningResult(side="2P", icons=(), total_count=0),
        )


# ============================
# テンプレ照合ヘルパー
# ============================


def _ncc_score(cell: np.ndarray, templ: np.ndarray) -> float:
    """セル画像とテンプレを NCC スライディングウィンドウで照合してスコアを返す。

    2026-04-27: 旧実装は templ を cell サイズに拡大していたため精度低下。
    templ ≤ cell のときは sliding window で最大スコアを取り、templ > cell
    のときのみ縮小してから照合する。
    """
    if cell.size == 0 or templ.size == 0:
        return 0.0
    h, w = cell.shape[:2]
    th, tw = templ.shape[:2]
    if th > h or tw > w:
        # テンプレが cell より大きい場合のみダウンスケール
        new_w = min(tw, w)
        new_h = min(th, h)
        templ = cv2.resize(templ, (new_w, new_h))
    # それ以外は sliding window で最大スコア
    res = cv2.matchTemplate(cell, templ, cv2.TM_CCOEFF_NORMED)
    return float(np.max(res))


__all__ = [
    "OjamaIcon",
    "OjamaWarningResult",
    "OjamaWarningDetector",
    "ICON_EMPTY",
    "ICON_SMALL",
    "ICON_LINE",
    "ICON_ROCK",
    "ICON_MOON",
    "ICON_CROWN",
    "ICON_BIG_CROWN",
    "ICON_SUPERCROWN",
    "COUNT_TABLE",
    "P1_BOARD_X",
    "P2_BOARD_X",
    "WARNING_TOP_Y",
    "WARNING_BOTTOM_Y",
    "CELL_COUNT",
    "CELL_WIDTH",
]
