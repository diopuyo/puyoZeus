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

# ============================
# 掛け算式 (連鎖中テロップ) 読取り定数 (2026-08-24)
# ============================
# 実測根拠 (memory reference_chain_formula_layout_2026-08-24、
# scratchpad formula_ocr_probe 634行 + 本セッション追検証 402 formula フレーム):
#   - 掛け算式は通常スコアと完全同一フォント・同一 8 桁グリッド・倍率 1.0
#   - レイアウト: 左辺 (消去数×10) = cell1-3 右詰め (1〜3桁、
#     3桁実例: c01 t=6712.9 「100×294」)、「×」= cell4 固定、
#     右辺 (ボーナス倍率) = cell5-7 右詰め (1〜3桁、実例 386/416/450)
#   - user 訂正 (2026-08-24): 左辺は 10 個同時消しで 3 桁になる (「100」)。
#     両辺とも 1〜3 桁を想定する。

# 「×」テンプレのファイル名 (DEFAULT_TEMPLATE_DIR 配下)。
# 不在なら掛け算式読取りは常に invalid (fail-safe、既存挙動へ影響なし)。
FORMULA_MULT_TEMPLATE_FILENAME: str = "formula_mult.png"

# 「×」セルの NCC 下限。実測分離: formula フレーム p05=0.868 / median=0.876、
# 通常スコアフレームの最大 0.312、グリッド外ノイズ最大 0.613 → 0.70 で余裕。
FORMULA_MULT_NCC_MIN: float = 0.70

# 掛け算式の数字セルを「数字」と判定する NCC 下限。
# formula 数字の実測 median 0.94-0.96 / 99% ≥ 0.85 に対し、空白セル
# (背景) の最大は 0.6 前後 → 0.70 で数字/空白を分離する。
FORMULA_DIGIT_NCC_MIN: float = 0.70

# 掛け算式グリッドのセル割当て (実測レイアウト、上記コメント参照)。
FORMULA_BLANK_CELL_INDEX: int = 0   # 常に空白 (左辺は最大3桁 = cell1-3)
FORMULA_LEFT_CELLS: tuple[int, ...] = (1, 2, 3)   # 左辺 右詰め
FORMULA_MULT_CELL_INDEX: int = 4                  # 「×」固定位置
FORMULA_RIGHT_CELLS: tuple[int, ...] = (5, 6, 7)  # 右辺 右詰め

# 左辺 (消去数×10) の物理制約。ぷよは 4 連結未満では消えないため
# 最小 4 個 × 10 = 40 (chain_detector.ERASURE_MIN_DROP=4 と同根)。
# 10 の倍数でない左辺は部分読み (フェード中の欠け) として棄却する。
FORMULA_LEFT_MIN: int = 40
FORMULA_LEFT_UNIT: int = 10

# 同一値の連続確認フレーム数。出現/消滅アニメ中の部分読み (実測 1.4%、
# 1〜2 フレーム持続) を除去する。機能D の CHAIN_FORMULA_CONSEC_FRAMES=2 と同根拠。
FORMULA_STEP_CONFIRM_FRAMES: int = 2

# 有効読取りが途絶えてからセッション (= 1 連鎖分の累積) を破棄するまでの秒数。
# 連鎖 1 段の表示周期は実測 ≈1.4 秒 (memory reference_chain_formula_per_step_
# 2026-08-22)。段間の遷移で読めない区間 (実測最大 ≈1.4 秒) を跨いでも
# セッションを維持しつつ、別の連鎖とは分離するための上限。
FORMULA_SESSION_RESET_SEC: float = 2.0

# 右辺 (ボーナス倍率) が減少した読取りを「新しい連鎖の開始」と受理するために
# 必要な最小消失ギャップ秒数。同一連鎖中は掛け算式が連続表示される
# (402 フレーム実測) ため、表示が途切れずに右辺が減る読取りは
# フェードアウト/遷移中の部分読み (例: 「50×386」→「50× 86」) であり棄却する。
FORMULA_NEW_SESSION_MIN_GAP_SEC: float = 0.5

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


@dataclass(frozen=True)
class FormulaReadResult:
    """掛け算式 (連鎖中テロップ) 1 サイド分の読取結果 (2026-08-24)。

    Attributes:
        valid: レイアウト検証まで通った完全な読取りか。
        left: 左辺 (消去数×10)。invalid 時 None。
        right: 右辺 (ボーナス倍率)。invalid 時 None。
        product: left × right = この段の素点。invalid 時 None。
        mult_ncc: 「×」セルのテンプレ NCC (0.0〜1.0)。
        reject_reason: invalid の理由 (デバッグ/計装用)。valid 時 None。
            "no_template"=×テンプレ不在 / "mult"=×不一致 /
            "lead_not_blank"=cell0 に数字 / "no_left"・"left_gap"=左辺不成立 /
            "left_implausible"=左辺が物理制約違反 (40未満 or 10の倍数でない) /
            "no_right"・"right_gap"=右辺不成立。
    """

    valid: bool
    left: int | None
    right: int | None
    product: int | None
    mult_ncc: float
    reject_reason: str | None = None


def _formula_invalid(reason: str, mult_ncc: float = 0.0) -> FormulaReadResult:
    """invalid な FormulaReadResult を生成する短縮ヘルパ。"""
    return FormulaReadResult(
        valid=False, left=None, right=None, product=None,
        mult_ncc=mult_ncc, reject_reason=reason,
    )


def parse_formula_cells(
    labels: "tuple[int | None, ...]",
    confs: "tuple[float, ...]",
    mult_ncc: float,
) -> FormulaReadResult:
    """8 セル分の数字分類結果と ×NCC から掛け算式をパースする (stateless 純関数)。

    レイアウト (実測、モジュール冒頭の定数コメント参照):
        [blank][左辺 1-3桁 右詰め][×][右辺 1-3桁 右詰め]
         cell0  cell1  cell2 cell3 c4  cell5 cell6 cell7

    Args:
        labels: 各セルの数字分類ラベル (None=分類失敗/空白)。長さ 8。
        confs: 各セルの NCC 信頼度。長さ 8。
        mult_ncc: cell4 の「×」テンプレ NCC。

    Returns:
        FormulaReadResult。部分読み (フェード中の欠け) はレイアウト検証で棄却する。
    """
    if mult_ncc < FORMULA_MULT_NCC_MIN:
        return _formula_invalid("mult", mult_ncc)

    def _digit(i: int) -> int | None:
        lab = labels[i]
        if lab is None or confs[i] < FORMULA_DIGIT_NCC_MIN:
            return None
        return int(lab)

    # cell0 は常に空白 (左辺は最大 3 桁)。数字があれば通常スコアの残像等。
    if _digit(FORMULA_BLANK_CELL_INDEX) is not None:
        return _formula_invalid("lead_not_blank", mult_ncc)
    # 左辺: cell1-3 右詰め。最下位 (cell3) は必須、上位桁は連続していること。
    d1, d2, d3 = (_digit(i) for i in FORMULA_LEFT_CELLS)
    if d3 is None:
        return _formula_invalid("no_left", mult_ncc)
    if d1 is not None and d2 is None:
        return _formula_invalid("left_gap", mult_ncc)
    left = d3
    if d2 is not None:
        left = d2 * 10 + d3
        if d1 is not None:
            left = d1 * 100 + left
    # 物理制約: 消去は 4 個以上 (=40) かつ左辺は 10 の倍数 (消去数×10)。
    if left < FORMULA_LEFT_MIN or left % FORMULA_LEFT_UNIT != 0:
        return _formula_invalid("left_implausible", mult_ncc)
    # 右辺: cell5-7 右詰め。最下位 (cell7) は必須、上位桁は連続していること。
    d5, d6, d7 = (_digit(i) for i in FORMULA_RIGHT_CELLS)
    if d7 is None:
        return _formula_invalid("no_right", mult_ncc)
    if d5 is not None and d6 is None:
        return _formula_invalid("right_gap", mult_ncc)
    right = d7
    if d6 is not None:
        right = d6 * 10 + d7
        if d5 is not None:
            right = d5 * 100 + right
    if right < 1:
        return _formula_invalid("right_zero", mult_ncc)
    return FormulaReadResult(
        valid=True, left=left, right=right, product=left * right,
        mult_ncc=mult_ncc, reject_reason=None,
    )


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
        enable_matmul_ncc: bool = False,
        mult_template: np.ndarray | None = None,
    ) -> None:
        """Args:
            templates: 0-9 → テンプレ画像 (50x40 BGR or grayscale) の辞書。
                未指定 (None) クラスは OCR で None 扱い。
            min_confidence: 各桁の NCC 最低スコア。これ未満なら None。
            margin_min: 1 位と 2 位のスコア差。これ未満なら曖昧と見て None。
            enable_matmul_ncc: NCC を行列積 1 回で一括計算する高速経路を使う
                (2026-07-30)。default False = 従来の matchTemplate ループ。
                詳細は `_ncc_scores_matmul` の docstring 参照。
            mult_template: 掛け算式「×」テンプレ画像 (2026-08-24 追加、
                optional)。None (default) なら read_formula_side は常に
                invalid を返す (既存挙動への影響ゼロ、backwards compat)。
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
        # 行列積 NCC (2026-07-30 高速化)。テンプレは不変なので正規化行列を
        # 初回利用時に 1 度だけ作って使い回す。
        self._enable_matmul_ncc: bool = bool(enable_matmul_ncc)
        self._tpl_matrix: tuple[tuple[int, ...], np.ndarray] | None = None
        # 掛け算式「×」テンプレ (2026-08-24)。数字セル (50x40) より小さい
        # グリフ (実測 32x31) をスライドマッチする。None なら機能無効。
        self._mult_template_gray: np.ndarray | None = (
            _to_gray(mult_template) if mult_template is not None else None
        )
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
        enable_matmul_ncc: bool = False,
    ) -> "ScoreOcr":
        """models/ui_templates/score_digits/ から digit_N.png を読み込む。

        enable_matmul_ncc: NCC を行列積1回に束ねる高速経路 (2026-07-30)。
        既定 OFF (float64 演算で cv2 の float32 と最大 5.5e-07 の差が出るため)。

        2026-08-24: 同ディレクトリの formula_mult.png (掛け算式「×」テンプレ)
        も存在すれば読み込む。不在でも従来と完全に同じ動作 (read_formula_side
        だけが常に invalid になる)。
        """
        templates = cls._load_templates_from_dir(template_dir)
        mult_path = template_dir / FORMULA_MULT_TEMPLATE_FILENAME
        mult_template = (
            cv2.imread(str(mult_path)) if mult_path.is_file() else None
        )
        return cls(
            enable_matmul_ncc=enable_matmul_ncc,
            templates=templates,
            min_confidence=min_confidence,
            margin_min=margin_min,
            avg_min_confidence=avg_min_confidence,
            mult_template=mult_template,
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

    def read_side_detail(
        self, frame: np.ndarray, side: Side
    ) -> tuple[int | None, float, tuple[int | None, ...], tuple[float, ...]]:
        """指定サイドを読み取り、セル単位の分類結果も返す (2026-08-24 追加)。

        掛け算式読取り (read_formula_side) がセル分類を再実行せずに済むよう、
        (score, min_conf, 各セルのラベル, 各セルの NCC) を返す。
        score/min_conf は read_side と同一の計算。

        Returns:
            (score, 信頼度, labels 長さ8, confs 長さ8)。frame 不正時は
            (None, 0.0, (None,)*8, (0.0,)*8)。
        """
        f = _ensure_1080p(frame)
        if f is None:
            empty_l: tuple[int | None, ...] = (None,) * DIGIT_COUNT
            empty_c: tuple[float, ...] = (0.0,) * DIGIT_COUNT
            return None, 0.0, empty_l, empty_c
        return self._read_one_side_detail(f, side)

    def read_formula_side(
        self,
        frame: np.ndarray,
        side: Side,
        digit_labels: "tuple[int | None, ...] | None" = None,
        digit_confs: "tuple[float, ...] | None" = None,
    ) -> FormulaReadResult:
        """掛け算式 (連鎖中テロップ)「左辺×右辺」を読み取る (2026-08-24)。

        掛け算式は通常スコアと同一フォント・同一 8 桁グリッドで表示される
        (実測、モジュール冒頭の定数コメント参照) ため、数字セルの分類は
        既存の _classify_digit をそのまま使う。新規の照合は「×」セル
        (cell4) の小テンプレ 1 枚のみ。

        Args:
            frame: 1920x1080 BGR フレーム (他解像度はリサイズされる)。
            side: "1P" or "2P"。
            digit_labels / digit_confs: 同一フレーム・同一サイドで既に
                read_side_detail 済みならその結果を渡す (セル分類の再実行を
                省略、追加コストは ×NCC のみになる)。None なら内部で分類する。

        Returns:
            FormulaReadResult。×テンプレ未登録時は常に invalid ("no_template")。
        """
        if self._mult_template_gray is None:
            return _formula_invalid("no_template")
        f = _ensure_1080p(frame)
        if f is None:
            return _formula_invalid("bad_frame")
        roi = _crop_score_roi(f, side)
        if roi is None or roi.size == 0:
            return _formula_invalid("bad_roi")
        mult_ncc = self._match_mult_cell(roi, side)
        if digit_labels is None or digit_confs is None:
            _score, _conf, digit_labels, digit_confs = (
                self._read_one_side_detail(f, side)
            )
        return parse_formula_cells(digit_labels, digit_confs, mult_ncc)

    def _match_mult_cell(self, roi: np.ndarray, side: Side) -> float:
        """score ROI の cell4 に「×」テンプレをスライドマッチし最大 NCC を返す。"""
        assert self._mult_template_gray is not None
        cell = _to_gray(_crop_digit_cell(roi, FORMULA_MULT_CELL_INDEX, side))
        tpl = self._mult_template_gray
        if cell.shape[0] < tpl.shape[0] or cell.shape[1] < tpl.shape[1]:
            return 0.0
        return float(cv2.matchTemplate(cell, tpl, cv2.TM_CCOEFF_NORMED).max())

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
        """従来 API (2026-08-24 に detail 版へ委譲、計算内容は不変)。"""
        score, conf, digits_t, _confs = self._read_one_side_detail(frame, side)
        return score, conf, digits_t

    def _read_one_side_detail(
        self, frame: np.ndarray, side: Side
    ) -> tuple[int | None, float, tuple[int | None, ...], tuple[float, ...]]:
        """1 サイド読取りの本体 (旧 _read_one_side + セル別 NCC も返す)。"""
        roi = _crop_score_roi(frame, side)
        if roi is None or roi.size == 0:
            empty: tuple[int | None, ...] = (None,) * DIGIT_COUNT
            return None, 0.0, empty, (0.0,) * DIGIT_COUNT
        digits: list[int | None] = []
        confidences: list[float] = []
        for i in range(DIGIT_COUNT):
            cell = _crop_digit_cell(roi, i, side)
            label, conf = self._classify_digit(cell)
            digits.append(label)
            confidences.append(conf)
        digits_t: tuple[int | None, ...] = tuple(digits)
        confs_t: tuple[float, ...] = tuple(float(c) for c in confidences)
        # 全桁が読めた時のみ score を確定
        if any(d is None for d in digits):
            min_conf = float(min(confidences)) if confidences else 0.0
            return None, min_conf, digits_t, confs_t
        # 平均 confidence チェック: 連鎖中の計算式表示で偶然 8 桁読めるケースを排除
        avg_conf = float(sum(confidences) / len(confidences))
        if avg_conf < self._avg_min_confidence:
            return None, avg_conf, digits_t, confs_t
        score = 0
        for d in digits:
            assert d is not None
            score = score * 10 + d
        min_conf = float(min(confidences))
        return score, min_conf, digits_t, confs_t

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
        scores = (
            self._ncc_scores_matmul(gray)
            if self._enable_matmul_ncc
            else self._ncc_scores_loop(gray)
        )
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

    def _ncc_scores_loop(self, gray: np.ndarray) -> dict[int, float]:
        """従来経路: テンプレごとに matchTemplate を呼んで NCC スコアを集める。

        Args:
            gray: (DIGIT_HEIGHT, DIGIT_WIDTH) のグレースケールセル。

        Returns:
            label -> NCC スコアの辞書。
        """
        return {
            label: float(
                cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED).max(),
            )
            for label, tpl in self._templates_gray.items()
        }

    def _ncc_scores_matmul(self, gray: np.ndarray) -> dict[int, float]:
        """高速経路: 行列積 1 回で全テンプレの NCC スコアを一括計算する。

        原理:
            同サイズ同士の TM_CCOEFF_NORMED は結果が 1x1 になり、値は
            **2 つの配列の Pearson 相関**に等しい。相関は「平均を引いて
            L2 正規化したベクトルの内積」なので、テンプレ側を前計算しておけば
            (テンプレ数 x 画素数) 行列と (画素数) ベクトルの積 1 回で
            全テンプレ分のスコアが同時に得られる。
            テンプレも入力も __init__ / _classify_digit で
            (DIGIT_HEIGHT, DIGIT_WIDTH) に揃えられるため、同サイズ条件は常に成立する。

        実測 (scripts/_diag_score_ocr_matmul_bench_2026-07-30.py):
            1セル分 (10テンプレ) が 1777us → 12.1us で **146倍速**。
            1フレーム換算 28.43ms → 0.19ms。認識全体の 19.5% を占めていた処理。
            torch CUDA は転送コストが支配的で 371us (numpy の 33倍遅い) だったため
            **GPU は使わない**。

        bit-identical にならない点:
            cv2 は内部を float32 で計算するのに対しこちらは float64 のため、
            スコアに最大 5.5e-07 程度の差が出る (実測)。ラベル決定は
            16セル全件で一致したが、`_min_confidence` / `_margin_min` の
            境界ぴったりでは判定が変わりうる。よって既定 OFF。

        Args:
            gray: (DIGIT_HEIGHT, DIGIT_WIDTH) のグレースケールセル。

        Returns:
            label -> NCC スコアの辞書。行列化できない構成では従来経路に fallback。
        """
        prepared = self._prepare_template_matrix()
        if prepared is None:
            return self._ncc_scores_loop(gray)
        labels, tpl_mat = prepared
        vec = gray.ravel().astype(np.float64)
        vec -= vec.mean()
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            # 均一セル: 相関は定義できない。従来経路 (cv2) の挙動に委ねる。
            return self._ncc_scores_loop(gray)
        raw = tpl_mat @ (vec / norm)
        return {int(label): float(raw[i]) for i, label in enumerate(labels)}

    def _prepare_template_matrix(
        self,
    ) -> tuple[tuple[int, ...], np.ndarray] | None:
        """テンプレを「平均引き + L2 正規化」した行列にして返す (初回のみ計算)。

        Returns:
            (label の順序, (テンプレ数 x 画素数) の正規化済み行列)。
            テンプレ不在・サイズ不揃い・分散ゼロのテンプレがある場合は None
            (呼び出し元が従来経路に fallback する)。
        """
        if self._tpl_matrix is not None:
            return self._tpl_matrix
        if not self._templates_gray:
            return None
        labels = tuple(self._templates_gray.keys())
        shapes = {self._templates_gray[label].shape for label in labels}
        if len(shapes) != 1:
            # サイズ不揃い = matchTemplate がスライドする挙動になるので等価でない
            return None
        mat = np.stack(
            [self._templates_gray[label].ravel().astype(np.float64) for label in labels],
        )
        mat -= mat.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        if not np.all(norms > 0.0):
            # 均一テンプレ (分散ゼロ) があると相関が定義できない
            return None
        mat /= norms
        self._tpl_matrix = (labels, mat)
        return self._tpl_matrix

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
        return self._apply_read(cur)

    def update_with_detail(
        self, frame: np.ndarray,
    ) -> tuple[ScoreDelta, tuple[int | None, ...], tuple[float, ...]]:
        """update() と同一の状態更新を行い、セル分類の詳細も返す (2026-08-24)。

        掛け算式読取り (ScoreOcr.read_formula_side) がセル分類を再実行せずに
        済むようにするための detail 版。score/delta の計算は update() と
        完全同一 (read_side と read_side_detail は同じ _read_one_side_detail
        に委譲される)。

        Returns:
            (ScoreDelta, 各セルのラベル 長さ8, 各セルの NCC 長さ8)。
        """
        cur, _conf, labels, confs = self._ocr.read_side_detail(
            frame, self._side,
        )
        return self._apply_read(cur), labels, confs

    def _apply_read(self, cur: int | None) -> ScoreDelta:
        """読取値 1 件で内部状態を更新し ScoreDelta を作る (update 共通部)。"""
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
# 掛け算式 段カウント累積器 (2026-08-24)
# ============================


@dataclass(frozen=True)
class FormulaStep:
    """確定した掛け算式 1 段分。

    Attributes:
        t_sec: 確定した時刻 (動画内秒)。
        left: 左辺 (消去数×10)。
        right: 右辺 (ボーナス倍率)。
    """

    t_sec: float
    left: int
    right: int

    @property
    def product(self) -> int:
        """この段の素点 (= left × right)。"""
        return self.left * self.right


class FormulaStepAccumulator:
    """掛け算式の読取り列から連鎖の段を数え、素点を積算する (1 サイド分)。

    user 伝授 (memory reference_chain_formula_per_step_2026-08-22):
        「掛け算式は消えるたびに出る。回数を数えれば連鎖数、値を足せば火力。
         推定も simulate も不要」

    確定規則 (定数の根拠はモジュール冒頭コメント参照):
        - 同一値が FORMULA_STEP_CONFIRM_FRAMES 連続で読めたら 1 段確定
          (出現/消滅アニメ中の部分読み 1.4% を除去)。
          **「連続」は文字どおり連続**: 無効読取り・幕間を 1 フレームでも
          挟んだら連続確認を破棄する (2026-08-25 P2-2 修正。破棄しないと
          非連続 2 観測が段に確定し、部分読みノイズ除去が機能しない)
        - **段の区切りは「幕間」で決める** (2026-08-24 Q-01 修正、下記)。
          幕間を挟んだ値の変化は、値が同一でも右辺が減っていても新しい段。
        - 幕間が観測できていない場合は従来規則にフォールバックする
          (`score_displayed` を渡さない呼出しでは Q-01 以前の区切り規則のまま。
           ただし上記 P2-2 の連続確認破棄は幕間の有無と無関係に常に効く)。
        - 有効読取りが FORMULA_SESSION_RESET_SEC 以上途絶えたら
          セッション破棄 (次の読取りから新しい連鎖として数える)

    ## 「幕間」とは (2026-08-24、Q-01 修正で導入)

    掛け算式と通常スコアは**排他**である
    (`src/recognition_pipeline.py:6272-6273` が通常スコアを読めたフレームでは
    掛け算式の読取りをスキップする)。したがって段と段の間には、
    通常スコアが表示される区間 = **幕間**が必ず入る。

    実測 (`logs/_diag_formula_fix_e2e_2026-08-24/trace_on.jsonl`、
    1P の 15 連鎖・13 境界をフレーム単位で計数):

        1 段の表示     = 各 28 フレーム (0.933 秒) で一定
        段間の幕間     = 13〜19 フレーム (0.433〜0.634 秒)
        幕間中の得点増 = 直前段の「左×右」と完全一致 (12/12)

    実画面でも確認済み (`screen_1p_t6697.6.jpg` = 「40× 1」、
    `screen_1p_t6698.9.jpg` = 同じ位置が「00000304」)。

    ## なぜ「右辺の単調増加」をやめたか

    旧規則は「右辺 (ボーナス倍率) は同一連鎖内で単調増加する」を前提にしていたが、
    `src/scoring.py` の公式ボーナステーブルに照らして**誤り**である。

        右辺 = max(1, 連鎖ボーナス + 連結ボーナス + 色数ボーナス)
                    ^^^^^^^^^^^^  ^^^^^^^^^^^^  ^^^^^^^^^^^^
                    単調増加       段ごとに変動   段ごとに変動

    連鎖ボーナス `[0, 8, 16, 32, 64, ...]` の 1→2 段の増分は **+8** しかないのに、
    連結 (最大 10/グループの和) + 色数 (4 色で 12) の変動幅は **最大 22 程度**ある。
    したがって右辺は同一連鎖内で**減少も同値もし得る**。

    さらに旧閾値 `FORMULA_NEW_SESSION_MIN_GAP_SEC = 0.5秒` は、実測の幕間分布
    0.433〜0.634 秒の**ど真ん中に刺さっていた**。右辺が一度でも減ると:

        幕間 13f (0.433秒 < 0.5秒) → 2 段目を棄却    (火力 1,100 のまま)
        幕間 19f (0.633秒 ≥ 0.5秒) → セッション破棄  (火力 320 に化ける)

    と、**幕間 0.1 秒の揺らぎだけで真逆に壊れ、火力が 3.4 倍ずれていた**
    (正しくは 2 段・1,420)。

    stateless 原則との関係: 観測 (read_formula_side) は stateless、
    本クラスはその外部 state-holding wrapper (ScoreTracker と同型)。
    """

    def __init__(
        self,
        confirm_frames: int = FORMULA_STEP_CONFIRM_FRAMES,
        session_reset_sec: float = FORMULA_SESSION_RESET_SEC,
        new_session_min_gap_sec: float = FORMULA_NEW_SESSION_MIN_GAP_SEC,
    ) -> None:
        self._confirm_frames = max(1, int(confirm_frames))
        self._session_reset_sec = float(session_reset_sec)
        self._new_session_min_gap_sec = float(new_session_min_gap_sec)
        self._steps: list[FormulaStep] = []
        self._pending: tuple[int, int] | None = None
        self._pending_count: int = 0
        self._last_valid_t: float | None = None
        # 直前に段が確定して以降、幕間 (通常スコア表示) を観測したか。
        # False の間は従来規則で判定するため、幕間を渡さない呼出しでは
        # 旧挙動と bit-identical になる (2026-08-24 Q-01 修正)。
        self._saw_interlude: bool = False
        # 幕間の連続フレーム数。1 フレームだけの通常スコア誤読で幕間フラグが
        # 立つと、直後のフェード部分読みを段として拾ってしまう (過大方向の事故)。
        # 実測の幕間は 13〜19 フレームなので、確認フレーム数を要求しても
        # 本物の幕間を取り逃さない (13 ≫ confirm_frames)。
        self._interlude_run: int = 0

    # ------------------------------
    # 公開プロパティ
    # ------------------------------

    @property
    def step_count(self) -> int:
        """現セッション (= 1 連鎖分) で確定した段数。"""
        return len(self._steps)

    @property
    def total_power(self) -> int:
        """現セッションの素点合計 (= Σ left×right)。"""
        return sum(s.product for s in self._steps)

    @property
    def steps(self) -> tuple[FormulaStep, ...]:
        return tuple(self._steps)

    @property
    def last_valid_t(self) -> float | None:
        """最後に有効読取りがあった時刻 (セッション健全性の確認用)。"""
        return self._last_valid_t

    def reset(self) -> None:
        """試合切替等での完全リセット。"""
        self._reset_session()
        self._last_valid_t = None

    def _reset_session(self) -> None:
        self._steps = []
        self._pending = None
        self._pending_count = 0
        self._saw_interlude = False
        self._interlude_run = 0

    # ------------------------------
    # 更新
    # ------------------------------

    def _discard_pending(self) -> None:
        """段の連続確認 (pending) を破棄する。確定済みの段は保持する。

        valid 観測が 1 フレームでも途切れたら呼ぶ (2026-08-25 P2-2 修正)。
        確定規則は「同一値が confirm_frames **連続**」であり、途切れを挟んだ
        観測を数え続けると非連続 2 観測で段が確定してしまう。
        """
        self._pending = None
        self._pending_count = 0

    def _on_interlude(self) -> None:
        """幕間 (通常スコアが読めたフレーム) 1 件を反映する。

        段の区切りとして記録するだけで、セッションは壊さない
        (幕間は同一連鎖の内部にも必ず入るため)。
        1 フレームだけの通常スコア誤読で区切りを立てないよう、
        掛け算式の段確定と同じ連続確認を要求する (実測の幕間は
        13〜19 フレームなので本物を取り逃さない)。
        """
        self._interlude_run += 1
        if self._interlude_run >= self._confirm_frames:
            self._saw_interlude = True
        # valid 観測が途切れたので連続確認は破棄する (2026-08-25 P2-2 修正)。
        self._discard_pending()

    def update(
        self,
        t_sec: float,
        result: "FormulaReadResult",
        *,
        score_displayed: bool = False,
    ) -> FormulaStep | None:
        """読取り 1 件を反映する。新規に段が確定したらその FormulaStep を返す。

        Args:
            t_sec: 現フレーム時刻 (単調増加前提)。
            result: read_formula_side の結果 (invalid も渡してよい)。
            score_displayed: このフレームで**通常スコアが読めた**か。
                True は「掛け算式が表示されていない」ことの**肯定的な観測**であり、
                段の区切り (幕間) を意味する。

                **読取り失敗では True にしてはいけない。**
                「読めなかった」と「表示されていないことが分かった」は別物で、
                前者を幕間と誤認すると**存在しない段を積む** (過大方向の事故)。
                呼出し側は `cached_score_val is not None` のときだけ True にする。

                既定 False = 幕間を渡さない従来の呼出し。このとき段の区切りは
                Q-01 以前の従来規則で判定する (backwards compat)。
                注: 2026-08-25 P2-2 修正 (無効読取りでの連続確認破棄) は
                幕間の有無と無関係に常に効くため、無効読取りを挟む系列では
                旧実装と結果が変わり得る (旧実装は非連続 2 観測を誤って
                「連続 2 フレーム」と数えていた)。

        Returns:
            新規確定した段 (なければ None)。
        """
        if score_displayed:
            self._on_interlude()
            return None
        self._interlude_run = 0
        if not result.valid or result.left is None or result.right is None:
            # 無効読取り: セッション破棄は時間経過でのみ判定する
            # (次の有効読取り時に評価)。ただし pending の連続確認は破棄する。
            # valid 観測が途切れた以上「連続 N フレーム」は成立しないため
            # (2026-08-25 P2-2 修正。破棄しないと valid→invalid→同じ valid の
            #  非連続 2 観測が段に確定し、部分読みノイズ除去が機能しない)。
            self._discard_pending()
            return None
        # セッションタイムアウト
        gap = (
            t_sec - self._last_valid_t
            if self._last_valid_t is not None else float("inf")
        )
        if gap > self._session_reset_sec:
            self._reset_session()
        self._last_valid_t = t_sec
        v = (int(result.left), int(result.right))
        if self._steps and not self._saw_interlude:
            # 幕間が観測できていない → 従来規則で判定する (Q-01 以前の区切り規則。
            # P2-2 の連続確認破棄は本分岐より前で常に効いている)。
            # 直前確定段と同一値 = 同じ段の再読 (何もしない)
            if (self._steps[-1].left, self._steps[-1].right) == v:
                self._pending = None
                self._pending_count = 0
                return None
            # 右辺の減少チェック (docstring の「なぜやめたか」も参照)
            if v[1] <= self._steps[-1].right:
                if gap >= self._new_session_min_gap_sec:
                    self._reset_session()  # 新しい連鎖の開始
                else:
                    return None  # フェード/遷移中の部分読み → 棄却
        # 同一値の連続確認
        if self._pending == v:
            self._pending_count += 1
        else:
            self._pending = v
            self._pending_count = 1
        if self._pending_count < self._confirm_frames:
            return None
        step = FormulaStep(t_sec=t_sec, left=v[0], right=v[1])
        self._steps.append(step)
        self._pending = None
        self._pending_count = 0
        # 段が確定したので幕間フラグを落とす。次の段には次の幕間が要る。
        self._saw_interlude = False
        return step


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
    "FORMULA_DIGIT_NCC_MIN",
    "FORMULA_LEFT_MIN",
    "FORMULA_LEFT_UNIT",
    "FORMULA_MULT_NCC_MIN",
    "FORMULA_MULT_TEMPLATE_FILENAME",
    "FORMULA_NEW_SESSION_MIN_GAP_SEC",
    "FORMULA_SESSION_RESET_SEC",
    "FORMULA_STEP_CONFIRM_FRAMES",
    "FormulaReadResult",
    "FormulaStep",
    "FormulaStepAccumulator",
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
    "parse_formula_cells",
]
