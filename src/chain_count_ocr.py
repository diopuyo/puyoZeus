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

## 表示仕様 (実フレームで確認済み: video_c54.mp4, 1P側, t=252.6〜257.7秒)

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
  **訂正 (2026-07-29)**: 本イベントは当初 4 連鎖で終了したと誤認していたが
  (旧npz の simulate() 結果と偶然一致していたため)、digit_5 テンプレ採取後の
  再検証で t≈257.55-257.70秒 に本物の「5 れんさ!」ポップアップ (glow演出付き)
  を実フレームで確認した。t=258.5秒以降は盤面静止・ポップアップ消滅のため、
  **真の連鎖数はこのイベントでは 5** (旧npz・新npz の simulate() は両方誤り)。
  score_consistency_ratio による整合判定は、真の連鎖数を保証するものでは
  ない好例として記録する (scripts/_verify_chain_count_screen_read_c54_2026-07-29.py
  の訂正コメントも参照)。
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

- **誤検出耐性は限定的**: ポップアップ非表示フレームでの最大誤検出スコアが
  0.578 まで observed (n=2、video_c54 1P側のみ)。閾値 0.60 で当該 2 件は
  排除できるが、この閾値は極小サンプルに基づく暫定値であり、全動画検証で
  再検証が必須。閾値を上げた副作用として、遷移中の弱い信号フレーム
  (n=3, score 0.465〜0.505) は検出できなくなるが、1 ステップの表示継続時間
  (約0.65秒) 内に高信頼度な peak フレームが必ず含まれる前提で許容している。
  → 2026-07-29 追加対応: window内の検出列を「1から始まり1ずつ増える
  連続列」として検証する `_extract_monotonic_max_chain_count()` を追加。
  孤立した誤検出 (例: 3の直後に7、1が出ていないのに4だけ) は連続列に
  乗らない限り採用されず、真の最大値を上回る誤検出のリスクを大幅に低減
  できる (下記「window内集計ロジック」参照)。ただし全動画検証は未実施。
- **digit_5〜digit_9 のテンプレ採取済み** (2026-07-29、video_c54実フレーム
  2件から採取: 1P側 game_idx=6 t≈640-638s は実際は6連鎖どまり(simulate()
  誤りでcc=9と誤表記、要注意)、2P側 game_idx=9 t≈626-638s で1→9まで実際に
  進行しテンプレ採取に成功)。**digit_0 も同時に採取済み** (2桁対応用、
  video_c11 1P側 game_idx=3 t≈599-611s で実際に「10 れんさ!」まで進行、
  simulate() は cc=12 と誤表記だったため実フレーム目視で10と確認)。
  全 10 クラス (0-9) のテンプレが揃った。ただし採取元は少数の実フレームで
  あり (各クラス概ね1サンプル)、動画・光源条件が変わると再現しない
  可能性がある。全動画検証は未実施。
- **2 桁の連鎖数 (10-19連鎖) に対応**。ぷよぷよの理論上の連鎖数上限は
  19連鎖 (MEMORY.md 記載) であり、2桁になる場合は十の位が必ず "1" になる
  という前提 (`CHAIN_TWO_DIGIT_TENS_LABEL`) で、"1" の右側 一定距離以内に
  もう1桁 (0-9) が検出された場合にのみ結合する (`_try_combine_two_digit`)。
  この前提は video_c11 の実測 1 件でのみ確認されており、行・間隔の許容量
  (`CHAIN_TWO_DIGIT_ROW_TOLERANCE_PX` 等) も単一サンプルからの暫定値。
  20連鎖以上 (理論上起こらない) は非対応。
- **2P側は「テンプレ自体は同一エンジン描画で通用する」ことは確認できたが、
  `read_max_in_window()` の集計結果はこのイベントで大幅に過小報告した
  (要改善、正直に記録)**。video_c54 2P側 game_idx=9 (t≈625.8-638.0s、実際は
  1→9まで実フレーム目視で確認済み) に `read_max_in_window` を実行したところ
  結果は 3 (真値の約1/3) だった。詳細:
    - digit_1・digit_5・digit_6・digit_7・digit_8・digit_9 は高信頼度
      (confidence 0.80〜1.00) かつ安定した位置で検出でき、フォント・
      位置ロジックとも同一エンジン描画の前提は実証された。
    - 一方 digit_2・digit_3・digit_4 はこのクリップでは confidence が
      0.60〜0.75 程度に留まり位置も不安定 (真の該当ステップが別の数字
      ラベルとして弱く誤分類される場面もあった)。原因は未特定 (真のステップが
      落下ぷよ等で一部隠れていた可能性、1P側で採取したテンプレとの光源差等、
      複数の仮説があるが未検証)。
    - 結果として「1」から「5」への連続列の橋渡しに失敗し
      (`CHAIN_STEP_MAX_GAP_SEC=2.5秒` を実際の空白期間 3.6秒以上が超過)、
      連続列は 3 で頭打ちになった。これは「真の最大値を過大評価するより
      過小評価を許容する」という設計方針どおりの挙動だが、実際の劣化幅
      (9 → 3) はこの1件の検証だけでも無視できない大きさであり、
      **全動画ロールアウト前に追加のテンプレ採取・閾値再検討が必須**。
  加えて、表示位置がROI (盤面) の左端付近になる場合、ポップアップ自体が
  ROI境界で切れて一部読み取れないケースも1件観測した (video_c54 2P側
  t≈637.0s の「9」、ROI左端で一部クリップ)。本モジュールは盤面ROI内のみを
  検索範囲とする設計のため、盤面外にはみ出す表示は原理的に対応できない
  (将来的にROIを盤面外側に少し広げる拡張が考えられるが未実装)。
- テンプレは手動crop (背景の隣接ぷよが一部写り込む) のため、score_ocr の
  digit テンプレほど背景ノイズを除去できていない。tight crop (数字グリフ
  中心、背景余白を最小化) により誤検出はある程度抑えたが、完全排除はできて
  いない (上記閾値の根拠を参照)。
- **色ベース mask + TM_CCORR_NORMED は不採用 (実験して悪化を確認済み)**。
  数字部分を HSV 色域 (H:5-35, S:80-255, V:80-255) でマスクし
  cv2.matchTemplate(..., mask=mask) で相関を取る手法を試したが、
  TM_CCORR_NORMED は平均を引かずに正規化するため全クラスのスコアが
  0.95〜1.0 に張り付いて識別力を失った (実験スクリプトで確認、本実装には
  含めていない)。現状は無地 (mask なし) TM_CCOEFF_NORMED を採用する。

## 【方式転換】得点裏取り集計 (2026-07-29 追加、userタスク指定)

上記「連続列必須」方式 (`_extract_monotonic_max_chain_count`) は、
video_c54 2P側の実9連鎖イベントで結果が **3** になる大きな過小評価を
実データで起こした (digit_2/3/4 のこのクリップでの信頼度低下・位置不安定に
より 1→9 の連続列の橋渡しに失敗、実際の空白期間がステップ間ギャップ許容
`CHAIN_STEP_MAX_GAP_SEC` を超えたため)。1桁でも読み損ねると全体の結果が
壊れる構造的脆弱性であり、テンプレが各クラス1サンプルしかない現状では
読み損ねは頻発しうる。

そこで、得点式 (src/scoring.py) が既知であることを利用し「連鎖数の候補ごとに
期待得点を概算し、実測 delta_score と最も整合する候補を採用する」得点裏取り
方式に転換する (`_aggregate_window_samples_score_backed`)。候補は window内で
検出された値の **和集合** (単独検出も含む、連続列である必要はない)。

期待得点の近似は「各ステップが 4 個消し・単色・連結ボーナスなしの最小構成
だった」と仮定した **下限値** を使う (`_approx_min_chain_score`)。実戦では
これより大きい連結・複数色同時消しが起きるため実得点は下限以上になるのが
通常だが、連鎖ボーナス (chain_power) が連鎖数に対し急峻に増加する
(0→8→16→32→64→...) ため、下限だけでも連鎖数を桁単位で強く制約できる。

実データ検証 (2026-07-29、video_c54):
- 2P game_idx=9 (delta_score=30920、実9連鎖・目視確認済み): 下限近似
  expected(9)=27880 (比率1.03・対数距離0.03) が expected(8)=20200
  (比率1.39・対数距離0.33) より明確に優位。9 を正しく選べる。
- 1P game_idx=1 (delta_score=7598): 当初 simulate() は 4 連鎖と誤認していたが
  後日の実フレーム再検証で真の連鎖数は 5 と判明済み (本ファイル上部の訂正
  コメント参照)。下限近似でも expected(5)=4840 (比率1.10・対数距離0.09) が
  expected(4)=2280 (比率1.74・対数距離0.55) より明確に優位で、真値 5 を
  正しく選べる (simulate() の誤りより得点裏取りの方が正確だった実例)。

一方、小さい連鎖数 (隣接1連鎖差) では下限近似の粗さにより選択がぶれうる
ことも実データで確認した (例: chain_count=5 だが delta_score=1140 の
イベントでは下限近似は N=3 を選好し simulate() の 5 と食い違った。
ただしこのケースは simulate() 側の認識誤りである可能性も排除できず、
どちらが正しいか本タスクの範囲では確証できない)。正直に記録する。

新方式は `read_max_in_window(..., delta_score=...)` で `delta_score` を
渡した場合のみ有効になる (省略時は既存の連続列方式のまま、backwards
compat)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.image_reader import BoardRegion, DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.scoring import (
    BASE_SCORE_PER_PUYO,
    MIN_BONUS_MULTIPLIER,
    chain_power,
    is_score_consistent,
    score_consistency_ratio,
)

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

# 検出対象の連鎖数範囲。ぷよぷよの理論上の連鎖数上限は 19 連鎖
# (MEMORY.md reference_puyo_rules_confirmed_2026-07-22 参照) のため、
# 2桁対応 (2026-07-29) により上限を 19 に拡張する。
CHAIN_COUNT_MIN: int = 1
CHAIN_COUNT_MAX: int = 19

# テンプレ (数字グリフ) クラスの範囲。0-9 の 10 クラス。
# "0" は最終結果として単体で出ることはない (CHAIN_COUNT_MIN=1) が、
# 2桁表示 (例: "10") の一の位として必要なため、テンプレクラスには含める。
CHAIN_DIGIT_LABELS: tuple[int, ...] = tuple(range(0, 10))

# 2桁表示 (10-19連鎖) の十の位は理論上つねに "1" (連鎖数上限19連鎖のため)。
# この前提を使い、"1" の右隣に別の数字が近接検出された場合のみ2桁として
# 結合する (_try_combine_two_digit 参照)。
CHAIN_TWO_DIGIT_TENS_LABEL: int = 1

# 2桁結合の位置判定パラメータ (2026-07-29 実測 1件のみ: video_c11,
# 「10 れんさ!」の "1" と "0" は隙間 2px 程度でほぼ密着していた)。
# 縦位置のずれ許容量 (px)。同一ポップアップの2桁は同じ行に描画されるはず。
CHAIN_TWO_DIGIT_ROW_TOLERANCE_PX: int = 20
# 一の位の左端 x 座標 - (十の位の左端 x 座標 + 十の位テンプレ幅) の許容範囲 (px)。
# 実測がほぼ密着 (2px) だったため、マイナス側 (わずかな重なり) も許容する。
CHAIN_TWO_DIGIT_MIN_GAP_PX: int = -20
CHAIN_TWO_DIGIT_MAX_GAP_PX: int = 30

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

# 連続列の「次のステップ」を受理する際の最大許容時間差 (秒)。
# 2026-07-29 実測 (video_c54, 1P側, 1→2→3→4 の実ステップ間隔は 1.1〜1.4秒)。
# 一方、同一 window 内で試合終了後の勝利演出画面 (盤面と無関係の背景) に
# ポップアップ非表示なのに「5」を弱く誤検出し (confidence 0.63、4の次で
# 数値上は連続列に見える) てしまう実例を検証で発見した (t_fire+1.0秒の
# window 終端が勝利演出に到達していたケース)。これは値の連続性だけでは
# 排除できないため、直前の受理から一定時間 (実測の倍以上の安全マージン) を
# 超えた検出は「別物」とみなして棄却する。
CHAIN_STEP_MAX_GAP_SEC: float = 2.5

# 入力フレームの想定サイズ (score_ocr と共通)
EXPECTED_FRAME_SHAPE: tuple[int, int] = (1080, 1920)

# 得点裏取り方式 (2026-07-29 追加) の期待得点近似で使う、1ステップあたりの
# 最小消去数の仮定 (4連結の下限。CONNECTION_BONUS_TABLE は 4 個消しから
# ボーナス対象になるため、4 未満のグループは通常の連鎖では発生しない)。
CHAIN_SCORE_APPROX_ERASED_PER_STEP: int = 4


# ============================
# 結果データクラス
# ============================


@dataclass(frozen=True)
class ChainCountReadResult:
    """ChainCountOcr.read_side() の結果。

    Attributes:
        chain_count: 読み取った連鎖数 (1-19、10以上は2桁結合の結果)。
            未検出/信頼度不足なら None。
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
        max_chain_count: window内で採用した連鎖数 (未検出/不整合なら None)。
        samples: 各サンプル時刻の生の読み取り結果 (デバッグ・検証用)。
        n_hits: chain_count が非 None だったサンプル数。
        method: 採用した集計方式 ("monotonic_run"=連続列方式 (既定) /
            "score_backed"=得点裏取り方式、2026-07-29 追加)。
            optional 追加フィールド (backwards compat、既定は旧方式名)。
        score_ratio: 得点裏取り方式のときの score_consistency_ratio
            (旧方式では None)。デバッグ・検証用。
    """

    max_chain_count: int | None
    samples: tuple[ChainCountReadResult, ...]
    n_hits: int
    method: str = "monotonic_run"
    score_ratio: float | None = None


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


# 1テンプレの ROI 内マッチ結果 (スコア, 左上座標, テンプレの(高さ,幅))
_DigitMatch = tuple[float, tuple[int, int], tuple[int, int]]


def _match_all_digits_in_roi(
    gray_roi: np.ndarray, templates_gray: dict[int, np.ndarray],
) -> dict[int, _DigitMatch]:
    """登録済み全クラスを ROI 全体でスキャンし、クラスごとの最良一致を返す。

    2桁結合判定 (_try_combine_two_digit) のため、単一の最良クラスだけでなく
    クラスごとの結果を全て保持する。
    """
    out: dict[int, _DigitMatch] = {}
    for label, tpl in templates_gray.items():
        score, loc = _match_digit_in_roi(gray_roi, tpl)
        out[label] = (score, loc, (tpl.shape[0], tpl.shape[1]))
    return out


def _try_combine_two_digit(
    matches: dict[int, _DigitMatch], min_confidence: float,
) -> ChainCountReadResult | None:
    """十の位=1固定 (連鎖数理論上限19) の2桁ポップアップを検出・結合する。

    "1" のテンプレが信頼度以上で見つかり、かつその右隣 (行が揃い、間隔が
    妥当な範囲) に別の数字 (0-9) が見つかった場合のみ 10+一の位 を返す。
    条件を満たさなければ None (呼び出し側は単一桁の通常ロジックにフォールバック)。
    """
    tens = matches.get(CHAIN_TWO_DIGIT_TENS_LABEL)
    if tens is None or tens[0] < min_confidence:
        return None
    tens_score, (tens_x, tens_y), (_tens_h, tens_w) = tens
    best_ones_label: int | None = None
    best_ones_score: float = -1.0
    for label, (score, (ones_x, ones_y), _shape) in matches.items():
        if label == CHAIN_TWO_DIGIT_TENS_LABEL or score < min_confidence:
            continue
        if abs(ones_y - tens_y) > CHAIN_TWO_DIGIT_ROW_TOLERANCE_PX:
            continue
        gap = ones_x - (tens_x + tens_w)
        if not (CHAIN_TWO_DIGIT_MIN_GAP_PX <= gap <= CHAIN_TWO_DIGIT_MAX_GAP_PX):
            continue
        if score > best_ones_score:
            best_ones_score, best_ones_label = score, label
    if best_ones_label is None:
        return None
    combined = CHAIN_TWO_DIGIT_TENS_LABEL * 10 + best_ones_label
    if combined > CHAIN_COUNT_MAX:
        return None
    confidence = min(tens_score, best_ones_score)
    return ChainCountReadResult(combined, confidence, (tens_x, tens_y))


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
            templates: 0-9 → テンプレ画像 (BGR or grayscale) の辞書。
                "0" は2桁表示の一の位判定にのみ使う (単体では出現しない)。
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
        """digit_N.png を全部スキャンする (N=0..9)。"""
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
        """ROI 全体をスキャンし、最良一致クラスを返す (2桁結合 → 1桁 の順で判定)。"""
        if not self._templates_gray:
            return ChainCountReadResult(None, 0.0, None)
        matches = _match_all_digits_in_roi(gray_roi, self._templates_gray)
        two_digit = _try_combine_two_digit(matches, self._min_confidence)
        if two_digit is not None:
            return two_digit
        return self._classify_single_digit(matches)

    def _classify_single_digit(
        self, matches: dict[int, _DigitMatch],
    ) -> ChainCountReadResult:
        """1桁分類のフォールバック ("0" は単体では出現しないため除外する)。"""
        best_label: int | None = None
        best_score: float = -1.0
        best_loc: tuple[int, int] = (0, 0)
        for label, (score, loc, _shape) in matches.items():
            if label == 0:
                continue  # "0" 単体は表示され得ない (2桁の一の位専用)
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
        delta_score: int | None = None,
    ) -> ChainCountWindowResult:
        """[t_start, t_end] を一定間隔でサンプリングし、連鎖数を集計する。

        「N れんさ!」は連鎖ステップが進むたびに増えるため、1回の連鎖window内で
        観測した値が最終連鎖数の手がかりになる (userタスク仕様)。

        Args:
            cap: cv2.VideoCapture (呼び出し側でオープン済み)。
            side: "1P" or "2P"。
            t_start: window開始時刻 (秒)。
            t_end: window終了時刻 (秒)。t_start 以上であること。
            sample_interval_sec: サンプリング間隔 (既定 0.05秒)。
            delta_score: 実測の得点差分 (省略可、既定 None)。指定した場合
                「得点裏取り」方式 (2026-07-29 追加) で集計する。連続列を
                要求せず、window内の検出値の和集合から delta_score に
                最も整合する値を選ぶ。省略時は既存の連続列方式のまま
                (backwards compat)。

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
        return _aggregate_window_samples(samples, sample_interval_sec, delta_score)


# ============================
# 純粋関数: window集計ロジック (stateless、CLAUDE.md 指標stateless原則)
# ============================


def _extract_monotonic_max_chain_count(
    hits: list[tuple[float, int]],
    max_step_gap_sec: float = CHAIN_STEP_MAX_GAP_SEC,
) -> int | None:
    """時刻順の (経過秒, 検出値) 列から「1始まり1ずつ増える連続列」の
    最大値を抽出する。

    「N れんさ!」は連鎖ステップが進むたびに 1 → 2 → 3 → … と必ず 1 ずつ
    増えながら表示される、という構造的性質を使い、孤立した誤検出 (例: 3の
    直後に7、1が出ていないのに4だけ) を棄却する (2026-07-29 userタスク指定)。

    アルゴリズム: 有効な「連続列」を複数並行して追跡する (複数の連続列候補が
    ある場合、すべて追跡し最後に最大値を採用する)。各検出値 v (時刻 t) について:
      - 既存の連続列のいずれかで v == 現在値 なら同一ステップの反復として無視
        (連続列は変化しないが最終更新時刻は t に進める)。
      - 既存の連続列のいずれかで v == 現在値+1 かつ 直前の更新時刻からの
        経過が max_step_gap_sec 以内なら、その連続列を v に進める。
      - どの連続列にも該当せず v == 1 なら、新しい連続列を開始する。
      - それ以外 (連続列の途中を飛ばした値・降下・1以外での新規開始・
        時間差が空きすぎている継続候補) は孤立検出として棄却する
        (連続列の状態は変えない)。
    途中欠け (例 1,2,(3が未検出),4) は保守的に「4を棄却」として扱う
    (真の最大値を過大評価するリスクより、過小評価の方を許容する設計判断)。

    時間差チェックの背景 (2026-07-29 実検証で発見): 数値上「+1」に見えても、
    試合終了後の勝利演出画面等 (盤面と無関係) を弱く誤検出すると、値の連続性
    だけでは誤って受理してしまう。実ステップ間隔 (実測1.1〜1.4秒) を大幅に
    超える経過時間での「継続」は別物とみなして棄却する。
    """
    active_runs: list[tuple[int, float]] = []  # (現在値, 最終更新時刻)
    for t, v in hits:
        matched = False
        for i, (run_max, last_t) in enumerate(active_runs):
            if v == run_max:
                active_runs[i] = (run_max, t)
                matched = True
            elif v == run_max + 1 and (t - last_t) <= max_step_gap_sec:
                active_runs[i] = (v, t)
                matched = True
        if not matched and v == 1:
            active_runs.append((1, t))
    return max((run_max for run_max, _ in active_runs), default=None)


def _aggregate_window_samples(
    samples: list[ChainCountReadResult],
    sample_interval_sec: float = CHAIN_WINDOW_SAMPLE_INTERVAL_SEC,
    delta_score: int | None = None,
) -> ChainCountWindowResult:
    """サンプル列から連鎖数を集計する (video I/O を含まない純粋関数)。

    delta_score が指定された場合は得点裏取り方式
    (`_aggregate_window_samples_score_backed`) を使う。省略時 (None、既定)
    は従来どおり `_extract_monotonic_max_chain_count` による連続列検証
    (値の連続性 + 経過時間チェック) を経由する (backwards compat)。
    samples は一定間隔 sample_interval_sec で採取された前提のため、
    リスト内インデックス × 間隔を経過時刻の代用とする。
    """
    if delta_score is not None:
        return _aggregate_window_samples_score_backed(samples, delta_score)
    hits_results = [s for s in samples if s.chain_count is not None]
    hits = [
        (i * sample_interval_sec, s.chain_count)
        for i, s in enumerate(samples)
        if s.chain_count is not None
    ]
    max_count = _extract_monotonic_max_chain_count(hits)
    return ChainCountWindowResult(
        max_chain_count=max_count,
        samples=tuple(samples),
        n_hits=len(hits_results),
        method="monotonic_run",
        score_ratio=None,
    )


# ============================
# 得点裏取り集計 (2026-07-29 追加、モジュール先頭「方式転換」節を参照)
# ============================


def _approx_min_chain_score(chain_count: int) -> int:
    """N連鎖の期待得点の下限近似 (各ステップ4個消し・単色・連結ボーナス0)。

    calculate_chain_score() は ChainStep の内訳 (連結サイズ・色数) を要求
    するため、連鎖数の整数値だけからは厳密な得点を計算できない。実戦では
    最小構成より大きい連結・複数色消しが起きるため実得点はこの値以上に
    なるのが通常だが、連鎖ボーナス (scoring.chain_power) は連鎖数に対して
    急峻に増加するため、候補の絞り込みには十分な判別力を持つ
    (モジュール先頭コメントの実データ検証を参照)。
    """
    if chain_count < 1:
        return 0
    total = 0
    for step in range(1, chain_count + 1):
        multiplier = max(MIN_BONUS_MULTIPLIER, chain_power(step))
        total += CHAIN_SCORE_APPROX_ERASED_PER_STEP * BASE_SCORE_PER_PUYO * multiplier
    return total


def _log_distance_from_ideal(ratio: float) -> float:
    """比率 1.0 (期待得点と実測が完全一致) からの対数距離。

    0 以下または無限大 (= 比較不能) は最も遠い候補として扱うため inf を返す。
    """
    if 0.0 < ratio < float("inf"):
        return abs(math.log(ratio))
    return float("inf")


def _select_chain_count_by_score(
    candidates: set[int], delta_score: int,
) -> tuple[int | None, float | None]:
    """delta_score に最も整合する連鎖数候補を選ぶ (得点裏取り)。

    候補ごとに `_approx_min_chain_score` の期待得点と `score_consistency_ratio`
    を計算し、比率が 1.0 に最も近い (対数距離最小) 候補を採用する。連続列で
    ある必要はない (2026-07-29 方式転換)。最有力候補でも
    `is_score_consistent` (許容比率 [0.5, 2.0]) を満たさない場合は、
    どの候補も信頼できないとみなし None を返す (誤った自信を防ぐ)。

    Returns:
        (採用した連鎖数 (信頼できる候補が無ければ None),
         採用した候補の score_consistency_ratio (候補が無ければ None))。
    """
    valid = {n for n in candidates if CHAIN_COUNT_MIN <= n <= CHAIN_COUNT_MAX}
    if not valid:
        return None, None
    scored = [
        (n, score_consistency_ratio(_approx_min_chain_score(n), delta_score))
        for n in sorted(valid)
    ]
    best_n, best_ratio = min(scored, key=lambda item: _log_distance_from_ideal(item[1]))
    if not is_score_consistent(_approx_min_chain_score(best_n), delta_score):
        return None, best_ratio
    return best_n, best_ratio


def _aggregate_window_samples_score_backed(
    samples: list[ChainCountReadResult], delta_score: int,
) -> ChainCountWindowResult:
    """得点裏取り方式での window内集計。

    連続列の完全性を要求せず、window内で検出された全ての値 (単独検出も
    含む) の和集合を候補とし、実測 delta_score と最も整合する候補を採用する。
    """
    hits_results = [s for s in samples if s.chain_count is not None]
    candidates = {s.chain_count for s in hits_results if s.chain_count is not None}
    chosen, ratio = _select_chain_count_by_score(candidates, delta_score)
    return ChainCountWindowResult(
        max_chain_count=chosen,
        samples=tuple(samples),
        n_hits=len(hits_results),
        method="score_backed",
        score_ratio=ratio,
    )


__all__ = [
    "CHAIN_COUNT_MAX",
    "CHAIN_COUNT_MIN",
    "CHAIN_DIGIT_HEIGHT",
    "CHAIN_DIGIT_LABELS",
    "CHAIN_DIGIT_WIDTH",
    "CHAIN_NCC_MIN_CONFIDENCE",
    "CHAIN_POPUP_DISPLAY_DURATION_SEC",
    "CHAIN_SCORE_APPROX_ERASED_PER_STEP",
    "CHAIN_TWO_DIGIT_MAX_GAP_PX",
    "CHAIN_TWO_DIGIT_MIN_GAP_PX",
    "CHAIN_TWO_DIGIT_ROW_TOLERANCE_PX",
    "CHAIN_TWO_DIGIT_TENS_LABEL",
    "CHAIN_WINDOW_SAMPLE_INTERVAL_SEC",
    "DEFAULT_CHAIN_TEMPLATE_DIR",
    "ChainCountOcr",
    "ChainCountReadResult",
    "ChainCountWindowResult",
]
