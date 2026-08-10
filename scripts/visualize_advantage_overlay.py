"""有利不利オーバーレイ動画 (試作) — tier1 軽量モデルで局面の有利不利を表示。

方針:
  - 学習データ (data/indicators_v2/study/labeled_win.csv) の差分特徴 (自−相手) で
    HistGBC を学習し、WinProb(1P) を出す。有利不利スコア = (p-0.5)*200。
  - 推論では重い火力系 (reach/immediate 等 484手探索) は使わず、
    collect_indicators_v2._fill_indicator_columns と同一の安価な指標関数のみ算出。
  - 認識は visualize_indicators_v2.generate と同じ load_default 経路。
  - 対象動画 (video_124_4min) は学習の study 動画 (v29-38) 外 → リークなし。

使い方:
    python -m scripts.visualize_advantage_overlay \
        --video data/frames/video_124_4min.mp4 \
        --out data/indicators_v2/overlay/advantage_v124.mp4 --max-sec 15
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.indicators_v2 as iv  # noqa: E402
from src.board import BOARD_COLS, BOARD_ROWS, Board  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.chain_detector import ChainEvent  # noqa: E402
from src.ojama_accounting import (  # noqa: E402
    OjamaAccountingTracker, OjamaAccountSnapshot,
    SCORE_RESET_THRESHOLD,  # 試合境界(score大幅減少)検知の既存定数を流用
)
from src.probability_calibration import (  # noqa: E402
    PlattCalibrationParams, apply_platt_calibration, load_platt_calibration,
)
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama  # noqa: E402
import scripts.mc_counter_estimator as mc_counter  # noqa: E402
from scripts.model_indicator_win import (  # noqa: E402
    GBC_PARAMS, load_labeled_csv, pair_sides_for_win, build_features,
)

OUT_W, OUT_H = 1280, 720
GRAPH_H = 150          # 下部に足すグラフ専用の黒帯高さ
TOP_H = 240            # 上部に足す情報パネル専用の黒帯高さ (盤面には一切描画しない)
CANVAS_H = TOP_H + OUT_H + GRAPH_H
# show_recognition=True 時のみ使用: scripts/visualize_recognition.py の ROI 定数
# (P1_ROI_X 等) はネイティブ解像度 1920x1080 で校正済みのため、認識色 overlay は
# そのネイティブ解像度で描いてから OUT_W/OUT_H に縮小する (2026-07-23 追加)。
NATIVE_W, NATIVE_H = 1920, 1080
DEFAULT_FPS = 30.0
EVEN_THRESHOLD = 5.0  # |有利不利| がこれ未満は「互角」
EMA_ALPHA = 0.25      # 有利不利の時間平滑
# (B) 持続圧力信号: board_ojama の増加を減衰累積 (着弾ダメージの記憶)
PRESSURE_DECAY = 0.985    # 毎フレーム減衰 (半減期 ~1.5s @30fps)
PRESSURE_SCALE = 6.0      # 圧力 → 有利不利[-100,100] 換算
# 能力低下ベースの圧力のスケール (2026-08-09)。 能力スコアは 0〜1 強の正規化値
# なので、 個数ベース (1 個 = 1.0) とは桁が違う。 「飽和連鎖量が 0.1 落ちる」
# = 「有利不利で 10 ポイント相当」を目安に 100 を置く (シーン逆算ではなく
# 指標の値域から決めた換算)。
CAPABILITY_PRESSURE_SCALE = 100.0
PRESSURE_BLEND_W = 0.6    # (旧2成分) 有利不利 = W×圧力 + (1-W)×現モデル
# (M3) お邪魔予告(incoming)信号: 相手に降る予告が多い=相手が埋まる=有利。
#   得点リード(結果)を廃し、予告(位置=これから相手が埋まる)へ置換(2026-07-14 user方針)。
FORECAST_SCALE = 1.4      # pending(まだ降る)お邪魔差 → 有利不利(72個≒満杯で±100)
FORECAST_DROP_PER_TURN = 30  # =OJAMA_MAX_DROP_PER_TURN(5段×6列)。ツモ1回で降る上限
# (b)ハイブリッド: 位置ブレンドに少量の得点リードを"タイブレーク"として加算。
#   ±SL_BIAS_CAP で頭打ち → 僅差局面のみ勝者側へ傾け、決着局面(位置が大)は支配しない。
SCORE_LEAD_SCALE = 1.2    # 得点差(お邪魔換算)→ バイアス値
SL_BIAS_CAP = 15.0        # 得点タイブレークの上限(±15。位置主・結果従)
# 4成分ブレンド: 圧力(着弾) + 予告(incoming) + 現モデル(位置) + threat(仕込み火力)
W_PRESSURE = 0.35
W_FORECAST = 0.30
W_MODEL = 0.20
W_THREAT = 0.15
# 打ち合い応手確率の重み (2026-08-09 user採用)。
# #24 の三つ巴比較で「併用スタッキングが全位相で有意勝ち」(rho 0.808 /
# AUC 0.837、中盤 0.856) と出ていた機構を有利不利へ接続する。
# ただし MC は 4 手先までしか読まず相手の応手を過小評価する既知バイアスが
# あるため、 モデル (0.20) と同程度に抑えて主役にはしない。
W_COUNTER = 0.20
THREAT_SCALE = 0.22       # 到達火力差(お邪魔個) → 有利不利換算
# 勝率較正: 有利不利→勝率。scripts.calibrate_winprob が実データで学習した
#   sigmoid(k×有利不利) を使う。ファイルが無ければ直線 0.5+adv/200 にフォールバック。
WINPROB_CALIB_PATH = Path("data/indicators_v2/winprob_calib.json")
# Platt scaling(全位相共通)後段校正: scripts.fit_platt_calibration が
#   model_indicator_win.py の全指標モデル(combined66データ)で学習した係数。
#   adv_to_winprob(adv) が返す「表示用勝率」に対する系統的自信過剰の補正として
#   最後段に1回だけ適用する(2026-07-29 追加、既定ON、user承認済み)。
#   注意(近似適用): 校正器は model_indicator_win.py の全指標HistGBCの出力分布に
#   対して学習されたものであり、本スクリプトの4成分ブレンドadv(pressure/
#   forecast/model/threat + kill_override)から adv_to_winprob() で得た確率とは
#   生成過程が異なる。両者とも「HistGBCの予測確率」という共通点はあるが、
#   厳密な分布一致は保証されない近似適用である(詳細は generate() docstring)。
PLATT_CALIBRATION_PATH = Path("data/indicators_v2/platt_calibration.json")


def _load_winprob_k() -> float | None:
    """較正の傾き k を読む。無ければ None(直線フォールバック)。"""
    try:
        d = json.loads(WINPROB_CALIB_PATH.read_text(encoding="utf-8"))
        return float(d["k"]) if d.get("kind") == "logistic_symmetric" else None
    except Exception:
        return None


_WINPROB_K = _load_winprob_k()


def adv_to_winprob(adv: float) -> float:
    """有利不利[-100,100] → 1P勝率[0,1]。較正があればsigmoid、無ければ直線。"""
    if _WINPROB_K is not None:
        return 1.0 / (1.0 + math.exp(-_WINPROB_K * adv))
    return max(0.0, min(1.0, 0.5 + adv / 200.0))


def _winprob_to_adv(p1: float) -> float:
    """adv_to_winprob() の厳密な逆変換 (較正sigmoidありならlogit/k、無ければ直線)。

    adv_to_winprob 側の分岐 (_WINPROB_K の有無) と完全に対にしないと、
    Platt を恒等変換 (a=1,b=0) にしても adv が往復せず変化してしまう
    (sigmoid較正 済みの p1 を直線式で逆変換すると値がズレるため)。
    """
    if _WINPROB_K is not None and _WINPROB_K != 0.0:
        p_clip = min(1.0 - 1e-9, max(1e-9, p1))
        return math.log(p_clip / (1.0 - p_clip)) / _WINPROB_K
    return (p1 - 0.5) * 200.0


def _apply_platt_to_display(
    adv: float, p1: float, platt_params: PlattCalibrationParams | None,
) -> tuple[float, float]:
    """表示用の (adv, p1) に Platt scaling 後段校正を適用する (無効時は素通し)。

    adv_to_winprob() の出力 p1 (系統的自信過剰を含む「表示用勝率」) を校正し、
    校正後 p1 から _winprob_to_adv() で adv を再構成して返す (-100〜100)。
    こうすることで EVEN_THRESHOLD判定・有利不利バー・グラフの全てに校正結果が
    反映される。kill_override 適用後の adv に対して事後的に適用する
    (kill_override は統計的予測ではなく物理判定=致死量オーバーライドのため
    校正対象にしない)。platt_params が None (校正無効/欠損) の場合は無変換。
    """
    if platt_params is None:
        return adv, p1
    calibrated_p1 = apply_platt_calibration(p1, platt_params)
    calibrated_adv = max(-100.0, min(100.0, _winprob_to_adv(calibrated_p1)))
    return calibrated_adv, calibrated_p1


# (B) キル判定(near-future): 「降るお邪魔量 > 受け容量」なら死=生存側の勝ち。
#   静止指標(threat/モデル)は「装填中の連鎖」を強みと誤読し、返しが足りず死ぬ寸前でも
#   有利側に出す。着弾を待たず pending と盤面空きから致死を検知し有利不利を生存側へ上書きする。
PLAYABLE_CELLS = 72       # 6列×12行(row0=隠し段は除く)
KILL_ROOM_FLOOR = 4       # 受け容量の下限(0除算/過敏回避。実質ほぼ窒息)
KILL_RATIO_MIN = 0.6      # 致死度差がこれ未満は上書きなし(通常の攻めは血流のまま)
KILL_RATIO_FULL = 1.5     # これ以上で完全上書き(g=1 → 生存側±100)
KILL_MIN_PENDING = 40     # pending がこれ未満は致死扱いしない(1ターン配送30個超=受け側が凌げない量)


def board_room(board) -> int:
    """受け側が窒息までに受けられるお邪魔のおおよその空き容量(セル数)。"""
    if board is None:
        return PLAYABLE_CELLS
    return max(0, PLAYABLE_CELLS - int(np.count_nonzero(board._grid[1:])))


def kill_override(adv: float, inc1: float, inc2: float,
                  room1: int, room2: int) -> float:
    """致死量を受ける側があれば有利不利を生存側へ寄せる(非致死なら不変)。

    inc1/inc2 = これから 1P/2P に降る pending お邪魔。room1/room2 = 各盤面の空き容量。
    致死度 = pending / 空き。致死度差 |l1-l2| に応じ g∈[0,1] で adv を±100側へブレンド。
    """
    # 小さな攻め(1ターン配送で捌ける量)は返し/掘りで凌げるので致死扱いしない
    l1 = inc1 / max(KILL_ROOM_FLOOR, room1) if inc1 >= KILL_MIN_PENDING else 0.0
    l2 = inc2 / max(KILL_ROOM_FLOOR, room2) if inc2 >= KILL_MIN_PENDING else 0.0
    lead = l1 - l2
    mag = abs(lead)
    if mag < KILL_RATIO_MIN:
        return adv
    g = min(1.0, (mag - KILL_RATIO_MIN) / (KILL_RATIO_FULL - KILL_RATIO_MIN))
    target = -100.0 if lead > 0 else 100.0  # 死ぬ側の逆へ
    return (1.0 - g) * adv + g * target


# (早期発火) 2026-07-29 userレビュー指摘1/2 対処: adv 全体が「両者STABLE」
#   (settled) までフリーズする設計(下記 generate() の settled ゲート参照) のため、
#   12連鎖のような返せない本線を撃っても連鎖アニメ中は勝率が動かず、連鎖完了の
#   瞬間に大きく飛ぶ(指摘1)。同様に相手の返し連鎖もアニメ中は無視されるため
#   「攻撃側 100%」のまま張り付く(指摘2)。
#   本トラッカーは chain_event 検知フレーム(掛け算式表示等、機能D。
#   src/recognition_pipeline.py:4053 _apply_chain_formula_early_fire 参照)で
#   即座に速報バイアスを加算する表示専用サイドチャネル。confirmed_board は
#   一切変更しない(STABLE限定評価の思想を維持、feedback_chain_phase_physics_only)。
EARLY_FIRE_CAP = 40.0     # 速報バイアス上限(過信防止。単独で100%表示は作らない)
# 毎フレーム減衰(半減期 約346フレーム@30fps≒11.5秒)。実測 (2026-07-29
# c56_g3 12連鎖) では settled フリーズが約19秒続いたため、その間ずっと
# 速報値を維持できるよう緩やかに設定 (settled 再計算が入れば on_settled() で
# 即クリアされるため、決着後に古い値が残り続けることはない)。
EARLY_FIRE_DECAY = 0.998


class EarlyFireTracker:
    """chain_event 検知フレームで即座に有利不利へ反映する速報バイアス(1P視点)。

    settled(両者STABLE)を待たず、起点盤面(ChainEvent.before_board)から
    iv.immediate_fire_power で即発火お邪魔量を見積もり、相手盤面への
    iv.ojama_damage(余裕浸食度)を bias として加算する。settled 再計算が
    入ったら on_settled() で bias をクリアし確定計算に道を譲る(二重計上防止)。
    """

    def __init__(self) -> None:
        self.bias = 0.0  # 1P視点 (正=1P有利)、範囲 ±EARLY_FIRE_CAP
        self._last_trigger_1p: float | None = None
        self._last_trigger_2p: float | None = None

    def _fire_damage(self, before: "Board | None", opponent: "Board | None",
                     elapsed: float) -> float:
        """1 回の発火が相手に与える速報ダメージ(0〜1)を見積もる。"""
        if before is None or opponent is None:
            return 0.0
        ojama = iv.immediate_fire_power(before, elapsed).raw
        return iv.ojama_damage(opponent, ojama).score

    def update(self, ev1: "ChainEvent | None", ev2: "ChainEvent | None",
              opp_board_for_1p: "Board | None", opp_board_for_2p: "Board | None",
              elapsed: float) -> float:
        """毎フレーム呼ぶ(settled 有無に関わらず)。現在の bias を返す。"""
        self.bias *= EARLY_FIRE_DECAY
        if ev1 is not None and ev1.trigger_sec != self._last_trigger_1p:
            self._last_trigger_1p = ev1.trigger_sec
            dmg = self._fire_damage(ev1.before_board, opp_board_for_1p, elapsed)
            self.bias += dmg * EARLY_FIRE_CAP
        if ev2 is not None and ev2.trigger_sec != self._last_trigger_2p:
            self._last_trigger_2p = ev2.trigger_sec
            dmg = self._fire_damage(ev2.before_board, opp_board_for_2p, elapsed)
            self.bias -= dmg * EARLY_FIRE_CAP
        self.bias = max(-EARLY_FIRE_CAP, min(EARLY_FIRE_CAP, self.bias))
        return self.bias

    def on_settled(self) -> None:
        """settled(確定)再計算が入ったら速報バイアスをクリアする(二重計上防止)。"""
        self.bias = 0.0


# (改修1) スコアリセット検知: 新ゲーム開始/全消し等でスコアが「前フレームから
#   大幅減少」または「両者ほぼ0」に戻ったら試合境界とみなし、凍結盤面(b1/b2)や
#   各種持続トラッカーを全て初期化する。空盤面(スコア0)なのに前試合の非空盤面
#   差分(例: 最大列高差)を表示し続ける「幻の差」バグの根治用。
#   drop 側の閾値は OjamaAccountingTracker が内部で使う既存定数を流用し重複させない。
SCORE_NEAR_ZERO_THRESHOLD = 20  # 両者スコアがこれ以下なら「0付近」とみなす(OCRノイズ許容)


def _detect_score_reset(
    score1: int | None, score2: int | None,
    prev1: int | None, prev2: int | None,
) -> bool:
    """スコア推移から試合境界(新ゲーム/全消しリセット)を検知する(純関数・state無し)。

    以下いずれかで True:
      - 前フレームからどちらかのスコアが SCORE_RESET_THRESHOLD 以上減少(新ゲーム開始等)
      - 両者のスコアが SCORE_NEAR_ZERO_THRESHOLD 以下(全消し直後/試合最初期)
    score が None(OCR失敗)の場合は判定不能として False を返す(誤リセット回避)。
    """
    if score1 is None or score2 is None:
        return False
    if prev1 is not None and prev1 - score1 >= SCORE_RESET_THRESHOLD:
        return True
    if prev2 is not None and prev2 - score2 >= SCORE_RESET_THRESHOLD:
        return True
    return score1 <= SCORE_NEAR_ZERO_THRESHOLD and score2 <= SCORE_NEAR_ZERO_THRESHOLD

# 学習・推論で共通の安価な差分特徴 (重い火力系は除外)。
# 既存の呼出元 (plot_move_diagnostics.py 等) がこの定数を直接 import して
# そのままモデル特徴量として使うため、値は変更しない (後方互換維持)。
FEATURES: tuple[str, ...] = (
    "board_color_puyo_total", "max_column_height", "column_bumpiness",
    "death_margin", "death_margin_neighbor", "current_max_chain",
    "conn_pair_count", "conn_triple_count",
    "ojama_net_balance", "ojama_forecast", "board_ojama_count", "dig_resistance",
)
# 特徴量の「候補」一覧。FEATURES に以下を末尾追加したもの:
#   - saturated_chain_count (飽和連鎖量、Round7b候補。中盤marginal寄与が未確定のため
#     まだ「候補」扱い。labeled_win.csv に列があれば _resolve_features() が
#     自動的に組み込む=現状のCSVには既に列が存在するため実際には有効化済み)
#   - ukeyasusa (受けやすさ、Round10で中盤 marginal +0.033=最大の中盤寄与と確定
#     → 2026-07 正式採用。従来は表示専用だったが本採用でモデル特徴量に格上げ)
#   - sub_chain_count (副砲、Round10で中盤 marginal +0.014・現在最大連鎖との相関
#     0.115で最も独立と確定 → 2026-07 正式採用)
#   - near_future_fire_k1..k5 (近未来最大火力、2026-07-22 user採否決定。
#     win-AUC検証で中盤 current_max_chain 比 +0.12〜+0.17・終盤 +0.04〜+0.08
#     の強いシグナルを確認、K増加で単調改善。序盤のみ current_max_chain 優位
#     のため、位相別 blend の要否は別途 viz レビューで検討する
#     (本追加はモデル特徴量候補としての組み込みのみ、overlay の見た目採否は別途))
#   - fire_stability_k2/4/6 (火力の受けの多さ=火力安定性、2026-07-22 user提案#30。
#     near_future_fire_power と同じビーム machinery の副産物。AUC検証結果は
#     別途報告 (本追加はモデル特徴量候補としての組み込みのみ、overlay の
#     見た目・位相別blend採否は別途 viz レビューで検討する))
#   - expected_fire_k1/k2 (平均ツモ期待火力、2026-07-22 user新指標。ランダム色
#     ツモを最適配置した時の火力のモンテカルロ平均=near_future(理想ツモ)の
#     対極。AUC検証結果は別途報告 (本追加はモデル特徴量候補としての組み込み
#     のみ、overlay の見た目・位相別blend採否は別途 viz レビューで検討する))
# labeled_win.csv に列が無い間は _resolve_features() が自動的に除外し、本モジュール
# 内の学習・推論は従来通り FEATURES のみで動く (列存在ガード。後方互換維持)。
NEAR_FUTURE_FIRE_COLS: tuple[str, ...] = tuple(
    f"near_future_fire_k{k}" for k in range(1, 6)
)
FIRE_STABILITY_COLS: tuple[str, ...] = tuple(
    f"fire_stability_k{k}" for k in (2, 4, 6)
)
EXPECTED_FIRE_COLS: tuple[str, ...] = tuple(
    f"expected_fire_k{k}" for k in (1, 2)
)
FEATURE_CANDIDATES: tuple[str, ...] = FEATURES + (
    "saturated_chain_count", "ukeyasusa", "sub_chain_count",
) + NEAR_FUTURE_FIRE_COLS + FIRE_STABILITY_COLS + EXPECTED_FIRE_COLS
# 主要ドライバ表示用の日本語ラベル
JP_LABEL: dict[str, str] = {
    "board_ojama_count": "盤面お邪魔数", "death_margin": "窒息余裕",
    "max_column_height": "最大列高", "current_max_chain": "現在最大連鎖",
    "board_color_puyo_total": "色ぷよ総数", "ojama_forecast": "お邪魔予告",
    "color_puyo_x_ojama_flat": "色ぷよ差×おじゃまフラット度",
    "ojama_flat_score": "おじゃまフラット度",
    "match_progress": "進行度", "color_puyo_x_earliness": "色ぷよ差×序盤度",
    "saturated_chain_count": "飽和連鎖量",
    "ukeyasusa": "受けやすさ", "sub_chain_count": "副砲連鎖数",
    "near_future_fire_k1": "近未来火力K1", "near_future_fire_k2": "近未来火力K2",
    "near_future_fire_k3": "近未来火力K3", "near_future_fire_k4": "近未来火力K4",
    "near_future_fire_k5": "近未来火力K5",
    "fire_stability_k2": "火力安定K2", "fire_stability_k4": "火力安定K4",
    "fire_stability_k6": "火力安定K6",
    "expected_fire_k1": "期待火力K1", "expected_fire_k2": "期待火力K2",
}
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\meiryo.ttc", "/mnt/c/Windows/Fonts/meiryo.ttc",
)
# 飽和連鎖量サブ行の縦オフセット(受けやすさ行から更に下へ何px か)
SATURATED_ROW_Y_OFFSET_PX = 22
# 上部情報パネル(TOP_H 内)のレイアウト定数。全て y 座標 (マジックナンバー禁止 → 定数化)。
# 盤面(ゲーム画面)には一切描画せず、この範囲 y∈[0, TOP_H) にのみ情報を集約する。
PANEL_TITLE_Y = 8                                         # タイトル行
PANEL_BAR_TOP = 44                                        # 有利不利バー上端
PANEL_BAR_H = 34                                          # バー高さ (従来 bar_h を踏襲)
PANEL_BAR_W = 720                                         # バー幅 (従来 bar_w を踏襲)
PANEL_WINPROB_Y = PANEL_BAR_TOP + PANEL_BAR_H + 14        # 勝率行
PANEL_DRIVERS_Y = PANEL_WINPROB_Y + 26                    # 主因行
PANEL_UKEY_Y = PANEL_DRIVERS_Y + 26                       # 受けやすさ行
PANEL_SAT_Y = PANEL_UKEY_Y + SATURATED_ROW_Y_OFFSET_PX    # 飽和連鎖行 (既存offset流用)


# ============================
# パネルレイアウト (2026-08-10 user指示、同日字幕余白追記) — 1920x1080 canvas
# ============================
# 左上に映像、左下にタイムライングラフ、右に縦長の情報パネル、下端に全幅の
# 字幕帯 (何も描画しない無地) を配置する新レイアウト。既存の overlay レイアウト
# (盤面に直接バー等を重ねる従来レイアウト、上記 PANEL_BAR_* 等) とは完全に
# 別経路であり、 --layout panel 指定時のみ使う (既定 layout=overlay は本ブロック
# を一切参照せず、既存出力は不変)。
# 字幕帯 (2026-08-10 追記): userが編集ソフトで字幕を載せるための余白。
# 下端 140px は video/graph/info のいずれの描画対象にも含めず、
# _draw_panel_layout が背景色で塗るだけで文字・図形を一切描かない
# (userの絶対要求「下端の帯には情報を一切描かない」)。
VALID_LAYOUTS: tuple[str, ...] = ("overlay", "panel")
PANEL_CANVAS_W = 1920                                  # 出力キャンバス全体
PANEL_CANVAS_H = 1080
PANEL_SUBTITLE_H = 140                                 # 下端字幕帯の高さ (無描画)
PANEL_CONTENT_H = PANEL_CANVAS_H - PANEL_SUBTITLE_H    # 映像+グラフ+情報パネルの高さ (940)
PANEL_VIDEO_W = 1408                                   # 左上の映像 (16:9 維持)
PANEL_VIDEO_H = 792
PANEL_INFO_W = PANEL_CANVAS_W - PANEL_VIDEO_W          # 右の情報パネル幅 (512)
PANEL_GRAPH_H = PANEL_CONTENT_H - PANEL_VIDEO_H        # 左下グラフ高さ (148)
PANEL_SUBTITLE_BG_COLOR = (10, 10, 12)                 # 字幕帯の背景色 (黒〜濃グレー)
# 情報パネル内の余白・行送り (マジックナンバー禁止 → 定数化)
PANEL_INFO_PAD = 24            # 左右余白
# _draw_bar は「1P」「2P」ラベルをバー本体の外側 (左へ34px/右へ6px+文字幅) に
# 描く (overlay レイアウトでは横幅720pxの余裕があり問題にならなかった)。
# 512px幅の情報パネルでは PANEL_INFO_PAD だけではラベルが左は映像領域に、
# 右はキャンバス外にクリップする (2026-08-10 自己検収 PNG で実測発見)。
# バー本体をさらに内側へ寄せてラベル分の余白を確保する。
PANEL_INFO_BAR_LABEL_MARGIN = 40
PANEL_INFO_BAR_TOP_OFFSET = 40  # パネル上端からバーまでの距離
PANEL_INFO_BAR_H = 54           # バー高さ (overlay版 34 より太くしてスマホ視認性を上げる)
PANEL_INFO_WINPROB_Y1 = 130     # 1P勝率%行
PANEL_INFO_WINPROB_Y2 = 190     # 2P勝率%行
PANEL_INFO_DRIVERS_Y = 260      # 主因見出し行
PANEL_INFO_DRIVER_LINE_H = 26   # 主因1件あたりの行送り
PANEL_INFO_STATE1_Y = 420       # 1P状態行
PANEL_INFO_STATE2_Y = 450       # 2P状態行
PANEL_INFO_COUNTER_Y = 490      # 応手情報行 (counter-reach有効時のみ)
PANEL_INFO_ELAPSED_BOTTOM_MARGIN = 50  # 経過時刻行 (情報パネル下端からの距離)


def panel_layout_regions() -> dict[str, tuple[int, int, int, int]]:
    """パネルレイアウトの4領域 (video/graph/info/subtitle) を (x0, y0, w, h) で返す。

    stateless な純関数 (座標計算のみ、副作用なし)。generate() と
    _draw_panel_layout() の両方が本関数を参照することで、座標がずれる
    バグ (二重管理) を構造的に防ぐ。4領域は 1920x1080 を隙間・重複なく分割する
    (video+graph の左列と info の右列が上部 940px を占め、下端 140px は
    subtitle が全幅で占める)。
    """
    return {
        "video": (0, 0, PANEL_VIDEO_W, PANEL_VIDEO_H),
        "graph": (0, PANEL_VIDEO_H, PANEL_VIDEO_W, PANEL_GRAPH_H),
        "info": (PANEL_VIDEO_W, 0, PANEL_INFO_W, PANEL_CONTENT_H),
        "subtitle": (0, PANEL_CONTENT_H, PANEL_CANVAS_W, PANEL_SUBTITLE_H),
    }


def _font(size: int) -> ImageFont.ImageFont:
    """meiryo を取得 (無ければ default)。"""
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _resolve_features(df: pd.DataFrame) -> list[str]:
    """FEATURE_CANDIDATES のうち labeled_win.csv に実在する列だけを返す(列存在ガード)。

    saturated_chain_count は再収集中のため現状 CSV に列が無い → 除外され
    従来と完全に同一の FEATURES(12指標)のみで学習される。再収集後に列が
    入れば自動的にモデル特徴量へ組み込まれる(このモジュール内の学習・推論
    経路のみ。FEATURES 定数そのものは変更しないため他スクリプトは無影響)。
    """
    missing = [c for c in FEATURE_CANDIDATES if c not in df.columns]
    if missing:
        print(f"[train] labeled_win.csv に未収集の列をスキップ(将来自動有効化): {missing}")
    return [c for c in FEATURE_CANDIDATES if c in df.columns]


# ============================
# 学習データ (2026-08-09 切り替え)
# ============================
# 従来は data/indicators_v2/study/labeled_win.csv (2026-07-22 時点・**10 動画**)
# を使っていた。 その後 2026-07-29 に 66 動画版が作られていたが参照先が
# 更新されておらず、 デモは 2 週間以上前の 10 動画で学習したモデルで
# 動いていた (2026-08-09 発覚)。
#   - 行数    40,112 -> 193,623 (4.8 倍)
#   - 動画数  10 -> 66
#   - 列      88 -> 96 (上位互換。 現行の全列を含む)
# ペア数は 6,049 -> 73,416。
TRAIN_CSV_PATH: str = (
    "data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv"
)

# ============================
# 交互作用特徴: 色ぷよ差 × おじゃまフラット度 (2026-08-09 user伝授)
# ============================
# user 伝授:
#   「色ぷよはお邪魔状況や予告お邪魔などがフラットな時に特に有利不利に
#     優位にうごきます」
# つまり色ぷよ差は **単独で常に効くのではなく、おじゃま状況が両者フラットな
# 局面で特に強く効く**。 色ぷよ差を単独特徴としてだけ持たせると、 おじゃまが
# 絡む局面のデータに引きずられて重みが薄まり、 フラット局面で効かなくなる。
#
# 実際 t=29 (1P の色ぷよが明らかに多い場面) では主因が
# 「副砲連鎖数差 +0.42 / 最大列高差 +0.17」= 1P 優位を示しているのに
# 総合は 2P有利80% と逆を向いていた。
#
# CLAUDE.md の原則「観測軸を提供 → 学習で重要度を発見」に従い、
# **フラット度を人が閾値で切らず連続量として与え、重要度は学習に決めさせる**。
COLOR_OJAMA_INTERACTION_COL: str = "color_puyo_x_ojama_flat"
# フラット度の減衰スケール。
# **指標は 0〜1 に正規化済み** (CLAUDE.md 規約) なので、 差の値域は概ね
# [-1, 1]。 当初これを「おじゃま個数」と誤解して 12.0 を置き、 全サンプルが
# フラット判定になる不具合を出した (2026-08-09)。
# ダメージ関数の第1折れ点 12 個 (memory `reference_ojama_damage_function`) を
# 正規化スケールへ写すと、 board_ojama_count の実測平均が 0.059 であることから
# 「1 段ぶん (6個) 前後の差」は正規化値でおよそ 0.1 に対応する。
# よって 0.1 を採用する (デモのシーンからの逆算ではなく、 物理量と実測分布から
# 導いた値)。
OJAMA_FLAT_SCALE: float = 0.1


def _ojama_flat_score(
    ojama_diff: "pd.Series | float", forecast_diff: "pd.Series | float",
) -> "pd.Series | float":
    """おじゃま状況のフラット度 (0〜1) を返す。

    盤面おじゃま差と予告おじゃま差の両方を見る (user 表現の
    「お邪魔状況や予告お邪魔など」)。 どちらの差も小さいほど 1 に近づく。

    Args:
        ojama_diff: 盤面おじゃま数の差 (自 - 相手)。
        forecast_diff: 予告おじゃまの差 (自 - 相手)。

    Returns:
        フラット度。 差が 0 なら 1.0、 差が大きいほど 0 に近づく。
    """
    total = np.abs(ojama_diff) + np.abs(forecast_diff)
    return np.exp(-total / OJAMA_FLAT_SCALE)


# おじゃまフラット度そのものを特徴量に加えるための列名 (2026-08-09)。
# HistGradientBoosting は木なので、 **フラット度を 1 列渡すだけで
# 「フラットなら色ぷよを見る / そうでなければ効率を見る」という分岐を
# 自力で学習できる**。 掛け算の交互作用を人が作り込む必要がない。
# 実測 (73,416ペア) で、 おじゃまフラット局面では効率系指標が全て AUC
# 0.49〜0.50 (無相関) になり、 色ぷよ総数だけが 0.5380 で効く。 逆に
# おじゃま差が大きい局面では効率系が 0.65〜0.75 で効く。 **局面によって
# 効く軸が入れ替わる**ため、 モデルに局面を教える列が要る。
OJAMA_FLAT_COL: str = "ojama_flat_score"


# ============================
# 進行度の文脈列 (2026-08-10 Phase1-1 B-2、user承認済み・アーキ設計)
# ============================
# 確定事実 (data/verify/j1_color_lead_clean_noinflight_2026-08-10.txt): 発火
# ±12s 除外のクリーン盤面で色ぷよ+8〜15リード側の実勝率は 序盤79.7% /
# 中盤65.2% (中立48.1%)。 色ぷよ差は序盤ほど強く効くが、 本モデルには
# 試合進行度を示す特徴列が一切無かった (アーキ設計が特定した第2の穴。
# B-1 対称化バグ修正とは独立)。
#
# 進行度は「位相は試合内相対進行率で切るべき」(memory
# `project_win_eval_regen_2026-07-26` 確定知見) に従い、 両者の盤面ぷよ総数
# (お邪魔含む。 src.indicators_v2.board_puyo_total、 既に ON_FIELD_CAP=72 で
# 正規化済み) の平均を使う。 時刻でなく盤面状態量そのものから出るため
# RT でも使える (二層設計思想、 memory
# `project_dual_mode_indicator_design_2026-07-22`)。
# _assign_phase_by_puyo_tertile (scripts/compute_exchange_delta_winprob.py)
# が使う「1P+2P盤面ぷよ合計の3分位」と同一コンセプト (2で割って[0,1]に
# 正規化した点のみ異なる)。
#
# board_puyo_total は「自−相手」の差分ではなく両者の合計から作る絶対量
# なので、 通常の FEATURES/FEATURE_CANDIDATES 経由で `_diff` 化すると
# 意味が壊れる。 また board_puyo_total_diff (=board_color_puyo_total_diff+
# board_ojama_count_diff、 完全共線) をそのままモデル特徴量に混ぜると
# 「材料」と「危険度」が相殺し合う既知の問題がある (修正J-2の教訓、
# scripts/compute_exchange_delta_winprob.py 冒頭コメント参照)。 そのため
# board_puyo_total 自体は FEATURE_CANDIDATES に追加せず、 本ブロック限定で
# 「両者の合計 (=進行度)」としてのみ使う。
MATCH_PROGRESS_COL: str = "match_progress"
# 「序盤ほど色ぷよ差が効く」を列にした交互作用 (色ぷよ差 × 早さ)。
# match_progress=0 (序盤) で係数最大、 1 (終盤) で 0 へ減衰する。
COLOR_EARLINESS_INTERACTION_COL: str = "color_puyo_x_earliness"


def _match_progress_from_totals(total_1p: "pd.Series | np.ndarray",
                                 total_2p: "pd.Series | np.ndarray") -> np.ndarray:
    """両者の正規化済み盤面ぷよ総数から進行度 (0〜1) を返す (stateless)。

    total_1p/total_2p は既に ON_FIELD_CAP で正規化済みの値
    (src.indicators_v2.board_puyo_total の score) を想定。 平均を取るだけで
    追加の学習データ由来定数を持たない (物理量の正規化のみ)。
    浮動小数の丸めで僅かに [0,1] を超えうるため clip で安全側に倒す。
    """
    total = (np.asarray(total_1p, dtype=float) + np.asarray(total_2p, dtype=float)) / 2.0
    return np.clip(total, 0.0, 1.0)


def _add_interaction_columns(
    feat: "pd.DataFrame", feat_cols: list[str],
    paired: "pd.DataFrame | None" = None,
) -> tuple["pd.DataFrame", list[str]]:
    """差分特徴に交互作用列を追加する (列が揃っていない場合は何もしない)。

    Args:
        paired: 1P/2P ペア済みの生データ (pair_sides_for_win の戻り値)。
            match_progress は「自−相手」の差分でなく両者の合計から作る
            絶対量のため、 build_features 後の feat には含まれない
            board_puyo_total_{1p,2p} をここから直接参照する。 省略時
            (既存呼出元との後方互換) は進行度列の追加をスキップする
            (列存在ガード、 呼出側は無変更で動作継続)。

    Returns:
        (列を追加した DataFrame, 交互作用列名を含む列名リスト)
    """
    feat = feat.copy()
    cols = [f"{c}_diff" for c in feat_cols]

    # ① 色ぷよ差 × おじゃまフラット度 (2026-08-09)
    ojama_need = ("board_color_puyo_total_diff", "board_ojama_count_diff",
                  "ojama_forecast_diff")
    if all(c in feat.columns for c in ojama_need):
        flat = _ojama_flat_score(
            feat["board_ojama_count_diff"], feat["ojama_forecast_diff"],
        )
        feat[f"{COLOR_OJAMA_INTERACTION_COL}_diff"] = (
            feat["board_color_puyo_total_diff"] * flat
        )
        # フラット度そのものも渡す (木が局面別の分岐を学習できるようにする)
        feat[f"{OJAMA_FLAT_COL}_diff"] = flat
        cols.append(f"{COLOR_OJAMA_INTERACTION_COL}_diff")
        cols.append(f"{OJAMA_FLAT_COL}_diff")

    # ② 進行度 + 色ぷよ差×早さ (2026-08-10 Phase1-1 B-2)
    progress_need = ("board_puyo_total_1p", "board_puyo_total_2p")
    has_progress_input = (
        paired is not None and all(c in paired.columns for c in progress_need)
    )
    if has_progress_input and "board_color_puyo_total_diff" in feat.columns:
        progress = _match_progress_from_totals(
            paired["board_puyo_total_1p"], paired["board_puyo_total_2p"],
        )
        feat[f"{MATCH_PROGRESS_COL}_diff"] = progress
        feat[f"{COLOR_EARLINESS_INTERACTION_COL}_diff"] = (
            feat["board_color_puyo_total_diff"] * (1.0 - progress)
        )
        cols.append(f"{MATCH_PROGRESS_COL}_diff")
        cols.append(f"{COLOR_EARLINESS_INTERACTION_COL}_diff")

    return feat, cols


# ============================
# 対称化 (side入れ替えミラー標本) の符号反転リスト (2026-08-10 バグ修正)
# ============================
# _train_model は side 入れ替え対称性 (「1P/2P を入れ替えたら予測も反転する
# べき」) を学習データに強制するため、 全特徴量を符号反転したミラー標本を
# 追加している。 しかしこれは **「自−相手」型の真の差分列にのみ正しい**。
#
# 一部の列は「side を入れ替えても値が変わらない」side非依存の絶対量であり、
# これを無条件反転すると本来あり得ない値 (例: フラット度が負) がミラー標本に
# 混入する。 具体的には:
#   - `ojama_flat_score_diff` (= _ojama_flat_score() の出力。 定義が
#     np.abs(おじゃま差) + np.abs(予告差) の減衰関数) は 1P/2P を入れ替えても
#     abs() の中身は符号だけ変わり絶対値は不変 → 出力は不変。 列名は
#     パイプライン都合で "_diff" サフィックスが付くが実体は「自−相手」の
#     差分ではないので反転してはいけない。
#   - 一方 `color_puyo_x_ojama_flat_diff` (交互作用列 = 色ぷよ差×フラット度)
#     は「符号可変量 × 符号不変量」なので side 入れ替えで符号が変わる。
#     これは反転が正しい (登録不要)。
#
# 新しい列を追加する際は、 1P/2P を入れ替えたときに符号が変わるかどうかを
# 必ず確認し、 変わらない (side非依存の絶対量・abs/exp/count等) 列だけを
# ここに追記すること。 登録漏れは「あり得ない値の学習データ混入」という
# サイレントバグを生む (2026-08-10 発見、 アーキ設計 案B-1)。
SIDE_INVARIANT_COLS: tuple[str, ...] = (
    # おじゃまフラット度 (np.abs ベースの絶対量。 1P/2P 非依存)
    f"{OJAMA_FLAT_COL}_diff",
    # 進行度 (両者の盤面ぷよ総数の平均。 1P/2P を入れ替えても平均は不変。
    # 2026-08-10 Phase1-1 B-2 追加)。 一方 color_puyo_x_earliness_diff は
    # 「符号可変(色ぷよ差) × 符号不変(1-進行度)」なので符号可変 → 未登録
    # (登録しないのが正しい、 反転して良い)。
    f"{MATCH_PROGRESS_COL}_diff",
)


def _mirror_sign(cols: list[str]) -> np.ndarray:
    """対称化ミラー標本用の列別符号ベクトルを返す (+1=そのまま複製, -1=反転)。

    SIDE_INVARIANT_COLS に登録された列は +1 (不変)、 それ以外は -1 (可変・
    「自−相手」差分は side 入れ替えで符号反転が正しい)。 テストで直接検証
    できるよう _train_model から切り出した (stateless、副作用なし)。
    """
    return np.array(
        [1.0 if c in SIDE_INVARIANT_COLS else -1.0 for c in cols], dtype=float,
    )


def _train_model(exclude_video: str | None = None):
    """study データの差分特徴で HistGBC を学習して返す。

    exclude_video: 指定動画IDの行を学習から除外 (対象動画のリーク防止)。
    実際に学習で使った特徴量列は model._puyo_feature_cols に格納する
    (列存在ガード対応。戻り値の型・呼出シグネチャは変更せず既存呼出元との
    互換を維持。_score_advantage 側がこの属性を参照して自動整合する)。
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    df = load_labeled_csv(TRAIN_CSV_PATH)
    if exclude_video is not None:
        before = len(df)
        df = df[df["video_id"].astype(str) != exclude_video].reset_index(drop=True)
        print(f"[train] {exclude_video} を学習除外: {before} -> {len(df)} 行")
    feat_cols = _resolve_features(df)
    paired = pair_sides_for_win(df, max_tdiff=1.0)
    feat = build_features(paired, feat_cols)
    # 交互作用 (色ぷよ差 × おじゃまフラット度、色ぷよ差 × 進行度の早さ) を
    # 追加。フラグ既定 ON だが必要列が揃わない場合は自動的に無効化される
    # (列存在ガードと同じ思想)。paired を渡すのは進行度列 (match_progress)
    # が board_puyo_total_{1p,2p} という build_features 後の feat には
    # 含まれない生列を必要とするため (_add_interaction_columns docstring 参照)。
    feat, cols = _add_interaction_columns(feat, feat_cols, paired)
    X = feat[cols].fillna(0.0).values
    y = paired["won_1p"].astype(int).values
    # 対称化: 差分を反転しラベルも反転したミラー標本を追加。
    # 有利不利は「側を入れ替えると符号反転」する反対称関数であるべきで、
    # これにより互角(差=0)の予測が厳密に 50% になり、勝ち数の偏りバイアスを除去。
    # ただし列ごとに符号可変/不変が異なる (SIDE_INVARIANT_COLS 参照) ため、
    # 全列一律の `-X` は誤り (2026-08-10 修正: 列別符号ベクトルを使う)。
    X_sym = np.vstack([X, X * _mirror_sign(cols)])
    y_sym = np.concatenate([y, 1 - y])
    model = HistGradientBoostingClassifier(**GBC_PARAMS)
    model.fit(X_sym, y_sym)
    model._puyo_feature_cols = feat_cols  # 列存在ガード後の実特徴量 (推論側で参照)
    # 交互作用列を使ったかを推論側へ伝える (使っていなければ推論も足さない)
    model._puyo_uses_interaction = (
        f"{COLOR_OJAMA_INTERACTION_COL}_diff" in cols
    )
    # 進行度文脈列 (match_progress / color_puyo_x_earliness) を使ったかを
    # 推論側へ伝える (2026-08-10 Phase1-1 B-2、既存の interaction フラグと同じ方式)
    model._puyo_uses_progress = (
        f"{MATCH_PROGRESS_COL}_diff" in cols
    )
    print(f"[train] 元n={len(y)} (1P勝ち{int(y.sum())}) -> 対称化後 {len(y_sym)}")
    return model


class PressureTracker:
    """(B) 持続圧力: 相手盤面お邪魔の増加(=攻撃着弾)を減衰累積した 1P視点の圧力。"""

    def __init__(self) -> None:
        self.pressure = 0.0
        self._prev1 = 0.0
        self._prev2 = 0.0

    def update(self, ojama_1p: float, ojama_2p: float) -> float:
        """毎フレーム呼ぶ。1P視点の圧力を [-100,100] で返す (正=1P攻勢)。"""
        self.pressure *= PRESSURE_DECAY
        self.pressure += max(0.0, ojama_2p - self._prev2) - max(0.0, ojama_1p - self._prev1)
        self._prev1, self._prev2 = ojama_1p, ojama_2p
        return float(max(-100.0, min(100.0, self.pressure * PRESSURE_SCALE)))


# ============================
# 打ち合い応手 (モンテカルロ) — 2026-08-09 user採用
# ============================
# user 確定定義 (memory reference_saisoku_exchange_model_2026-07-22):
#   「有効な催促 = 着弾までに相手が返せる見込みが 50% 以下」
#
# **相手が打てる手数は固定値ではない** (2026-08-09 user指摘)。 確定知見
# (memory reference_ojama_landing_gated_by_placement_2026-07-29) の通り、
# おじゃまは「連鎖完了後・受け側のツモが着地したとき」に降るため、
#     相手が打てる手数 ≒ floor(連鎖アニメ時間 ÷ 1手時間) + 1
# であり、 局面ごとに変わる。
#
# 使う実装: `scripts/mc_counter_estimator.estimate_counter_distribution`。
# これは **時間予算 (秒) を渡すと、 その中で打てるだけ手を進める**ロールアウト
# で、 K=4 の頭打ちが無い (#24 K拡張、 user承認方針「K は近似値として出すのが
# 正しい」)。 既知ネクスト (Next 表示) も渡せる。
# 当初 `counter_reach_probability_fast` を K 固定で呼ぶ実装を書いたが、
# **既に上限を外した実装があった**ため差し替えた。

# 応手判定に使う「返し」の閾値 (お邪魔換算個数)。
# ダメージ関数の第1折れ点 12 個 (2段ぶん、
# memory reference_ojama_damage_function) を採用する物理由来の値。
COUNTER_THRESHOLD_OJAMA: float = 12.0
# 応手優位 → 有利不利[-100,100] への換算。 確率差 (最大 1.0) をフルスケールに
# はせず控えめに置く (MC の近似誤差を主役にしないため)。
COUNTER_SCALE: float = 40.0
# ロールアウト本数 (既定 200 は重いので表示用に減らす)。
COUNTER_N_ROLLOUTS: int = 60


class CounterReachTracker:
    """相手が返せるかを時間予算ベースのモンテカルロで見る打ち合い優位。

    盤面 + 時間予算が同じなら結果も同じ (実装が決定論的) なのでキャッシュする。
    """

    def __init__(self) -> None:
        self._cache: dict[bytes, float] = {}
        # 直近に使った時間予算と平均打手数 (デバッグ・表示用)
        self.last_budget_sec: float = 0.0
        self.last_hands: float = 0.0

    def _reach(
        self, board: Board, budget_sec: float,
        known_pairs: "tuple[tuple[int, int], ...]",
    ) -> tuple[float, float]:
        """(閾値以上を返せる確率, 平均打手数) を返す。"""
        dist = mc_counter.estimate_counter_distribution(
            board, budget_sec,
            known_pairs=known_pairs,
            thresholds_ojama=(COUNTER_THRESHOLD_OJAMA,),
            n_rollouts=COUNTER_N_ROLLOUTS,
        )
        return (
            float(dist.prob_at_least.get(COUNTER_THRESHOLD_OJAMA, 0.0)),
            float(dist.mean_hands_used),
        )

    def update(
        self, b1: Board, b2: Board, budget_sec: float = 0.0,
        next1: "tuple[int, int] | None" = None,
        next2: "tuple[int, int] | None" = None,
    ) -> tuple[float, float, float]:
        """(1P視点の優位[-100,100], 1Pの応手確率, 2Pの応手確率) を返す。

        budget_sec: 着弾までの時間予算 [秒]。 **手数はこの予算から決まる**
            (固定値を使わない)。 0 以下なら判定不能として 0 を返す。
        next1/next2: 各 side の既知ネクスト (あれば精度が上がる)。
        """
        if budget_sec <= 0.0:
            return 0.0, float("nan"), float("nan")
        self.last_budget_sec = budget_sec
        out: list[float] = []
        hands: list[float] = []
        for b, nx in ((b1, next1), (b2, next2)):
            known = (nx,) if nx and nx[0] > 0 and nx[1] > 0 else ()
            base = b.grid_bytes() if hasattr(b, "grid_bytes") else b._grid.tobytes()
            key = base + f"|{budget_sec:.2f}|{known}".encode()
            if key not in self._cache:
                if len(self._cache) > 256:
                    self._cache.clear()
                self._cache[key] = self._reach(b, budget_sec, known)
            p, h = self._cache[key]
            out.append(p)
            hands.append(h)
        p1, p2 = out
        self.last_hands = float(np.mean(hands)) if hands else 0.0
        # 相手が返せないほど 1P 有利
        adv = ((1.0 - p2) - (1.0 - p1)) * COUNTER_SCALE
        return float(max(-100.0, min(100.0, adv))), p1, p2


class CapabilityPressureTracker:
    """(2026-08-09 user伝授) 攻撃の効果を **相手の盤面能力の低下量** で測る圧力。

    従来の `PressureTracker` は「相手盤面のおじゃま個数が何個増えたか」だけを
    減衰累積していた。 しかし user 伝授の通り、 評価すべきは個数ではなく
    **おじゃまによって盤面の機能がどれだけ落ちたか**である:

      - 飽和連鎖量が減った (組める最大の連鎖が小さくなった)
      - 連鎖がつながりづらくなった (連結が分断された)

    同じ 10 個でも土台を割られた場合と上に乗っただけでは損害がまったく違う。
    個数だけで測ると同じ扱いになってしまう
    (CLAUDE.md「形は手段、機能が本質」の直接の適用)。

    本トラッカーは各 side の能力スコア
        capability = 飽和連鎖量 + 連結(pair/triple の正規化和)
    を毎回計算し、 **その低下量** を減衰累積する。 相手の能力が落ちれば
    1P 側にプラス、 自分の能力が落ちればマイナス。
    """

    def __init__(self) -> None:
        self.pressure = 0.0
        self._prev1: float | None = None
        self._prev2: float | None = None

    @staticmethod
    def _capability(board: Board) -> float:
        """盤面の攻撃能力を 0〜1 スケール相当で返す (指標は正規化済み)。"""
        co, _ = iv.connectivity_observation(board)
        # 連結は個数なので盤面セル数で割って 0〜1 に寄せる (指標規約に合わせる)
        conn = (co.pair_count + co.triple_count) / float(BOARD_ROWS * BOARD_COLS)
        return float(iv.saturated_chain_count(board).score) + conn

    def update(self, b1: Board, b2: Board) -> float:
        """毎フレーム。1P視点の圧力を [-100,100] で返す (正=1P攻勢)。"""
        c1 = self._capability(b1)
        c2 = self._capability(b2)
        if self._prev1 is None or self._prev2 is None:
            self._prev1, self._prev2 = c1, c2
            return 0.0
        self.pressure *= PRESSURE_DECAY
        # 相手の能力が落ちた分 = こちらの攻撃が効いた分
        self.pressure += max(0.0, self._prev2 - c2) - max(0.0, self._prev1 - c1)
        self._prev1, self._prev2 = c1, c2
        return float(max(-100.0, min(100.0,
                                     self.pressure * CAPABILITY_PRESSURE_SCALE)))


class ScoreLeadTracker:
    """(M2) 得点リード: 各side の現在スコア差(=どちらが多く攻撃を通したか)を追う。

    連鎖を撃った側のスコアが伸びる。累積差なので「今実行中の側」だけでなく
    「既により多く撃った側」を正しく評価する。velocity版は2番目に撃つ小連鎖側を
    誤って有利にしたため、累積リードへ変更(2026-07-14 M2改)。
    スコアは試合開始で0にリセットされるため試合内相対で機能する。
    """

    def __init__(self) -> None:
        self._s1 = 0
        self._s2 = 0

    def update(self, score1: int | None, score2: int | None,
               rate: float = 70.0) -> float:
        """毎フレーム。1P視点の得点リードを [-100,100] で返す(正=1Pが多く攻撃)。"""
        if score1 is not None:
            self._s1 = score1
        if score2 is not None:
            self._s2 = score2
        lead = (self._s1 - self._s2) / rate  # お邪魔換算のスコア差
        return float(max(-100.0, min(100.0, lead * SCORE_LEAD_SCALE)))


def _side_feats(board: Board, net: int, forecast: int) -> dict[str, float]:
    """1 side の安価な指標 (collect と同一関数)。"""
    co, _ = iv.connectivity_observation(board)
    return {
        "board_color_puyo_total": iv.board_color_puyo_total(board).score,
        "max_column_height": iv.max_column_height(board).score,
        "column_bumpiness": iv.column_bumpiness(board).score,
        "death_margin": iv.death_margin(board).score,
        "death_margin_neighbor": iv.death_margin_neighbor(board).score,
        "current_max_chain": iv.current_max_chain(board).score,
        "conn_pair_count": float(co.pair_count),
        "conn_triple_count": float(co.triple_count),
        "ojama_net_balance": iv.ojama_net_balance(net).score,
        "ojama_forecast": iv.ojama_forecast(forecast).score,
        "board_ojama_count": iv.board_ojama_count(board).score,
        "dig_resistance": iv.dig_resistance(board).score,
        # モデル入力 (cols) には含めない (FEATURES/FEATURE_CANDIDATES 未登録)。
        # match_progress (進行度) を算出するためだけに保持する軽量な値
        # (count_puyos()/72 のみ、simulate 不要で他指標と同程度に安価)。
        "board_puyo_total": iv.board_puyo_total(board).score,
    }


def _fill_near_future_candidate(
    f1: dict[str, float], f2: dict[str, float], b1: Board, b2: Board,
    cols: list[str],
) -> None:
    """near_future_fire_k1..k5 が cols に含まれる場合のみ計算する (列存在ガード)。

    K=1..5 は1回のビームサーチで同時取得できるため (near_future_fire_power の
    設計)、5列のうちどれか1つでも使われていれば1回だけ計算する
    (saturated_chain_count 等と同じ「使われる時だけ計算」方針を踏襲)。
    next_pair/dnext_pair は overlay のリアルタイム経路では未配線のため省略
    (理想ツモにフォールバック、iv.near_future_fire_power 側の既定動作。
    overlay 見た目採否レビュー時に next_pair 配線を検討する)。
    """
    if not any(c in cols for c in NEAR_FUTURE_FIRE_COLS):
        return
    if NEAR_FUTURE_FIRE_COLS[0] in f1:
        return
    nf1 = iv.near_future_fire_power(b1)
    nf2 = iv.near_future_fire_power(b2)
    for k in range(1, 6):
        name = f"near_future_fire_k{k}"
        f1[name] = nf1.values[k].score
        f2[name] = nf2.values[k].score


def _fill_fire_stability_candidate(
    f1: dict[str, float], f2: dict[str, float], b1: Board, b2: Board,
    cols: list[str],
) -> None:
    """fire_stability_k2/4/6 が cols に含まれる場合のみ計算する (列存在ガード)。

    _fill_near_future_candidate と同じ方針 (1回のビームサーチで K=2,4,6を
    同時取得、next_pair/dnext_pair は overlay 未配線のため省略)。
    """
    if not any(c in cols for c in FIRE_STABILITY_COLS):
        return
    if FIRE_STABILITY_COLS[0] in f1:
        return
    fs1 = iv.fire_stability(b1)
    fs2 = iv.fire_stability(b2)
    for k in (2, 4, 6):
        name = f"fire_stability_k{k}"
        f1[name] = fs1.values[k].score
        f2[name] = fs2.values[k].score


def _fill_expected_fire_candidate(
    f1: dict[str, float], f2: dict[str, float], b1: Board, b2: Board,
    cols: list[str],
) -> None:
    """expected_fire_k1/k2 が cols に含まれる場合のみ計算する (列存在ガード)。

    _fill_near_future_candidate と同じ方針。モンテカルロ (既定 N=48) のため
    他の候補指標より重い (~150-300ms/盤面、scripts/_tmp_bench_expected_fire.py
    実測) 点に留意 (呼出元 HeavyAdvCache は間引き実行のため許容コスト)。
    """
    if not any(c in cols for c in EXPECTED_FIRE_COLS):
        return
    if EXPECTED_FIRE_COLS[0] in f1:
        return
    ef1 = iv.expected_fire_power(b1)
    ef2 = iv.expected_fire_power(b2)
    for k in (1, 2):
        name = f"expected_fire_k{k}"
        f1[name] = ef1.values[k].score
        f2[name] = ef2.values[k].score


def _score_advantage(
    model, b1: Board, b2: Board, snap: OjamaAccountSnapshot,
    feature_cols: tuple[str, ...] | list[str] | None = None,
) -> tuple[float, float, list[tuple[str, float]]]:
    """両盤面 → (有利不利[-100..100], 1P勝率, 主要ドライバ)。

    feature_cols: optional。省略時は model._puyo_feature_cols (学習時に
    _train_model が格納した実特徴量列) を使い、無ければ従来通り FEATURES に
    フォールバックする。既存呼出元は本引数を渡さないため挙動は変わらない。
    """
    cols = list(feature_cols) if feature_cols is not None else list(
        getattr(model, "_puyo_feature_cols", FEATURES))
    f1 = _side_feats(b1, snap.net_balance_capped, snap.forecast_p1)
    f2 = _side_feats(b2, -snap.net_balance_capped, snap.forecast_p2)
    # 候補指標 (saturated_chain_count/ukeyasusa/sub_chain_count) は cols に
    # 含まれる場合のみ計算 (列存在ガードで通常は不要な board sim コストが
    # 発生しない)。呼出元(HeavyAdvCache)は間引き実行のため許容コスト。
    _candidate_fns = {
        "saturated_chain_count": iv.saturated_chain_count,
        "ukeyasusa": iv.ukeyasusa,
        "sub_chain_count": iv.sub_chain_count,
    }
    for name, fn in _candidate_fns.items():
        if name in cols and name not in f1:
            f1[name] = fn(b1).score
            f2[name] = fn(b2).score
    _fill_near_future_candidate(f1, f2, b1, b2, cols)
    _fill_fire_stability_candidate(f1, f2, b1, b2, cols)
    _fill_expected_fire_candidate(f1, f2, b1, b2, cols)
    diff = {c: f1[c] - f2[c] for c in cols}
    # 交互作用 (色ぷよ差 × おじゃまフラット度) を学習時と同じ順序で末尾に足す。
    # 学習側が使っていなければ足さない (model._puyo_uses_interaction で判定)。
    x_cols = list(cols)
    if getattr(model, "_puyo_uses_interaction", False):
        flat = float(_ojama_flat_score(
            diff.get("board_ojama_count", 0.0),
            diff.get("ojama_forecast", 0.0),
        ))
        diff[COLOR_OJAMA_INTERACTION_COL] = (
            diff.get("board_color_puyo_total", 0.0) * flat
        )
        x_cols.append(COLOR_OJAMA_INTERACTION_COL)
        # フラット度そのもの (学習時と同じ順序で末尾に付ける)
        diff[OJAMA_FLAT_COL] = flat
        x_cols.append(OJAMA_FLAT_COL)
    # 進行度文脈列 (match_progress / color_puyo_x_earliness) を学習時と
    # 同じ順序で末尾に足す (2026-08-10 Phase1-1 B-2)。学習側が使っていなければ
    # 足さない (model._puyo_uses_progress で判定、既存 interaction と同じ方式)。
    if getattr(model, "_puyo_uses_progress", False):
        progress = float(_match_progress_from_totals(
            f1.get("board_puyo_total", 0.0), f2.get("board_puyo_total", 0.0),
        ))
        diff[MATCH_PROGRESS_COL] = progress
        x_cols.append(MATCH_PROGRESS_COL)
        diff[COLOR_EARLINESS_INTERACTION_COL] = (
            diff.get("board_color_puyo_total", 0.0) * (1.0 - progress)
        )
        x_cols.append(COLOR_EARLINESS_INTERACTION_COL)
    x = np.array([[diff[c] for c in x_cols]], dtype=float)
    p1 = float(model.predict_proba(x)[0, 1])
    adv = (p1 - 0.5) * 200.0
    drivers = sorted(
        ((c, diff[c]) for c in JP_LABEL if c in diff),
        key=lambda kv: -abs(kv[1]))[:3]
    return adv, p1, drivers


def _threat(b1: Board, b2: Board, sp1, sp2, elapsed: float) -> float:
    """(3/M1) 火力threat = 到達火力差 1P−2P を [-100,100] で返す。

    reach_fire_power(実 next/dnext ペアで2手先読み)を使う。潜在火力
    (potential_fire_power)は greedy 探索が大連鎖を過小評価するバグがあり
    (2026-07-14 Phase1: あん実816に対し potential=360 だが reach=956 と的中)、
    「実際に撃てる火力」を測る reach の方が有利不利に正確なため置換。
    """
    r1 = iv.reach_fire_power(b1, sp1.next_pair, sp1.dnext_pair, elapsed).value.raw
    r2 = iv.reach_fire_power(b2, sp2.next_pair, sp2.dnext_pair, elapsed).value.raw
    return float(max(-100.0, min(100.0, (r1 - r2) * THREAT_SCALE)))


def _forecast_signal(snap: OjamaAccountSnapshot) -> float:
    """(旧M3) 会計の確定予告(連鎖終了時生成)差。ラグがあるため RealtimeForecastTracker を使う。"""
    diff = float(snap.forecast_p2 - snap.forecast_p1)  # 2P incoming - 1P incoming
    return float(max(-100.0, min(100.0, diff * FORECAST_SCALE)))


class RealtimeForecastTracker:
    """(M3改B) pending お邪魔を「相殺 + 30個/ターン配送」でリアルタイム管理する位置信号。

    お邪魔会計(OjamaAccountingTracker)と同じ配送モデルを、連鎖終了を待たず実行中に:
      - 相手が発火(score増)→ 生成量を算出。まず自分の pending を相殺、余剰を相手 pending へ。
      - 自分がツモを置く(tsumo増)→ 自分の pending から最大30個/ターンを盤面へ配送(pending減)。
    「累積スコア(結果)」でも「時間減衰(粗い)」でもなく、実際の降り方(5段=30個ずつ)を
    反映した「まだ降る残りお邪魔=位置」。大連鎖でも一気に埋まらず、返し(相殺)で相殺される。
    配送された分は盤面お邪魔(=圧力)が引き継ぐ。
    """

    def __init__(self) -> None:
        self.inc1 = 0.0  # 1P にこれから降る pending
        self.inc2 = 0.0  # 2P にこれから降る pending
        self._s1: int | None = None
        self._s2: int | None = None
        self._t1: int | None = None
        self._t2: int | None = None

    def _fire(self, gen: float, own_pending: float) -> tuple[float, float]:
        """発火生成 gen で自分の pending を相殺し (残own_pending, 相手へ余剰) を返す。"""
        canceled = min(gen, own_pending)
        return own_pending - canceled, gen - canceled

    def update(self, score1: int | None, score2: int | None,
               tsumo1: int, tsumo2: int, rate: float = 70.0) -> float:
        """1P視点の予告信号 [-100,100](正=2Pに多く降る=1P有利)。"""
        # 試合境界(スコア大幅減)で pending クリア
        if ((score1 is not None and self._s1 is not None and self._s1 - score1 >= 1000)
                or (score2 is not None and self._s2 is not None and self._s2 - score2 >= 1000)):
            self.inc1 = self.inc2 = 0.0
        # 発火(score増)→ 相殺 + 余剰を相手 pending へ
        if score1 is not None:
            if self._s1 is not None and score1 > self._s1:
                self.inc1, surplus = self._fire((score1 - self._s1) / rate, self.inc1)
                self.inc2 += surplus
            self._s1 = score1
        if score2 is not None:
            if self._s2 is not None and score2 > self._s2:
                self.inc2, surplus = self._fire((score2 - self._s2) / rate, self.inc2)
                self.inc1 += surplus
            self._s2 = score2
        # 配送(ツモ増)→ 自分の pending から 30個/ターン 盤面へ(=pending減、圧力が引継ぐ)
        if self._t1 is not None and tsumo1 > self._t1:
            self.inc1 = max(0.0, self.inc1 - FORECAST_DROP_PER_TURN * (tsumo1 - self._t1))
        if self._t2 is not None and tsumo2 > self._t2:
            self.inc2 = max(0.0, self.inc2 - FORECAST_DROP_PER_TURN * (tsumo2 - self._t2))
        self._t1, self._t2 = tsumo1, tsumo2
        return float(max(-100.0, min(100.0, (self.inc2 - self.inc1) * FORECAST_SCALE)))


class ThreatTracker:
    """threat(reach火力=重い)の計算を間引くキャッシュ。

    reach_fire_power は満杯盤面で高コスト。threat は連鎖ビルドに従い緩やかに
    変化するため、every フレームに1回だけ再計算し間は前回値を再利用する。
    毎フレーム呼んでも reach 実計算は 1/every に削減。
    """

    def __init__(self, every: int = 9) -> None:  # ~0.3s @30fps
        self._every = max(1, every)
        self._last = 0.0
        self._n = 0

    def update(self, b1: Board, b2: Board, sp1, sp2, elapsed: float) -> float:
        if self._n % self._every == 0:
            self._last = _threat(b1, b2, sp1, sp2, elapsed)
        self._n += 1
        return self._last


class HeavyAdvCache:
    """重い盤面由来計算(_score_advantage の current_max_chain 等 + threat の reach)を
    ~0.3s 毎に計算してキャッシュ。盤面は STABLE 時しか変わらないため精度低下は最小。

    毎フレーム呼んでも重い simulate 群は 1/every に削減(オーバーレイ実用速度の要)。
    圧力(board_ojama)・得点リード(score)・会計は安価なので呼出側で毎フレーム更新のまま。
    ukeyasusa (dig_resistance 含む連鎖シミュ) もここで間引き計算してキャッシュする。
    2026-07 正式採用によりモデル特徴量 (FEATURE_CANDIDATES) としても
    _score_advantage 側で使われる(こちらは表示専用の重複計算)。
    saturated_chain_count (飽和連鎖量) は依然コア adv 非混入の表示専用候補。
    """

    def __init__(self, model, every: int = 9) -> None:  # ~0.3s @30fps
        self._model = model
        self._every = max(1, every)
        self._n = 0
        self._adv = 0.0
        self._threat = 0.0
        self._drivers: list[tuple[str, float]] = []
        # 受けやすさキャッシュ (0〜1 正規化済み)
        self._ukey1: float = 0.0
        self._ukey2: float = 0.0
        # 飽和連鎖量キャッシュ (0〜1 正規化済み。表示専用・コア adv には混ぜない)
        self._sat1: float = 0.0
        self._sat2: float = 0.0

    def update(self, b1: Board, b2: Board, snap: OjamaAccountSnapshot,
               sp1, sp2, elapsed: float,
               ) -> tuple[float, float, list[tuple[str, float]], float, float, float, float]:
        """(モデル有利不利, threat, 主要ドライバ, ukey1, ukey2, sat1, sat2) を返す(間引きキャッシュ)。

        ukey1/ukey2: 受けやすさ (0〜1)。dig_resistance 込みのため毎フレーム計算は高コスト。
        sat1/sat2: 飽和連鎖量 (0〜1、iv.saturated_chain_count の score)。board sim を
        伴うため同様に every 間引きで十分(盤面は STABLE 時しか変わらない)。
        """
        if self._n % self._every == 0:
            self._adv, _, self._drivers = _score_advantage(self._model, b1, b2, snap)
            self._threat = _threat(b1, b2, sp1, sp2, elapsed)
            # 受けやすさ: dig_resistance を含むため every と同じタイミングで計算
            self._ukey1 = iv.ukeyasusa(b1).score
            self._ukey2 = iv.ukeyasusa(b2).score
            # 飽和連鎖量: 表示専用ライブ計算 (ukeyasusa と同じ間引きキャッシュ)
            self._sat1 = iv.saturated_chain_count(b1).score
            self._sat2 = iv.saturated_chain_count(b2).score
        self._n += 1
        return (self._adv, self._threat, self._drivers,
                self._ukey1, self._ukey2, self._sat1, self._sat2)


def _fresh_trackers(
    model,
) -> tuple[OjamaAccountingTracker, "_SideTracker", "_SideTracker",
           PressureTracker, RealtimeForecastTracker, ScoreLeadTracker, HeavyAdvCache,
           EarlyFireTracker]:
    """スコアリセット検知時に持続トラッカー一式を初期状態で作り直す。

    各トラッカーは内部 state が僅少 (数個の float/Counter) のため、都度
    再生成するだけで「初期化」と等価であり専用 reset() の追加は不要
    (OjamaAccountingTracker のみ既存 reset() を呼び互換 API を維持する)。
    戻り値末尾に EarlyFireTracker を追加 (2026-07-29、既存呼出元は1箇所のみで
    アンパック先も同時更新済みのため後方互換上の実害なし)。
    """
    tracker = OjamaAccountingTracker()
    tracker.reset()
    return (tracker, _SideTracker(), _SideTracker(),
            PressureTracker(), RealtimeForecastTracker(), ScoreLeadTracker(),
            HeavyAdvCache(model), EarlyFireTracker())


def _pick_recog_display_board(
    side_result: object, frozen_board: "Board | None",
) -> tuple["Board | None", bool]:
    """認識色 overlay 描画用に、CHAIN/GRAVITY_SETTLE 中は estimated_board
    (物理推論スルー盤面、src/recognition_pipeline.py 反復5-6) を、それ以外は
    凍結済み STABLE 盤面 (frozen_board) を返す (反復10 viz拡張)。

    src/ は無改修 (SideResult の既存 optional フィールドを参照するのみ)。
    古い SideResult (estimated_board 未搭載) でも getattr で安全に動く。

    Args:
        side_result: PipelineResult.p1 または .p2 (SideResult)。
        frozen_board: 直近 STABLE で確定した盤面 (従来の b1/b2)。

    Returns:
        (描画対象 board, 低信頼度フラグ)。board が None なら描画対象なし。
    """
    state = getattr(side_result, "state", None)
    if state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE):
        estimated = getattr(side_result, "estimated_board", None)
        if estimated is not None:
            provenance = getattr(side_result, "board_provenance", "observed")
            return estimated, provenance == "chain_estimate_low_confidence"
    return frozen_board, False


def _graph_geometry(
    render_area: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int, int]:
    """グラフ描画領域 (gx0, gx1, gy0, gy1, title_y) を返す (stateless・単体テスト対象)。

    render_area が None なら従来の overlay レイアウト (ゲーム画面の下の黒帯、
    TOP_H/OUT_W/OUT_H/CANVAS_H から算出) を計算する (後方互換、値は既存と
    完全一致)。render_area=(x0, y0, w, h) を指定した場合は、その矩形内に
    タイトル+枠+プロットを収める (panel レイアウト用)。
    """
    if render_area is None:
        game_bottom = TOP_H + OUT_H  # ゲーム画面の下端 y 座標
        gx0, gx1 = 40, OUT_W - 40
        gy0, gy1 = game_bottom + 26, CANVAS_H - 12
        return gx0, gx1, gy0, gy1, gy0 - 20
    x0, y0, w, h = render_area
    margin_x = 40
    gx0, gx1 = x0 + margin_x, x0 + w - margin_x
    gy0, gy1 = y0 + 30, y0 + h - 12
    return gx0, gx1, gy0, gy1, y0 + 6


def _draw_graph(
    d: "ImageDraw.ImageDraw", history: list[tuple[float, float]],
    t_rel: float, total: float,
    render_area: tuple[int, int, int, int] | None = None,
) -> None:
    """リアルタイム評価値グラフ (将棋風) を描画する。進行に合わせ伸びる。

    render_area: optional。省略時は従来通りゲーム画面下の黒帯に描く
    (後方互換、既存呼出元は挙動不変)。panel レイアウトでは (x0,y0,w,h) の
    矩形を明示して左下グラフ領域に描く (_draw_panel_layout 参照)。
    """
    gx0, gx1, gy0, gy1, title_y = _graph_geometry(render_area)
    gyc = (gy0 + gy1) // 2
    gw, gh = gx1 - gx0, gy1 - gy0
    d.rectangle([gx0 - 4, title_y, gx1 + 4, gy1 + 4], fill=(0, 0, 0, 150))
    d.text((gx0, title_y), "有利不利グラフ (0=互角 上1P/下2P)", font=_font(15),
           fill=(255, 255, 255))
    total = max(total, 1.0)

    def _px(t: float) -> int:
        return int(gx0 + (t / total) * gw)

    def _py(a: float) -> int:
        return int(gyc - (max(-100, min(100, a)) / 100.0) * (gh / 2))
    for t, a in history:  # 各点で中央線から値まで縦線 (塗り面風)
        col = (90, 140, 220) if a >= 0 else (210, 90, 90)
        d.line([(_px(t), gyc), (_px(t), _py(a))], fill=col, width=2)
    d.line([(gx0, gyc), (gx1, gyc)], fill=(255, 255, 255), width=1)
    ph = _px(t_rel)  # 再生ヘッド
    d.line([(ph, gy0), (ph, gy1)], fill=(255, 255, 0), width=2)
    d.rectangle([gx0, gy0, gx1, gy1], outline=(255, 255, 255), width=1)


def _draw_ukeyasusa(
    d: "ImageDraw.ImageDraw", ukey1: float, ukey2: float,
    x0: int, y_top: int,
) -> None:
    """受けやすさ (1P / 2P) を小さく描画するサブ関数。

    ukey1/ukey2: 0〜1 正規化値。差分(1P-2P)を色と数値で表示する。
    本関数自体は表示専用(HeavyAdvCache の間引きキャッシュ値を渡すだけ)だが、
    2026-07 の正式採用により受けやすさは FEATURE_CANDIDATES 経由でモデル特徴量
    (ukeyasusa_diff) としても使われる。表示値と学習内部の計算は独立(重複計算)
    だが同じ iv.ukeyasusa() を呼ぶため値は一致する。
    """
    diff = ukey1 - ukey2  # 正=1P有利
    diff_col = (120, 200, 120) if diff >= 0 else (200, 120, 120)
    label = (
        f"受けやすさ  1P {ukey1:.2f} / 2P {ukey2:.2f}"
        f"  (差 {diff:+.2f})"
    )
    d.text((x0, y_top), label, font=_font(15), fill=diff_col)


def _draw_saturated(
    d: "ImageDraw.ImageDraw", sat1: float, sat2: float,
    x0: int, y_top: int,
) -> None:
    """飽和連鎖量 (1P / 2P) を小さく描画するサブ関数 (ukeyasusa と同一パターン)。

    sat1/sat2: 0〜1 正規化値 (iv.saturated_chain_count の score)。差分(1P-2P)を
    色と数値で表示する。飽和連鎖量はコアの有利不利スコアに混ぜない(表示専用)。
    """
    diff = sat1 - sat2  # 正=1P有利
    diff_col = (120, 200, 120) if diff >= 0 else (200, 120, 120)
    label = (
        f"飽和連鎖  1P {sat1:.2f} / 2P {sat2:.2f}"
        f"  (差 {diff:+.2f})"
    )
    d.text((x0, y_top), label, font=_font(15), fill=diff_col)


def _draw_bar(
    d: "ImageDraw.ImageDraw", adv: float, waiting: bool, cx: int, x0: int,
    top: int = PANEL_BAR_TOP, bar_w: int = PANEL_BAR_W, bar_h: int = PANEL_BAR_H,
    label_font_size: int = 22, verdict_font_size: int = 24,
) -> None:
    """有利不利バー本体(色分け矩形 + 判定テキスト + 1P/2Pラベル)を描画するサブ関数。

    _draw_overlay から分離 (1関数50行以内の規約対応)。top/bar_w/bar_h/
    label_font_size/verdict_font_size は既定値が従来の PANEL_BAR_* 定数と
    一致するため、呼出元が指定しなければ従来の overlay レイアウトと完全に
    同じ描画になる (backwards compat)。panel レイアウトは _draw_panel_info
    から異なる座標・サイズを明示的に渡す。
    """
    d.rectangle([x0, top, cx, top + bar_h], fill=(70, 110, 200, 180))   # 1P側(青)
    d.rectangle([cx, top, x0 + bar_w, top + bar_h], fill=(200, 80, 80, 180))  # 2P側(赤)
    d.rectangle([x0, top, x0 + bar_w, top + bar_h], outline=(255, 255, 255), width=2)
    if waiting:
        d.text((cx - 90, top + 4), "STABLE 待ち", font=_font(label_font_size),
               fill=(255, 255, 255))
        return
    mx = int(cx - (max(-100, min(100, adv)) / 100.0) * (bar_w // 2))  # adv>0=1P=左
    d.rectangle([mx - 3, top - 6, mx + 3, top + bar_h + 6], fill=(255, 255, 255))
    verdict = ("互角" if abs(adv) < EVEN_THRESHOLD
               else f"{'1P' if adv > 0 else '2P'} 有利  {abs(adv):.0f}")
    d.text((cx - 70, top + 4), verdict, font=_font(verdict_font_size), fill=(0, 0, 0))
    d.text((x0 - 34, top + 4), "1P", font=_font(label_font_size), fill=(150, 200, 255))
    d.text((x0 + bar_w + 6, top + 4), "2P", font=_font(label_font_size), fill=(255, 180, 180))


def _draw_overlay(
    frame: np.ndarray, adv: float, p1: float,
    drivers: list[tuple[str, float]], waiting: bool,
    history: list[tuple[float, float]], t_rel: float, total: float,
    ukey1: float = 0.0, ukey2: float = 0.0,
    sat1: float = 0.0, sat2: float = 0.0,
) -> np.ndarray:
    """上部情報パネル(黒帯)+ ゲーム画面(無地) + 下部グラフ帯 を合成して1フレーム描画する。

    2026-07 改修: 盤面(ゲーム画面 y∈[TOP_H, TOP_H+OUT_H))には一切描画しない
    (旧版は盤面に重ねて視認性を損なっていた)。有利不利バー/勝率/主因/
    受けやすさ/飽和連鎖の全情報は上部パネル (y∈[0, TOP_H)) に集約する。
    ukey1/ukey2/sat1/sat2: optional。HeavyAdvCache 由来の表示専用値。
    saturated_chain_count(sat1/sat2)は依然コアの adv 計算に混ぜない候補指標
    (表示専用)。ukeyasusa(ukey1/ukey2)は2026-07 正式採用によりモデル特徴量
    (FEATURE_CANDIDATES 経由)としても adv に寄与するが、本関数へ渡す表示値
    自体は独立計算(値は一致するが adv 計算のパイプラインとは別経路)。
    """
    canvas = Image.new("RGB", (OUT_W, CANVAS_H), (12, 12, 16))
    canvas.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (0, TOP_H))
    img = canvas
    d = ImageDraw.Draw(img, "RGBA")
    cx = OUT_W // 2
    x0 = cx - PANEL_BAR_W // 2
    d.text((x0, PANEL_TITLE_Y), "有利不利オーバーレイ (試作・tier1軽量モデル)",
           font=_font(20), fill=(255, 255, 0))
    _draw_bar(d, adv, waiting, cx, x0)
    if not waiting:
        d.text((x0, PANEL_WINPROB_Y),
               f"勝率  1P {p1 * 100:.0f}%   /   2P {(1 - p1) * 100:.0f}%",
               font=_font(20), fill=(255, 255, 255))
        dl = "  ".join(f"{JP_LABEL[c]}差 {v:+.2f}" for c, v in drivers)
        d.text((x0, PANEL_DRIVERS_Y), f"主因: {dl}", font=_font(16), fill=(230, 230, 180))
        # 受けやすさ差・飽和連鎖差 (コアの adv には影響しない表示専用)
        _draw_ukeyasusa(d, ukey1, ukey2, x0, PANEL_UKEY_Y)
        _draw_saturated(d, sat1, sat2, x0, PANEL_SAT_Y)
    if history:
        _draw_graph(d, history, t_rel, total)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _draw_panel_drivers(
    d: "ImageDraw.ImageDraw", drivers: list[tuple[str, float]],
    x0: int, y_top: int,
) -> None:
    """主因を1行1項目で縦に並べて描画する (情報パネルの狭い幅向けの折り返し)。

    _score_advantage が返す drivers は既に上位3件に絞られているため、
    1件ずつ改行して描くだけで480px幅パネル内に収まる (横一列連結だと
    従来の overlay レイアウトでは横幅720pxを使えたが、パネルはそれより
    狭いため折り返しが必須)。
    """
    d.text((x0, y_top), "主因:", font=_font(18), fill=(230, 230, 180))
    for i, (c, v) in enumerate(drivers):
        label = f"{JP_LABEL[c]}差 {v:+.2f}"
        d.text((x0, y_top + 24 + i * PANEL_INFO_DRIVER_LINE_H), label,
               font=_font(18), fill=(230, 230, 180))


def _draw_panel_info(
    d: "ImageDraw.ImageDraw", box: tuple[int, int, int, int],
    adv: float, p1: float, waiting: bool, drivers: list[tuple[str, float]],
    state1: str, state2: str, counter_text: str, elapsed_sec: float,
) -> None:
    """右側縦長情報パネル (バー/勝率/主因/状態/経過時刻) を描画する。

    box: (x0, y0, w, h) の矩形 (panel_layout_regions()["info"])。
    バー描画は既存 _draw_bar を座標だけ差し替えて再利用する
    (「読みやすさ優先」のため overlay 版よりバー太め・勝率フォント大きめ)。
    """
    x0, y0, w, h = box
    d.rectangle([x0, y0, x0 + w, y0 + h], fill=(18, 18, 24))
    pad = PANEL_INFO_PAD
    cx = x0 + w // 2
    # バー本体は「1P」「2P」ラベル分の余白 (PANEL_INFO_BAR_LABEL_MARGIN) を
    # さらに内側へ寄せて確保する (パネル外・映像領域へのラベル溢れ防止)。
    bar_inset = pad + PANEL_INFO_BAR_LABEL_MARGIN
    bar_w = w - bar_inset * 2
    _draw_bar(d, adv, waiting, cx, x0 + bar_inset,
              top=y0 + PANEL_INFO_BAR_TOP_OFFSET, bar_w=bar_w, bar_h=PANEL_INFO_BAR_H,
              label_font_size=20, verdict_font_size=26)
    if waiting:
        return
    d.text((x0 + pad, y0 + PANEL_INFO_WINPROB_Y1), f"1P {p1 * 100:.0f}%",
           font=_font(52), fill=(150, 200, 255))
    d.text((x0 + pad, y0 + PANEL_INFO_WINPROB_Y2), f"2P {(1 - p1) * 100:.0f}%",
           font=_font(52), fill=(255, 180, 180))
    _draw_panel_drivers(d, drivers, x0 + pad, y0 + PANEL_INFO_DRIVERS_Y)
    d.text((x0 + pad, y0 + PANEL_INFO_STATE1_Y), f"1P状態: {state1}",
           font=_font(22), fill=(200, 220, 255))
    d.text((x0 + pad, y0 + PANEL_INFO_STATE2_Y), f"2P状態: {state2}",
           font=_font(22), fill=(255, 210, 210))
    if counter_text:
        d.text((x0 + pad, y0 + PANEL_INFO_COUNTER_Y), counter_text,
               font=_font(18), fill=(220, 220, 150))
    d.text((x0 + pad, y0 + h - PANEL_INFO_ELAPSED_BOTTOM_MARGIN),
           f"経過 {elapsed_sec:.0f} 秒", font=_font(20), fill=(200, 200, 200))


def _draw_panel_layout(
    frame: np.ndarray, adv: float, p1: float,
    drivers: list[tuple[str, float]], waiting: bool,
    history: list[tuple[float, float]], t_rel: float, total: float,
    state1: str, state2: str, counter_text: str, elapsed_sec: float,
) -> np.ndarray:
    """パネルレイアウト (左上映像+左下グラフ+右情報パネル+下端字幕帯、1920x1080)
    で1フレーム描画する。

    2026-08-10 user指示の新レイアウト (同日、下端字幕帯の追記込み)。既存の
    _draw_overlay (盤面上に直接バー等を重ねる従来レイアウト) とは完全に独立
    した経路であり、 --layout panel 指定時のみ呼ばれる (既定 layout=overlay
    では未使用、既存出力は一切変わらない)。下端の字幕帯 (regions["subtitle"])
    には背景色を塗るだけで文字・図形を一切描かない (user要求の絶対条件)。
    """
    regions = panel_layout_regions()
    canvas = Image.new("RGB", (PANEL_CANVAS_W, PANEL_CANVAS_H), (12, 12, 16))
    vx, vy, vw, vh = regions["video"]
    video_rgb = cv2.cvtColor(
        cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    canvas.paste(Image.fromarray(video_rgb), (vx, vy))
    d = ImageDraw.Draw(canvas, "RGBA")
    sx, sy, sw, sh = regions["subtitle"]
    d.rectangle([sx, sy, sx + sw, sy + sh], fill=PANEL_SUBTITLE_BG_COLOR)  # 字幕帯: 無描画
    if history:
        _draw_graph(d, history, t_rel, total, render_area=regions["graph"])
    _draw_panel_info(d, regions["info"], adv, p1, waiting, drivers,
                     state1, state2, counter_text, elapsed_sec)
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


# ============================
# 認識フラグの既定値解決 (2026-07-31)
# ============================

# 「スクリプト側が明示 False を渡すため、ライブラリ既定 True の認識改善が
# viz では一切効いていない」罠を潰すための仕組み。
# 実測 (inspect) で食い違っていたのは 7 つ:
#   enable_landing_observed_color / enable_match_start_full_clear /
#   enable_recovery_counter_carryover / enable_cnn_flicker_hsv_fallback /
#   enable_initial_confirm_vote / enable_drift_resync_match_start_guard /
#   enable_drift_resync_hsv_gate
# いずれもライブラリ既定 True に対しスクリプトが False を渡していた。
# → **viz で承認したのに本番は別挙動**という事故が起きうる (レビュー制度の破綻)。
#
# 方式: スクリプト側の既定を None にし、None のときは
# RecognitionPipeline.load_default の既定値を inspect で読んで採用する。
# 明示指定 (--flag / --no-flag) されたときだけそれを尊重する。
# (scripts/measure_stable_cell_acc.py:354 と同じ方式)


def _pipeline_default(name: str) -> bool:
    """RecognitionPipeline.load_default の当該引数の既定値を返す。

    引数が存在しない場合は False (呼び出し側で無害な既定として扱う)。
    """
    params = inspect.signature(RecognitionPipeline.load_default).parameters
    param = params.get(name)
    return bool(param.default) if param is not None else False


def _resolve_flag(name: str, value: "bool | None") -> bool:
    """None ならライブラリ既定に解決し、明示指定はそのまま返す。"""
    return _pipeline_default(name) if value is None else bool(value)


def _build_counter_text(counter_p1: float, counter_p2: float) -> str:
    """panel レイアウトの応手情報行の文字列を作る (無効/未計算時は空文字)。

    counter-reach 無効時 (enable_counter_reach=False) は counter_p1/p2 が
    nan のままなので空文字を返し、_draw_panel_info 側で行自体を描かない。
    """
    if math.isnan(counter_p1) or math.isnan(counter_p2):
        return ""
    return f"応手確率  1P {counter_p1 * 100:.0f}%  /  2P {counter_p2 * 100:.0f}%"


def generate(video: Path, out: Path, max_sec: float, sample_interval: float,
             start_sec: float = 0.0, end_sec: float = 0.0,
             exclude_video: str | None = None, warmup_sec: float = 0.0,
             show_recognition: bool = False,
             enable_landing_observed_color: bool | None = None,
             force_in_match: bool = True,
             enable_drift_guards: bool | None = None,
             enable_match_start_full_clear: bool | None = None,
             enable_recovery_counter_carryover: bool | None = None,
             enable_cnn_flicker_hsv_fallback: bool | None = None,
             enable_initial_confirm_vote: bool | None = None,
             enable_platt_calibration: bool = False,
             enable_early_fire_reaction: bool = False,
             enable_per_side_settled: bool = False,
             disable_score_lead_bias: bool = False,
             enable_capability_pressure: bool = False,
             disable_pressure: bool = False,
             enable_counter_reach: bool = False,
             enable_puyo_to_empty_hsv_guard: bool | None = None,
             layout: str = "overlay") -> int:
    """有利不利オーバーレイ動画を生成。書き出しフレーム数を返す。

    start_sec: 書き出し開始秒 (ゲームの真の開始=スコア0の瞬間)。
    warmup_sec: start_sec の何秒前から「処理だけ」始めるか (状態機械/会計の初期化用。
        この区間は認識を通すが動画には書き出さない)。
    end_sec: 書き出し終了秒。
    show_recognition: True で scripts/visualize_recognition.py の認識色 overlay
        (盤面セルに認識色記号+ state 枠) を合成する (2026-07-23 追加)。
        既定 False = 従来通り無地の盤面 (後方互換、既存呼出元は挙動不変)。
        認識自体は従来通り縮小済み frame で行う(推論経路は不変)。表示専用に
        ネイティブ解像度 (1920x1080) でオーバーレイを描いてから縮小するため、
        visualize_recognition.py の ROI 定数がそのまま使える。
    enable_landing_observed_color: RecognitionPipeline.load_default に渡す
        着地セル CNN==HSV 一致色補正フラグ (2026-07-25 レビュー動画#49で追加)。
        既定 False = 従来挙動 (後方互換、既存呼出元は挙動不変)。
    force_in_match: RecognitionPipeline.load_default に渡す試合中強制フラグ。
        既定 True = 従来挙動 (後方互換、既存呼出元は挙動不変)。本スクリプト
        本来の想定 (既に試合中で始まる短いクリップ) では MatchStateDetector が
        導入部の欠如で NOT_IN_MATCH 誤判定するのを避けるため True が必要だが、
        試合0本の境界(前試合のリザルト/ロード演出)を跨ぐフル試合レンダでは
        逆に「試合外」凍結が効かず、ロード演出中の装飾アイコンを盤面と誤認して
        書き込む副作用がある (2026-07-25 c34 game1 「青2個」実測で確認)。
        試合境界を跨ぐ場合は False を明示指定する (2026-07-25 追加)。
    enable_drift_guards: True で DriftDetector 再同期暴走ガード2種
        (試合開始15秒保護窓 + HSV較正3色未満抑制) を有効化する
        (2026-07-25 レビュー動画v3で追加)。既定 False = 従来挙動。
    enable_match_start_full_clear: RecognitionPipeline.load_default に渡す
        前試合盤面残骸リーク修正フラグ (幽霊B対策、2026-07-23 追加)。
        既定 False = 従来挙動 (後方互換、既存呼出元は挙動不変)。
    enable_recovery_counter_carryover: RecognitionPipeline.load_default に渡す
        #51 復旧カウンタ carryover フラグ (2026-07-26 追加)。既定 False = 従来挙動
        (後方互換、既存呼出元は挙動不変)。
    enable_cnn_flicker_hsv_fallback: RecognitionPipeline.load_default に渡す
        #51 後半 CNN 乱高下セル HSV フォールバックフラグ (2026-07-26 追加)。
        既定 False = 従来挙動 (後方互換、既存呼出元は挙動不変)。
    enable_initial_confirm_vote: RecognitionPipeline.load_default に渡す
        初回 STABLE 確定の多数決ガードフラグ (2026-07-27 追加)。既定 False = 従来挙動
        (後方互換、既存呼出元は挙動不変)。
    enable_platt_calibration: 表示用勝率 (adv_to_winprob の出力) に Platt scaling
        後段校正を適用する (2026-07-29 追加、data/indicators_v2/platt_calibration.json
        を読む)。**既定 False (暫定)**。user は「入れる」と承認済みで最終的には既定 ON に
        すべきだが、以下2点が未解決のため暫定的に False としている:
          (1) 校正器ファイル自体がまだ生成されていない (学習ジョブが setsid detach
              されておらず親の終了で kill された。CPU が空いてから再実行が必要)。
              既定 True のままだと generate() を既定引数で呼ぶだけで必ず例外になる。
          (2) 学習時分布と適用時分布が一致しない。校正器は model_indicator_win.py の
              全指標 HistGBC (combined66) で学習されるが、適用先は本ファイルの4成分
              ブレンド (model 成分の重みは W_MODEL=0.20 のみ) を adv_to_winprob の
              別 sigmoid 較正で確率化したもの。実測した改善値 (ECE 0.0264->0.0189、
              終盤 p95 +72.3->+47.8) は HistGBC に対する数値で、**本表示に対する
              数値ではない**。overlay 自身の校正 (EVEN判定頻度・p95の変化) は未測定。
        → 上記(1)を解消し(2)を overlay 経路で実測して妥当性を確認したら既定 True に
          戻すこと。False にすると従来挙動 (校正なし) を完全再現する。
        True かつ校正器ファイルが無い場合は CalibrationFileMissingError を
        処理開始前(動画を1フレームも読む前)に送出する(黙って未校正で通さない
        ためのガード。fail-fast のため重い動画処理を無駄にしない)。
        校正器は model_indicator_win.py の全指標モデル(combined66データ)で
        学習されたものであり、本スクリプトの4成分ブレンドモデルとは生成過程が
        異なるため近似適用である点に注意 (詳細は PLATT_CALIBRATION_PATH 定義部・
        _apply_platt_to_display のコメント参照)。
    enable_early_fire_reaction: True で EarlyFireTracker (chain_event 検知フレームで
        即座に反映する速報バイアス) を表示に加算する (2026-07-29 userレビュー指摘1/2
        対処、追加)。既定 False = 従来挙動 (settled ゲートのみ、後方互換、既存呼出元は
        挙動不変)。confirmed_board は変更しないサイドチャネル (詳細は EarlyFireTracker
        docstring 参照)。adv_ema/p1_last 自体 (EMA 内部状態) には混ぜず、表示直前
        (グラフ点・バー・勝率テキスト) にのみ加算するため、無効時は完全に従来経路と
        ビット一致する。
    enable_per_side_settled: True で「片側でも STABLE なら再計算」に切り替える
        (2026-08-08 追加)。従来の「両者同時 STABLE」ゲートは実測で試合時間の
        72.3%・最長 13.97 秒も評価を凍結させており、大連鎖の撃ち合い中に
        盤面が激変しても判定が動かない原因だった。b1/b2 は片側ずつ更新される
        凍結盤面なので、片側 STABLE でも最新同士で計算できる。
        既定 False = 従来挙動と完全一致 (backwards compat)。
    disable_score_lead_bias: True で得点タイブレーク (ScoreLeadTracker) を
        無効化する (2026-08-09 user伝授)。スコアはおじゃまを送る手段であり、
        送った時点で意味を失う (送ったぶんは予告おじゃま/盤面おじゃまとして
        既に観測できるため二重計上になる)。既定 False = 従来挙動維持。
    enable_puyo_to_empty_hsv_guard: RecognitionPipeline.load_default に渡す
        色→空 HSV 照合ガード (コミット 97445cc, 2026-07-30 追加)。True にすると
        NON-STABLE→STABLE 復帰 merge の色→空 遷移について HSV が色を保持する
        cell を消さない (列デッドロックの初発を停止、実測は
        scripts/_diag_column_deadlock_trace_2026-07-30.py 参照)。ただし 4動画測定で
        c58/c26 の 2P tail 悪化・c26/c69 の 1P 効果ゼロ、汎化未確認のため
        load_default 既定 OFF。既定 False = 従来挙動不変 (後方互換、A/B比較用)。
    layout: "overlay"(既定、従来通り盤面に直接バー等を重ねるレイアウト)または
        "panel"(2026-08-10 user指示。左上に映像、左下にタイムライングラフ、
        右に縦長情報パネルを配置する新レイアウト、出力キャンバスは1920x1080)。
        既定 "overlay" = 従来挙動不変 (backwards compat)。認識・有利不利の
        計算経路は layout に関わらず完全に同一で、最終合成のみ分岐する。
    """
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"未知の layout: {layout!r} (有効値: {VALID_LAYOUTS})")
    platt_params: PlattCalibrationParams | None = None
    if enable_platt_calibration:
        platt_params = load_platt_calibration(PLATT_CALIBRATION_PATH, required=True)
    model = _train_model(exclude_video)
    _draw_recog_cells = _draw_recog_state = None
    _vr_rois: tuple[int, int, int, int] | None = None
    _vr_roi_size: tuple[int, int] | None = None
    if show_recognition:
        from scripts.visualize_recognition import (
            draw_cell_overlay as _draw_recog_cells,
            draw_state_label as _draw_recog_state,
            P1_ROI_X, P1_ROI_Y, P2_ROI_X, P2_ROI_Y, ROI_W, ROI_H,
        )
        _vr_rois = (P1_ROI_X, P1_ROI_Y, P2_ROI_X, P2_ROI_Y)
        _vr_roi_size = (ROI_W, ROI_H)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] open失敗: {video}", file=sys.stderr)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    proc_frame = int(max(0.0, start_sec - warmup_sec) * fps)  # 処理開始 (ウォームアップ込み)
    write_frame = int(start_sec * fps)                        # 書き出し開始 (ゲーム頭)
    if end_sec > 0:
        n = min(n, int(end_sec * fps))
    elif max_sec > 0:
        n = min(n, write_frame + int(max_sec * fps))
    start_frame = proc_frame
    if proc_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, proc_frame)
        print(f"[seek] 処理開始 {proc_frame / fps:.1f}s / 書き出し開始 "
              f"{write_frame / fps:.1f}s (ウォームアップ {warmup_sec:.0f}s)")
    out.parent.mkdir(parents=True, exist_ok=True)
    # layout="panel" は出力キャンバスサイズが異なる (1920x1080)。認識・有利不利の
    # 計算経路 (OUT_W/OUT_H で処理するフレーム) は layout に関わらず不変。
    canvas_size = ((PANEL_CANVAS_W, PANEL_CANVAS_H) if layout == "panel"
                   else (OUT_W, CANVAS_H))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, canvas_size)
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=force_in_match,
        # 未指定 (None) はライブラリ既定に解決する = 本番と同じ挙動を描画する
        enable_landing_observed_color=_resolve_flag(
            "enable_landing_observed_color", enable_landing_observed_color),
        enable_drift_resync_match_start_guard=_resolve_flag(
            "enable_drift_resync_match_start_guard", enable_drift_guards),
        enable_drift_resync_hsv_gate=_resolve_flag(
            "enable_drift_resync_hsv_gate", enable_drift_guards),
        enable_match_start_full_clear=_resolve_flag(
            "enable_match_start_full_clear", enable_match_start_full_clear),
        enable_recovery_counter_carryover=_resolve_flag(
            "enable_recovery_counter_carryover", enable_recovery_counter_carryover),
        enable_cnn_flicker_hsv_fallback=_resolve_flag(
            "enable_cnn_flicker_hsv_fallback", enable_cnn_flicker_hsv_fallback),
        enable_initial_confirm_vote=_resolve_flag(
            "enable_initial_confirm_vote", enable_initial_confirm_vote),
        enable_puyo_to_empty_hsv_guard=_resolve_flag(
            "enable_puyo_to_empty_hsv_guard", enable_puyo_to_empty_hsv_guard))
    import re
    m = re.search(r"(v\d+|video_\d+)", video.name)
    if m and hasattr(pipe, "set_video_id"):
        pipe.set_video_id(m.group(1))
    tracker = OjamaAccountingTracker(); tracker.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    ps1 = ps2 = BoardState.MENU
    b1 = b2 = None
    adv_ema = 0.0
    p1_last = 0.5
    drivers: list[tuple[str, float]] = []
    # 受けやすさ・飽和連鎖量キャッシュ初期値 (HeavyAdvCache が更新するまで 0.0)
    ukey1: float = 0.0
    ukey2: float = 0.0
    sat1: float = 0.0
    sat2: float = 0.0
    # 打ち合い応手確率 (panel レイアウトの応手情報行用。counter-reach 無効時は
    # nan のままで counter_text が空文字になる、_score_advantage 系と同じ
    # 「使われる時だけ意味を持つ」設計)。
    counter_p1: float = float("nan")
    counter_p2: float = float("nan")
    ptracker = PressureTracker()
    fctracker = RealtimeForecastTracker()
    svtracker = ScoreLeadTracker()
    # 能力低下ベースの圧力 (2026-08-09)。 enable_capability_pressure=True の
    # ときだけ使う。 リセットは _fresh_trackers と同じタイミングで行う。
    cap_ptracker = CapabilityPressureTracker()
    # 打ち合い応手確率 (2026-08-09 user採用)
    counter_tracker = CounterReachTracker()
    hcache = HeavyAdvCache(model)
    efire_tracker = EarlyFireTracker()  # (早期発火) 既定 OFF 時も生成のみ(コスト僅少)
    prev_score1: int | None = None  # (改修1) スコアリセット検知用の前フレーム値
    prev_score2: int | None = None
    history: list[tuple[float, float]] = []  # (ゲーム開始からの秒, 有利不利) 累積
    total_dur = max(1.0, (n / fps) - start_sec)  # グラフ横軸の総尺
    step = max(1, int(round(sample_interval * fps)))
    written = 0
    for fi in range(start_frame, n):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        # show_recognition=True 時のみネイティブ解像度のコピーを保持
        # (認識色 overlay 描画用。推論には使わないため計算経路は不変)。
        raw_native = frame.copy() if show_recognition else None
        if frame.shape[:2] != (OUT_H, OUT_W):
            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        t = fi / fps
        # お邪魔会計は密な駆動が必須のため pipe.update / _drive_ojama は毎フレーム。
        r = pipe.update(fi, t, frame)
        # (改修1) 試合境界(score大幅減少/両者0付近)を検知したら凍結盤面・持続
        # トラッカー・表示状態を全て初期化する(前試合の「幻の差」持ち越し防止)。
        if _detect_score_reset(r.p1.score, r.p2.score, prev_score1, prev_score2):
            print(f"[reset] t={t:.1f}s score大幅減少/0付近を検知 -> 評価を互角にリセット")
            b1 = b2 = None
            adv_ema, p1_last, drivers = 0.0, 0.5, []
            ukey1 = ukey2 = sat1 = sat2 = 0.0
            counter_p1 = counter_p2 = float("nan")
            history.clear()
            (tracker, tp1, tp2, ptracker, fctracker, svtracker, hcache,
             efire_tracker) = _fresh_trackers(model)
            cap_ptracker = CapabilityPressureTracker()
            counter_tracker = CounterReachTracker()
        prev_score1, prev_score2 = r.p1.score, r.p2.score
        if r.p1.state == BoardState.STABLE and r.p1.confirmed_board is not None:
            b1 = r.p1.confirmed_board
        if r.p2.state == BoardState.STABLE and r.p2.confirmed_board is not None:
            b2 = r.p2.confirmed_board
        snap = _drive_ojama(tracker, r.p1, r.p2, ps1, ps2, t,
                            tracker_p1=tp1, tracker_p2=tp2, pipeline=pipe)
        ps1, ps2 = r.p1.state, r.p2.state
        # (早期発火) chain_event 検知フレームで即座に速報バイアスを更新する。
        # settled ゲートの外側 (= 非STABLE中も毎フレーム) で呼ぶことが本修正の要
        # (2026-07-29 userレビュー指摘1/2対処、詳細は EarlyFireTracker docstring)。
        if enable_early_fire_reaction:
            efire_tracker.update(r.p1.chain_event, r.p2.chain_event, b2, b1,
                                 tracker._elapsed(t))
        # (改修2) 両者STABLE(=連鎖終了+お邪魔会計済み)の瞬間のみ有利不利を再計算。
        # 連鎖中/非STABLE中(どちらかが未着地)は前回確定した adv_ema/p1_last/drivers
        # を保持する(着弾前に生値で乱高下させない)。お邪魔会計自体は上の
        # _drive_ojama で毎フレーム密に駆動済みのため、ここで止めても会計は失われない。
        # (2026-08-08) 片側独立更新モード。
        # 従来は「両者同時 STABLE」でしか再計算せず、 実測で **試合時間の 72.3%、
        # 最長 13.97 秒** 評価が凍結していた (scripts/_measure_settled_freeze_
        # 2026-08-08.py)。 片方が連鎖中・おじゃま落下中だと両者とも止まるため、
        # 1P が撃ち切って空・2P がおじゃまで埋まって窒息寸前でも「互角 54%」の
        # まま動かない (user 指摘)。
        # b1/b2 は上で **片側ずつ** STABLE 時に更新される凍結盤面なので、
        # 片側でも STABLE なら「最新の凍結盤面同士」で再計算できる。 生値では
        # ないため、 元の設計意図 (連鎖中に生値で乱高下させない) は保たれる。
        # 既定 False = 従来挙動と完全一致 (backwards compat)。
        if enable_per_side_settled:
            settled = (
                r.p1.state == BoardState.STABLE
                or r.p2.state == BoardState.STABLE
            )
        else:
            settled = (
                r.p1.state == BoardState.STABLE
                and r.p2.state == BoardState.STABLE
            )
        if b1 is not None and b2 is not None and settled:
            # 重い盤面由来(モデルadv/threat/ukeyasusa/飽和連鎖)はキャッシュ間引き、安価な圧力/リードは毎フレーム
            model_adv, threat, drivers, ukey1, ukey2, sat1, sat2 = hcache.update(
                b1, b2, snap, r.p1, r.p2, tracker._elapsed(t))
            # 圧力の測り方 (2026-08-09 user伝授)。
            # 従来: 相手のおじゃま「個数」の増加を累積 → 個数は現象であって
            #       盤面がどれだけ壊れたかを表さない (同じ10個でも土台を割られた
            #       場合と上に乗っただけでは損害が違う)。
            # 新  : 相手の盤面「能力」(飽和連鎖量 + 連結) の低下量を累積する。
            if disable_pressure:
                # 圧力成分そのものを外す (2026-08-09 user要望)。
                # 圧力は「攻撃を通した履歴」だが、 その効果は既に相手の盤面
                # (おじゃま数・連結・飽和連鎖量) としてモデルが見ている。
                # 本当に独立した情報を足しているのかを確かめるための版。
                pres = 0.0
            elif enable_capability_pressure:
                pres = cap_ptracker.update(b1, b2)
            else:
                pres = ptracker.update(iv.board_ojama_count(b1).raw,
                                       iv.board_ojama_count(b2).raw)
            fc = fctracker.update(r.p1.score, r.p2.score,
                                  pipe.tsumo_count("1P"), pipe.tsumo_count("2P"))  # (M3改B)配送予告
            # (b) 得点タイブレーク。
            # **2026-08-09 user伝授: スコア差そのものに意味はない**。
            # スコアはおじゃまを送るための手段であり、 送った時点で意味を失う
            # (送ったぶんは相手の予告おじゃま/盤面おじゃまとして既に観測できる
            #  ので、 スコアでも数えると二重計上になる)。 しかもスコアは累積
            # なので一度差がつくと残り続け、 連鎖中は伸び続けるため
            # 「同時に連鎖しているのに一方へ寄り続ける」症状の原因になる。
            # 意味があるのは「おじゃまに変換されていない繰越 (落下ボーナス・
            # 全消しボーナスのスタック)」だけ。
            # 評価は予告おじゃま + フィールド状況で行うのが正しい。
            if disable_score_lead_bias:
                sl_bias = 0.0
            else:
                sl_bias = max(-SL_BIAS_CAP, min(SL_BIAS_CAP,
                                                svtracker.update(r.p1.score, r.p2.score)))
            # 打ち合い応手確率 (2026-08-09 user採用)。
            # 相手が返せない攻撃を持っている側を有利にする。
            # 着弾までの時間予算を、 **そのときの連鎖数から** 出す。
            # 固定値ではなく局面依存 (2026-08-09 user指摘)。
            # estimate_chain_anim_duration_sec は 23 動画 418 イベントの実測
            # ベース (0.4 秒/連鎖)。
            _cc = 0
            for _sr in (r.p1, r.p2):
                _ev = getattr(_sr, "chain_event", None)
                _n = getattr(_ev, "chain_count", None) if _ev is not None else None
                if _n:
                    _cc = max(_cc, int(_n))
            _budget = iv.estimate_chain_anim_duration_sec(float(_cc)) if _cc else 0.0
            counter_adv, counter_p1, counter_p2 = (
                counter_tracker.update(
                    b1, b2, _budget,
                    next1=getattr(r.p1, "next_pair", None),
                    next2=getattr(r.p2, "next_pair", None),
                ) if enable_counter_reach
                else (0.0, float("nan"), float("nan"))
            )
            adv = (W_PRESSURE * pres + W_FORECAST * fc
                   + W_MODEL * model_adv + W_THREAT * threat
                   + W_COUNTER * counter_adv) + sl_bias
            adv = max(-100.0, min(100.0, adv))
            adv = kill_override(adv, fctracker.inc1, fctracker.inc2,  # (B)キル判定で生存側へ
                                board_room(b1), board_room(b2))
            p1 = adv_to_winprob(adv)  # 表示用勝率(較正sigmoid or 直線)
            adv, p1 = _apply_platt_to_display(adv, p1, platt_params)  # Platt後段校正
            adv_ema = EMA_ALPHA * adv + (1 - EMA_ALPHA) * adv_ema
            p1_last = EMA_ALPHA * p1 + (1 - EMA_ALPHA) * p1_last
            if enable_early_fire_reaction:
                efire_tracker.on_settled()  # 確定計算が入ったので速報バイアスをクリア
        # (早期発火) 表示直前にのみ bias を加算する (adv_ema/p1_last の EMA 内部状態
        # 自体には混ぜない = 無効時は従来経路とビット一致)。
        disp_adv, disp_p1 = adv_ema, p1_last
        if enable_early_fire_reaction and efire_tracker.bias != 0.0:
            disp_adv = max(-100.0, min(100.0, adv_ema + efire_tracker.bias))
            disp_p1 = adv_to_winprob(disp_adv)
        if fi >= write_frame and fi % step == 0 and b1 is not None and b2 is not None:
            # settled=False の間も直近確定値(保持中)を同値追記 → グラフは平坦を維持
            history.append((t - start_sec, disp_adv))
        if fi < write_frame:
            continue  # ウォームアップ区間は書き出さない
        waiting = b1 is None or b2 is None
        display_frame = frame
        if show_recognition:
            # 認識色 overlay はネイティブ解像度 (1920x1080) で描いてから縮小する
            # (visualize_recognition.py の ROI 定数をそのまま使うため。
            # STABLE 凍結済みの b1/b2 = 評価と同条件の盤面を描画)。
            if raw_native.shape[:2] != (NATIVE_H, NATIVE_W):
                raw_native = cv2.resize(
                    raw_native, (NATIVE_W, NATIVE_H), interpolation=cv2.INTER_AREA)
            p1_x, p1_y, p2_x, p2_y = _vr_rois
            # 反復10 viz拡張: CHAIN/GRAVITY_SETTLE 中は estimated_board (物理
            # 推論スルー盤面) を描画する。無ければ従来通り凍結済み b1/b2。
            disp_b1, low_conf_1 = _pick_recog_display_board(r.p1, b1)
            disp_b2, low_conf_2 = _pick_recog_display_board(r.p2, b2)
            if disp_b1 is not None:
                _draw_recog_cells(raw_native, disp_b1, p1_x, p1_y)
            if disp_b2 is not None:
                _draw_recog_cells(raw_native, disp_b2, p2_x, p2_y)
            _draw_recog_state(raw_native, r.p1.state, p1_x, p1_y,
                              score=r.p1.score or 0, label_prefix="1P:")
            _draw_recog_state(raw_native, r.p2.state, p2_x, p2_y,
                              score=r.p2.score or 0, label_prefix="2P:")
            # board_provenance=="chain_estimate_low_confidence" の側は
            # ROI に橙色の枠を描いて低信頼であることを視覚化する。
            if _vr_roi_size is not None:
                roi_w, roi_h = _vr_roi_size
                if low_conf_1:
                    cv2.rectangle(raw_native, (p1_x, p1_y),
                                  (p1_x + roi_w, p1_y + roi_h), (0, 165, 255), 4)
                if low_conf_2:
                    cv2.rectangle(raw_native, (p2_x, p2_y),
                                  (p2_x + roi_w, p2_y + roi_h), (0, 165, 255), 4)
            display_frame = cv2.resize(
                raw_native, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        if layout == "panel":
            frame_out = _draw_panel_layout(
                display_frame, disp_adv, disp_p1, drivers, waiting,
                history, t - start_sec, total_dur,
                state1=r.p1.state.name, state2=r.p2.state.name,
                counter_text=_build_counter_text(counter_p1, counter_p2),
                elapsed_sec=t - start_sec)
        else:
            frame_out = _draw_overlay(display_frame, disp_adv, disp_p1, drivers, waiting,
                                      history, t - start_sec, total_dur,
                                      ukey1=ukey1, ukey2=ukey2, sat1=sat1, sat2=sat2)
        writer.write(frame_out)
        written += 1
        if written % 300 == 0:
            print(f"  ... {written} frames (t={t:.1f}s adv={disp_adv:+.0f})")
    cap.release(); writer.release()
    print(f"[done] {written} frames -> {out}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/frames/video_124_4min.mp4")
    ap.add_argument("--out", default="data/indicators_v2/overlay/advantage_v124.mp4")
    ap.add_argument("--max-sec", type=float, default=0.0)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=0.0)
    ap.add_argument("--warmup-sec", type=float, default=0.0,
                    help="start_sec の何秒前から処理だけ始めるか (状態機械初期化用)")
    ap.add_argument("--exclude-video", default=None,
                    help="学習から除外する動画ID (対象動画のリーク防止)")
    ap.add_argument("--sample-interval", type=float, default=0.15)
    ap.add_argument(
        "--show-recognition", action="store_true", dest="show_recognition",
        help="scripts/visualize_recognition.py の認識色 overlay (盤面セル色記号+"
             "state枠) を合成する (2026-07-23 追加、既定 False = 従来通り無地)。",
    )
    ap.add_argument(
        "--landing-observed-color", action=argparse.BooleanOptionalAction, default=None,
        dest="enable_landing_observed_color",
        help="真因 A 対処: 着地セルの CNN==HSV 一致色補正を有効化 "
             "(RecognitionPipeline.load_default に転送、 2026-07-25 レビュー動画#49で追加)。 "
             "デフォルト OFF = 従来挙動不変 (backwards compat)。",
    )
    ap.add_argument(
        "--no-force-in-match", action="store_false", default=True,
        dest="force_in_match",
        help="RecognitionPipeline.load_default の force_in_match を False にする "
             "(試合境界を跨ぐフル試合レンダ専用。既定 True = 従来挙動不変。 "
             "2026-07-25 c34 game1 境界レビューで追加、詳細は generate() docstring)。",
    )
    ap.add_argument(
        "--drift-guards", action=argparse.BooleanOptionalAction, default=None,
        dest="enable_drift_guards",
        help="DriftDetector再同期暴走ガード2種(開始15秒保護窓+HSV較正3色未満抑制)を"
             "有効化 (2026-07-25 レビュー動画v3で追加)。既定 OFF = 従来挙動不変。",
    )
    ap.add_argument(
        "--match-start-full-clear", action=argparse.BooleanOptionalAction, default=None,
        dest="enable_match_start_full_clear",
        help="前試合盤面残骸リーク修正(幽霊B対策)を有効化 "
             "(RecognitionPipeline.load_default に転送、2026-07-23 追加)。"
             "デフォルト OFF = 従来挙動不変 (backwards compat)。",
    )
    ap.add_argument(
        "--recovery-counter-carryover", action=argparse.BooleanOptionalAction, default=None,
        dest="enable_recovery_counter_carryover",
        help="#51: 復旧カウンタ carryover (非STABLE滞在が短時間なら"
             "stable_recovery_counters/recovery_cellsを引き継ぐ) を有効化 "
             "(RecognitionPipeline.load_default に転送、2026-07-26 追加)。"
             "デフォルト OFF = 従来挙動不変 (backwards compat)。",
    )
    ap.add_argument(
        "--cnn-flicker-hsv-fallback", action=argparse.BooleanOptionalAction, default=None,
        dest="enable_cnn_flicker_hsv_fallback",
        help="#51後半: CNN乱高下セル(直近8フレームで出力変化3回以上)を"
             "HSV出力にフォールバックさせる復旧ゲート緩和を有効化 "
             "(RecognitionPipeline.load_default に転送、2026-07-26 追加)。"
             "デフォルト OFF = 従来挙動不変 (backwards compat)。",
    )
    ap.add_argument(
        "--initial-confirm-vote", action=argparse.BooleanOptionalAction, default=None,
        dest="enable_initial_confirm_vote",
        help="初回STABLE確定を直前NON-STABLE滞在中のCNN履歴多数決で構成する"
             "ガードを有効化 (RecognitionPipeline.load_default に転送、"
             "2026-07-27 追加)。デフォルト OFF = 従来挙動不変 (backwards compat)。",
    )
    # CLI 既定は generate() の既定 (enable_platt_calibration=False) と必ず一致させる。
    # 2026-07-29: 関数側を暫定 False に戻した際 (0a0b014) CLI 側の default=True を
    # 直し忘れ、フラグ無指定で呼ぶと校正器ファイル欠損で必ず落ちる不整合が発生した。
    # 校正を使う場合は --platt-calibration を明示する。
    ap.add_argument(
        "--platt-calibration", action="store_true", default=False,
        dest="enable_platt_calibration",
        help="表示用勝率へ Platt scaling 後段校正を適用する (2026-07-29 追加)。"
             "既定 OFF = 従来挙動 (校正なし) 。data/indicators_v2/platt_calibration.json "
             "が必要で、無い場合は動画を読む前に例外になる。A/B比較用。",
    )
    # 旧フラグ名の後方互換: --no-platt-calibration は既定OFFなので実質no-opだが、
    # 既存スクリプトが渡しても落ちないよう受け付ける。
    ap.add_argument(
        "--no-platt-calibration", action="store_false",
        dest="enable_platt_calibration",
        help="(後方互換) 校正を明示的に無効化する。既定が OFF なので通常は不要。",
    )
    ap.add_argument(
        "--early-fire-reaction", action="store_true", default=False,
        dest="enable_early_fire_reaction",
        help="chain_event 検知フレーム(掛け算式表示等)で即座に速報バイアスを"
             "反映する EarlyFireTracker を有効化する (2026-07-29 userレビュー"
             "指摘1/2対処: 12連鎖等の大型本線が settled ゲートで長時間反映されず"
             "連鎖終了時に急変する問題、および相手の返し連鎖アニメ中の見落としに"
             "対処)。既定 OFF = 従来挙動不変 (backwards compat)。A/B比較用。",
    )
    ap.add_argument(
        "--counter-reach", action="store_true", default=False,
        dest="enable_counter_reach",
        help="打ち合い応手確率 (モンテカルロ) を有利不利に加える (2026-08-09 "
             "user採用)。相手が閾値以上を返せる確率を見て、返せない攻撃を"
             "持っている側を有利にする。既定は無効 (後方互換)。",
    )
    ap.add_argument(
        "--no-pressure", action="store_true", default=False,
        dest="disable_pressure",
        help="圧力成分を完全に外す (2026-08-09)。圧力は攻撃の履歴だが、その効果は"
             "既に相手の盤面としてモデルが見ているため、独立した情報を足して"
             "いるのか検証するための版。既定は外さない。",
    )
    ap.add_argument(
        "--capability-pressure", action="store_true", default=False,
        dest="enable_capability_pressure",
        help="圧力を「おじゃま個数の増加」でなく「相手の盤面能力(飽和連鎖量+連結)"
             "の低下量」で測る (2026-08-09 user伝授)。既定は従来の個数ベース。",
    )
    ap.add_argument(
        "--no-score-lead-bias", action="store_true", default=False,
        dest="disable_score_lead_bias",
        help="得点タイブレークを無効化する (2026-08-09)。スコア差はおじゃまを"
             "送る手段の中間値にすぎず、送ったぶんは予告/盤面で既に観測できる"
             "ため二重計上になる。既定は無効化しない (後方互換)。",
    )
    ap.add_argument(
        "--per-side-settled", action="store_true", default=False,
        dest="enable_per_side_settled",
        help="片側でも STABLE なら有利不利を再計算する (2026-08-08)。"
             "従来の両者同時 STABLE ゲートは試合時間の 72.3%% を凍結させていた。"
             "既定は無効 (後方互換)。",
    )
    ap.add_argument(
        "--puyo-to-empty-hsv-guard", action=argparse.BooleanOptionalAction, default=None,
        dest="enable_puyo_to_empty_hsv_guard",
        help="色→空 HSV 照合ガードを有効化 (RecognitionPipeline.load_default に転送、"
             "コミット 97445cc, 2026-07-30 追加)。c34 型の列デッドロックには有効だが"
             "c58/c26 の 2P で tail 悪化・c26/c69 の 1P で効果ゼロ (汎化未確認)。"
             "デフォルト OFF = 従来挙動不変 (backwards compat)。A/B比較用。",
    )
    ap.add_argument(
        "--layout", choices=VALID_LAYOUTS, default="overlay", dest="layout",
        help="出力レイアウト (2026-08-10 user指示追加)。'overlay'(既定)は従来通り"
             "盤面に直接バー等を重ねる。'panel' は左上に映像・左下にタイムライン"
             "グラフ・右に縦長情報パネルを配置する新レイアウト (1920x1080)。",
    )
    a = ap.parse_args()
    generate(Path(a.video), Path(a.out), a.max_sec, a.sample_interval,
             start_sec=a.start_sec, end_sec=a.end_sec,
             exclude_video=a.exclude_video, warmup_sec=a.warmup_sec,
             show_recognition=a.show_recognition,
             enable_landing_observed_color=a.enable_landing_observed_color,
             force_in_match=a.force_in_match,
             enable_drift_guards=a.enable_drift_guards,
             enable_match_start_full_clear=a.enable_match_start_full_clear,
             enable_recovery_counter_carryover=a.enable_recovery_counter_carryover,
             enable_cnn_flicker_hsv_fallback=a.enable_cnn_flicker_hsv_fallback,
             enable_initial_confirm_vote=a.enable_initial_confirm_vote,
             enable_platt_calibration=a.enable_platt_calibration,
             enable_early_fire_reaction=a.enable_early_fire_reaction,
             enable_per_side_settled=a.enable_per_side_settled,
             disable_score_lead_bias=a.disable_score_lead_bias,
             enable_capability_pressure=a.enable_capability_pressure,
             disable_pressure=a.disable_pressure,
             enable_counter_reach=a.enable_counter_reach,
             enable_puyo_to_empty_hsv_guard=a.enable_puyo_to_empty_hsv_guard,
             layout=a.layout)


if __name__ == "__main__":
    main()
