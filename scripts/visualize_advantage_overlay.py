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
import functools
import inspect
import json
import math
import sys
import zlib
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.indicators_v2 as iv  # noqa: E402
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_OJAMA, Board  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.chain_count_truth import select_chain_count_high_confidence_band  # noqa: E402
from src.chain_detector import ChainEvent  # noqa: E402
from src.death_confirmation import (  # noqa: E402
    DeathConfirmStats,  # Gate 3R-6 本体: 候補/猶予/確定/解除の母数付きカウンタ (dump専用)
    DeathConfirmTracker,  # Gate 3R-6 本体: 1サイド分の死亡確定状態機械 (既定OFF)
    NEXT_STATIONARY_CONFIRM_SEC,  # ネクスト不動確定閾値の既定値 (user指定の暫定値)
    # 2026-08-26 決着ホールド根治: 新しいツモの落下開始 (*→TSUMO_FALL) は
    # 「ネクストが動いた」ことの物理的な証拠。色ペア値の比較と違い、次ツモが
    # 同じ色ペアでも取りこぼさない。既存の純関数をそのまま再利用する。
    is_new_tsumo_fall_start,
    # 2026-08-25 第3版 (Codex 承認条件対応): 両側 pending の ambiguous 判定を
    # 含む境界処理の単一実装 (二重実装防止)。
    resolve_boundary_confirmations,
)
from src.exchange_episode_tracker import (  # noqa: E402
    ChainEventObservation,
    GenerationObservation,
    GrossCounterDeltaClassification,
    classify_gross_counter_delta,  # Gate 3R-5: gross 累積カウンタ差分の会計分類 (dump 専用)
)
from src.exchange_ledger import LedgerSnapshot, PhysicalContext, Side  # noqa: E402
from src.live_exchange_episode_tracker import (  # noqa: E402
    LiveEpisodeSnapshot,
    LiveExchangeEpisodeTracker,
)
from src.exchange_virtual_board import (  # noqa: E402
    land_pending_ojama_onto_board,
    resolve_mutual_exchange,
)
from src.fps_normalize import resolve_normalize_fps_30_stride  # noqa: E402
from src.ojama_accounting import (  # noqa: E402
    CHAIN_TOTAL_MIN_SCORE,  # #9 決着先読みの発火ノイズガードに流用 (ResolvedExchangeTracker)
    GrossOjamaCounters,  # Gate 3R-5: cap 前 gross 累積カウンタ (dump 専用、既定 OFF)
    OjamaAccountingTracker, OjamaAccountSnapshot,
    PENDING_ABS_CAP,  # ホールド延長の安全弁 (RESOLVED_HOLD_LANDING_MAX_WAIT_SEC) 算出用
    SCORE_RESET_THRESHOLD,  # 試合境界(score大幅減少)検知の既存定数を流用
    THEORY_DROP_PER_TURN,  # 同上 (1ターンの最大落下量)
)
from src.probability_calibration import (  # noqa: E402
    PhaseCalibrationParams, PlattCalibrationParams, apply_platt_calibration,
    load_phase_platt_calibration, load_platt_calibration,
    phase_label_for_progress,
)
from src.production_config import (  # noqa: E402
    ATTRIBUTION_EXCLUDED_INDICATORS,
    COUNTER_REACH_ENABLED_BY_DEFAULT,
    GHOST_CHAIN_RULE_ENABLED,  # kill_override 連鎖完走後是正のフォールバック simulate 用
    OVERLAY_NORMALIZE_FPS_30_ENABLED_BY_DEFAULT,
    OVERLAY_PRODUCTION_RECOGNITION_ENABLED_BY_DEFAULT,
    OVERLAY_RESIZE_1080P_ENABLED_BY_DEFAULT,
    recognition_load_default_kwargs,
    reorg_removed_indicator_names,
)
from src.scoring import calculate_chain_score  # noqa: E402
from src.recognition_pipeline import (  # noqa: E402
    PipelineResult,
    RecognitionPipeline,
    # 連鎖の終わりの絶対律のうち「ネクストが動いた」判定 (stateless)。
    # 2026-08-26: 決着ホールドの解除条件を絶対律へ差し替えるにあたり、
    # 同じ判定を書き直さず本番稼働中の実装をそのまま再利用する
    # (memory feedback_check_existing_before_building_2026-08-21)。
    _is_game_event_chain_exit,
)
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama  # noqa: E402
import scripts.mc_counter_estimator as mc_counter  # noqa: E402
from scripts.model_indicator_win import (  # noqa: E402
    GBC_PARAMS, load_labeled_csv, pair_sides_for_win, build_features,
)
# 評価済みモデル成果物 (2026-08-14 coordinator指示) の47列スキーマは
# build_labeled_win_from_npz.py の b-2 分類定数が単一情報源。ここで列名の
# 分類表を再定義せず import して再利用する (重複定義によるドリフト防止、
# CLAUDE.md「マジックナンバー禁止・単一情報源」原則)。
from scripts.build_labeled_win_from_npz import (  # noqa: E402
    COLOR_OJAMA_RATIO_EPS,
    DIFF_KEEP_OWN_HEAVY_COLUMNS,
    DIFF_KEEP_OWN_NEW_COLUMNS,
    DIFF_KEEP_OWN_PAIR_COLUMNS,
    DIFF_REPLACE_OWN_COLUMNS,
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
# 位相別 Platt scaling (2026-08-11 Phase1-2 追加、既存の全位相共通Plattと排他)。
#   scripts.fit_phase_platt_calibration が本ファイルの tier1 モデル
#   (B-1対称化修正+B-2進行度列込み) の OOF 予測から学習した位相別係数。
#   全位相共通 Platt と同じ「適用先モデルの出力分布に対して直接学習した」
#   校正器のため、単一Plattより近似度が高い (詳細は generate() docstring)。
PHASE_CALIBRATION_PATH = Path("data/indicators_v2/phase_platt_calibration.json")


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


def _match_progress_for_boards(b1: Board, b2: Board) -> float:
    """両盤面から表示用の進行度 (match_progress、[0,1]) を計算する (stateless)。

    学習時 (_add_interaction_columns / _match_progress_from_totals) と同じ
    board_puyo_total ベースの定義を使い、RT でも学習時と一貫させる
    (2026-08-11 Phase1-2、位相別 Platt scaling の位相選択に使用)。
    """
    total_1p = iv.board_puyo_total(b1).score
    total_2p = iv.board_puyo_total(b2).score
    return float(_match_progress_from_totals(total_1p, total_2p))


def _resolve_display_platt(
    progress: float,
    platt_params: PlattCalibrationParams | None,
    phase_platt_params: PhaseCalibrationParams | None,
) -> PlattCalibrationParams | None:
    """表示に適用する Platt パラメータを選ぶ (2026-08-11 Phase1-2 追加)。

    位相別校正器 (phase_platt_params) が有効ならそちらを優先し、進行度から
    位相を判定して該当パラメータを返す。位相別が無効 (None) なら従来通り
    全位相共通の platt_params を返す (後方互換、既存経路と完全一致)。
    generate() 側で両フラグの同時指定は禁止しているため、通常は片方のみ
    非 None になる。
    """
    if phase_platt_params is not None:
        label = phase_label_for_progress(
            progress, phase_platt_params.early_bound, phase_platt_params.late_bound,
        )
        return phase_platt_params.phases[label]
    return platt_params


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


# kill_override 連鎖完走後是正 (2026-08-22 修正①) のフォールバック simulate
# 専用の共有インスタンス。幽霊連鎖ルールは本番採用フラグ (production_config.py
# が単一情報源) と同じ設定を使う (他の simulate 経路 exchange_virtual_board.py
# 等と同一の考え方)。フレーム毎の新規生成コストを避けるためモジュール定数化。
_CHAIN_COMPLETION_SIMULATOR = ChainSimulator(
    exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)


def _board_hash(board: "Board | None") -> int:
    """盤面グリッドの決定論的ハッシュ (タイムラインdump用、2026-08-11 追加)。

    組込み `hash()` は文字列/バイト列でハッシュランダム化 (PYTHONHASHSEED) の
    影響を受け実行のたびに値が変わるため使わない。dump はある1回の生成プロセスの
    出力を後で別プロセス (走査器) が読むため、決定論的である必要がある。
    None (未確定盤面) は 0 を返す。
    """
    if board is None:
        return 0
    return int(zlib.crc32(board._grid.tobytes()))


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


def _chain_event_gen_ojama(ev: "ChainEvent", elapsed_sec: float) -> float:
    """1つの ChainEvent が生成するお邪魔換算量を求める (既存資産のみ再利用)。

    `ev.total_score` が信頼できる値 (baseline 実測、または W7
    `enable_pseudo_chain_score_fill` 充填済み推定、`>= CHAIN_TOTAL_MIN_SCORE`)
    ならそのまま使う (`ResolvedExchangeTracker._resolve()` と同一の使い方、
    追加 simulate なし)。信頼できない場合 (formula/landing 経路 +
    `enable_pseudo_chain_score_fill=False`、現行本番既定、`total_score=0`
    固定になる W7 既知の弱点) のみ、`ev.before_board` を自前で simulate し
    `calculate_chain_score` (`_fill_pseudo_chain_score` と同一関数) で得点を
    確定するフォールバックに落ちる。simulate は最悪1回のみ。

    既知の近似 (黙って握りつぶさず明記する): フォールバック simulate は
    全消し持ち越しボーナス (`ALL_CLEAR_BONUS`、`VideoChainTracker`/
    `RecognitionPipeline` 内部の外部状態でボード単体からは復元不能) を
    含まない。大型連鎖の判定への影響は軽微と判断し許容する。
    """
    if ev.total_score >= CHAIN_TOTAL_MIN_SCORE:
        return float(iv._score_to_ojama_count(float(ev.total_score), elapsed_sec))
    fallback = _CHAIN_COMPLETION_SIMULATOR.simulate(ev.before_board)
    if fallback.chain_count < 1:
        return 0.0  # 起点盤面に連鎖が実在しない (ノイズ疑い)
    fallback_score = calculate_chain_score(fallback).total_score
    return float(iv._score_to_ojama_count(float(fallback_score), elapsed_sec))


class ChainGenerationAccumulator:
    """連鎖中の自分の生成量 (お邪魔換算) を、複数回に分かれる chain_event
    トリガーにまたがって累積する (2026-08-22 改良②、根治①の実測で確認した
    根本原因への対処)。

    【実測で確認した問題】t=6717.5 (logs/_diag_t6717_kill_override_root_
    cause_2026-08-22) で計装したところ、1P の連鎖は t=6708.1 (CHAIN 突入)
    から t=6718.3 (STABLE 復帰) まで約10.2秒続き、その間に盤面の空きは
    5→67 まで回復する本物の大型連鎖だった。しかし単発の `ChainEvent` だけを
    見ると t=6713.5 時点で観測できるのは chain_count=5・生成量84個の
    イベント1個だけで、連鎖全体のごく一部にすぎなかった (是正① 単体では
    pending 216→102 までしか下がらず、kill_override の閾値 KILL_RATIO_FULL
    =1.5 を依然大きく上回り続けて誤爆が解消しなかった)。

    原因: `_apply_chain_formula_early_fire` (formula 機構) は「既に
    アクティブな疑似 ChainEvent があれば新規発火しない」設計のため、長い
    連鎖はホールド期限 (`chain_hold_base_sec + chain_hold_per_step_sec ×
    chain_count`) が切れるたびに「その時点で残っている分」だけを対象にした
    **新しい**疑似イベントに置き換わる。単発の ChainEvent だけを見ると
    連鎖全体の生成量を大幅に過小評価する。`EarlyFireTracker`
    (2026-07-29 指摘1/2) が全く同じ理由で「trigger_sec の変化を見て加算」
    を採用しているのと同じ対処をここでも行う (新しい発想ではなく既存の
    確立されたパターンの横展開)。

    生成量は `trigger_sec` が変わるたびに加算 (二重計上防止)。busy状態
    (CHAIN/GRAVITY_SETTLE) を離れたら (=真の連鎖終了、以後は本物の確定
    盤面で判定できる) 累積をリセットする。完走後盤面 (room 算出用) は
    直近の `chain_event.before_board` を使う (連鎖生理上、最新の途中経過
    盤面を完走までシミュレートすれば最終結果に収束するため、履歴を
    保持する必要はない — 生成量の累積とは異なり冪等)。

    [2026-08-22 user指摘・対症療法の実測欠陥] 累積 (`accumulate=True`) は
    「まだ画面に見えていない残り連鎖ぶんまで既に生成し終えた」という架空の
    完了状態を仮定するため、raw モデル (実際の盤面画素を見ている) との
    間に新しい不一致時間帯を作る実測結果が出た (t=6717.5: 元の2.03秒の
    符号不一致は解消したが、直前に6.63秒の新しい不一致が発生、
    logs/_diag_kill_override_chain_completion_v2 系列の全編再走査で確認)。
    根治 (CHAIN 保持時間の実測較正 `chain_hold_base_sec`/
    `chain_hold_per_step_sec` 配線) が入れば断片化そのものが減るため、
    `accumulate=False` (既定) では「直近1件の chain_event が示す生成量を
    そのまま使う」(=trigger_sec が変わるたびに **加算せず置き換える**)
    保守的な動作にする。掛け算式表示等で connectivity が失われ本当に
    複数フラグメントに分裂してしまう残存ケースの保険として `accumulate=True`
    を明示指定すれば残せる (対症療法として完全に削除はしない、
    ただし根治との併用時は二重計上にならないよう `accumulate=False` を
    既定にした)。
    """

    def __init__(self, accumulate: bool = False) -> None:
        self._accumulate = bool(accumulate)
        self._accum_gen: dict[str, float] = {"1p": 0.0, "2p": 0.0}
        self._last_trigger: dict[str, "float | None"] = {"1p": None, "2p": None}
        self._latest_before_board: dict[str, "Board | None"] = {"1p": None, "2p": None}

    def _update_side(
        self, key: str, r_side, elapsed_sec: float,
    ) -> "tuple[float, Board | None]":
        busy = getattr(r_side, "state", None) in _LIVE_DEFENDER_BUSY_STATES
        if not busy:
            self._accum_gen[key] = 0.0
            self._last_trigger[key] = None
            self._latest_before_board[key] = None
            return 0.0, None
        ev = getattr(r_side, "chain_event", None)
        if ev is not None and ev.chain_count >= 1:
            self._latest_before_board[key] = ev.before_board
            if ev.trigger_sec != self._last_trigger[key]:
                self._last_trigger[key] = ev.trigger_sec
                gen = _chain_event_gen_ojama(ev, elapsed_sec)
                if self._accumulate:
                    self._accum_gen[key] += gen
                else:
                    # [対症療法回避、2026-08-22] 架空の完了状態を仮定せず、
                    # 直近1件の chain_event が示す生成量だけを使う (置き換え)。
                    self._accum_gen[key] = gen
        return self._accum_gen[key], self._latest_before_board[key]

    def update(
        self, r_p1, r_p2, elapsed_sec: float,
    ) -> "tuple[float, Board | None, float, Board | None]":
        """毎フレーム呼ぶ (settled 有無に関わらず、EarlyFireTracker と同じ理由:
        settled ゲートの外側でも trigger_sec の変化を取りこぼさないため)。

        Returns:
            (1P累積生成量, 1Pの直近before_board, 2P累積生成量, 2Pの直近before_board)。
            非busy中はそれぞれ (0.0, None)。
        """
        gen1, board1 = self._update_side("1p", r_p1, elapsed_sec)
        gen2, board2 = self._update_side("2p", r_p2, elapsed_sec)
        return gen1, board1, gen2, board2


def _kill_override_chain_completion_inputs(
    snap: OjamaAccountSnapshot,
    b1: "Board", b2: "Board", room1: int, room2: int,
    gen1: float, before1: "Board | None",
    gen2: float, before2: "Board | None",
    pending_p1_override: "float | None" = None,
    pending_p2_override: "float | None" = None,
) -> "tuple[int, int, float, float]":
    """`kill_override` への入力を「連鎖完走後」に是正する (2026-08-22 修正①)。

    背景 (根治対象、6日越しの user 判断待ち事項): 2026-08-03 user決定
    「修正B: 連鎖中の仮想盤面評価」— 発火前の凍結盤面をそのまま致死判定に
    使うと「消費中の連鎖を保有火力として二重計上」する。2026-08-13 指摘#9で
    その完成形として `resolve_mutual_exchange` (両者決着の完走シミュレーション
    + 相殺 + 1ターン上限着弾) を実装したが、起動条件が
    `ev1 is not None and ev2 is not None` (両者同時発火) のため、**片側だけ
    連鎖している場面 (実際の誤爆7件全てがこれ) ではクラス全体が起動せず**、
    無条件で毎フレーム呼ばれる per-frame `kill_override` (`:4764` 付近) には
    この保護が一切配線されていなかった。

    生成量・完走前盤面は呼出側 (`ChainGenerationAccumulator.update()`) が
    複数トリガーにまたがって解決済みのものを渡す設計に変更した (2026-08-22
    改良②、`ChainGenerationAccumulator` docstring の実測根拠参照)。
    本関数自体は「解決済みの (生成量, 起点盤面) を `resolve_mutual_exchange`
    に渡して完走後の room/pending を得るだけ」の薄い純関数に単純化した。

    発火していない側 (`before1`/`before2` が None、または `gen<=0`) は
    「現在の確定盤面 + 生成量0」として渡す。両側とも生成量0ならこの完走
    シミュレーション自体を呼ばず入力をそのまま返す (=無関係フレームでは
    bit-identical、余計な simulate コストも払わない)。

    Returns:
        (room1_有効値, room2_有効値, pending1_有効値, pending2_有効値)。
        どちらの側も発火していなければ (room1, room2, pending_p1, pending_p2)
        をそのまま返す。

    pending_p1_override/pending_p2_override (2026-08-24 A案「規模の比較」、
    optional 引数のみ追加 = backwards compat): None (既定) なら従来通り
    snap.pending_p1/p2 (PENDING_ABS_CAP=216 で丸め済みの額面) を使う。
    値が渡された場合はそれを相殺の基礎 pending として使う。呼出側は
    「cap 前の実額 (snap.pending_p1/p2_uncapped) + 未登録送付分
    (PostChainUnregisteredSentTracker)」を渡す設計 (根因①②の同時是正)。
    """
    base_pending1 = (
        float(snap.pending_p1) if pending_p1_override is None
        else float(pending_p1_override))
    base_pending2 = (
        float(snap.pending_p2) if pending_p2_override is None
        else float(pending_p2_override))
    if gen1 <= 0.0 and gen2 <= 0.0:
        return room1, room2, base_pending1, base_pending2
    before1_final = before1 if before1 is not None else b1
    before2_final = before2 if before2 is not None else b2
    result = resolve_mutual_exchange(
        before1_final, before2_final, int(round(gen1)), int(round(gen2)),
        int(round(base_pending1)), int(round(base_pending2)))
    return (
        board_room(result.board_p1_after), board_room(result.board_p2_after),
        float(result.leftover_p1), float(result.leftover_p2),
    )


# ============================
# B案: 致死上書きの確信度ゲート (2026-08-24、±100即断の構造的禁止)
# 根因: memory project_pm100_display_flip_2026-08-24。per-frame kill_override
# の g=1 完全上書きが ChainEvent 断片化等の観測ノイズ1フレームで ±100→∓100
# へ反転し、EMA (τ=0.116秒) は段差入力を実質素通しする。表示の安定は平滑化
# では作れないため、上書き側で「確信度は持続時間で獲得する」構造にする。
# 定数は全て物理量からの導出 (シーン逆算禁止、feedback_overfitting_awareness)。
# ============================

# 設置→盤面反映の受け入れ基準 8フレーム (memory feedback_placement_reflection_
# 8frames_2026-07-25) を実効30fps (--normalize-fps-30 済み) の秒に換算した認識
# 反映遅延。持続確認の時間窓に認識側の遅れぶんのマージンとして加える。
PLACEMENT_REFLECTION_LATENCY_SEC: float = 8.0 / 30.0

# 方向反転のクールダウン = 1手の平均設置時間 (0.348秒、24.6万件実測の
# PLACEMENT_SPEED_BY_ROW_SEC 単純平均 = mc_counter.BEAM_ROLLOUT_AVG_STEP_TIME_SEC)。
# 物理根拠: おじゃまは受け側の設置でしか降らず (reference_ojama_landing_gated_
# by_placement)、新しい連鎖の発火にも設置が要る。1手も置けない時間内に致死の
# 「向き」が実世界で入れ替わることはないため、その間の逆向き上書きは観測ノイズ
# (ChainEvent 断片化 = 根因③) として保留する。
KILL_FLIP_COOLDOWN_SEC: float = mc_counter.BEAM_ROLLOUT_AVG_STEP_TIME_SEC

# 完全上書き (±100) を許すまでの同方向持続時間。
# 物理根拠: 受け側が反証 (撃ち返しの発火) に即座に使える持ち手は「現在手 +
# ネクスト」の2手 (画面に見えており思考済みで置ける分)。その設置時間
# 2×0.348秒 の間、致死判定が同方向のまま反証が観測されなければ確定とみなす。
# 認識の設置→盤面反映遅延 (上記 8f≈0.267秒) をマージンとして加算 → ≈0.963秒。
KILL_CONFIRM_PERSIST_SEC: float = (
    2.0 * mc_counter.BEAM_ROLLOUT_AVG_STEP_TIME_SEC + PLACEMENT_REFLECTION_LATENCY_SEC)

# 持続確認前の上書き上限 (絶対値)。根拠: 生モデルが全編で実際に出す上限帯は
# p99=91〜96 / max 99.4 (logs/_diag_zenchi_pm100_stats_2026-08-24.log 実測)。
# 未確定の致死推定がモデル自身の出力域を超えて断定する (99%/1% を名乗る) のを
# 禁じ、その下側の 90 に置く。方向情報は失わない (±90 ≈ 勝率9割台の強い有利)。
KILL_UNCONFIRMED_ABS_CAP: float = 90.0
# episode の「未解決」は勝率99%相当まで出してよいという意味ではない。
# 既存の評価→勝率較正を逆変換し、勝率90%/10%を専用の表示上限にする。
# B案の時間ヒステリシス用 KILL_UNCONFIRMED_ABS_CAP は互換維持のため不変。
EPISODE_UNRESOLVED_WINPROB_CAP: float = 0.90
EPISODE_UNRESOLVED_ABS_CAP: float = _winprob_to_adv(
    EPISODE_UNRESOLVED_WINPROB_CAP)


class KillOverrideConfidenceGate:
    """per-frame `kill_override` の出力に確信度制御を掛ける (2026-08-24 B案)。

    3つの規則 (すべて時間ベース、settled フレームの粗密に依存しない):
      1. 方向反転時クールダウン: 前回と逆方向の致死上書きは、反転検知から
         KILL_FLIP_COOLDOWN_SEC (1手時間) の間は適用を保留し、上書き前の
         ブレンド値 (adv_pre) をそのまま返す。1フレームだけの反転
         (根因③ ChainEvent 断片化) はこれで構造的に表示へ出なくなる。
      2. 持続による確定: 同一方向の上書きが KILL_CONFIRM_PERSIST_SEC 続いて
         初めて完全上書き (±100 到達) を許す。
      3. 未確定中の上限: 確定前は |adv| ≤ KILL_UNCONFIRMED_ABS_CAP に丸める
         (adv_pre 自身が既にそれを超えている場合は adv_pre を下限に採用し、
         安全弁がモデルより弱い方向へ働かないようにする)。

    stateless 実装原則の例外 (状態は外部 wrapper とする CLAUDE.md 規約に従い、
    kill_override 純関数自体は変更せず、状態を持つ本ゲートを外側に置く)。
    試合境界では呼出側 (generate ループ) が本オブジェクトを作り直す。
    """

    def __init__(self) -> None:
        self._direction: int = 0            # +1=2P致死(adv増側) / -1=1P致死 / 0=なし
        self._dir_since_sec: float = 0.0    # 現在方向の開始時刻
        self._last_fired_sec: "float | None" = None
        self._cooldown_until_sec: float = float("-inf")

    def apply(self, adv_pre: float, adv_post: float, t_sec: float) -> float:
        """kill_override 適用前後の値からゲート済みの adv を返す。

        adv_pre: kill_override 適用前の4成分ブレンド値。
        adv_post: kill_override 適用後の値 (未発火なら adv_pre と同一)。
        """
        fired = adv_post != adv_pre
        if not fired:
            # 発火が1手時間を超えて途切れた = 判定根拠 (pending/room比) が
            # 非致死へ戻った。方向の記憶を破棄し、次の発火は新規事象として扱う。
            if (self._last_fired_sec is not None
                    and t_sec - self._last_fired_sec > KILL_FLIP_COOLDOWN_SEC):
                self._direction = 0
                self._last_fired_sec = None
            return adv_post
        direction = 1 if adv_post > adv_pre else -1
        if direction != self._direction:
            reversal = self._direction != 0
            self._direction = direction
            self._dir_since_sec = t_sec
            if reversal:
                self._cooldown_until_sec = t_sec + KILL_FLIP_COOLDOWN_SEC
        self._last_fired_sec = t_sec
        if t_sec < self._cooldown_until_sec:
            return adv_pre  # 反転直後 (1手未満) は上書きを保留
        if t_sec - self._dir_since_sec >= KILL_CONFIRM_PERSIST_SEC:
            return adv_post  # 同方向の持続で確定 → 完全上書きを許可
        lower = min(adv_pre, -KILL_UNCONFIRMED_ABS_CAP)
        upper = max(adv_pre, KILL_UNCONFIRMED_ABS_CAP)
        return float(min(max(adv_post, lower), upper))


# ============================
# A案(i-a): 連鎖完走後の未登録送付分トラッカー (2026-08-24「規模の比較」)
# ============================

# 未登録送付分の保持期限。導出: 会計 finalize の実測最大遅延 11.5秒
# (t=186.0 の1P連鎖完走 → t=197.53 の pending 登録、memory project_pm100_
# display_flip_2026-08-24 根因①) + score settle 待ち (K_SETTLE_FRAMES=20f
# ≈0.67秒) + 連鎖合体窓 (CHAIN_COALESCE_WINDOW_SEC=2.5秒) ≈ 14.7秒 を包含する
# 安全側の丸め。期限までに会計登録 (pending増加) にも盤面着弾 (おじゃまセル
# 増加) にも現れない送付分は、生成量推定の誤observationとみなして破棄する。
UNREGISTERED_SENT_EXPIRE_SEC: float = 20.0


class PostChainUnregisteredSentTracker:
    """「送付済みだが会計未登録」のおじゃまを連鎖完走時に即時捕捉する (A案(i-a))。

    根因① (memory project_pm100_display_flip_2026-08-24): お邪魔会計の
    finalize は score settle 待ちを要するため、撃ち合いでは相手側の pending
    登録が連鎖完走から最大11.5秒遅れる。その間に相手が撃ち返すと、送付済みの
    実弾 (例: 1Pの517個) が相殺の計算から丸ごと欠落し、撃ち返し (720個) の
    全量が新規攻撃として kill_override に渡って方向が反転する。

    本トラッカーは各サイドの busy (CHAIN/GRAVITY_SETTLE) → 非busy 遷移を
    連鎖完走とみなし (reference_chain_end_absolute_signals と同じ観測面)、
    ChainGenerationAccumulator の生成量推定を「宛先側への未登録 pending」
    として保持する。保持分は次のいずれかで減額/消滅する (全て実観測駆動):
      - 会計登録: 宛先の snap.pending_*_uncapped が増えた分だけ減額
        (会計が追い付いた)。
      - 盤面着弾: 宛先の凍結盤面のおじゃまセルが増えた分だけ減額
        (会計を経ずに降った)。
      - 相互相殺: 宛先側が自分の連鎖を完走したら、その生成量でまず消す
        (ゲーム内の相殺と同じ向き)。
      - 期限切れ: UNREGISTERED_SENT_EXPIRE_SEC。
    減額は二重に効く場合があるが、常に「供給不足側」(=従来挙動に近づく側)
    に倒れる保守設計 (架空の攻撃を作らない)。kill_override の入力専用で、
    会計本体・表示値には一切書き込まない。
    """

    def __init__(self) -> None:
        self._prev_busy: dict[str, bool] = {"1p": False, "2p": False}
        self._last_gen: dict[str, float] = {"1p": 0.0, "2p": 0.0}
        self._held: dict[str, float] = {"1p": 0.0, "2p": 0.0}  # key=送り主
        self._held_since: dict[str, float] = {"1p": 0.0, "2p": 0.0}
        self._prev_pending_unc: dict[str, "float | None"] = {"1p": None, "2p": None}
        self._prev_board_ojama: dict[str, "int | None"] = {"1p": None, "2p": None}

    @staticmethod
    def _ojama_cells(board: "Board | None") -> "int | None":
        """可視領域 (row1-12、board_room と同じ範囲) のおじゃまセル数。"""
        if board is None:
            return None
        return int(np.count_nonzero(board._grid[1:] == COLOR_OJAMA))

    def _absorb_dest_observations(
        self, dest_key: str, pending_unc_now: float, board: "Board | None",
    ) -> None:
        """宛先側の観測 (会計登録/盤面着弾) で送り主の保持分を減額する。"""
        sender = "2p" if dest_key == "1p" else "1p"
        prev_pend = self._prev_pending_unc[dest_key]
        if prev_pend is not None and pending_unc_now > prev_pend:
            self._held[sender] = max(
                0.0, self._held[sender] - (pending_unc_now - prev_pend))
        self._prev_pending_unc[dest_key] = pending_unc_now
        cells = self._ojama_cells(board)
        prev_cells = self._prev_board_ojama[dest_key]
        if cells is not None:
            if prev_cells is not None and cells > prev_cells:
                self._held[sender] = max(
                    0.0, self._held[sender] - float(cells - prev_cells))
            self._prev_board_ojama[dest_key] = cells

    def _track_completion(
        self, key: str, r_side, gen: float, own_pending_unc: float, t_sec: float,
    ) -> None:
        """busy→非busy 遷移 (連鎖完走) で未登録送付分を捕捉する。"""
        busy = getattr(r_side, "state", None) in _LIVE_DEFENDER_BUSY_STATES
        if busy:
            # 断片化 (根因③) で gen が瞬間的に落ちても、同一連鎖内の最大値は
            # 「少なくともこれだけは生成した」の下限として保持できる。
            self._last_gen[key] = max(self._last_gen[key], float(gen))
            self._prev_busy[key] = True
            return
        if self._prev_busy[key]:
            sent = self._last_gen[key]
            other = "2p" if key == "1p" else "1p"
            # (1) 自分宛てに飛翔中の相手の未登録分をまず相殺 (ゲーム内と同じ向き)
            consumed = min(self._held[other], sent)
            self._held[other] -= consumed
            sent -= consumed
            # (2) 自分宛ての会計 pending は実会計が finalize 時に自ら相殺する
            #     ため、ここでは保持へ登録する量から差し引くだけ (二重相殺防止)。
            sent = max(0.0, sent - own_pending_unc)
            if sent > 0.0:
                self._held[key] += sent
                self._held_since[key] = t_sec
        self._last_gen[key] = 0.0
        self._prev_busy[key] = False

    def update(
        self, r_p1, r_p2, snap: OjamaAccountSnapshot,
        b1: "Board | None", b2: "Board | None",
        gen1: float, gen2: float, t_sec: float,
    ) -> tuple[float, float]:
        """毎フレーム呼ぶ (settled 有無に関わらず、完走遷移と登録差分を逃さない)。

        Returns:
            (1P宛て未登録分 [=2Pが送った], 2P宛て未登録分 [=1Pが送った])。
        """
        for key in ("1p", "2p"):
            if (self._held[key] > 0.0
                    and t_sec - self._held_since[key] > UNREGISTERED_SENT_EXPIRE_SEC):
                self._held[key] = 0.0
        self._absorb_dest_observations("1p", float(snap.pending_p1_uncapped), b1)
        self._absorb_dest_observations("2p", float(snap.pending_p2_uncapped), b2)
        self._track_completion("1p", r_p1, gen1, float(snap.pending_p1_uncapped), t_sec)
        self._track_completion("2p", r_p2, gen2, float(snap.pending_p2_uncapped), t_sec)
        return self._held["2p"], self._held["1p"]


# 主因表示に使う合成キー (2026-08-22 修正④)。JP_LABEL に対応する日本語文言を
# 登録する (下記 JP_LABEL 定義参照)。実指標ではなく表示専用の合成キーのため
# ATTRIBUTION_EXCLUDED_INDICATORS 等の指標系リストには含めない。
KILL_OVERRIDE_DRIVER_KEY_P1: str = "kill_override_p1_lethal"
KILL_OVERRIDE_DRIVER_KEY_P2: str = "kill_override_p2_lethal"


def _kill_override_attribution_entry(
    adv_before: float, adv_after: float,
    pending1: float, pending2: float, room1: int, room2: int,
) -> "tuple[str, float]":
    """`kill_override` 発火時の主因表示エントリを作る (2026-08-22 修正④)。

    従来は安全弁 (`kill_override`) が結論を上書きしても、主因欄には安全弁
    適用前の生モデル寄与度だけが並び、「主因は1P有利の根拠なのに結論は
    2P有利」という自己矛盾した表示になっていた (2026-08-22 実測: t=6717.5
    で主因1位「連結対数差差 +8.00」= 1P有利根拠、結論は2P有利99%)。
    本関数は予測値には一切関与せず、表示する (キー, 値) を1件返すだけ。

    致死判定された側 (adv が悪化した側) を、`adv_before` と `adv_after` の
    大小関係から特定する (`kill_override` 内部の `lead>0 → target=-100`
    という規約と同じ向き: adv が下がった = 1P が致死判定された)。
    値には「死ぬと判定された側が受ける相殺後 pending」を渡す
    (画面には既存の `f"{JP_LABEL[c]}差 {v:+.2f}"` 形式でそのまま乗る)。
    """
    if adv_after < adv_before:
        return KILL_OVERRIDE_DRIVER_KEY_P1, float(pending1)
    return KILL_OVERRIDE_DRIVER_KEY_P2, float(pending2)


def _drivers_for_display(
    drivers: list[tuple[str, float]],
    kill_override_note: "tuple[str, float] | None",
) -> list[tuple[str, float]]:
    """表示直前に安全弁の理由を主因先頭へ差し込む (2026-08-22 修正④)。

    dump 用の `drivers` (raw モデル寄与度、自己無矛盾性が要件) は一切変更
    せず、新しいリストを返すだけ。`kill_override_note` が None (安全弁
    未発火、またはフラグ既定 False) の場合は `drivers` をそのまま返す
    (bit-identical)。
    """
    if kill_override_note is None:
        return drivers
    _override_keys = (KILL_OVERRIDE_DRIVER_KEY_P1, KILL_OVERRIDE_DRIVER_KEY_P2)
    rest = [d for d in drivers if d[0] not in _override_keys]
    return [kill_override_note] + rest[:2]


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
        # [2026-08-22 修正②] 直近に確認した OjamaAccountSnapshot.
        # chain_total_score_p1/p2 (finalize 検知用、finalized_since_last_check
        # 参照)。None = 未確認 (初回呼び出しでは「変化あり」と誤判定しない)。
        self._prev_chain_total_p1: int | None = None
        self._prev_chain_total_p2: int | None = None

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

    def on_settled(self, finalized: bool = True) -> None:
        """settled(確定)再計算が入ったら速報バイアスをクリアする(二重計上防止)。

        finalized: [2026-08-22 修正② 追加、optional] 既定 True = 従来通り
            「settled 再計算が走ったら常にクリア」(bit-identical)。
            `--per-side-settled` 下では相手が STABLE の間 settled 再計算が
            毎フレーム走るため、大連鎖の速報バイアスが連鎖の finalize
            (会計反映) より前に毎フレーム消えてしまう不具合があった。
            呼出側が `finalized_since_last_check()` 等で「今回の settled
            再計算で実際に finalize が起きたか」を判定し False を渡すと
            クリアをスキップして bias を維持できる。
        """
        if finalized:
            self.bias = 0.0

    def finalized_since_last_check(
        self, chain_total_p1: int, chain_total_p2: int,
    ) -> bool:
        """直近の確認以降に連鎖の finalize が会計に反映されたかを判定する。

        [2026-08-22 修正②] `OjamaAccountSnapshot.chain_total_score_p1/p2`
        (docstring「最後の連鎖合計得点(検証用)」) は
        `OjamaAccountingTracker._finalize_chain_end` の中でのみ更新される
        (ojama_accounting.py:691 `s.last_chain_total_score = chain_total`)。
        よって前回確認時からの値の変化を検知すれば、その間に少なくとも
        1回 finalize が実行され pending の相殺・繰越が会計に反映された
        ことが分かる (新しい会計ロジックは作らず既存フィールドの読み取りの
        みで判定する)。呼出側が settled 再計算のたびに呼ぶことを想定
        (`update()`/`on_settled()` と同じ呼び出し頻度)。

        初回呼び出し (内部状態が未確認 None) は必ず False を返す
        (=「変化なし」の安全側、finalize 済みかどうか判断材料が無いため
        誤ってクリアしない)。
        """
        prev1, prev2 = self._prev_chain_total_p1, self._prev_chain_total_p2
        self._prev_chain_total_p1 = chain_total_p1
        self._prev_chain_total_p2 = chain_total_p2
        if prev1 is None or prev2 is None:
            return False
        return chain_total_p1 != prev1 or chain_total_p2 != prev2


# ============================
# れんさ数表示 (--show-chain-count、2026-08-15 user要望、既定 OFF)
# ============================
# user要望: 「得点よりれんさ数の方が重要指標。実際の連鎖数と評価がどう動いたか、
# どちらの盤面での連鎖か明確に。認識性能検証としても使えるように」。
#
# 設計方針 (推定/実測の両論併記、単一の断定値を出さない):
#   - 推定連鎖数 = `ChainEvent.chain_count`。ただしこれは常に simulate 由来
#     (chain_detector.py `_try_emit_event` の `sim.chain_count`) であり、
#     真値8連鎖→1と過小評価する壊滅例が実測済み
#     (project_chain_count_both_untrustworthy_2026-07-30)。「推定」と明示する。
#   - 実測得点差 = `OjamaAccountSnapshot.chain_total_score_p1/p2`
#     (`chain_end_triggered_p1/p2` が立った瞬間のみ真)。これは score OCR の
#     連鎖前後差分そのもの (src/ojama_accounting.py `_finalize_chain_end`:
#     `chain_total = score_after - score_start`) であり、`ChainEvent.total_score`
#     (これも simulate 由来、chain_detector.py:282,285) とは別物・こちらが
#     真の観測値。
#   - 得点逆算連鎖数 = 実測得点差を `select_chain_count_high_confidence_band`
#     (src/chain_count_truth.py) に通した高信頼帯判定 (カバレッジ19.1%、
#     判定不能なら None=「-」表示。単独では真値扱いしない)。
#   - 推定 (simulate) と逆算 (得点) が食い違ったら、それ自体が
#     「認識性能検証」の価値そのもの (どちらか、あるいは両方が誤っている
#     証拠) → 表示側で目立たせる (色を変える)。
CHAIN_DISPLAY_HOLD_SEC: float = 4.0  # 連鎖情報のパルス表示を保持する秒数
# (根拠: EARLY_FIRE_DECAY の半減期11.5秒より短いが、"数字を読んで比較する"
# UI 用途には十分な長さを固定値として確保する。連鎖アニメ自体は最長でも
# 十数秒に収まる実測 (project_chain_count_both_untrustworthy) との比較で
# 極端に短くも長くもない値)。


@dataclass(frozen=True)
class ChainCountDisplayInfo:
    """`ChainCountDisplayTracker.snapshot` が返す片側 (1P/2P) 分の表示情報。

    いずれのフィールドも None なら「該当データなし・保持期限切れ」を意味し、
    呼び出し側は "-" 等のプレースホルダで描画する。
    """

    estimated_chain_count: int | None      # 推定連鎖数 (simulate 由来)
    actual_score: int | None               # 実測得点差 (score OCR 由来)
    derived_chain_count: int | None        # 得点逆算連鎖数 (高信頼帯、不明なら None)


class ChainCountDisplayTracker:
    """認識性能検証用の「推定連鎖数 vs 実測得点差」を側ごとに保持する (表示専用)。

    stateless な `select_chain_count_high_confidence_band` を毎フレーム
    再利用するだけで、本クラス自体は「パルスで一瞬しか立たない値を
    一定時間読める形で保持する」という外部 wrapper の責務のみを持つ
    (CLAUDE.md 「観測指標は stateless 実装、state-holding は外部 wrapper」)。
    adv/winprob の計算には一切関与しない (表示専用、コアロジックへの
    副作用ゼロ)。
    """

    def __init__(self) -> None:
        self._est_p1: int | None = None
        self._est_p1_until: float = -1.0
        self._est_p2: int | None = None
        self._est_p2_until: float = -1.0
        self._score_p1: int | None = None
        self._derived_p1: int | None = None
        self._score_p1_until: float = -1.0
        self._score_p2: int | None = None
        self._derived_p2: int | None = None
        self._score_p2_until: float = -1.0

    def update(
        self, ev1: "ChainEvent | None", ev2: "ChainEvent | None",
        snap: OjamaAccountSnapshot, t_sec: float,
    ) -> None:
        """毎フレーム呼ぶ (render 有無・settled 有無に関わらず、パルス取りこぼし防止)。"""
        if ev1 is not None:
            self._est_p1 = ev1.chain_count
            self._est_p1_until = t_sec + CHAIN_DISPLAY_HOLD_SEC
        if ev2 is not None:
            self._est_p2 = ev2.chain_count
            self._est_p2_until = t_sec + CHAIN_DISPLAY_HOLD_SEC
        if snap.chain_end_triggered_p1:
            self._score_p1 = snap.chain_total_score_p1
            self._derived_p1 = select_chain_count_high_confidence_band(
                snap.chain_total_score_p1).chain_count
            self._score_p1_until = t_sec + CHAIN_DISPLAY_HOLD_SEC
        if snap.chain_end_triggered_p2:
            self._score_p2 = snap.chain_total_score_p2
            self._derived_p2 = select_chain_count_high_confidence_band(
                snap.chain_total_score_p2).chain_count
            self._score_p2_until = t_sec + CHAIN_DISPLAY_HOLD_SEC

    def snapshot(self, side: str, t_sec: float) -> "ChainCountDisplayInfo | None":
        """side ("1P"/"2P") の現在の表示情報を返す。両方とも保持期限切れなら None。"""
        if side == "1P":
            est = self._est_p1 if t_sec <= self._est_p1_until else None
            score = self._score_p1 if t_sec <= self._score_p1_until else None
            derived = self._derived_p1 if t_sec <= self._score_p1_until else None
        else:
            est = self._est_p2 if t_sec <= self._est_p2_until else None
            score = self._score_p2 if t_sec <= self._score_p2_until else None
            derived = self._derived_p2 if t_sec <= self._score_p2_until else None
        if est is None and score is None:
            return None
        return ChainCountDisplayInfo(est, score, derived)


def _build_chain_display_text(
    side_label: str, info: "ChainCountDisplayInfo | None",
) -> tuple[str, bool]:
    """1行分の表示文字列と「推定≠逆算」食い違いフラグを組み立てる (純粋関数)。

    info が None (連鎖無し・保持期限切れ) の場合は空文字を返し、呼び出し側
    (`_draw_panel_info`) の optional-if-truthy パターン (`counter_text` と
    同様) で行自体を描かない。
    """
    if info is None:
        return "", False
    est_s = f"{info.estimated_chain_count}連鎖" if info.estimated_chain_count is not None else "-"
    score_s = f"{info.actual_score:+d}点" if info.actual_score is not None else "-"
    derived_s = f"{info.derived_chain_count}連鎖" if info.derived_chain_count is not None else "-"
    mismatch = (
        info.estimated_chain_count is not None
        and info.derived_chain_count is not None
        and info.estimated_chain_count != info.derived_chain_count
    )
    tag = " [推定≠逆算]" if mismatch else ""
    text = f"{side_label} 推定{est_s} / 実測{score_s} (逆算{derived_s}){tag}"
    return text, mismatch


# ============================
# #9 両者同時発火の決着先読み (2026-08-13、docs/DEMO_REVIEW_2026-08-13.md #9)
# ============================
# --resolved-exchange-eval で有効化 (既定 OFF)。両側の chain_event が同時に
# アクティブになった瞬間、双方の発火直前盤面 (ChainEvent.before_board) から
# 連鎖を完走シミュレーションし決着後の仮想盤面を1回だけ評価する。連鎖終了
# (両側の chain_event が両方 None に戻る) まで結果を保持し、その間は再評価
# しない (「確定済みの未来を逐次再評価しない」という #9 の原理そのもの)。
# 片側のみの発火はトリガー対象外 (EarlyFireTracker の領分のまま)。
#
# [指摘11対処、2026-08-14] 「両側 chain_event が両方 None に戻る」だけでは
# 連鎖アニメ終了の瞬間でしかなく、相殺後おじゃまの**着弾**はまだ完了して
# いない (docs/DEMO_REVIEW_2026-08-13.md #11: 1Pは2連鎖で対応するも5段くらい
# 降る状況で1P有利表示になっていた事象の根因)。ホールドは「着弾完了」まで
# 延長する (ResolvedExchangeTracker._landing_complete 参照)。

# 安全弁 (指摘11): 着弾完了シグナルが何らかの理由で成立しないまま無限に
# ホールドし続けないよう、延長フェーズに上限時間を設ける。シーンからの
# 逆算ではなく物理量のみから導く (feedback_overfitting_awareness_2026-08-04
# 準拠): 予告おじゃまの絶対上限 PENDING_ABS_CAP (=216個、on-field 3面分、
# src.ojama_accounting) を 1ターンの最大落下量 THEORY_DROP_PER_TURN
# (=30個/ターン) で捌き切るのに要するターン数の天井 × 1手あたりの実測秒数
# (iv.SEC_PER_HAND=0.733秒、labeled_win.csv 実測中央値)。
RESOLVED_HOLD_LANDING_MAX_WAIT_SEC: float = (
    math.ceil(PENDING_ABS_CAP / THEORY_DROP_PER_TURN) * iv.SEC_PER_HAND
)

# [2026-08-26] 決着ホールドを「連鎖の終わりの絶対律」で解除するときの確認フレーム数。
# 絶対律の2信号 (ネクストが動いた / お邪魔が落ちた) はどちらも保持セッション開始時の
# スナップショットとの比較なので、いったん成立すれば下がらない (ラッチ性がある)。
# それでも next_pair の単発誤読で早期解除しないよう、連続フレームの確認を挟む。
# 値は既存の「連続2フレーム確認」の作法 (src/score_ocr.py FormulaStepAccumulator の
# confirm_frames) に合わせる。シーンからの逆算ではない
# (memory feedback_overfitting_awareness_2026-08-04)。
RESOLVED_ABS_END_CONFIRM_FRAMES: int = 2
# 発火させた設置による NEXT 移動を「連鎖終了」と誤認しないため、保持開始直後は
# 終了判定の基準値を取らない。RecognitionPipeline が短い疑似 CHAIN を抑えるために
# 実運用している最小表示時間を単一情報源として再利用する。
RESOLVED_ABS_BASELINE_DELAY_SEC: float = RecognitionPipeline.CHAIN_MIN_DISPLAY_SEC
RESOLVED_ABS_BASELINE_CONFIRM_FRAMES: int = 2
# 同一物理連鎖の段間最大1.4秒 (`src/chain_count_ocr.py`) に30fps量子化と
# OCR遅延の余裕0.1秒を足す。この間の40点以上の得点段は物理継続証拠になる。
RESOLVED_ABS_CHAIN_ACTIVITY_QUIET_SEC: float = 1.5
# 同じ物理交換の断片化イベントを再武装しないための中立継続時間。新しい1手を
# 置く実測中央値を単一情報源として使い、瞬間的な None gap を中立とみなさない。
RESOLVED_ABS_REARM_NEUTRAL_SEC: float = iv.SEC_PER_HAND

# [指摘14 案1、2026-08-15] `enable_live_defender_strict=True` 時に
# 「defender_side が今まさに自分の連鎖を処理中」とみなす状態機械 state の集合
# (ResolvedExchangeTracker docstring 指摘14節参照)。BoardState.CHAIN =
# 「連鎖中(消去+重力)」そのもの、BoardState.GRAVITY_SETTLE = その直後の
# 重力settle/着地完了までの継続window (board_state_machine.py の docstring
# 通り「CHAIN 終了後から board が物理的に静止するまでの window」)。
# 実測 (計装ログ logs/_diag_issue14_reeval_calls_2026-08-15.log) で、
# 誤爆は旧連鎖の chain_event hold が切れてから新連鎖の trigger が検出
# されるまでの GRAVITY_SETTLE 区間で発生することを確認済み。
# BoardState.TSUMO_FALL/OJAMA_FALL は意図的に含めない (指摘13が意図した
# 「受け側は連鎖中も置き続ける」正当な自由行動を塞がないため)。
_LIVE_DEFENDER_BUSY_STATES: frozenset["BoardState"] = frozenset({
    BoardState.CHAIN, BoardState.GRAVITY_SETTLE,
})


class ResolvedExchangeTracker:
    """両者同時発火の決着を先読みし、連鎖終了まで固定表示する (#9 対処)。

    決着計算は `resolve_mutual_exchange` (連鎖完走シミュレーション+相殺+着弾、
    src.exchange_virtual_board の既存資産、Step2 `reconstruct_virtual_board_pair`
    の姉妹関数) → `_score_advantage` (既存学習モデル) の2段のみ (pressure/
    threat/counter 等の他ライブ成分は含めない、決着先読みの対象はあくまで
    「モデルが見た仮想盤面ペアの勝率」)。simulate の連結欠損由来の過小評価
    (既知事故: 真値8連鎖→simulate1連鎖、project_chain_count_both_untrustworthy)
    への対策として、保持中に **確定済み** 連鎖合計得点
    (`OjamaAccountSnapshot.chain_total_score_p1/p2`、`chain_end_triggered_p1/p2`
    が立った瞬間のみ真、K_SETTLE_FRAMES 連続不変を確認済みの値) が予測総得点を
    超えたら、それを下限として即時再決着する。生 score OCR (掛け算式アニメ中は
    上昇し続ける途中値) を直接比較すると、上昇アニメの毎フレームで
    「観測>予測」が成立してしまい絶えず再決着し続ける (=乱高下の再現、
    実測で確認済み) ため、既存の settle 確定済み値のみを見る。

    [指摘11対処、2026-08-14] 両側の chain_event が両方 None に戻っても
    即座に解放せず、相殺後おじゃまの着弾完了 (`_landing_complete`) まで
    保持を延長する。安全弁として `RESOLVED_HOLD_LANDING_MAX_WAIT_SEC` を
    超えたら強制解放する。

    [指摘10対処、2026-08-14] `enable_decisive_amplify=True` の場合、決着値に
    「受け側の応手不能度」を統合する。受け側限定の応手確率を計算し
    (既存 `CounterReachTracker` の受け側限定経路を再利用)、既存
    `_counter_defender_adv` (物理由来の非線形ダメージ関数 `iv.ojama_damage`
    + 応手確率 + 専用定数 `RESOLVED_AMPLIFY_SCALE`) と同一の式で決着値へ
    加算する。既定 False = 従来 (#9 のみ) の決着値と完全に同一
    (backwards compat)。

    [指摘12対処、2026-08-14] `_amplify_decisive` が使う時間予算は
    `_chain_remaining_time_budget_sec` (#3 の経過時間控除 + E[最終|N観測]
    条件付き期待) に一本化した。以前は `iv.estimate_chain_anim_duration_sec`
    (観測連鎖数×0.4秒、経過時間控除なし) を直呼びしており、連鎖の後半ほど
    残り時間を過小評価 → 応手0%と誤断 → 決定度増幅が全量発動、という
    二重の誤りが重なっていた (実演出8.1秒 vs 旧式算出2.4秒、指摘12の実測)。
    増幅強度も専用定数 `RESOLVED_AMPLIFY_SCALE` (= `COUNTER_SCALE * W_COUNTER`)
    に分離し、ライブ per-frame 経路が重み付け後に持つ実効上限と揃えることで
    モデル評価との二重計上を避ける (同定数のコメント参照)。

    [指摘12 修正4、2026-08-14 意味論バグ対処] 応手確率 (`CounterReachTracker`
    への入力盤面) は `resolve_mutual_exchange` の `board_p1_pre_landing`/
    `board_p2_pre_landing` (自分の連鎖は消化済みだが、相殺後の余剰おじゃまは
    まだ配置していない盤面) を使う。時間予算修正 (修正1) 後もなお時間予算13秒
    (mean_hands≈13、手数は十分) にもかかわらず応手0%になる事象が残っていた
    根因がこれで、`board_p1_after`/`board_p2_after` (=着弾**後**、余剰おじゃま
    が既に降り切った盤面) を渡していたため、実際にはまだ空中のおじゃまが
    盤面を埋めた状態から MC を回していた (「おじゃまは連鎖完了後・受け側
    ツモ設置時まで降らない」ルールとの意味論不一致、memory
    reference_ojama_landing_gated_by_placement)。ダメージ計算
    (`_counter_defender_adv` の `iv.ojama_damage`) は「返せなかった場合に
    何が起きるか」を測るものなので、こちらは着弾後盤面のままが正しい
    (両者を混同しないこと、下記実装参照)。

    [指摘13対処、2026-08-15] 従来は「片側だけ連鎖アニメが終わった」瞬間も
    「両側とも連鎖継続中」と同じ完全凍結ブランチに合流していた (両側
    chain_event が両方 None になるまで hold_* を一切動かさない)。これは
    「決着済みの攻撃側の帰結」と「生きている受け側の応手力」を区別せず
    両方凍結する設計不備だった (user指摘: 受け側は連鎖中も置き続けており、
    実際に応手力は変化する)。`enable_live_defender_reeval=True` の場合、
    片側のみ連鎖中 (攻撃側継続・受け側は自由行動) の間、または strict ONで
    両側eventが残っていても受け側の物理stateが自由な間、
    `COUNTER_RECOMPUTE_INTERVAL_SEC` (0.5秒、既存の応手判定周期と同一) ごとに
    以下を再評価する (`_reevaluate_live_defender` 参照):
      - 凍結維持: 攻撃側の連鎖帰結 (`_decisive_defender` が返す飛来量
        incoming、攻撃側の仮想盤面 `board_pX_after`) — 攻撃側の生盤面は
        アニメ中で信用できない (physics_only 原則) ため使わない。
      - 生値で再評価: 受け側の**現在の**盤面 (呼出側 generate() が保持する
        sticky な `b1`/`b2`、片側STABLE時のみ更新される) + 残り時間予算
        (`_chain_remaining_time_budget_sec` に現在の `t_sec` を都度渡すだけで
        経過時間控除が自然に効く)。
    増幅 (`enable_decisive_amplify=True` の場合のみ) とモデル評価
    (`_score_advantage`) の両方をこの更新済み値で再計算し hold_adv/hold_p1に
    反映するため、表示が「攻撃側の帰結起点→受け側の組みに応じて漸移→
    撃ち返しで通常経路に反転」という連続的な挙動になる。
    `enable_live_defender_reeval=False` (既定) では本節は一切実行されず、
    従来 (両側 chain_event が両方 None になるまで完全凍結) と bit-identical
    (backwards compat)。

    [指摘13 方向反転修正、2026-08-15、docs/KNOWN_WEAKNESSES.md W12]
    上記の forecast 全量差し替え (`_live_defender_snap`) だけでは直らない
    方向反転が残っていた (診断: logs/_diag_issue13_direction_flip_2026-08-15.log)。
    根因は学習モデルが forecast 特徴をほぼ無視すること (W12実測:
    着弾前局面の実勝率48.4%=ノーペナルティ学習、重要度26位/47列)。
    受け側の生盤面 (未着弾=クリーン) をそのままモデルへ渡す限り forecast を
    どれだけ差し替えても「降るまでは無傷」判定は解消しない。
    対処 (意味論の統一): モデルへ渡す前に、未着弾分を物理的に着弾させた
    仮想盤面を再構成する (`land_pending_ojama_onto_board`、
    `resolve_mutual_exchange` と同じ着弾原理 + `OJAMA_MAX_DROP_PER_TURN`
    上限を再利用、新規の着弾実装は書かない)。着弾させる量は「凍結時の
    incoming 固定」ではなく `_live_remaining_incoming` が現在の会計
    スナップショット (`snap.pending_pX`、実際に降り進んでいれば減る) から
    都度求める (二重計上防止)。会計が0を示すのに凍結時飛来予測が正の場合
    (baseline reset 等で会計がこの交換を追跡できていない場合) は、盤面上の
    増加分を凍結時飛来量から控除するフォールバックへ切り替える。
    forecast (`_live_defender_snap`) は「このフレームで着弾しきれず残った
    分」(`leftover_now`、`OJAMA_MAX_DROP_PER_TURN` 超過分) だけに揃え、
    凍結経路 `_resolve()` (forecast=leftover) と意味論を一致させる。
    応手確率MC (`CounterReachTracker`) は引き続き着弾**前**の生盤面を使う
    (指摘12 修正4 と同じ「降られる前に撃てるか」の意味論、モデル評価用の
    着弾後仮想盤面とは別物であり混同しないこと、下記実装のコメント参照)。

    [指摘14 案1、2026-08-15、既定OFF、docs/DEMO_REVIEW_2026-08-13.md #14]
    `enable_live_defender_strict=True` の場合、`_reevaluate_live_defender` の
    起動条件を厳格化する。

    【誤爆の実機構(計装で確定、推測でなく実測)】
    実動画 (review_demo_2026-08-12.mp4、絶対t=195.3秒) で
    `scripts/_diag_issue14_reeval_calls_2026-08-15.py` により
    `_reevaluate_live_defender` の全呼び出しを計装したところ、致死退行を
    引き起こした初回呼び出しは ev1_cc=9(1P、継続中)/ev2 が None
    (defender=2P) という XOR 条件下で発生していたが、そのフレームの
    defender(2P)自身の **状態機械 state は `BoardState.GRAVITY_SETTLE`**
    (直前の小連鎖の消去は終わったが重力 settle 中でまだ物理的に静止して
    いない) であり、次の瞬間には2Pがさらに7連鎖という**本物の別の連鎖**を
    新規発火させていた。つまり ev(ChainEvent)は「trigger 検知フレームで
    1度だけ発行され chain_hold_base_sec+chain_hold_per_step_sec×chain_count
    秒だけ保持後 None に戻る」パルス方式 (src/chain_detector.py
    VideoChainTracker._try_emit_event 参照) のため、**旧連鎖の hold が
    切れてから新連鎖の trigger が検出されるまでの短い settle 区間で
    ev が None になる瞬間が生じる**。この区間は defender が真に「自由」
    なのではなく、直前の消去の重力settleの真っ最中 (=すぐ次の本物の連鎖に
    突入する寸前) であり、chain_event の有無だけでは判定できない
    (旧・案1初版はここで defender_ev is not None だけを見ており、この
    settle gap では ev が None のため誤って「自由」と判定し続けてしまい
    A/B実測でも strict が baseline と1桁も違わない=無効化していた)。

    【対処】状態機械の `state` (`r_p1.state`/`r_p2.state`、既存の
    BoardStateMachine 出力、新しい推測ロジックではない) を使う。
    defender_side 自身の state が `_LIVE_DEFENDER_BUSY_STATES`
    (= {BoardState.CHAIN, BoardState.GRAVITY_SETTLE}) に含まれる間は
    「現在まさに連鎖(消去+重力)を処理中」と判定し再評価をスキップして
    直前の保持値を維持する。`BoardState.CHAIN` は毎フレーム直接観測される
    (chain_event のようなパルス+hold-window方式ではない) ため、旧連鎖の
    hold 切れ〜新連鎖 trigger 検知までの gap でも「defender は今まさに
    連鎖中」を取りこぼさない。`BoardState.TSUMO_FALL`(ツモ設置)・
    `BoardState.OJAMA_FALL`(おじゃま着弾) は意図的に busy 扱いしない
    (defender が普通にツモを置いている/着弾中は指摘13が意図した
    「受け側は連鎖中も置き続けており応手力が変化する」正当な自由行動
    そのものであり、ここを塞ぐと指摘13の効果自体が失われる)。
    既定 False = 従来挙動と完全に同一 (backwards compat)。

    [指摘19 根治、2026-08-16、coordinator決定(b)、docs/DEMO_REVIEW_2026-08-13.md #19]
    `enable_kill_override_counter_aware` (状態ゲート方式) は致死上書き
    (`kill_override`) という**安全弁1個だけ**を止める対症療法であり、その
    手前の `hold_adv` 自体の計算 (`_resolve`→`resolve_mutual_exchange` が
    使う gen1/gen2 = 各 side が生成したお邪魔換算値) が非対称に壊れている
    という根本原因は残っていた。

    【実機構(計装で確定、logs/_diag_issue19_root_cause_trace_2026-08-16.log)】
    `_maybe_redecide` は `OjamaAccountSnapshot.chain_end_triggered_pX` が
    True の**最初の1フレーム**だけ `chain_total_score_pX` を読み、以後は
    `_redecidedX` で永久に latch して無視する(docstring 指摘11節参照、
    「2回目以降のsettleは別の連鎖の可能性が高い」という設計判断)。しかし
    `OjamaAccountingTracker` の実装 (src/ojama_accounting.py
    `_finalize_chain_end`) では、`chain_end_triggered_pX` は settle 開始
    (TSUMO_FALL/OJAMA_FALL 遷移) の瞬間に True になり、**同一の連鎖が
    coalesce window 内で複数回に分けて finalize されるたびに
    `chain_total_score_pX` を段階的に上書きしながら True であり続ける**
    (実測: t=5.23で総額0(未確定)→t=5.87で1260→t=8.43で4020、この間ずっと
    `chain_end_triggered_p1=True` 継続)。「1回きり」latch は運悪く**未確定
    (0や小さい途中値) の瞬間に固定してしまい、その後実際に育っていく
    真の確定値を二度と拾わない**。これが victim 側の gen が過小評価される
    直接原因。attacker 側は既に完全終了している (settle が1回で完結する)
    ため latch の悪影響を受けず「即時確定値」のまま正しく動く
    (`_maybe_redecide` は両 side 対称のコードだが、片方だけ症状が出るのは
    settle 過程の非対称さ=「相手は終わっている・自分は今まさに終わりつつ
    ある」という指摘19 前提そのものに起因する)。

    【対処】`enable_resolved_victim_gen_live=True` の場合、`_redecide_obs`
    (下記) で「1回きり」制限を「`chain_end_triggered_pX` が True の間は
    `COUNTER_RECOMPUTE_INTERVAL_SEC` (0.5秒、既存の応手判定周期と同一) ごと
    に再チェックし、より大きい確定値が出るたびに追従する」へ緩和する。
    `_maybe_redecide` の `max(pred, obs)` 合成は変更しない (単調増加のみ
    許容、後退しない)。victim/attacker を明示的に区別する新しい判定は
    追加しない (どちらの side も同じコードを通るが、既に決着済みの
    attacker 側は `chain_total_score_pX` がそれ以上変化しないため
    実質ノーオペ=「攻撃側は従来通り即時確定値のまま」が自然に成り立つ)。
    既定 False = 従来 (`_redecided1/2` による1回きり latch) と完全に同一
    (backwards compat)。

    【実測による正直な報告、2026-08-16、コーダ検証】本フラグは `_maybe_
    redecide` の latch バグそのものは実在し (logs/_diag_issue19_root_cause_
    trace_2026-08-16.log で 0→1260→4020 の段階確定と latch を実証、
    ユニットテストでも再現・修正を確認済み)、この latch バグ単体としては
    正しい修正である。**しかし指摘19 の実受入窓 (review_demo_2026-08-12.mp4
    絶対t=201.2-203.4) の再現には効果が無かった** (logs/_diag_issue19_
    victim_gen_live_ab_2026-08-16.log: フラグ ON でも 0.7% のまま変化なし)。
    詳細計装 (logs/_diag_issue19_pinpoint_mechanism_2026-08-16.log) で判明した
    実際の機構: この窓では `_resolve()` が t_abs=201.23 に**既に正しい**
    hold_adv=5.24 (56%相当) を1回の呼び出しで算出済み (gen1/gen2 とも
    ev.total_score の simulate 値をそのまま使った初回決着で、latch バグに
    ぶつかる前に正解に到達している)。ところが `hold_after_kill_override` が
    `_incoming_total_p1=262` という**この hold_adv 自体が既に織り込み済みの
    値**に対して独自の pending/room 比ヒューリスティックで即座に致死断定し、
    victim(1P) 自身の state が `_LIVE_DEFENDER_BUSY_STATES` (自分の反撃連鎖が
    アニメ中= CHAIN) である間 `enable_kill_override_counter_aware` の
    busy ゲートが「busy=安全弁を弱めない」と判断し続けるため、1P 自身の
    連鎖アニメが終わり state が STABLE に戻る t_abs=203.43 まで 2.2秒間
    -100.00 (0.7%) に固定され続ける。つまり 指摘19 の真因は「victim の
    gen が過小評価される」ことではなく、「**resolve_mutual_exchange が
    既に正しく解決した結果を kill_override の独立ヒューリスティックが
    busy 状態ゲート越しに上書きし続ける**」ことだった (根治対象を誤認、
    本フラグは撤回はしないが 指摘19 の解決フラグとしては不採用)。

    さらに 指摘13 の既存合格窓 (t=234.87-245.5) で非退行確認したところ、
    本フラグ ON 時に t=236.23-243.30 の区間で BASE (2-6%) から FIX
    (最大26.4%) へ一時的に乖離することを検出した (最終収束値 56.3% は
    3構成とも一致、既定 OFF では当然 bit-identical)。ON にした場合の
    他シーンへの波及は未レビューのため、実運用フラグとしての採用は
    現時点で推奨しない (「_redecided1/2 latch バグの根治」としては
    有効、別課題として扱う)。指摘19 自体の次の一手は
    `hold_after_kill_override` 側 (kill_override 発火条件そのものの
    見直し、または「同一 _resolve セッション内で既に確定済みの hold_adv
    は kill_override の busy ゲートより優先する」設計) が本命候補。
    """

    def __init__(
        self, model,
        attribution_exclude: tuple[str, ...] = ATTRIBUTION_EXCLUDED_INDICATORS,
        enable_decisive_amplify: bool = False,
        enable_live_defender_reeval: bool = False,
        enable_live_defender_strict: bool = False,
        enable_pending_landing_gate: bool = False,
        enable_kill_override_counter_aware: bool = False,
        enable_resolved_victim_gen_live: bool = False,
        enable_episode_physical_redecide: bool = False,
        enable_episode_physical_consistency_guard: bool = False,
        episode_physical_stats: "dict | None" = None,
        enable_counter_placement_reuse: bool = False,
        enable_counter_budget_quantize: bool = False,
        enable_absolute_chain_end: bool = False,
    ) -> None:
        self._model = model
        self._attribution_exclude = attribution_exclude
        self._enable_decisive_amplify = enable_decisive_amplify
        # [指摘13、2026-08-15] 片側のみ連鎖中の間の受け側ライブ再評価 (既定OFF、
        # クラス docstring 指摘13節参照)。
        self._enable_live_defender_reeval = enable_live_defender_reeval
        # [指摘14 案1、2026-08-15] 上記ライブ再評価の起動条件厳格化 (既定OFF、
        # クラス docstring 指摘14節参照)。enable_live_defender_reeval=False の
        # 間は本フラグの値に関わらず _reevaluate_live_defender 自体が呼ばれない。
        self._enable_live_defender_strict = enable_live_defender_strict
        # [予告おじゃまの降下条件、2026-08-21 user 仕様伝授] **既定 OFF**。
        #
        # 経緯 (判断の揺れを記録しておく):
        #   1. user から降下の正確な仕様を伝授され、条件を見ずに降らせている
        #      実装のバグを特定 (下記 _reevaluate_live_defender 内のコメント参照)
        #   2. user 判断「一旦は振らせていい」→ 既定 OFF で実装
        #   3. user 指摘「モデルが予告があっても無傷では不利ではない、は
        #      時系列を無視すればあっている (相殺の可能性があるため)」を受け、
        #      物理的に降らせるのは過剰補正だと理解 → 既定 ON に変更
        #   4. **しかし ON にすると指摘13 で直した事象 (受け側が無傷に見えて
        #      有利判定が出る) が戻る恐れがある**。3の指摘は「モデルの学習は
        #      正しい」という理解の訂正であって「降らせる処理をやめる」という
        #      指示ではなかった。user 同意のうえ既定 OFF に戻した (2026-08-21)。
        #
        # 判断を保留する理由: ON/OFF どちらが正しいかは**実際の映像で確認
        # しないと決められない**。30先動画の作成後、指摘13 の検収シーンで
        # 両方の挙動を見比べてから決める。理屈では「連鎖が終われば降る」ので
        # 両立するはずだが、未検証。
        #
        # True では攻撃側が連鎖中の間は降らせず、受け側の生盤面で評価する。
        # 予告は snapshot 側が forecast として保持し続けるので情報は失われない。
        # 実装位置: _reevaluate_live_defender 内の land_pending_ojama_onto_board
        # 呼び出し直前。
        self._enable_pending_landing_gate = enable_pending_landing_gate
        # [指摘19、2026-08-15、状態ゲート方式] hold_after_kill_override の
        # 致死断定を受け側の状態機械 state (_LIVE_DEFENDER_BUSY_STATES) で
        # ゲートする (既定OFF、hold_after_kill_override docstring 参照)。
        # enable_resolved_kill_override=False の間は本フラグの値に関わらず
        # hold_after_kill_override 自体が呼ばれない (孫フラグ)。
        self._enable_kill_override_counter_aware = enable_kill_override_counter_aware
        # [指摘19 根治、2026-08-16] 保持セッション中「1回きり」の再決着 latch
        # を、chain_end_triggered_pX が True の間 0.5秒ごとに追従する方式へ
        # 緩和する (既定OFF、クラス docstring 指摘19根治節参照)。
        self._enable_resolved_victim_gen_live = enable_resolved_victim_gen_live
        # 交換台帳の純残量を、進行中連鎖の物理下限として決着値へ戻す。
        # 既定OFFでは純残量を参照せず、従来経路と同一にする。
        self._enable_episode_physical_redecide = enable_episode_physical_redecide
        # 台帳・直前モデル・直前確定値の3者が同方向の時だけ、未解決中の
        # 単独逆転を保留する。再決着とは独立にA/Bできる既定OFFの安全策。
        self._enable_episode_physical_consistency_guard = (
            enable_episode_physical_consistency_guard)
        # [2026-08-21 user承認・設置ごと量子化] 受け側限定応手MC
        # (CounterReachTracker._update_defender_only) の再計算を「受け側の
        # 盤面bytesが変化した (=設置が起きた) とき」だけに限定する近似
        # (既定OFF、CounterReachTracker クラス docstring 参照)。
        # `self._counter_tracker.update(..., reuse_if_board_unchanged=...)`
        # へそのまま貫通させるだけで、本クラス自身は判定ロジックを持たない。
        self._enable_counter_placement_reuse = enable_counter_placement_reuse
        # [2026-08-21 user承認・残り秒数の量子化、上記とは独立の別機構]
        # `self._counter_tracker.update(..., quantize_budget_sec=...)` へ
        # そのまま貫通させる (CounterReachTracker クラス docstring 参照)。
        # 盤面一致による再利用 (上記) とは独立に効果を測れるよう別フラグ。
        self._enable_counter_budget_quantize = enable_counter_budget_quantize
        # [2026-08-26 決着ホールド根治、user決定] **既定 OFF**。
        # ホールドの解除条件は従来「両側の chain_event が None」だが、ChainEvent は
        # 長い連鎖で 1.4秒ごとに断片化するため打ち合い中は長時間 None にならず、
        # 実測で最大 45.07秒 settled 再計算が止まっていた (ホールドが潰した settled
        # 1880/3964 = 47.43%)。True では user 伝授の絶対律
        # 「連鎖の終わり = 連鎖している側のネクストが動いた瞬間 OR
        #   連鎖している側にお邪魔が落ちた瞬間」
        # (memory reference_chain_end_absolute_signals_2026-08-21) を観測する。
        # NEXT系の終了候補は次段CHAINで撤回し、物理連鎖の段間では解除しない。
        # 安全弁 RESOLVED_HOLD_LANDING_MAX_WAIT_SEC は従来どおり維持する。
        self._enable_absolute_chain_end = enable_absolute_chain_end
        # 絶対律の side 別追跡。index 0=1P / 1=2P。本クラスの他の状態は _ev1/_ev2 の
        # ように数字サフィックスで持つが、ここは 1P/2P で完全に同じ処理を1つの
        # ヘルパーで回したいため2要素リストにする (処理の二重実装を避ける)。
        self._abs_next_at_start: list = [None, None]
        self._abs_dropped_at_start: list = [0.0, 0.0]
        self._abs_ended: list = [False, False]
        self._abs_end_kind: list[str | None] = [None, None]
        self._abs_pending_frames: list = [0, 0]
        # 直前フレームの state 名 (`*→TSUMO_FALL` の立ち上がり検出用)。
        self._abs_prev_state: list = [None, None]
        self._abs_prev_score: list[int | None] = [None, None]
        self._abs_last_chain_activity_sec: list[float | None] = [None, None]
        # TSUMO_FALL の立ち上がりは1フレームだけの**エッジ**なので、そのままでは
        # 連続フレーム確認 (デバウンス) を通れない。セッション内で一度観測したら
        # ラッチして保持する (next 値比較と着弾量は開始時との比較なので元々ラッチ性がある)。
        self._abs_saw_tsumo: list = [False, False]
        # セッション開始時の NEXT は、発火させた設置に伴うスライド途中でありうる。
        # 実際の CHAIN を観測し、開始直後の物理待ちを過ぎた後に初めて基準化する。
        self._abs_arm_t_sec: float = 0.0
        self._abs_saw_chain: list[bool] = [False, False]
        self._abs_baseline_ready: list[bool] = [False, False]
        self._abs_baseline_candidate: list = [None, None]
        self._abs_baseline_frames: list[int] = [0, 0]
        # 絶対律で解放した直後、同じ lingering ChainEvent から別セッションを
        # 作らない。両側のイベントが一度ともに None になるまで再武装を禁止する。
        self._abs_rearm_blocked: bool = False
        self._abs_rearm_neutral_started_sec: float | None = None
        self._abs_legacy_neutral_frames: int = 0
        # どの信号で閉じたかの母数付きカウンタ。0 が「起きなかった」なのか
        # 「測っていない」なのかを取り違えないため、必ず母数と並べて出す
        # (memory feedback_zero_needs_denominator_2026-08-25)。
        self.abs_end_stats: dict = {
            "sessions": 0,          # 保持セッションの総数 (母数)
            "released_by_abs": 0,   # 絶対律で解除したセッション数
            "released_by_legacy": 0,  # 旧条件 (両側 chain_event None) で解除
            "ended_by_boundary": 0,  # 保持中に試合そのものが終了したセッション
            "total_boundaries": 0,   # 観測した正式試合境界 (母数)
            "end_by_next": [0, 0],   # side 別: ネクストの色ペアが変わって終了
            "end_by_slide": [0, 0],  # side 別: NEXT の物理スライドで終了
            "end_by_tsumo": [0, 0],  # side 別: 新ツモの落下開始で終了
            "end_by_ojama": [0, 0],  # side 別: お邪魔着弾で終了
            "reopened_by_chain": [0, 0],  # 終了候補後の次段 CHAIN で撤回
            "rearm_blocked_frames": 0,  # 同一交換の再武装を拒否したフレーム数
            # 実解除の瞬間の観測 (t, state1, state2, 終了候補の根拠)。
            # dump の空白終端ではなく、実際に _release() へ進む箇所で記録する。
            "release_states": [],
            # 終了候補が成立し着弾待ちへ入った瞬間。次段 CHAIN で候補を
            # 撤回した場合も残るため、release_states とは意味を分ける。
            "end_candidate_states": [],
        }
        self._active = False
        self._ev1: "ChainEvent | None" = None
        self._ev2: "ChainEvent | None" = None
        self._pred_score1 = 0.0
        self._pred_score2 = 0.0
        # 各 side につき再決着は1回まで (下記 _maybe_redecide 参照)。
        # enable_resolved_victim_gen_live=True の場合のみ _redecide_obs が
        # この latch を「0.5秒ごとの追従」へ読み替える。
        self._redecided1 = False
        self._redecided2 = False
        # [指摘19 根治] 上記の 0.5秒間引き用の直近再決着時刻 (raw elapsed_sec、
        # COUNTER_RECOMPUTE_INTERVAL_SEC と同じ周期)。新しい保持セッション
        # 開始時 (update() の trigger ブロック) に None へ戻す。
        self._victim_live_last_t1: "float | None" = None
        self._victim_live_last_t2: "float | None" = None
        self._episode_physical_net_last: "float | None" = None
        # 試合境界で tracker 本体を作り直しても、動画全体の母数を失わない共有器。
        # None の場合は単体利用として新規作成するため、既存呼出元は不変。
        self._episode_physical_stats: dict = (
            episode_physical_stats if episode_physical_stats is not None else {
                "redecide": 0, "fallback": 0, "fallback_times": []})
        self.hold_adv = 0.0    # 保持中の決着後有利不利 (1P視点)
        self.hold_p1 = 0.5     # 保持中の決着後1P勝率
        self.hold_drivers: list[tuple[str, float]] = []
        # [2026-08-27 userレビュー 3:50] decisive defender が連鎖中に、反対側が
        # hold開始後に新しく発火した side-local chain の前後だけを追跡する。
        self._nondef_cycle_armed: list[bool] = [False, False]
        self._nondef_cycle_in_chain: list[bool] = [False, False]
        self._nondef_cycle_prev_state: list[BoardState | None] = [None, None]
        self._nondef_cycle_before_board: list[Board | None] = [None, None]
        self._nondef_cycle_score_start: list[int | None] = [None, None]
        self._nondef_cycle_stable_since: list[float | None] = [None, None]
        self._nondef_cycle_event: list[ChainEvent | None] = [None, None]
        self._nondef_cycle_hold_anchor: list[float | None] = [None, None]
        self._nondef_cycle_saw_new_hand: list[bool] = [False, False]
        self._nondef_fallback_active: list[bool] = [False, False]
        self._nondef_fallback_anchor: list[float | None] = [None, None]
        self._nondef_fallback_before: list[Board | None] = [None, None]
        self._nondef_cycle_chain_id: list[int | None] = [None, None]
        self._nondef_applied_event_keys: set[tuple[str, int]] = set()
        self._resolved_root_chain_id: list[int | None] = [None, None]
        self.nondef_cycle_stats: dict[str, int] = {
            "armed": 0, "started": 0, "applied": 0,
            "rejected_score": 0, "rejected_same_board": 0,
            "rejected_late": 0, "rejected_stale_event": 0,
            "rejected_no_new_hand": 0,
            "rejected_sim_mismatch": 0, "rejected_direction": 0,
            "fresh_replaced": 0, "fallback_applied": 0,
        }
        self.nondef_cycle_deltas: list[tuple] = []
        self.nondef_cycle_trace: list[tuple] = []
        # [指摘11] 着弾完了待ちの延長フェーズ管理用 (両側 chain_event が
        # None化した後、着弾完了までの間だけ True)。
        self._awaiting_landing: bool = False
        self._landing_wait_started_sec: "float | None" = None
        self._landing_end_kind: "str | None" = None
        # [指摘11] 決着計算時点で予測した各 side の最終お邪魔到達量 (着弾
        # 完了判定①用)。incoming が 0 なら「元々受け側でない」= 常に達成扱い。
        self._target_ojama_p1: float = 0.0
        self._target_ojama_p2: float = 0.0
        self._incoming_total_p1: float = 0.0
        self._incoming_total_p2: float = 0.0
        # [指摘10] 応手不能度の増幅で使う受け側限定 MC (既存 CounterReachTracker
        # をこのクラス専用にもう1個持つ。ライブ per-frame 用インスタンスとは
        # 独立、決着計算は頻度が低い=1保持につき高々数回のためキャッシュ共有は不要)。
        self._counter_tracker = CounterReachTracker()
        # [指摘12 修正1, 2026-08-14] #3 で実装済みの経過時間控除+条件付き
        # 期待最終連鎖数テーブル (_chain_remaining_time_budget_sec が読む)。
        # enable_decisive_amplify=False の間は使われないため I/O を避ける
        # (既存の `_chain_len_table` ロード方針と同じ、generate() 内コメント参照)。
        self._chain_len_table: "dict[int, float]" = (
            _load_chain_length_conditional_table() if enable_decisive_amplify else {}
        )
        # [指摘12 修正1] `update()` 呼び出し時点の動画内絶対時刻 (raw t、
        # `ChainEvent.trigger_sec` と同じ時間軸)。試合開始からの経過秒
        # `elapsed_sec` とは意味論が異なる (定数オフセット分ずれる) ため
        # 別属性で保持する (_amplify_decisive の時間予算計算専用)。
        self._t_sec: float = 0.0
        # [指摘12 修正3, 2026-08-14] ホールド中のパネル表示用 (受け側限定
        # 応手情報)。判定値 (hold_adv/hold_p1) には一切混ぜず、表示専用の
        # サイドチャネルとして公開する (呼出側 generate() が参照)。
        self.hold_defender_side: "str | None" = None
        self.hold_incoming_ojama: float = 0.0
        self.hold_defender_prob: float = float("nan")
        # [指摘13、2026-08-15] 直近の _resolve() が保持した決着結果 (凍結成分)。
        # 片側のみ連鎖中のライブ再評価 (`_reevaluate_live_defender`) が
        # 攻撃側の帰結 (board_pX_after/pre_landing、飛来量) を読み直すために
        # 保持する (_resolve() 内のローカル変数だった result/resolved_snap を
        # インスタンス属性へ格上げ、他の既存フィールドには影響しない)。
        self._result: "MutualExchangeResult | None" = None
        self._resolved_snap: "OjamaAccountSnapshot | None" = None
        # [指摘13] ライブ再評価の間引き用の直近実行時刻 (raw t_sec、
        # COUNTER_RECOMPUTE_INTERVAL_SEC と同じ 0.5秒周期)。_resolve() の
        # たびに None へ戻し、新しい決着セッション開始直後は間引きせず
        # 即座に1回評価させる (段差回避)。
        self._last_live_reeval_t: "float | None" = None

    def _store_resolved_result(
        self, snap: OjamaAccountSnapshot, result: "MutualExchangeResult",
        *, reset_nondefender_cycles: bool = True,
    ) -> None:
        """決着結果をモデル評価し、ホールド用の状態へ原子的に反映する。"""
        self._update_landing_targets(result)
        # 決着後盤面には着弾分 (leftover 未満) が既に反映済みのため、snap の
        # forecast/net_balance だけを決着後の残り (leftover) に差し替える
        # (他のフィールドは会計連続性のため元 snap のまま流用、dataclasses.replace)。
        resolved_snap = dataclass_replace(
            snap,
            net_balance_capped=result.leftover_p2 - result.leftover_p1,
            forecast_p1=result.leftover_p1, forecast_p2=result.leftover_p2,
        )
        # [指摘13] 片側のみ連鎖中のライブ再評価が読む凍結成分を保持する。
        self._result, self._resolved_snap = result, resolved_snap
        self._last_live_reeval_t = None
        adv, p1, drivers = _score_advantage(
            self._model, result.board_p1_after, result.board_p2_after,
            resolved_snap, attribution_exclude=self._attribution_exclude,
        )
        if self._enable_decisive_amplify:
            adv, p1 = self._amplify_decisive(adv, result)
        if reset_nondefender_cycles:
            self._reset_nondefender_cycles()
        else:
            self._refresh_nondefender_cycle_anchors(adv)
        self.hold_adv, self.hold_p1, self.hold_drivers = adv, p1, drivers

    def _resolve(
        self, snap: OjamaAccountSnapshot, elapsed_sec: float,
        score1: float, score2: float,
        *, reset_nondefender_cycles: bool = True,
    ) -> None:
        """発火時または確定時の得点から、従来どおり決着値を計算する。"""
        gen1 = iv._score_to_ojama_count(score1, elapsed_sec)
        gen2 = iv._score_to_ojama_count(score2, elapsed_sec)
        result = resolve_mutual_exchange(
            self._ev1.before_board, self._ev2.before_board, gen1, gen2,
            snap.pending_p1, snap.pending_p2,
        )
        self._store_resolved_result(
            snap, result,
            reset_nondefender_cycles=reset_nondefender_cycles)
        self._pred_score1, self._pred_score2 = score1, score2

    def _maybe_redecide_physical_net(
        self, snap: OjamaAccountSnapshot, physical_net_raw: "float | None",
        physical_is_unresolved: bool,
    ) -> None:
        """未解決交換の純残量が変化した時だけ、物理下限で再決着する。"""
        if not self._enable_episode_physical_redecide or not physical_is_unresolved:
            return
        if physical_net_raw is None or abs(physical_net_raw) < 1e-9:
            return
        if self._episode_physical_net_last == physical_net_raw:
            return
        gen1 = max(0, int(round(physical_net_raw)))
        gen2 = max(0, int(round(-physical_net_raw)))
        result = resolve_mutual_exchange(
            self._ev1.before_board, self._ev2.before_board, gen1, gen2, 0, 0)
        self._store_resolved_result(
            snap, result, reset_nondefender_cycles=False)
        self._episode_physical_net_last = float(physical_net_raw)
        self._episode_physical_stats["redecide"] += 1

    @property
    def episode_physical_stats(self) -> dict:
        """試合境界をまたいで共有する動画全体の物理追従カウンタ。"""
        return self._episode_physical_stats

    @property
    def episode_physical_redecide_count(self) -> int:
        return int(self._episode_physical_stats["redecide"])

    @property
    def episode_consistency_fallback_count(self) -> int:
        return int(self._episode_physical_stats["fallback"])

    @property
    def episode_consistency_fallback_times(self) -> list[float]:
        return self._episode_physical_stats["fallback_times"]

    def has_untrusted_minimum_active_chain(
        self, state1: BoardState, state2: BoardState,
    ) -> bool:
        """進行中連鎖の完走予測が最小40点に潰れているかを返す。

        ChainEvent の simulate は認識連結欠損時に本物の多連鎖を1連鎖・40点と
        過小評価する。物理 state が CHAIN の間だけ「未確定の下限」と扱い、
        終了後の実測確定値まで否定しない。
        """
        if not self._active:
            return False
        pairs = (
            (self._ev1, state1, self._pred_score1),
            (self._ev2, state2, self._pred_score2),
        )
        untrusted = [
            state == BoardState.CHAIN
            and event is not None
            and event.chain_count == 1
            and event.total_score == CHAIN_TOTAL_MIN_SCORE
            and predicted_score <= CHAIN_TOTAL_MIN_SCORE
            for event, state, predicted_score in pairs
        ]
        if sum(untrusted) != 1:
            return False
        other_index = 1 - untrusted.index(True)
        return pairs[other_index][2] > CHAIN_TOTAL_MIN_SCORE

    def apply_episode_consistency(
        self, candidate_adv: float, candidate_p1: float,
        stable_adv: float, stable_p1: float, model_adv: float,
        physical_net_raw: float, *, is_unresolved: bool,
        allows_hard_override: bool,
    ) -> "tuple[float, float, bool]":
        """台帳と生モデルの両方に逆らう未解決中の単独反転を保留する。

        `stable_adv` は直前の resolved hold を解除時に引き継ぐため、常に独立な
        STABLE評価とは限らない。台帳と生モデルが一致した場合はこの2票を採用し、
        stableも同方向なら較正済みstable値、そうでなければ生モデル値へ戻す。
        """
        enabled = self._enable_episode_physical_consistency_guard and is_unresolved
        if not enabled or allows_hard_override or abs(physical_net_raw) < 1e-9:
            return candidate_adv, candidate_p1, False
        conflicts = candidate_adv * physical_net_raw < 0.0
        model_supports = (
            abs(model_adv) > EVEN_THRESHOLD
            and model_adv * physical_net_raw > 0.0)
        if not conflicts or not model_supports:
            return candidate_adv, candidate_p1, False
        stable_supports = (
            abs(stable_adv) > EVEN_THRESHOLD
            and stable_adv * physical_net_raw > 0.0)
        self._episode_physical_stats["fallback"] += 1
        self._episode_physical_stats["fallback_times"].append(
            round(float(self._t_sec), 3))
        if stable_supports:
            return stable_adv, stable_p1, True
        return model_adv, adv_to_winprob(model_adv), True

    def _update_landing_targets(self, result: "MutualExchangeResult") -> None:
        """決着計算が予測した各 side の最終お邪魔到達量 (着弾完了判定①用) を保持する。

        [指摘11] 目標 = 発火直前の盤面おじゃま数 + 今回の交換で確定した
        飛来量総量 (即時落下分 dropped + 次ターン繰越 leftover)。
        """
        base1 = float(iv.board_ojama_count(self._ev1.before_board).raw)
        base2 = float(iv.board_ojama_count(self._ev2.before_board).raw)
        self._incoming_total_p1 = float(result.dropped_to_p1 + result.leftover_p1)
        self._incoming_total_p2 = float(result.dropped_to_p2 + result.leftover_p2)
        self._target_ojama_p1 = base1 + self._incoming_total_p1
        self._target_ojama_p2 = base2 + self._incoming_total_p2

    def _decisive_defender(
        self, result: "MutualExchangeResult",
    ) -> "tuple[str | None, float]":
        """今回の交換で最終的に飛来量が大きい側 (受け側) を返す (指摘10)。

        `_resolve_defender_threat` と同じ「脅威が無ければ None / 両方向とも
        あれば大きい方を優先」という判断パターンを、決着計算の生データ
        (result) 向けに踏襲する (入力が異なるため別関数、ロジックは対応)。
        """
        candidates = [
            (side, amount) for side, amount in (
                ("1P", float(result.dropped_to_p1 + result.leftover_p1)),
                ("2P", float(result.dropped_to_p2 + result.leftover_p2)),
            ) if amount > 0.0
        ]
        if not candidates:
            return None, 0.0
        return max(candidates, key=lambda c: c[1])

    def _amplify_decisive(
        self, adv: float, result: "MutualExchangeResult",
    ) -> "tuple[float, float]":
        """[指摘10] 受け側の応手不能度を決着値に統合する (既定 enable 時のみ呼ばれる)。

        受け側限定の応手確率を既存 CounterReachTracker (受け側限定経路) から
        求め、既存 `_counter_defender_adv` (`RESOLVED_AMPLIFY_SCALE` +
        iv.ojama_damage) と同一式で adv に加算する。応手不能 (確率低) かつ
        飛来量大なら決定的側へ増幅し、受け側が高確率で返せる場合はほぼ
        無効果のまま。

        [指摘12 修正1、2026-08-14] 時間予算は #3 で実装済みの
        `_chain_remaining_time_budget_sec` (経過時間控除 +
        E[最終連鎖数|観測N到達] の条件付き期待) **のみ**で計算する
        (旧式 `iv.estimate_chain_anim_duration_sec(観測連鎖数)` の直呼びは
        禁止 — `test_amplify_decisive_source_never_calls_legacy_time_budget_directly`
        で静的に検査する)。旧式は経過時間を控除しないため、連鎖の後半ほど
        残り時間を過小評価し「応手不能」と誤断する系統バイアスがあった
        (指摘12: 実演出8.1秒に対し旧式2.4秒と算出、応手0%→過剰増幅の直接原因)。

        [指摘12 修正4、2026-08-14 意味論バグ対処] 応手確率の MC 入力盤面は
        `board_p1_pre_landing`/`board_p2_pre_landing` (着弾**前**、自分の
        連鎖は消化済みだが余剰おじゃまはまだ配置していない盤面) を使う。
        修正1 (時間予算) 後も応手0%が残っていた根因がこれで、着弾**後**
        盤面 (`board_p1_after`/`board_p2_after`、既に降り切っている) から
        MC を回すと、まだ空中のはずのおじゃまが盤面を埋めた状態で判定して
        しまい不当に過小評価される。一方 `_counter_defender_adv` の
        ダメージ計算 (iv.ojama_damage) は「返せなかった場合に何が起きるか」
        を測るものなので着弾後盤面のままが正しい (下記で使い分ける、
        混同しないこと)。
        """
        defender_side, incoming = self._decisive_defender(result)
        self.hold_defender_side, self.hold_incoming_ojama = defender_side, incoming
        if defender_side is None:
            self.hold_defender_prob = float("nan")
            return adv, adv_to_winprob(adv)
        # 回復時間 = 相手 (攻撃側) の隙時間。
        # [2026-08-15 コメント是正] 以前の記述は「実観測 ChainEvent.chain_count
        # (simulate由来のchain_countは使わない)」としていたが誤り。
        # ChainEvent.chain_count 自体が常に simulate 由来 (chain_detector.py
        # `_try_emit_event` の `sim.chain_count`、chain_detector.py:280,300) で
        # あり、「simulate由来でない実観測 chain_count」は存在しない
        # (project_chain_count_both_untrustworthy の過小評価事故はこの値
        # そのものが対象)。ここで attacker_event.chain_count を使うのは値を
        # 信頼しているからではなく、`_expected_final_chain_count` が
        # 「観測 N 到達」を入力に E[最終連鎖数|N到達] を引く条件付き期待値
        # テーブル (#3 修正) の**入力**として扱うことで simulate 過小評価を
        # 補正する設計だからである。#3 修正と同じ「時間予算の計算箇所」
        # 1関数 (_chain_remaining_time_budget_sec) だけを経由する
        # (他の場所で時間予算を計算しない、根本修正)。
        attacker_event = self._ev2 if defender_side == "1P" else self._ev1
        budget = _chain_remaining_time_budget_sec(
            attacker_event.chain_count, attacker_event.trigger_sec, self._t_sec,
            self._chain_len_table,
        )
        # 応手確率の判定は着弾前盤面 (修正4)。ダメージ計算 (下の
        # _counter_defender_adv) は着弾後盤面のまま — 意味論が異なるため
        # 混同しない。
        _, cp1, cp2 = self._counter_tracker.update(
            result.board_p1_pre_landing, result.board_p2_pre_landing, budget,
            defender_side=defender_side, threshold_ojama=incoming,
            reuse_if_board_unchanged=self._enable_counter_placement_reuse,
            quantize_budget_sec=self._enable_counter_budget_quantize,
        )
        defender_prob = cp1 if defender_side == "1P" else cp2
        self.hold_defender_prob = defender_prob
        amp = _counter_defender_adv(
            defender_side, defender_prob, incoming,
            result.board_p1_after, result.board_p2_after,
            scale=RESOLVED_AMPLIFY_SCALE,
        )
        adv = max(-100.0, min(100.0, adv + amp))
        return adv, adv_to_winprob(adv)

    def _live_defender_snap(
        self, defender_side: str, leftover_now: float,
    ) -> OjamaAccountSnapshot:
        """[指摘13、2026-08-15 方向反転修正で意味論変更] 受け側の forecast を
        「このフレームで物理着弾させた後に残った未着弾分」に差し替えた
        snapshot を返す (`_reevaluate_live_defender` docstring 参照)。

        受け側の盤面はこのフレームで `leftover_now` を除く分を
        `land_pending_ojama_onto_board` により物理的に着弾済みにしたため、
        forecast も凍結経路 `_resolve()` と同じ意味論 (= 着弾済み分は盤面が
        語る、forecast は残りだけ) に揃える。旧実装 (全量 `_incoming_total_pX`
        を forecast に積む方式) は forecast 特徴がモデルにほぼ無視される
        (W12) ため方向反転を解消できず撤回した。攻撃側は盤面
        (`board_pX_after`) を変えていないため forecast は元の leftover の
        まま (二重計上しない)。net_balance_capped も同じ差し替え後の値の
        差分に揃える (`_side_feats` の net/forecast 引数と整合)。
        """
        if defender_side == "1P":
            forecast_p1, forecast_p2 = leftover_now, self._resolved_snap.forecast_p2
        else:
            forecast_p1, forecast_p2 = self._resolved_snap.forecast_p1, leftover_now
        return dataclass_replace(
            self._resolved_snap,
            forecast_p1=forecast_p1, forecast_p2=forecast_p2,
            net_balance_capped=forecast_p2 - forecast_p1,
        )

    def _live_remaining_incoming(
        self, defender_side: str, live_defender_board: Board,
        snap: "OjamaAccountSnapshot | None",
    ) -> float:
        """[指摘13 方向反転修正、2026-08-15] 受け側の「まだ着弾していない」量を、
        現在の会計優先・盤面差分フォールバックで求める。

        優先: `snap.pending_pX` (実際の tsumo 着地の度に drain される会計値、
        現在までに本当に降った分は自然に減っている=二重計上しない)。
        フォールバック: 会計側が0を示しているのに凍結時点の飛来予測
        (`self._incoming_total_pX`) が正の場合 (baseline reset (score大幅
        減少検知) 等で会計がこの交換を追跡できていない場合)、凍結時点の
        飛来量から盤面上で既に増えた分を控除した残りを使う
        (`self._target_ojama_pX` = 凍結時盤面 + 飛来総量、なので
        `target - 現在盤面値` = 未着弾の残り)。
        """
        incoming_total = (
            self._incoming_total_p1 if defender_side == "1P" else self._incoming_total_p2)
        if incoming_total <= 0.0:
            return 0.0
        if snap is not None:
            acct_pending = float(
                snap.pending_p1 if defender_side == "1P" else snap.pending_p2)
            if acct_pending > 0.0:
                return min(acct_pending, incoming_total)
        target = (
            self._target_ojama_p1 if defender_side == "1P" else self._target_ojama_p2)
        current = float(iv.board_ojama_count(live_defender_board).raw)
        return max(0.0, target - current)

    def _reset_nondefender_cycles(self) -> None:
        """新しい決着結果では side-local chain の基準を取り直す。"""
        self._nondef_cycle_armed = [False, False]
        self._nondef_cycle_in_chain = [False, False]
        self._nondef_cycle_prev_state = [None, None]
        self._nondef_cycle_before_board = [None, None]
        self._nondef_cycle_score_start = [None, None]
        self._nondef_cycle_stable_since = [None, None]
        self._nondef_cycle_event = [None, None]
        self._nondef_cycle_hold_anchor = [None, None]
        self._nondef_cycle_saw_new_hand = [False, False]
        self._nondef_cycle_chain_id = [None, None]
        self._nondef_fallback_active = [False, False]
        self._nondef_fallback_anchor = [None, None]
        self._nondef_fallback_before = [None, None]

    def _refresh_nondefender_cycle_anchors(self, adv: float) -> None:
        """同じ交換内の再決着後も観測中の連鎖を残し、基準値だけ更新する。"""
        for idx, in_chain in enumerate(self._nondef_cycle_in_chain):
            if in_chain:
                self._nondef_cycle_hold_anchor[idx] = adv

    def _apply_nondefender_cycle_after(
        self, side: str, after: Board, anchor: float, source: str,
        before: "Board | None" = None,
    ) -> bool:
        """同じ実盤面系列の連鎖前後差だけを反映し、方向反転は拒否する。"""
        idx = 0 if side == "1P" else 1
        observed_before = (
            before if before is not None else self._nondef_cycle_before_board[idx])
        if (
            self._result is None or self._resolved_snap is None
            or observed_before is None
        ):
            return False
        frozen = (
            self._result.board_p2_after if side == "1P"
            else self._result.board_p1_after)
        ref_b1, ref_b2 = (
            (observed_before, frozen) if side == "1P"
            else (frozen, observed_before))
        new_b1, new_b2 = (after, frozen) if side == "1P" else (frozen, after)
        ref_adv, _ref_p1, _ref_drivers = _score_advantage(
            self._model, ref_b1, ref_b2, self._resolved_snap,
            attribution_exclude=self._attribution_exclude)
        new_adv, _new_p1, _new_drivers = _score_advantage(
            self._model, new_b1, new_b2, self._resolved_snap,
            attribution_exclude=self._attribution_exclude)
        delta = new_adv - ref_adv
        candidate = max(-100.0, min(100.0, anchor + delta))
        if abs(anchor) > EVEN_THRESHOLD and anchor * candidate < 0.0:
            self.nondef_cycle_stats["rejected_direction"] += 1
            self.nondef_cycle_trace.append((
                "direction_reject", round(float(self._t_sec), 3), side, source,
                anchor, ref_adv, new_adv, delta, candidate,
                _board_hash(observed_before), _board_hash(after)))
            return False
        self.hold_adv, self.hold_p1 = candidate, adv_to_winprob(candidate)
        self.nondef_cycle_deltas.append((
            round(float(self._t_sec), 3), side, source,
            anchor, ref_adv, new_adv, delta, candidate))
        return True

    @staticmethod
    def _simulate_cycle_after(
        event: "ChainEvent | None", observed_score_delta: int,
    ) -> "Board | None":
        """観測得点・連鎖数と整合する時だけChainEventの完走盤面を返す。"""
        if event is None or observed_score_delta < CHAIN_TOTAL_MIN_SCORE:
            return None
        result = _CHAIN_COMPLETION_SIMULATOR.simulate(event.before_board)
        if result.chain_count < 1 or result.chain_count != event.chain_count:
            return None
        expected_score = int(calculate_chain_score(result).total_score)
        tolerance = max(10, int(0.05 * expected_score))
        if abs(observed_score_delta - expected_score) > tolerance:
            return None
        return result.final_board

    def nondef_cycle_summary(self) -> str:
        """side-local chain補正の母数・棄却数・実効果を1行へ整形する。"""
        s = self.nondef_cycle_stats
        return (
            f"ARM {s['armed']} / 開始 {s['started']} / 適用 {s['applied']} / "
            f"得点棄却 {s['rejected_score']} / 同盤面棄却 {s['rejected_same_board']} / "
            f"遅延棄却 {s['rejected_late']} / stale event {s['rejected_stale_event']} / "
            f"新規手なし {s['rejected_no_new_hand']} / "
            f"simulate不整合 {s['rejected_sim_mismatch']} / "
            f"方向拒否 {s['rejected_direction']} / fallback {s['fallback_applied']} / "
            f"fresh置換 {s['fresh_replaced']} / "
            f"差分 {self.nondef_cycle_deltas} / trace {self.nondef_cycle_trace}"
        )

    def _clear_nondefender_cycle(self, idx: int) -> None:
        """side-local chain 1周期分の一時状態だけを解放する。"""
        self._nondef_cycle_in_chain[idx] = False
        self._nondef_cycle_before_board[idx] = None
        self._nondef_cycle_score_start[idx] = None
        self._nondef_cycle_stable_since[idx] = None
        self._nondef_cycle_event[idx] = None
        self._nondef_cycle_hold_anchor[idx] = None
        self._nondef_cycle_chain_id[idx] = None

    def _replace_nondefender_fallback_if_fresh(
        self, idx: int, side: str, board: "Board | None",
        state: "BoardState | None",
    ) -> bool:
        """fallback後、次の行動前にfresh盤面が来た時だけanchorから置換する。"""
        if not self._nondef_fallback_active[idx]:
            return False
        if state != BoardState.STABLE:
            self._nondef_fallback_active[idx] = False
            self._nondef_fallback_anchor[idx] = None
            self._nondef_fallback_before[idx] = None
            return False
        before = self._nondef_fallback_before[idx]
        if board is None or before is None or board == before:
            return True
        anchor = self._nondef_fallback_anchor[idx]
        if anchor is not None and self._apply_nondefender_cycle_after(
                side, board, anchor, "fresh_replace", before=before):
            self.nondef_cycle_stats["fresh_replaced"] += 1
        self._nondef_fallback_active[idx] = False
        self._nondef_fallback_anchor[idx] = None
        self._nondef_fallback_before[idx] = None
        return True

    def _start_nondefender_cycle(
        self, idx: int, side: str, board: "Board | None", score: "int | None",
        event: "ChainEvent | None", physical_chain_id: "int | None",
    ) -> None:
        """ARM後の新しいeventだけをside-local chain開始として保存する。"""
        original = self._ev1 if side == "1P" else self._ev2
        trigger = getattr(event, "trigger_sec", None)
        original_trigger = getattr(original, "trigger_sec", None)
        key = (side, int(physical_chain_id)) if physical_chain_id is not None else None
        stale = (
            key is None or key in self._nondef_applied_event_keys
            or physical_chain_id == self._resolved_root_chain_id[idx]
            or (original_trigger is not None and trigger <= original_trigger))
        if stale:
            self.nondef_cycle_stats["rejected_stale_event"] += 1
            return
        event_before = event.before_board if event is not None else None
        before = event_before if event_before is not None else board
        self._nondef_cycle_in_chain[idx] = True
        self._nondef_cycle_before_board[idx] = before.copy() if before is not None else None
        self._nondef_cycle_score_start[idx] = score
        self._nondef_cycle_stable_since[idx] = None
        self._nondef_cycle_event[idx] = event
        self._nondef_cycle_hold_anchor[idx] = self.hold_adv
        self._nondef_cycle_chain_id[idx] = physical_chain_id
        self.nondef_cycle_stats["started"] += 1
        self.nondef_cycle_trace.append((
            "start", round(float(self._t_sec), 3), side, score,
            trigger, original_trigger, physical_chain_id,
            self._resolved_root_chain_id[idx],
            _board_hash(board), _board_hash(event_before)))

    def _finish_nondefender_cycle(
        self, idx: int, side: str, board: "Board | None", score: "int | None",
    ) -> bool:
        """fresh盤面を優先し、0.5秒後は整合済みsimulateへ限定fallbackする。"""
        before = self._nondef_cycle_before_board[idx]
        start = self._nondef_cycle_score_start[idx]
        anchor = self._nondef_cycle_hold_anchor[idx]
        event = self._nondef_cycle_event[idx]
        observed_delta = score - start if score is not None and start is not None else -1
        score_ready = observed_delta >= CHAIN_TOTAL_MIN_SCORE
        board_ready = before is not None and board is not None and before != board
        stable_since = self._nondef_cycle_stable_since[idx]
        waited = 0.0 if stable_since is None else self._t_sec - stable_since
        if score_ready and board_ready and anchor is not None:
            applied = self._apply_nondefender_cycle_after(
                side, board, anchor, "fresh", before=before)
        elif waited <= COUNTER_RECOMPUTE_INTERVAL_SEC:
            return False
        elif not score_ready:
            self.nondef_cycle_stats["rejected_score"] += 1
            applied = False
        else:
            simulated = self._simulate_cycle_after(event, observed_delta)
            if simulated is None or anchor is None:
                self.nondef_cycle_stats["rejected_sim_mismatch"] += 1
                applied = False
            else:
                applied = self._apply_nondefender_cycle_after(
                    side, simulated, anchor, "simulate_fallback", before=before)
                if applied:
                    self.nondef_cycle_stats["fallback_applied"] += 1
                    self._nondef_fallback_active[idx] = True
                    self._nondef_fallback_anchor[idx] = anchor
                    self._nondef_fallback_before[idx] = before
        physical_chain_id = self._nondef_cycle_chain_id[idx]
        if physical_chain_id is not None:
            self._nondef_applied_event_keys.add((side, int(physical_chain_id)))
        self.nondef_cycle_stats["applied"] += int(applied)
        self._clear_nondefender_cycle(idx)
        return True

    def _observe_nondefender_cycle(
        self, side: str, board: "Board | None", state: "BoardState | None",
        score: "int | None", chain_event: "ChainEvent | None",
        chain_end_confirmed: bool, physical_chain_id: "int | None",
    ) -> None:
        """hold後に開始から終了まで見えた新規side-local chainだけを補正する。"""
        idx = 0 if side == "1P" else 1
        prev = self._nondef_cycle_prev_state[idx]
        self._nondef_cycle_prev_state[idx] = state
        if self._replace_nondefender_fallback_if_fresh(idx, side, board, state):
            return
        if not self._nondef_cycle_armed[idx]:
            if (
                chain_end_confirmed and state == BoardState.STABLE
                and board is not None
            ):
                self._nondef_cycle_armed[idx] = True
                self._nondef_cycle_saw_new_hand[idx] = False
                self.nondef_cycle_stats["armed"] += 1
            return
        if state == BoardState.TSUMO_FALL:
            self._nondef_cycle_saw_new_hand[idx] = True
        if not self._nondef_cycle_in_chain[idx]:
            if state == BoardState.CHAIN and prev not in _LIVE_DEFENDER_BUSY_STATES:
                if self._nondef_cycle_saw_new_hand[idx]:
                    self._start_nondefender_cycle(
                        idx, side, board, score, chain_event, physical_chain_id)
                    self._nondef_cycle_saw_new_hand[idx] = False
                else:
                    self.nondef_cycle_stats["rejected_no_new_hand"] += 1
            return
        if state != BoardState.STABLE:
            self._nondef_cycle_stable_since[idx] = None
            return
        before = self._nondef_cycle_before_board[idx]
        if self._nondef_cycle_stable_since[idx] is None:
            self._nondef_cycle_stable_since[idx] = self._t_sec
            self.nondef_cycle_trace.append((
                "stable", round(float(self._t_sec), 3), side, score,
                _board_hash(before), _board_hash(board)))
        self._finish_nondefender_cycle(idx, side, board, score)

    def _reevaluate_live_defender(
        self, b1: "Board | None", b2: "Board | None",
        snap: "OjamaAccountSnapshot | None" = None,
        state1: "BoardState | None" = None, state2: "BoardState | None" = None,
        score1: "int | None" = None, score2: "int | None" = None,
        event1: "ChainEvent | None" = None, event2: "ChainEvent | None" = None,
        chain_id1: "int | None" = None, chain_id2: "int | None" = None,
    ) -> None:
        """[指摘13、2026-08-15] 交換中、受け側の現在盤面+残り
        時間逓減で hold_adv/hold_p1/hold_drivers を再評価する。

        `update()` 側で `enable_live_defender_reeval=True` かつ、従来の片側event
        条件、または strict ONで受け側の物理stateがfreeの場合に呼ばれる。
        `COUNTER_RECOMPUTE_INTERVAL_SEC` (0.5秒、既存の応手判定
        周期と同一) 未満の連続呼び出しは即 return しキャッシュ値を保持する
        (無効化中の判定は行わない=呼出元のフラグゲートが単一情報源)。

        b1/b2: 呼出側 generate() が保持する「受け側の現在の STABLE 確定盤面」
        (sticky、片側STABLE時のみ更新・非STABLE中は前回値を保持)。凍結対象の
        攻撃側盤面は `self._result.board_pX_after`/`board_pX_pre_landing`
        (直近 `_resolve()` が保持した仮想盤面) をそのまま使い続ける — 攻撃側
        の生盤面は連鎖アニメ中で信用できない (physics_only 原則)。
        snap: [方向反転修正、2026-08-15 追加の optional 引数] 呼出側 `update()`
            が保持する現在の `OjamaAccountSnapshot` (`_live_remaining_incoming`
            参照)。省略時 (None、backwards compat) は会計フォールバック側の
            盤面差分のみで残量を求める。
        state1/state2: [指摘14 案1、2026-08-15 追加の optional 引数、案1初版の
            ev1/ev2 から差し替え] 呼出側 `update()` が同フレームで観測した
            `r_p1.state`/`r_p2.state` (状態機械 BoardState)。
            `enable_live_defender_strict=True` の場合のみ、defender_side 自身の
            state が `_LIVE_DEFENDER_BUSY_STATES` (今まさに連鎖処理中) に
            含まれるかの検証に使う (クラス docstring 指摘14節、chain_event
            ではなく state を使う理由の実測根拠を参照)。strict=False (既定)
            では未使用 (省略しても backwards compat)。

        [方向反転修正、2026-08-15、docs/KNOWN_WEAKNESSES.md W12]
        受け側の生盤面 (未着弾=クリーン) を直接モデルへ渡す限り、forecast を
        どれだけ差し替えても方向反転は解消しない (モデルは forecast 特徴を
        ほぼ無視する)。モデル評価に使う盤面は `land_pending_ojama_onto_board`
        で未着弾分 (`_live_remaining_incoming` が求める、`OJAMA_MAX_DROP_
        PER_TURN` 上限まで) を物理的に着弾させた仮想盤面に差し替える。
        forecast (`_live_defender_snap`) はこのフレームで着弾しきれず残った
        `leftover_now` のみに揃え、凍結経路 `_resolve()` (forecast=leftover)
        と意味論を一致させる (二重計上しない)。
        """
        if self._result is None or self._resolved_snap is None:
            return  # _resolve 未実行 (理論上到達しない、安全側 no-op)
        defender_side, incoming = self._decisive_defender(self._result)
        if defender_side is None:
            return  # 脅威なし(相殺で完全相殺等)は再評価対象なし、既存値を保持
        if self._enable_live_defender_strict:
            # [指摘14 案1、状態機械 state 版] defender_side 自身が今まさに
            # 自分の連鎖 (消去+重力) を処理中でないことを状態機械 state で
            # 確認する。chain_event の有無 (旧版) は「trigger 検知フレームで
            # 1度だけ発行され一定時間 hold 後 None に戻る」パルス方式のため、
            # 旧連鎖の hold 切れ〜新連鎖 trigger 検知までの settle gap で
            # 「defender 自身の ev が None」になる瞬間が生じ、そこを「自由」と
            # 誤判定していた (実測: logs/_diag_issue14_reeval_calls_2026-08-15.log、
            # クラス docstring 指摘14節参照)。state は毎フレーム直接観測される
            # ため gap が生じない。不一致 (busy) なら再評価せず直前の保持値を
            # 維持する (安全側、新規推測ロジックは追加せず既存の state 観測を
            # そのまま使う)。
            defender_state = state1 if defender_side == "1P" else state2
            if defender_state in _LIVE_DEFENDER_BUSY_STATES:
                # 初回STABLE復帰は基準化だけ。その後に開始→終了を両方観測した
                # 反対側の新規連鎖だけを paired-delta で1回補正する。
                side = "2P" if defender_side == "1P" else "1P"
                self._observe_nondefender_cycle(
                    side, b2 if side == "2P" else b1,
                    state2 if side == "2P" else state1,
                    score2 if side == "2P" else score1,
                    event2 if side == "2P" else event1,
                    bool(
                        getattr(snap, "chain_end_triggered_p2", False)
                        if side == "2P" else
                        getattr(snap, "chain_end_triggered_p1", False)),
                    chain_id2 if side == "2P" else chain_id1)
                return
        if (
            self._last_live_reeval_t is not None
            and (self._t_sec - self._last_live_reeval_t) < COUNTER_RECOMPUTE_INTERVAL_SEC
        ):
            return
        attacker_event = self._ev2 if defender_side == "1P" else self._ev1
        if attacker_event is None:
            return  # 攻撃側イベント不明(理論上到達しない防御的ガード)
        live_defender_board = b1 if defender_side == "1P" else b2
        if live_defender_board is None:
            return  # 受け側の STABLE 盤面をまだ一度も観測していない(安全側、段差回避)
        attacker_board_frozen = (
            self._result.board_p2_after if defender_side == "1P"
            else self._result.board_p1_after)
        remaining = self._live_remaining_incoming(defender_side, live_defender_board, snap)
        # [予告おじゃまの降下条件、2026-08-21 user 仕様伝授]
        #
        # おじゃまが実際に降るには順に3条件が必要:
        #   (1) 相手の連鎖が確定する (相手のネクストが動き始める瞬間)
        #   (2) 受け側の現在の手がフィールドに置かれる
        #   (3) その手で連鎖が起きない
        # (3) で連鎖が起きた場合はその連鎖終了後に降るが、相殺で予告が
        # 消えれば降らない。
        #
        # 従来はこの条件を一切見ず「予告量 > 0 なら降らせる」だった
        # (land_pending_ojama_onto_board は量だけを見る)。そのため
        # **攻撃側がまだ連鎖している最中に受け側へ降らせて評価**しており、
        # 受け側がその間に連鎖を撃って相殺できる可能性を潰して不利に
        # 見せていた。指摘13 (受け側の生盤面をそのまま渡すと無傷に見える)
        # への対処として降らせる方式を採ったが、逆方向に振れた形。
        #
        # 修正: 攻撃側が連鎖中 (= 条件(1)未成立) の間は降らせない。
        # 予告は「迫っている量」として snapshot 側 (_live_defender_snap) が
        # forecast として保持し続けるので、情報が失われるわけではない。
        # 攻撃側の連鎖が終わっていれば従来どおり降らせる (条件(2)(3)は
        # 受け側 state が BUSY でないこと=上の strict ガードで近似される)。
        attacker_state = state2 if defender_side == "1P" else state1
        attacker_still_chaining = attacker_state in _LIVE_DEFENDER_BUSY_STATES
        if self._enable_pending_landing_gate and attacker_still_chaining:
            # まだ降っていない: 受け側の生盤面をそのまま評価に使う。
            landed_defender_board = live_defender_board
            leftover_now = int(round(max(0.0, remaining)))
        else:
            landed_defender_board, _dropped_now, leftover_now = (
                land_pending_ojama_onto_board(
                    live_defender_board, attacker_board_frozen, remaining))
        board_p1 = (
            landed_defender_board if defender_side == "1P" else self._result.board_p1_after)
        board_p2 = (
            landed_defender_board if defender_side == "2P" else self._result.board_p2_after)
        live_snap = self._live_defender_snap(defender_side, leftover_now)
        adv, p1, drivers = _score_advantage(
            self._model, board_p1, board_p2, live_snap,
            attribution_exclude=self._attribution_exclude,
        )
        if self._enable_decisive_amplify:
            budget = _chain_remaining_time_budget_sec(
                attacker_event.chain_count, attacker_event.trigger_sec, self._t_sec,
                self._chain_len_table,
            )
            # 応手確率MC (CounterReachTracker) は着弾**前**の生盤面を使う
            # (指摘12 修正4 と同じ「降られる前に撃てるか」の意味論)。上の
            # board_p1/board_p2 (モデル評価用、着弾後仮想盤面) とは別物
            # — 混同しないこと。非受け側は未使用のため元の pre_landing を
            # 渡す (既存 _amplify_decisive と同じ組み方)。
            counter_b1 = (
                live_defender_board if defender_side == "1P"
                else self._result.board_p1_pre_landing)
            counter_b2 = (
                live_defender_board if defender_side == "2P"
                else self._result.board_p2_pre_landing)
            _, cp1, cp2 = self._counter_tracker.update(
                counter_b1, counter_b2, budget,
                defender_side=defender_side, threshold_ojama=incoming, t_sec=self._t_sec,
                reuse_if_board_unchanged=self._enable_counter_placement_reuse,
                quantize_budget_sec=self._enable_counter_budget_quantize,
            )
            defender_prob = cp1 if defender_side == "1P" else cp2
            self.hold_defender_side, self.hold_incoming_ojama, self.hold_defender_prob = (
                defender_side, incoming, defender_prob)
            amp = _counter_defender_adv(
                defender_side, defender_prob, incoming,
                self._result.board_p1_after, self._result.board_p2_after,
                scale=RESOLVED_AMPLIFY_SCALE,
            )
            adv = max(-100.0, min(100.0, adv + amp))
            p1 = adv_to_winprob(adv)
        self.hold_adv, self.hold_p1, self.hold_drivers = adv, p1, drivers
        self._last_live_reeval_t = self._t_sec

    def hold_after_kill_override(
        self, b1: "Board | None", b2: "Board | None",
        state1: "BoardState | None" = None, state2: "BoardState | None" = None,
    ) -> "tuple[float, float]":
        """[指摘14 案2、2026-08-15、既定では呼ばれない] 決着ホールド値
        (hold_adv/hold_p1) に致死上書き (`kill_override`) を適用した値を返す。

        従来 `kill_override` はライブ per-frame 経路 (通常の4成分ブレンド)
        にのみ配線されており、決着ホールド中 (`ResolvedExchangeTracker` が
        disp_adv/disp_p1 を丸ごと上書きする経路) には未配線だった。その結果
        pending/room 比が致死水準でも決着ホールド中は安全弁が発火しない
        (実測: 589/50≈11.8 ≫ KILL_RATIO_FULL=1.5 でも無発火)。呼出側
        `generate()` の `--resolved-kill-override` (既定OFF) 有効時のみ、
        表示直前にこのメソッドを呼んで hold_adv/hold_p1 を上書きする。

        room/pending 比の材料は新規に増やさず既存の観測量のみ再利用する:
        pending = `self._incoming_total_p1/p2` (直近 `_resolve()` が
        `_update_landing_targets` で確定した決着済み飛来量総量。指摘11の
        着弾完了判定 `_landing_complete` と同一値、二重定義しない)。
        room = `board_room(b1)/board_room(b2)` (呼出側が保持する現在の
        sticky 確定盤面、モジュール既存の `board_room` をそのまま使う)。

        二重計上防止 (amplify との優先順位): ライブ経路 (3861-3870行付近) と
        同じく `kill_override` を「最終段」として適用する。`kill_override` は
        致死度差が `KILL_RATIO_FULL` 以上で g=1 (完全上書き) となり adv を
        target(±100) へ完全に置換するため、g=1 の場面では amplify 由来の
        寄与 (既に adv に混ざっている) は自動的に上書きされ二重計上しない。
        g<1 (部分ブレンド) の場面では amplify 込みの adv を (1-g) 分だけ残す
        設計を意図的に踏襲する (ライブ経路と同一の優先順位、既存の
        `kill_override` の意味論を変えない)。

        [指摘19対処、2026-08-15、既定OFF、docs/DEMO_REVIEW_2026-08-13.md #19、
        状態ゲート方式へ設計変更 (coordinator判断、確率ブレンド版は撤回)]
        `enable_kill_override_counter_aware=True` の場合、致死断定した側
        (victim_side) 自身の**状態機械 state** が `_LIVE_DEFENDER_BUSY_STATES`
        (= {CHAIN, GRAVITY_SETTLE}、指摘14案1で既に確立済みの状態集合を
        そのまま再利用、新設しない) に含まれなければ致死断定を発火させず
        `hold_adv`/`hold_p1` をそのまま返す (kill_override は安全弁=物理的に
        動けない相手への致死量見落とし対策であり、勝率の微調整装置では
        ないため二値判定にする)。

        【却下した先行案、実測で確定 (logs/_diag_issue19_dampen_trace_
        2026-08-15.log)】応手確率 (MC、CounterReachTracker/
        mc_counter_estimator) による連続ブレンドを先に実装したが、
        (a) `_amplify_decisive` の凍結済み確率をそのまま使うと受け側が
        盤面を積み上げても2秒以上0.0167に凍結されたまま退行が解消せず、
        (b) 現在盤面へフレッシュ再計算しても実測で応手確率は25-40%止まり
        (docs/KNOWN_WEAKNESSES.md W15、mc_counter_estimator の既知の推定
        精度限界) であり、target=±100 固定の線形ブレンドでは56%前後という
        合格水準に届かなかった。指摘14案1が chain_event のパルス方式
        依存(誤爆)から状態機械ベースへ切り替えて解決した前例と同じ
        パターンを踏襲し、確率推定の精度に一切依存しない二値の state ゲート
        に置き換えた。

        state1/state2: 呼出側 `generate()` が同フレームで観測した
        `r.p1.state`/`r.p2.state` (状態機械 BoardState、
        `_reevaluate_live_defender` の state1/state2 と同じ観測量を再利用、
        新しい観測量は増やさない)。省略時 (None、backwards compat) は
        「busy かどうか判定不能」として警戒を緩めず従来通り発火する
        (fail-silent を避ける、既存呼出元・既存テストは無変化)。
        """
        room1, room2 = board_room(b1), board_room(b2)
        adv = kill_override(
            self.hold_adv, self._incoming_total_p1, self._incoming_total_p2, room1, room2)
        if adv == self.hold_adv:
            return self.hold_adv, self.hold_p1
        if self._enable_kill_override_counter_aware:
            victim_state = state1 if adv < self.hold_adv else state2
            if victim_state is not None and victim_state not in _LIVE_DEFENDER_BUSY_STATES:
                return self.hold_adv, self.hold_p1  # 受け側は自由行動中=致死断定しない
        return adv, adv_to_winprob(adv)

    def _redecide_obs(
        self, triggered: bool, total_score: int, side_is_1p: bool, elapsed_sec: float,
    ) -> "float | None":
        """1 side 分の再決着観測値を返す (`_maybe_redecide` 補助、指摘19根治)。

        既定 (enable_resolved_victim_gen_live=False): 従来通り保持セッション
        中 1回きり (`_redecided1/2` latch、bit-identical)。

        True の場合: `chain_end_triggered_pX` が True の間 (=同一連鎖の
        settle が継続中。ojama_accounting.py `_finalize_chain_end` は
        coalesce window 内で `chain_total_score_pX` を複数回に分けて段階的に
        確定する、クラス docstring 指摘19根治節の実測ログ参照)、
        `COUNTER_RECOMPUTE_INTERVAL_SEC` (0.5秒) ごとに再チェックし追従する
        (`_maybe_redecide` 側の `max(pred, obs)` が単調増加を保証)。
        """
        redecided = self._redecided1 if side_is_1p else self._redecided2
        if not triggered:
            return None
        if not self._enable_resolved_victim_gen_live:
            obs = None if redecided else float(total_score)
        else:
            last_t = self._victim_live_last_t1 if side_is_1p else self._victim_live_last_t2
            due = last_t is None or (elapsed_sec - last_t) >= COUNTER_RECOMPUTE_INTERVAL_SEC
            obs = float(total_score) if (not redecided or due) else None
            if obs is not None:
                if side_is_1p:
                    self._victim_live_last_t1 = elapsed_sec
                else:
                    self._victim_live_last_t2 = elapsed_sec
        if side_is_1p:
            self._redecided1 = redecided or triggered
        else:
            self._redecided2 = redecided or triggered
        return obs

    def _maybe_redecide(self, snap: OjamaAccountSnapshot, elapsed_sec: float) -> None:
        """確定済み連鎖合計得点が予測総得点を超えたら下限として即時再決着する。

        `chain_end_triggered_p1/p2` は OjamaAccountingTracker が
        K_SETTLE_FRAMES 連続でscore不変を確認した瞬間だけ True になる
        edge-trigger (1連鎖=1回) — 掛け算式アニメ中の生 score OCR を毎フレーム
        比較すると常に「観測>予測(未完のため)」が成立し続けて再決着が
        乱発するため、確定済み値のみを見る (クラス docstring 参照)。
        各 side、保持セッション中に1回まで (`_redecided1/2`)。2回目以降の
        settle は別の (後続の) 連鎖である可能性が高く、その場合は元の
        before_board を使い回した再決着はむしろ不整合を生むため見送る
        (簡明優先、hold の解除は両側 chain_event が None に戻るのを待つ)。

        [指摘19 根治、2026-08-16] `enable_resolved_victim_gen_live=True` の
        場合、上記「1回きり」は `_redecide_obs` により「chain_end_triggered_pX
        が True の間 0.5秒ごとに追従」へ緩和される (クラス docstring 参照)。
        既定 False では本メソッドの挙動は従来と完全に同一 (bit-identical)。
        """
        obs1 = self._redecide_obs(
            snap.chain_end_triggered_p1, snap.chain_total_score_p1, True, elapsed_sec)
        obs2 = self._redecide_obs(
            snap.chain_end_triggered_p2, snap.chain_total_score_p2, False, elapsed_sec)
        exceeded = (
            (obs1 is not None and obs1 > self._pred_score1)
            or (obs2 is not None and obs2 > self._pred_score2)
        )
        if not exceeded:
            return
        self._resolve(
            snap, elapsed_sec,
            max(self._pred_score1, obs1 or 0.0), max(self._pred_score2, obs2 or 0.0),
            reset_nondefender_cycles=False,
        )

    def _landing_complete(self, r_p1, r_p2, snap: OjamaAccountSnapshot) -> bool:
        """[指摘11] 着弾完了を検知する (どちらかの成立で True)。

        ① 会計の未着弾量が0になった: 既存 `OjamaAccountSnapshot.pending_p1/p2`
           (=forecast_incoming、tsumo 着地の度に drain される実測値) が両者
           とも0以下。最も単純で堅牢な既存シグナル、通常はこちらが先に成立する。
        ② 受け側盤面のおじゃま数が決着計算の予測着弾量に達した: `_resolve`
           時点で保持した `_target_ojama_p{1,2}` に、実際の確定盤面
           (STABLE時のみ信用、CLAUDE.md「STABLE確定盤面のみで評価」原則) の
           `iv.board_ojama_count` が到達したか。①より判定が遅れがちだが
           (confirmed_board は非STABLE中フリーズ)、①が何らかの理由で
           成立しない場合の保険として OR で残す。incoming が0の side は
           「元々受け側でない」ため常に達成扱い。
        """
        if snap.pending_p1 <= 0 and snap.pending_p2 <= 0:
            return True
        ok1 = self._incoming_total_p1 <= 0.0
        ok2 = self._incoming_total_p2 <= 0.0
        if not ok1 and r_p1.state == BoardState.STABLE and r_p1.confirmed_board is not None:
            ok1 = iv.board_ojama_count(r_p1.confirmed_board).raw >= self._target_ojama_p1
        if not ok2 and r_p2.state == BoardState.STABLE and r_p2.confirmed_board is not None:
            ok2 = iv.board_ojama_count(r_p2.confirmed_board).raw >= self._target_ojama_p2
        return ok1 and ok2

    def _release(self) -> "tuple[bool, bool]":
        """ホールドを解除する共通処理 (指摘11、着弾完了/安全弁のどちらでも同じ)。"""
        self._active = False
        self._awaiting_landing = False
        self._landing_wait_started_sec = None
        self._landing_end_kind = None
        return False, True

    def on_game_boundary(self) -> None:
        """正式試合境界で、保持中セッションを物理終端として記録する。"""
        if not self._enable_absolute_chain_end:
            return
        self.abs_end_stats["total_boundaries"] += 1
        if self._active:
            self.abs_end_stats["ended_by_boundary"] += 1

    def _abs_side_inputs(self, r_side, snap, idx: int) -> "tuple":
        """絶対律の入力を返す。

        Returns:
            (next_pair, その side への累積お邪魔着弾量, 物理スライド中か, state名)。

        `getattr` で防御するのは、軽量テストダブル (Signals スタブ) が属性を
        持たない場合があるため。本番の `SideResult` は全て持つ。
        """
        nxt = getattr(r_side, "next_pair", None)
        key = "total_dropped_to_p1" if idx == 0 else "total_dropped_to_p2"
        dropped = float(getattr(snap, key, 0.0) or 0.0)
        slide = bool(getattr(r_side, "next_slide_motion", False) or False)
        st = getattr(r_side, "state", None)
        return nxt, dropped, slide, getattr(st, "name", "") if st is not None else ""

    def _update_abs_baseline(
        self, idx: int, nxt, slide: bool, state: str,
    ) -> None:
        """実CHAIN観測後、開始設置の移動が収まったNEXTを基準化する。"""
        if state == BoardState.CHAIN.name:
            self._abs_saw_chain[idx] = True
        if self._abs_baseline_ready[idx] or not self._abs_saw_chain[idx]:
            return
        if self._t_sec - self._abs_arm_t_sec < RESOLVED_ABS_BASELINE_DELAY_SEC:
            return
        if nxt is None or slide:
            self._abs_baseline_candidate[idx] = None
            self._abs_baseline_frames[idx] = 0
            return
        if nxt != self._abs_baseline_candidate[idx]:
            self._abs_baseline_candidate[idx] = nxt
            self._abs_baseline_frames[idx] = 1
            return
        self._abs_baseline_frames[idx] += 1
        if self._abs_baseline_frames[idx] >= RESOLVED_ABS_BASELINE_CONFIRM_FRAMES:
            self._abs_next_at_start[idx] = nxt
            self._abs_baseline_ready[idx] = True

    def _abs_end_signal(self, idx: int, r_side, snap) -> "str | None":
        """その side に「連鎖が終わった」信号が出ていれば種別名を返す (純判定)。

        user 伝授の絶対律「連鎖の終わり = 連鎖している側のネクストが動いた瞬間
        OR 連鎖している側にお邪魔が落ちた瞬間」を、取りこぼしの無い形で見る:

          - "next":  ネクストの**色ペア値**が変わった (`_is_game_event_chain_exit`)。
                     精度は高いが、次ツモが同じ色ペアだと検出できない (4色で約6.25%)。
          - "slide": NEXT ROI の**物理スライド**。色が同じでも取りこぼさない。
                     ただし連鎖演出の発光で誤検知する既知の事故源なので、
                     必ず連続フレーム確認と併用する (単独の根拠にしない)。
          - "tsumo": 新しいツモの落下開始 (`*→TSUMO_FALL`)。置けた = ネクストが
                     動いた、の物理的な証拠。エッジなのでセッション内でラッチする。
          - "ojama": その side への累積着弾量がセッション開始時より増えた。
        """
        nxt, dropped, slide, state = self._abs_side_inputs(r_side, snap, idx)
        new_tsumo = is_new_tsumo_fall_start(self._abs_prev_state[idx] or "", state)
        self._update_abs_baseline(idx, nxt, slide, state)
        if not self._abs_saw_chain[idx]:
            return None
        last_activity = self._abs_last_chain_activity_sec[idx]
        chain_recent = (
            state == BoardState.CHAIN.name
            and last_activity is not None
            and self._t_sec - last_activity <= RESOLVED_ABS_CHAIN_ACTIVITY_QUIET_SEC
        )
        if chain_recent:
            return None
        if dropped > self._abs_dropped_at_start[idx]:
            return "ojama"
        if not self._abs_baseline_ready[idx]:
            return None
        # NEXT変化は発火させた設置でも起きる。変化自体は開始時基準との比較で
        # ラッチされるため、own state が CHAIN を抜けるまで確定を待てば、終了後に
        # 同じ信号を回収できる。state遅延は解除を遅らせるだけで早めない。
        if new_tsumo:
            self._abs_saw_tsumo[idx] = True
        if _is_game_event_chain_exit(nxt, self._abs_next_at_start[idx]):
            return "next"
        if slide:
            return "slide"
        if self._abs_saw_tsumo[idx]:
            return "tsumo"
        return None

    def _arm_absolute_chain_end(self, r_p1, r_p2, snap) -> None:
        """保持セッション開始時に、絶対律の基準スナップショットを取る。

        以後の判定は「この時点から変わったか」で見るため、いったん成立すれば
        下がらない (ラッチ性がある)。瞬間値の比較ではないので、連鎖アニメ中の
        chain_event 点滅 (W30) に引きずられない。
        """
        self._abs_arm_t_sec = float(self._t_sec)
        for idx, r_side in ((0, r_p1), (1, r_p2)):
            nxt, dropped, _slide, state = self._abs_side_inputs(r_side, snap, idx)
            self._abs_next_at_start[idx] = None
            self._abs_dropped_at_start[idx] = dropped
            self._abs_ended[idx] = False
            self._abs_end_kind[idx] = None
            self._abs_pending_frames[idx] = 0
            self._abs_prev_state[idx] = state
            score = getattr(r_side, "score", None)
            self._abs_prev_score[idx] = None if score is None else int(score)
            self._abs_last_chain_activity_sec[idx] = (
                float(self._t_sec) if state == BoardState.CHAIN.name else None)
            self._abs_saw_tsumo[idx] = False
            self._abs_saw_chain[idx] = state == BoardState.CHAIN.name
            self._abs_baseline_ready[idx] = False
            self._abs_baseline_candidate[idx] = None
            self._abs_baseline_frames[idx] = 0
        self.abs_end_stats["sessions"] += 1

    def _observe_absolute_chain_end(self, r_p1, r_p2, snap) -> bool:
        """毎フレーム呼ぶ。両 side とも絶対律で連鎖が終わっていれば True。

        side ごとの終了条件 (user 伝授の絶対律、どちらか成立で終了):
          A. その side のネクストが動いた (`_is_game_event_chain_exit` を再利用。
             `next_pair=None` は「未検知」であって「不動」ではないため、
             両方が有効値のときだけ比較される)
          B. その side にお邪魔が落ちた (累積着弾量がセッション開始時より増えた)

        単発の誤読で早期解除しないよう、`RESOLVED_ABS_END_CONFIRM_FRAMES`
        フレーム連続で成立してから確定させる。

        長連鎖では段間に一時的な非CHAINが入り、次段で再びCHAINになる。
        その場合、前段末尾で立った終了候補を撤回する。撤回しないと5連鎖と
        6連鎖の段間を物理連鎖の終了と誤認する。
        """
        for idx, r_side in ((0, r_p1), (1, r_p2)):
            state = self._abs_side_inputs(r_side, snap, idx)[3]
            score_value = getattr(r_side, "score", None)
            score = None if score_value is None else int(score_value)
            prev_score = self._abs_prev_score[idx]
            chain_started = (
                state == BoardState.CHAIN.name
                and self._abs_prev_state[idx] != BoardState.CHAIN.name)
            score_step = (
                score is not None and prev_score is not None
                and score - prev_score >= CHAIN_TOTAL_MIN_SCORE)
            physical_activity = chain_started or score_step
            if physical_activity:
                self._abs_last_chain_activity_sec[idx] = float(self._t_sec)
            if self._abs_ended[idx] and physical_activity:
                self._abs_ended[idx] = False
                self._abs_end_kind[idx] = None
                self._abs_pending_frames[idx] = 0
                self.abs_end_stats["reopened_by_chain"][idx] += 1
            sig = self._abs_end_signal(idx, r_side, snap)
            # prev_state は「終了済み」の side でも更新し続ける (次セッションの
            # 立ち上がり判定が前セッションの残骸に引きずられないように)。
            self._abs_prev_state[idx] = state
            self._abs_prev_score[idx] = score
            if self._abs_ended[idx]:
                continue
            if sig is None:
                self._abs_pending_frames[idx] = 0
                continue
            self._abs_pending_frames[idx] += 1
            if self._abs_pending_frames[idx] < RESOLVED_ABS_END_CONFIRM_FRAMES:
                continue
            self._abs_ended[idx] = True
            self._abs_end_kind[idx] = sig
            self.abs_end_stats["end_by_" + sig][idx] += 1
        return self._abs_ended[0] and self._abs_ended[1]

    def _safe_abs_legacy_done(self, r_p1, r_p2, ev1, ev2, snap) -> bool:
        """ON時だけ、断片化Noneを除外した旧終了条件を返す。"""
        if ev1 is not None or ev2 is not None:
            self._abs_legacy_neutral_frames = 0
            return False
        states = (
            self._abs_side_inputs(r_p1, snap, 0)[3],
            self._abs_side_inputs(r_p2, snap, 1)[3],
        )
        if (not all(self._abs_saw_chain)
                or BoardState.CHAIN.name in states):
            self._abs_legacy_neutral_frames = 0
            return False
        self._abs_legacy_neutral_frames += 1
        return self._abs_legacy_neutral_frames >= RESOLVED_ABS_END_CONFIRM_FRAMES

    def _wait_abs_rearm_neutral(self, ev1, ev2) -> bool:
        """両イベントNoneが1手分続けば再武装を許す。待機中はTrue。"""
        if ev1 is not None or ev2 is not None:
            self._abs_rearm_neutral_started_sec = None
            self.abs_end_stats["rearm_blocked_frames"] += 1
            return True
        if self._abs_rearm_neutral_started_sec is None:
            self._abs_rearm_neutral_started_sec = float(self._t_sec)
        waited = float(self._t_sec) - self._abs_rearm_neutral_started_sec
        if waited < RESOLVED_ABS_REARM_NEUTRAL_SEC:
            self.abs_end_stats["rearm_blocked_frames"] += 1
            return True
        self._abs_rearm_blocked = False
        self._abs_rearm_neutral_started_sec = None
        return False

    def abs_end_summary(self) -> str:
        """絶対律による解除の母数付きサマリ (診断用)。

        母数 (保持セッション総数) と必ず並べて出す。`0` が「起きなかった」
        なのか「測っていない」なのかを取り違えないため
        (memory feedback_zero_needs_denominator_2026-08-25)。
        """
        st = self.abs_end_stats
        n = st["sessions"]
        def _pair(key: str) -> str:
            return "1P={} 2P={}".format(st[key][0], st[key][1])

        rel = st["release_states"]
        # 解除の瞬間にどちらかが CHAIN = 連鎖アニメ中に解いた疑い (早期解除)。
        early = [r for r in rel if r[1] == "CHAIN" or r[2] == "CHAIN"]
        return (
            "保持セッション {n} / 絶対律で解除 {a}/{n} / 旧条件で解除 {b}/{n} / "
            "試合境界で終端 {c}/{n} (正式境界 {bc}) / "
            "終了信号[色ペア変化 {s1}] [物理スライド {s2}] [新ツモ落下 {s3}] "
            "[お邪魔着弾 {s4}] / 次段CHAINで撤回 {s5} / "
            "**実解除時にCHAIN継続 {e}/{r}** / "
            "再武装拒否 {q}frame {ex}"
        ).format(
            n=n, a=st["released_by_abs"], b=st["released_by_legacy"],
            c=st["ended_by_boundary"], bc=st["total_boundaries"],
            s1=_pair("end_by_next"), s2=_pair("end_by_slide"),
            s3=_pair("end_by_tsumo"), s4=_pair("end_by_ojama"),
            s5=_pair("reopened_by_chain"),
            e=len(early), r=len(rel), q=st["rearm_blocked_frames"],
            ex=("-> " + str([(r[0], r[1], r[2], r[3]) for r in early[:6]])
                if early else ""),
        )

    def _await_landing(
        self, r_p1, r_p2, snap: OjamaAccountSnapshot, elapsed_sec: float,
    ) -> "tuple[bool, bool]":
        """[指摘11] 両側 chain_event が None化した後の着弾完了待ちフェーズ。

        着弾完了なら解放、未完了でも安全弁 (`RESOLVED_HOLD_LANDING_MAX_WAIT_SEC`)
        を超えたら強制解放する (無限ホールド防止)。
        """
        if not self._awaiting_landing:
            self._awaiting_landing = True
            self._landing_wait_started_sec = elapsed_sec
        release_reason = "landing" if self._landing_complete(r_p1, r_p2, snap) else None
        if release_reason is None:
            waited = elapsed_sec - self._landing_wait_started_sec
            if waited >= RESOLVED_HOLD_LANDING_MAX_WAIT_SEC:
                release_reason = "timeout"
        if release_reason is not None:
            kind = self._landing_end_kind or "unknown"
            self._abs_rearm_blocked = True
            self._abs_rearm_neutral_started_sec = None
            key = "released_by_legacy" if kind == "legacy" else "released_by_abs"
            self.abs_end_stats[key] += 1
            self.abs_end_stats["release_states"].append((
                round(float(self._t_sec), 3),
                self._abs_side_inputs(r_p1, snap, 0)[3],
                self._abs_side_inputs(r_p2, snap, 1)[3],
                f"{kind}:{release_reason}",
            ))
            return self._release()
        return True, False

    def update(
        self, r_p1, r_p2, snap: OjamaAccountSnapshot, elapsed_sec: float,
        t_sec: "float | None" = None,
        b1: "Board | None" = None, b2: "Board | None" = None,
        physical_net_raw: "float | None" = None,
        physical_is_unresolved: bool = False,
        physical_chain_id_p1: "int | None" = None,
        physical_chain_id_p2: "int | None" = None,
    ) -> "tuple[bool, bool]":
        """毎フレーム呼ぶ。(is_active, just_deactivated) を返す

        (決着値そのものは hold_adv/hold_p1/hold_drivers 属性を直接参照する)。

        t_sec: 呼出側の動画内絶対時刻 (raw、`ChainEvent.trigger_sec` と同じ
            時間軸、2026-08-14 指摘12 修正1 で追加の optional 引数)。
            `_amplify_decisive` の時間予算計算 (経過時間控除) 専用で、
            省略時 (None、backwards compat) は `elapsed_sec` にフォール
            バックする — 呼び出し元が新規引数を渡さない既存テスト等は
            従来と同じ挙動 (enable_decisive_amplify=False では未使用のため
            実害なし)。
        b1/b2: [指摘13、2026-08-15 追加の optional 引数] 呼出側 generate() が
            保持する「受け側の現在の STABLE 確定盤面」(sticky、片側STABLE時
            のみ更新・非STABLE中は前回値を保持)。`enable_live_defender_reeval
            =True` かつ受け側が物理的に自由な間だけ `_reevaluate_live_defender` が
            参照する。省略時 (None、backwards compat) はライブ再評価自体が
            盤面欠損として no-op になる (既存呼出元は無変化)。
        snap: 既存の必須引数 (`_maybe_redecide` 用) を `_reevaluate_live_
            defender` にもそのまま渡す (2026-08-15 方向反転修正)。受け側の
            未着弾量を現在の会計から求める (`_live_remaining_incoming`)。
        """
        self._t_sec = elapsed_sec if t_sec is None else t_sec
        ev1, ev2 = r_p1.chain_event, r_p2.chain_event
        if not self._active:
            if self._enable_absolute_chain_end and self._abs_rearm_blocked:
                if self._wait_abs_rearm_neutral(ev1, ev2):
                    return False, False
            # CHAIN_TOTAL_MIN_SCORE 未満 (OCR誤読ノイズ由来の疑いが濃い極小連鎖、
            # ojama_accounting.py と同じ判断基準) はトリガー対象外にする。
            #
            # 根治③ (W7, 2026-08-13, docs/KNOWN_WEAKNESSES.md): 当初案は
            # 「total_score>=40 または simulate検証済み chain_count>=1」の
            # OR ゲートへ拡張する計画だったが、根治① (score_estimated 充填、
            # src/recognition_pipeline.py `_fill_pseudo_chain_score`) を
            # 実装した結果、拡張は不要と判明したため見送った (簡素化)。
            # 根拠: calculate_step_score (src/scoring.py) は消去グループ
            # size>=4 (ぷよ消去の最小単位) でのみステップを生成し、その
            # 最小得点は 4×10×max(1,0)=40=CHAIN_TOTAL_MIN_SCORE と厳密に
            # 一致する。つまり chain_count>=1 の simulate 検証済み結果は
            # 必ず total_score>=40 になり、既存ゲートを素通しで満たす
            # (tests/test_scoring.py の不変条件テストで固定)。
            # score_estimated=False かつ total_score=0 (根治①未実装/OFF時の
            # 旧来ハードコード値、または simulate 失敗時の fail-safe) だけが
            # 引き続き「スコア未計算」としてノイズゲート対象になる。
            if (ev1 is not None and ev2 is not None
                    and ev1.total_score >= CHAIN_TOTAL_MIN_SCORE
                    and ev2.total_score >= CHAIN_TOTAL_MIN_SCORE):
                self._ev1, self._ev2 = ev1, ev2
                self._redecided1 = self._redecided2 = False
                # [指摘19 根治] 新しい保持セッション開始、0.5秒間引きタイマも
                # リセットする (enable_resolved_victim_gen_live=False では
                # 未使用のため実害なし)。
                self._victim_live_last_t1 = self._victim_live_last_t2 = None
                self._episode_physical_net_last = None
                self._awaiting_landing = False
                self._landing_wait_started_sec = None
                self._landing_end_kind = None
                self._resolve(
                    snap, elapsed_sec, float(ev1.total_score), float(ev2.total_score))
                # この決着を構成した物理連鎖IDを基準として保存する。状態機械の
                # TSUMO/STABLE揺れで同じ連鎖が再びCHAINへ見えても、新しい応手
                # として二重に補正しないための識別子。省略時は安全側で補正しない。
                self._resolved_root_chain_id = [
                    physical_chain_id_p1, physical_chain_id_p2]
                self._active = True
                # [2026-08-26 決着ホールド根治] 既定 OFF では呼ばない
                # (状態も統計も一切動かないため bit-identical)。
                if self._enable_absolute_chain_end:
                    self._arm_absolute_chain_end(r_p1, r_p2, snap)
                    self._abs_legacy_neutral_frames = 0
            return self._active, False
        self._maybe_redecide(snap, elapsed_sec)
        self._maybe_redecide_physical_net(
            snap, physical_net_raw, physical_is_unresolved)
        # [2026-08-26 決着ホールド根治、user決定] 絶対律による終了判定。
        # 既定 OFF では `_abs_done` は常に False で、下の条件式は従来と同一。
        _abs_done = False
        if self._enable_absolute_chain_end:
            _abs_done = self._observe_absolute_chain_end(r_p1, r_p2, snap)
        _legacy_done = (
            self._safe_abs_legacy_done(r_p1, r_p2, ev1, ev2, snap)
            if self._enable_absolute_chain_end
            else ev1 is None and ev2 is None)
        # ON時の旧条件は、瞬間的なNone gapを除外した安全版だけをORする。
        # OFF時は従来の `ev1 is None and ev2 is None` と完全に同一。
        if _legacy_done or _abs_done:
            # [指摘11] 連鎖アニメは終わったが、相殺後おじゃまの着弾完了までは
            # ホールドを延長する (「着弾前の空白」で通常評価に戻さない)。
            if self._enable_absolute_chain_end and not self._awaiting_landing:
                self._landing_end_kind = "legacy" if _legacy_done else "abs"
                # これは実解除でなく、着弾待ちへ入る終了候補。次段 CHAIN が
                # 見えた場合は下の継続分岐へ戻り、候補を撤回する。
                self.abs_end_stats["end_candidate_states"].append((
                    round(float(self._t_sec), 3),
                    self._abs_side_inputs(r_p1, snap, 0)[3],
                    self._abs_side_inputs(r_p2, snap, 1)[3],
                    self._landing_end_kind,
                ))
            return self._await_landing(r_p1, r_p2, snap, elapsed_sec)
        # 連鎖アニメがまだ続いている (片側だけ終わった場合も含む、検収指摘⑤)。
        self._awaiting_landing = False
        self._landing_wait_started_sec = None
        self._landing_end_kind = None
        should_live_reevaluate = (
            (ev1 is None) != (ev2 is None)
            or self._enable_live_defender_strict)
        if self._enable_live_defender_reeval and should_live_reevaluate:
            # [指摘13 + 2026-08-27 userレビュー] chain_event は保持パルスであり、
            # 受け側が既に自由でも両側に古い event が残ることがある。strict ON
            # では XOR を入口にせず、受け側の物理 state が CHAIN/
            # GRAVITY_SETTLE かを `_reevaluate_live_defender` 内で直接判定する。
            # strict OFF は従来の XOR 条件を維持して後方互換を保つ。
            # getattr フォールバック (state 属性を持たない軽量テスト
            # ダブルとの後方互換用。本番の Signals は必ず state を持つ)。
            self._reevaluate_live_defender(
                b1, b2, snap,
                state1=getattr(r_p1, "state", None), state2=getattr(r_p2, "state", None),
                score1=getattr(r_p1, "score", None), score2=getattr(r_p2, "score", None),
                event1=ev1, event2=ev2,
                chain_id1=physical_chain_id_p1,
                chain_id2=physical_chain_id_p2)
        return True, False


def resolved_hold_freezes_settled(
    enable_resolved_exchange_eval: bool, resolved_active: bool,
) -> bool:
    """決着ホールド中は per_side_settled の片側 STABLE 判定を素通りさせない
    (検収指摘⑤、2026-08-14)。

    `ResolvedExchangeTracker.update` 自体は「両側 chain_event が None」まで
    正しく hold を維持する (`ev1 is None and ev2 is None` の AND ゲート)。
    だが呼出側 (main ループ) の `enable_per_side_settled` は片側だけ STABLE
    になった瞬間に settled=True へ OR で倒す仕様のため、片方の連鎖が先に
    終わった瞬間 (chain_event が None化 = 状態が STABLE に戻る瞬間) に
    settled 再計算が起動し、「片方は最新盤面・もう片方は連鎖前の凍結盤面」
    という不整合ペアで内部 EMA (adv_ema/p1_last) が汚染される。 hold 解除
    直後にその汚染が漏れて表示値が跳ぶ (「部分解放」に見える実体)。
    True を返したら呼出側は settled を強制的に False に上書きする
    (= hold 中は内部状態も含め完全凍結、両側 chain_event が None に戻り
    tracker が deactivate するまで settled 再計算そのものを止める)。
    `enable_resolved_exchange_eval` 無効時は常に False (従来挙動と完全一致)。
    """
    return enable_resolved_exchange_eval and resolved_active


def _minimum_prediction_guard_applies(
    tracker: ResolvedExchangeTracker, state1: BoardState, state2: BoardState,
    stable_adv: float,
) -> bool:
    """40点下限予測だけで直前評価を極端に反転する場合に限り保留する。"""
    return (
        tracker.has_untrusted_minimum_active_chain(state1, state2)
        and abs(tracker.hold_adv) > EPISODE_UNRESOLVED_ABS_CAP
        and tracker.hold_adv * stable_adv < 0.0
    )


# (改修1) スコアリセット検知: 新ゲーム開始/全消し等でスコアが「前フレームから
#   大幅減少」または「両者ほぼ0」に戻ったら試合境界とみなし、凍結盤面(b1/b2)や
#   各種持続トラッカーを全て初期化する。空盤面(スコア0)なのに前試合の非空盤面
#   差分(例: 最大列高差)を表示し続ける「幻の差」バグの根治用。
#   drop 側の閾値は OjamaAccountingTracker が内部で使う既存定数を流用し重複させない。
SCORE_NEAR_ZERO_THRESHOLD = 20  # 両者スコアがこれ以下なら「0付近」とみなす(OCRノイズ許容)
# タイムラインdump (2026-08-11 追加) の game_idx 用デバウンス。スコアが0付近に
# 留まる間 _detect_score_reset は毎フレーム True になりうるため、それを都度
# game_idx += 1 すると境界1回につき数十〜数百回進んでしまう。実試合は最短でも
# 14秒あるため、直前の進行から5秒未満は再進行させない
# (scripts/collect_boards_lean.py の GAME_BOUNDARY_DEBOUNCE_SEC と同値、
# 意図的に同じ値を保つ。import で結合させず定数を独立定義するに留める理由は
# collect_boards_lean.py が別用途の重い収集スクリプトであるため)。
GAME_BOUNDARY_DEBOUNCE_SEC: float = 5.0


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


def accept_formal_boundary(
    reset_now: bool, latched: bool, t_sec: float,
    last_formal_t: float | None,
    debounce_sec: float = GAME_BOUNDARY_DEBOUNCE_SEC,
) -> bool:
    """この reset 信号を「正式な試合境界イベント」として受理するか (純関数)。

    【P1 是正 2026-08-26、Codex 第26報レビュー】`_detect_score_reset()` は
    両スコアが `SCORE_NEAR_ZERO_THRESHOLD` 以下の間**毎フレーム True** を返す。
    従来は `game_idx += 1` だけが debounce されており、死亡確定の境界処理
    (`resolve_boundary_confirmations`) は debounce の外にあったため、実境界
    約6件に対し `total_boundaries=715` 回呼ばれていた (Codex 実測)。
    `DeathConfirmTracker.on_game_boundary()` は毎回 `_post_boundary_armed`
    を False へ戻すので、新試合冒頭の再武装や死亡候補を取りこぼしうる。

    受理条件は次の両方。片方だけでは不足する:
      1. reset 信号の**立ち上がり** (`latched` が False)。低得点が続く間の
         毎フレーム再受理を防ぐ。時間ではなく信号の縁で切るので、低得点が
         5秒より長く続いても同一境界を再受理しない。
      2. 前回受理から `debounce_sec` 秒以上経過。OCR ちらつきで信号が一瞬
         落ちて再度立ち上がる場合に、同一境界を2回受理しないための保険。

    実試合は最短でも14秒あるため、この二重条件で真の境界は落とさない。

    Args:
        reset_now: このフレームの `_detect_score_reset()` の結果。
        latched: 直前フレームで reset 信号が立っていたか
            (`update_score_reset_latch()` の戻り値を持ち回る)。
        t_sec: 現在時刻 (秒)。
        last_formal_t: 最後に正式受理した境界の時刻。未受理なら None。
        debounce_sec: 再受理を抑制する秒数。既定は `GAME_BOUNDARY_DEBOUNCE_SEC`。

    Returns:
        受理するなら True。
    """
    if not reset_now or latched:
        return False
    return last_formal_t is None or (t_sec - last_formal_t) >= debounce_sec


def update_score_reset_latch(
    latched: bool, reset_now: bool, score1: int | None, score2: int | None,
) -> bool:
    """reset 信号の立ち上がりラッチを更新する (純関数)。

    `_detect_score_reset()` は score が None (OCR 失敗) のとき False を返すが、
    それは「境界が終わった」ではなく「**判定不能**」である。ここでラッチを
    解除すると、OCR の瞬断のたびに同じ境界を再受理してしまう
    (`0` が「合格」なのか「測っていない」なのかを取り違える誤りと同型)。
    そのため、両者のスコアが実際に読めていて、かつ reset 条件を満たさなく
    なったときだけ解除する。

    Args:
        latched: 現在のラッチ状態。
        reset_now: このフレームの `_detect_score_reset()` の結果。
        score1: 1P のスコア。None は OCR 失敗。
        score2: 2P のスコア。None は OCR 失敗。

    Returns:
        更新後のラッチ状態。
    """
    if reset_now:
        return True
    if score1 is None or score2 is None:
        return latched
    return False

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
# center_bulge (中央凸度、2026-08-12 壁打ちuser仕様) は新指標のため FEATURES
# 定数そのものは変更せず (直接 import する既存呼出元への影響回避、後方互換
# 維持)、候補一覧 FEATURE_CANDIDATES の末尾に追加する。labeled_win.csv に
# 列が無い間は _resolve_features() の列存在ガードで自動的に除外され、
# 収集後に自動有効化される (saturated_chain_count 等と同じ方式)。
CENTER_BULGE_COL: str = "center_bulge"
# saturated_chain_count は a-1決定 (2026-08-12、src.production_config.
# INDICATOR_REORG_DECISIONS 参照) で削除確定済みだが、本タプルへの反映が
# 漏れていた (build_labeled_win_from_npz.py にしか反映されていなかった、
# docs/CROSS_CUTTING_AUDIT_2026-08-13.md P2)。2026-08-13 是正: 削除台帳
# `src.production_config.REORG_REMOVED_INDICATORS` を単一情報源とし、末尾の
# フィルタで機械的に除外する (台帳を更新するだけで本タプルも自動追従する)。
FEATURE_CANDIDATES: tuple[str, ...] = tuple(
    c for c in (
        FEATURES + (
            "saturated_chain_count", "ukeyasusa", "sub_chain_count",
        ) + NEAR_FUTURE_FIRE_COLS + FIRE_STABILITY_COLS + EXPECTED_FIRE_COLS + (
            CENTER_BULGE_COL,
        )
    )
    if c not in reorg_removed_indicator_names()
)
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
    "center_bulge": "中央凸度",
    # --- 評価済みモデル成果物 (47列、2026-08-14) 用の追加ラベル ---
    "center_bulge_color": "中央凸度(色)", "center_bulge_ojama": "中央凸度(おじゃま)",
    "color_diversity_evenness": "色多様性均等度", "buried_hole_count": "埋没穴数",
    "immediate_fire_power": "即時火力", "chain_efficiency": "連鎖効率",
    "min_puyos_to_ignite": "発火最小ぷよ数", "second_chain_potential": "副砲潜在力",
    "main_linked_pair_count": "本線連結対数", "isolated_pair_count": "孤立対数",
    "main_linked_ratio": "本線連結比率", "ignition_point_count": "発火点数",
    "multi_color_ignition": "多色発火性", "simultaneous_pop_richness": "同時消し豊富度",
    "saturation_chain_upper": "飽和連鎖(上限探索)",
    "chain_articulation_point_count": "連鎖関節点数",
    "conn_max_group_size": "最大連結サイズ",
    "all_clear_bonus_pending": "全消しボーナス予約中",
    "opp_all_clear_bonus_pending": "相手全消しボーナス予約中",
    "ojama_net_balance_synced": "おじゃま収支(再同期)", "ojama_margin": "おじゃま猶予量",
    "color_ojama_ratio_own": "色ぷよ比率", "color_diff_x_ojama_diff": "色ぷよ差×おじゃま差",
    "diff_max_column_height": "最大列高差", "diff_column_bumpiness": "凹凸差",
    "diff_death_margin": "窒息余裕差", "diff_death_margin_neighbor": "窒息余裕差(隣接)",
    "diff_conn_pair_count": "連結対数差", "diff_conn_max_group_size": "最大連結サイズ差",
    "diff_board_color_puyo_total": "色ぷよ総数差", "diff_board_puyo_total": "盤面ぷよ総数差",
    "diff_board_ojama_count": "盤面お邪魔数差", "diff_center_bulge_color": "中央凸度差(色)",
    "diff_center_bulge_ojama": "中央凸度差(おじゃま)", "diff_current_max_chain": "現在最大連鎖差",
    "diff_dig_resistance": "掘り耐性差", "diff_ukeyasusa": "受けやすさ差",
    "diff_sub_chain_count": "副砲連鎖数差",
    # --- 主因表示の合成キー (2026-08-22 修正④、実指標ではなく表示専用) ---
    "kill_override_p1_lethal": "致死判定(1P着弾見込)",
    "kill_override_p2_lethal": "致死判定(2P着弾見込)",
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
# れんさ数表示 (--show-chain-count、2026-08-15 追加、既定 OFF)。
# 応手情報行 (PANEL_INFO_COUNTER_Y) の下、経過時刻行 (下端 h-50) より十分上。
PANEL_INFO_CHAIN_Y1 = PANEL_INFO_COUNTER_Y + 30   # 1P れんさ表示行
PANEL_INFO_CHAIN_Y2 = PANEL_INFO_CHAIN_Y1 + PANEL_INFO_DRIVER_LINE_H  # 2P れんさ表示行
# 推定連鎖数(simulate)と得点逆算連鎖数が食い違う場合に強調する警告色 (橙)。
# 認識性能検証の主目的 (食い違いを目立たせる) のため、通常の 1P/2P 色より
# 明らかに異なる色調にする。
PANEL_CHAIN_MISMATCH_COLOR = (255, 140, 0)
PANEL_INFO_ELAPSED_BOTTOM_MARGIN = 50  # 経過時刻行 (情報パネル下端からの距離)


def panel_layout_regions(
    subtitle_h: int = PANEL_SUBTITLE_H,
) -> dict[str, tuple[int, int, int, int]]:
    """パネルレイアウトの4領域 (video/graph/info/subtitle) を (x0, y0, w, h) で返す。

    stateless な純関数 (座標計算のみ、副作用なし)。generate() と
    _draw_panel_layout() の両方が本関数を参照することで、座標がずれる
    バグ (二重管理) を構造的に防ぐ。4領域は 1920x1080 を隙間・重複なく分割する
    (video+graph の左列と info の右列が上部コンテンツ高を占め、下端は
    subtitle が全幅で占める)。

    subtitle_h: 下端字幕帯の高さ (2026-08-21 user指示「グラフ広げて」で追加)。
        既定 PANEL_SUBTITLE_H (140) = 従来と完全一致 (backwards compat)。
        0 を渡すと字幕帯を無くし、その分をグラフ (左下) と情報パネル (右)
        の高さへ丸ごと回す。PANEL_CONTENT_H/PANEL_GRAPH_H の定数値自体は
        変更せず、本関数内で subtitle_h から動的に導出する
        (定数は「既定呼び出し時の従来値」を表す資料値として残す)。
    """
    content_h = PANEL_CANVAS_H - subtitle_h  # 字幕帯を除いた上部コンテンツ高
    graph_h = content_h - PANEL_VIDEO_H      # グラフはコンテンツ高から映像分を引いた残り
    return {
        "video": (0, 0, PANEL_VIDEO_W, PANEL_VIDEO_H),
        "graph": (0, PANEL_VIDEO_H, PANEL_VIDEO_W, graph_h),
        "info": (PANEL_VIDEO_W, 0, PANEL_INFO_W, content_h),
        "subtitle": (0, content_h, PANEL_CANVAS_W, subtitle_h),
    }


@functools.lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.ImageFont:
    """meiryo を取得 (無ければ default)。

    2026-08-21 速度改善: `size` のみに依存する純関数のため `lru_cache` で
    包む。改修前は 1 フレームあたり平均 12.5 回 (60秒区間で 22,419 回) 呼ばれ、
    毎回 `ImageFont.truetype` がディスクからフォントファイルを再読込していた
    (実測: フォント読み込みだけで 124〜131 秒 / 全体 294〜318 秒)。呼出し側
    (本ファイル内 `_font(...)` の全呼出し箇所) が渡す size は全て整数リテラル
    (`{15, 16, 18, 20, 22, 24, 26, 52}` の 8 種、`_draw_bar` の
    label_font_size/verdict_font_size 引数経由の値も含む) で種類数は有限のため
    maxsize=32 (実測種類数の4倍の余裕) でメモリ増大の心配なく全件キャッシュ
    できる。フォント不在時の fallback (`ImageFont.load_default()`) も
    size ごとに同じ戻り値を返す純関数のままなのでキャッシュ対象として安全
    (2 回目以降のディスク存在チェックも省略される、後方互換: 戻り値は
    キャッシュ前と bit-identical)。
    """
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
# 2026-08-14 正式切替: 148本フル・2026-08-14学習し直し結果 (AUC 0.657
# /終盤0.839、scripts/_retrain148_2026-08-14.py) を受けて、暫定 light63
# (2026-08-12、63本のみ・仮特徴量セット) から正式昇格させる。
#   旧 (暫定light63): data/verify/npz_light_smoke_2026-08-12/labeled_win_light63.csv
#   旧旧 (66本): data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv
#
# [重要・既知の列名不一致に関する注記 (2026-08-14 コーダ実装時検証)]
# _retrain148_2026-08-14.py が報告した AUC 0.657 は「CSVの47列(非meta数値列)を
# そのまま行単位特徴量として使う」独自パイプラインの数値であり、本モジュールの
# _train_model()/_resolve_features() (FEATURE_CANDIDATES の裸の列名一致 →
# pair_sides_for_win → build_features で 1P-2P 差分を作る方式) とは別物。
# npz→CSV変換ツールは b-2 決定 (2026-08-12 実測) により
# max_column_height/column_bumpiness/death_margin/death_margin_neighbor/
# conn_pair_count/conn_max_group_size を「own列」ではなく「diff_own列」の
# 形でのみ出力するため (scripts/build_labeled_win_from_npz.py の
# DIFF_REPLACE_OWN_COLUMNS 参照)、本モジュールの列存在ガードはこれらの
# base 列名を見つけられず学習から除外する (center_bulge も
# center_bulge_color/_ojama に分解済みで同様に見つからない)。
# 実際に _resolve_features() で解決される列は 9 個 (旧 light63 の 9 個とは
# 中身が入れ替わり: 新規に current_max_chain/ojama_net_balance/
# ojama_forecast/dig_resistance/ukeyasusa/sub_chain_count を獲得する一方、
# max_column_height/column_bumpiness/death_margin/death_margin_neighbor/
# conn_pair_count/center_bulge を失う)。「47列を自動で拾う」という期待は
# 現状のこのパイプラインでは成立しないため、diff_ プレフィックス列の
# 取り込みは別途フォローアップとして検討する (このコメントと合わせて
# _train_model() 内のスキップログで実際の解決列を都度確認できる)。
TRAIN_CSV_PATH: str = (
    "data/verify/labeled_win_full148_2026-08-14/labeled_win_full148.csv"
)
# [2026-08-14 追記] 上記コメントで報告した「AUC 0.657/47列との乖離」は
# 下記 MODEL_ARTIFACT_PATH 経由の直読みで解消済み (coordinator指示)。
# TRAIN_CSV_PATH は artifact 不在時のフォールバック学習でのみ使われる
# (_acquire_model 参照)。

# ============================
# 評価済みモデル成果物の直読み (2026-08-14 coordinator指示)
# ============================
# 「評価したモデル (AUC 0.657/終盤0.839, scripts/_retrain148_2026-08-14.py)
# = デモが使うモデル」を構造的に一致させる。従来の起動時学習 (_train_model,
# 1P/2P差分9列・109秒) は評価と別パイプラインだったため、評価済み成果物
# (全144動画・47列・row単位=side単独で勝率を出す方式) を直接ロードする。
# 特徴量列リストは成果物と同じディレクトリの feature_cols_full.json
# (_retrain148_2026-08-14.py が既に保存済み、47列) を単一情報源として使う。
MODEL_ARTIFACT_DIR: Path = Path("data/verify/retrain148_2026-08-14")
MODEL_ARTIFACT_PATH: Path = MODEL_ARTIFACT_DIR / "model_full148_full_features.joblib"
MODEL_ARTIFACT_FEATURE_COLS_PATH: Path = MODEL_ARTIFACT_DIR / "feature_cols_full.json"

# 評価成果物モデルが使う grid-only 指標 (Board -> IndicatorV2Value)。
# scripts/build_labeled_win_from_npz.py の GRID_ONLY_INDICATORS +
# GRID_ONLY_HEAVY_INDICATORS と同一の関数集合 (薄い委譲構造を踏襲、
# indicators_v2.py への委譲のみで新規ロジックを持たない)。native分岐は
# 使わず常に既存Python実装を直接呼ぶ (ライブ推論では正確性を優先、
# native/Python parityは別テストで担保済みのためこの選択で値は変わらない)。
FULL_MODEL_GRID_REGISTRY: dict[str, Callable[[Board], "iv.IndicatorV2Value"]] = {
    "board_color_puyo_total": iv.board_color_puyo_total,
    "board_puyo_total": iv.board_puyo_total,
    "max_column_height": iv.max_column_height,
    "column_bumpiness": iv.column_bumpiness,
    "death_margin": iv.death_margin,
    "death_margin_neighbor": iv.death_margin_neighbor,
    "center_bulge_color": iv.center_bulge_color,
    "center_bulge_ojama": iv.center_bulge_ojama,
    "board_ojama_count": iv.board_ojama_count,
    "color_diversity_evenness": iv.color_diversity_evenness,
    "buried_hole_count": iv.buried_hole_count,
    "current_max_chain": iv.current_max_chain,
    "dig_resistance": iv.dig_resistance,
    "ukeyasusa": iv.ukeyasusa,
    "sub_chain_count": iv.sub_chain_count,
    "immediate_fire_power": iv.immediate_fire_power,
    "chain_efficiency": iv.chain_efficiency,
    "min_puyos_to_ignite": iv.min_puyos_to_ignite,
    "second_chain_potential": iv.second_chain_potential,
    "main_linked_pair_count": iv.main_linked_pair_count,
    "isolated_pair_count": iv.isolated_pair_count,
    "main_linked_ratio": iv.main_linked_ratio,
    "ignition_point_count": iv.ignition_point_count,
    "multi_color_ignition": iv.multi_color_ignition,
    "simultaneous_pop_richness": iv.simultaneous_pop_richness,
    "saturation_chain_upper": iv.saturation_chain_upper,
    "chain_articulation_point_count": iv.chain_articulation_point_count,
}


def _side_feats_full_base(board: Board) -> dict[str, float]:
    """評価成果物モデル用の grid-only 指標 (27種+conn3種=30列) を1回で計算する。"""
    row = {name: fn(board).score for name, fn in FULL_MODEL_GRID_REGISTRY.items()}
    total_conn, _ = iv.connectivity_observation(board)
    row["conn_pair_count"] = float(total_conn.pair_count)
    row["conn_triple_count"] = float(total_conn.triple_count)
    row["conn_max_group_size"] = float(total_conn.max_group_size)
    return row


def _side_feats_full(
    self_base: dict[str, float], opp_base: dict[str, float],
    net: int, forecast: int,
) -> dict[str, float]:
    """1 side ぶんの47列特徴量を組み立てる (評価成果物モデルと同一スキーマ)。

    diff_*/own+diff の分類は build_labeled_win_from_npz.py の
    DIFF_REPLACE_OWN_COLUMNS 等 (import 済み、単一情報源) と完全一致させる。
    ojama_net_balance_synced/all_clear_bonus_pending 系は permutation
    importance 実測 (rank30以降、importance<=0.0002、
    data/verify/retrain148_2026-08-14/permutation_importance_full.csv) で
    予測寄与がほぼ0と確認済みのため、ライブでは簡略値で代替する
    (詳細下記コメント。将来 VideoChainTracker.all_clear_pending 配線で
    厳密化する余地は残すが、重要度が低く今回は見送り)。
    """
    feat = dict(self_base)
    diff_targets = (
        DIFF_REPLACE_OWN_COLUMNS + DIFF_KEEP_OWN_PAIR_COLUMNS
        + DIFF_KEEP_OWN_NEW_COLUMNS + DIFF_KEEP_OWN_HEAVY_COLUMNS
    )
    for c in diff_targets:
        feat[f"diff_{c}"] = self_base[c] - opp_base[c]
    for c in DIFF_REPLACE_OWN_COLUMNS:
        feat.pop(c, None)  # own→diff完全置換 (b-2決定、own列はCSVに乗せない)
    feat["ojama_net_balance"] = iv.ojama_net_balance(net).score
    feat["ojama_forecast"] = iv.ojama_forecast(forecast).score
    # 簡略化 (rank31, importance=0.0001): オフライン版は両側の直近確定値を
    # merge_asofして平均する再同期版だが、ライブでは常に同一フレームの両盤面
    # が既に同期済みのため ojama_net_balance と同値で代替する。
    feat["ojama_net_balance_synced"] = feat["ojama_net_balance"]
    absorption_raw = iv.ON_FIELD_CAP - self_base["board_puyo_total"] * iv.ON_FIELD_CAP
    margin_raw = absorption_raw - max(0.0, float(forecast))
    feat["ojama_margin"] = iv.ojama_net_balance(margin_raw).score
    # 簡略化 (rank30/40、importance<=0.0001): 全消しボーナス予約中フラグは
    # 常に0固定 (VideoChainTracker.all_clear_pending 配線は将来課題、
    # 重要度が低く現時点では見送り)。
    feat["all_clear_bonus_pending"] = 0.0
    feat["opp_all_clear_bonus_pending"] = 0.0
    color, ojama = feat["board_color_puyo_total"], feat["board_ojama_count"]
    feat["color_ojama_ratio_own"] = color / (color + ojama + COLOR_OJAMA_RATIO_EPS)
    feat["color_diff_x_ojama_diff"] = (
        feat["diff_board_color_puyo_total"] * feat["diff_board_ojama_count"]
    )
    return feat


class ModelArtifactMissingError(RuntimeError):
    """--model-dir 明示指定時、モデル成果物ファイルが見つからない/ロードできない場合に送出。

    (2026-08-18 追加) 既定ディレクトリ (MODEL_ARTIFACT_DIR) 使用時は従来通り
    fail-safe (None を返して CSV 起動時学習にフォールバック) を維持するが、
    ユーザーが明示的に `--model-dir` で別モデルを指定した場合は
    「無言で古いモデルにフォールバックする」事故 (fail-silent) を防ぐため、
    ここで即座に例外化する。
    """


class ModelArtifactFeatureMismatchError(RuntimeError):
    """モデルが学習時に期待する特徴量数と feature_cols_full.json の列数が
    食い違う場合に送出 (2026-08-18 追加)。model_dir 指定の有無に関わらず、
    この不一致は成果物ディレクトリの取り違え/破損を示す実データ異常のため
    常にエラーにする (数値だけで採否を決めず、まず測定器=成果物ペアの
    整合性を確認する原則)。
    """


def _load_artifact_model(model_dir: Path | None = None, *, strict: bool = False):
    """評価済みモデル成果物をロードする (2026-08-14 coordinator指示、
    2026-08-18 model_dir/strict 追加)。

    model_dir: 省略時 (None) は従来通り module 定数 MODEL_ARTIFACT_PATH /
        MODEL_ARTIFACT_FEATURE_COLS_PATH をそのまま使う (既存呼出元・既存
        テストの monkeypatch と完全互換、後方互換必須)。指定時はそのディレ
        クトリ配下の同名ファイル (model_full148_full_features.joblib /
        feature_cols_full.json) を使う (`--model-dir`)。
    strict: True の場合、ファイル欠如・ロード失敗・特徴量列不一致を
        fail-silent フォールバックせず即座に例外を送出する
        (`--model-dir` 明示指定時のみ `_acquire_model` が True で呼ぶ)。
        False (既定) では従来通り None を返し、呼出元 `_acquire_model` が
        起動時学習へフォールバックする。

    ロードしたモデルには `_puyo_feature_mode="full_row"` を付与し、
    `_score_advantage` がこの属性を見て行単位 (side単独) 推論に分岐する
    (`_score_advantage_full_row` 参照、既存の diff ベース経路は無変更)。
    """
    if model_dir is None:
        model_path, cols_path = MODEL_ARTIFACT_PATH, MODEL_ARTIFACT_FEATURE_COLS_PATH
    else:
        model_path = model_dir / MODEL_ARTIFACT_PATH.name
        cols_path = model_dir / MODEL_ARTIFACT_FEATURE_COLS_PATH.name
    if not model_path.exists() or not cols_path.exists():
        if strict:
            missing = [str(p) for p in (model_path, cols_path) if not p.exists()]
            raise ModelArtifactMissingError(
                f"--model-dir で指定されたモデル成果物が見つかりません: {missing}"
                " (フォールバックせずに停止します)"
            )
        return None
    try:
        import joblib
        model = joblib.load(model_path)
        cols = json.loads(cols_path.read_text(encoding="utf-8"))
    except (ImportError, OSError, ValueError) as e:
        if strict:
            raise ModelArtifactMissingError(
                f"--model-dir 指定先のモデルロードに失敗しました"
                f" (model={model_path}, cols={cols_path}): {e!r}"
            ) from e
        print(f"[model] 成果物ロード失敗 ({e!r})。CSV起動時学習にフォールバックします。")
        return None
    cols = [str(c) for c in cols]
    n_expected = getattr(model, "n_features_in_", None)
    if n_expected is not None and n_expected != len(cols):
        raise ModelArtifactFeatureMismatchError(
            f"モデルが期待する特徴量数 ({n_expected}) と {cols_path} の列数"
            f" ({len(cols)}) が不一致です (model={model_path})"
        )
    model._puyo_feature_mode = "full_row"
    model._puyo_full_cols = cols
    print(f"[model] 評価済み成果物をロード: {model_path} (列数={len(cols)})")
    return model


def _acquire_model(exclude_video: str | None = None, model_dir: Path | None = None):
    """モデルを確保する: 評価済み成果物を優先ロードし、無ければ従来の起動時
    学習 (`_train_model`) にフォールバックする (2026-08-14 coordinator指示、
    2026-08-18 model_dir 追加)。

    exclude_video (動画リーク防止用) 指定時は成果物 (全144動画で学習済み・
    除外不可) を使わず、必ず CSV起動時学習にフォールバックする
    (黙って全動画学習済みモデルを返すサイレント不整合を防ぐ、fail-silent警戒)。
    model_dir より exclude_video を優先する (リーク防止が最優先)。

    model_dir: 省略時 (None) は従来通り既定ディレクトリ (MODEL_ARTIFACT_DIR)
        を fail-safe (欠如時は CSV 起動時学習へフォールバック) でロードする
        (既存動作・後方互換、既存呼出元は挙動不変)。指定時 (`--model-dir`)
        はそのディレクトリの成果物のみを使い、欠如/列不一致は
        ModelArtifactMissingError / ModelArtifactFeatureMismatchError を
        送出して即座に停止する (フォールバックしない、fail-silent 厳禁)。
    """
    if exclude_video is not None:
        if model_dir is not None:
            print(f"[model] exclude_video={exclude_video!r} 指定のため --model-dir"
                  f" ({model_dir}) は無視し CSV起動時学習にフォールバックします"
                  " (動画リーク防止を優先)。")
        print(f"[model] exclude_video={exclude_video!r} 指定のため成果物"
              f" (全144動画で学習済み・除外不可) を使わず CSV起動時学習にフォールバックします。")
        return _train_model(exclude_video)
    if model_dir is not None:
        return _load_artifact_model(model_dir, strict=True)
    model = _load_artifact_model()
    if model is not None:
        return model
    print(f"[model] 成果物未検出 ({MODEL_ARTIFACT_PATH})。CSV起動時学習にフォールバックします。")
    return _train_model(exclude_video)


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

# [指摘12 修正2, 2026-08-14] 決着ホールドの決定度増幅 (ResolvedExchangeTracker.
# _amplify_decisive) 専用の強度定数。
# 従来は本増幅がライブ per-frame 用 COUNTER_SCALE (=40.0) をそのまま**全量**
# adv に加算していたが、ライブ per-frame の同種成分 (打ち合い応手 counter_adv)
# は 4成分ブレンド式 `adv = W_PRESSURE*pres + W_FORECAST*fc + W_MODEL*model_adv
# + W_THREAT*threat + W_COUNTER*counter_adv` で W_COUNTER(=0.20) 倍に抑えて
# から合成される — つまりライブ経路での実効上限は COUNTER_SCALE*W_COUNTER=8.0。
# 決着増幅だけがこの重み付けを経ずに COUNTER_SCALE を素通しで加算していたため、
# モデルの評価 (既に着弾後仮想盤面のおじゃま量/連結崩壊等を織り込み済み) の上に
# 同じ機構を実効5倍で二重計上していたのが 指摘12 (84%→97%飽和) の増幅側の
# 根因。既存定数 (COUNTER_SCALE, W_COUNTER) の積のみで構成し、シーンからの
# 逆算では決めない (feedback_overfitting_awareness_2026-08-04 準拠、
# 新規マジックナンバー無し)。
RESOLVED_AMPLIFY_SCALE: float = COUNTER_SCALE * W_COUNTER

# 再計算の時間間引き間隔 [秒] (2026-08-12 追加)。
# `_reach` は1回0.35〜数秒かかる MC 計算で、打ち合い場面では毎フレーム
# (相手盤面が変わる度) キャッシュキーが変わり毎秒複数回走っていた
# = デモ生成が遅い主犯。 表示更新はそもそも「1秒に2〜3回」の粒度で
# 十分 (公開仕様) であり、 `HeavyAdvCache every=9 (~0.3s@30fps)` と同じ
# 思想で 0.5 秒間引きを入れる。
COUNTER_RECOMPUTE_INTERVAL_SEC: float = 0.5

# [2026-08-21 user承認・残り秒数の量子化] `_update_defender_only` の
# キャッシュキーに入る budget_sec (着弾までの残り秒) を、受け側が打てる
# 手数の単位 (= 1手あたりの平均設置時間) に丸めてからキーへ入れる近似。
# 「受け側が打てる手数は floor(残り時間 ÷ 1手の時間) で決まる」
# (memory reference_ojama_landing_gated_by_placement) ため、同じ商になる
# budget_sec は同じ答えのはず、という前提でキャッシュの粒度を粗くする
# (実際の MC 計算 `_reach` には常に元の budget_sec を渡すため、
# キャッシュミス時の計算結果自体は量子化の影響を一切受けない)。
#
# 量子化幅は mc_counter_estimator.BEAM_ROLLOUT_AVG_STEP_TIME_SEC
# (`PLACEMENT_SPEED_BY_ROW_SEC` 段別実測値 0.134〜0.496秒の単純平均、
# 既存の物理実測値からの導出であり本ファイルで再フィットはしない
# feedback_overfitting_awareness_2026-08-04 準拠)。実測値 ≈0.348秒。
COUNTER_BUDGET_QUANTUM_SEC: float = mc_counter.BEAM_ROLLOUT_AVG_STEP_TIME_SEC


class CounterReachTracker:
    """相手が返せるかを時間予算ベースのモンテカルロで見る打ち合い優位。

    盤面 + 時間予算が同じなら結果も同じ (実装が決定論的) なのでキャッシュする。
    さらに (2026-08-12) 呼び出し側が `t_sec` を渡した場合、
    `COUNTER_RECOMPUTE_INTERVAL_SEC` 未満の間隔では前回の結果を再利用する
    (t_sec 省略時は従来通り毎回計算=後方互換)。

    [2026-08-21 user承認・設置ごと量子化] `_update_defender_only` 経路 (受け側
    限定) は時間予算 (残り秒) をキャッシュキーに含むため、盤面が変わらない
    間も残り時間が単調減少し続け `f"{budget_sec:.2f}"` が毎回変わる。実測
    (scripts配下の一時計測、60秒クリップ2本) でキャッシュヒット率は共に
    **0.00%** だった (盤面同一でも時間差で必ずミスする設計上の必然)。
    user承認の近似「自分の盤面は自分が置いた時しか変わらないので、置くまでは
    前回の答えを使い回してよい」を `reuse_if_board_unchanged` (既定 False)
    で実装し、有効時のみ `_placement_reuse` (受け側+閾値のスコープごとに
    直近の盤面bytesと結果を保持) を参照する。閾値が変われば別スコープになり
    自動的に再計算される (「閾値が違えば再計算が必要」という指摘を反映)。
    """

    def __init__(self) -> None:
        self._cache: dict[bytes, float] = {}
        # 直近に使った時間予算と平均打手数 (デバッグ・表示用)
        self.last_budget_sec: float = 0.0
        self.last_hands: float = 0.0
        # 時間間引き用の直近状態 (2026-08-12 追加)
        self._last_result: tuple[float, float, float] | None = None
        self._last_t_sec: float | None = None
        # 設置ごと量子化 (2026-08-21 追加、既定 reuse_if_board_unchanged=False
        # の間は一切参照されない=完全に無害)。scope_key -> (直近盤面bytes, (p, h))。
        self._placement_reuse: "dict[str, tuple[bytes, tuple[float, float]]]" = {}

    def _reach(
        self, board: Board, budget_sec: float,
        known_pairs: "tuple[tuple[int, int], ...]",
        threshold_ojama: float = COUNTER_THRESHOLD_OJAMA,
    ) -> tuple[float, float]:
        """(閾値以上を返せる確率, 平均打手数) を返す。

        threshold_ojama: 到達確率を判定するお邪魔換算の閾値 (2026-08-13 追加
            の optional 引数、既定は従来の固定値 COUNTER_THRESHOLD_OJAMA。
            --counter-defender-only では実際の飛来量を渡す、backwards compat)。
        """
        dist = mc_counter.estimate_counter_distribution(
            board, budget_sec,
            known_pairs=known_pairs,
            thresholds_ojama=(threshold_ojama,),
            n_rollouts=COUNTER_N_ROLLOUTS,
        )
        return (
            float(dist.prob_at_least.get(threshold_ojama, 0.0)),
            float(dist.mean_hands_used),
        )

    def update(
        self, b1: Board, b2: Board, budget_sec: float = 0.0,
        next1: "tuple[int, int] | None" = None,
        next2: "tuple[int, int] | None" = None,
        t_sec: float | None = None,
        defender_side: str | None = None,
        threshold_ojama: float | None = None,
        reuse_if_board_unchanged: bool = False,
        quantize_budget_sec: bool = False,
    ) -> tuple[float, float, float]:
        """(1P視点の優位[-100,100], 1Pの応手確率, 2Pの応手確率) を返す。

        budget_sec: 着弾までの時間予算 [秒]。 **手数はこの予算から決まる**
            (固定値を使わない)。 0 以下なら判定不能として 0 を返す。
        next1/next2: 各 side の既知ネクスト (あれば精度が上がる)。
        t_sec: 呼び出し側の動画内時刻 [秒] (省略可、後方互換の optional 引数)。
            指定した場合のみ `COUNTER_RECOMPUTE_INTERVAL_SEC` 間引きを適用する。
            ただし budget_sec が 0↔正 で遷移した直後 (打ち合い開始/終了) は
            反応遅れを避けるため間引きを無視して即計算する。
        defender_side: "1P"/"2P" を指定すると、その side の盤面**のみ**を
            対象に応手確率を計算する (2026-08-13 #4/#5 修正、
            --counter-defender-only 用)。指定 side の応手確率のみが返り値の
            該当 index に入り、もう片方は NaN になる。返り値[0] (adv) は
            0.0 固定 (呼び出し側 generate() が ojama_damage ベースの新しい
            統合式で計算するため、本メソッドは確率算出のみ担当する設計分離)。
            既定 None = 従来通り両側計算 (backwards compat)。時間ベース間引き
            (COUNTER_RECOMPUTE_INTERVAL_SEC) は defender_side 指定時は適用
            しない (閾値/対象 side がフレームごとに変わり得るため単純化、
            計算コストは従来の片側分のみで元々半減している)。
        threshold_ojama: defender_side 指定時に使う到達確率の閾値 (お邪魔
            換算、実際の飛来量)。None なら COUNTER_THRESHOLD_OJAMA へ
            フォールバックする。
        reuse_if_board_unchanged: [2026-08-21 追加] defender_side 指定時のみ
            有効。True で「受け側の盤面bytesが前回計算時と同一なら、時間予算
            (budget_sec) が変わっていても前回の (応手確率, 平均打手数) を
            再利用する」近似 (user承認、クラス docstring 参照)。閾値
            (threshold_ojama) が変わればスコープが別物になり自動的に
            再計算される。既定 False = 従来通り毎回 budget_sec も一致条件に
            含める (backwards compat)。defender_side が None の経路
            (両側同時計算) では未対応 (該当なし、無視される)。
        quantize_budget_sec: [2026-08-21 追加、reuse_if_board_unchanged とは
            独立の別機構] defender_side 指定時のみ有効。True で budget_sec
            (残り秒数) をキャッシュキーに入れる際 `COUNTER_BUDGET_QUANTUM_SEC`
            (1手あたりの平均設置時間、物理実測値由来) 単位に丸める
            (`_budget_cache_key_part` 参照)。実際の MC 計算には常に元の
            budget_sec を渡すため、キャッシュミス時の計算結果自体は不変。
            既定 False = 従来通り `.2f` 精度 (backwards compat)。
        """
        if defender_side is not None:
            return self._update_defender_only(
                b1, b2, budget_sec, next1, next2, t_sec, defender_side,
                threshold_ojama, reuse_if_board_unchanged, quantize_budget_sec)
        budget_transitioned = (
            (budget_sec <= 0.0) != (self.last_budget_sec <= 0.0)
        )
        if budget_sec <= 0.0:
            self.last_budget_sec = budget_sec
            self._last_result = (0.0, float("nan"), float("nan"))
            self._last_t_sec = t_sec
            return self._last_result
        if (
            t_sec is not None and not budget_transitioned
            and self._last_result is not None and self._last_t_sec is not None
            and (t_sec - self._last_t_sec) < COUNTER_RECOMPUTE_INTERVAL_SEC
        ):
            return self._last_result
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
        result = (float(max(-100.0, min(100.0, adv))), p1, p2)
        self._last_result = result
        self._last_t_sec = t_sec
        return result

    def _placement_reuse_scope_key(self, defender_side: str, threshold_ojama: float) -> str:
        """設置ごと量子化のスコープキー (受け側 + 閾値で分離)。

        閾値が変わればスコープが別物になるため、盤面が同じでも閾値が違えば
        自動的に再計算される (別スコープ=直近盤面が未記録=キャッシュ不成立)。
        """
        return f"def={defender_side}|th={threshold_ojama:.2f}"

    def _try_reuse_by_board(
        self, scope_key: str, base: bytes,
    ) -> "tuple[float, float] | None":
        """同一スコープで直近盤面bytesと完全一致する場合のみ前回結果を返す。

        時間予算 (budget_sec) の差異は無視する近似 (クラス docstring
        「設置ごと量子化」参照、user承認)。不一致・未記録なら None。
        """
        cached = self._placement_reuse.get(scope_key)
        if cached is None:
            return None
        last_base, last_ph = cached
        return last_ph if last_base == base else None

    def _budget_cache_key_part(self, budget_sec: float, quantize: bool) -> str:
        """キャッシュキーに入れる budget_sec 部分を返す (量子化 optional)。

        quantize=True: 1手あたりの平均設置時間 (COUNTER_BUDGET_QUANTUM_SEC)
        単位に floor 量子化したバケット番号を返す (「同じ手数=同じ答え」
        近似、クラス docstring 参照)。quantize=False (既定) は従来通り
        `.2f` 精度の生値を返す (backwards compat)。
        """
        if not quantize:
            return f"{budget_sec:.2f}"
        bucket = int(budget_sec // COUNTER_BUDGET_QUANTUM_SEC)
        return f"qbucket={bucket}"

    def _update_defender_only(
        self, b1: Board, b2: Board, budget_sec: float,
        next1: "tuple[int, int] | None", next2: "tuple[int, int] | None",
        t_sec: float | None, defender_side: str, threshold_ojama: float | None,
        reuse_if_board_unchanged: bool = False,
        quantize_budget_sec: bool = False,
    ) -> tuple[float, float, float]:
        """`update()` の defender_side 指定時の実体 (#4/#5 修正、2026-08-13)。

        受け側の盤面 1 つだけを対象に MC を1回だけ回す (従来の両側計算より
        コストが半分で済む)。既存の `_cache` を再利用するが、キーに
        defender_side/threshold を含めて対称経路のキャッシュと衝突しない
        ようにする。時間ベース間引きは適用しない (docstring 参照)。

        reuse_if_board_unchanged: [2026-08-21 追加] True で設置ごと量子化
        (クラス docstring 参照) を有効化する。既定 False では
        `_placement_reuse` を一切読み書きせず従来と完全に同一の挙動。
        quantize_budget_sec: [2026-08-21 追加、reuse_if_board_unchanged とは
        独立の別機構] True で budget_sec をキャッシュキーに入れる際だけ
        `COUNTER_BUDGET_QUANTUM_SEC` 単位に丸める (`_budget_cache_key_part`
        参照)。キャッシュミス時に `_reach` へ渡す budget_sec は常に元の値
        (量子化しない) なので、ミス時の計算結果自体は不変。既定 False =
        従来通り `.2f` 精度をキーに使う (backwards compat)。
        """
        if budget_sec <= 0.0:
            result = (0.0, float("nan"), float("nan"))
            self.last_budget_sec = budget_sec
            self._last_result = result
            self._last_t_sec = t_sec
            return result
        self.last_budget_sec = budget_sec
        board = b1 if defender_side == "1P" else b2
        nx = next1 if defender_side == "1P" else next2
        th = COUNTER_THRESHOLD_OJAMA if threshold_ojama is None else threshold_ojama
        known = (nx,) if nx and nx[0] > 0 and nx[1] > 0 else ()
        base = board.grid_bytes() if hasattr(board, "grid_bytes") else board._grid.tobytes()
        scope_key = self._placement_reuse_scope_key(defender_side, th)
        reused = (
            self._try_reuse_by_board(scope_key, base)
            if reuse_if_board_unchanged else None
        )
        if reused is not None:
            p, h = reused
        else:
            budget_part = self._budget_cache_key_part(budget_sec, quantize_budget_sec)
            key = base + f"|{budget_part}|{known}|def={defender_side}|th={th:.2f}".encode()
            if key not in self._cache:
                if len(self._cache) > 256:
                    self._cache.clear()
                self._cache[key] = self._reach(board, budget_sec, known, threshold_ojama=th)
            p, h = self._cache[key]
            if reuse_if_board_unchanged:
                self._placement_reuse[scope_key] = (base, (p, h))
        self.last_hands = h
        result = (
            (0.0, p, float("nan")) if defender_side == "1P" else (0.0, float("nan"), p)
        )
        self._last_result = result
        self._last_t_sec = t_sec
        return result


# ============================
# #3/#4/#5 修正 (2026-08-13、docs/DEMO_REVIEW_2026-08-13.md)
# ============================
# #3  --counter-remaining-time: 打ち合い応手の時間予算の意味論修正
#       (経過時間の控除 + 観測連鎖数を最終連鎖数と誤認しない条件付き補正)。
# #4/#5 --counter-defender-only: 受け側限定・実飛来量ベースの応手判定。
# 両フラグは独立 (それぞれ既定 False = 従来挙動、組み合わせ自由)。
# 攻撃側検知 (_detect_chain_attacker) だけを共有する。

# #3 で使う条件付き分布テーブルの既定パス
# (scripts/_build_chain_length_conditional_2026-08-13.py が生成)。
CHAIN_LENGTH_CONDITIONAL_PATH = Path("data/verify/chain_length_conditional_2026-08-13.json")


@dataclass(frozen=True)
class _ChainAttackObservation:
    """攻撃側検知の下ごしらえ結果 (#3 と #4/#5 が共通で使う)。"""
    chain_count: int                       # 観測された連鎖数 (0=攻撃なし)
    trigger_sec: float                     # 攻撃側 ChainEvent.trigger_sec
    attacker_side: "str | None"            # "1P"/"2P"/None (攻撃なし)
    attacker_event: "ChainEvent | None"     # 攻撃側の生 ChainEvent


def _detect_chain_attacker(r_p1, r_p2, t_sec: float) -> _ChainAttackObservation:
    """両 side の chain_event から、より大きい連鎖数を出している側 (攻撃側) を
    検知する (旧来の `_cc` 計算をそのまま関数化しただけ、挙動不変)。
    """
    cc = 0
    trigger_sec = 0.0
    attacker_side: "str | None" = None
    attacker_event = None
    for label, sr in (("1P", r_p1), ("2P", r_p2)):
        ev = getattr(sr, "chain_event", None)
        n = getattr(ev, "chain_count", None) if ev is not None else None
        if n and int(n) > cc:
            cc = int(n)
            trigger_sec = float(getattr(ev, "trigger_sec", t_sec))
            attacker_side = label
            attacker_event = ev
    return _ChainAttackObservation(cc, trigger_sec, attacker_side, attacker_event)


def _load_chain_length_conditional_table(
    path: Path = CHAIN_LENGTH_CONDITIONAL_PATH,
) -> "dict[int, float]":
    """#3 で使う E[最終連鎖数|観測N連鎖到達] テーブルを読み込む。

    ファイル不在・壊れている場合は空dictを返す (fail-safe、レンダを止めない)。
    呼び出し側 `_expected_final_chain_count` は空dictを「観測値=最終値と
    みなす」保守的フォールバック (旧来の近似と同じ) として扱う。
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        table = data.get("expected_final_given_reached_n", {})
        return {int(k): float(v) for k, v in table.items()}
    except (json.JSONDecodeError, OSError, ValueError, AttributeError):
        return {}


def _expected_final_chain_count(observed_n: int, table: "dict[int, float]") -> float:
    """観測連鎖数 N から期待最終連鎖数 E[最終|N到達] を引く (テーブル無ければ
    保守的フォールバック=観測値そのまま、実測上限超は外挿せずクランプする)。
    """
    if observed_n <= 0:
        return 0.0
    if not table:
        return float(observed_n)
    if observed_n in table:
        return table[observed_n]
    max_key = max(table)
    return table[max_key] if observed_n > max_key else float(observed_n)


def _chain_remaining_time_budget_sec(
    chain_count: int, trigger_sec: float, t_sec: float,
    table: "dict[int, float]",
) -> float:
    """#3 修正 + #12 フォローアップ較正 (2026-08-14): 経過時間控除 + 条件付き
    期待最終連鎖数で残り時間を求める (stateless)。

    残り時間 = anim(E[最終|N]) − 経過時間 + 着弾ラグ (+1手、
    reference_ojama_landing_gated_by_placement と整合させるため
    SEC_PER_HAND を流用、新規定数を作らない)。

    anim() は calibration="empirical_table_2026_08_14" (連鎖数別演出時間の
    実測中央値テーブル、docs/DEMO_REVIEW_2026-08-13.md #12 案B) を使う。
    従来の calibration="legacy" (0.4秒/連鎖) は大連鎖ほど実演出を大幅に
    過小評価していた (6連鎖: 実測中央値9.5秒 vs 旧2.4秒)。この関数は
    enable_counter_remaining_time (既定False) 配下でのみ呼ばれる opt-in
    経路のため、legacy 経路 (_resolve_counter_time_budget の
    enable_remaining_time=False 分岐) には一切影響しない。
    """
    if chain_count <= 0:
        return 0.0
    expected_final = _expected_final_chain_count(chain_count, table)
    total_anim = iv.estimate_chain_anim_duration_sec(
        expected_final, calibration="empirical_table_2026_08_14")
    elapsed = max(0.0, t_sec - trigger_sec)
    remaining = total_anim - elapsed + iv.SEC_PER_HAND
    return max(0.0, remaining)


def _resolve_counter_time_budget(
    obs: _ChainAttackObservation, t_sec: float,
    enable_remaining_time: bool, chain_len_table: "dict[int, float]",
) -> float:
    """#3: 打ち合い応手の時間予算を求める (旧来経路/新経路の切替)。

    enable_remaining_time=False (既定) では従来通り
    `estimate_chain_anim_duration_sec(観測連鎖数)` そのもの (bit-identical、
    backwards compat)。
    """
    if obs.chain_count <= 0:
        return 0.0
    if not enable_remaining_time:
        return iv.estimate_chain_anim_duration_sec(float(obs.chain_count))
    return _chain_remaining_time_budget_sec(
        obs.chain_count, obs.trigger_sec, t_sec, chain_len_table)


def _incoming_ojama_for_defender(
    defender_side: str, obs: _ChainAttackObservation,
    snap: OjamaAccountSnapshot, elapsed_sec: float,
) -> float:
    """defender_side に実際に飛んでくるおじゃま概算量を求める (1方向分)。

    脅威 = 攻撃側 (defender_side の相手) の進行中連鎖の現時点得点
    (score_to_ojama 換算、`iv._score_to_ojama_count` 再利用) + 既存の
    予告おじゃま (pending)。固定閾値ではなく実際の飛来量そのものを使う
    (docs/DEMO_REVIEW_2026-08-13.md #5、reference_ojama_damage_nonlinear
    「二値でなく期待ダメージ」原則)。
    """
    attacker_side = "2P" if defender_side == "1P" else "1P"
    pending = snap.pending_p1 if defender_side == "1P" else snap.pending_p2
    chain_ojama = (
        iv._score_to_ojama_count(float(obs.attacker_event.total_score), elapsed_sec)
        if obs.attacker_side == attacker_side and obs.attacker_event is not None
        else 0
    )
    return float(chain_ojama) + float(max(0, pending))


def _resolve_defender_threat(
    obs: _ChainAttackObservation, snap: OjamaAccountSnapshot, elapsed_sec: float,
) -> "tuple[str | None, float]":
    """#4/#5: 受け側と実際の飛来おじゃま概算量を求める (脅威が無ければ None, 0.0)。

    脅威の対象は「相手の連鎖イベントが進行中」または「予告おじゃま
    (pending) > 0」のどちらか (docs/DEMO_REVIEW_2026-08-13.md #5 item1、
    連鎖イベントが無くても既に着弾待ちの予告だけで脅威は成立しうる)。
    両方向 (1P向け/2P向け) を独立に確認し、両方に脅威がある稀なケースでは
    より大きい飛来量の方を優先する (本アーキは単一 defender のみを扱う
    設計上の制約、2026-08-13)。
    """
    incoming_1p = _incoming_ojama_for_defender("1P", obs, snap, elapsed_sec)
    incoming_2p = _incoming_ojama_for_defender("2P", obs, snap, elapsed_sec)
    candidates = [
        (side, amount) for side, amount in (("1P", incoming_1p), ("2P", incoming_2p))
        if amount > 0.0
    ]
    if not candidates:
        return None, 0.0
    return max(candidates, key=lambda c: c[1])


def _counter_defender_adv(
    defender_side: str, defender_prob: float, incoming_ojama: float,
    b1: Board, b2: Board, scale: float = COUNTER_SCALE,
) -> float:
    """#4: 受け側限定の応手成分を有利不利へ変換する (2026-08-13)。

    counter成分 = 攻撃側方向 × (1-受け側応手確率) × 飛来量ダメージ。
    飛来量ダメージは `iv.ojama_damage` (reference_ojama_damage_function の
    折れ点12/18個・受け側の残り容量 [headroom] に依存する既存の非線形関数)
    を再利用する。 受け側が高確率で返せる (defender_prob→1) ほど 0 に
    近づき、 極端化を抑える方向へ効く (docs/DEMO_REVIEW_2026-08-13.md #4)。
    値域は [-scale, +scale] に収まる。

    scale: 呼び出し元ごとに強度を切り替える optional 引数 (2026-08-14
        指摘12 修正2 で追加、backwards compat)。既定はライブ per-frame 用
        `COUNTER_SCALE` (従来と bit-identical)。決着ホールドの決定度増幅
        (`ResolvedExchangeTracker._amplify_decisive`) は専用の
        `RESOLVED_AMPLIFY_SCALE` を渡す (二重計上防止、同定数のコメント参照)。
    """
    if math.isnan(defender_prob):
        return 0.0
    defender_board = b1 if defender_side == "1P" else b2
    damage = iv.ojama_damage(defender_board, incoming_ojama).score
    direction = -1.0 if defender_side == "1P" else 1.0
    return direction * damage * (1.0 - defender_prob) * scale


def _build_counter_text_defender_only(
    defender_side: "str | None", defender_prob: float, incoming_ojama: float,
) -> str:
    """#4/#5 修正版の応手情報行 (受け側のみ・脅威が無ければ空文字)。

    従来の `_build_counter_text` は固定閾値を両者常時計算・表示していたが
    (docs/DEMO_REVIEW_2026-08-13.md #5)、受け側のみ・実際の飛来量が
    条件のときだけ表示する仕様に修正する (--counter-defender-only 有効時)。
    """
    if defender_side is None or math.isnan(defender_prob):
        return ""
    return (
        f"{defender_side}応手 {defender_prob * 100:.0f}%"
        f"  (飛来おじゃま概算{incoming_ojama:.0f}個)"
    )


def _resolve_counter_text(
    enable_defender_only: bool, defender_side: "str | None",
    counter_p1: float, counter_p2: float, incoming_ojama: float,
) -> str:
    """panel レイアウトの応手情報行テキストを、フラグに応じて選ぶ。"""
    if not enable_defender_only:
        return _build_counter_text(counter_p1, counter_p2)
    defender_prob = (
        counter_p1 if defender_side == "1P"
        else counter_p2 if defender_side == "2P" else float("nan")
    )
    return _build_counter_text_defender_only(defender_side, defender_prob, incoming_ojama)


def _resolve_counter_text_for_display(
    enable_defender_only: bool, resolved_hold_active: bool,
    hold_defender_side: "str | None", hold_defender_prob: float,
    hold_incoming_ojama: float,
    defender_side: "str | None", counter_p1: float, counter_p2: float,
    incoming_ojama: float,
) -> str:
    """[指摘12 修正3、2026-08-14] パネルの応手%行を、決着ホールド中かどうかで
    出所を切り替える。

    ホールド中 (`resolved_hold_active=True`) は毎フレームの settled 再計算
    そのものが凍結される (`resolved_hold_freezes_settled`) ため、ライブ
    per-frame の `defender_side`/`counter_p1`/`counter_p2`/`incoming_ojama`
    はホールド開始前の**古い値のまま**になる (指摘12 副次バグ: 表示が古い
    値に張り付き、受け側の向きも実際の決着と食い違う)。ホールド中は代わりに
    `ResolvedExchangeTracker` が決着計算 (`_amplify_decisive`) の内部で求めた
    受け側 side・飛来量・応手確率 (`hold_defender_side`/`hold_incoming_ojama`/
    `hold_defender_prob`) を表示専用に使う。これは表示のみの分岐であり
    `hold_adv`/`hold_p1` (判定値) には一切触れない
    (test_resolved_hold_display_does_not_affect_judgment で確認)。
    `enable_defender_only=False` の場合は従来通り両側常時表示に戻す
    (backwards compat)。
    """
    if resolved_hold_active and enable_defender_only:
        return _build_counter_text_defender_only(
            hold_defender_side, hold_defender_prob, hold_incoming_ojama)
    return _resolve_counter_text(
        enable_defender_only, defender_side, counter_p1, counter_p2, incoming_ojama)


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


def _driver_value(name: str, f1: dict[str, float], f2: dict[str, float]) -> float:
    """主因表示用の1P視点の値を返す (2026-08-14 追加、_score_advantage_full_row用)。

    `diff_` プレフィックス付き列名は既にそれ自体が「自−相手」の相対量
    (build_labeled_win_from_npz.py の b-2 diff化)なので f1 の値をそのまま
    使う (f1-f2 すると二重差分になってしまうため区別が必要)。それ以外の
    own列は従来通り f1-f2 の差分を使う。
    """
    if name.startswith("diff_"):
        return f1.get(name, 0.0)
    return f1.get(name, 0.0) - f2.get(name, 0.0)


def _score_advantage_full_row(
    model, b1: Board, b2: Board, snap: OjamaAccountSnapshot,
    attribution_exclude: tuple[str, ...],
) -> tuple[float, float, list[tuple[str, float]]]:
    """評価済み成果物モデル (47列、side単独row単位) 用の推論経路 (2026-08-14)。

    1P/2P それぞれ独立に「このsideが勝つ確率」を予測する (学習時の `won`
    ラベルが side単独行に対する二値ラベルのため、対の差分は取らない)。
    2予測は反対称保証が無いため、project既定の「有利不利は反対称関数」
    原則 (_train_model のミラー標本と同じ思想) に従い対称化して統合する。
    """
    cols = model._puyo_full_cols
    base1, base2 = _side_feats_full_base(b1), _side_feats_full_base(b2)
    f1 = _side_feats_full(base1, base2, snap.net_balance_capped, snap.forecast_p1)
    f2 = _side_feats_full(base2, base1, -snap.net_balance_capped, snap.forecast_p2)
    x1 = np.array([[np.nan_to_num(f1.get(c, 0.0)) for c in cols]], dtype=float)
    x2 = np.array([[np.nan_to_num(f2.get(c, 0.0)) for c in cols]], dtype=float)
    p_1p_wins = float(model.predict_proba(x1)[0, 1])
    p_2p_wins = float(model.predict_proba(x2)[0, 1])
    p1 = 0.5 * (p_1p_wins + (1.0 - p_2p_wins))
    adv = (p1 - 0.5) * 200.0
    all_candidates = sorted(
        ((c, _driver_value(c, f1, f2)) for c in JP_LABEL if c in f1),
        key=lambda kv: -abs(kv[1]))
    drivers = [kv for kv in all_candidates if kv[0] not in attribution_exclude][:3]
    return adv, p1, drivers


def _score_advantage(
    model, b1: Board, b2: Board, snap: OjamaAccountSnapshot,
    feature_cols: tuple[str, ...] | list[str] | None = None,
    attribution_exclude: tuple[str, ...] = ATTRIBUTION_EXCLUDED_INDICATORS,
) -> tuple[float, float, list[tuple[str, float]]]:
    """両盤面 → (有利不利[-100..100], 1P勝率, 主要ドライバ)。

    feature_cols: optional。省略時は model._puyo_feature_cols (学習時に
    _train_model が格納した実特徴量列) を使い、無ければ従来通り FEATURES に
    フォールバックする。既存呼出元は本引数を渡さないため挙動は変わらない。

    attribution_exclude: 「主因」候補から除外する指標名の集合 (2026-08-11
    ロードマップ Phase1-3)。 既定は src.production_config.
    ATTRIBUTION_EXCLUDED_INDICATORS (勝敗と無相関と実測済みの指標。根拠は
    同定数のコメント参照)。 **予測 (adv/p1) には一切影響しない** — この関数は
    adv/p1 を計算し終えた後の「表示候補の絞り込み」としてのみ使う。
    デバッグ目的で除外前の全候補を見たい場合は空 tuple `()` を渡す
    (--show-excluded-attribution 経由、scripts.visualize_advantage_overlay.main 参照)。

    model._puyo_feature_mode == "full_row" (評価済み成果物モデル、
    2026-08-14 coordinator指示) の場合は `_score_advantage_full_row` に
    完全委譲する (以下の従来 diff ベース経路は無変更、feature_cols 引数も
    full_row 分岐では無視される)。
    """
    if getattr(model, "_puyo_feature_mode", None) == "full_row":
        return _score_advantage_full_row(model, b1, b2, snap, attribution_exclude)
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
        CENTER_BULGE_COL: iv.center_bulge,
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
    # 主因候補: |差分| の大きい順。 attribution_exclude に含まれる指標は
    # 「実際の予測寄与を測らず差分の大きさだけで選ぶ」現行方式では無情報でも
    # 1位に出得るため、既定でここで弾く (adv/p1 の計算は上で完了済みで無関係)。
    all_candidates = sorted(
        ((c, diff[c]) for c in JP_LABEL if c in diff),
        key=lambda kv: -abs(kv[1]))
    drivers = [kv for kv in all_candidates if kv[0] not in attribution_exclude][:3]
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

    `every=9 (~0.3s @30fps)` という見積りは「1回の update() 呼び出し =
    処理対象になった1フレーム」という前提に立つ (2026-08-12 補足)。
    normalize_fps_30=True (既定) で 60fps 動画を stride=2 に間引くと、
    generate() のメインループは stride 対象フレームでしか update() を呼ばない
    ため、update() 呼び出し1回あたりに進む実時間も 2/60s (=1/30s) になり、
    「9回=0.3秒」という近似は 60fps 動画でも 30fps 動画と同じ**実時間**を
    指すようになる (stride 導入前は 60fps 動画で誤って 9/60=0.15秒 相当しか
    間引けていなかった=フレーム数ベース定数の意図が半分の実時間で発火する
    問題と同根)。
    """

    def __init__(
        self, model, every: int = 9,
        attribution_exclude: tuple[str, ...] = ATTRIBUTION_EXCLUDED_INDICATORS,
    ) -> None:  # ~0.3s @30fps (normalize_fps_30=True なら 60fps 動画でも同じ実時間)
        """attribution_exclude: `_score_advantage` に渡す主因除外リスト
        (2026-08-11 追加、optional 引数)。既定は production_config の
        ATTRIBUTION_EXCLUDED_INDICATORS。既存呼出元は本引数を渡さないため
        挙動は変わらない (後方互換)。
        """
        self._model = model
        self._every = max(1, every)
        self._attribution_exclude = attribution_exclude
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
            self._adv, _, self._drivers = _score_advantage(
                self._model, b1, b2, snap,
                attribution_exclude=self._attribution_exclude)
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
    attribution_exclude: tuple[str, ...] = ATTRIBUTION_EXCLUDED_INDICATORS,
    enable_chain_gen_accumulate: bool = False,
    accounting_tracker: OjamaAccountingTracker | None = None,
) -> tuple[OjamaAccountingTracker, "_SideTracker", "_SideTracker",
           PressureTracker, RealtimeForecastTracker, ScoreLeadTracker, HeavyAdvCache,
           EarlyFireTracker, "ChainGenerationAccumulator"]:
    """スコアリセット検知時に持続トラッカー一式を初期状態で作り直す。

    各トラッカーは内部 state が僅少 (数個の float/Counter) のため、都度
    再生成するだけで「初期化」と等価であり専用 reset() の追加は不要。
    `accounting_tracker` 指定時だけ OjamaAccountingTracker を保持する
    (Gate 3R-5 gross dump の境界ワイプ累積用)。省略時は従来どおり新規生成し
    reset() を呼ぶため、既定 OFF の実行経路は変わらない。
    戻り値末尾に EarlyFireTracker (2026-07-29)、続けて
    ChainGenerationAccumulator (2026-08-22 修正① 改良②) を追加 (既存呼出元は
    1箇所のみでアンパック先も同時更新済みのため後方互換上の実害なし)。

    attribution_exclude: HeavyAdvCache へそのまま渡す主因除外リスト
    (2026-08-11 追加、optional 引数。後方互換)。
    enable_chain_gen_accumulate: ChainGenerationAccumulator の累積モード
    (2026-08-22 user判断で既定 False に変更、`ChainGenerationAccumulator`
    docstring 参照。optional 引数、後方互換)。
    """
    # gross dump 有効時だけ、累積会計トラッカーを試合境界越しに保持する。
    # 境界フレームは呼出元の通常の _drive_ojama で一度だけ処理されるため、
    # 旧 tracker 内の pending が boundary_wiped_uncapped_* へ記録される。
    tracker = accounting_tracker
    if tracker is None:
        tracker = OjamaAccountingTracker()
        tracker.reset()
    return (tracker, _SideTracker(), _SideTracker(),
            PressureTracker(), RealtimeForecastTracker(), ScoreLeadTracker(),
            HeavyAdvCache(model, attribution_exclude=attribution_exclude),
            EarlyFireTracker(),
            ChainGenerationAccumulator(accumulate=enable_chain_gen_accumulate))


# ============================
# タイムラインdump (2026-08-11 追加)
# ============================
# 背景: scripts/scan_judgment_anomalies.py のフル走査 (148動画・全 settled
# 更新を再計算) は実測で約39日かかる (同スクリプトのモジュール docstring
# 「性能の発見」参照)。判定計算は generate() 内で1回しか行わない設計に変え、
# その結果を npz に保存して走査器はそれを読むだけにする (本モジュールがその
# 「1回だけ回す」側、走査器が「読むだけ」側)。
#
# adv_raw / adv_ema / p1 / p1_raw の意味論 (重要、恒久記録。2026-08-11
# アーキ審査で p1_raw 追加・raw/display 分離を決定):
#   adv_raw = HeavyAdvCache.update() が返す model_adv (= _score_advantage() の
#     生モデル出力、drivers と同じ呼び出しから出るため自己無矛盾)。
#     4成分ブレンド(pressure/forecast/threat/counter)や kill_override・
#     Platt較正・EMA は一切含まない。scan_judgment_anomalies.py の D0
#     (主因⇔結論の符号矛盾) は「モデル自身が出した diff と adv の符号一致」
#     という自己無矛盾性の検査であり、npz 再計算モード (ダミー会計だが同じ
#     _score_advantage() 直呼び) と同じ量を比較できるよう、意図的に
#     4成分ブレンド後の値ではなくこちらを採用する。D0 はこの raw 段階に
#     固定したままで正しい (kill_override の正当な符号反転を誤検知しない
#     ため、2026-08-11 アーキ判定)。
#   p1_raw = adv_to_winprob(adv_raw)。adv_raw と対になる「生モデルの勝率」
#     (kill_override/4成分ブレンド/校正/EMA 適用前)。
#   adv_ema = 4成分ブレンド + kill_override + 校正 + EMA を経た、実際に画面に
#     表示される値 (generate() 内のローカル変数 adv_ema そのもの)。
#   p1 = adv_ema に対応する EMA 後の表示用勝率 (ローカル変数 p1_last)。
#   D1a/D1b (「これだけ無視できない状況なのに有利判定」検出) は raw
#     (adv_raw/p1_raw) と display (adv_ema/p1) を**別々に**判定し、
#     Suspect.stage で "raw_only"(内部品質バックログ、kill_override 等で
#     是正済みのため表示は無害)/"display"(表示自体が矛盾=リリースブロッカー)/
#     "both" を区別する (scripts.scan_judgment_anomalies の detect_d1a/
#     detect_d1b・JudgmentRecord docstring 参照)。集計・合否ゲートは
#     display(+both) のみを基準にする (raw_only は別集計、コーディネーター
#     2026-08-11 決定)。
TIMELINE_DUMP_SCORE_NONE_SENTINEL: int = -1  # score OCR失敗(None)の npz 格納値


@dataclass(frozen=True)
class DisplayTimelineRow:
    """画面へ実際に出した値の密な1サンプル。settled限定dumpとは分離する。"""

    t_sec: float
    game_idx: int
    state1: str
    state2: str
    display_adv: float
    display_p1: float
    adv_raw_last: float
    source: str
    resolved_active: bool
    settled_ran: bool
    state1: str
    state2: str
    score1: int
    score2: int


def _display_timeline_source(
    resolved_active: bool, just_deactivated: bool, settled_ran: bool,
    episode_consistency_fallback: bool = False,
    minimum_prediction_guarded: bool = False,
) -> str:
    """表示値の由来を、優先順位どおりに分類する。"""
    if episode_consistency_fallback:
        return "episode_guard"
    if minimum_prediction_guarded:
        return "minimum_prediction_guard"
    if resolved_active:
        return "resolved_hold"
    if just_deactivated and not settled_ran:
        return "resolved_release"
    return "settled" if settled_ran else "frozen"


def save_display_timeline(
    path: Path, video_id: str, rows: list[DisplayTimelineRow],
) -> None:
    """実表示の密な時系列をnpzへ保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = tuple(DisplayTimelineRow.__dataclass_fields__)
    cols = {key: np.asarray([getattr(row, key) for row in rows]) for key in keys}
    np.savez_compressed(path, video_id=np.asarray(video_id), **cols)

# (2026-08-25 Gate 3R-5) gross 累積カウンタ dump 列のキー名 (単一情報源)。
# TimelineDumpRow のフィールド名 = npz キー名。値は settled 更新間の差分
# (単調カウンタの Δ、classify_gross_counter_delta の出力) を記録する。
# gross_pending_unc_* のみ絶対値 (uncapped 純残量)、gross_residual_* のみ float
# (保存則残差)。gross_inspected_sides は検査母数 (0=この行は未検査、2=両side検査。
# 残差0が「合っている」のか「測っていない」のかを行単位で区別する、
# memory feedback_zero_needs_denominator_2026-08-25)。
_TIMELINE_GROSS_INT_KEYS: tuple[str, ...] = (
    "gross_gen_p1", "gross_gen_p2",            # gross 生成 (cap前) Δ
    "gross_offset_p1", "gross_offset_p2",      # cap前 相殺 Δ
    "gross_dropped_p1", "gross_dropped_p2",    # cap前 着弾 Δ
    "gross_wiped_p1", "gross_wiped_p2",        # 境界ワイプ量 Δ (回数ではなく量)
    "gross_clamp_loss_p1", "gross_clamp_loss_p2",  # サニティ切り捨て量 Δ
    "gross_pending_unc_p1", "gross_pending_unc_p2",  # uncapped 純残量 (絶対値)
    "gross_inspected_sides",                   # 検査母数 (0 or 2)
)
_TIMELINE_GROSS_FLOAT_KEYS: tuple[str, ...] = (
    "gross_residual_p1", "gross_residual_p2",  # 保存則残差 (期待値 0)
)
_TIMELINE_GROSS_KEYS: tuple[str, ...] = (
    _TIMELINE_GROSS_INT_KEYS + _TIMELINE_GROSS_FLOAT_KEYS
)
# 保存則残差の非0判定の許容誤差 (probe v5 `_record_result` と同じ値。
# シーン逆算ではなく浮動小数点比較の一般的な許容誤差)。
_GROSS_RESIDUAL_EPS: float = 1e-9


@dataclass(frozen=True)
class TimelineDumpRow:
    """タイムラインdump 1レコード (1回の settled 更新に対応)。"""

    t_sec: float
    game_idx: int
    adv_raw: float
    adv_ema: float
    p1: float
    p1_raw: float  # adv_to_winprob(adv_raw)。kill_override/4成分ブレンド/校正/EMA
                    # 適用前 (2026-08-11 アーキ審査追加)。adv_raw と対で D1a/D1b の
                    # 「生モデル段階」判定に使う (drivers と同じ呼び出しに由来)。
    pending_p1: int
    pending_p2: int
    room1: int
    room2: int
    is_dead1: bool
    is_dead2: bool
    drivers_top1_name: str
    drivers_top1_val: float
    drivers_top3_names: tuple[str, str, str]
    drivers_top3_vals: tuple[float, float, float]
    score1: int
    score2: int
    b1_hash: int
    b2_hash: int
    state1: str
    state2: str
    # (2026-08-23 根治①) kill_override に実際に渡された「連鎖完走後」是正値
    # (`_kill_override_chain_completion_inputs` の戻り値、生成呼び出し側
    # :5149-5160 参照)。enable_kill_override_chain_completion が False、
    # または是正条件が成立しなかったフレームでは pending_p1/p2・room1/room2
    # と完全に同じ値になる (=「是正が効いたか」は k* と生値の等値比較だけで
    # 判定できる、専用フラグ列は追加しない)。
    # None を渡すと生値をそのまま複製する (__post_init__ 参照、後方互換:
    # 旧呼び出し・旧npz を読んだ場合もこのフィールドが自動的に埋まる)。
    kpending_p1: float | None = None
    kpending_p2: float | None = None
    kroom1: int | None = None
    kroom2: int | None = None
    # (2026-08-25 Gate 3R-5) gross 累積カウンタ列 (既定 OFF)。
    # None = 未記録 (フラグ OFF / 旧 npz)。k* 系と異なり __post_init__ で
    # 代替値を埋めない (「測っていない」を None で明示し、0 と区別する)。
    # フィールドの意味は _TIMELINE_GROSS_INT_KEYS 定義部コメント参照。
    gross_gen_p1: int | None = None
    gross_gen_p2: int | None = None
    gross_offset_p1: int | None = None
    gross_offset_p2: int | None = None
    gross_dropped_p1: int | None = None
    gross_dropped_p2: int | None = None
    gross_wiped_p1: int | None = None
    gross_wiped_p2: int | None = None
    gross_clamp_loss_p1: int | None = None
    gross_clamp_loss_p2: int | None = None
    gross_pending_unc_p1: int | None = None
    gross_pending_unc_p2: int | None = None
    gross_inspected_sides: int | None = None
    gross_residual_p1: float | None = None
    gross_residual_p2: float | None = None
    # (2026-08-25 Gate 3R-6 本体) 候補→猶予→確定の時間的死亡確定 (既定 OFF)。
    # None = 未記録 (フラグ OFF / 旧 npz)。is_dead1/is_dead2 (Board.is_dead()
    # の静的占有判定、即時反映) とは異なる列として並存させる
    # (`--death-confirm-sequence` docstring 参照。既存 is_dead1/is_dead2 は
    # 一切変更しない、backwards compat)。
    is_dead1_confirmed: bool | None = None
    is_dead2_confirmed: bool | None = None

    def __post_init__(self) -> None:
        """新4フィールドが未指定 (None) なら生値で埋める (後方互換の要)。

        frozen dataclass のため object.__setattr__ で直接書き込む。
        """
        if self.kpending_p1 is None:
            object.__setattr__(self, "kpending_p1", float(self.pending_p1))
        if self.kpending_p2 is None:
            object.__setattr__(self, "kpending_p2", float(self.pending_p2))
        if self.kroom1 is None:
            object.__setattr__(self, "kroom1", int(self.room1))
        if self.kroom2 is None:
            object.__setattr__(self, "kroom2", int(self.room2))


def _pad_drivers_top3(
    drivers: list[tuple[str, float]],
) -> tuple[tuple[str, str, str], tuple[float, float, float]]:
    """drivers (最大3件、既に上位3件に絞られている前提) を固定長3に0埋め整形する。"""
    names = [d[0] for d in drivers[:3]]
    vals = [d[1] for d in drivers[:3]]
    while len(names) < 3:
        names.append("")
        vals.append(0.0)
    return (names[0], names[1], names[2]), (vals[0], vals[1], vals[2])


def _build_gross_dump_fields(
    prev: GrossOjamaCounters | None, curr: GrossOjamaCounters,
    prev_pending_unc: tuple[int, int] | None, curr_pending_unc: tuple[int, int],
    game_idx: int,
) -> dict[str, int | float]:
    """settled 1回分の gross dump 列 kwargs を組み立てる (Gate 3R-5、純関数)。

    prev が None (処理開始直後) の行は
    差分を計算できないため、全 Δ を 0・`gross_inspected_sides=0` で返す
    (=「この行は検査していない」の明示。残差 0 の行と区別できる、
    memory feedback_zero_needs_denominator_2026-08-25)。
    それ以外は classify_gross_counter_delta (推測なしのカテゴリ別 Δ 復元)
    の結果をそのまま列に写す。uncapped 純残量は常に絶対値で記録する。
    """
    fields: dict[str, int | float] = {key: 0 for key in _TIMELINE_GROSS_INT_KEYS}
    fields.update({key: 0.0 for key in _TIMELINE_GROSS_FLOAT_KEYS})
    fields["gross_pending_unc_p1"] = int(curr_pending_unc[0])
    fields["gross_pending_unc_p2"] = int(curr_pending_unc[1])
    if prev is None or prev_pending_unc is None:
        return fields  # gross_inspected_sides=0 のまま = 未検査行
    c = classify_gross_counter_delta(
        prev, curr,
        (float(prev_pending_unc[0]), float(prev_pending_unc[1])),
        (float(curr_pending_unc[0]), float(curr_pending_unc[1])), game_idx)
    s = c.settlement
    fields.update(
        gross_gen_p1=c.generated_by_1p, gross_gen_p2=c.generated_by_2p,
        gross_offset_p1=int(s.canceled_by_1p) if s is not None else 0,
        gross_offset_p2=int(s.canceled_by_2p) if s is not None else 0,
        gross_dropped_p1=int(s.landed_on_1p) if s is not None else 0,
        gross_dropped_p2=int(s.landed_on_2p) if s is not None else 0,
        gross_wiped_p1=c.boundary_wiped_on_1p,
        gross_wiped_p2=c.boundary_wiped_on_2p,
        gross_clamp_loss_p1=c.clamp_loss_on_1p,
        gross_clamp_loss_p2=c.clamp_loss_on_2p,
        gross_residual_p1=c.conservation_residual_p1,
        gross_residual_p2=c.conservation_residual_p2,
        gross_inspected_sides=c.inspected_side_count,
    )
    return fields


@dataclass
class _GrossDumpStats:
    """gross dump 列の母数付き集計 (Gate 3R-5、2026-08-25)。

    0 は必ず母数と並べて表示する (「残差非0が0 side」は検査 side 数が
    0 より大きいときにのみ「合っている」を意味する、
    memory feedback_zero_needs_denominator_2026-08-25)。
    """

    rows_total: int = 0            # gross 列を記録した dump 行数 (母数)
    rows_inspected: int = 0        # 差分検査を実行できた行数
    sides_inspected: int = 0       # 検査 side 数の合計 (=検査母数)
    nonzero_residual_sides: int = 0  # 保存則残差が非0だった side 数
    residual_abs_max: float = 0.0  # |残差| の最大値
    wiped_total: int = 0           # 境界ワイプ量の合計 (回数ではなく量)
    clamp_loss_total: int = 0      # サニティ切り捨て量の合計 (期待値 0)

    def record(self, fields: dict[str, int | float]) -> None:
        """dump 1行分の gross 列を集計する。"""
        self.rows_total += 1
        if int(fields["gross_inspected_sides"]) <= 0:
            return
        self.rows_inspected += 1
        self.sides_inspected += int(fields["gross_inspected_sides"])
        residuals = (float(fields["gross_residual_p1"]),
                     float(fields["gross_residual_p2"]))
        self.nonzero_residual_sides += sum(
            abs(r) > _GROSS_RESIDUAL_EPS for r in residuals)
        self.residual_abs_max = max(
            self.residual_abs_max, *(abs(r) for r in residuals))
        self.wiped_total += (int(fields["gross_wiped_p1"])
                             + int(fields["gross_wiped_p2"]))
        self.clamp_loss_total += (int(fields["gross_clamp_loss_p1"])
                                  + int(fields["gross_clamp_loss_p2"]))

    def summary(self) -> str:
        """母数付きの可視化文字列 (0/0 = 一度も検査していない、と区別可能)。"""
        return (
            f"保存則残差非0 {self.nonzero_residual_sides}/{self.sides_inspected} side"
            f" (検査 {self.rows_inspected}/{self.rows_total} 行、"
            f"|残差|最大 {self.residual_abs_max:.6g}、"
            f"境界ワイプ量 {self.wiped_total}、clamp loss {self.clamp_loss_total})"
        )


@dataclass(frozen=True)
class EpisodeTimelineRow:
    """条件5専用の密な交換episode監査行。既存timelineとは別ファイルに保存する。"""

    t_sec: float
    game_idx: int
    state1: str
    state2: str
    episode_id: int
    stage: str
    status: str
    net_raw: float
    net_display: float
    total_generated: float
    total_canceled: float
    total_landed: float
    unreconciled: float
    provisional_residual: float
    is_unresolved: bool
    allows_hard_override: bool
    hard_override_target: float
    chain_id_p1: int
    chain_id_p2: int
    generation_p1: float
    generation_p2: float
    resolved_chain_count: int
    active_chain_count: int
    closed_episode_count: int
    closed_unreconciled_total: float
    closed_normal_unreconciled_count: int
    last_close_reason: str
    last_closed_status: str
    last_closed_generated: float
    last_closed_canceled: float
    last_closed_landed: float
    last_closed_unreconciled: float
    last_closed_has_settlement: bool
    last_closed_oversettled: float
    last_closed_oversettled_chain_count: int
    unattributed_settlement_total: float
    open_episode_outstanding: float
    ledger_residual_all: float
    simulate_excluded_chain_count: int
    simulate_excluded_amount: float
    formula_step_observation_count: int
    provisional_score_decrease_ignored_count: int
    boundary_count: int
    boundary_settlement_excluded_count: int
    boundary_settlement_excluded_amount: float
    forced_close_count: int
    chain_id_force_cut_count: int
    unbacked_residual_count: int
    finalize_divergence: float
    finalize_gate_held: bool
    oversettled_total: float
    retired_chain_count: int
    retired_unreconciled: float
    duplicate_generated_suppressed_count: int
    duplicate_generated_suppressed_amount: float
    finalize_rejected_count: int
    finalize_rejected_amount: float
    retired_canceled: float
    retired_landed: float
    retired_generated: float
    post_close_settlement_dropped_count: int
    post_close_settlement_dropped_amount: float
    post_close_settlement_backfilled_count: int
    post_close_settlement_backfilled_amount: float
    post_close_finalize_backfilled_count: int
    post_close_finalize_backfilled_amount: float
    post_close_finalize_dropped_count: int
    post_close_finalize_dropped_amount: float
    post_retire_backfilled_count: int
    post_retire_backfilled_amount: float
    post_close_outstanding_delta_total: float
    post_close_growth_backfilled_count: int
    post_close_growth_backfilled_amount: float
    post_close_growth_dropped_count: int
    post_close_growth_dropped_amount: float
    gross_inspected_sides: int
    gross_residual_p1: float
    gross_residual_p2: float
    hard_override_candidate: bool
    hard_override_applied: bool
    hard_override_path: str
    hard_override_hold_reason: str


@dataclass(frozen=True)
class _EpisodeDriveResult:
    """毎フレームのlive台帳とgross保存則検査結果。"""

    snapshot: LiveEpisodeSnapshot
    gross_inspected_sides: int = 0
    gross_residual_p1: float = 0.0
    gross_residual_p2: float = 0.0


def _exchange_chain_event_key(ev: ChainEvent | None) -> tuple | None:
    """公開ChainEventの値変化だけをresolverへ送るための同一性キー。"""
    if ev is None:
        return None
    return (round(ev.trigger_sec, 3), ev.mechanism, ev.chain_count, ev.total_score)


class _LiveEpisodeOverlayAdapter:
    """overlayの公開観測をLiveExchangeEpisodeTrackerへ毎フレーム供給する。"""

    def __init__(self) -> None:
        self.tracker = LiveExchangeEpisodeTracker(enabled=True)
        self._last_chain_key: dict[str, tuple | None] = {"1P": None, "2P": None}
        self._prev_gross: GrossOjamaCounters | None = None
        self._prev_pending: tuple[float, float] | None = None

    def update(
        self, *, result: PipelineResult, accounting: OjamaAccountingTracker,
        account_snapshot: OjamaAccountSnapshot, t_sec: float, game_idx: int,
        room1: int, room2: int, dead1: bool, dead2: bool,
    ) -> _EpisodeDriveResult:
        """認識・gross・死亡確定を同一frameの原子的入力として台帳へ渡す。"""
        curr_gross = accounting.get_gross_counters(t_sec)
        curr_pending = (
            float(account_snapshot.pending_p1_uncapped),
            float(account_snapshot.pending_p2_uncapped))
        classification = self._classify(curr_gross, curr_pending, game_idx)
        chains = self._chain_observations(
            result, t_sec, game_idx, accounting._elapsed(t_sec))
        generations = self._generation_observations(classification, t_sec, game_idx)
        context = PhysicalContext(
            p1_chaining=result.p1.state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE),
            p2_chaining=result.p2.state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE),
            p1_pending_uncapped=curr_pending[0], p2_pending_uncapped=curr_pending[1],
            p1_room=room1, p2_room=room2, p1_dead=dead1, p2_dead=dead2,
            game_idx=game_idx)
        live = self.tracker.observe_frame(
            t_sec=t_sec, context=context, chain_observations=chains,
            generation_observations=generations,
            settlement=(classification.settlement if classification else None),
            wiped_sides=(classification.wiped_sides if classification else ()))
        self._prev_gross, self._prev_pending = curr_gross, curr_pending
        return self._drive_result(live, classification)

    def _classify(
        self, curr: GrossOjamaCounters, pending: tuple[float, float], game_idx: int,
    ) -> GrossCounterDeltaClassification | None:
        if self._prev_gross is None or self._prev_pending is None:
            return None
        return classify_gross_counter_delta(
            self._prev_gross, curr, self._prev_pending, pending, game_idx)

    def _chain_observations(
        self, result: PipelineResult, t_sec: float, game_idx: int, elapsed_sec: float,
    ) -> tuple[ChainEventObservation, ...]:
        observations: list[ChainEventObservation] = []
        for label, side_result in (("1P", result.p1), ("2P", result.p2)):
            ev = side_result.chain_event
            key = _exchange_chain_event_key(ev)
            # chain_event は煙や認識gapで一時的に None になり、同じ物理連鎖が
            # 同じ値で再出現する。None を前回キーにすると、再出現が別IDになる。
            if ev is None or key == self._last_chain_key[label]:
                continue
            self._last_chain_key[label] = key
            observations.append(ChainEventObservation(
                side=label, t_sec=t_sec, mechanism=ev.mechanism or "",
                chain_count=ev.chain_count, total_score=ev.total_score,
                ojama_sent=ev.ojama_sent, game_idx=game_idx,
                elapsed_sec=elapsed_sec))
        return tuple(observations)

    def _generation_observations(
        self, classification: GrossCounterDeltaClassification | None,
        t_sec: float, game_idx: int,
    ) -> tuple[GenerationObservation, ...]:
        if classification is None:
            return ()
        generated = (
            ("1P", classification.generated_by_1p),
            ("2P", classification.generated_by_2p))
        return tuple(GenerationObservation(
            side=side, t_sec=t_sec, game_idx=game_idx, generated_delta=amount)
            for side, amount in generated if amount > 0)

    def _drive_result(
        self, live: LiveEpisodeSnapshot,
        classification: GrossCounterDeltaClassification | None,
    ) -> _EpisodeDriveResult:
        if classification is None:
            return _EpisodeDriveResult(snapshot=live)
        return _EpisodeDriveResult(
            snapshot=live,
            gross_inspected_sides=classification.inspected_side_count,
            gross_residual_p1=classification.conservation_residual_p1,
            gross_residual_p2=classification.conservation_residual_p2)


def _apply_episode_hard_override_gate(
    before: float, after: float, allows_hard_override: bool,
    *, is_unresolved: bool = False,
    hard_override_target: float | None = None,
    physical_net_raw: float | None = None,
) -> tuple[float, bool, bool, str]:
    """未解決episode中の±100完全上書きだけを方向付きで制御する。"""
    if is_unresolved and hard_override_target is not None:
        lower = min(before, -EPISODE_UNRESOLVED_ABS_CAP)
        upper = max(before, EPISODE_UNRESOLVED_ABS_CAP)
        capped = max(lower, min(upper, hard_override_target))
        candidate = capped != before
        if not candidate:
            return capped, False, False, ""
        return capped, True, False, "episode_physical_target_capped"
    candidate = after != before and abs(after) >= 100.0 - 1e-9
    if not candidate:
        return after, False, False, ""
    if (is_unresolved and physical_net_raw is not None
            and abs(physical_net_raw) > 1e-9
            and after * physical_net_raw < 0.0):
        return before, True, False, "episode_direction_conflict"
    if not allows_hard_override:
        lower = min(before, -EPISODE_UNRESOLVED_ABS_CAP)
        upper = max(before, EPISODE_UNRESOLVED_ABS_CAP)
        return (
            max(lower, min(upper, after)), True, False,
            "episode_unresolved_capped")
    if is_unresolved:
        lower = min(before, -EPISODE_UNRESOLVED_ABS_CAP)
        upper = max(before, EPISODE_UNRESOLVED_ABS_CAP)
        return (
            max(lower, min(upper, after)), True, False,
            "episode_unresolved_capped")
    return after, True, True, ""


def _sync_probability_after_episode_gate(
    before_adv: float, before_p1: float, gated_adv: float,
) -> float:
    """episode gate が表示値を変えた場合だけ勝率表示を同期する。"""
    if abs(gated_adv - before_adv) <= 1e-9:
        return before_p1
    return adv_to_winprob(gated_adv)


def _cap_unresolved_episode_display(
    adv: float, p1: float, *, is_unresolved: bool,
) -> tuple[float, float, bool]:
    """未解決交換の最終表示を勝率10〜90%相当へ制限する。

    live kill override だけでなく、古い resolved hold・EMA・整合性fallback の
    再露出も表示直前で一律に覆う。内部の学習値やhold状態は変更せず、交換が
    解決した次フレームから確定値をそのまま表示できるようにする。
    """
    if not is_unresolved or abs(adv) <= EPISODE_UNRESOLVED_ABS_CAP + 1e-9:
        return adv, p1, False
    capped = math.copysign(EPISODE_UNRESOLVED_ABS_CAP, adv)
    return capped, adv_to_winprob(capped), True


def _ensure_display_probability_direction(adv: float, p1: float) -> float:
    """EVEN外で有利側と勝率50%超側が食い違う表示だけを同期する。"""
    if abs(adv) <= EVEN_THRESHOLD:
        return p1
    if (adv > 0.0 and p1 >= 0.5) or (adv < 0.0 and p1 <= 0.5):
        return p1
    return adv_to_winprob(adv)


def _episode_kill_override_inputs(net_raw: float) -> tuple[float, float]:
    """交換台帳の1P視点純残量を、受け側別の致死入力へ変換する。

    正値は1Pが2Pへ送る残量、負値は2Pが1Pへ送る残量。相殺後の純残量なので
    同時に両側へ足さず、負けている受け側だけへcapなしで渡す。
    """
    return max(0.0, -net_raw), max(0.0, net_raw)


def _append_episode_hard_path(current: str, path: str, candidate: bool) -> str:
    """同一frameでlive/hold両候補が出ても供給経路を失わない。"""
    if not candidate or path in current.split("+"):
        return current
    return path if not current else f"{current}+{path}"


def _episode_timeline_row(
    drive: _EpisodeDriveResult, *, t_sec: float, game_idx: int,
    state1: str, state2: str,
    hard_candidate: bool, hard_applied: bool, hard_path: str, hard_reason: str,
) -> EpisodeTimelineRow:
    """live snapshotを固定スキーマのsidecar行へ変換する。"""
    snap = drive.snapshot
    ledger = snap.ledger
    values: dict[str, object] = dict(
        t_sec=t_sec, game_idx=game_idx, state1=state1, state2=state2,
        episode_id=-1 if ledger.episode_id is None else ledger.episode_id,
        stage="" if ledger.stage is None else ledger.stage.name,
        status="" if ledger.status is None else ledger.status.name,
        net_raw=ledger.net_raw, net_display=ledger.net_display,
        total_generated=ledger.total_generated, total_canceled=ledger.total_canceled,
        total_landed=ledger.total_landed, unreconciled=ledger.unreconciled,
        provisional_residual=ledger.provisional_residual,
        is_unresolved=ledger.is_unresolved,
        allows_hard_override=ledger.allows_hard_override,
        hard_override_target=(
            0.0 if ledger.hard_override_target is None
            else ledger.hard_override_target),
        **_episode_live_audit_fields(snap),
        **_episode_ledger_audit_fields(ledger),
        gross_inspected_sides=drive.gross_inspected_sides,
        gross_residual_p1=drive.gross_residual_p1,
        gross_residual_p2=drive.gross_residual_p2,
        hard_override_candidate=hard_candidate,
        hard_override_applied=hard_applied,
        hard_override_path=hard_path,
        hard_override_hold_reason=hard_reason)
    return EpisodeTimelineRow(**values)


def _episode_live_audit_fields(snap: LiveEpisodeSnapshot) -> dict[str, object]:
    """resolver・close要約・未帰属量をsidecar列へ展開する。"""
    return dict(
        chain_id_p1=-1 if snap.latest_chain_id_p1 is None else snap.latest_chain_id_p1,
        chain_id_p2=-1 if snap.latest_chain_id_p2 is None else snap.latest_chain_id_p2,
        generation_p1=snap.latest_generation_p1,
        generation_p2=snap.latest_generation_p2,
        resolved_chain_count=snap.resolved_chain_count,
        active_chain_count=len(snap.active_chains),
        closed_episode_count=snap.closed_episode_count,
        closed_unreconciled_total=snap.closed_unreconciled_total,
        closed_normal_unreconciled_count=snap.closed_normal_unreconciled_count,
        last_close_reason=snap.last_close_reason,
        last_closed_status=snap.last_closed_status,
        last_closed_generated=snap.last_closed_generated,
        last_closed_canceled=snap.last_closed_canceled,
        last_closed_landed=snap.last_closed_landed,
        last_closed_unreconciled=snap.last_closed_unreconciled,
        last_closed_has_settlement=snap.last_closed_has_settlement,
        last_closed_oversettled=snap.last_closed_oversettled,
        last_closed_oversettled_chain_count=(
            snap.last_closed_oversettled_chain_count),
        unattributed_settlement_total=snap.unattributed_settlement_total,
        open_episode_outstanding=snap.open_episode_outstanding,
        ledger_residual_all=snap.ledger_residual_all,
        simulate_excluded_chain_count=snap.simulate_excluded_chain_count,
        simulate_excluded_amount=snap.simulate_excluded_amount,
        formula_step_observation_count=snap.formula_step_observation_count,
        provisional_score_decrease_ignored_count=(
            snap.provisional_score_decrease_ignored_count),
        boundary_count=snap.boundary_count,
        boundary_settlement_excluded_count=snap.boundary_settlement_excluded_count,
        boundary_settlement_excluded_amount=snap.boundary_settlement_excluded_amount)


def _episode_ledger_audit_fields(ledger: LedgerSnapshot) -> dict[str, object]:
    """台帳の異常検知カウンタをsidecar列へ展開する。"""
    return dict(
        forced_close_count=ledger.forced_close_count,
        chain_id_force_cut_count=ledger.chain_id_force_cut_count,
        unbacked_residual_count=ledger.unbacked_residual_count,
        finalize_divergence=ledger.finalize_divergence,
        finalize_gate_held=ledger.finalize_gate_held,
        oversettled_total=ledger.oversettled_total,
        retired_chain_count=ledger.retired_chain_count,
        retired_unreconciled=ledger.retired_unreconciled,
        duplicate_generated_suppressed_count=(
            ledger.duplicate_generated_suppressed_count),
        duplicate_generated_suppressed_amount=(
            ledger.duplicate_generated_suppressed_amount),
        finalize_rejected_count=ledger.finalize_rejected_count,
        finalize_rejected_amount=ledger.finalize_rejected_amount,
        retired_canceled=ledger.retired_canceled,
        retired_landed=ledger.retired_landed,
        retired_generated=ledger.retired_generated,
        post_close_settlement_dropped_count=(
            ledger.post_close_settlement_dropped_count),
        post_close_settlement_dropped_amount=(
            ledger.post_close_settlement_dropped_amount),
        post_close_settlement_backfilled_count=(
            ledger.post_close_settlement_backfilled_count),
        post_close_settlement_backfilled_amount=(
            ledger.post_close_settlement_backfilled_amount),
        post_close_finalize_backfilled_count=(
            ledger.post_close_finalize_backfilled_count),
        post_close_finalize_backfilled_amount=(
            ledger.post_close_finalize_backfilled_amount),
        post_close_finalize_dropped_count=(
            ledger.post_close_finalize_dropped_count),
        post_close_finalize_dropped_amount=(
            ledger.post_close_finalize_dropped_amount),
        post_retire_backfilled_count=ledger.post_retire_backfilled_count,
        post_retire_backfilled_amount=ledger.post_retire_backfilled_amount,
        post_close_outstanding_delta_total=(
            ledger.post_close_outstanding_delta_total),
        post_close_growth_backfilled_count=(
            ledger.post_close_growth_backfilled_count),
        post_close_growth_backfilled_amount=(
            ledger.post_close_growth_backfilled_amount),
        post_close_growth_dropped_count=ledger.post_close_growth_dropped_count,
        post_close_growth_dropped_amount=ledger.post_close_growth_dropped_amount)


def save_episode_timeline(
    path: Path, video_id: str, rows: list[EpisodeTimelineRow],
) -> None:
    """条件5 sidecarを既存dumpとは独立したnpzへ保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = tuple(EpisodeTimelineRow.__dataclass_fields__)
    columns = {key: np.asarray([getattr(row, key) for row in rows]) for key in keys}
    np.savez_compressed(path, video_id=np.asarray(video_id), **columns)


def _build_timeline_dump_row(
    t_sec: float, game_idx: int, adv_raw: float, adv_ema: float, p1: float,
    p1_raw: float,
    pending_p1: int, pending_p2: int, room1: int, room2: int,
    b1: Board, b2: Board, drivers: list[tuple[str, float]],
    score1: int | None, score2: int | None, state1: str, state2: str,
    kpending_p1: float | None = None, kpending_p2: float | None = None,
    kroom1: int | None = None, kroom2: int | None = None,
    is_dead1: bool | None = None, is_dead2: bool | None = None,
    gross_fields: dict[str, int | float] | None = None,
    is_dead1_confirmed: bool | None = None, is_dead2_confirmed: bool | None = None,
) -> TimelineDumpRow:
    """1回分の settled 更新から TimelineDumpRow を組み立てる (純関数)。

    kpending_p1/p2・kroom1/kroom2 (2026-08-23 根治①追加): kill_override に
    実際に渡された是正後の値。省略時 (既定 None) は pending_p1/p2・room1/room2
    と同じ値になる (TimelineDumpRow.__post_init__ 参照、旧呼び出し元との
    bit-identical を保証する optional 引数)。

    is_dead1/is_dead2 (2026-08-25 Gate 3R-6 案A追加): 記録する窒息判定の
    override。省略時 (既定 None) は従来通り b1/b2 の Board.is_dead() を
    その場で評価する (旧呼び出し元との bit-identical を保証する optional
    引数)。enable_nonstable_hold_is_dead=True の呼出元が「非STABLE中は
    保留 (False)」に解決済みの値を渡す用途 (_resolve_nonstable_hold_is_dead
    参照)。

    gross_fields (2026-08-25 Gate 3R-5追加): `_build_gross_dump_fields` が
    返す gross_* 列の kwargs 辞書。省略時 (既定 None) は TimelineDumpRow の
    gross_* が全て既定値 None のまま (旧呼び出し元との bit-identical を
    保証する optional 引数)。

    is_dead1_confirmed/is_dead2_confirmed (2026-08-25 Gate 3R-6 本体追加):
    `DeathConfirmTracker.resolved_is_dead()` の値。省略時 (既定 None) は
    TimelineDumpRow の is_dead1_confirmed/is_dead2_confirmed が None のまま
    (旧呼び出し元との bit-identical を保証する optional 引数、既存
    is_dead1/is_dead2 列は一切変更しない)。
    """
    top1_name, top1_val = drivers[0] if drivers else ("", 0.0)
    top3_names, top3_vals = _pad_drivers_top3(list(drivers))
    return TimelineDumpRow(
        t_sec=t_sec, game_idx=game_idx, adv_raw=adv_raw, adv_ema=adv_ema, p1=p1,
        p1_raw=p1_raw,
        pending_p1=pending_p1, pending_p2=pending_p2, room1=room1, room2=room2,
        is_dead1=b1.is_dead() if is_dead1 is None else is_dead1,
        is_dead2=b2.is_dead() if is_dead2 is None else is_dead2,
        drivers_top1_name=top1_name, drivers_top1_val=top1_val,
        drivers_top3_names=top3_names, drivers_top3_vals=top3_vals,
        score1=score1 if score1 is not None else TIMELINE_DUMP_SCORE_NONE_SENTINEL,
        score2=score2 if score2 is not None else TIMELINE_DUMP_SCORE_NONE_SENTINEL,
        b1_hash=_board_hash(b1), b2_hash=_board_hash(b2),
        state1=state1, state2=state2,
        kpending_p1=kpending_p1, kpending_p2=kpending_p2,
        kroom1=kroom1, kroom2=kroom2,
        is_dead1_confirmed=is_dead1_confirmed, is_dead2_confirmed=is_dead2_confirmed,
        # (2026-08-25 Gate 3R-5) gross_fields=None (既定) では何も追加せず
        # TimelineDumpRow の gross_* は全て既定値 None のまま
        # (bit-identical、backwards compat)。
        **(gross_fields if gross_fields is not None else {}),
    )


def save_timeline_dump(path: Path, video_id: str, rows: list[TimelineDumpRow]) -> None:
    """タイムラインdump (1動画分) を npz に保存する。

    走査器 (scripts/scan_judgment_anomalies.py --from-dump) はこのファイルを
    読むだけで D0/D1a/D1b を検出でき、_score_advantage の再計算・モデル学習が
    一切不要になる。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    names3 = np.empty((n, 3), dtype=object)
    vals3 = np.empty((n, 3), dtype=np.float64)
    for i, row in enumerate(rows):
        names3[i] = row.drivers_top3_names
        vals3[i] = row.drivers_top3_vals
    data: dict[str, np.ndarray] = dict(
        video_id=np.array(video_id),
        t_sec=np.array([r.t_sec for r in rows], dtype=np.float64),
        game_idx=np.array([r.game_idx for r in rows], dtype=np.int32),
        adv_raw=np.array([r.adv_raw for r in rows], dtype=np.float64),
        adv_ema=np.array([r.adv_ema for r in rows], dtype=np.float64),
        p1=np.array([r.p1 for r in rows], dtype=np.float64),
        p1_raw=np.array([r.p1_raw for r in rows], dtype=np.float64),
        pending_p1=np.array([r.pending_p1 for r in rows], dtype=np.int32),
        pending_p2=np.array([r.pending_p2 for r in rows], dtype=np.int32),
        room1=np.array([r.room1 for r in rows], dtype=np.int32),
        room2=np.array([r.room2 for r in rows], dtype=np.int32),
        is_dead1=np.array([r.is_dead1 for r in rows], dtype=bool),
        is_dead2=np.array([r.is_dead2 for r in rows], dtype=bool),
        drivers_top1_name=np.array([r.drivers_top1_name for r in rows], dtype=object),
        drivers_top1_val=np.array([r.drivers_top1_val for r in rows], dtype=np.float64),
        drivers_top3_names=names3,
        drivers_top3_vals=vals3,
        score1=np.array([r.score1 for r in rows], dtype=np.int32),
        score2=np.array([r.score2 for r in rows], dtype=np.int32),
        b1_hash=np.array([r.b1_hash for r in rows], dtype=np.int64),
        b2_hash=np.array([r.b2_hash for r in rows], dtype=np.int64),
        state1=np.array([r.state1 for r in rows], dtype=object),
        state2=np.array([r.state2 for r in rows], dtype=object),
        # (2026-08-23 根治①) kill_override 是正後の実入力値。旧 dump にはこの
        # 4キーが存在しない (load_timeline_dump 側の後方互換分岐で吸収する)。
        kpending_p1=np.array([r.kpending_p1 for r in rows], dtype=np.float64),
        kpending_p2=np.array([r.kpending_p2 for r in rows], dtype=np.float64),
        kroom1=np.array([r.kroom1 for r in rows], dtype=np.int32),
        kroom2=np.array([r.kroom2 for r in rows], dtype=np.int32),
    )
    data.update(_gross_dump_columns(rows))
    data.update(_death_confirm_dump_columns(rows))
    np.savez_compressed(str(path), **data)


def _death_confirm_dump_columns(rows: list[TimelineDumpRow]) -> dict[str, np.ndarray]:
    """死亡確定 (Gate 3R-6 本体、2026-08-25) の2列を npz 保存用配列辞書へ変換する。

    `enable_death_confirm_sequence=False` (既定) では全行
    `is_dead1_confirmed` が None のままなので空辞書を返し、npz へキー自体を
    一切追加しない (旧 dump と bit-identical、`_gross_dump_columns` と同じ
    設計。load_timeline_dump 側は `"is_dead1_confirmed" not in d.files` で判定)。
    """
    if not rows or rows[0].is_dead1_confirmed is None:
        return {}
    return {
        "is_dead1_confirmed": np.array(
            [r.is_dead1_confirmed for r in rows], dtype=bool),
        "is_dead2_confirmed": np.array(
            [r.is_dead2_confirmed for r in rows], dtype=bool),
    }


def _gross_dump_columns(rows: list[TimelineDumpRow]) -> dict[str, np.ndarray]:
    """gross累積カウンタ列 (Gate 3R-5、2026-08-25) を npz 保存用配列辞書へ変換する。

    `enable_gross_ledger_dump=False` (既定) では全行 `gross_inspected_sides`
    が None のままなので空辞書を返し、npz へキー自体を一切追加しない
    (旧 dump と bit-identical、backwards compat。load_timeline_dump 側は
    `"gross_inspected_sides" not in d.files` で判定する)。
    """
    if not rows or rows[0].gross_inspected_sides is None:
        return {}
    cols: dict[str, np.ndarray] = {
        key: np.array([getattr(r, key) for r in rows], dtype=np.int64)
        for key in _TIMELINE_GROSS_INT_KEYS
    }
    cols.update({
        key: np.array([getattr(r, key) for r in rows], dtype=np.float64)
        for key in _TIMELINE_GROSS_FLOAT_KEYS
    })
    return cols


def _correct_dead_run_for_side(
    rows: list[TimelineDumpRow], is_dead_attr: str, state_attr: str,
) -> list[bool]:
    """1side分の is_dead 列を、同一 game_idx 内で「非STABLE中に凍結された
    is_dead」を「直後に STABLE 復帰した行の値」で遡及訂正した配列を返す
    (2026-08-24、`_retroactively_correct_dead_dump_rows` の内部ヘルパー)。

    game_idx が変わらない範囲を後方 (未来 -> 過去) に走査する。
    STABLE 行に出会うたびに `pending` (直近未来側の真値) を更新し、
    非STABLE 行はその `pending` で上書きする。**game_idx 終端まで一度も
    STABLE に出会わない (=試合終了まで復帰しない) 区間は pending が
    定まらないため一切変更しない** (死亡見逃しゼロを最優先する安全側)。
    1関数50行以内のため配列構築本体は呼出元 `_retroactively_correct_
    dead_dump_rows` に任せ、ここでは1side分の訂正ロジックのみを担う。
    """
    n = len(rows)
    corrected = [bool(getattr(r, is_dead_attr)) for r in rows]
    game_idx = [r.game_idx for r in rows]
    state = [getattr(r, state_attr) for r in rows]
    group_start = 0
    for i in range(1, n + 1):
        if i == n or game_idx[i] != game_idx[group_start]:
            pending: bool | None = None
            for k in range(i - 1, group_start - 1, -1):
                if state[k] == BoardState.STABLE.name:
                    pending = corrected[k]
                elif pending is not None:
                    corrected[k] = pending
            group_start = i
    return corrected


def _retroactively_correct_dead_dump_rows(
    rows: list[TimelineDumpRow],
) -> list[TimelineDumpRow]:
    """`enable_stable_confirmed_is_dead` 用の後処理 (2026-08-24 追加)。

    dump_rows (npz保存直前、メモリ上のリスト) に対してのみ作用する純関数。
    ライブの adv_ema/kill_override/描画/indicators_v2 には一切触れない
    (これらは別経路で b1/b2 を直接消費しており、本関数の戻り値を参照
    しないため無関係、`generate()` の enable_stable_confirmed_is_dead
    docstring 参照)。

    アルゴリズム: 同一 game_idx 内で、is_dead が非STABLE 中に凍結されて
    いた区間の直後に該当 side が STABLE へ復帰する行があれば、その値で
    遡って上書きする。試合終了まで復帰がない区間は変更しない
    (死亡見逃しゼロを最優先する安全側、docstring 参照)。
    """
    new_is_dead1 = _correct_dead_run_for_side(rows, "is_dead1", "state1")
    new_is_dead2 = _correct_dead_run_for_side(rows, "is_dead2", "state2")
    return [
        dataclass_replace(r, is_dead1=new_is_dead1[i], is_dead2=new_is_dead2[i])
        for i, r in enumerate(rows)
    ]


def _resolve_nonstable_hold_is_dead(
    raw_dead: bool, state_name: str,
) -> tuple[bool, bool]:
    """非STABLE中の is_dead 判定を保留する (Gate 3R-6 案A、2026-08-25)。

    user 伝授の絶対律「盤面が高い ＝ 窒息ではない。設置前 / 積み上げ中 /
    連鎖直前 / 連鎖中は窒息としない」(memory
    `reference_full_board_is_not_death_2026-08-22`) をリアルタイム制約下で
    実装する。**未来参照なし** — 「own state が STABLE でない = 今この側は
    落ち着いていない」という現在の状態だけで保留を決める
    (`enable_stable_confirmed_is_dead` の遡及訂正は未来の STABLE 値を使う
    後処理でリアルタイム不可、本関数はその代替でなく併存する別機構)。

    保留の表現は「False + 保留フラグ」の2値ペア (= unknown の明示エンコード。
    dump の state 列が STABLE か否かで保留行は完全に復元できるため、
    npz に新列は追加しない)。窒息を**主張しない**だけで「生存確定」の
    意味ではない点に注意。

    stateless な純関数 (コーディング規約: 観測ロジックは state を持たない。
    保留回数の集計は外部 wrapper `_IsDeadHoldStats` が担う)。

    Args:
        raw_dead: Board.is_dead() の生判定 (非STABLE中は凍結盤面由来)。
        state_name: その side の現在 state 名 (BoardState.*.name)。

    Returns:
        (記録する is_dead 値, 保留したか)。STABLE なら (raw_dead, False)。
    """
    if state_name == BoardState.STABLE.name:
        return raw_dead, False
    return False, True


@dataclass
class _IsDeadHoldStats:
    """is_dead 保留の母数付きカウンタ (Gate 3R-6 案A、2026-08-25)。

    「0 が『起きていない』のか『測っていない』のか」を区別するため、
    保留数は必ず母数 (dump 行数) 付きの `held/total` 形式で表示する
    (memory `feedback_zero_needs_denominator_2026-08-25`)。
    suppressed = 生判定 True を保留で False に変えた行数 (実効行数)。
    """

    total: int = 0
    held1: int = 0
    held2: int = 0
    suppressed1: int = 0
    suppressed2: int = 0

    def record(self, held1: bool, suppressed1: bool,
               held2: bool, suppressed2: bool) -> None:
        """dump 1行分の保留結果を集計する。"""
        self.total += 1
        self.held1 += int(held1)
        self.held2 += int(held2)
        self.suppressed1 += int(suppressed1)
        self.suppressed2 += int(suppressed2)

    def summary(self) -> str:
        """母数付きの可視化文字列 (例: 1P 保留 3/29 行 ...)。"""
        return (
            f"1P 保留 {self.held1}/{self.total} 行"
            f" (生判定True抑制 {self.suppressed1}/{self.total}) / "
            f"2P 保留 {self.held2}/{self.total} 行"
            f" (生判定True抑制 {self.suppressed2}/{self.total})"
        )


_TIMELINE_DUMP_ARRAY_KEYS: tuple[str, ...] = (
    "t_sec", "game_idx", "adv_raw", "adv_ema", "p1", "p1_raw",
    "pending_p1", "pending_p2", "room1", "room2",
    "is_dead1", "is_dead2",
    "drivers_top1_name", "drivers_top1_val",
    "drivers_top3_names", "drivers_top3_vals",
    "score1", "score2", "b1_hash", "b2_hash", "state1", "state2",
)


def _load_timeline_dump_k_fields(
    d: "np.lib.npyio.NpzFile",
) -> "tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]":
    """kill_override 是正値4列 (kpending_p1/p2, kroom1/kroom2) を読み込む
    (2026-08-23 根治①)。旧 dump にはこの4キーが存在しないため、無ければ
    全て None を返す (呼出側で TimelineDumpRow.__post_init__ の自動補完
    =生値と同じ値、に委ねる後方互換)。
    """
    if "kpending_p1" not in d.files:
        return None, None, None, None
    return d["kpending_p1"], d["kpending_p2"], d["kroom1"], d["kroom2"]


def _load_timeline_dump_gross_fields(
    d: "np.lib.npyio.NpzFile",
) -> "dict[str, np.ndarray] | None":
    """gross累積カウンタ dump列 (Gate 3R-5、2026-08-25) を読み込む。

    旧 dump にはこの列が存在しないため、無ければ None を返す
    (呼出側で TimelineDumpRow の gross_* を既定値 None のままにする
    後方互換分岐、_load_timeline_dump_k_fields と同じ設計)。
    """
    if "gross_inspected_sides" not in d.files:
        return None
    return {key: d[key] for key in _TIMELINE_GROSS_KEYS}


def _load_timeline_dump_death_confirm_fields(
    d: "np.lib.npyio.NpzFile",
) -> "tuple[np.ndarray | None, np.ndarray | None]":
    """死亡確定2列 (Gate 3R-6 本体、2026-08-25) を読み込む。

    旧 dump にはこの列が存在しないため、無ければ (None, None) を返す
    (呼出側で TimelineDumpRow の is_dead1_confirmed/is_dead2_confirmed を
    既定値 None のままにする後方互換分岐、_load_timeline_dump_k_fields と
    同じ設計)。
    """
    if "is_dead1_confirmed" not in d.files:
        return None, None
    return d["is_dead1_confirmed"], d["is_dead2_confirmed"]


def _gross_kwargs_from_arrays(
    gross: "dict[str, np.ndarray] | None", i: int,
) -> dict[str, int | float]:
    """gross配列辞書から1行分の TimelineDumpRow kwargs を組み立てる。

    gross=None (旧 dump / フラグ OFF) なら空辞書を返し、呼出元の
    TimelineDumpRow 構築が gross_* を既定値 None のままにする。
    """
    if gross is None:
        return {}
    return {
        key: (float(gross[key][i]) if key in _TIMELINE_GROSS_FLOAT_KEYS
              else int(gross[key][i]))
        for key in _TIMELINE_GROSS_KEYS
    }


def _timeline_dump_row_from_arrays(
    f: "dict[str, np.ndarray]", i: int,
    kpending_p1: "np.ndarray | None", kpending_p2: "np.ndarray | None",
    kroom1: "np.ndarray | None", kroom2: "np.ndarray | None",
    gross: "dict[str, np.ndarray] | None" = None,
    death_confirm1: "np.ndarray | None" = None,
    death_confirm2: "np.ndarray | None" = None,
) -> TimelineDumpRow:
    """事前ロード済み配列辞書 (npz から1回だけ読み出し済み) から1行を組み立てる。"""
    return TimelineDumpRow(
        t_sec=float(f["t_sec"][i]), game_idx=int(f["game_idx"][i]),
        adv_raw=float(f["adv_raw"][i]), adv_ema=float(f["adv_ema"][i]),
        p1=float(f["p1"][i]), p1_raw=float(f["p1_raw"][i]),
        pending_p1=int(f["pending_p1"][i]), pending_p2=int(f["pending_p2"][i]),
        room1=int(f["room1"][i]), room2=int(f["room2"][i]),
        is_dead1=bool(f["is_dead1"][i]), is_dead2=bool(f["is_dead2"][i]),
        drivers_top1_name=str(f["drivers_top1_name"][i]),
        drivers_top1_val=float(f["drivers_top1_val"][i]),
        drivers_top3_names=tuple(str(x) for x in f["drivers_top3_names"][i]),
        drivers_top3_vals=tuple(float(x) for x in f["drivers_top3_vals"][i]),
        score1=int(f["score1"][i]), score2=int(f["score2"][i]),
        b1_hash=int(f["b1_hash"][i]), b2_hash=int(f["b2_hash"][i]),
        state1=str(f["state1"][i]), state2=str(f["state2"][i]),
        kpending_p1=float(kpending_p1[i]) if kpending_p1 is not None else None,
        kpending_p2=float(kpending_p2[i]) if kpending_p2 is not None else None,
        kroom1=int(kroom1[i]) if kroom1 is not None else None,
        kroom2=int(kroom2[i]) if kroom2 is not None else None,
        is_dead1_confirmed=(
            bool(death_confirm1[i]) if death_confirm1 is not None else None),
        is_dead2_confirmed=(
            bool(death_confirm2[i]) if death_confirm2 is not None else None),
        **_gross_kwargs_from_arrays(gross, i),
    )


def load_timeline_dump(path: Path) -> tuple[str, list[TimelineDumpRow]]:
    """save_timeline_dump() が書いた npz を (video_id, レコード列) に復元する。

    (2026-08-23 根治③・性能バグ修正) 旧実装は `d["t_sec"][i]` のように
    各フィールドをレコード毎に npz から直接参照しており、圧縮 npz
    (savez_compressed) は参照のたびに該当配列を丸ごと解凍するため、
    1区間 (n≈2万行) の読込で27GB超の read が発生していた。本修正は
    全フィールドをループの外で1回だけ読み出す形に直す。出力はビット同一
    (tests/test_advantage_overlay_timeline_dump.py 参照)。

    gross_* 列 (2026-08-25 Gate 3R-5) は旧 dump / フラグ OFF の dump には
    存在しないため `_load_timeline_dump_gross_fields` が None を返し、
    全行の gross_* が既定値 None のまま復元される (bit-identical)。
    is_dead1_confirmed/is_dead2_confirmed 列 (2026-08-25 Gate 3R-6 本体) も
    同様に旧 dump / フラグ OFF では存在せず、None のまま復元される。
    """
    d = np.load(str(path), allow_pickle=True)
    video_id = str(d["video_id"])
    fields = {k: d[k] for k in _TIMELINE_DUMP_ARRAY_KEYS}
    kpending_p1, kpending_p2, kroom1, kroom2 = _load_timeline_dump_k_fields(d)
    gross = _load_timeline_dump_gross_fields(d)
    death_confirm1, death_confirm2 = _load_timeline_dump_death_confirm_fields(d)
    n = int(fields["t_sec"].shape[0])
    rows = [
        _timeline_dump_row_from_arrays(
            fields, i, kpending_p1, kpending_p2, kroom1, kroom2, gross,
            death_confirm1, death_confirm2)
        for i in range(n)
    ]
    return video_id, rows


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


def _graph_relative_time(t_sec: float, start_sec: float, game_start_sec: float) -> float:
    """#8 修正: 下部グラフの横軸に使う「現在の試合開始からの相対時間」。

    game_start_sec は境界検知のたびに `(t_sec - start_sec)` の値へ更新される
    (`_reset_graph_origin` 参照)。境界が一度も起きない動画では
    game_start_sec=0.0 のままなので、戻り値は従来の `t_sec - start_sec` と
    完全に一致する (backwards compat)。
    """
    return (t_sec - start_sec) - game_start_sec


def _reset_graph_origin(
    t_sec: float, start_sec: float, n_frames: int, fps: float,
) -> "tuple[float, float]":
    """#8 修正: 試合境界検知時にグラフの原点とスケールを巻き直す。

    docs/DEMO_REVIEW_2026-08-13.md #8: history.clear() だけでは history に
    積む座標が動画全体の絶対時間のままだったため、境界後の曲線が絶対位置
    (=途中) から始まって見えるバグだった。原点を現在時刻に更新し、スケール
    (グラフ横軸の総尺) も「この時点からの残り動画尺」に巻き直す (試合ごとに
    巻き直してよい、簡明な実装を優先)。

    Returns:
        (新しい game_start_sec, 新しい graph_total)。
    """
    game_start_sec = t_sec - start_sec
    graph_total = max(1.0, (n_frames / fps) - t_sec)
    return game_start_sec, graph_total


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
    chain_text_p1: str = "", chain_text_p2: str = "",
    chain_mismatch_p1: bool = False, chain_mismatch_p2: bool = False,
) -> None:
    """右側縦長情報パネル (バー/勝率/主因/状態/経過時刻) を描画する。

    box: (x0, y0, w, h) の矩形 (panel_layout_regions()["info"])。
    バー描画は既存 _draw_bar を座標だけ差し替えて再利用する
    (「読みやすさ優先」のため overlay 版よりバー太め・勝率フォント大きめ)。
    chain_text_p1/p2: --show-chain-count (既定 OFF) 用のれんさ表示行。
        既定 "" = 従来通り行自体を描かない (counter_text と同じ
        optional-if-truthy パターン、backwards compat)。
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
    if chain_text_p1:
        color = PANEL_CHAIN_MISMATCH_COLOR if chain_mismatch_p1 else (150, 200, 255)
        d.text((x0 + pad, y0 + PANEL_INFO_CHAIN_Y1), chain_text_p1,
               font=_font(16), fill=color)
    if chain_text_p2:
        color = PANEL_CHAIN_MISMATCH_COLOR if chain_mismatch_p2 else (255, 180, 180)
        d.text((x0 + pad, y0 + PANEL_INFO_CHAIN_Y2), chain_text_p2,
               font=_font(16), fill=color)
    d.text((x0 + pad, y0 + h - PANEL_INFO_ELAPSED_BOTTOM_MARGIN),
           f"経過 {elapsed_sec:.0f} 秒", font=_font(20), fill=(200, 200, 200))


def _draw_panel_layout(
    frame: np.ndarray, adv: float, p1: float,
    drivers: list[tuple[str, float]], waiting: bool,
    history: list[tuple[float, float]], t_rel: float, total: float,
    state1: str, state2: str, counter_text: str, elapsed_sec: float,
    chain_text_p1: str = "", chain_text_p2: str = "",
    chain_mismatch_p1: bool = False, chain_mismatch_p2: bool = False,
    subtitle_h: int = PANEL_SUBTITLE_H,
) -> np.ndarray:
    """パネルレイアウト (左上映像+左下グラフ+右情報パネル+下端字幕帯、1920x1080)
    で1フレーム描画する。

    2026-08-10 user指示の新レイアウト (同日、下端字幕帯の追記込み)。既存の
    _draw_overlay (盤面上に直接バー等を重ねる従来レイアウト) とは完全に独立
    した経路であり、 --layout panel 指定時のみ呼ばれる (既定 layout=overlay
    では未使用、既存出力は一切変わらない)。下端の字幕帯 (regions["subtitle"])
    には背景色を塗るだけで文字・図形を一切描かない (user要求の絶対条件)。

    subtitle_h: 字幕帯の高さ (既定 PANEL_SUBTITLE_H = 従来と完全一致、
        2026-08-21 追加)。panel_layout_regions() へそのまま渡す
        (座標計算の単一情報源はそちら、本関数は描画のみ)。0 を渡すと
        字幕帯領域が高さ0となり矩形塗り自体が no-op になる (PIL は
        ゼロ高矩形を安全に無視する)。
    """
    regions = panel_layout_regions(subtitle_h=subtitle_h)
    canvas = Image.new("RGB", (PANEL_CANVAS_W, PANEL_CANVAS_H), (12, 12, 16))
    vx, vy, vw, vh = regions["video"]
    video_rgb = cv2.cvtColor(
        cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    canvas.paste(Image.fromarray(video_rgb), (vx, vy))
    d = ImageDraw.Draw(canvas, "RGBA")
    sx, sy, sw, sh = regions["subtitle"]
    if sh > 0:
        d.rectangle([sx, sy, sx + sw, sy + sh], fill=PANEL_SUBTITLE_BG_COLOR)  # 字幕帯: 無描画
    if history:
        _draw_graph(d, history, t_rel, total, render_area=regions["graph"])
    _draw_panel_info(d, regions["info"], adv, p1, waiting, drivers,
                     state1, state2, counter_text, elapsed_sec,
                     chain_text_p1=chain_text_p1, chain_text_p2=chain_text_p2,
                     chain_mismatch_p1=chain_mismatch_p1, chain_mismatch_p2=chain_mismatch_p2)
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
             model_dir: Path | None = None,
             show_recognition: bool = False,
             enable_landing_observed_color: bool | None = None,
             force_in_match: bool = True,
             enable_drift_guards: bool | None = None,
             enable_match_start_full_clear: bool | None = None,
             enable_recovery_counter_carryover: bool | None = None,
             enable_cnn_flicker_hsv_fallback: bool | None = None,
             enable_initial_confirm_vote: bool | None = None,
             enable_platt_calibration: bool = False,
             enable_phase_calibration: bool = False,
             enable_early_fire_reaction: bool = False,
             enable_per_side_settled: bool = False,
             disable_score_lead_bias: bool = False,
             enable_capability_pressure: bool = False,
             disable_pressure: bool = False,
             enable_counter_reach: bool = COUNTER_REACH_ENABLED_BY_DEFAULT,
             enable_counter_remaining_time: bool = False,
             enable_counter_defender_only: bool = False,
             enable_resolved_exchange_eval: bool = False,
             enable_resolved_decisive_amplify: bool = False,
             enable_resolved_live_defender: bool = False,
             enable_resolved_live_defender_strict: bool = False,
             enable_resolved_kill_override: bool = False,
             enable_kill_override_chain_completion: bool = False,
             enable_kill_override_chain_gen_accumulate: bool = False,
             enable_kill_override_attribution: bool = False,
             enable_early_fire_clear_on_finalize: bool = False,
             enable_resolved_kill_override_counter_aware: bool = False,
             enable_resolved_victim_gen_live: bool = False,
             enable_resolved_episode_physical_redecide: bool = False,
             enable_resolved_episode_physical_consistency_guard: bool = False,
             enable_resolved_minimum_prediction_guard: bool = False,
             enable_resolved_counter_placement_reuse: bool = False,
             enable_resolved_counter_budget_quantize: bool = False,
             enable_resolved_absolute_chain_end: bool = False,
             enable_puyo_to_empty_hsv_guard: bool | None = None,
             stable_majority_window: bool | None = None,
             enable_ojama_fall_placement_override: bool | None = None,
             enable_ojama_fall_entry_hardening: bool | None = None,
             enable_ojama_fall_scoped_exit: bool | None = None,
             enable_pseudo_chain_score_fill: bool = False,
             chain_hold_base_sec: "float | None" = None,
             chain_hold_per_step_sec: "float | None" = None,
             enable_slide_exit_min_display_guard: bool = False,
             layout: str = "overlay",
             panel_subtitle_h: int = PANEL_SUBTITLE_H,
             enable_resolved_pending_landing_gate: bool = False,
             show_excluded_attribution: bool = False,
             render: bool = True,
             dump_timeline_path: Path | None = None,
             debug_history_out: list[tuple[float, float]] | None = None,
             normalize_fps_30: bool = OVERLAY_NORMALIZE_FPS_30_ENABLED_BY_DEFAULT,
             use_production_recognition: bool = (
                 OVERLAY_PRODUCTION_RECOGNITION_ENABLED_BY_DEFAULT),
             resize_1080p: bool = OVERLAY_RESIZE_1080P_ENABLED_BY_DEFAULT,
             show_chain_count: bool = False,
             enable_stable_confirmed_is_dead: bool = False,
             enable_kill_override_hysteresis: bool = False,
             enable_kill_override_scale_compare: bool = False,
             enable_nonstable_hold_is_dead: bool = False,
             enable_gross_ledger_dump: bool = False,
             enable_death_confirm_sequence: bool = False,
             death_next_stationary_sec: float = NEXT_STATIONARY_CONFIRM_SEC,
             dump_display_timeline_path: Path | None = None,
             enable_exchange_episode_gate: bool = False,
             dump_exchange_episode_timeline_path: Path | None = None,
             ) -> int:
    """有利不利オーバーレイ動画を生成。書き出しフレーム数を返す。

    start_sec: 書き出し開始秒 (ゲームの真の開始=スコア0の瞬間)。
    warmup_sec: start_sec の何秒前から「処理だけ」始めるか (状態機械/会計の初期化用。
        この区間は認識を通すが動画には書き出さない)。
    end_sec: 書き出し終了秒。
    model_dir: 有利不利判定に使う学習済みモデル成果物のディレクトリ
        (`--model-dir`、2026-08-18 追加)。配下に
        `model_full148_full_features.joblib` / `feature_cols_full.json` を
        置く (`scripts/_retrain148_2026-08-14.py` の出力形式と同一)。
        既定 None = 従来通り MODEL_ARTIFACT_DIR
        (data/verify/retrain148_2026-08-14) を使う (後方互換、既存呼出元は
        挙動不変)。指定時にファイル欠如・特徴量列不一致があれば
        ModelArtifactMissingError / ModelArtifactFeatureMismatchError を
        送出して即座に停止する (黙って旧モデルにフォールバックしない、
        fail-silent 厳禁)。exclude_video 指定時は model_dir より
        exclude_video を優先し CSV 起動時学習にフォールバックする
        (`_acquire_model` 参照)。
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
    enable_phase_calibration: 表示用勝率に「進行度 (match_progress) 別」の
        Platt scaling 後段校正を適用する (2026-08-11 Phase1-2 追加、
        data/indicators_v2/phase_platt_calibration.json を読む)。既定 False
        (後方互換、既存呼出元は挙動不変)。全位相共通の enable_platt_calibration
        と同時 True は禁止 (ValueError、どちらを使うか呼出元が明示する設計)。
        全位相共通 Platt は memory `project_calibration_overconfident_2026-07-29`
        で終盤の ECE が改善しにくいと判明しており (終盤0.056→0.035程度)、
        B-1(対称化修正)+B-2(進行度列)後の tier1 モデル自身の OOF 予測から
        位相別に学習した校正器を使うことでより高い改善を狙う
        (scripts/fit_phase_platt_calibration.py が学習・
        data/verify/calibration_phase_2026-08-11 に効果測定値を保存)。
        True かつ校正器ファイルが無い場合は CalibrationFileMissingError を
        処理開始前に送出する (enable_platt_calibration と同じ fail-fast 設計)。
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
    enable_counter_remaining_time: True で打ち合い応手の時間予算の意味論を
        修正する (2026-08-13、docs/DEMO_REVIEW_2026-08-13.md #3)。従来
        (既定 False) は「観測済み連鎖数 × 0.4秒」を毎回の総時間予算として
        丸ごと渡していた (経過時間を控除せず・観測連鎖数を最終連鎖数と
        誤認する二重のズレ)。True にすると (1) 攻撃側 ChainEvent.trigger_sec
        からの経過時間を時計で控除し、 (2) 「N連鎖まで観測された連鎖の
        最終連鎖数」の条件付き期待値テーブル (CHAIN_LENGTH_CONDITIONAL_PATH、
        scripts/_build_chain_length_conditional_2026-08-13.py が148動画の
        実測 chain_trigger_sec イベントから生成、無ければ観測値そのまま=
        旧来近似へ自動フォールバック) を使う。既定 False = 従来挙動と
        完全に同一 (backwards compat)。
    enable_counter_defender_only: True で打ち合い応手確率を受け側限定・
        実飛来量ベースに切り替える (2026-08-13、docs/DEMO_REVIEW_
        2026-08-13.md #4/#5)。従来 (既定 False) は (a) 固定閾値
        COUNTER_THRESHOLD_OJAMA を両者に常時計算・表示し (b) 攻撃が無い
        側にも表示していた。True にすると、攻撃 (相手の連鎖イベントまたは
        予告おじゃま) が飛んでいる側のみを対象に、閾値を実際の飛来量
        (進行中連鎖の現時点得点→おじゃま換算 + 既存予告分) から動的に
        算出する。有利不利への統合も
        「攻撃側方向 × (1-受け側応手確率) × 飛来量ダメージ
        (iv.ojama_damage、受け側の残り容量に依存する非線形関数)」に
        変わり、受け側が高確率で返せる場合に極端化を抑える方向へ効く。
        既定 False = 従来挙動と完全に同一 (backwards compat)。
        enable_counter_reach=False の場合はどちらのフラグも無効
        (打ち合い応手成分自体が計算されない)。
    enable_resolved_exchange_eval: True で ResolvedExchangeTracker (両者同時
        発火の決着先読み) を有効化する (2026-08-13、docs/DEMO_REVIEW_
        2026-08-13.md #9)。従来 (既定 False) は連鎖アニメ中の観測到着ごとに
        settled ゲート越しに逐次再評価するため、両者が撃ち合った未来は
        物理的に確定しているにもかかわらず勝率が乱高下する。True にすると、
        両側の chain_event が同時にアクティブになった瞬間に一度だけ
        `resolve_mutual_exchange` (連鎖完走シミュレーション+相殺+着弾、
        src.exchange_virtual_board) → `_score_advantage` の2段で決着後勝率を
        計算し、両側の chain_event が両方 None に戻った (=両者の連鎖アニメ
        終了) 後もさらに相殺後おじゃまの**着弾完了**まで表示を固定する
        (2026-08-14 指摘11対処、ResolvedExchangeTracker._landing_complete
        参照。着弾完了は判定不能でも安全弁 RESOLVED_HOLD_LANDING_MAX_WAIT_SEC
        で必ず解放する)。表示は adv_ema/p1_last の EMA 内部状態には混ぜず
        直接ホールド (無効時は完全に従来経路とビット一致)。片側のみの発火は
        トリガー対象外 (--early-fire-reaction の領分のまま、両フラグは独立)。
        simulate の連結欠損由来の過小評価 (既知事故: 真値8連鎖→simulate1連鎖)
        対策として、保持中に観測 score が予測総得点を超えたら下限として
        即時再決着する。既定 False = 従来挙動と完全に同一 (backwards compat)。
    enable_resolved_decisive_amplify: True で決着値に「受け側の応手不能度」を
        統合する (2026-08-14 指摘10対処)。着弾後仮想盤面で受け側限定の応手
        確率を既存 CounterReachTracker (受け側限定経路) から求め、既存
        `_counter_defender_adv` (専用定数 RESOLVED_AMPLIFY_SCALE +
        iv.ojama_damage) と同一式で決着値に加算する。応手不能 (確率低) かつ
        飛来量大なほど決定的側へ増幅し、受け側が高確率で返せる場合はほぼ
        無効果のまま。時間予算は #3 と同じ `_chain_remaining_time_budget_sec`
        に一本化済み (指摘12対処、2026-08-14、旧式の直呼びは静的テストで
        禁止)。ホールド中のパネル応手%表示も本メソッドの内部値
        (`hold_defender_side`/`hold_incoming_ojama`/`hold_defender_prob`) を
        使うよう切り替わる (指摘12 副次バグ対処、判定値には影響しない)。
        `enable_resolved_exchange_eval=False` の場合は無視される (#9 サブ
        フラグ)。既定 False = 従来 (#9 のみ) の決着値と完全に同一
        (backwards compat)。
    enable_resolved_live_defender: True で「片側のみ連鎖中 (攻撃側継続・
        受け側自由行動)」の間、決着値を `COUNTER_RECOMPUTE_INTERVAL_SEC`
        (0.5秒) ごとにライブ再評価する (2026-08-15 指摘13対処、
        docs/DEMO_REVIEW_2026-08-13.md #13)。従来は両側 chain_event が両方
        None に戻るまで hold_adv を完全凍結していたが、受け側は連鎖中も
        置き続けており応手力は実際に変化する (user指摘)。凍結を維持する
        成分 (攻撃側の連鎖帰結=飛来量・攻撃側の仮想盤面 board_pX_after) と
        生値で動かす成分 (受け側の現在 STABLE 確定盤面・残り時間予算の逓減)
        を分離し、モデル評価と決定度増幅 (enable_resolved_decisive_amplify
        有効時) の両方をこの更新済み値で再計算する
        (`ResolvedExchangeTracker._reevaluate_live_defender` 参照)。表示が
        「攻撃側の帰結起点→受け側の組みに応じて漸移→撃ち返しで反転」という
        連続的な挙動になる。`enable_resolved_exchange_eval=False` の場合は
        無視される (#9 サブフラグ)。既定 False = 従来 (両側終了まで完全凍結)
        と完全に同一 (backwards compat)。
    enable_resolved_live_defender_strict: True で `enable_resolved_live_defender`
        の起動条件を厳格化する (2026-08-15 指摘14 案1、docs/DEMO_REVIEW_
        2026-08-13.md #14)。従来 (既定 False) の XOR 条件だけでは「両者が
        本当に同時に本線を撃ち合い攻撃側のアニメだけ先に終わった」ケースを
        「受け側が自由行動中」と誤分類し、実際にはまだ連鎖継続中の受け側の
        着地前盤面 (おじゃま僅少) をモデルへ渡して致死量を見落とす
        (実測: 589個飛来を受ける2Pに誤って生存率18.9%を5.2秒表示、正しくは
        3.9%)。計装 (`scripts/_diag_issue14_reeval_calls_2026-08-15.py`) で
        誤爆の実機構を確認したところ、chain_event の有無 (案1初版) では
        「旧連鎖の hold が切れてから新連鎖の trigger が検出されるまでの
        settle gap」を取りこぼすと判明した (この gap では defender 自身は
        `BoardState.GRAVITY_SETTLE` = 今まさに重力settle中だが ev は既に
        None)。True にすると defender_side 自身の**状態機械 state**
        (`r_p1.state`/`r_p2.state`) が `_LIVE_DEFENDER_BUSY_STATES`
        (= {CHAIN, GRAVITY_SETTLE}) に含まれるかを追加確認し、含まれれば
        (=今まさに自分の連鎖処理中) 再評価をスキップして直前の保持値を
        維持する (`ResolvedExchangeTracker._reevaluate_live_defender` 参照)。
        `TSUMO_FALL`/`OJAMA_FALL` は busy 扱いしない (指摘13が意図した
        「受け側は連鎖中も置き続ける」正当な自由行動を塞がないため)。
        `enable_resolved_live_defender=False` の場合は無視される (孫フラグ)。
        既定 False = 従来挙動と完全に同一 (backwards compat)。
    enable_resolved_kill_override: True で決着ホールド値 (hold_adv/hold_p1)
        にも致死上書き (`kill_override`) を適用する (2026-08-15 指摘14 案2、
        docs/DEMO_REVIEW_2026-08-13.md #14)。従来 (既定 False) は
        `kill_override` がライブ per-frame 経路にのみ配線されており、決着
        ホールド中は pending/room 比が致死水準でも安全弁が発火しなかった
        (実測: 589/50≈11.8 ≫ KILL_RATIO_FULL=1.5 でも無発火)。True にすると
        `ResolvedExchangeTracker.hold_after_kill_override` を表示直前に通す
        (既存の `kill_override` 関数・既存の `_incoming_total_p1/p2`・
        `board_room` をそのまま再利用、新規の観測量は増やさない)。
        `enable_resolved_exchange_eval=False` の場合は無視される (#9 サブ
        フラグ)。既定 False = 従来挙動と完全に同一 (backwards compat)。
    enable_kill_override_chain_completion: True で per-frame `kill_override`
        (`:4764` 付近、無条件で毎フレーム呼ばれる経路) の入力を「連鎖完走後」
        に是正する (2026-08-22 修正①、`_kill_override_chain_completion_inputs`
        参照)。従来は死ぬと判定されうる側が自分自身の連鎖 (CHAIN/
        GRAVITY_SETTLE) を撃っている最中でも、発火前の凍結盤面の空きと
        相殺前の額面 pending をそのまま使っており、「自分の連鎖が pending を
        相殺しきる」ケースまで致死断定してしまっていた (実測7件、
        logs/killoverride_wrong_2026-08-22/一覧.tsv)。True にすると、
        発火中の側の `chain_event.total_score` (simulate 由来の完走予測、
        アニメ進行度に非依存) を使って既存 `resolve_mutual_exchange`
        (#9 決着計算と同一関数、新規の会計計算は追加しない) で完走後の
        盤面空き・相殺後の残存 pending を求め、それを `kill_override` へ渡す。
        どちらの側も発火していないフレームでは完全に従来と同一の値を返す
        (bit-identical)。既定 False = 従来挙動と完全に同一 (backwards compat)。
    enable_kill_override_chain_gen_accumulate: `ChainGenerationAccumulator`
        の累積モード切替 (2026-08-22 user判断)。`enable_kill_override_
        chain_completion=False` の場合は無視される (孫フラグ)。既定 False =
        累積せず直近1件の chain_event の値に置き換える (対症療法の実測欠陥
        により既定を非累積化、`ChainGenerationAccumulator` docstring 参照:
        累積は「まだ画面に見えていない残り連鎖ぶんまで既に生成し終えた」
        架空の完了状態を仮定するため raw モデルとの間に新しい不一致時間帯を
        作ることが全編再走査の実測で判明した)。True にすると従来の累積動作
        (複数の formula 再トリガーにまたがって生成量を合算) に戻せる
        (CHAIN 保持時間の実測較正 [chain_hold_base_sec/chain_hold_per_step_sec]
        で断片化そのものを減らす根治と併用する場合、二重計上を避けるため
        既定 False のままにすること)。
    enable_kill_override_hysteresis: True で per-frame `kill_override` の出力に
        確信度ゲート (`KillOverrideConfidenceGate`、2026-08-24 B案) を適用する。
        同一方向に KILL_CONFIRM_PERSIST_SEC (≈0.96秒 = 受け側の持ち手2手 +
        認識反映8f) 持続して初めて完全上書き (±100) を許し、それまでは
        |adv| ≤ KILL_UNCONFIRMED_ABS_CAP (90) に制限、方向反転時は
        KILL_FLIP_COOLDOWN_SEC (≈0.35秒 = 1手時間) のクールダウンで上書きを
        保留する。根因③ (ChainEvent 断片化による1フレーム ±100→∓100 反転、
        memory project_pm100_display_flip_2026-08-24) の構造的禁止。
        既定 False = 従来挙動と完全に同一 (backwards compat)。
    enable_kill_override_scale_compare: True で per-frame `kill_override` の
        pending 入力を「規模の比較」に是正する (2026-08-24 A案、根因①②)。
        (i) 会計 finalize 遅延中の「送付済み未登録」分を
        `PostChainUnregisteredSentTracker` が連鎖完走時に即時捕捉して受け側
        pending に加算し、(ii) 相殺の引き算には PENDING_ABS_CAP=216 で丸める
        前の実額 (`snap.pending_p1/p2_uncapped`、src/ojama_accounting.py の
        並行帳簿) を使う。表示用の pending_p1/p2 (cap 済み) は従来のまま
        (docs/KNOWN_WEAKNESSES.md:1010- の cap は表示用として維持)。
        `enable_kill_override_chain_completion` と独立に動作するが、納品構成
        では併用する (完走シミュレーションの基礎 pending がこの是正値になる)。
        既定 False = 従来挙動と完全に同一 (backwards compat)。
    enable_early_fire_clear_on_finalize: True で `EarlyFireTracker` の速報
        バイアスをクリアする条件を「settled 再計算が走った」から「連鎖の
        finalize (`OjamaAccountingTracker._finalize_chain_end`) が会計に
        反映された」に変える (2026-08-22 修正②、`EarlyFireTracker.
        finalized_since_last_check` 参照)。`--per-side-settled` 下では相手が
        STABLE の間 settled 再計算が毎フレーム走るため、従来 (既定 False)
        は大連鎖の速報バイアスが finalize 前に毎フレーム消えていた。
        既定 False = 従来挙動と完全に同一 (backwards compat)。
        `enable_early_fire_reaction=False` の場合は無視される (孫フラグ)。
    enable_kill_override_attribution: True で `kill_override` が発火したフレーム
        の主因表示 (パネルの「主因:」欄) に致死判定の理由を明示する
        (2026-08-22 修正④、`_kill_override_attribution_entry` 参照)。
        従来は安全弁適用前の生モデル寄与度だけが並び、安全弁が結論を上書き
        した場合に表示と結論が矛盾していた (例: 主因1位が1P有利の根拠なのに
        結論は2P有利)。**予測値 (adv/p1) には一切影響しない表示専用の追加**。
        既定 False = 従来挙動と完全に同一 (backwards compat)。
    enable_resolved_kill_override_counter_aware: True で
        `enable_resolved_kill_override` の致死断定を「受け側が実際に応手
        可能か」で減衰する (2026-08-15 指摘19、docs/DEMO_REVIEW_2026-08-13.md
        #19、`ResolvedExchangeTracker.hold_after_kill_override` docstring
        参照)。従来 (既定 False) は pending/room 比のみで致死断定するため、
        受け側が STABLE で応手可能な局面でも致死断定する誤りがあった
        (実測 t=201.4-203.4: 1P 0.7% と誤表示、直後に撃ち返し score 42065
        vs 19729 で勝利)。応手確率は新規に推測ロジックを作らず、既存の
        `CounterReachTracker` 経由で同一フレームに算出済みの
        `hold_defender_prob`/`hold_defender_side` を再利用する。
        `enable_resolved_kill_override=False` の場合は無視される (孫フラグ)。
        既定 False = 従来挙動と完全に同一 (backwards compat)。
    enable_resolved_victim_gen_live: True で `_maybe_redecide` の「保持
        セッション中1回きり」再決着 latch を、`chain_end_triggered_pX` が
        True の間 `COUNTER_RECOMPUTE_INTERVAL_SEC` (0.5秒) ごとに追従する
        方式へ緩和する (2026-08-16 指摘19 根治、coordinator決定(b)、
        docs/DEMO_REVIEW_2026-08-13.md #19、`ResolvedExchangeTracker` docstring
        指摘19根治節参照)。従来 (既定 False) は「1回きり」latch が settle
        開始直後の未確定 (しばしば0の) `chain_total_score_pX` で永久に
        固定してしまい、その後段階的に育っていく真の確定値 (実測:
        0→1260→4020 と複数回に分けて確定) を二度と拾わなかった。これが
        自分の連鎖を処理中の側 (victim) の生成お邪魔量が過小評価される
        直接原因 (`--resolved-kill-override-counter-aware` は対症療法として
        致死上書きだけを止めていたが、その手前の hold_adv 自体は非対称の
        まま残っていた)。`--resolved-exchange-eval` 無効時は無視される
        (#9 サブフラグ)。既定 False = 従来挙動と完全に同一 (backwards
        compat)。
    enable_resolved_minimum_prediction_guard: True で、物理的に連鎖中の
        ChainEvent 完走予測が1連鎖・40点の最小値に潰れた場合、その値を
        決着確定として表示せず直前の STABLE 評価を維持する。認識の連結欠損で
        本物の多連鎖が1連鎖へ縮む既知事故への安全策で、連鎖終了後は実測結果を
        通常どおり評価する。既定 False = 従来挙動と完全に同一。
    enable_resolved_counter_placement_reuse: [2026-08-21 user承認・配線]
        `CounterReachTracker._update_defender_only` (受け側限定応手MC) の
        再計算を「受け側の盤面bytesが変化した (=設置が起きた) とき」だけに
        限定する近似 (ResolvedExchangeTracker/CounterReachTracker docstring
        参照)。実測 (60秒クリップ2本、_measure_counter_cache_hitrate
        スクリプト) では従来キャッシュのヒット率が **0.00%** だった
        (時間予算がキーに入っており毎回ミスする設計上の必然)。閾値
        (相手の予告おじゃま量) が変わればスコープが別物になり自動的に
        再計算される。表示更新の周期 (`COUNTER_RECOMPUTE_INTERVAL_SEC`
        0.5秒ごと) 自体は変えない (計算頻度だけを盤面変化に紐づける)。
        `--resolved-exchange-eval` 無効時は無視される (#9 サブフラグ)。
        既定 False = 従来挙動と完全に同一 (backwards compat)。
    enable_resolved_absolute_chain_end: [2026-08-26 決着ホールド根治、
        user決定] 決着ホールドの解除条件に、user 伝授の絶対律
        「連鎖の終わり = 連鎖している側のネクストが動いた瞬間 OR
        連鎖している側にお邪魔が落ちた瞬間」を **OR で足す**。
        従来の解除条件「両側の chain_event が None」は、ChainEvent が長い
        連鎖で 1.4秒ごとに断片化するため打ち合い中は長時間成立せず、実測で
        最大 45.07秒 settled 再計算が止まっていた (ホールドが潰した settled
        1880/3964 = 47.43%、評価行が旧比 −26%)。新条件は解除を早めるだけで
        遅くはしないため、無限ホールドの新規リスクを作らない。
        `--resolved-exchange-eval` 無効時は無視される (#9 サブフラグ)。
        既定 False = 従来挙動と完全に同一 (backwards compat)。
    enable_resolved_counter_budget_quantize: [2026-08-21 user承認、上記
        counter_placement_reuse とは独立の別機構] `CounterReachTracker` の
        キャッシュキーに入る budget_sec (着弾までの残り秒) を
        `COUNTER_BUDGET_QUANTUM_SEC` (1手あたりの平均設置時間、
        `mc_counter_estimator.PLACEMENT_SPEED_BY_ROW_SEC` の単純平均
        ≈0.348秒、既存の物理実測値からの導出であり再フィットしない) 単位に
        丸めてからキーへ入れる。「受け側が打てる手数は floor(残り時間÷1手の
        時間) で決まる」ため、同じ商になる budget_sec は同じ答えのはず、
        という近似 (根拠: memory reference_ojama_landing_gated_by_placement)。
        キャッシュミス時に実際の MC 計算へ渡す budget_sec は常に元の値
        (量子化しない) のため、ミス時の計算結果自体は不変。
        `--resolved-exchange-eval` 無効時は無視される (#9 サブフラグ)。
        既定 False = 従来挙動と完全に同一 (backwards compat)。
    enable_puyo_to_empty_hsv_guard: RecognitionPipeline.load_default に渡す
        色→空 HSV 照合ガード (コミット 97445cc, 2026-07-30 追加)。True にすると
        NON-STABLE→STABLE 復帰 merge の色→空 遷移について HSV が色を保持する
        cell を消さない (列デッドロックの初発を停止、実測は
        scripts/_diag_column_deadlock_trace_2026-07-30.py 参照)。ただし 4動画測定で
        c58/c26 の 2P tail 悪化・c26/c69 の 1P 効果ゼロ、汎化未確認のため
        load_default 既定 OFF。既定 False = 従来挙動不変 (後方互換、A/B比較用)。
    stable_majority_window: RecognitionPipeline.load_default に渡す盤面確定窓
        3中2多数決 (2026-08-13 user承認、認識99.5%物差し条件付き採用)。True で
        初回STABLE確定窓が「stable_frame_count 連続厳密一致」から「直近3観測中
        2一致」に切り替わる (src/board_state_machine.py 参照)。
        None (既定) = load_default 本体の既定値 (False) に従う
        (後方互換、A/B比較用)。
    layout: "overlay"(既定、従来通り盤面に直接バー等を重ねるレイアウト)または
        "panel"(2026-08-10 user指示。左上に映像、左下にタイムライングラフ、
        右に縦長情報パネルを配置する新レイアウト、出力キャンバスは1920x1080)。
        既定 "overlay" = 従来挙動不変 (backwards compat)。認識・有利不利の
        計算経路は layout に関わらず完全に同一で、最終合成のみ分岐する。
    panel_subtitle_h: --layout panel 時の下端字幕帯の高さ (2026-08-21
        user指示「グラフ広げて」で追加)。既定 PANEL_SUBTITLE_H (140px) =
        従来と完全一致 (backwards compat)。0 を指定すると字幕帯を無くし、
        その分を左下グラフ (148→288px) と右の情報パネル (940→1080px) の
        高さへ丸ごと回す (panel_layout_regions() 参照、layout="overlay"
        では無視される)。
    enable_resolved_pending_landing_gate: ResolvedExchangeTracker の
        enable_pending_landing_gate をCLI/generate() から渡すための配線
        (2026-08-21 追加。クラス定義側は2026-08-21 導入済みだったが
        generate() のシグネチャに引数が無く渡す手段が存在しなかった配線漏れの
        是正)。攻撃側が連鎖中の間、予告おじゃまを受け側の生盤面へ着地させず
        保留する (ResolvedExchangeTracker docstring 参照)。
        既定 False = 従来挙動と完全に同一 (backwards compat)。
    show_excluded_attribution: True で「主因」候補から
        src.production_config.ATTRIBUTION_EXCLUDED_INDICATORS
        (勝敗と無相関と実測済みの指標、根拠は同定数コメント参照) を除外**しない**
        デバッグ表示に切り替える (2026-08-11 ロードマップ Phase1-3 追加)。
        既定 False = 除外リストを適用 (通常表示、2026-08-11 から既定挙動)。
        adv/p1 (有利不利の判定値そのもの) には一切影響しない
        (主因欄の表示候補を絞り込むだけ)。
    render: False で動画の合成・書き出し (VideoWriter生成・描画・writer.write)
        を一切行わず、判定計算だけ実行する (2026-08-11 タイムラインdump工事で
        追加)。認識 (RecognitionPipeline) と有利不利判定は render に関わらず
        完全に同一の経路・同一の間引き (HeavyAdvCache.every 等) で動作する
        (dump は本番が実際に出す判定の記録であるべきため、間引きも本番のまま
        温存する設計)。既定 True = 従来通り描画・書き出しする (backwards
        compat、既存呼出元は挙動不変)。
    dump_timeline_path: 指定すると、settled 更新 (有利不利判定が再計算される
        瞬間) のたびに1レコードを収集し、終了時に npz として保存する
        (2026-08-11 タイムラインdump工事で追加。スキーマは
        `TimelineDumpRow` 参照)。
        `scripts/scan_judgment_anomalies.py --from-dump` がこれを読むだけで
        「ありえない判定」検出でき、判定の再計算 (148動画で約39日と実測済み、
        同スクリプトのモジュール docstring 参照) が不要になる。
        既定 None = dump しない (backwards compat、既存呼出元は挙動不変)。
        `render=False` と併用すると計算のみを最速で回せる。
    debug_history_out: 指定すると、実際に画面へ表示する値 disp_adv (t_sec, disp_adv)
        を history と同じ間引き (sample_interval) で毎回追記する (2026-08-13
        #9 検証用デバッグフック、CLI 未配線)。settled 更新時のみ記録する
        dump_timeline_path と異なり、非settled中の保持値も含めた「実際の表示値」
        の全サンプルが得られる (#9 の乱高下削減効果を分散で数値検証するための
        側路であり、本番挙動には一切影響しない)。既定 None = 何もしない
        (backwards compat、既存呼出元は挙動不変)。
    dump_display_timeline_path: 指定すると、非settled中の凍結値・決着ホールド値を
        含む「実際の表示」を sample_interval と同じ密度で別npzへ保存する。
        settled限定の dump_timeline_path は会計・認識監査用として変更しない。
        既定 None では分岐内へ入らず、既存出力に影響しない。
    enable_exchange_episode_gate: 交換episodeが未解決の間、ライブ経路と決着
        ホールド経路の完全上書き±100を禁止する。既定Falseではtracker生成・
        gross毎frame分類・判定分岐を一切行わず、従来出力と互換。
    dump_exchange_episode_timeline_path: 条件5専用の密な台帳sidecar。既定Noneでは
        ファイルも列も生成しない。enable_exchange_episode_gate=Trueが必須。
    normalize_fps_30: True (既定、2026-08-12 追加) で 60fps 等の動画を
        stride 相当 (実効30fps) に間引く
        (src.fps_normalize.resolve_normalize_fps_30_stride)。
        collect_boards_lean.py (収集) が 2026-07-30 から既定採用している
        正規化と**同一関数**であり、CLI フラグ名・既定値も対称にしてある
        (`--normalize-fps-30` / `--no-normalize-fps-30`)。
        認識状態機械のフレーム数定数 (STABLE_RECOVERY_MIN_FRAMES=8 等) は
        「30fps で1フレーム進む=1/30秒」を前提にコメントされているため、
        60fps 動画を全フレーム処理すると実時間がその半分になり STABLE 遷移が
        過多になる (A/B実測 +23%)。本フラグにより収集・学習データと同じ
        認識意味論で動く。30fps 以下の動画は stride=1 に丸まり挙動不変。
        既定 True = 収集側 (normalize_fps_30 既定 True) と揃える
        (OVERLAY_NORMALIZE_FPS_30_ENABLED_BY_DEFAULT、
        src.production_config が単一情報源)。False にすると従来の全フレーム
        処理を完全再現する (A/B比較・基準データ収集で全フレームが必須な場合用)。
    use_production_recognition: True (既定) で本番採用の認識フラグ群
        (src.production_config.RECOGNITION_ADOPTED: effect-gate/burst-guard-v2/
        transition-merge-guard/burst-gate-open-threshold 0.954/
        hidden-row-burst-guard/match-transition-debounce) を
        recognition_load_default_kwargs() 経由で load_default() へ自動適用する
        (2026-08-13 是正、根因調査の副次発見)。従来はこれらを一切転送しておらず、
        デモ/レビュー動画が本番より劣化した認識で生成されていた
        (2026-08-08 の --early-fire-reaction 付け忘れ事故と同型)。False にすると
        従来通り load_default の関数既定値 (全て無効) で動く (A/B比較用、
        backwards compat)。個別に上書きした引数 (enable_landing_observed_color
        等) との衝突は無い (RECOGNITION_ADOPTED の6キーはそれらと重複しない)。
    resize_1080p: True (既定) で認識入力を 1920x1080 に正規化してから
        RecognitionPipeline.update() に渡す (collect_boards_lean.py:1050 と
        同一の正規化。2026-08-13 是正、根因調査の副次発見)。従来は表示キャンバス
        用サイズ OUT_W/OUT_H(1280x720) へ直接縮小したフレームをそのまま認識にも
        渡しており、BoardRegion の絶対px座標較正 (1920x1080前提) と不整合だった
        (CLAUDE.md「他解像度は1920x1080にリサイズしてから認識する」原則違反。
        720p 入力 + burst-guard 有効でクラッシュすることを診断で実証済み)。
        認識用フレームと表示用フレーム (OUT_W/OUT_H) は独立に生成するため、
        本フラグは出力動画の解像度・レイアウトに一切影響しない。
        False にすると従来 (bit-identical) の挙動に戻る (A/B比較用)。
    enable_pseudo_chain_score_fill: RecognitionPipeline.load_default に渡す
        W7根治①フラグ (2026-08-13、docs/KNOWN_WEAKNESSES.md)。formula/landing
        経路の疑似 ChainEvent の total_score/base_score に simulate 推定値を
        充填する。既定 False = 従来挙動 (backwards compat、未採用のため
        use_production_recognition 経由でも自動 ON にはならない)。
    chain_hold_base_sec / chain_hold_per_step_sec: RecognitionPipeline.
        load_default に渡す CHAIN 保持時間モデルの実測較正値 (2026-08-22
        修正②根治)。`src/recognition_pipeline.py:731-736` (A0、2026-07-24
        計装実測) で「固定項2.61s + 係数1.17s×連鎖数」が原点通過モデルより
        有意に良く適合 (23動画418イベント、R²=0.356) と較正済みだったが、
        **本ファイルには一度も配線されておらず既定 (base=0.0、
        per_step=0.3固定) のまま**だった (2026-08-22 実測で発覚: formula
        機構の再トリガー間隔が保持時間切れの周期と一致し、長い連鎖が
        複数の断片 ChainEvent に分裂する直接原因)。既定 None = 従来通り
        ライブラリ既定 (0.0 / 0.3、backwards compat)。値を指定すると
        `RecognitionPipeline` へそのまま渡る (未検証のため既定では有効化
        しない、`--chain-hold-base-sec 2.61 --chain-hold-per-step-sec 1.17`
        で明示指定して A/B 比較すること)。
    show_chain_count: True で --layout panel の情報パネルに「推定連鎖数
        (ChainEvent.chain_count、simulate 由来) / 実測得点差
        (OjamaAccountSnapshot.chain_total_score_pX、score OCR 由来) / 得点逆算
        連鎖数 (select_chain_count_high_confidence_band、判定不能なら"-")」を
        1P/2P それぞれ並べて表示する (2026-08-15 user要望「れんさ数の方が
        重要指標・認識性能検証としても使える形に」)。単一の断定値でなく
        3値併記なのは、simulate 推定と得点逆算の食い違い自体が認識性能
        検証の価値だから (一致していれば連鎖数は信頼できる、食い違えば
        どちらか/両方が誤っている証拠。食い違い時は情報パネル側でオレンジ
        色に切り替えて強調する)。既定 False = 行を一切描かない
        (bit-identical、backwards compat)。layout="overlay" には未配線
        (_draw_overlay は変更していないため無関係)。
    enable_stable_confirmed_is_dead: True で dump_timeline_path 出力の
        is_dead1/is_dead2 列を「凍結盤面への誤判定」から根治する
        (2026-08-24、`src.board.Board.is_dead` docstring 「STABLE確定盤面
        への静的判定として使う」契約の是正)。
        根因: b1/b2 (settled 再計算に使う確定盤面) は該当 side が
        `BoardState.STABLE` の瞬間にのみ更新され、非STABLE中 (CHAIN/
        GRAVITY_SETTLE/TSUMO_FALL/OJAMA_FALL) は直前の確定盤面が凍結
        される (原則4「非STABLE中は前回STABLE盤面を凍結」通りの仕様動作)。
        `--per-side-settled` 採用後は「相手側だけSTABLEでも再計算」する
        ため、この凍結盤面に対して is_dead() が連鎖中も呼ばれ続け、
        実測で試合時間の約8%・569秒 (CHAINを含む区間に限定、走査器
        D1a の過検出原因の過半) が「連鎖で盤面がほぼ空になっている実画面
        なのに窒息判定 True」という誤りになっていた
        (証拠: logs/is_dead_persist_2026-08-23/)。
        本フラグは **dump_rows (npz保存直前の診断用リストのみ) を
        後処理で遡及訂正**する (ライブの adv_ema/kill_override/描画/
        indicators_v2 には一切触れない、docstring 冒頭の
        `_retroactively_correct_dead_dump_rows` 参照)。
        同一 game_idx 内で「非STABLE中に凍結されていた is_dead」の
        直後に該当 side が STABLE へ復帰する行があれば、その復帰後
        最初の STABLE 行の is_dead 値で遡って上書きする (=連鎖が
        解決してから初めて分かる真の生死を過去に反映)。
        試合終了 (game_idx 変化) まで一度も STABLE に復帰しない区間は
        **一切変更しない** (受け入れ条件「死亡見逃しゼロ」を最優先し、
        「最後の STABLE 時点では生きていた」ケースを誤って False に
        書き換えるリスクを排除する安全側設計、2026-08-24)。
        既定 False = 従来通り b1.is_dead()/b2.is_dead() をそのまま記録
        (bit-identical、backwards compat)。dump_timeline_path が None
        (=dump しない) のときは no-op。
    enable_nonstable_hold_is_dead: True で dump の is_dead1/is_dead2 を
        「own state が STABLE でない行では判定保留 (False+state列で復元可能)」
        として記録する (2026-08-25 Gate 3R-6 案A、
        `_resolve_nonstable_hold_is_dead` docstring 参照)。
        enable_stable_confirmed_is_dead (未来の STABLE 値で遡及訂正する
        dump後処理、リアルタイム不可) と異なり、**現在の state だけ**で
        決めるためリアルタイム (配信オーバーレイ) でも成立する。
        保留した行数は終了時に母数付き (`保留 x/y 行`) で必ず表示する
        (0 が「起きていない」のか「測っていない」のかを区別するため)。
        既定 False = 従来通り (bit-identical、backwards compat)。
        dump_timeline_path が None のときは no-op。
    enable_gross_ledger_dump: True で dump に cap前 gross 累積カウンタ列
        (`_TIMELINE_GROSS_KEYS`) を追加する (2026-08-25 Gate 3R-5、
        `docs/EXCHANGE_GROSS_SUPPLY_DESIGN_2026-08-25.md` §3.3/§3.4)。
        `OjamaAccountingTracker.get_gross_counters()` の cap 前累積値を
        settled 更新ごとに読み、`classify_gross_counter_delta` で
        生成/相殺/着弾/境界ワイプ量/clamp loss/保存則残差へ**推測せず**
        分解する (交換台帳・production_config には一切配線しない、
        dump 専用の読み取り経路)。試合境界では会計トラッカーだけを保持し、
        既存 `_drive_ojama` が旧 pending の境界ワイプ量を累積値へ記録する。
        これにより境界をまたいでも単調カウンタ差分と保存則を検査できる。
        終了時に検査 side 数を分母とした母数付き要約を必ず表示する。
        既定 False = 全 gross_* 列は None のまま (save_timeline_dump は
        列自体を npz へ追加しない、旧 dump と bit-identical、
        backwards compat)。dump_timeline_path が None のときは no-op。
    enable_death_confirm_sequence: True で dump に
        `is_dead1_confirmed`/`is_dead2_confirmed` 列を追加する
        (2026-08-25 Gate 3R-6 本体、`src/death_confirmation.py` 参照)。
        user伝授の死亡確定条件「12段目に設置して連鎖が起きない」「おじゃまが
        降って12段目が埋まる」を「候補→猶予/解除→確定」の3段階として
        リアルタイムに実装する。既存の `is_dead1`/`is_dead2` (即時の占有
        判定、`Board.is_dead()`) は一切変更せず**別列として並存**させる。
        確定条件は【設計訂正 2026-08-25】済み: 当初案「次の事象 (次のツモ
        が置けた/さらにおじゃまが降った)」は死亡すると発火しない
        (死亡=次のツモを置けない、の逆理)ため撤回し、**「own chain が
        始まらないまま `death_next_stationary_sec` 秒ネクストが動かない」**
        簡易検出に差し替えた (`DeathConfirmTracker` docstring 参照。
        底抜け演出検出による根治は後回し、user承認済み)。
        候補・解除・確定・閾値未到達での境界消滅 (いずれも発生源別:
        設置/おじゃま) の回数と確定遅延分布は終了時に母数付きで必ず表示
        する。既定 False = is_dead1_confirmed/is_dead2_confirmed 列は npz
        に一切追加されない (旧 dump と bit-identical、backwards compat)。
        dump_timeline_path が None のときは no-op。
        `src/production_config.py` へは未登録 (測定値を user が確認して
        から採否判断)。
    death_next_stationary_sec: enable_death_confirm_sequence の確定閾値
        (秒、既定 `NEXT_STATIONARY_CONFIRM_SEC`=1.5)。**user 指定の暫定値
        であり Claude がシーンから逆算した値ではない** (2026-08-25 user
        指示。底抜け演出検出による根治までの簡易実装)。感度を事後測定
        できるよう CLI (`--death-next-stationary-sec`) から変更可能。
    """
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"未知の layout: {layout!r} (有効値: {VALID_LAYOUTS})")
    if enable_platt_calibration and enable_phase_calibration:
        raise ValueError(
            "enable_platt_calibration と enable_phase_calibration は同時指定不可"
            " (どちらを使うか呼出元が明示すること)"
        )
    if dump_exchange_episode_timeline_path is not None and not enable_exchange_episode_gate:
        raise ValueError(
            "dump_exchange_episode_timeline_path は enable_exchange_episode_gate=True が必須")
    if enable_exchange_episode_gate and (
        enable_kill_override_chain_completion or enable_kill_override_scale_compare
    ):
        raise ValueError(
            "exchange episode gate と旧 ChainGenerationAccumulator は排他")
    platt_params: PlattCalibrationParams | None = None
    if enable_platt_calibration:
        platt_params = load_platt_calibration(PLATT_CALIBRATION_PATH, required=True)
    phase_platt_params: PhaseCalibrationParams | None = None
    if enable_phase_calibration:
        phase_platt_params = load_phase_platt_calibration(
            PHASE_CALIBRATION_PATH, required=True,
        )
    model = _acquire_model(exclude_video, model_dir)
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
    # 60fps→実効30fps 正規化 (2026-08-12 追加)。stride=1 (30fps 以下の動画、
    # または normalize_fps_30=False) では以下の間引き分岐は常にスキップされ、
    # 挙動は従来と完全一致する (backwards compat)。t (時刻) は絶対フレーム
    # 番号 fi から計算する (下記ループの `t = fi / fps`) ため stride の値に
    # 関わらず実時間 (収集側 collect_boards_lean.py:831 と同じ方式)。
    stride = resolve_normalize_fps_30_stride(fps) if normalize_fps_30 else 1
    effective_fps = fps / stride  # VideoWriter に渡す実効fps (再生時間を保つ)
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
    # render=False (2026-08-11 タイムラインdump工事) は動画出力を一切行わない
    # ため、出力先ディレクトリの作成も VideoWriter の生成も行わない
    # (計算のみを最速で回す用途、backwards compat: render 既定 True)。
    writer: cv2.VideoWriter | None = None
    if render:
        out.parent.mkdir(parents=True, exist_ok=True)
        # layout="panel" は出力キャンバスサイズが異なる (1920x1080)。認識・有利不利の
        # 計算経路 (OUT_W/OUT_H で処理するフレーム) は layout に関わらず不変。
        canvas_size = ((PANEL_CANVAS_W, PANEL_CANVAS_H) if layout == "panel"
                       else (OUT_W, CANVAS_H))
        # stride 間引き後は書き出しフレーム数が 1/stride になるため、出力fps も
        # effective_fps (= fps/stride) にして再生時間 (実時間) を保つ
        # (normalize_fps_30=False/30fps以下入力なら stride=1 で fps と同値、
        # 従来挙動と完全一致)。
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 effective_fps, canvas_size)
    # RECOGNITION_ADOPTED 採用 (2026-08-15): enable_ojama_fall_placement_override が
    # 標準採用7キーの1つに移った。本ファイルは元々この1キーだけ独立CLI引数
    # (--enable-ojama-fall-placement-override, 2026-08-13導入, BooleanOptionalAction)
    # として明示 kwargs 供給しており、下記 ** 展開 (recognition_load_default_kwargs())
    # にもそのまま同名キーが含まれるようになるため、二重供給 (TypeError, 直下コメント
    # 「重複時は TypeError で早期に気付ける設計」参照) を避けて事前にマージする。
    _production_recognition_kwargs = (
        recognition_load_default_kwargs() if use_production_recognition else {}
    )
    # tri-state 解決 (2026-08-15): 明示指定 (not None) は常にそれを使う。未指定
    # (None) かつ production_recognition が OFF のときはキー自体を渡さない
    # (test_no_production_recognition_skips_adopted_kwargs が要求する「RECOGNITION_
    # ADOPTED のキーは一切渡らない」不変条件を、標準7キー目のこのフラグにも
    # 適用するため)。未指定かつ ON のときのみ採用値 True を明示供給する。
    _ojama_override_kwargs: dict[str, bool] = {}
    if enable_ojama_fall_placement_override is not None:
        _ojama_override_kwargs["enable_ojama_fall_placement_override"] = bool(
            enable_ojama_fall_placement_override)
    elif use_production_recognition:
        _ojama_override_kwargs["enable_ojama_fall_placement_override"] = bool(
            _production_recognition_kwargs.get(
                "enable_ojama_fall_placement_override", False))
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
            "enable_puyo_to_empty_hsv_guard", enable_puyo_to_empty_hsv_guard),
        stable_majority_window=_resolve_flag(
            "stable_majority_window", stable_majority_window),
        # OJAMA_FALL誤分類の修正フラグ3種 (2026-08-13 デモレビュー対応)。
        # placement_override のみ 2026-08-15 に RECOGNITION_ADOPTED 採用済のため
        # 上で計算済みの _ojama_override_kwargs (tri-state, キー自体を条件付きで
        # 渡す) を末尾の ** 展開で合流させる。entry_hardening/scoped_exit の2つは
        # 未採用のため従来通り _resolve_flag (ライブラリ既定 False に解決) のまま。
        enable_ojama_fall_entry_hardening=_resolve_flag(
            "enable_ojama_fall_entry_hardening",
            enable_ojama_fall_entry_hardening),
        enable_ojama_fall_scoped_exit=_resolve_flag(
            "enable_ojama_fall_scoped_exit", enable_ojama_fall_scoped_exit),
        # 根治① (W7, 2026-08-13): 疑似 ChainEvent の simulate 推定スコア充填。
        # 未採用のため RECOGNITION_ADOPTED には含めず、直接引き渡す
        # (既定 False、backwards compat)。
        enable_pseudo_chain_score_fill=enable_pseudo_chain_score_fill,
        # CHAIN 保持時間の実測較正値 (2026-08-22 修正②根治、上記docstring参照)。
        # 既定 None ならキー自体を渡さずライブラリ既定 (0.0/0.3) のまま
        # (backwards compat、未採用のため自動 ON にはしない)。
        **({"chain_hold_base_sec": chain_hold_base_sec}
           if chain_hold_base_sec is not None else {}),
        **({"chain_hold_per_step_sec": chain_hold_per_step_sec}
           if chain_hold_per_step_sec is not None else {}),
        # スライド誤検知抑制ガード (2026-08-22 修正③根治、上記docstring参照)。
        # 既定 False = 従来挙動完全維持 (backwards compat、未採用のため自動 ON
        # にはしない、明示指定のみで有効化)。
        enable_slide_exit_min_display_guard=enable_slide_exit_min_display_guard,
        # 本番採用の認識フラグ群 (2026-08-13 是正)。RECOGNITION_ADOPTED の
        # 残り6キーは上記の個別 kwargs と重複しないため ** 展開で安全に合流できる
        # (重複時は TypeError で早期に気付ける設計、静かな上書きは起きない)。
        # enable_ojama_fall_placement_override (7キー目、2026-08-15採用) だけは
        # 上の tri-state 解決 (_ojama_override_kwargs) 経由でマージするため
        # 生の recognition_load_default_kwargs() からは除外する (二重供給防止)。
        **{k: v for k, v in _production_recognition_kwargs.items()
           if k != "enable_ojama_fall_placement_override"},
        **_ojama_override_kwargs)
    import re
    m = re.search(r"(v\d+|video_\d+)", video.name)
    if m and hasattr(pipe, "set_video_id"):
        pipe.set_video_id(m.group(1))
    tracker = OjamaAccountingTracker(); tracker.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    # (2026-08-25 Gate 3R-6 本体) 死亡確定の時間的ロジック。2サイド分の
    # 独立インスタンス (_SideTracker と同じ「1サイド1インスタンス」パターン)。
    # enable_death_confirm_sequence=False でも生成・update() 自体は行うが
    # (コスト僅少)、dump 列へは一切反映しない (bit-identical)。
    # stationary_confirm_sec: user 指定の暫定閾値 (既定 NEXT_STATIONARY_
    # CONFIRM_SEC=1.5秒、--death-next-stationary-sec で変更可能)。
    death_tracker1 = DeathConfirmTracker(stationary_confirm_sec=death_next_stationary_sec)
    death_tracker2 = DeathConfirmTracker(stationary_confirm_sec=death_next_stationary_sec)
    death_confirm_stats = DeathConfirmStats()
    ps1 = ps2 = BoardState.MENU
    b1 = b2 = None
    adv_ema = 0.0
    p1_last = 0.5
    model_adv_last = float("nan")
    drivers: list[tuple[str, float]] = []
    # kill_override が直近の settled 再計算で実際に発火したか (2026-08-22
    # 修正④、_kill_override_attribution_entry 参照)。dump 用の drivers
    # (raw モデル寄与度、self己無矛盾性が要件) には一切混ぜず、表示専用の
    # 別変数として持ち回る。settled 再計算のたびに更新、非 settled 中は
    # 直前値を保持 (adv_ema 等と同じ「凍結」パターン)。
    kill_override_note: "tuple[str, float] | None" = None
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
    # #4/#5 修正 (--counter-defender-only) 用。脅威が無い間は None/0.0 のまま
    # (パネル表示行を出さない・統合式へも寄与させない、既定挙動)。
    defender_side: str | None = None
    incoming_ojama: float = 0.0
    # #3 修正 (--counter-remaining-time) 用の条件付き分布テーブル (E[最終|N到達])。
    # フラグ OFF 時は読み込まない (I/O 無駄を避ける、無効時は完全に旧経路)。
    _chain_len_table = (
        _load_chain_length_conditional_table() if enable_counter_remaining_time else {}
    )
    ptracker = PressureTracker()
    fctracker = RealtimeForecastTracker()
    svtracker = ScoreLeadTracker()
    # 能力低下ベースの圧力 (2026-08-09)。 enable_capability_pressure=True の
    # ときだけ使う。 リセットは _fresh_trackers と同じタイミングで行う。
    cap_ptracker = CapabilityPressureTracker()
    # 打ち合い応手確率 (2026-08-09 user採用)
    counter_tracker = CounterReachTracker()
    # 主因除外リスト (2026-08-11 Phase1-3)。show_excluded_attribution=True の
    # ときだけ空にしてデバッグ表示 (除外前の全候補) に切り替える。
    attribution_exclude = () if show_excluded_attribution else ATTRIBUTION_EXCLUDED_INDICATORS
    hcache = HeavyAdvCache(model, attribution_exclude=attribution_exclude)
    efire_tracker = EarlyFireTracker()  # (早期発火) 既定 OFF 時も生成のみ(コスト僅少)
    # 修正① 既定OFF時も生成のみ(コスト僅少)。累積モードは既定 False (2026-08-22
    # user判断、ChainGenerationAccumulator docstring 参照)。
    chain_gen_tracker = ChainGenerationAccumulator(
        accumulate=enable_kill_override_chain_gen_accumulate)
    # kill_override 連鎖完走後是正 (修正①) の当該フレーム分プレースホルダ。
    # enable_kill_override_chain_completion=True の間は毎フレーム
    # chain_gen_tracker.update() で上書きされる (settled ゲートの外側で
    # 更新、詳細はその呼出し箇所のコメント参照)。False の間は未使用。
    chain_gen1 = chain_gen2 = 0.0
    chain_gen_before1: "Board | None" = None
    chain_gen_before2: "Board | None" = None
    # (2026-08-24 B案) 致死上書きの確信度ゲート。既定 OFF 時は生成のみで
    # apply() を一切呼ばない (bit-identical、コストゼロ)。
    kill_gate = KillOverrideConfidenceGate()
    # (2026-08-24 A案(i-a)) 未登録送付分トラッカー。既定 OFF 時は生成のみ。
    unregistered_sent_tracker = PostChainUnregisteredSentTracker()
    unregistered_extra_p1 = unregistered_extra_p2 = 0.0
    # れんさ数表示 (--show-chain-count、2026-08-15)。show_chain_count=False 時も
    # 生成・update() 自体は行うが (コスト僅少)、_draw_panel_layout へ渡す文字列を
    # 組み立てないため描画には一切現れない (bit-identical、後方互換)。
    chain_display_tracker = ChainCountDisplayTracker()
    # #9 両者同時発火の決着先読み (2026-08-13)。enable_resolved_exchange_eval=False
    # 時も生成のみ (コスト僅少、update() 呼び出し自体は毎フレーム行うが chain_event
    # が両方非None になるまで内部で何もしない)。
    resolved_tracker = ResolvedExchangeTracker(
        model, attribution_exclude=attribution_exclude,
        enable_decisive_amplify=enable_resolved_decisive_amplify,
        enable_live_defender_reeval=enable_resolved_live_defender,
        enable_live_defender_strict=enable_resolved_live_defender_strict,
        enable_kill_override_counter_aware=enable_resolved_kill_override_counter_aware,
        enable_resolved_victim_gen_live=enable_resolved_victim_gen_live,
        enable_episode_physical_redecide=enable_resolved_episode_physical_redecide,
        enable_episode_physical_consistency_guard=(
            enable_resolved_episode_physical_consistency_guard),
        enable_pending_landing_gate=enable_resolved_pending_landing_gate,
        enable_counter_placement_reuse=enable_resolved_counter_placement_reuse,
        enable_counter_budget_quantize=enable_resolved_counter_budget_quantize,
        enable_absolute_chain_end=enable_resolved_absolute_chain_end)
    # (2026-08-26 決着ホールド根治) 試合境界をまたいで母数付きカウンタを積む器。
    # 同じ dict オブジェクトを後続の tracker へ渡すため、動画全体の合計になる。
    _abs_end_stats_carry = resolved_tracker.abs_end_stats
    _episode_physical_stats_carry = resolved_tracker.episode_physical_stats
    prev_score1: int | None = None  # (改修1) スコアリセット検知用の前フレーム値
    prev_score2: int | None = None
    history: list[tuple[float, float]] = []  # (試合開始からの秒, 有利不利) 累積
    total_dur = max(1.0, (n / fps) - start_sec)  # グラフ横軸の総尺 (1試合目/境界無し用)
    # (#8 修正、2026-08-13) 下部グラフの横軸原点。試合境界を検知するたびに
    # 「(t - start_sec) - game_start_sec」が0から再スタートするよう更新する
    # (history.clear() だけでは history に積む座標が動画全体の絶対時間の
    # ままだったため、境界後の曲線が絶対位置=途中から始まって見えるバグ
    # だった。docs/DEMO_REVIEW_2026-08-13.md #8)。
    game_start_sec = 0.0
    # グラフ横軸のスケール。試合ごとに巻き直す (簡明な実装を優先、境界検知の
    # たびに「その時点からの残り動画尺」で再計算する。1試合目/境界が一度も
    # 起きない動画では total_dur のまま=従来と完全に同一の挙動)。
    graph_total = total_dur
    step = max(1, int(round(sample_interval * fps)))
    written = 0
    # タイムラインdump (2026-08-11 追加)。dump_timeline_path が None の間引き
    # 判定は「常に空リストを回すだけ」で済むよう、dump_rows は常に生成する
    # (dump 無効時のコストは空リスト append 判定1回のみ、無視できる)。
    game_idx = 0  # スコアリセット検知で進む試合境界カウンタ (video 内ローカル)
    # 【P1 是正 2026-08-26、Codex 第26報レビュー】`_detect_score_reset` は
    # 両スコアが SCORE_NEAR_ZERO_THRESHOLD 以下の間**毎フレーム True** になる。
    # 従来は `game_idx += 1` だけが debounce されており、
    # `resolve_boundary_confirmations()` は debounce の外にあったため、
    # 実境界 約6件に対して total_boundaries=715 回呼ばれていた (Codex 実測)。
    # `on_game_boundary()` は毎回 `_post_boundary_armed=False` へ戻すので、
    # 新試合冒頭の再武装や死亡候補を取りこぼしうる。
    # そこで reset 信号を**立ち上がりでラッチ**し、低得点状態が何秒続いても
    # 同一境界を再受理しないようにする。
    #
    # 【第2版 2026-08-26、Codex 第27報レビュー】第1版は `resolve_boundary_
    # confirmations` だけを正式境界へ寄せ、`game_idx` 加算と各種トラッカーの
    # 初期化は従来の「raw reset の各フレーム」に残していた。しかしそれでは
    # 低得点が debounce 秒より長く続くと**正式境界なしに game_idx が進み**、
    # 死亡候補の `_pending_game_idx` と次の `_ending_game_idx` が食い違って
    # **真の死亡が `rejected_game_idx_mismatch` になりうる**。
    # 第1版が据え置いた理由 (OFF 出力の bit-identical 維持) は、pre-gate
    # 条件1〜4 の再取得が決定済みであるため成立しない。よって境界処理
    # 全体を**正式境界イベント1回**へ統合する。
    _score_reset_latched = False       # 直前フレームで reset 信号が立っていたか
    _last_formal_boundary_t: float | None = None  # 最後に「正式受理」した境界の時刻
    dump_rows: list[TimelineDumpRow] = []
    display_dump_rows: list[DisplayTimelineRow] = []
    episode_dump_rows: list[EpisodeTimelineRow] = []
    episode_adapter = (
        _LiveEpisodeOverlayAdapter() if enable_exchange_episode_gate else None)
    # (2026-08-25 Gate 3R-6 案A) is_dead 保留の母数付きカウンタ。
    # enable_nonstable_hold_is_dead=False では一切 record されない
    # (dump 経路も従来と bit-identical)。
    isdead_hold_stats = _IsDeadHoldStats()
    # (2026-08-25 Gate 3R-5) gross累積カウンタ dump 用の前回値。dump 有効時は
    # 会計 tracker 自体を試合境界越しに保持するため、ここも破棄せず単調差分を
    # 継続する。dump 無効時は一切更新されず _build_gross_dump_fields も
    # 呼ばれない (bit-identical、backwards compat)。
    prev_gross_counters: "GrossOjamaCounters | None" = None
    prev_gross_pending_unc: tuple[int, int] | None = None
    gross_dump_stats = _GrossDumpStats()
    for fi in range(start_frame, n):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        # --- 60fps→30fps 正規化 (2026-08-12 追加) ---
        # cap.read() は毎フレーム呼んでデコードし (シーク禁止、収集側
        # collect_boards_lean.py:819-827 と同じ方式)、stride 非対象フレームは
        # pipe.update() もオーバーレイ計算も一切行わず出力動画にも書かない
        # (stride=1 の時は常に False で従来挙動と完全一致、backwards compat)。
        if (fi - start_frame) % stride != 0:
            continue
        # show_recognition=True かつ render=True の時のみネイティブ解像度の
        # コピーを保持する (認識色 overlay 描画用。推論には使わないため計算
        # 経路は不変。render=False では描画自体を行わないため無駄なコピーを
        # 避ける、2026-08-11 追加)。
        raw_native = frame.copy() if (show_recognition and render) else None
        # --- 認識入力の 1080p 正規化 (2026-08-13 是正) ---
        # RecognitionPipeline.update() は BoardRegion の絶対px座標較正
        # (DEFAULT_P1_REGION 等、1920x1080 前提) を使うため、collect_boards_lean.py
        # (TARGET_W/TARGET_H=1920,1080) と同じ正規化をここでも行う必要がある。
        # 従来は下記の表示キャンバス用リサイズ (OUT_W/OUT_H=1280x720) の結果を
        # そのまま認識にも渡しており座標系が不整合だった。認識用フレーム
        # (recog_frame) と表示用フレーム (frame) は元の native frame から
        # 独立に生成する (表示解像度は resize_1080p の有無に関わらず不変)。
        if resize_1080p:
            recog_frame = (
                frame if frame.shape[:2] == (NATIVE_H, NATIVE_W)
                else cv2.resize(frame, (NATIVE_W, NATIVE_H),
                                interpolation=cv2.INTER_AREA)
            )
        else:
            # 逃げ道 (--no-resize-1080p): 従来と bit-identical
            # (表示用縮小フレームをそのまま認識に渡す、A/B比較用)。
            recog_frame = (
                frame if frame.shape[:2] == (OUT_H, OUT_W)
                else cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            )
        if frame.shape[:2] != (OUT_H, OUT_W):
            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        t = fi / fps
        # お邪魔会計は密な駆動が必須のため pipe.update / _drive_ojama は毎フレーム。
        r = pipe.update(fi, t, recog_frame)
        # (改修1) 試合境界(score大幅減少/両者0付近)を検知したら凍結盤面・持続
        # トラッカー・表示状態を全て初期化する(前試合の「幻の差」持ち越し防止)。
        _reset_now = _detect_score_reset(
            r.p1.score, r.p2.score, prev_score1, prev_score2)
        # 【P1 是正 第2版】この reset 信号を「正式な境界イベント」として受理する
        # かをまず決める (判定本体は accept_formal_boundary の docstring 参照)。
        # 受理しなかったフレームでは境界処理を**一切**行わない。
        _formal_boundary = accept_formal_boundary(
            reset_now=_reset_now, latched=_score_reset_latched,
            t_sec=t, last_formal_t=_last_formal_boundary_t)
        if _formal_boundary:
            print(f"[reset] t={t:.1f}s score大幅減少/0付近を検知 -> 評価を互角にリセット")
            # (2026-08-25 第3版、Codex 承認条件対応) 死亡確定の境界判定
            # (`resolve_boundary_confirmations`) に渡す game_idx は、この
            # フレームで加算される**前**の値 (=今終わろうとしている試合の
            # game_idx) を使う。加算後の値を渡すと、候補発生時に記録した
            # `_pending_game_idx` (終わろうとしている試合の game_idx) と
            # 一致しなくなり、正常系まで `rejected_game_idx_mismatch` に
            # 誤判定してしまうため (`DeathConfirmTracker.on_game_boundary`
            # docstring 参照)。
            # 【P1 是正 第2版】加算前スナップショットと game_idx 加算を、この
            # 正式境界イベントの中で**1回だけ**行う。両者が同じイベントで動く
            # ので、候補発生時の `_pending_game_idx` と食い違わない。
            _ending_game_idx = game_idx
            _last_formal_boundary_t = t
            game_idx += 1
            b1 = b2 = None
            adv_ema, p1_last, drivers = 0.0, 0.5, []
            model_adv_last = float("nan")
            kill_override_note = None  # 前試合の安全弁理由を持ち越さない (修正④)
            ukey1 = ukey2 = sat1 = sat2 = 0.0
            counter_p1 = counter_p2 = float("nan")
            defender_side, incoming_ojama = None, 0.0
            history.clear()
            # (#8 修正) グラフ横軸をこの試合の開始 (=現在の t - start_sec) を
            # 原点にリセットし、スケールも「この時点からの残り動画尺」に
            # 巻き直す (試合ごとに巻き直してよい、簡明な実装を優先)。
            game_start_sec, graph_total = _reset_graph_origin(t, start_sec, n, fps)
            (tracker, tp1, tp2, ptracker, fctracker, svtracker, hcache,
             efire_tracker, chain_gen_tracker) = _fresh_trackers(
                model, attribution_exclude=attribution_exclude,
                # 境界ワイプ量を失わないため、dump 有効時だけ会計 tracker を保持。
                # OFF は従来どおり新規 tracker を生成し、実行経路を変えない。
                accounting_tracker=(
                    tracker if (enable_gross_ledger_dump or enable_exchange_episode_gate)
                    else None),
                enable_chain_gen_accumulate=enable_kill_override_chain_gen_accumulate)
            chain_gen1 = chain_gen2 = 0.0
            chain_gen_before1 = chain_gen_before2 = None
            # (2026-08-24) 確信度ゲート/未登録送付分も前試合の状態を持ち越さない
            kill_gate = KillOverrideConfidenceGate()
            unregistered_sent_tracker = PostChainUnregisteredSentTracker()
            unregistered_extra_p1 = unregistered_extra_p2 = 0.0
            cap_ptracker = CapabilityPressureTracker()
            counter_tracker = CounterReachTracker()
            # 決着ホールド中に試合自体が終わった場合は、未解除ではなく正式な
            # 物理終端として母数を閉じてから、次試合用 tracker へ差し替える。
            if enable_resolved_absolute_chain_end:
                resolved_tracker.on_game_boundary()
            resolved_tracker = ResolvedExchangeTracker(
                model, attribution_exclude=attribution_exclude,
                enable_decisive_amplify=enable_resolved_decisive_amplify,
                enable_live_defender_reeval=enable_resolved_live_defender,
                enable_live_defender_strict=enable_resolved_live_defender_strict,
                enable_kill_override_counter_aware=enable_resolved_kill_override_counter_aware,
                enable_resolved_victim_gen_live=enable_resolved_victim_gen_live,
                enable_episode_physical_redecide=(
                    enable_resolved_episode_physical_redecide),
                enable_episode_physical_consistency_guard=(
                    enable_resolved_episode_physical_consistency_guard),
                episode_physical_stats=(
                    _episode_physical_stats_carry
                    if (enable_resolved_episode_physical_redecide
                        or enable_resolved_episode_physical_consistency_guard)
                    else None),
                enable_pending_landing_gate=enable_resolved_pending_landing_gate,
                enable_counter_placement_reuse=enable_resolved_counter_placement_reuse,
                enable_counter_budget_quantize=enable_resolved_counter_budget_quantize,
                enable_absolute_chain_end=enable_resolved_absolute_chain_end)
            # (2026-08-26 決着ホールド根治) 判定状態は試合ごとに作り直すが、
            # **母数付きカウンタだけは引き継ぐ**。引き継がないと動画全体の
            # サマリが「最後の試合ぶんだけ」になり、母数が壊れる
            # (memory feedback_zero_needs_denominator_2026-08-25)。
            # 既定 OFF では代入自体を行わない (実行経路を変えない)。
            if enable_resolved_absolute_chain_end:
                resolved_tracker.abs_end_stats = _abs_end_stats_carry
            # 試合境界で前試合の保持中パルス(推定連鎖数/実測得点差)を持ち越さない
            # (2026-08-15、前試合最後の連鎖表示が次試合冒頭に残る誤表示を防止)。
            chain_display_tracker = ChainCountDisplayTracker()
            # (2026-08-25 Gate 3R-6 本体、実測で根治) 死亡確定の候補/猶予中の
            # 遷移追跡は即座にリセットするが、確定済みフラグは新規インスタンス
            # への差し替えでは消さない (`DeathConfirmTracker.on_game_boundary`
            # docstring 参照)。実測 (2P・実試合2・t=223 真の窒息) で、旧実装
            # (即時差し替え) は決着演出〜結果表示中の settled recompute 空白
            # 区間の内側でこの境界検知が発生し、確定フラグが dump に一度も
            # 現れず見逃しに見える不具合があった。
            # 【設計の穴の是正 2026-08-25 第2版・第3版 (Codex 承認条件対応)】
            # 境界検知の瞬間にまだ猶予中 (未確定・未解除) だった候補は
            # 「閾値未到達で消滅」させるのではなく、以下をすべて満たす場合に
            # 限り**その場で死亡確定する** (`resolve_boundary_confirmations`
            # 内部で `on_game_boundary()` を呼ぶ、`death_confirmation.py`
            # モジュール docstring「候補のまま試合終了 = 死亡確定」/
            # 「Codex 条件付き承認・追加要件」節参照): (1) 候補発生時と同一
            # game_idx、(2) own CHAIN 開始で解除されていない、(3) next変化/
            # 新規ツモ落下等の生存証拠が無い、(4) 両側同時 pending
            # (ambiguous) ではない。人が本当に詰んだとき試合はその場で
            # 終わるため、ネクスト不動猶予の条件は真の死亡では原理的に
            # 満たせない (実測: t=223 真の窒息が候補後0.37秒で試合終了、
            # 旧設計では検出0/5083行だった)。境界処理の結果は母数付きで
            # 記録する (0/0 と区別するため enable_death_confirm_sequence の
            # ときのみ記録、既定 False では record 自体を呼ばない)。
            # 【P1 是正】`resolve_boundary_confirmations` の docstring は元から
            # 「1境界イベントにつき本関数を1回だけ呼ぶこと」と定めており、契約が
            # 書かれていたのに呼出側が守っていなかった。第2版ではブロック全体が
            # 正式境界の内側なので、ここでの追加判定は不要。
            resolve_boundary_confirmations(
                death_tracker1, death_tracker2, game_idx=_ending_game_idx,
                stats=(death_confirm_stats
                       if enable_death_confirm_sequence else None))
        # 【P1 是正 2026-08-26、要件3】reset 信号の立ち上がりラッチを更新する
        # (判定本体は update_score_reset_latch の docstring 参照)。
        _score_reset_latched = update_score_reset_latch(
            _score_reset_latched, _reset_now, r.p1.score, r.p2.score)
        prev_score1, prev_score2 = r.p1.score, r.p2.score
        if r.p1.state == BoardState.STABLE and r.p1.confirmed_board is not None:
            b1 = r.p1.confirmed_board
        if r.p2.state == BoardState.STABLE and r.p2.confirmed_board is not None:
            b2 = r.p2.confirmed_board
        # (2026-08-25 Gate 3R-6 本体) settled ゲートの外側 (=非STABLE中も
        # 毎フレーム) で state 遷移を観測する必要がある (efire_tracker.update()
        # と同じ理由、死亡候補は STABLE 以外の state 遷移も見て判定するため)。
        # 既定 False (enable_death_confirm_sequence) でも呼ぶだけでコスト僅少
        # (dump へは反映しない、bit-identical)。
        _death_occ1 = (
            r.p1.state == BoardState.STABLE and b1 is not None and b1.is_dead())
        _death_occ2 = (
            r.p2.state == BoardState.STABLE and b2 is not None and b2.is_dead())
        # (2026-08-25 設計訂正) 確定条件は「ネクスト不動」(次の事象では
        # ない、死亡確定の項参照)。next_pair は既存公開フィールド
        # (`SideResult.next_pair`)。テスト用モック (_FakeResult 等) には
        # 存在しない場合があるため getattr で安全に None へ倒す
        # (既存の同種防御パターンと同じ)。is_match_active (2026-08-25
        # 実測で追加): まちうけ画面の背景誤検出+ネクスト固定表示による
        # 誤確定を防ぐ (DeathConfirmTracker.update docstring 参照)。
        _match_active = getattr(r, "is_match_active", True)
        _death_event1, _death_delay1 = death_tracker1.update(
            r.p1.state.name, _death_occ1, t,
            next_key=getattr(r.p1, "next_pair", None),
            is_match_active=_match_active, game_idx=game_idx)
        _death_event2, _death_delay2 = death_tracker2.update(
            r.p2.state.name, _death_occ2, t,
            next_key=getattr(r.p2, "next_pair", None),
            is_match_active=_match_active, game_idx=game_idx)
        if enable_death_confirm_sequence:
            death_confirm_stats.record(_death_event1, _death_delay1)
            death_confirm_stats.record(_death_event2, _death_delay2)
        snap = _drive_ojama(tracker, r.p1, r.p2, ps1, ps2, t,
                            tracker_p1=tp1, tracker_p2=tp2, pipeline=pipe)
        ps1, ps2 = r.p1.state, r.p2.state
        episode_drive: _EpisodeDriveResult | None = None
        episode_hard_candidate = False
        episode_hard_applied = False
        episode_hard_path = ""
        episode_hard_reason = ""
        if episode_adapter is not None:
            episode_dead1 = (
                death_tracker1.resolved_is_dead_for_game(game_idx)
                if enable_death_confirm_sequence else _death_occ1)
            episode_dead2 = (
                death_tracker2.resolved_is_dead_for_game(game_idx)
                if enable_death_confirm_sequence else _death_occ2)
            episode_drive = episode_adapter.update(
                result=r, accounting=tracker, account_snapshot=snap,
                t_sec=t, game_idx=game_idx,
                room1=(board_room(b1) if b1 is not None else BOARD_ROWS * BOARD_COLS),
                room2=(board_room(b2) if b2 is not None else BOARD_ROWS * BOARD_COLS),
                dead1=episode_dead1, dead2=episode_dead2)
        # れんさ数表示 (--show-chain-count) 用の保持更新。adv/history/その他の
        # コアロジックには一切影響しない (表示専用トラッカー、毎フレーム更新
        # してもコスト僅少なため show_chain_count フラグでのゲートは不要)。
        chain_display_tracker.update(r.p1.chain_event, r.p2.chain_event, snap, t)
        # (早期発火) chain_event 検知フレームで即座に速報バイアスを更新する。
        # settled ゲートの外側 (= 非STABLE中も毎フレーム) で呼ぶことが本修正の要
        # (2026-07-29 userレビュー指摘1/2対処、詳細は EarlyFireTracker docstring)。
        if enable_early_fire_reaction:
            efire_tracker.update(r.p1.chain_event, r.p2.chain_event, b2, b1,
                                 tracker._elapsed(t))
        # (2026-08-22 修正① 改良②) kill_override 連鎖完走後是正の生成量累積。
        # settled ゲートの外側で毎フレーム呼ぶことが要 (efire_tracker.update と
        # 同じ理由: formula 機構の再トリガー [trigger_sec の変化] を settled
        # 間引きの間に取りこぼさないため、ChainGenerationAccumulator docstring
        # 参照)。既定 False では呼ぶだけでコスト僅少 (busy 判定のみ)。
        # (2026-08-24 A案) scale-compare も生成量推定を使うため同じ条件で更新する
        # (完走時の未登録送付分の捕捉に必要。chain_completion=False でも
        # scale-compare=True なら回す。両方 False なら従来通り呼ばない =
        # bit-identical)。
        if enable_kill_override_chain_completion or enable_kill_override_scale_compare:
            (chain_gen1, chain_gen_before1,
             chain_gen2, chain_gen_before2) = chain_gen_tracker.update(
                r.p1, r.p2, tracker._elapsed(t))
        # (2026-08-24 A案(i-a)) 未登録送付分の更新も settled ゲートの外側で
        # 毎フレーム行う (busy→非busy の完走遷移・会計登録の増分・盤面着弾は
        # settled フレーム以外でも起きるため)。
        if enable_kill_override_scale_compare:
            unregistered_extra_p1, unregistered_extra_p2 = (
                unregistered_sent_tracker.update(
                    r.p1, r.p2, snap, b1, b2, chain_gen1, chain_gen2, t))
        # (#9 決着先読み) settled ゲートの外側 (= 非STABLE中も毎フレーム) で
        # 呼ぶことが要 (chain_event の両側同時アクティブ検知は STABLE ゲート内
        # では起きない、EarlyFireTracker と同じ理由)。
        resolved_active, resolved_just_deactivated = False, False
        if enable_resolved_exchange_eval:
            # b1/b2 (受け側の現在 STABLE 確定盤面、直上で更新済み) を渡す
            # (指摘13、enable_resolved_live_defender=False では tracker 内部で
            # 未使用のため無害)。
            resolved_active, resolved_just_deactivated = resolved_tracker.update(
                r.p1, r.p2, snap, tracker._elapsed(t), t_sec=t, b1=b1, b2=b2,
                physical_net_raw=(
                    episode_drive.snapshot.ledger.net_raw
                    if episode_drive is not None else None),
                physical_is_unresolved=(
                    episode_drive.snapshot.ledger.is_unresolved
                    if episode_drive is not None else False),
                physical_chain_id_p1=(
                    episode_drive.snapshot.latest_chain_id_p1
                    if episode_drive is not None else None),
                physical_chain_id_p2=(
                    episode_drive.snapshot.latest_chain_id_p2
                    if episode_drive is not None else None))
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
        # (検収指摘⑤ 2026-08-14) 決着ホールド中は settled 再計算そのものを
        # 止める (resolved_hold_freezes_settled docstring 参照)。フラグ無効時
        # /非hold中は常に False = 従来判定を素通り (backwards compat)。
        if resolved_hold_freezes_settled(enable_resolved_exchange_eval, resolved_active):
            settled = False
        # #9 決着先読みの保持値を adv_ema/p1_last へ引き継ぐ際 (deactivate 直後)、
        # 同一フレームで settled 再計算が走っていればそちらを優先する
        # (真の観測後盤面 > 決着先読みの1回評価、フラグ無効時は常に False)。
        settled_ran_this_frame = False
        if b1 is not None and b2 is not None and settled:
            settled_ran_this_frame = True
            # 重い盤面由来(モデルadv/threat/ukeyasusa/飽和連鎖)はキャッシュ間引き、安価な圧力/リードは毎フレーム
            model_adv, threat, drivers, ukey1, ukey2, sat1, sat2 = hcache.update(
                b1, b2, snap, r.p1, r.p2, tracker._elapsed(t))
            model_adv_last = float(model_adv)
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
            # (2026-08-13 #3/#4/#5 修正、docs/DEMO_REVIEW_2026-08-13.md 参照)
            # 時間予算の算出 (#3) と 受け側限定・実飛来量ベース (#4/#5) は
            # それぞれ独立フラグで、下ごしらえ (_detect_chain_attacker) だけ共有する。
            attack_obs = _detect_chain_attacker(r.p1, r.p2, t)
            _budget = _resolve_counter_time_budget(
                attack_obs, t, enable_counter_remaining_time, _chain_len_table)
            if not enable_counter_reach:
                counter_adv, counter_p1, counter_p2 = 0.0, float("nan"), float("nan")
                defender_side, incoming_ojama = None, 0.0
            elif enable_counter_defender_only:
                defender_side, incoming_ojama = _resolve_defender_threat(
                    attack_obs, snap, tracker._elapsed(t))
                _, counter_p1, counter_p2 = counter_tracker.update(
                    b1, b2, _budget,
                    next1=getattr(r.p1, "next_pair", None),
                    next2=getattr(r.p2, "next_pair", None),
                    t_sec=t, defender_side=defender_side,
                    threshold_ojama=incoming_ojama if defender_side else None,
                )
                defender_prob = (
                    counter_p1 if defender_side == "1P"
                    else counter_p2 if defender_side == "2P" else float("nan")
                )
                counter_adv = (
                    _counter_defender_adv(defender_side, defender_prob, incoming_ojama, b1, b2)
                    if defender_side is not None else 0.0
                )
            else:
                defender_side, incoming_ojama = None, 0.0
                counter_adv, counter_p1, counter_p2 = counter_tracker.update(
                    b1, b2, _budget,
                    next1=getattr(r.p1, "next_pair", None),
                    next2=getattr(r.p2, "next_pair", None),
                    t_sec=t,  # 時間ベース間引き (2026-08-12、CounterReachTracker.update 参照)
                )
            adv = (W_PRESSURE * pres + W_FORECAST * fc
                   + W_MODEL * model_adv + W_THREAT * threat
                   + W_COUNTER * counter_adv) + sl_bias
            adv = max(-100.0, min(100.0, adv))
            # room1/room2 はタイムラインdump (2026-08-11) でも再利用するため
            # ローカル変数に保持する (値は従来の board_room(b1)/board_room(b2)
            # 直呼び出しと完全に同一、キャッシュしただけで挙動不変)。
            room1, room2 = board_room(b1), board_room(b2)
            # (2026-08-22 修正) 従来は fctracker.inc1/inc2 (得点差÷70ヒューリ
            # スティック+ツモ毎30個減衰の粗い推定) を渡していたが、実測
            # (t=886.5s) で真値は2Pに216個なのに inc1=616.73/inc2=0.00 と
            # 1P側へ逆方向に出る事故があった (scripts/_diag_kill_override_
            # wiring_2026-08-22.py)。確定会計 (OjamaAccountingTracker.snap.
            # pending_p1/p2、resolve_mutual_exchange や hold_after_kill_
            # override の _incoming_total_p1/p2 と同系統の値) に差し替える。
            # fc(4成分ブレンドの予告項、直上 fctracker.update() 呼出し)は
            # 据え置き(安全弁の入力だけを正す局所修正、二重計上ではない)。
            # (2026-08-22 修正①、改良②で生成量を複数トリガー累積対応に変更)
            # 死ぬと判定されうる側が自分の連鎖 (CHAIN/GRAVITY_SETTLE) を
            # 撃っている最中なら、連鎖前の凍結盤面の空きと相殺前の額面
            # pending ではなく「連鎖完走後」の値で判定する
            # (`_kill_override_chain_completion_inputs`/
            # `ChainGenerationAccumulator` docstring 参照。既定 False では
            # room1/room2・snap.pending_p1/p2 をそのまま返すため bit-identical)。
            # (2026-08-24 A案「規模の比較」) scale-compare 有効時は kill の
            # pending 基礎値を「cap 前の実額 (並行帳簿) + 未登録送付分」に
            # 差し替える。None のままなら従来入力 (snap.pending_p1/p2) =
            # bit-identical。表示用 pending (cap 済み) はここでは触らない。
            if enable_kill_override_scale_compare:
                kill_base_pend1 = (
                    float(snap.pending_p1_uncapped) + unregistered_extra_p1)
                kill_base_pend2 = (
                    float(snap.pending_p2_uncapped) + unregistered_extra_p2)
            else:
                kill_base_pend1 = kill_base_pend2 = None
            if enable_kill_override_chain_completion:
                kroom1, kroom2, kpending1, kpending2 = (
                    _kill_override_chain_completion_inputs(
                        snap, b1, b2, room1, room2,
                        chain_gen1, chain_gen_before1,
                        chain_gen2, chain_gen_before2,
                        pending_p1_override=kill_base_pend1,
                        pending_p2_override=kill_base_pend2))
            elif enable_kill_override_scale_compare:
                kroom1, kroom2 = room1, room2
                kpending1, kpending2 = kill_base_pend1, kill_base_pend2
            elif episode_drive is not None and episode_drive.snapshot.ledger.is_unresolved:
                # 条件5は旧ChainGenerationAccumulatorと排他。単に旧経路をOFFに
                # するだけでは掛け算式の累積生成量が致死判定から消えるため、
                # chain_idで統合・相殺済みの台帳純残量を置換入力として使う。
                kroom1, kroom2 = room1, room2
                kpending1, kpending2 = _episode_kill_override_inputs(
                    episode_drive.snapshot.ledger.net_raw)
            else:
                kroom1, kroom2 = room1, room2
                kpending1, kpending2 = float(snap.pending_p1), float(snap.pending_p2)
            adv_pre_kill_override = adv
            adv = kill_override(adv, kpending1, kpending2,  # (B)キル判定で生存側へ
                                kroom1, kroom2)
            # (2026-08-24 B案) 確信度ゲート: 反転クールダウン + 持続確認 +
            # 未確定中の ±90 上限。既定 OFF ではこの分岐自体を通らない
            # (bit-identical)。ゲートが上書きを保留した場合 adv は
            # adv_pre_kill_override へ戻るため、下の kill_override_note も
            # 自然に None になる (矛盾した主因表示を出さない)。
            if enable_kill_override_hysteresis:
                adv = kill_gate.apply(adv_pre_kill_override, adv, t)
            if episode_drive is not None:
                (adv, episode_hard_candidate, episode_hard_applied,
                 episode_hard_reason) = _apply_episode_hard_override_gate(
                    adv_pre_kill_override, adv,
                    episode_drive.snapshot.ledger.allows_hard_override,
                    is_unresolved=episode_drive.snapshot.ledger.is_unresolved,
                    hard_override_target=(
                        episode_drive.snapshot.ledger.hard_override_target),
                    physical_net_raw=episode_drive.snapshot.ledger.net_raw)
                episode_hard_path = _append_episode_hard_path(
                    episode_hard_path, "live", episode_hard_candidate)
            # (2026-08-22 修正④) 安全弁 (kill_override) が結論を決めた場合、
            # その理由を主因表示に明示する (`_kill_override_attribution_entry`
            # docstring 参照)。予測値 (adv/p1) には一切影響しない表示専用の
            # 追加情報で、dump 用の `drivers` (raw モデル寄与度) には混ぜない。
            kill_override_note = (
                _kill_override_attribution_entry(
                    adv_pre_kill_override, adv, kpending1, kpending2,
                    kroom1, kroom2)
                if enable_kill_override_attribution and adv != adv_pre_kill_override
                else None
            )
            p1 = adv_to_winprob(adv)  # 表示用勝率(較正sigmoid or 直線)
            # Platt後段校正 (全位相共通 or 位相別、2026-08-11 Phase1-2)。
            # 両方 False (既定) なら progress 計算自体を省き従来経路とビット一致させる。
            if platt_params is not None or phase_platt_params is not None:
                progress = _match_progress_for_boards(b1, b2)
                _chosen_platt = _resolve_display_platt(
                    progress, platt_params, phase_platt_params)
                adv, p1 = _apply_platt_to_display(adv, p1, _chosen_platt)
            adv_ema = EMA_ALPHA * adv + (1 - EMA_ALPHA) * adv_ema
            p1_last = EMA_ALPHA * p1 + (1 - EMA_ALPHA) * p1_last
            if enable_early_fire_reaction:
                # (2026-08-22 修正②) 既定 False では従来通り settled 再計算の
                # たびに無条件クリア (bit-identical)。True では finalize が
                # 実際に会計へ反映されたときだけクリアする。
                if enable_early_fire_clear_on_finalize:
                    _finalized = efire_tracker.finalized_since_last_check(
                        snap.chain_total_score_p1, snap.chain_total_score_p2)
                    efire_tracker.on_settled(finalized=_finalized)
                else:
                    efire_tracker.on_settled()  # 確定計算が入ったので速報バイアスをクリア
            if dump_timeline_path is not None:
                # settled 更新のたびに1レコード追記する (本番の間引き
                # HeavyAdvCache.every はそのまま=dump は本番が実際に出す
                # 判定の記録、2026-08-11 タイムラインdump工事)。
                # p1_raw: adv_to_winprob(model_adv)。kill_override/4成分
                # ブレンド/校正/EMA を一切通していない生モデル勝率
                # (2026-08-11 アーキ審査追加、D1a/D1b の raw 段階判定に使う)。
                # (2026-08-25 Gate 3R-6 案A) 非STABLE中の is_dead 判定保留。
                # 既定 False では dead1_rec/dead2_rec は None のままで
                # _build_timeline_dump_row が従来通り b1/b2.is_dead() を
                # 評価する (bit-identical)。adv/p1 の判定値には一切影響
                # しない (is_dead は dump 記録専用列)。
                dead1_rec: bool | None = None
                dead2_rec: bool | None = None
                if enable_nonstable_hold_is_dead:
                    _raw_dead1, _raw_dead2 = b1.is_dead(), b2.is_dead()
                    dead1_rec, _held1 = _resolve_nonstable_hold_is_dead(
                        _raw_dead1, r.p1.state.name)
                    dead2_rec, _held2 = _resolve_nonstable_hold_is_dead(
                        _raw_dead2, r.p2.state.name)
                    isdead_hold_stats.record(
                        held1=_held1, suppressed1=(_raw_dead1 and not dead1_rec),
                        held2=_held2, suppressed2=(_raw_dead2 and not dead2_rec))
                # (2026-08-25 Gate 3R-5) gross累積カウンタ列。既定 False では
                # gross_fields=None のまま _build_timeline_dump_row に渡り、
                # TimelineDumpRow の gross_* は全て None (bit-identical)。
                gross_fields: dict[str, int | float] | None = None
                if enable_gross_ledger_dump:
                    curr_gross = tracker.get_gross_counters(t)
                    curr_pending_unc = (
                        int(snap.pending_p1_uncapped), int(snap.pending_p2_uncapped))
                    gross_fields = _build_gross_dump_fields(
                        prev_gross_counters, curr_gross,
                        prev_gross_pending_unc, curr_pending_unc, game_idx)
                    gross_dump_stats.record(gross_fields)
                    prev_gross_counters = curr_gross
                    prev_gross_pending_unc = curr_pending_unc
                # (2026-08-25 Gate 3R-6 本体) 死亡確定の時間的ロジック。
                # 既定 False では is_dead1_confirmed/is_dead2_confirmed は
                # None のままで npz へ列自体が追加されない (bit-identical)。
                # death_tracker1/2.update() は settled ゲートの外側で毎フレーム
                # 既に呼び済み (上記 :5837-5848 参照)、ここでは現在の確定状態
                # (resolved_is_dead()) を読むだけ。
                death1_rec: bool | None = None
                death2_rec: bool | None = None
                if enable_death_confirm_sequence:
                    death1_rec = death_tracker1.resolved_is_dead()
                    death2_rec = death_tracker2.resolved_is_dead()
                dump_rows.append(_build_timeline_dump_row(
                    t_sec=t, game_idx=game_idx, adv_raw=model_adv,
                    adv_ema=adv_ema, p1=p1_last, p1_raw=adv_to_winprob(model_adv),
                    pending_p1=snap.pending_p1, pending_p2=snap.pending_p2,
                    room1=room1, room2=room2, b1=b1, b2=b2, drivers=drivers,
                    score1=r.p1.score, score2=r.p2.score,
                    state1=r.p1.state.name, state2=r.p2.state.name,
                    is_dead1=dead1_rec, is_dead2=dead2_rec,
                    # (2026-08-23 根治①) kill_override に実際に渡された
                    # 是正後の値 (:5149-5160 kroom1/kroom2/kpending1/kpending2)。
                    # enable_kill_override_chain_completion=False、または
                    # 是正未発火のフレームでは room1/room2・snap.pending_p1/p2
                    # と完全に同じ値 (_kill_override_chain_completion_inputs
                    # 仕様通り)。
                    kpending_p1=kpending1, kpending_p2=kpending2,
                    kroom1=kroom1, kroom2=kroom2,
                    gross_fields=gross_fields,
                    is_dead1_confirmed=death1_rec, is_dead2_confirmed=death2_rec,
                ))
        # (早期発火) 表示直前にのみ bias を加算する (adv_ema/p1_last の EMA 内部状態
        # 自体には混ぜない = 無効時は従来経路とビット一致)。
        disp_adv, disp_p1 = adv_ema, p1_last
        drivers_before_resolved = drivers
        episode_consistency_fallback = False
        minimum_prediction_guarded = False
        if enable_early_fire_reaction and efire_tracker.bias != 0.0:
            disp_adv = max(-100.0, min(100.0, adv_ema + efire_tracker.bias))
            disp_p1 = adv_to_winprob(disp_adv)
        # (#9 決着先読み) 保持中は EMA/早期発火バイアスを完全に上書きする
        # (「確定済みの未来を逐次再評価しない」= disp を決着値に固定表示、
        # docs/DEMO_REVIEW_2026-08-13.md #9)。hold 解除直後は、同フレームで
        # 真の settled 再計算が走っていなければ adv_ema/p1_last を決着値で
        # 継続させ、次回 settled 更新からのジャンプを最小化する
        # (実装は簡明優先、既定 False 時は本ブロック自体を評価しない)。
        if enable_resolved_exchange_eval:
            if resolved_active:
                minimum_prediction_guarded = (
                    enable_resolved_minimum_prediction_guard
                    and _minimum_prediction_guard_applies(
                        resolved_tracker, r.p1.state, r.p2.state, adv_ema))
                disp_adv, disp_p1 = (
                    resolved_tracker.hold_adv, resolved_tracker.hold_p1)
                drivers = resolved_tracker.hold_drivers
                # [指摘14 案2、2026-08-15] 決着ホールド値にも致死上書きを通す
                # (既定 OFF、ResolvedExchangeTracker.hold_after_kill_override
                # docstring 参照)。
                if enable_resolved_kill_override and not minimum_prediction_guarded:
                    candidate_adv, candidate_p1 = resolved_tracker.hold_after_kill_override(
                        b1, b2, state1=r.p1.state, state2=r.p2.state)
                    if episode_drive is not None:
                        before_gated_adv = disp_adv
                        (gated_adv, candidate, applied,
                         reason) = _apply_episode_hard_override_gate(
                            disp_adv, candidate_adv,
                            episode_drive.snapshot.ledger.allows_hard_override,
                            is_unresolved=(
                                episode_drive.snapshot.ledger.is_unresolved),
                            hard_override_target=(
                                episode_drive.snapshot.ledger.hard_override_target),
                            physical_net_raw=(
                                episode_drive.snapshot.ledger.net_raw))
                        disp_p1 = _sync_probability_after_episode_gate(
                            before_gated_adv, disp_p1, gated_adv)
                        disp_adv = gated_adv
                        episode_hard_candidate |= candidate
                        episode_hard_applied |= applied
                        episode_hard_path = _append_episode_hard_path(
                            episode_hard_path, "hold_active", candidate)
                        episode_hard_reason = episode_hard_reason or reason
                    else:
                        disp_adv, disp_p1 = candidate_adv, candidate_p1
                if episode_drive is not None:
                    ledger = episode_drive.snapshot.ledger
                    (disp_adv, disp_p1,
                     episode_consistency_fallback) = (
                        resolved_tracker.apply_episode_consistency(
                            disp_adv, disp_p1, adv_ema, p1_last,
                            model_adv_last, ledger.net_raw,
                            is_unresolved=ledger.is_unresolved,
                            allows_hard_override=ledger.allows_hard_override))
                    if episode_consistency_fallback:
                        drivers = drivers_before_resolved
                # 40点の下限予測より、実量を確認済みの episode fallback を優先する。
                # fallback も無い間だけ、誤った確定表示を直前STABLE値へ保留する。
                if (minimum_prediction_guarded
                        and not episode_consistency_fallback):
                    disp_adv, disp_p1 = adv_ema, p1_last
                    drivers = drivers_before_resolved
            elif resolved_just_deactivated and not settled_ran_this_frame:
                adv_ema, p1_last = resolved_tracker.hold_adv, resolved_tracker.hold_p1
                if enable_resolved_kill_override:
                    candidate_adv, candidate_p1 = resolved_tracker.hold_after_kill_override(
                        b1, b2, state1=r.p1.state, state2=r.p2.state)
                    if episode_drive is not None:
                        before_gated_adv = adv_ema
                        (gated_adv, candidate, applied,
                         reason) = _apply_episode_hard_override_gate(
                            adv_ema, candidate_adv,
                            episode_drive.snapshot.ledger.allows_hard_override,
                            is_unresolved=(
                                episode_drive.snapshot.ledger.is_unresolved),
                            hard_override_target=(
                                episode_drive.snapshot.ledger.hard_override_target),
                            physical_net_raw=(
                                episode_drive.snapshot.ledger.net_raw))
                        p1_last = _sync_probability_after_episode_gate(
                            before_gated_adv, p1_last, gated_adv)
                        adv_ema = gated_adv
                        episode_hard_candidate |= candidate
                        episode_hard_applied |= applied
                        episode_hard_path = _append_episode_hard_path(
                            episode_hard_path, "hold_deactivated", candidate)
                        episode_hard_reason = episode_hard_reason or reason
                    else:
                        adv_ema, p1_last = candidate_adv, candidate_p1
                disp_adv, disp_p1 = adv_ema, p1_last
        # (2026-08-22 修正④) 安全弁の発火理由を主因先頭へ差し込む。決着ホールド
        # 中 (resolved_active) は `drivers` が `resolved_tracker.hold_drivers`
        # (別経路の生成物) に差し替わっており、ライブ per-frame kill_override
        # の note (`kill_override_note`) は無関係になるため対象外にする
        # (混線防止、既定 False では本行自体が no-op で bit-identical)。
        if enable_kill_override_attribution and not (
                enable_resolved_exchange_eval and resolved_active):
            drivers = _drivers_for_display(drivers, kill_override_note)
        if episode_adapter is not None:
            ledger = episode_drive.snapshot.ledger
            disp_adv, disp_p1, final_display_capped = (
                _cap_unresolved_episode_display(
                    disp_adv, disp_p1, is_unresolved=ledger.is_unresolved))
            if final_display_capped:
                episode_hard_candidate = True
                episode_hard_path = _append_episode_hard_path(
                    episode_hard_path, "final_display", True)
                episode_hard_reason = (
                    episode_hard_reason or "episode_unresolved_display_capped")
            disp_p1 = _ensure_display_probability_direction(disp_adv, disp_p1)
        # (#8 修正) グラフに積む時刻は「現在の試合の開始からの相対時間」
        # (= (t - start_sec) - game_start_sec)。境界検知直後は game_start_sec が
        # (t - start_sec) と一致するため必ず 0 から始まる。境界が一度も
        # 起きない動画では game_start_sec=0.0 のままなので従来 (t - start_sec)
        # と完全に同一 (backwards compat)。
        t_rel = _graph_relative_time(t, start_sec, game_start_sec)
        if fi >= write_frame and fi % step == 0:
            if b1 is not None and b2 is not None:
                # settled=False の間も直近確定値(保持中)を同値追記 → グラフは平坦。
                history.append((t_rel, disp_adv))
                if debug_history_out is not None:
                    debug_history_out.append((t, disp_adv))
            if dump_display_timeline_path is not None:
                # 実画面は盤面未確定の待機中も値を表示する。b1/b2 の有無で落とすと
                # 試合境界が密dumpの欠測になり、張り付き時間の分母が再び壊れる。
                display_dump_rows.append(DisplayTimelineRow(
                    t_sec=t, game_idx=game_idx, display_adv=disp_adv,
                    display_p1=disp_p1, adv_raw_last=model_adv_last,
                    source=_display_timeline_source(
                        resolved_active, resolved_just_deactivated,
                        settled_ran_this_frame,
                        episode_consistency_fallback,
                        minimum_prediction_guarded),
                    resolved_active=resolved_active,
                    settled_ran=settled_ran_this_frame,
                    state1=r.p1.state.name, state2=r.p2.state.name,
                    score1=(TIMELINE_DUMP_SCORE_NONE_SENTINEL
                            if r.p1.score is None else int(r.p1.score)),
                    score2=(TIMELINE_DUMP_SCORE_NONE_SENTINEL
                            if r.p2.score is None else int(r.p2.score)),
                ))
            if dump_exchange_episode_timeline_path is not None:
                if episode_drive is None:
                    raise RuntimeError("交換episode sidecar有効なのにlive snapshotが無い")
                episode_dump_rows.append(_episode_timeline_row(
                    episode_drive, t_sec=t, game_idx=game_idx,
                    state1=r.p1.state.name, state2=r.p2.state.name,
                    hard_candidate=episode_hard_candidate,
                    hard_applied=episode_hard_applied,
                    hard_path=episode_hard_path,
                    hard_reason=episode_hard_reason))
        if fi < write_frame:
            continue  # ウォームアップ区間は書き出さない
        if not render:
            # 判定計算のみ (2026-08-11 追加)。描画・エンコードを一切行わない。
            written += 1
            if written % 300 == 0:
                print(f"  ... {written} frames (t={t:.1f}s adv={disp_adv:+.0f}) [no-render]")
            continue
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
            # れんさ数表示 (--show-chain-count、既定 OFF)。OFF 時は空文字/False
            # のままとなり _draw_panel_info の optional-if-truthy 分岐で行自体が
            # 描かれない (=既存出力と bit-identical、後方互換)。
            chain_text_p1, chain_mismatch_p1 = "", False
            chain_text_p2, chain_mismatch_p2 = "", False
            if show_chain_count:
                chain_text_p1, chain_mismatch_p1 = _build_chain_display_text(
                    "1P", chain_display_tracker.snapshot("1P", t))
                chain_text_p2, chain_mismatch_p2 = _build_chain_display_text(
                    "2P", chain_display_tracker.snapshot("2P", t))
            frame_out = _draw_panel_layout(
                display_frame, disp_adv, disp_p1, drivers, waiting,
                history, t_rel, graph_total,
                state1=r.p1.state.name, state2=r.p2.state.name,
                counter_text=_resolve_counter_text_for_display(
                    enable_counter_defender_only,
                    enable_resolved_exchange_eval and resolved_active,
                    resolved_tracker.hold_defender_side,
                    resolved_tracker.hold_defender_prob,
                    resolved_tracker.hold_incoming_ojama,
                    defender_side, counter_p1, counter_p2, incoming_ojama),
                # (2026-08-22 修正) 従来の t - start_sec は動画全体の絶対
                # 経過秒で、試合境界 (game_start_sec リセット) を反映しな
                # かったため区間分割の継ぎ目 (t=893.7s等) で 893秒→6秒と
                # 逆行していた。グラフ横軸 (:4826 の t_rel =
                # _graph_relative_time(t, start_sec, game_start_sec)) と
                # 同じ試合相対時間に揃える
                # (境界が一度も起きない動画では game_start_sec=0.0 のまま
                # なので従来値と完全一致、backwards compat)。
                elapsed_sec=t_rel,
                chain_text_p1=chain_text_p1, chain_text_p2=chain_text_p2,
                chain_mismatch_p1=chain_mismatch_p1, chain_mismatch_p2=chain_mismatch_p2,
                subtitle_h=panel_subtitle_h)
        else:
            frame_out = _draw_overlay(display_frame, disp_adv, disp_p1, drivers, waiting,
                                      history, t_rel, graph_total,
                                      ukey1=ukey1, ukey2=ukey2, sat1=sat1, sat2=sat2)
        writer.write(frame_out)
        written += 1
        if written % 300 == 0:
            print(f"  ... {written} frames (t={t:.1f}s adv={disp_adv:+.0f})")
    cap.release()
    if writer is not None:
        writer.release()
    if dump_timeline_path is not None:
        video_id = video.stem
        # (2026-08-24) is_dead1/is_dead2 の凍結盤面誤判定の遡及訂正。
        # dump_rows (診断用リストのみ) に作用し、ライブ判定には無関係
        # (enable_stable_confirmed_is_dead docstring 参照)。既定 False では
        # 呼ばないため save_timeline_dump への入力は従来と完全一致する。
        if enable_stable_confirmed_is_dead:
            dump_rows = _retroactively_correct_dead_dump_rows(dump_rows)
        save_timeline_dump(dump_timeline_path, video_id, dump_rows)
        print(f"[dump] {len(dump_rows)} records -> {dump_timeline_path}")
        # (2026-08-25 Gate 3R-6 案A) 保留カウンタは母数付きで必ず表示する
        # (0/0 なら「dump行なし=測っていない」、0/N なら「保留は起きていない」
        # と区別できる、memory feedback_zero_needs_denominator_2026-08-25)。
        if enable_nonstable_hold_is_dead:
            print(f"[is_dead保留] {isdead_hold_stats.summary()}")
        # (2026-08-26 決着ホールド根治) どの信号でホールドを解除したかを母数付きで
        # 必ず表示する。既定 OFF では print 自体を行わない (出力も bit-identical)。
        if enable_resolved_absolute_chain_end:
            print(f"[決着ホールド解除] {resolved_tracker.abs_end_summary()}")
        if enable_resolved_live_defender and enable_resolved_live_defender_strict:
            print(f"[side-local連鎖補正] {resolved_tracker.nondef_cycle_summary()}")
        if (enable_resolved_episode_physical_redecide
                or enable_resolved_episode_physical_consistency_guard):
            print(
                "[決着物理追従] 再決着 "
                f"{resolved_tracker.episode_physical_redecide_count}回 / "
                "矛盾保留 "
                f"{resolved_tracker.episode_consistency_fallback_count}frame / "
                "時刻先頭 "
                f"{resolved_tracker.episode_consistency_fallback_times[:8]}")
        # (2026-08-25 Gate 3R-5) gross累積カウンタも母数付きで必ず表示する
        # (0/0 なら「dump行なし=測っていない」と区別できる、
        # memory feedback_zero_needs_denominator_2026-08-25)。
        if enable_gross_ledger_dump:
            print(f"[gross台帳] {gross_dump_stats.summary()}")
        # (2026-08-25 Gate 3R-6 本体) 候補/解除/確定は母数付きで必ず表示する
        # (0/0 なら「dump行なし=測っていない」と区別できる、
        # memory feedback_zero_needs_denominator_2026-08-25)。
        if enable_death_confirm_sequence:
            print(f"[死亡確定] {death_confirm_stats.summary()}")
    if dump_display_timeline_path is not None:
        save_display_timeline(
            dump_display_timeline_path, video.stem, display_dump_rows)
        print(f"[display-dump] {len(display_dump_rows)} records -> "
              f"{dump_display_timeline_path}")
    if dump_exchange_episode_timeline_path is not None:
        save_episode_timeline(
            dump_exchange_episode_timeline_path, video.stem, episode_dump_rows)
        print(f"[episode-dump] {len(episode_dump_rows)} records -> "
              f"{dump_exchange_episode_timeline_path}")
    if render:
        print(f"[done] {written} frames -> {out}")
    else:
        print(f"[done] {written} frames (no-render)")
    return written


def main() -> None:
    # 2026-08-21: 1プロセス1コアに固定する (並列実行時のスレッド過剰生成の防止)。
    # 30先動画を8並列でレンダしたところ、各プロセスが 109 スレッドを立てて
    # 16コアに 872 スレッドが群がり、実効スループットが事前実測の 5.7分の1
    # (0.668 -> 0.1166 動画秒/壁秒、完了見込み3.0時間 -> 16.5時間) に落ちた。
    # 収集側は「cv2.setNumThreads(1) × 14並列が最適」と実測済み
    # (memory project_collect_indicators_v2_perf) だが、レンダ経路は未適用だった。
    # 単独実行時は 1 コアに絞られるが、認識だけなら実時間並みの速度が出ており
    # (27.44ms/update 実測)、並列数でスケールさせる方が総スループットは高い。
    cv2.setNumThreads(1)
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
    ap.add_argument(
        "--model-dir", type=Path, default=None, dest="model_dir",
        help=(
            "有利不利判定に使う学習済みモデルのディレクトリ (2026-08-18 追加)。"
            " 配下に model_full148_full_features.joblib / feature_cols_full.json"
            " が必要 (scripts/_retrain148_2026-08-14.py の出力形式と同一)。"
            f" 既定 None = 従来通り {MODEL_ARTIFACT_DIR} を使う"
            " (後方互換、未指定時は挙動不変)。指定先にファイルが無い場合や"
            " モデルが期待する特徴量数と feature_cols_full.json の列数が"
            " 食い違う場合は、フォールバックせず即座にエラー終了する。"
        ),
    )
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
    # 位相別 Platt (2026-08-11 Phase1-2 追加)。--platt-calibration と排他
    # (generate() 側で同時指定を ValueError にする)。
    ap.add_argument(
        "--phase-calibration", action="store_true", default=False,
        dest="enable_phase_calibration",
        help="表示用勝率へ「進行度 (match_progress) 別」の Platt scaling 後段校正を"
             "適用する (2026-08-11 追加)。既定 OFF = 従来挙動 (校正なし)。"
             "data/indicators_v2/phase_platt_calibration.json が必要で、無い場合は"
             "動画を読む前に例外になる。--platt-calibration とは同時指定不可。",
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
        "--early-fire-clear-on-finalize", action="store_true", default=False,
        dest="enable_early_fire_clear_on_finalize",
        help="EarlyFireTracker の速報バイアスをクリアする条件を「settled再計算"
             "が走った」から「連鎖の finalize が会計に反映された」に変える"
             "(2026-08-22 修正②)。--per-side-settled 下では相手STABLE中は"
             "settled が毎フレーム走り、finalize前に速報が毎回消えていた。"
             "既定 OFF = 従来挙動不変 (backwards compat)。--early-fire-reaction"
             "が OFF の場合は無視される (孫フラグ)。",
    )
    ap.add_argument(
        "--kill-override-chain-completion", action="store_true", default=False,
        dest="enable_kill_override_chain_completion",
        help="致死判定 (kill_override) の入力を「連鎖完走後」に是正する"
             "(2026-08-22 修正①)。死ぬと判定されうる側が自分自身の連鎖"
             "(CHAIN/GRAVITY_SETTLE) を撃っている最中でも従来は発火前の凍結"
             "盤面の空きと相殺前の額面 pending を使っており、自分の連鎖が"
             "pending を相殺しきるケースまで致死断定していた (実測7件、"
             "logs/killoverride_wrong_2026-08-22/一覧.tsv)。既存"
             "resolve_mutual_exchange を再利用して完走後の盤面・相殺後の"
             "残存pendingで判定し直す。既定 OFF = 従来挙動不変 (backwards"
             "compat)。",
    )
    ap.add_argument(
        "--kill-override-chain-gen-accumulate", action="store_true", default=False,
        dest="enable_kill_override_chain_gen_accumulate",
        help="ChainGenerationAccumulator の累積モード (2026-08-22 user判断)。"
             "既定OFF=直近1件のchain_eventの値に置き換える(合算しない)。"
             "累積は架空の完了状態を仮定し新しい不一致時間帯を作ると実測で"
             "判明したため既定OFFにした。--kill-override-chain-completion"
             "がOFFの場合は無視される(孫フラグ)。",
    )
    ap.add_argument(
        "--kill-override-attribution", action="store_true", default=False,
        dest="enable_kill_override_attribution",
        help="kill_override 発火時に主因表示欄へ致死判定の理由を明示する"
             "(2026-08-22 修正④、表示専用・予測値には影響しない)。従来は"
             "安全弁が結論を上書きしても主因欄は適用前の生モデル寄与度の"
             "ままで、表示と結論が矛盾していた。既定 OFF = 従来挙動不変。",
    )
    ap.add_argument(
        "--kill-override-hysteresis", action="store_true", default=False,
        dest="enable_kill_override_hysteresis",
        help="致死上書き (kill_override) に確信度ゲートを掛ける (2026-08-24 "
             "B案)。同一方向に約0.96秒 (受け側の持ち手2手+認識反映8f) 持続して"
             "初めて完全上書き (±100) を許し、それまでは |adv|≤90 に制限、"
             "方向反転時は1手時間 (約0.35秒) のクールダウンで上書きを保留する。"
             "ChainEvent 断片化による1フレーム ±100→∓100 反転 (納品動画で"
             "実測、memory project_pm100_display_flip_2026-08-24 根因③) の"
             "構造的禁止。既定 OFF = 従来挙動不変 (backwards compat)。",
    )
    ap.add_argument(
        "--kill-override-scale-compare", action="store_true", default=False,
        dest="enable_kill_override_scale_compare",
        help="致死上書きの pending 入力を「規模の比較」に是正する (2026-08-24 "
             "A案、根因①②)。(i) 会計 finalize 遅延 (実測最大11.5秒) 中の"
             "「送付済み未登録」分を連鎖完走時に即時捕捉して受け側 pending に"
             "供給し、(ii) 相殺の引き算には PENDING_ABS_CAP=216 で丸める前の"
             "実額 (会計の並行帳簿 pending_p1/p2_uncapped) を使う。表示用の"
             "cap は従来のまま。撃ち返し全量が新規攻撃として誤計上され方向が"
             "反転する事故 (納品動画 seg01 game2 で実測) の根治。"
             "既定 OFF = 従来挙動不変 (backwards compat)。",
    )
    ap.add_argument(
        "--counter-reach", action="store_true", default=COUNTER_REACH_ENABLED_BY_DEFAULT,
        dest="enable_counter_reach",
        help="打ち合い応手確率 (モンテカルロ) を有利不利に加える。相手が閾値"
             "以上を返せる確率を見て、返せない攻撃を持っている側を有利にする。"
             "src.production_config.COUNTER_REACH_ENABLED_BY_DEFAULT により"
             "既定 ON (2026-08-12 正式採用、指標大整理提案書0-4)。無効化は"
             "--no-counter-reach。",
    )
    ap.add_argument(
        "--no-counter-reach", action="store_false", dest="enable_counter_reach",
        help="(A/B比較用) --counter-reach を明示的に無効化する。既定が ON の"
             "ため通常は不要。",
    )
    ap.add_argument(
        "--counter-remaining-time", action="store_true", default=False,
        dest="enable_counter_remaining_time",
        help="打ち合い応手の時間予算の意味論を修正する (2026-08-13、"
             "docs/DEMO_REVIEW_2026-08-13.md #3)。経過時間の控除 + 観測連鎖数"
             "を最終連鎖数と誤認しない条件付き補正 (E[最終|N到達]) に切り替える。"
             "既定 OFF = 従来挙動 (観測連鎖数×0.4秒をそのまま予算にする)。",
    )
    ap.add_argument(
        "--counter-defender-only", action="store_true", default=False,
        dest="enable_counter_defender_only",
        help="打ち合い応手確率を受け側限定・実飛来量ベースに切り替える "
             "(2026-08-13、docs/DEMO_REVIEW_2026-08-13.md #4/#5)。既定 OFF = "
             "従来挙動 (固定閾値12個を両側常時計算)。",
    )
    ap.add_argument(
        "--resolved-exchange-eval", action="store_true", default=False,
        dest="enable_resolved_exchange_eval",
        help="両者同時発火の決着を先読みし連鎖終了まで固定表示する "
             "(2026-08-13、docs/DEMO_REVIEW_2026-08-13.md #9)。両側の "
             "chain_event が同時にアクティブになった瞬間に一度だけ連鎖を"
             "完走シミュレーションし決着後勝率で固定する。着弾完了 (2026-08-14 "
             "指摘11対処) まで保持を延長する。既定 OFF = 従来挙動 "
             "(観測到着ごとの逐次再評価、連鎖中の乱高下あり)。",
    )
    ap.add_argument(
        "--resolved-decisive-amplify", action="store_true", default=False,
        dest="enable_resolved_decisive_amplify",
        help="--resolved-exchange-eval の決着値に受け側の応手不能度を統合する "
             "(2026-08-14、docs/DEMO_REVIEW_2026-08-13.md #10)。応手不能かつ "
             "飛来量大なら決定的側へ増幅する。--resolved-exchange-eval 無効時 "
             "は無視される。既定 OFF = #9 のみの決着値と完全に同一。",
    )
    ap.add_argument(
        "--resolved-live-defender", action=argparse.BooleanOptionalAction, default=False,
        dest="enable_resolved_live_defender",
        help="交換中、受け側が物理的に自由な間は決着値を0.5秒ごと"
             "ライブ再評価する (2026-08-15、2026-08-27拡張)。"
             "攻撃側の帰結 (飛来量・仮想盤面) は凍結維持したまま、受け側の現在盤面"
             "+残り時間逓減でモデル評価/決定度増幅を再計算する。"
             "--resolved-exchange-eval 無効時は無視される。既定 OFF = 従来"
             " (両側終了まで完全凍結) と完全に同一。",
    )
    ap.add_argument(
        "--resolved-live-defender-strict", action=argparse.BooleanOptionalAction,
        default=False, dest="enable_resolved_live_defender_strict",
        help="--resolved-live-defender の起動条件を厳格化する (2026-08-15 "
             "指摘14 案1、docs/DEMO_REVIEW_2026-08-13.md #14)。従来の XOR "
             "条件だけでは「両者が本当に同時に本線を撃ち合い攻撃側のアニメ"
             "だけ先に終わった」ケースを誤って受け側再評価対象にしてしまう "
             "(実測: 589個飛来を受ける側の生存率を18.9%%と誤表示、正しくは"
             "3.9%%)。defender_side 自身の状態機械 state が CHAIN/"
             "GRAVITY_SETTLE (今まさに自分の連鎖処理中) であることを追加"
             "確認する (chain_event 有無では settle gap を取りこぼすと計装で"
             "確認済み、docstring 参照)。--resolved-live-defender 無効時は"
             "無視。既定 OFF = 従来挙動と完全に同一。",
    )
    ap.add_argument(
        "--resolved-kill-override", action=argparse.BooleanOptionalAction,
        default=False, dest="enable_resolved_kill_override",
        help="決着ホールド値 (hold_adv/hold_p1) にも致死上書き (kill_override) "
             "を適用する (2026-08-15 指摘14 案2、docs/DEMO_REVIEW_2026-08-13.md "
             "#14)。従来 kill_override はライブ per-frame 経路にのみ配線され、"
             "決着ホールド中は pending/room 比が致死水準でも安全弁が発火"
             "しなかった (実測: 589/50≈11.8 ≫ KILL_RATIO_FULL=1.5 でも無発火)。"
             "--resolved-exchange-eval 無効時は無視される。既定 OFF = 従来"
             "挙動と完全に同一。",
    )
    ap.add_argument(
        "--resolved-kill-override-counter-aware", action=argparse.BooleanOptionalAction,
        default=False, dest="enable_resolved_kill_override_counter_aware",
        help="--resolved-kill-override の致死断定を受け側の応手確率で減衰する "
             "(2026-08-15 指摘19、docs/DEMO_REVIEW_2026-08-13.md #19)。従来は"
             "pending/room 比のみで致死断定するため、受け側がSTABLEで応手"
             "可能な局面でも致死断定していた (実測 t=201.4-203.4: 1P 0.7%%"
             "と誤表示、直後に撃ち返し勝利)。既存の CounterReachTracker が"
             "同一フレームで算出済みの hold_defender_prob/hold_defender_side"
             "を再利用し新規の推測ロジックは追加しない。--resolved-kill-"
             "override 無効時は無視される。既定 OFF = 従来挙動と完全に同一。",
    )
    ap.add_argument(
        "--resolved-victim-gen-live", action=argparse.BooleanOptionalAction,
        default=False, dest="enable_resolved_victim_gen_live",
        help="決着の再決着 (_maybe_redecide) を「保持セッション中1回きり」"
             "から「chain_end_triggered_pX が True の間0.5秒ごとに追従」"
             "へ緩和する (2026-08-16 指摘19 根治、docs/DEMO_REVIEW_2026-08-13"
             ".md #19)。従来の1回きり latch は settle 開始直後の未確定"
             "(しばしば0の) chain_total_score_pX で永久に固定してしまい、"
             "段階的に育つ真の確定値 (実測: 0→1260→4020) を拾わなかった。"
             "自分の連鎖を処理中の側の生成お邪魔量が過小評価される直接原因"
             "だった。--resolved-exchange-eval 無効時は無視される。"
             "既定 OFF = 従来挙動と完全に同一。",
    )
    ap.add_argument(
        "--resolved-episode-physical-redecide",
        action=argparse.BooleanOptionalAction, default=False,
        dest="enable_resolved_episode_physical_redecide",
        help="交換台帳の未解決純残量が変化するたび決着ホールドを再計算する。"
             "--exchange-episode-gate と --resolved-exchange-eval の併用時のみ"
             "有効。既定 OFF。",
    )
    ap.add_argument(
        "--resolved-episode-physical-consistency-guard",
        action=argparse.BooleanOptionalAction, default=False,
        dest="enable_resolved_episode_physical_consistency_guard",
        help="未解決中、台帳方向と直前モデル方向の両方に逆らう単独反転を"
             "直前確定値へ保留する。再決着は行わない。既定 OFF。",
    )
    ap.add_argument(
        "--resolved-minimum-prediction-guard",
        action=argparse.BooleanOptionalAction, default=False,
        dest="enable_resolved_minimum_prediction_guard",
        help="進行中連鎖の完走予測が1連鎖・40点の最小値に潰れた場合、"
             "決着値を確定表示せず直前STABLE値を維持する。連鎖終了後は"
             "実測結果へ戻る。既定 OFF。",
    )
    ap.add_argument(
        "--resolved-pending-landing-gate", action=argparse.BooleanOptionalAction,
        default=False, dest="enable_resolved_pending_landing_gate",
        help="ResolvedExchangeTracker.enable_pending_landing_gate の配線 "
             "(2026-08-21 配線是正。クラス引数自体は2026-08-21導入済みだったが "
             "generate()/CLIに渡す手段が無く常にFalse固定だった配線漏れの是正)。"
             "攻撃側が連鎖中の間は予告おじゃまを受け側の生盤面へ着地させず "
             "保留する。--resolved-exchange-eval 無効時は無視される。"
             "既定 OFF = 従来挙動と完全に同一。",
    )
    ap.add_argument(
        "--resolved-counter-placement-reuse", action=argparse.BooleanOptionalAction,
        default=False, dest="enable_resolved_counter_placement_reuse",
        help="受け側限定応手MC (CounterReachTracker._update_defender_only) の "
             "再計算を「受け側の盤面bytesが変化した(=設置が起きた)とき」だけに "
             "限定する近似 (2026-08-21 user承認)。実測で従来キャッシュのヒット率は "
             "0.00%% (時間予算がキーに入り毎回ミスする設計上の必然)。閾値が変われば "
             "別スコープとなり自動的に再計算される。表示更新周期 (0.5秒ごと) は "
             "変えない。--resolved-exchange-eval 無効時は無視される。"
             "既定 OFF = 従来挙動と完全に同一。",
    )
    ap.add_argument(
        "--resolved-absolute-chain-end", action=argparse.BooleanOptionalAction,
        default=False, dest="enable_resolved_absolute_chain_end",
        help="決着ホールドの解除条件に、連鎖の終わりの絶対律 "
             "(連鎖している側のネクストが動いた OR その側にお邪魔が落ちた) を "
             "OR で足す (2026-08-26 user決定の根治)。従来の「両側 chain_event が "
             "None」は ChainEvent の断片化で打ち合い中に長時間成立せず、実測で "
             "最大45.07秒 評価が止まっていた。新条件は解除を早めるだけで遅く "
             "しない。--resolved-exchange-eval 無効時は無視される。"
             "既定 OFF = 従来挙動と完全に同一。",
    )
    ap.add_argument(
        "--resolved-counter-budget-quantize", action=argparse.BooleanOptionalAction,
        default=False, dest="enable_resolved_counter_budget_quantize",
        help="応手MCのキャッシュキーに入る budget_sec (着弾までの残り秒) を "
             "COUNTER_BUDGET_QUANTUM_SEC (1手あたりの平均設置時間、"
             "mc_counter_estimator.PLACEMENT_SPEED_BY_ROW_SEC の単純平均 "
             "≈0.348秒、物理実測値からの導出) 単位に丸める (2026-08-21 "
             "user承認)。--resolved-counter-placement-reuse とは独立の別機構 "
             "(盤面一致による再利用/残り秒数の丸め、それぞれ単独でも効果を "
             "測れる)。キャッシュミス時に実際の MC 計算へ渡す budget_sec は "
             "常に元の値 (量子化しない)。--resolved-exchange-eval 無効時は "
             "無視される。既定 OFF = 従来挙動と完全に同一。",
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
        "--stable-majority-window", action=argparse.BooleanOptionalAction, default=None,
        dest="stable_majority_window",
        help="盤面確定窓 3中2多数決を有効化 (RecognitionPipeline.load_default に転送、"
             "2026-08-13 user承認、認識99.5%%物差し条件付き採用)。初回STABLE確定窓を"
             "「stable_frame_count 連続厳密一致」から「直近3観測中2一致」に切り替える。"
             "デフォルト None = load_default 本体の既定値 (False) に従う。A/B比較用。",
    )
    ap.add_argument(
        "--enable-ojama-fall-placement-override", action=argparse.BooleanOptionalAction,
        default=None, dest="enable_ojama_fall_placement_override",
        help="OJAMA_FALL中の実設置検知で即exit (案2、2026-08-13)。None=既定OFF")
    ap.add_argument(
        "--enable-ojama-fall-entry-hardening", action=argparse.BooleanOptionalAction,
        default=None, dest="enable_ojama_fall_entry_hardening",
        help="OJAMA_FALL入口の実時間化+連鎖直後の割込抑制 (案4-lite、2026-08-13)。None=既定OFF")
    ap.add_argument(
        "--enable-ojama-fall-scoped-exit", action=argparse.BooleanOptionalAction,
        default=None, dest="enable_ojama_fall_scoped_exit",
        help="OJAMA_FALL出口のおじゃま限定監視+会計連動 (Stage2根治、2026-08-13)。None=既定OFF")
    ap.add_argument(
        "--enable-pseudo-chain-score-fill", action=argparse.BooleanOptionalAction,
        default=False, dest="enable_pseudo_chain_score_fill",
        help="W7根治① (2026-08-13、docs/KNOWN_WEAKNESSES.md): formula/landing "
             "経路の疑似ChainEventにsimulate推定スコアを充填する。既定OFF"
             " (bit-identical)。")
    ap.add_argument(
        "--chain-hold-base-sec", type=float, default=None,
        dest="chain_hold_base_sec",
        help="CHAIN保持時間モデルの固定項 (2026-08-22 修正②根治)。実測較正値"
             " (recognition_pipeline.py:731-736、23動画418イベント) は2.61。"
             "既定None=ライブラリ既定0.0のまま (backwards compat)。")
    ap.add_argument(
        "--chain-hold-per-step-sec", type=float, default=None,
        dest="chain_hold_per_step_sec",
        help="CHAIN保持時間モデルの連鎖数係数 (2026-08-22 修正②根治)。実測較正値"
             "は1.17。既定None=ライブラリ既定0.3のまま (backwards compat)。")
    ap.add_argument(
        "--enable-slide-exit-min-display-guard",
        action=argparse.BooleanOptionalAction,
        default=False, dest="enable_slide_exit_min_display_guard",
        help="修正③根治 (2026-08-22): NextSlide即終了経路のX1/X4誤検知抑制ガード"
             " (連鎖中1.37秒周期の断片化対策)。既定OFF (bit-identical)。")
    ap.add_argument(
        "--layout", choices=VALID_LAYOUTS, default="overlay", dest="layout",
        help="出力レイアウト (2026-08-10 user指示追加)。'overlay'(既定)は従来通り"
             "盤面に直接バー等を重ねる。'panel' は左上に映像・左下にタイムライン"
             "グラフ・右に縦長情報パネルを配置する新レイアウト (1920x1080)。",
    )
    ap.add_argument(
        "--panel-subtitle-h", type=int, default=PANEL_SUBTITLE_H,
        dest="panel_subtitle_h",
        help="--layout panel の下端字幕帯の高さ (px、2026-08-21 user指示"
             "「グラフ広げて」で追加)。既定 140px = 従来と完全一致。0を指定"
             "すると字幕帯を無くし、その分を左下グラフと右の情報パネルの"
             "高さへ丸ごと回す (layout=overlay 時は無視される)。",
    )
    ap.add_argument(
        "--show-chain-count", action="store_true", default=False,
        dest="show_chain_count",
        help="--layout panel の情報パネルに推定連鎖数(simulate)/実測得点差"
             "(score OCR)/得点逆算連鎖数を1P/2Pそれぞれ表示する "
             "(2026-08-15 user要望、認識性能検証用)。既定OFF=行を描かない"
             " (bit-identical)。3値が食い違う場合はオレンジで強調表示する。",
    )
    ap.add_argument(
        "--show-excluded-attribution", action="store_true", default=False,
        dest="show_excluded_attribution",
        help="デバッグ専用。「主因」欄から通常除外している指標 "
             "(src.production_config.ATTRIBUTION_EXCLUDED_INDICATORS、勝敗と"
             "無相関と実測済み) も候補に含めた、除外前の表示に戻す "
             "(2026-08-11 ロードマップ Phase1-3 追加)。既定 OFF = 除外リスト適用"
             " (通常表示)。adv/p1 の判定値には無関係 (表示候補の絞り込みのみ)。",
    )
    ap.add_argument(
        "--no-render", action="store_false", default=True, dest="render",
        help="動画の合成・書き出しを行わず判定計算だけ行う (2026-08-11 "
             "タイムラインdump工事で追加)。--dump-timeline と併用して"
             "dump生成を高速化する用途。既定 (指定なし) は従来通りレンダリングする。",
    )
    ap.add_argument(
        "--dump-timeline", type=Path, default=None, dest="dump_timeline_path",
        help="settled 更新 (有利不利判定の再計算) のたびに1レコードを収集し、"
             "終了時に npz として保存する (2026-08-11 追加)。"
             "scripts/scan_judgment_anomalies.py --from-dump がこれを読むだけで"
             "検出でき、判定の再計算(148動画で約39日と実測済み)が不要になる。"
             "既定 None = 保存しない。",
    )
    ap.add_argument(
        "--dump-display-timeline", type=Path, default=None,
        dest="dump_display_timeline_path",
        help="実際の表示値を非settled中も含めて毎サンプル保存する。"
             "settled限定の --dump-timeline とは別npzへ出力する。",
    )
    ap.add_argument(
        "--exchange-episode-gate", action="store_true", default=False,
        dest="enable_exchange_episode_gate",
        help="条件5: 交換episode未解決中の完全上書き±100をライブ/決着hold両経路で"
             "禁止する。既定OFF、本番登録なし。",
    )
    ap.add_argument(
        "--dump-exchange-episode-timeline", type=Path, default=None,
        dest="dump_exchange_episode_timeline_path",
        help="条件5のepisode/chain/gross/hard-overrideを毎サンプルsidecar npzへ保存。"
             "--exchange-episode-gate が必須。",
    )
    ap.add_argument(
        "--stable-confirmed-is-dead", action="store_true", default=False,
        dest="enable_stable_confirmed_is_dead",
        help="--dump-timeline の is_dead1/is_dead2 列を、非STABLE中(連鎖等)に"
             "凍結された盤面への誤判定から遡及訂正する (2026-08-24)。実測で"
             "試合時間の約8%%が「連鎖で盤面が空になった実画面なのに窒息判定"
             "True」の誤りだった (logs/is_dead_persist_2026-08-23/)。dump専用の"
             "後処理でライブ判定/描画には無関係。既定 OFF = 従来通り"
             "(bit-identical)。",
    )
    ap.add_argument(
        "--nonstable-hold-is-dead", action="store_true", default=False,
        dest="enable_nonstable_hold_is_dead",
        help="--dump-timeline の is_dead1/is_dead2 を、own state が STABLE で"
             "ない行では判定保留 (False) として記録する (2026-08-25 Gate 3R-6 "
             "案A)。user伝授の絶対律「設置前/積み上げ中/連鎖直前/連鎖中は"
             "窒息としない」の実装で、未来参照なし=リアルタイム可能 "
             "(--stable-confirmed-is-dead の遡及訂正はリアルタイム不可の後処理"
             "、併用時は遡及訂正が後段で上書きする)。保留行数は終了時に母数"
             "付きで表示。既定 OFF = 従来通り (bit-identical)。",
    )
    ap.add_argument(
        "--gross-ledger-dump", action="store_true", default=False,
        dest="enable_gross_ledger_dump",
        help="--dump-timeline へ cap前 gross 累積カウンタ列 "
             "(gross_gen/offset/dropped/wiped/clamp_loss/pending_unc/residual/"
             "inspected_sides) を追加する (2026-08-25 Gate 3R-5、"
             "docs/EXCHANGE_GROSS_SUPPLY_DESIGN_2026-08-25.md)。"
             "OjamaAccountingTracker.get_gross_counters() と "
             "classify_gross_counter_delta の読み取り専用経路で、交換台帳・"
             "src/production_config.py へは一切配線しない (dump専用、"
             "本番の判定表示には無関係)。保存則残差は検査 side 数を分母とし"
             "母数付きで終了時に表示する。既定 OFF = gross_* 列は npz に"
             "一切追加されず、旧 dump と bit-identical (backwards compat)。",
    )
    ap.add_argument(
        "--death-confirm-sequence", action="store_true", default=False,
        dest="enable_death_confirm_sequence",
        help="--dump-timeline へ is_dead1_confirmed/is_dead2_confirmed 列を"
             "追加する (2026-08-25 Gate 3R-6 本体、src/death_confirmation.py)。"
             "user伝授の死亡確定条件 (12段目に設置して連鎖が起きない / "
             "おじゃまが降って12段目が埋まる) を「候補→猶予/解除→確定」の"
             "3段階でリアルタイムに実装する。own chain 開始 (掛け算式実読) "
             "で候補を解除し、連鎖なしで --death-next-stationary-sec 秒"
             "ネクストが動かなければ確定する (設計訂正2026-08-25: 「次の"
             "ツモが置けたら確定」は死亡すると発火しない逆理のため撤回、"
             "ネクスト不動の簡易検出に差し替え済み)。既存 is_dead1/is_dead2 "
             "(Board.is_dead() の即時占有判定) は変更せず別列として並存。"
             "候補/解除/確定/閾値未到達での境界消滅 (発生源別) と確定遅延"
             "分布は母数付きで終了時に表示する。既定 OFF = "
             "is_dead1_confirmed/is_dead2_confirmed 列は npz に一切追加"
             "されず、旧 dump と bit-identical (backwards compat)。"
             "(2026-08-25 Codex 独立レビュー NG 対応で修正: ①不動猶予は"
             "候補発生時刻を基準にする ②next=None は不動の証拠にしない "
             "③試合境界後は検証済みの実ゲーム開始まで候補受付を再凍結する。"
             "詳細は src/death_confirmation.py docstring 参照)。",
    )
    ap.add_argument(
        "--death-next-stationary-sec", type=float,
        default=NEXT_STATIONARY_CONFIRM_SEC,
        dest="death_next_stationary_sec",
        help="--death-confirm-sequence の確定閾値 (秒、既定 "
             f"{NEXT_STATIONARY_CONFIRM_SEC})。user 指定の暫定値であり "
             "Claude がシーンから逆算した値ではない (2026-08-25 user指示、"
             "底抜け演出検出による根治までの簡易実装)。感度の事後測定用に"
             "変更可能にしている。",
    )
    # collect_boards_lean.py と同名・同既定 (2026-08-12 追加、対称化)。
    ap.add_argument(
        "--normalize-fps-30", action="store_true", dest="normalize_fps_30",
        help=(
            "60fps 等の動画を stride-2 相当 (実効30fps) に間引く "
            "(src.fps_normalize.resolve_normalize_fps_30_stride、2026-08-12 追加。"
            "collect_boards_lean.py が2026-07-30から既定採用している正規化と"
            "同一関数)。"
            "2026-08-12 既定 True 化により本フラグは実質 no-op "
            "(明示しなくても既定で有効)。後方互換のため残置。"
            "無効化するには --no-normalize-fps-30 を使う。"
        ),
    )
    ap.add_argument(
        "--no-normalize-fps-30", action="store_true", dest="no_normalize_fps_30",
        help=(
            "60fps stride 正規化を明示的に無効化する (2026-08-12 追加、既定 "
            "True 化に伴う逃げ道)。--normalize-fps-30 と同時指定した場合は本"
            "フラグ (無効化) が優先される。全フレームであることが要件の"
            "基準データ収集等、既定 ON では困る用途で使う。"
        ),
    )
    # 本番採用の認識フラグ群の自動適用 (2026-08-13 是正、対称化パターンは
    # --normalize-fps-30 と同一)。
    ap.add_argument(
        "--production-recognition", action="store_true",
        dest="production_recognition",
        help=(
            "本番採用の認識フラグ群 (src.production_config.RECOGNITION_ADOPTED: "
            "effect-gate/burst-guard-v2/transition-merge-guard/"
            "burst-gate-open-threshold 0.954/hidden-row-burst-guard/"
            "match-transition-debounce) を load_default() へ自動適用する "
            "(2026-08-13 追加)。既定 True 化により本フラグは実質 no-op "
            "(明示しなくても既定で有効)。後方互換のため残置。"
            "無効化するには --no-production-recognition を使う。"
        ),
    )
    ap.add_argument(
        "--no-production-recognition", action="store_true",
        dest="no_production_recognition",
        help=(
            "本番採用の認識フラグ群の自動適用を明示的に無効化する "
            "(2026-08-13 追加、既定 True 化に伴う逃げ道)。--production-recognition "
            "と同時指定した場合は本フラグ (無効化) が優先される。過去の劣化"
            "構成の再現・A/B比較用。"
        ),
    )
    ap.add_argument(
        "--resize-1080p", action="store_true", dest="resize_1080p",
        help=(
            "認識入力を1920x1080へ正規化してから RecognitionPipeline.update() に"
            "渡す (collect_boards_lean.py:1050 と同一の正規化、2026-08-13 追加)。"
            "既定 True 化により本フラグは実質 no-op (明示しなくても既定で有効)。"
            "後方互換のため残置。無効化するには --no-resize-1080p を使う。"
        ),
    )
    ap.add_argument(
        "--no-resize-1080p", action="store_true", dest="no_resize_1080p",
        help=(
            "1080p正規化を明示的に無効化し、表示キャンバス用サイズ "
            "(1280x720) へ直接縮小したフレームをそのまま認識に渡す従来挙動を"
            "再現する (2026-08-13 追加、既定 True 化に伴う逃げ道。過去の劣化"
            "構成の再現・A/B比較用)。"
        ),
    )
    a = ap.parse_args()
    # 既定値解決 (collect_boards_lean.py と同じ方式): 明示 --no-normalize-fps-30 が
    # 最優先で無効化する。それ以外は --normalize-fps-30 の有無に関わらず既定 True
    # (generate() 関数側の既定と一致させる)。
    normalize_fps_30 = not a.no_normalize_fps_30
    use_production_recognition = not a.no_production_recognition
    resize_1080p = not a.no_resize_1080p
    generate(Path(a.video), Path(a.out), a.max_sec, a.sample_interval,
             start_sec=a.start_sec, end_sec=a.end_sec,
             exclude_video=a.exclude_video, warmup_sec=a.warmup_sec,
             model_dir=a.model_dir,
             show_recognition=a.show_recognition,
             enable_landing_observed_color=a.enable_landing_observed_color,
             force_in_match=a.force_in_match,
             enable_drift_guards=a.enable_drift_guards,
             enable_match_start_full_clear=a.enable_match_start_full_clear,
             enable_recovery_counter_carryover=a.enable_recovery_counter_carryover,
             enable_cnn_flicker_hsv_fallback=a.enable_cnn_flicker_hsv_fallback,
             enable_initial_confirm_vote=a.enable_initial_confirm_vote,
             enable_platt_calibration=a.enable_platt_calibration,
             enable_phase_calibration=a.enable_phase_calibration,
             enable_early_fire_reaction=a.enable_early_fire_reaction,
             enable_early_fire_clear_on_finalize=a.enable_early_fire_clear_on_finalize,
             enable_kill_override_chain_completion=a.enable_kill_override_chain_completion,
             enable_kill_override_chain_gen_accumulate=(
                 a.enable_kill_override_chain_gen_accumulate),
             enable_kill_override_attribution=a.enable_kill_override_attribution,
             enable_kill_override_hysteresis=a.enable_kill_override_hysteresis,
             enable_kill_override_scale_compare=(
                 a.enable_kill_override_scale_compare),
             enable_per_side_settled=a.enable_per_side_settled,
             disable_score_lead_bias=a.disable_score_lead_bias,
             enable_capability_pressure=a.enable_capability_pressure,
             disable_pressure=a.disable_pressure,
             enable_counter_reach=a.enable_counter_reach,
             enable_counter_remaining_time=a.enable_counter_remaining_time,
             enable_counter_defender_only=a.enable_counter_defender_only,
             enable_resolved_exchange_eval=a.enable_resolved_exchange_eval,
             enable_resolved_decisive_amplify=a.enable_resolved_decisive_amplify,
             enable_resolved_live_defender=a.enable_resolved_live_defender,
             enable_resolved_live_defender_strict=a.enable_resolved_live_defender_strict,
             enable_resolved_kill_override=a.enable_resolved_kill_override,
             enable_resolved_kill_override_counter_aware=(
                 a.enable_resolved_kill_override_counter_aware),
             enable_resolved_victim_gen_live=a.enable_resolved_victim_gen_live,
             enable_resolved_episode_physical_redecide=(
                 a.enable_resolved_episode_physical_redecide),
             enable_resolved_episode_physical_consistency_guard=(
                 a.enable_resolved_episode_physical_consistency_guard),
             enable_resolved_minimum_prediction_guard=(
                 a.enable_resolved_minimum_prediction_guard),
             enable_resolved_counter_placement_reuse=(
                 a.enable_resolved_counter_placement_reuse),
             enable_resolved_counter_budget_quantize=(
                 a.enable_resolved_counter_budget_quantize),
             enable_resolved_absolute_chain_end=(
                 a.enable_resolved_absolute_chain_end),
             enable_puyo_to_empty_hsv_guard=a.enable_puyo_to_empty_hsv_guard,
             stable_majority_window=a.stable_majority_window,
             enable_ojama_fall_placement_override=a.enable_ojama_fall_placement_override,
             enable_ojama_fall_entry_hardening=a.enable_ojama_fall_entry_hardening,
             enable_ojama_fall_scoped_exit=a.enable_ojama_fall_scoped_exit,
             enable_pseudo_chain_score_fill=a.enable_pseudo_chain_score_fill,
             chain_hold_base_sec=a.chain_hold_base_sec,
             chain_hold_per_step_sec=a.chain_hold_per_step_sec,
             enable_slide_exit_min_display_guard=(
                 a.enable_slide_exit_min_display_guard),
             layout=a.layout,
             panel_subtitle_h=a.panel_subtitle_h,
             enable_resolved_pending_landing_gate=a.enable_resolved_pending_landing_gate,
             show_excluded_attribution=a.show_excluded_attribution,
             render=a.render,
             dump_timeline_path=a.dump_timeline_path,
             dump_display_timeline_path=a.dump_display_timeline_path,
             enable_exchange_episode_gate=a.enable_exchange_episode_gate,
             dump_exchange_episode_timeline_path=(
                 a.dump_exchange_episode_timeline_path),
             normalize_fps_30=normalize_fps_30,
             use_production_recognition=use_production_recognition,
             resize_1080p=resize_1080p,
             show_chain_count=a.show_chain_count,
             enable_stable_confirmed_is_dead=a.enable_stable_confirmed_is_dead,
             enable_nonstable_hold_is_dead=a.enable_nonstable_hold_is_dead,
             enable_gross_ledger_dump=a.enable_gross_ledger_dump,
             enable_death_confirm_sequence=a.enable_death_confirm_sequence,
             death_next_stationary_sec=a.death_next_stationary_sec)


if __name__ == "__main__":
    main()
