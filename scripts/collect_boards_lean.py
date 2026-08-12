"""軽量 board 抽出パス — SiameseBoardCNN 学習用 npz を高速収集する。

collect_indicators_v2 の重い処理(全指標計算・ojama_disruption 等)を省略し、
confirmed_board グリッドと勝敗 won ラベルのみを蓄積する。

⚠️ 2026-08-12 追加: お邪魔会計 (OjamaAccountingTracker) は例外的に「軽量な
真値記録」として常時駆動する (ojama_net_balance / ojama_forecast 列)。
net収支/forecast は npz からの事後復元が不可能と確定した (score近似v1/v2は
相関0.33-0.38で不合格、tsumo_countゲートv3も不可判定) ため、収集は認識
パイプラインをフル実行しているという前提を活かし、追加コスト僅少 (会計は
辞書演算のみで ChainSimulator 等の重い計算を含まない) な会計計算だけを
本スクリプトに例外的に組み込んだ。全指標計算・ojama_disruption (モンテ
カルロ) は依然として省略する。

## 省略する処理 (既定)
- 全指標計算 (indicators_v2 モジュール呼び出しなし。お邪魔会計のみ例外的に
  上記の通り常時駆動する)
- ojama_disruption (モンテカルロ計算なし)
- NextDetector (load_next_detector=False。--with-next 指定時のみ有効化)
- VideoChainTracker (enable_chain_tracker=False。--enable-chain-tracker 指定時のみ有効化。
  2026-07-30 追加: 機能D (掛け算式検知) 単独では CHAIN 検知が実運用で 0 件
  だった実測があり、CHAIN 中の盤面凍結が機能しない欠陥の疑いがあるため、
  基準データ収集ではこのフラグを明示指定して VideoChainTracker を有効化する。
  省略時は従来通り無効 (後方互換、既存 boards_lean_fixed 系 npz の再現性維持))

## 出力 npz 形式
collect_indicators_v2 --board-npz と同形式 + won / score 列を追加:
  grids      : (N, 13, 6) int8
  video_id   : (N,) str
  side       : (N,) str  "1P" / "2P"
  t_sec      : (N,) float32
  game_idx   : (N,) int32
  frame_idx  : (N,) int32
  won        : (N,) float32  1P視点の勝敗 (1.0/0.0/NaN)
  score      : (N,) int32    スコア (-1 = None)
  next1_a    : (N,) int8     現ネクスト軸ぷよ色 (1-5、未検出/未取得は -1)
  next1_b    : (N,) int8     現ネクスト子ぷよ色 (1-5、未検出/未取得は -1)
  dnext_a    : (N,) int8     ダブルネクスト軸ぷよ色 (1-5、未検出/未取得は -1)
  dnext_b    : (N,) int8     ダブルネクスト子ぷよ色 (1-5、未検出/未取得は -1)
  chain_trigger_sec : (N,) float32  機能D (掛け算表示) 検知時刻。未検知は NaN
                       (2026-07-29 追加、連鎖完了時刻の新方式較正用。
                        既存 boards_lean_fixed 系 npz には存在しない新規キー
                        であり、次回の再収集で初めて実値が入る)
  tsumo_count : (N,) int32   試合開始からの確定ツモ設置数 (手数)。
                       RecognitionPipeline.tsumo_count(side) の値をそのまま
                       記録する (2026-08-12 追加、おじゃま収支近似復元 v3 の
                       着地イベントゲート用。dedup済み STABLE snapshot は
                       1着地に対応しないため、この列の増分を「着地イベント」
                       の代理指標として使う)。取得不能時は -1
                       (TSUMO_COUNT_UNKNOWN)。既存 boards_lean_fixed 系 npz
                       には存在しない新規キーであり、次回の再収集で初めて
                       実値が入る (後方互換: 既存 npz 読み出し側のキー集合
                       には影響しない)。
  all_clear_pending : (N,) int8  全消しボーナス予約中フラグ (0/1)。
                       src.chain_detector.VideoChainTracker.all_clear_pending
                       (公式ルール通りの全消しボーナス未消費ラッチ) をそのまま
                       記録する (2026-08-12 追加。post-hoc の score 跳ね検出
                       近似は過検出気味 (c143実測 ON率6.7%) と判明したため、
                       実運用パイプラインが厳密追跡済みの値を直接保存する)。
                       enable_chain_tracker=False (既定) の収集では
                       VideoChainTracker 自体が無効化されており取得不能 → -1
                       (ALL_CLEAR_PENDING_UNKNOWN)。既存 boards_lean_fixed 系
                       npz には存在しない新規キーであり、次回の再収集で
                       初めて実値が入る (後方互換: 既存 npz 読み出し側の
                       キー集合には影響しない)。
  ojama_net_balance : (N,) float32  お邪魔収支 net (own-perspective、
                       2026-08-12 追加)。ojama_net_balance / ojama_forecast は
                       予測貢献度1〜2位の主力指標だが、npz からの事後復元は
                       不可能と確定した (score近似v1/v2は相関0.33-0.38で
                       不合格、tsumo_countゲートv3も不可判定)。そのため収集
                       中に src.ojama_accounting.OjamaAccountingTracker を
                       実際に駆動し、STABLE snapshot ごとに真値を記録する。
                       値は snapshot.net_balance_capped を own-perspective に
                       変換したもの (1P はそのまま、2P は符号反転)。自分有利
                       方向が正。取得不能時は NaN (OJAMA_NET_BALANCE_UNKNOWN)。
                       ⚠️ 試合境界のリセットは OjamaAccountingTracker.
                       on_state_transition が MENU 遷移/score 大幅減少を検知
                       して内部で自動処理する。本スクリプト側では動画処理
                       開始時に reset() を 1 回呼ぶだけでよく、game_idx が
                       進むたびに外部から reset() してはならない (c系20本の
                       学習データで判明した教訓: 収集を秒区間ごとに分割して
                       都度 reset() すると、区間境界をまたぐ pending お邪魔が
                       消えて会計が壊れる、2026-08-12発見)。既存
                       boards_lean_fixed 系 npz には存在しない新規キーであり、
                       次回の再収集で初めて実値が入る (後方互換)。
  ojama_forecast    : (N,) float32  お邪魔予告 forecast (own-perspective、
                       2026-08-12 追加)。ojama_net_balance と同じ tracker
                       駆動で得る snapshot.forecast_p1/forecast_p2 (自分に
                       向かう予告個数、負値は 0 にクリップ) を side 別に選択
                       した値。取得不能時は NaN (OJAMA_FORECAST_UNKNOWN)。
                       既存 boards_lean_fixed 系 npz には存在しない新規キー
                       (後方互換)。

  ⚠️ next1_*/dnext_* は --with-next を指定した収集時のみ実値が入る。
  未指定 (既定) の場合は NextDetector が無効なため全て -1 (後方互換、
  既存 boards_lean_fixed の再利用に影響なし)。
  ⚠️ chain_trigger_sec は enable_chain_formula_detection (RecognitionPipeline
  既定 True) が有効な収集であれば常に記録される (--with-next 等の追加指定は
  不要)。ただし既存の boards_lean_fixed / boards_lean_fixed_regen_2026-07-28
  npz は本キー追加より前に収集済みのため、chain_trigger_sec 列そのものが
  存在しない (再収集しない限り遡って取得できない)。

## 勝敗 won の自己ラベル付け
score のリセット(前値 - 現値 >= SCORE_RESET_THRESHOLD)でゲーム境界を検知し
game_idx を振る。動画末尾で最終 score が大きい side を勝者とし、
そのゲームの各 snapshot に 1P 視点 won を付与する。
(1P 盤面なら 1P 勝ち=1、1P 負け=0 / 2P 盤面は逆転)

## 使い方
    python -m scripts.collect_boards_lean \\
        --video data/frames/video_29.mp4 \\
        --out-npz /tmp/lean29.npz \\
        --max-sec 30

## --sample-interval による高速化
    --sample-interval 0.1 を指定すると fps*0.1 フレームに 1 回だけ
    pipeline.update を呼ぶ (collect_indicators_v2 と同じ間引き方式)。
    cap.read() は毎フレーム呼んでデコードし、間引き対象フレームは continue
    でスキップする。

    スコアリセット検知への影響:
      score は STABLE snapshot 取得時にのみ読む設計のため、
      間引きで STABLE でない短命フレームを飛ばしても実害なし。
      ゲーム終了時の score リセットは STABLE 直後の数フレームで起こるが
      0.1 秒≒3 フレーム間引きなら次の STABLE フレームで検知できる。
      (worst-case: 間引き幅 * fps フレーム = 約 0.2 秒の検知遅延)

    推奨値: 0.1〜0.2 秒 (≒3×〜6× 高速化、snapshot 数ほぼ変わらず)。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board import Board  # noqa: E402
from src.board_quality import is_phantom_board  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.fps_normalize import resolve_normalize_fps_30_stride  # noqa: E402
from src.ojama_accounting import (  # noqa: E402
    OjamaAccountingTracker,
    OjamaAccountSnapshot,
)
from src.recognition_pipeline import RecognitionPipeline, SideResult  # noqa: E402

# ============================
# 定数
# ============================

# 出力解像度 (認識は 1920x1080 前提)
TARGET_W: int = 1920
TARGET_H: int = 1080
DEFAULT_FPS: float = 30.0

# 試合境界検知: score がこの値以上減少したら新しい試合とみなす
SCORE_RESET_THRESHOLD: int = 500

# ゲーム境界の共有カウンタのデバウンス幅 [秒] (2026-07-31)。
# 1P/2P が同じ境界を検知したときに 2 回進めないための窓。
# 実際の 1 試合は最短 14 秒 (勝利数パネル実測) なので誤抑制しない。
GAME_BOUNDARY_DEBOUNCE_SEC: float = 5.0

# サンプル間引き幅の下限 (0 以下指定は 1 フレームおき = 全フレームに丸める)
MIN_SAMPLE_INTERVAL_FRAMES: int = 1


def _resolve_sample_interval_frames(
    sample_interval_sec: float,
    fps: float,
    sample_interval_frames: Optional[int] = None,
) -> int:
    """認識サンプル間隔を「実際に使うフレーム数」として一意に確定する。

    collect_indicators_v2._resolve_sample_interval_frames と同じ仕様
    (2026-07-28 user指示: fps に依存しない「Nフレームに1回」指定への統一)。
    sample_interval_frames が指定された場合はそれを最優先し、fps に関係なく
    そのフレーム数ごとに 1 回だけ認識する。省略時 (None) は従来通り
    sample_interval_sec (秒) を fps 換算する (完全後方互換)。

    Args:
        sample_interval_sec: 認識サンプル間隔秒 (0 = 全フレーム)。
        fps: 動画の fps。
        sample_interval_frames: 認識サンプル間隔フレーム数 (優先指定、省略可)。
            0 以下が渡された場合も不正値として扱い、下限 1 に丸める。

    Returns:
        実際に使うフレーム間引き幅 (最小 MIN_SAMPLE_INTERVAL_FRAMES)。
    """
    if sample_interval_frames is not None:
        resolved = sample_interval_frames
    else:
        resolved = int(round(sample_interval_sec * fps))
    return max(MIN_SAMPLE_INTERVAL_FRAMES, resolved)


# 勝敗ラベルが付与できない試合の won 値
WON_UNKNOWN: float = float("nan")

# next_pair/dnext_pair が None (未検出 / NextDetector 無効) の場合の埋め値。
# ぷよ色は 1-5 のため -1 は安全な sentinel。
NEXT_COLOR_UNKNOWN: int = -1

# chain_trigger_sec (機能D 掛け算表示検知時刻) が未検知/取得不能の場合の埋め値
# (2026-07-29 追加)。t_sec は常に >= 0 のため NaN は安全な sentinel。
CHAIN_TRIGGER_SEC_UNKNOWN: float = float("nan")

# chain_mechanism (発火検知経路、2026-08-02 Step2 追加) が未検知/取得不能の
# 場合の埋め値。空文字列は CHAIN_MECHANISM_* のどの値とも衝突しないため安全。
CHAIN_MECHANISM_UNKNOWN: str = ""

# tsumo_count (試合開始からの確定ツモ設置数、2026-08-12 追加) が未取得の場合の
# 埋め値。RecognitionPipeline.tsumo_count(side) は常に 0 以上の整数を返すため
# -1 は安全な sentinel。
TSUMO_COUNT_UNKNOWN: int = -1

# all_clear_pending (全消しボーナス予約中フラグ、2026-08-12 追加) が未取得の
# 場合の埋め値。値は 0 (予約なし) / 1 (予約中) の二値のため -1 は安全な
# sentinel (enable_chain_tracker=False 等で VideoChainTracker 自体が無効な
# 収集では常にこの値になる)。
ALL_CLEAR_PENDING_UNKNOWN: int = -1

# ojama_net_balance / ojama_forecast (お邪魔会計の真値、2026-08-12 追加) が
# 未取得の場合の埋め値。t_sec 等と同じ float32 系のため chain_trigger_sec と
# 同方式で NaN sentinel を使う (0 は「収支ゼロ/予告ゼロ」という正当な実値と
# 衝突するため sentinel に使えない)。
OJAMA_NET_BALANCE_UNKNOWN: float = float("nan")
OJAMA_FORECAST_UNKNOWN: float = float("nan")


# ============================
# 蓄積バッファ
# ============================

@dataclass
class _LeanNpzAccumulator:
    """board グリッド + won ラベル蓄積バッファ。

    confirmed_board と score 情報を蓄積し、動画末尾で won を付与して npz 保存する。
    score を保存することで、収集後にオフラインで何度でも勝者ラベルを再作成できる。
    """
    grids: list[np.ndarray] = field(default_factory=list)
    video_ids: list[str] = field(default_factory=list)
    sides: list[str] = field(default_factory=list)
    t_secs: list[float] = field(default_factory=list)
    game_idxs: list[int] = field(default_factory=list)
    frame_idxs: list[int] = field(default_factory=list)
    # won は後付け (動画末尾で付与する)
    wons: list[float] = field(default_factory=list)
    # score を保存: オフライン再ラベル付けを可能にする (None は -1 として保存)
    scores: list[int] = field(default_factory=list)
    # ネクスト情報 (指標①本命版検証用、2026-07 追加)。
    # None は NEXT_COLOR_UNKNOWN (-1) として保存する。既存キー・既存呼び出しの
    # 後方互換のため append() では末尾の optional 引数として追加する。
    next1_as: list[int] = field(default_factory=list)
    next1_bs: list[int] = field(default_factory=list)
    dnext_as: list[int] = field(default_factory=list)
    dnext_bs: list[int] = field(default_factory=list)
    # 機能D (掛け算表示) 検知時刻 (2026-07-29 追加、連鎖完了時刻の新方式較正用)。
    # RecognitionPipeline.SideResult.chain_event.trigger_sec をそのまま記録する。
    # chain_event が None (掛け算表示未検知/連鎖なし) の場合は
    # CHAIN_TRIGGER_SEC_UNKNOWN (NaN) で埋める。既存呼び出し (引数省略) では
    # 常に NaN のまま保存される (後方互換: 挙動不変)。
    chain_trigger_secs: list[float] = field(default_factory=list)
    # 発火検知経路 (CHAIN_MECHANISM_*、2026-08-02 Step2 追加)。
    # 既存呼び出し (mechanism 省略) では CHAIN_MECHANISM_UNKNOWN ("") のまま
    # 蓄積され、save() 時に一度も実値が入らなければ npz キー自体を書かない
    # (後方互換: 既存 npz 読み出し側のキー集合を変えない)。
    chain_mechanisms: list[str] = field(default_factory=list)
    # 試合開始からの確定ツモ設置数 (手数、2026-08-12 追加)。
    # RecognitionPipeline.tsumo_count(side) をそのまま記録する。おじゃま収支
    # 近似復元 v3 の着地イベントゲート用 (dedup済み STABLE snapshot は
    # 1着地に対応しないため、この列の増分を着地イベントの代理指標として使う)。
    # None は TSUMO_COUNT_UNKNOWN (-1) として保存する。既存呼び出し
    # (tsumo_count 省略) では常に -1 のまま保存される (後方互換: 挙動不変)。
    tsumo_counts: list[int] = field(default_factory=list)
    # 全消しボーナス予約中フラグ (0/1、2026-08-12 追加)。
    # VideoChainTracker.all_clear_pending (chain_detector.py) をそのまま
    # 記録する。None は ALL_CLEAR_PENDING_UNKNOWN (-1) として保存する。
    # 既存呼び出し (all_clear_pending 省略) では常に -1 のまま保存される
    # (後方互換: 挙動不変)。
    all_clear_pendings: list[int] = field(default_factory=list)
    # お邪魔会計の真値 (own-perspective、2026-08-12 追加)。
    # OjamaAccountingTracker.get_snapshot() を毎処理フレーム駆動して得た
    # net_balance_capped / forecast_p1・p2 を side 別 own-perspective に
    # 変換した値。None は OJAMA_NET_BALANCE_UNKNOWN / OJAMA_FORECAST_UNKNOWN
    # (NaN) として保存する。既存呼び出し (省略) では常に NaN のまま保存される
    # (後方互換: 挙動不変)。
    ojama_net_balances: list[float] = field(default_factory=list)
    ojama_forecasts: list[float] = field(default_factory=list)

    def append(
        self,
        grid: np.ndarray,
        video_id: str,
        side: str,
        t_sec: float,
        game_idx: int,
        frame_idx: int,
        score: int | None = None,
        next_pair: tuple[int, int] | None = None,
        dnext_pair: tuple[int, int] | None = None,
        chain_trigger_sec: float | None = None,
        mechanism: str | None = None,
        tsumo_count: int | None = None,
        all_clear_pending: int | None = None,
        ojama_net_balance: float | None = None,
        ojama_forecast: float | None = None,
    ) -> None:
        """1 STABLE snapshot を追加する。won は NaN で仮置き。

        Args:
            grid: 確定盤面グリッド (13, 6)。
            video_id: 動画 ID 文字列。
            side: "1P" または "2P"。
            t_sec: タイムスタンプ (秒)。
            game_idx: ゲーム境界カウンタ。
            frame_idx: フレーム絶対番号。
            score: スコア OCR 値。None は -1 に変換して保存。
            next_pair: (軸ぷよ色, 子ぷよ色)。None は NEXT_COLOR_UNKNOWN で保存
                (後方互換: 省略時は既存呼び出しと同じ挙動)。
            dnext_pair: ダブルネクストの (軸ぷよ色, 子ぷよ色)。同上。
            chain_trigger_sec: この snapshot 時点で有効な機能D 検知時刻
                (RecognitionPipeline.SideResult.chain_event.trigger_sec)。
                None は CHAIN_TRIGGER_SEC_UNKNOWN (NaN) で保存する
                (後方互換: 省略時は既存呼び出しと同じ挙動、2026-07-29 追加)。
            mechanism: この snapshot 時点で有効な chain_event.mechanism
                (CHAIN_MECHANISM_* のいずれか)。None は
                CHAIN_MECHANISM_UNKNOWN ("") で保存する (後方互換、
                2026-08-02 追加)。
            tsumo_count: この snapshot 時点の RecognitionPipeline.tsumo_count
                (side) の値 (試合開始からの確定ツモ設置数)。None は
                TSUMO_COUNT_UNKNOWN (-1) で保存する (後方互換: 省略時は既存
                呼び出しと同じ挙動、2026-08-12 追加)。
            all_clear_pending: この snapshot 時点の
                VideoChainTracker.all_clear_pending (bool、全消しボーナス
                予約中フラグ) の値。0/1/None を受け付ける。None は
                ALL_CLEAR_PENDING_UNKNOWN (-1) で保存する (後方互換: 省略時は
                既存呼び出しと同じ挙動、2026-08-12 追加)。
            ojama_net_balance: この snapshot 時点の OjamaAccountingTracker
                収支 (own-perspective、自分有利方向が正)。None は
                OJAMA_NET_BALANCE_UNKNOWN (NaN) で保存する (後方互換: 省略時は
                既存呼び出しと同じ挙動、2026-08-12 追加)。
            ojama_forecast: この snapshot 時点の OjamaAccountingTracker 予告
                個数 (own-perspective、自分に向かう予告個数)。None は
                OJAMA_FORECAST_UNKNOWN (NaN) で保存する (後方互換、
                2026-08-12 追加)。
        """
        self.grids.append(grid.copy())
        self.video_ids.append(video_id)
        self.sides.append(side)
        self.t_secs.append(t_sec)
        self.game_idxs.append(game_idx)
        self.frame_idxs.append(frame_idx)
        self.wons.append(WON_UNKNOWN)
        self.scores.append(score if score is not None else -1)
        n_a, n_b = next_pair if next_pair is not None else (NEXT_COLOR_UNKNOWN, NEXT_COLOR_UNKNOWN)
        d_a, d_b = dnext_pair if dnext_pair is not None else (NEXT_COLOR_UNKNOWN, NEXT_COLOR_UNKNOWN)
        self.next1_as.append(int(n_a))
        self.next1_bs.append(int(n_b))
        self.dnext_as.append(int(d_a))
        self.dnext_bs.append(int(d_b))
        self.chain_trigger_secs.append(
            chain_trigger_sec if chain_trigger_sec is not None else CHAIN_TRIGGER_SEC_UNKNOWN
        )
        self.chain_mechanisms.append(
            mechanism if mechanism is not None else CHAIN_MECHANISM_UNKNOWN
        )
        self.tsumo_counts.append(
            int(tsumo_count) if tsumo_count is not None else TSUMO_COUNT_UNKNOWN
        )
        self.all_clear_pendings.append(
            int(all_clear_pending) if all_clear_pending is not None
            else ALL_CLEAR_PENDING_UNKNOWN
        )
        self.ojama_net_balances.append(
            float(ojama_net_balance) if ojama_net_balance is not None
            else OJAMA_NET_BALANCE_UNKNOWN
        )
        self.ojama_forecasts.append(
            float(ojama_forecast) if ojama_forecast is not None
            else OJAMA_FORECAST_UNKNOWN
        )

    def assign_won_labels(
        self,
        game_final_scores: dict[int, dict[str, int | None]],
    ) -> None:
        """各 game_idx の最終 score から 1P 視点 won を付与する。

        スコア判定を主とし、スコアが同点または欠損の場合のみ
        _winner_by_survival フォールバックで窒息判定を補助する。

        Args:
            game_final_scores: {game_idx: {"1P": score_int|None, "2P": score_int|None}}
        """
        winner_by_game: dict[int, str | None] = {}
        for gidx, scores in game_final_scores.items():
            s1 = scores.get("1P")
            s2 = scores.get("2P")
            if s1 is not None and s2 is not None and s1 != s2:
                # スコアで判定できる場合: 高得点側が勝者
                winner_by_game[gidx] = "1P" if s1 > s2 else "2P"
            else:
                # スコア同点・欠損時: 窒息フォールバック
                winner_by_game[gidx] = _winner_by_survival(self, gidx)

        for i in range(len(self.wons)):
            gidx = self.game_idxs[i]
            winner = winner_by_game.get(gidx)
            if winner is None:
                continue
            # 1P 視点: 自 side が勝者なら 1、負けなら 0
            self.wons[i] = 1.0 if self.sides[i] == winner else 0.0

    def save(self, path: Path) -> None:
        """npz 形式で保存する。grids=(N,13,6) int8、won=(N,) float32、score=(N,) int32。

        next1_a/next1_b/dnext_a/dnext_b (int8) を追加保存する (既存キーは不変、
        後方互換)。--with-next 未指定の収集では全て NEXT_COLOR_UNKNOWN (-1)。
        chain_trigger_sec (float32、2026-07-29 追加) も同様に追加保存する。
        機能D 検知時刻を記録しないだけの既存呼び出しでは全て NaN
        (CHAIN_TRIGGER_SEC_UNKNOWN) になる (後方互換、既存 npz 読み出し側の
        挙動には影響しない新規キー)。

        chain_mechanism (str、2026-08-02 追加) は一度でも実値
        (CHAIN_MECHANISM_UNKNOWN 以外) が記録された場合のみキーを書く。
        一度も記録されなかった (mechanism 未指定の呼び出しのみ、または
        ChainEvent.mechanism が全て None) 場合はキー自体を省略する
        (後方互換: 既存 npz 読み出し側の `set(d.keys())` 依存コードを壊さない)。

        tsumo_count (int32、2026-08-12 追加) は next1_a 等と同様に常に
        追加保存する (既存キーは不変、後方互換)。tsumo_count 未指定の
        既存呼び出しでは全て TSUMO_COUNT_UNKNOWN (-1) になる (後方互換、
        既存 npz 読み出し側の挙動には影響しない新規キー)。

        all_clear_pending (int8、2026-08-12 追加) も同様に常に追加保存する
        (既存キーは不変、後方互換)。all_clear_pending 未指定の既存呼び出し
        では全て ALL_CLEAR_PENDING_UNKNOWN (-1) になる (後方互換、既存 npz
        読み出し側の挙動には影響しない新規キー)。

        ojama_net_balance / ojama_forecast (float32、2026-08-12 追加) も
        同様に常に追加保存する (既存キーは不変、後方互換)。両方未指定の
        既存呼び出しでは全て NaN (OJAMA_NET_BALANCE_UNKNOWN /
        OJAMA_FORECAST_UNKNOWN) になる (後方互換、既存 npz 読み出し側の
        挙動には影響しない新規キー)。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict[str, np.ndarray] = dict(
            grids=np.array(self.grids, dtype=np.int8) if self.grids
                  else np.array([], dtype=np.int8),
            video_id=np.array(self.video_ids),
            side=np.array(self.sides),
            t_sec=np.array(self.t_secs, dtype=np.float32),
            game_idx=np.array(self.game_idxs, dtype=np.int32),
            frame_idx=np.array(self.frame_idxs, dtype=np.int32),
            won=np.array(self.wons, dtype=np.float32),
            score=np.array(self.scores, dtype=np.int32),
            next1_a=np.array(self.next1_as, dtype=np.int8),
            next1_b=np.array(self.next1_bs, dtype=np.int8),
            dnext_a=np.array(self.dnext_as, dtype=np.int8),
            dnext_b=np.array(self.dnext_bs, dtype=np.int8),
            chain_trigger_sec=np.array(self.chain_trigger_secs, dtype=np.float32),
            tsumo_count=np.array(self.tsumo_counts, dtype=np.int32),
            all_clear_pending=np.array(self.all_clear_pendings, dtype=np.int8),
            ojama_net_balance=np.array(self.ojama_net_balances, dtype=np.float32),
            ojama_forecast=np.array(self.ojama_forecasts, dtype=np.float32),
        )
        if any(m != CHAIN_MECHANISM_UNKNOWN for m in self.chain_mechanisms):
            save_kwargs["chain_mechanism"] = np.array(self.chain_mechanisms)
        np.savez_compressed(str(path), **save_kwargs)


# ============================
# 窒息フォールバック判定ヘルパ
# ============================

# 窒息判定: 3列目(index=2)の画面内最上段(row=1、隠し段row0は除く)にぷよがあれば窒息。
# 2026-07-22 ルール是正: 旧 row=0(隠し段)は窒息検知漏れ(完全オーバーフローしないと発火せず)。
# board.py DEATH_ROW と同じ定義に統一。
_DEATH_ROW: int = 1
_DEATH_COL: int = 2


def _winner_by_survival(
    acc: "_LeanNpzAccumulator",
    game_idx: int,
) -> str | None:
    """スコア判定不能時のフォールバック: 窒息していない側を勝者とする。

    各 game_idx の末尾 snapshot の grid で窒息セル (row=_DEATH_ROW=1, col=2 != 0) を確認する。
    どちらも窒息なし / 両方窒息 / snapshot なしの場合は None を返す。

    Args:
        acc: スナップショット蓄積バッファ。
        game_idx: 対象ゲームのインデックス。

    Returns:
        "1P" / "2P" / None (判定不能)
    """
    # game_idx に属するインデックスを side 別に収集
    idx_by_side: dict[str, list[int]] = {"1P": [], "2P": []}
    for i, (gidx, side) in enumerate(zip(acc.game_idxs, acc.sides)):
        if gidx == game_idx and side in idx_by_side:
            idx_by_side[side].append(i)

    def _is_suffocated(indices: list[int]) -> bool | None:
        """末尾 snapshot で窒息しているか判定する。"""
        if not indices:
            return None
        last_i = max(indices, key=lambda i: acc.t_secs[i])
        return bool(acc.grids[last_i][_DEATH_ROW, _DEATH_COL] != 0)

    suf_1p = _is_suffocated(idx_by_side["1P"])
    suf_2p = _is_suffocated(idx_by_side["2P"])

    if suf_1p is None or suf_2p is None:
        return None  # どちらかの snapshot がない
    if suf_1p and not suf_2p:
        return "2P"  # 1P が窒息 → 2P 勝ち
    if suf_2p and not suf_1p:
        return "1P"  # 2P が窒息 → 1P 勝ち
    return None  # 両方窒息・両方生存は判定不能


# ============================
# 1 side の状態管理
# ============================

@dataclass
class _SideState:
    """1 side の間引き・ゲーム境界管理用状態。"""
    game_idx: int = 0
    prev_score: int | None = None
    last_emitted_grid: bytes | None = None
    # game_idx ごとの最終 score を追跡
    final_scores: dict[int, int | None] = field(default_factory=dict)
    # おじゃま会計 tsumo delta drain 用の前回手数 (2026-08-12 追加)。
    # collect_indicators_v2._drain_by_tsumo_delta と同じ役割だが、
    # pipeline.tsumo_count() を再度呼ばず main loop が既に取得済みの値を
    # 再利用するための保持先 (呼び出し回数を変えないための設計)。
    ojama_prev_tsumo: int = 0


@dataclass
class _SharedGameCounter:
    """1P/2P で共有するゲーム境界カウンタ (2026-07-31 の desync 根治)。

    旧実装は `_SideState.game_idx` を **side ごとに独立して**進めていた。
    ゲーム境界は「両者共通の 1 つの事象」なのに、検知は各 side の score
    リセットに依存するため:
      - 検知フレームがずれると game_idx がずれる (実測 57.6% が 5秒超のずれ)
      - **片側の score OCR が壊れている動画 (c26/c58 等) ではその side が
        game 0 に留まり続け、以降すべてのゲームが対応しなくなる**
    その結果 `_merge_final_scores` や won 付与が別ゲーム同士を突き合わせる。

    共有カウンタにすると「どちらかが検知すれば両者が進む」ので、
    片側の検知失敗に耐える。両者が同じ境界を検知したときに 2 回進まないよう
    直前の進行から GAME_BOUNDARY_DEBOUNCE_SEC 以内は進めない
    (実際の 1 試合は最短 14 秒なので誤抑制しない)。
    """
    game_idx: int = 0
    # 最後に境界を進めた時刻 [秒]。None = まだ一度も進めていない。
    last_advance_sec: float | None = None

    def advance_if_new(self, t_sec: float) -> bool:
        """境界を進める (デバウンス内なら進めない)。進めたら True。"""
        if (
            self.last_advance_sec is not None
            and t_sec - self.last_advance_sec < GAME_BOUNDARY_DEBOUNCE_SEC
        ):
            return False
        self.game_idx += 1
        self.last_advance_sec = t_sec
        return True


def _update_game_boundary(
    state: _SideState,
    score: int | None,
    shared: "_SharedGameCounter | None" = None,
    t_sec: float = 0.0,
) -> None:
    """score リセット検知で game_idx を進める。旧ゲームの最終 score は
    リセット直前の prev_score (高値) を記録する。

    バグ修正 (旧): 旧実装はリセット発生フレームで final_scores に ≈0 の低値を
    書き込んでから game_idx を進めていたため、旧ゲームの最終スコアが
    リセット後低値で上書きされ勝者判定が全て None になっていた。

    バグ修正 (2026-07-31): `shared` を渡すと **1P/2P 共有のカウンタ**を使い、
    どちらかが境界を検知すれば両 side が同じ game_idx に揃う。
    shared=None のときは従来の side 独立カウンタ (後方互換)。

    Args:
        state: 対象 side の状態。
        score: 今フレームの score OCR 値 (None は無視)。
        shared: 共有カウンタ。None なら side 独立 (旧挙動)。
        t_sec: 現在時刻 [秒]。shared のデバウンス判定に使う。
    """
    if score is None:
        return
    is_reset = (
        state.prev_score is not None
        and state.prev_score - score >= SCORE_RESET_THRESHOLD
    )
    if is_reset:
        # 旧ゲームの最終スコア = リセット直前の高値を確定
        state.final_scores[state.game_idx] = state.prev_score
        if shared is None:
            state.game_idx += 1
        else:
            shared.advance_if_new(t_sec)
    if shared is not None:
        # 共有カウンタに追従 (相手側が検知した境界にも乗る)
        state.game_idx = shared.game_idx
    # 現ゲームの暫定最終スコア (次フレームで上書きされ続け、最後は真の最終値)
    state.final_scores[state.game_idx] = score
    state.prev_score = score


def _should_emit(
    state: _SideState, board: Board, bstate: BoardState,
    exclude_phantom: bool = False,
) -> bool:
    """STABLE かつ重複でない盤面かを判定する。

    Args:
        exclude_phantom: 幻盤面ガード (2026-08-08)。True で非試合画面
            (対戦カード紹介・ロビー・順位表) 由来の満杯おじゃま盤面を
            記録対象から外す。 collect 側は force_in_match=True で
            MatchStateDetector を無効化しているため、 これらの画面が
            素通りして npz に混入している (実測 0.875%、全 123 本)。
            背景明度による分離は実測で不可能と判明したため
            (幻 min=0.0 / 正常 max=226.9 で完全に重なる)、 盤面の物理的
            整合 (src/board_quality.py) で弾く。
            既定 False = 従来挙動完全維持 (backwards compat)。
    """
    if bstate != BoardState.STABLE or board is None:
        return False
    # 全消し直後 / 試合開始直後 (盤面ぷよ 0) は除外
    if board.count_puyos() == 0:
        return False
    # 幻盤面ガード: 実戦なら窒息死が目前で安定継続し得ない盤面を弾く
    if exclude_phantom and is_phantom_board(board._grid):
        return False
    # 直前と同一盤面なら間引き
    grid_bytes = board._grid.tobytes()
    if grid_bytes == state.last_emitted_grid:
        return False
    return True


# ============================
# おじゃま会計 (2026-08-12 追加)
# ============================
#
# ojama_net_balance / ojama_forecast は npz からの事後復元が不可能と確定した
# ため (score近似v1/v2は相関0.33-0.38で不合格、tsumo_countゲートv3も不可判定)、
# 収集中に OjamaAccountingTracker を実際に駆動して真値を記録する。
# collect_indicators_v2._drive_ojama / _drain_by_tsumo_delta と同じロジック
# だが、以下の点で意図的に分離実装している:
#   1. pipeline.tsumo_count() を再度呼ばない (main loop が tsumo_count npz
#      列用に既に取得済みの値を再利用する。呼び出し回数を変えると既存
#      _FakeLeanPipeline 系テストの呼び出し回数アサーションを壊すため)。
#   2. tsumo_count 未対応 pipeline でも例外にならない (delta=None は skip)。


def _drive_ojama_accounting_lean(
    tracker: OjamaAccountingTracker,
    state_p1: _SideState,
    state_p2: _SideState,
    prev_bstate_p1: BoardState,
    prev_bstate_p2: BoardState,
    p1: SideResult,
    p2: SideResult,
    tsumo_count_1p: int | None,
    tsumo_count_2p: int | None,
    t_sec: float,
) -> OjamaAccountSnapshot:
    """OjamaAccountingTracker を毎処理フレーム駆動し、現在の snapshot を返す。

    試合境界のリセットは tracker.on_state_transition が MENU 遷移/score
    大幅減少を検知して内部で自動処理する。呼び出し側 (collect_lean) は
    動画処理開始時に reset() を 1 回呼ぶだけでよく、本関数からは reset() を
    一切呼ばない (c系20本の学習データで判明した教訓: 収集を秒区間ごとに
    分割して都度 reset() すると、区間境界をまたぐ pending お邪魔がリセット
    で消えて会計が壊れる、2026-08-12発見)。

    Args:
        tracker: お邪魔会計追跡器 (動画 1 本につき 1 個、呼出元で保持)。
        state_p1, state_p2: tsumo delta drain 用の前回手数を保持する状態。
        prev_bstate_p1, prev_bstate_p2: 前フレームの各 side の状態。
        p1, p2: 今フレームの pipeline.update() 結果 (side 別)。
        tsumo_count_1p, tsumo_count_2p: 今フレームの
            RecognitionPipeline.tsumo_count(side) の値 (main loop で既に
            取得済み)。None は取得不能 (drain しない)。
        t_sec: 現在時刻 (秒)。
    """
    tracker.on_state_transition("p1", prev_bstate_p1, p1.state, p1.score, t_sec)
    tracker.on_state_transition("p2", prev_bstate_p2, p2.state, p2.score, t_sec)
    _drain_ojama_by_tsumo_delta_lean(tracker, "p1", state_p1, tsumo_count_1p, t_sec)
    _drain_ojama_by_tsumo_delta_lean(tracker, "p2", state_p2, tsumo_count_2p, t_sec)
    return tracker.get_snapshot(t_sec)


def _drain_ojama_by_tsumo_delta_lean(
    tracker: OjamaAccountingTracker,
    ojama_key: str,
    state: _SideState,
    tsumo_count: int | None,
    t_sec: float,
) -> None:
    """tsumo_count の増分 delta 回 on_tsumo_settled を呼ぶ。

    試合境界 (手数リセット) では delta < 0 になるため skip する
    (会計は on_state_transition の MENU/score減少検知で既にリセット済み)。
    tsumo_count=None (pipeline 未対応/未取得) の場合は何もしない。
    """
    if tsumo_count is None:
        return
    delta = tsumo_count - state.ojama_prev_tsumo
    if delta > 0:
        for _ in range(delta):
            tracker.on_tsumo_settled(ojama_key, t_sec)
    state.ojama_prev_tsumo = tsumo_count


def _ojama_snapshot_to_own_perspective(
    snap: OjamaAccountSnapshot,
) -> tuple[float, float, float, float]:
    """snapshot を 1P/2P 双方の own-perspective (net, forecast) に変換する。

    net は snap.net_balance_capped の own-perspective 変換 (1P はそのまま、
    2P は符号反転)。forecast は snap.forecast_p1/p2 を負値 0 クリップして
    side 別に選択する (src.indicators_v2.ojama_net_balance/ojama_forecast の
    .raw 定義と一致させ、収集後の値と学習時の値を一致させる)。

    Returns:
        (net_1p, forecast_1p, net_2p, forecast_2p) の 4 要素タプル。
    """
    net_1p = float(snap.net_balance_capped)
    net_2p = -net_1p
    forecast_1p = float(max(0, snap.forecast_p1))
    forecast_2p = float(max(0, snap.forecast_p2))
    return net_1p, forecast_1p, net_2p, forecast_2p


# ============================
# メイン収集ループ
# ============================

def collect_lean(
    video_path: Path,
    out_npz: Path,
    max_sec: float = 0.0,
    start_sec: float = 0.0,
    sample_interval_sec: float = 0.0,
    capture_next: bool = False,
    sample_interval_frames: Optional[int] = None,
    enable_chain_tracker: bool = False,
    normalize_fps_30: bool = True,
    enable_effect_gate: bool = False,
    effect_gate_persist_sec: Optional[float] = None,
    enable_effect_visual_gate: bool = False,
    enable_burst_guard_v2: bool = False,
    enable_transition_merge_guard: bool = False,
    burst_gate_open_threshold: Optional[float] = None,
    enable_hidden_row_burst_guard: bool = False,
    enable_burst_close_extension: bool = False,
    burst_chain_gap_max_sec: Optional[float] = None,
    enable_online_hsv_refresh: bool = False,
    enable_match_transition_debounce: bool = False,
    enable_ojama_entry_gravity_settle_guard: bool = False,
    enable_gravity_settle_reset_on_exit: bool = False,
    enable_phantom_board_guard: bool = False,
    enable_margin_time_rate: bool = False,
    enable_stable_majority_window: bool = False,
) -> int:
    """1 動画を処理して盤面 npz を出力する。指標計算は一切行わない。

    Args:
        video_path: 入力動画パス。
        out_npz: 出力 npz パス。
        max_sec: 処理最大秒数 (0=全長)。
        start_sec: 処理開始オフセット秒。
        sample_interval_sec: フレーム間引き間隔 (秒)。0 = 全フレーム処理
            (従来挙動)。collect_indicators_v2 と同じ間引き方式を採用:
            cap.read() は毎フレーム呼び、sample_interval_frames おきに
            pipeline.update を呼ぶ。
        capture_next: True で NextDetector を有効化し next1_a/next1_b/
            dnext_a/dnext_b を実値で記録する (指標①本命版検証用)。
            既定 False = 従来挙動 (NextDetector 無効、全て -1 で保存、
            後方互換)。
        sample_interval_frames: フレーム間引き間隔 (フレーム数、省略可)。
            指定すると fps に関係なくそのフレーム数ごとに 1 回認識し、
            sample_interval_sec より優先される (2026-07-28 追加)。
            省略時 (None) は sample_interval_sec の従来挙動を完全維持する
            (後方互換)。実際に使われた間引き幅は標準出力にログされる。
        enable_chain_tracker: True で VideoChainTracker (1P/2P) を有効化する。
            既定 False = 従来挙動 (無効、後方互換、既存 boards_lean_fixed 系
            npz の再現性を維持する)。
            2026-07-30 追記: 機能D (掛け算式検知, enable_chain_formula_detection)
            のみでは実運用で CHAIN 検知が 0 件だった実測 (chain_trigger_sec
            非NaN率 0.0%) があり、CHAIN 期間中に盤面が凍結されず消去途中の
            盤面が STABLE 扱いされる欠陥の疑いがある。VideoChainTracker は
            visualize_advantage_overlay.py / collect_indicators_v2.py の
            本番経路で既定 True (実績あり) のため、基準データ収集ではこちらを
            有効化する。
        normalize_fps_30: True (既定) で 60fps 動画を stride-2 (実効30fps) に
            間引く (src.fps_normalize.resolve_normalize_fps_30_stride、
            2026-07-30 追加)。60fps 動画を全フレーム処理すると
            board_state_machine.py 等のフレーム数定数が想定する実時間の半分に
            なる問題への対処。優先順位は「明示 --sample-interval-frames > 自動
            --normalize-fps-30」: sample_interval_frames が明示指定されている
            場合は本フラグを無視する。
            2026-07-30 既定 True 化 (user承認済み): A/B実測で 60fps stride-2 は
            連鎖数誤り10.4%・列まるごと欠損0%・盤面相違26.0%(中央値1セル)と、
            30fps動画の15fps間引き (26.1%・21.7%・48.4%) より一貫して良好、
            かつ既存24本の30fps動画と時間解像度が揃う (データセットの世代混在
            を防ぐ)。無効化するには明示 --no-normalize-fps-30 (または本関数を
            呼ぶ側で normalize_fps_30=False) を指定する。False 指定時は
            従来挙動・bit-identical (30fps未満動画では stride=1 で常に無変化)。
        enable_effect_gate: エフェクト時間ゲート (2026-08-03、A/B 計測用)。
            True で相手連鎖中/自お邪魔着弾直後 window の間、自盤面上段
            (board_state_machine.EFFECT_GATE_TOP_ROWS) の cell 更新に実秒
            ベース持続確認を要求する (満杯盤面 47 セル誤り根治の検証用)。
            既定 False = 従来挙動完全維持 (backwards compat)。
        effect_gate_persist_sec: 上記ゲートの確定に必要な持続秒数。
            None (既定) なら RecognitionPipeline 既定値 (EFFECT_PERSIST_SEC
            =0.4秒) を使う。enable_effect_gate=False の間は無視される。
        enable_effect_visual_gate: 案B 4条件AND拡張 (2026-08-04、A/B 計測用)。
            True で effect_gate_window_active を「(既存時間窓) AND (not 自
            連鎖中) AND (not 全消しラッチ) AND (視覚グロー検出)」に拡張する。
            enable_effect_gate=False の間は無視される (時間窓自体が発生しない
            ため)。既定 False = 従来挙動完全維持 (backwards compat)。
        enable_burst_guard_v2: バーストガード再設計 Stage1 (2026-08-05、A/B
            計測用、docs/BURST_GUARD_DESIGN_2026-08-05.md)。True で
            effect_gate_window_active の計算を Schmitt trigger 視覚トリガー
            + ハード凍結方式に切り替える (案Bの enable_effect_visual_gate
            経路とは排他)。enable_effect_gate=False の間は no-op (警告ログ)。
            既定 False = 従来挙動完全維持 (backwards compat)。
        enable_transition_merge_guard: バーストガード Stage1.5 (2026-08-05
            アーキ追補、A/B 計測用)。True で NON-STABLE→STABLE 遷移merge
            (`_merge_diff_only`) の直前に、物理的期待値フィルタ
            (`_filter_transition_new_cnn_for_burst_guard`) を
            effect_gate_window_active 中のみ適用する。
            enable_burst_guard_v2=False の間は no-op (警告ログ)。
            既定 False = 従来挙動完全維持 (backwards compat)。
        burst_gate_open_threshold: バーストガード緊急較正 (2026-08-05、
            factorialバックテスト用)。None (既定) なら BURST_GATE_OPEN_
            THRESHOLD (=0.97) を使う (bit-identical)。CLOSE も同値運用。
        enable_hidden_row_burst_guard: バーストガード Stage1.5b (2026-08-05
            アーキ追補、§11、A/B 計測用)。True で row1-3 凍結中/close直後
            クールダウン中の infer_hidden_row 呼び出しをスキップし、row0
            (隠し段) の確信度100%誤色書き込みを防ぐ。
            enable_burst_guard_v2=False の間は no-op (警告ログ)。
            既定 False = 従来挙動完全維持 (backwards compat)。
        enable_burst_close_extension: バーストガード §12 close側再設計
            (2026-08-05 アーキ確定、A/B 計測用)。True で生 is_open と実効
            active信号 (遷移mergeフィルタ+hard freeze の適用条件) を分離し、
            close後 BURST_GATE_POST_CLOSE_COOLDOWN_SEC のクールダウン、
            および相手連鎖継続中の延長 (トリガーではない) を実効側に反映する。
            enable_burst_guard_v2=False の間は no-op (警告ログ)。
            既定 False = 従来挙動完全維持 (backwards compat)。
        burst_chain_gap_max_sec: バーストガード §12 緊急パラメータ化
            (2026-08-05、A/B 計測用)。相手連鎖延長の再点火間隔上限を上書き
            する。None (既定) = モジュール定数 BURST_GATE_OPPONENT_CHAIN_
            GAP_MAX_SEC (=3.3、bit-identical)。**0.0 を渡すと延長を常に
            不成立にできる** (差分実験で busy局面の凍結連鎖の犯人と確定した
            延長機構をA/B測定で切る用途、close後クールダウン0.9秒は無関係
            のため無改修で残る)。
        enable_online_hsv_refresh: 長時間劣化修正 A+B (2026-08-06、
            docs/LONGRUN_DEGRADATION_INVESTIGATION_2026-08-06.md §1/§4、
            A/B 計測用)。True で (A)試合毎に OnlineHsvCalibrator の較正を
            リセット、(B)inject後もupdate()+再inject判定を継続する
            (凍結ガード撤廃)。詳細は RecognitionPipeline.__init__ 参照。
            既定 False = 従来挙動完全維持 (backwards compat)。
        enable_match_transition_debounce: 長時間劣化修正 A' (2026-08-06、
            docs/LONGRUN_DEGRADATION_INVESTIGATION_2026-08-06.md §4追補)。
            True で is_active の True/False遷移を対称デバウンスし、
            MATCH_TRANSITION_DEBOUNCE_SEC (1.0秒) 未満のフリッカーによる
            _match_active_started_frame/_time の誤再アーム/リセットを防ぐ。
            既定 False = 従来挙動完全維持 (backwards compat)。
        enable_ojama_entry_gravity_settle_guard: 修正B (2026-08-08、
            振動バグB+C の修正)。True で GRAVITY_SETTLE 中の OJAMA_FALL
            新規発火を禁止する。連鎖の段間重力待ちを横取りされると
            GravitySettleDetector 内部カウンタが残留し、次回進入時に誤って
            1 frame で STABLE 化するバグ (バグC) を誘発する。
            enable_gravity_settle_reset_on_exit と対で使う。
            既定 False = 従来挙動完全維持 (backwards compat)。
        enable_gravity_settle_reset_on_exit: 修正C (2026-08-08、
            振動バグB+C の修正)。True で GRAVITY_SETTLE が他 detector に
            横取りされて弾き出された際、GravitySettleDetector の内部
            カウンタ (_settle_start_frame 等) をその場でリセットする。
            既定 False = 従来挙動完全維持 (backwards compat)。
        enable_margin_time_rate: マージンタイム逓減 (2026-08-09)。True で
            おじゃま判定の閾値を経過時間に応じた実効レートにする。
            従来は 70 点固定で、長い試合の後半 (実レートが 22 点まで下がる)
            では着弾を丸ごと見逃していた (npz 実測で全着弾の 6.27%)。
            起点は最初の1手から 95.5 秒 (user伝授)。
            既定 False = 従来挙動完全維持 (backwards compat)。
        enable_stable_majority_window: 盤面確定窓 3中2多数決 (2026-08-13
            user承認、認識99.5%物差し条件付き採用)。True で初回STABLE確定窓が
            「stable_frame_count 連続厳密一致」から「直近3観測中2一致」に
            切り替わる (src/board_state_machine.py 参照)。148動画収集走行中
            のため既定 False 必須 (backwards compat)。
        enable_phantom_board_guard: 幻盤面ガード (2026-08-08)。True で
            非試合画面 (対戦カード紹介・ロビー・順位表) 由来の満杯おじゃま
            盤面を snapshot として記録しない。本スクリプトは
            force_in_match=True で MatchStateDetector を無効化しているため
            これらの画面が素通りしており、実測で全 123 本・0.875% の
            スナップショットが該当した (実画面 4/4 で真陽性)。背景明度に
            よる分離は実測不可能と判明したため (幻 min=0.0 / 正常
            max=226.9 で完全に重なる)、盤面の物理的整合
            (src/board_quality.py) で弾く。
            既定 False = 従来挙動完全維持 (backwards compat)。

    Returns:
        蓄積した snapshot 数。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[lean] cannot open: {video_path}", file=sys.stderr)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps) if start_sec > 0.0 else 0
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    if max_sec > 0:
        end_frame = min(total_frames, start_frame + int(max_sec * fps))
    else:
        end_frame = total_frames
    n_frames = max(0, end_frame - start_frame)

    video_id = video_path.stem

    # --- fps正規化 (2026-07-30 追加、既定 OFF) ---
    # 明示 --sample-interval-frames が優先。未指定かつ normalize_fps_30=True の
    # ときのみ、60fps 等の動画を実効30fps に揃える stride を自動注入する。
    if sample_interval_frames is None and normalize_fps_30:
        sample_interval_frames = resolve_normalize_fps_30_stride(fps)

    # --- フレーム間引き設定 (collect_indicators_v2 と同じ計算式) ---
    # sample_interval_frames 指定時はそちらを優先、省略時は従来通り秒換算
    effective_interval_frames = _resolve_sample_interval_frames(
        sample_interval_sec, fps, sample_interval_frames,
    )
    # 実際に使われた間引き幅を明示ログ (fps 違いによる意図しない間引きの
    # 見落としを後から気付けるようにするため、2026-07-28 追加)。
    print(
        f"[lean] sample_interval: {effective_interval_frames} frames "
        f"(fps={fps:.3f}, sample_interval_sec={sample_interval_sec}, "
        f"sample_interval_frames_arg={sample_interval_frames})"
    )

    # NextDetector / ChainTracker は既定 OFF で高速化
    # (capture_next=True の場合のみ NextDetector を有効化、指標①本命版検証用)
    # (enable_chain_tracker=True の場合のみ VideoChainTracker を有効化、
    #  2026-07-30 基準データ収集で CHAIN 期間中の盤面凍結を機能させるため追加)
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=enable_chain_tracker,
        temporal_smoothing=1,
        load_next_detector=capture_next,
        force_in_match=True,
        enable_effect_gate=enable_effect_gate,
        effect_gate_persist_sec=effect_gate_persist_sec,
        enable_effect_visual_gate=enable_effect_visual_gate,
        enable_burst_guard_v2=enable_burst_guard_v2,
        enable_transition_merge_guard=enable_transition_merge_guard,
        burst_gate_open_threshold=burst_gate_open_threshold,
        enable_hidden_row_burst_guard=enable_hidden_row_burst_guard,
        enable_burst_close_extension=enable_burst_close_extension,
        burst_chain_gap_max_sec=burst_chain_gap_max_sec,
        enable_online_hsv_refresh=enable_online_hsv_refresh,
        enable_match_transition_debounce=enable_match_transition_debounce,
        enable_ojama_entry_gravity_settle_guard=(
            enable_ojama_entry_gravity_settle_guard
        ),
        enable_gravity_settle_reset_on_exit=enable_gravity_settle_reset_on_exit,
        enable_margin_time_rate=enable_margin_time_rate,
        stable_majority_window=enable_stable_majority_window,
    )
    # 動画 ID をセット (per-video HSV プロファイル自動ロード用)
    vid_match = __import__("re").search(r"(v\d+|video_\d+)", video_path.name)
    if vid_match and hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(vid_match.group(1))

    acc = _LeanNpzAccumulator()
    state_p1 = _SideState()
    state_p2 = _SideState()
    # ゲーム境界は両者共通の 1 つの事象なので共有カウンタで管理する
    # (2026-07-31 desync 根治)。片側の score OCR が壊れていても揃う。
    shared_game = _SharedGameCounter()

    # おじゃま会計 (2026-08-12 追加): 動画処理開始時に一度だけ生成・リセット
    # する。試合 (game_idx) が進むたびに reset() してはならない
    # (_drive_ojama_accounting_lean のコメント参照、c系20本の教訓)。
    ojama_tracker = OjamaAccountingTracker()
    ojama_tracker.reset()
    prev_bstate_p1 = BoardState.MENU
    prev_bstate_p2 = BoardState.MENU

    for local_i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        # --- フレーム間引き: effective_interval_frames おきに pipeline.update を呼ぶ ---
        # cap.read() は毎フレーム呼んでデコードし、間引き対象フレームはスキップ。
        # (collect_indicators_v2 と同じ方式)
        if local_i % effective_interval_frames != 0:
            continue
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        fi = start_frame + local_i
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)

        # 試合開始からの確定ツモ設置数 (2026-08-12 追加、着地イベント代理指標用)。
        # pipeline が tsumo_count 未対応 (フェイク等) の場合は None のまま
        # (後方互換: _process_side_lean 側で TSUMO_COUNT_UNKNOWN に埋められる)。
        get_tsumo_count = getattr(pipeline, "tsumo_count", None)
        tsumo_count_1p = get_tsumo_count("1P") if callable(get_tsumo_count) else None
        tsumo_count_2p = get_tsumo_count("2P") if callable(get_tsumo_count) else None

        # おじゃま会計 (2026-08-12 追加): dedup済み STABLE snapshot だけでなく
        # 毎処理フレーム密に駆動する (スコア変化・連鎖終了の密な観測が必要
        # なため、production_config の --sample-interval 0 採用根拠と同じ
        # 理由)。tsumo_count_1p/2p は直前で取得済みの値を再利用し、
        # pipeline.tsumo_count() を再度呼ばない (呼び出し回数を変えない)。
        ojama_snap = _drive_ojama_accounting_lean(
            ojama_tracker, state_p1, state_p2,
            prev_bstate_p1, prev_bstate_p2,
            result.p1, result.p2,
            tsumo_count_1p, tsumo_count_2p, t_sec,
        )
        prev_bstate_p1 = result.p1.state
        prev_bstate_p2 = result.p2.state
        ojama_net_1p, ojama_forecast_1p, ojama_net_2p, ojama_forecast_2p = (
            _ojama_snapshot_to_own_perspective(ojama_snap)
        )

        # 全消しボーナス予約中フラグ (2026-08-12 追加)。VideoChainTracker.
        # all_clear_pending (chain_detector.py) が公式ルール通りの厳密ラッチを
        # 保持しているためそのまま取得する。RecognitionPipeline は
        # 現時点で side 別の公開 getter を持たないため、内部で保持する
        # side 別 VideoChainTracker (_chain_tracker_1p / _chain_tracker_2p) に
        # getattr で安全に参照する。enable_chain_tracker=False (既定) の
        # 収集では tracker が None のため None のまま
        # (後方互換: _process_side_lean 側で ALL_CLEAR_PENDING_UNKNOWN に埋める)。
        chain_tracker_1p = getattr(pipeline, "_chain_tracker_1p", None)
        chain_tracker_2p = getattr(pipeline, "_chain_tracker_2p", None)
        all_clear_pending_1p = getattr(chain_tracker_1p, "all_clear_pending", None)
        all_clear_pending_2p = getattr(chain_tracker_2p, "all_clear_pending", None)

        _process_side_lean(
            acc, state_p1, "1P", result.p1.confirmed_board,
            result.p1.state, result.p1.score, video_id, t_sec, fi,
            next_pair=result.p1.next_pair, dnext_pair=result.p1.dnext_pair,
            chain_event=result.p1.chain_event, shared_game=shared_game,
            exclude_phantom=enable_phantom_board_guard,
            tsumo_count=tsumo_count_1p,
            all_clear_pending=all_clear_pending_1p,
            ojama_net_balance=ojama_net_1p,
            ojama_forecast=ojama_forecast_1p,
        )
        _process_side_lean(
            acc, state_p2, "2P", result.p2.confirmed_board,
            result.p2.state, result.p2.score, video_id, t_sec, fi,
            next_pair=result.p2.next_pair, dnext_pair=result.p2.dnext_pair,
            chain_event=result.p2.chain_event, shared_game=shared_game,
            exclude_phantom=enable_phantom_board_guard,
            tsumo_count=tsumo_count_2p,
            all_clear_pending=all_clear_pending_2p,
            ojama_net_balance=ojama_net_2p,
            ojama_forecast=ojama_forecast_2p,
        )
    cap.release()

    # 勝敗ラベルを付与して保存
    combined_final = _merge_final_scores(state_p1, state_p2)
    acc.assign_won_labels(combined_final)
    acc.save(out_npz)
    return len(acc.grids)


def _process_side_lean(
    acc: _LeanNpzAccumulator,
    state: _SideState,
    side_label: str,
    board: Optional[Board],
    bstate: BoardState,
    score: int | None,
    video_id: str,
    t_sec: float,
    frame_idx: int,
    next_pair: tuple[int, int] | None = None,
    dnext_pair: tuple[int, int] | None = None,
    chain_event: object | None = None,
    shared_game: "_SharedGameCounter | None" = None,
    exclude_phantom: bool = False,
    tsumo_count: int | None = None,
    all_clear_pending: int | None = None,
    ojama_net_balance: float | None = None,
    ojama_forecast: float | None = None,
) -> None:
    """1 side の STABLE snapshot を蓄積する。指標計算は行わない。

    next_pair/dnext_pair は capture_next=False (既定) の呼び出しでは常に
    None (SideResult 既定値) となり、acc.append 側で -1 埋めされる
    (後方互換)。

    Args:
        chain_event: result.p{1,2}.chain_event (src.recognition_pipeline.
            ChainEvent | None、循環import回避のため object 型ヒント)。
            機能D 検知時刻 (.trigger_sec) を chain_trigger_sec として記録する
            (2026-07-29 追加、既存呼び出しは省略可・挙動不変)。
            .mechanism (CHAIN_MECHANISM_* | None) を chain_mechanism として
            記録する (2026-08-02 Step2 追加、同様に省略可・挙動不変)。
        shared_game: 1P/2P 共有のゲーム境界カウンタ (2026-07-31)。
            渡すと片側の score OCR 破綻でも game_idx がずれない。
            None なら従来の side 独立カウンタ (後方互換)。
        tsumo_count: RecognitionPipeline.tsumo_count(side) の値 (この
            snapshot 時点の試合開始からの確定ツモ設置数)。None は
            acc.append 側で TSUMO_COUNT_UNKNOWN (-1) に埋められる
            (後方互換: 既存呼び出しは省略可・挙動不変、2026-08-12 追加)。
        all_clear_pending: この snapshot 時点の VideoChainTracker.
            all_clear_pending (全消しボーナス予約中フラグ) の値。None は
            acc.append 側で ALL_CLEAR_PENDING_UNKNOWN (-1) に埋められる
            (後方互換: 既存呼び出しは省略可・挙動不変、2026-08-12 追加)。
        ojama_net_balance: この snapshot 時点のお邪魔会計 net 収支
            (own-perspective)。None は acc.append 側で
            OJAMA_NET_BALANCE_UNKNOWN (NaN) に埋められる (後方互換: 既存
            呼び出しは省略可・挙動不変、2026-08-12 追加)。
        ojama_forecast: この snapshot 時点のお邪魔会計予告個数
            (own-perspective)。None は acc.append 側で
            OJAMA_FORECAST_UNKNOWN (NaN) に埋められる (後方互換、
            2026-08-12 追加)。
        exclude_phantom: 幻盤面ガード (2026-08-08)。True で非試合画面由来の
            満杯おじゃま盤面を記録しない。既定 False = 従来挙動完全維持。
    """
    _update_game_boundary(state, score, shared=shared_game, t_sec=t_sec)
    if board is None or not _should_emit(
        state, board, bstate, exclude_phantom=exclude_phantom,
    ):
        return
    trigger_sec = getattr(chain_event, "trigger_sec", None) if chain_event is not None else None
    mechanism = getattr(chain_event, "mechanism", None) if chain_event is not None else None
    acc.append(
        board._grid, video_id, side_label,
        round(t_sec, 3), state.game_idx, frame_idx,
        score=score, next_pair=next_pair, dnext_pair=dnext_pair,
        chain_trigger_sec=trigger_sec, mechanism=mechanism,
        tsumo_count=tsumo_count, all_clear_pending=all_clear_pending,
        ojama_net_balance=ojama_net_balance, ojama_forecast=ojama_forecast,
    )
    state.last_emitted_grid = board._grid.tobytes()


def _merge_final_scores(
    state_p1: _SideState,
    state_p2: _SideState,
) -> dict[int, dict[str, int | None]]:
    """両 side の final_scores を game_idx をキーに統合する。

    Returns:
        {game_idx: {"1P": score_or_None, "2P": score_or_None}}
    """
    all_games: set[int] = set(state_p1.final_scores) | set(state_p2.final_scores)
    result: dict[int, dict[str, int | None]] = {}
    for gidx in all_games:
        result[gidx] = {
            "1P": state_p1.final_scores.get(gidx),
            "2P": state_p2.final_scores.get(gidx),
        }
    return result


# ============================
# CLI エントリポイント
# ============================

def main() -> int:
    """CLI エントリポイント。"""
    parser = argparse.ArgumentParser(description="軽量 board 抽出 (SiameseBoardCNN 学習用)")
    parser.add_argument("--video", type=Path, required=True, help="入力動画パス")
    parser.add_argument("--out-npz", type=Path, required=True, help="出力 npz パス")
    parser.add_argument(
        "--max-sec", type=float, default=0.0,
        help="処理する最大秒数 (0=全長)",
    )
    parser.add_argument(
        "--start-sec", type=float, default=0.0,
        help="処理開始オフセット秒",
    )
    parser.add_argument(
        "--sample-interval", type=float, default=0.0,
        dest="sample_interval",
        help=(
            "フレーム間引き間隔 (秒)。0 = 全フレーム処理 (既定)。"
            "0.1 で約 3×、0.2 で約 6× 高速化。"
            "STABLE 検出・勝者判定には影響しない。"
            "--sample-interval-frames 指定時はそちらが優先される"
        ),
    )
    parser.add_argument(
        "--sample-interval-frames", type=int, default=None,
        dest="sample_interval_frames",
        help=(
            "フレーム間引き間隔 (フレーム数、省略可、整数)。"
            "fps に関係なくこのフレーム数ごとに 1 回認識する。"
            "--sample-interval (秒) より優先される。"
            "例: 8フレームに1回 (60fps 想定) なら --sample-interval-frames 8"
        ),
    )
    parser.add_argument(
        "--with-next", action="store_true", dest="with_next",
        help=(
            "NextDetector を有効化し next1_a/next1_b/dnext_a/dnext_b を"
            "実値で記録する (指標①本命版検証用)。既定は無効 (-1 埋め、後方互換)。"
        ),
    )
    parser.add_argument(
        "--enable-chain-tracker", action="store_true", dest="enable_chain_tracker",
        help=(
            "VideoChainTracker (1P/2P) を有効化する。既定は無効 (後方互換、"
            "既存 boards_lean_fixed 系 npz の再現性維持)。"
            "2026-07-30 追加: 機能D (掛け算式検知) 単独では CHAIN 検知が"
            "実運用で 0 件だった実測があり、基準データ収集ではこちらを有効化する。"
        ),
    )
    parser.add_argument(
        "--normalize-fps-30", action="store_true", dest="normalize_fps_30",
        help=(
            "60fps 等の動画を stride-2 相当 (実効30fps) に間引く "
            "(src.fps_normalize.resolve_normalize_fps_30_stride、2026-07-30 追加)。"
            "--sample-interval-frames が明示指定されている場合はそちらが優先され、"
            "本フラグは無視される。"
            "2026-07-30 既定 True 化 (user承認済み) により本フラグは実質 no-op "
            "(明示しなくても既定で有効)。後方互換のため残置。"
            "無効化するには --no-normalize-fps-30 を使う。"
        ),
    )
    parser.add_argument(
        "--no-normalize-fps-30", action="store_true", dest="no_normalize_fps_30",
        help=(
            "60fps stride 正規化を明示的に無効化する (2026-07-30 追加、既定 "
            "True 化に伴う逃げ道)。--normalize-fps-30 と同時指定した場合は本"
            "フラグ (無効化) が優先される。全フレームであることが要件の"
            "基準データ収集等、既定 ON では困る用途で使う。"
        ),
    )
    parser.add_argument(
        "--enable-effect-gate", action="store_true", dest="enable_effect_gate",
        help=(
            "エフェクト時間ゲート (2026-08-03、A/B 計測用) を有効化する。"
            "満杯盤面 47 セル誤り根治の効果測定に使う。既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--enable-effect-visual-gate", action="store_true",
        dest="enable_effect_visual_gate",
        help=(
            "案B 4条件AND拡張 (2026-08-04、A/B 計測用) を有効化する。"
            "--enable-effect-gate の時間窓に (not 自連鎖中) AND (not 全消し"
            "ラッチ) AND (視覚グロー検出) を追加する。--enable-effect-gate が"
            "無効の間は無視される。既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--effect-gate-persist-sec", type=float, default=None,
        dest="effect_gate_persist_sec",
        help="エフェクト時間ゲートの確定に必要な持続秒数 (既定 0.4秒)。",
    )
    parser.add_argument(
        "--enable-burst-guard-v2", action="store_true",
        dest="enable_burst_guard_v2",
        help=(
            "バーストガード再設計 Stage1 (2026-08-05、A/B 計測用) を有効化する。"
            "docs/BURST_GUARD_DESIGN_2026-08-05.md。Schmitt trigger視覚トリガー"
            "+ハード凍結方式に effect_gate_window_active の計算を切り替える"
            "(--enable-effect-visual-gate とは排他)。--enable-effect-gate が"
            "無効の間は no-op (警告ログ)。既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--enable-transition-merge-guard", action="store_true",
        dest="enable_transition_merge_guard",
        help=(
            "バーストガード Stage1.5 (2026-08-05 アーキ追補、A/B 計測用) を"
            "有効化する。docs/BURST_GUARD_DESIGN_2026-08-05.md §10。"
            "NON-STABLE→STABLE 遷移merge直前に物理的期待値フィルタを"
            "effect_gate_window_active 中のみ適用する。"
            "--enable-burst-guard-v2 が無効の間は no-op (警告ログ)。"
            "既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--burst-gate-open-threshold", type=float, default=None,
        dest="burst_gate_open_threshold",
        help=(
            "バーストガード緊急較正 (2026-08-05、factorialバックテスト用)。"
            "Schmitt trigger の開窓閾値を上書きする (CLOSE も同値運用)。"
            "既定 None = BURST_GATE_OPEN_THRESHOLD (0.97)。"
        ),
    )
    parser.add_argument(
        "--enable-hidden-row-burst-guard", action="store_true",
        dest="enable_hidden_row_burst_guard",
        help=(
            "バーストガード Stage1.5b (2026-08-05 アーキ追補、§11) を有効化"
            "する。docs/BURST_GUARD_DESIGN_2026-08-05.md §11。row1-3 凍結"
            "中/close直後クールダウン中の infer_hidden_row 呼び出しをスキップし"
            "row0 (隠し段) の確信度100%%誤色書き込みを防ぐ。"
            "--enable-burst-guard-v2 が無効の間は no-op (警告ログ)。"
            "既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--enable-burst-close-extension", action="store_true",
        dest="enable_burst_close_extension",
        help=(
            "バーストガード §12 close側再設計 (2026-08-05 アーキ確定) を"
            "有効化する。docs/BURST_GUARD_DESIGN_2026-08-05.md §12.2。"
            "生 is_open と実効active信号 (遷移mergeフィルタ+hard freeze の"
            "適用条件) を分離し、close後クールダウン (BURST_GATE_POST_"
            "CLOSE_COOLDOWN_SEC) と相手連鎖継続中の延長を実効側に反映する。"
            "--enable-burst-guard-v2 が無効の間は no-op (警告ログ)。"
            "既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--burst-chain-gap-max", type=float, default=None,
        dest="burst_chain_gap_max_sec",
        help=(
            "バーストガード §12 緊急パラメータ化 (2026-08-05)。相手連鎖延長の"
            "再点火間隔上限を上書きする。既定 None = モジュール定数 3.3。"
            "0.0 を渡すと延長を常に不成立にできる (差分実験で busy局面の"
            "凍結連鎖の犯人と確定した延長機構をA/B測定で切る用途)。"
        ),
    )
    parser.add_argument(
        "--enable-online-hsv-refresh", action="store_true",
        dest="enable_online_hsv_refresh",
        help=(
            "長時間劣化修正 A+B (2026-08-06) を有効化する。"
            "docs/LONGRUN_DEGRADATION_INVESTIGATION_2026-08-06.md §1/§4。"
            "試合毎のOnlineHsvCalibrator較正リセット (A) + inject後の凍結"
            "ガード撤廃 (B) の両方を有効にする。既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--enable-match-transition-debounce", action="store_true",
        dest="enable_match_transition_debounce",
        help=(
            "長時間劣化修正 A' (2026-08-06、§4追補) を有効化する。"
            "docs/LONGRUN_DEGRADATION_INVESTIGATION_2026-08-06.md。"
            "is_active の True/False遷移を対称デバウンスし、"
            "MATCH_TRANSITION_DEBOUNCE_SEC (1.0秒) 未満のフリッカーによる "
            "_match_active_started_frame/_time の誤再アーム/リセットを防ぐ。"
            "既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--enable-ojama-entry-gravity-settle-guard", action="store_true",
        dest="enable_ojama_entry_gravity_settle_guard",
        help=(
            "修正B (2026-08-08、状態機械振動バグB+C の修正) を有効化する。"
            "GRAVITY_SETTLE 中の OJAMA_FALL 新規発火を禁止する。"
            "--enable-gravity-settle-reset-on-exit と対で使う。"
            "既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--enable-gravity-settle-reset-on-exit", action="store_true",
        dest="enable_gravity_settle_reset_on_exit",
        help=(
            "修正C (2026-08-08、状態機械振動バグB+C の修正) を有効化する。"
            "GRAVITY_SETTLE が他 detector に横取りされて弾き出された際、"
            "GravitySettleDetector の内部カウンタをその場でリセットする。"
            "既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--margin-time-rate", action="store_true",
        dest="enable_margin_time_rate",
        help=(
            "マージンタイム逓減 (2026-08-09) を有効化する。おじゃま判定の閾値を"
            "経過時間に応じた実効レートにする (最初の1手から95.5秒で減衰開始)。"
            "従来の固定70では長い試合の後半で着弾の6.27%%を見逃していた。"
            "既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--enable-phantom-board-guard", action="store_true",
        dest="enable_phantom_board_guard",
        help=(
            "幻盤面ガード (2026-08-08) を有効化する。非試合画面 (対戦カード"
            "紹介・ロビー・順位表) で誤認識された満杯おじゃま盤面を snapshot "
            "として記録しない。実測で全123本・0.875%% が該当。"
            "既定は無効 (後方互換)。"
        ),
    )
    parser.add_argument(
        "--stable-majority-window", action="store_true",
        dest="enable_stable_majority_window",
        help=(
            "盤面確定窓 3中2多数決 (2026-08-13 user承認、認識99.5%%物差し"
            "条件付き採用) を有効化する。初回STABLE確定窓を「stable_frame_count "
            "連続厳密一致」から「直近3観測中2一致」に切り替える (1フレーム"
            "ノイズで振り出しに戻る問題・2値交互ノイズへの弱さの対策)。"
            "既定は無効 (後方互換、148動画収集走行中のため既定OFF必須)。"
        ),
    )
    args = parser.parse_args()
    # 既定値解決 (2026-07-30 既定 True 化): 明示 --no-normalize-fps-30 が
    # 最優先で無効化する。それ以外は --normalize-fps-30 の有無に関わらず
    # 新既定 True (collect_lean() 関数側の既定と一致させる)。
    normalize_fps_30 = not args.no_normalize_fps_30
    n = collect_lean(
        args.video, args.out_npz,
        max_sec=args.max_sec,
        start_sec=args.start_sec,
        sample_interval_sec=args.sample_interval,
        capture_next=args.with_next,
        sample_interval_frames=args.sample_interval_frames,
        enable_chain_tracker=args.enable_chain_tracker,
        normalize_fps_30=normalize_fps_30,
        enable_effect_gate=args.enable_effect_gate,
        effect_gate_persist_sec=args.effect_gate_persist_sec,
        enable_effect_visual_gate=args.enable_effect_visual_gate,
        enable_burst_guard_v2=args.enable_burst_guard_v2,
        enable_transition_merge_guard=args.enable_transition_merge_guard,
        burst_gate_open_threshold=args.burst_gate_open_threshold,
        enable_hidden_row_burst_guard=args.enable_hidden_row_burst_guard,
        enable_burst_close_extension=args.enable_burst_close_extension,
        burst_chain_gap_max_sec=args.burst_chain_gap_max_sec,
        enable_online_hsv_refresh=args.enable_online_hsv_refresh,
        enable_match_transition_debounce=args.enable_match_transition_debounce,
        enable_ojama_entry_gravity_settle_guard=(
            args.enable_ojama_entry_gravity_settle_guard
        ),
        enable_gravity_settle_reset_on_exit=args.enable_gravity_settle_reset_on_exit,
        enable_phantom_board_guard=args.enable_phantom_board_guard,
        enable_margin_time_rate=args.enable_margin_time_rate,
        enable_stable_majority_window=args.enable_stable_majority_window,
    )
    print(f"[lean] {args.video.name} -> {args.out_npz} : {n} snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
