"""「ありえない判定」走査器 — D0/D1a/D1b (2026-08-11、タイムラインdump対応)。

## 背景
`scripts/visualize_advantage_overlay.py` の有利不利判定は 45 指標 + 学習済
HistGBC の合成値で、局所的に「見た目には説明のつかない」判定を出すことが
過去のレビューで複数回確認されている (memory
`project_youtube_demo_advantage_issues_2026-08-08` t=29 等)。本スクリプトは
そのうち **閾値不要の論理矛盾 3 種** を全動画横断で機械的に検出する
「走査器」(D0 + D1a + D1b)。

## D0: 主因⇔結論の符号矛盾
`_score_advantage()` が組み立てる主因 (drivers、`ATTRIBUTION_EXCLUDED_
INDICATORS` 除外後の**表示用**上位候補) の 1 位の符号と、総合判定 adv の
符号が逆であれば矛盾 (例: 主因1位「色ぷよ差 +0.67 (1P有利)」なのに
adv は 2P有利)。

## D1a: 確定死の無視
STABLE の confirmed_board が `Board.is_dead()==True` (DEATH_ROW=1 直読み)
の側を、その瞬間に有利判定 (adv 符号がその側 or 勝率>0.5) している場合に
矛盾とする。ただし判定タイミングと勝敗確定の間で正当に is_dead=True が
一瞬出うる (試合境界直後の残存フレーム等) ため、ゲーム境界 ±
`GAME_BOUNDARY_GUARD_SEC` 秒は除外する (物理的ガード、閾値でなく試合境界
という実在イベントからの相対時間)。

## D1b: 致死確定 (pending/room) の無視 (2026-08-11 追加)
D1a が「盤面が既に窒息済み」を対象にするのに対し、D1b は「まだ盤面上は
窒息していないが、これから降る pending お邪魔だけで受け容量を超えることが
確定している」側を対象にする。`kill_override()` が生存側へ 100% 寄せる
基準と同じ `KILL_RATIO_FULL` (=1.5) を再利用し、
`pending / max(KILL_ROOM_FLOOR, room) >= KILL_RATIO_FULL` の側が有利判定
されていれば矛盾とする。D1a と同じゲーム境界ガードを適用する。

## 計算経路と既知の近似 (重要、恒久記録)
`visualize_advantage_overlay.py` の `generate()` は 4 成分ブレンド
(pressure/forecast/model/threat + kill_override + EMA + Platt較正) を
動画のライブ認識ループで組み立てる。本走査器には2つの動作モードがある:

  - **npz 再計算モード** (`scan_video()`, `--npz-dir`): npz (STABLE snapshot
    のみ、盤面グリッド+score+t_sec+game_idx) から `_score_advantage()` の
    生モデル出力 (adv, p1, drivers) だけを再計算する軽量版。npz には
    BoardState 遷移履歴が無く、OjamaAccountingTracker の本物の会計
    (連鎖終了イベント駆動) を再構築できないため、`ojama_net_balance`/
    `ojama_forecast` は常に 0 のダミー会計 (`_dummy_snapshot`) で代用する
    (pending も常に 0)。room はダミーではなく盤面グリッドから実値を計算する
    (`board_room()` は盤面の空きセル数のみに依存し会計非依存のため)。
    148動画のフル再計算で **約39日** かかることが実測され (`scripts/
    _diag_scan_speed_2026-08-11.py` 等)、非現実的と判明した。
  - **dump 読み出しモード** (`scan_video_from_dump()`, `--from-dump`):
    `visualize_advantage_overlay.generate(..., dump_timeline_path=...)` が
    settled 更新のたびに書き出した npz (`TimelineDumpRow`) を読むだけ。
    モデル学習・盤面再計算が一切不要なため大幅に高速。

  `JudgmentRecord.adv`/`.drivers` は両モードとも「`_score_advantage()` の
  生モデル出力 (model_adv)」を表す (dump モードでは `TimelineDumpRow.adv_raw`)
  ため、D0 (「モデル自身が出した diff と adv の符号一致」という自己無矛盾性
  の検査) は両モードで意味論が揃い、比較可能である。一方 `JudgmentRecord.p1`
  は dump モードでは **4成分ブレンド+kill_override+校正+EMA を経た実際の
  表示値** (`TimelineDumpRow.p1`、model_adv とはステージが異なる) であり、
  npz 再計算モードの p1 (model_adv と対の生確率) とは異なる量である。
  D1a/D1b は `adv`(生モデル) と `p1`(dump モードでは表示値) を OR 条件で
  見るため、この非対称性により **D1a/D1b は両モード間で一致するとは限らない**
  (会計がダミーな npz 再計算モードでは pending が常に 0 のため D1b はほぼ
  発火しない、という違いも重なる)。D0 のみが両モード比較に適する
  (`visualize_advantage_overlay.TimelineDumpRow` docstring 併記も参照)。

検出ロジック本体 (`detect_d0`/`detect_d1a`/`detect_d1b`) は npz/model/dump
のいずれにも直接依存しない純関数として分離してあり、`JudgmentRecord` という
共通の中間表現だけを介して両モードから呼べる。

## CPU 使用に関する注記 (2026-08-11 実測に基づく追記)
`_train_model()` の `HistGradientBoostingClassifier.fit()` は既定で論理
コア数ぶんのスレッドを内部生成する (BLAS/OpenMP 経由)。Phase L 全域 regen
(10 並列) 走行中に無制限スレッドで実行したところ、データ整形段階
(`pair_sides_for_win` 等) は 108 秒で終わるにもかかわらず、学習
(`.fit()`) 込みの全体が 2 時間経っても完了せず harness に kill された
(2026-08-11 実測、`scripts/_diag_train_timing_2026-08-11.py` で切り分け
済み)。原因はスレッド過剰発行によるオーバーサブスクリプション
(自分のスレッド数 × 競合プロセス数 >> 論理コア数) と推定される。
本スクリプトは既定で `--max-threads` (既定 2) により
`threadpoolctl.threadpool_limits` でスレッド数を絞る。CLAUDE.md の
プロセス管理ルール「CPUを食いすぎない」を本スクリプト自身の既定挙動として
組み込む (呼出側が明示指定を忘れても安全側に倒す)。

## 使い方
    python -m scripts.scan_judgment_anomalies \\
        --npz-dir data/indicators_v2/boards_lean_phase_l_2026-08-07 \\
        --limit-videos 3
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import threadpoolctl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board import Board  # noqa: E402
from src.ojama_accounting import OjamaAccountSnapshot  # noqa: E402
from scripts.visualize_advantage_overlay import (  # noqa: E402
    JP_LABEL, KILL_RATIO_FULL, KILL_ROOM_FLOOR, _score_advantage, _train_model,
    board_room, load_timeline_dump,
)

# 既定スレッド上限 (2026-08-11 実測に基づく、モジュール docstring 参照)。
# 大きくしすぎると他ジョブとの競合下でオーバーサブスクリプションを再発する。
DEFAULT_MAX_THREADS: int = 2

# ============================
# 定数
# ============================

# ゲーム境界 (試合開始/終了) からこの秒数以内の判定は D1a の対象から除外する
# (user伝授: 「判定タイミングと勝敗確定の間のフレームで正当に is_dead が
# 出る可能性を考慮」。境界前後どちらの残存フレームもカバーするため
# 「境界時刻との絶対差」で判定する)。
GAME_BOUNDARY_GUARD_SEC: float = 2.0

# 検出器の重大度 (D0/D1a/D1b はどれも閾値不要の論理矛盾のため固定)
SEVERITY_CRITICAL: str = "CRITICAL"

# dump モードで trigger_side を特定できない (settled 更新は片側/両側どちらの
# 更新でも起こりうり、dump は「誰の更新か」を記録しないため) ときのプレース
# ホルダ。npz 再計算モードの "1P"/"2P" とは異なる第三の値として明示する。
TRIGGER_SIDE_UNKNOWN: str = "dump"

# デフォルト npz ディレクトリの探索パターン (Phase L 系の全域盤面収集)
_PHASE_L_GLOB: str = "boards_lean_phase_l_*"


# ============================
# レコード / 検出結果 (npz・model 非依存の純データ)
# ============================

@dataclass(frozen=True)
class JudgmentRecord:
    """1 判定断面 (両者の盤面が両方判明している瞬間)。

    npz 再計算モード (`scan_video`) と dump 読み出しモード
    (`scan_video_from_dump`) の両方がこの共通中間表現を生成する設計。
    `detect_d0`/`detect_d1a`/`detect_d1b` はどちらの生成元にも依存しない。

    pending_p1/pending_p2/room1/room2 は D1b 専用 (2026-08-11 追加、
    デフォルト値ありのため既存コンストラクタ呼び出しは変更不要=後方互換)。
    npz 再計算モードでは会計がダミーのため pending は常に 0 (D1b はほぼ
    発火しない)、room は盤面グリッドから実値を計算する
    (モジュール docstring「計算経路と既知の近似」参照)。
    """

    video_id: str
    t_sec: float
    game_idx: int
    trigger_side: str  # このレコードを生んだ側の更新 ("1P" / "2P" / dump時は"dump")
    adv: float  # _score_advantage の生モデル出力 (model_adv、drivers と対)
    p1: float  # 1P 勝率 (0-1)。npz再計算モードは model_adv と対の生確率、
               # dump モードは表示用 EMA 後勝率 (モジュール docstring 参照)
    drivers: tuple[tuple[str, float], ...]  # 表示用主因 (除外後、|差分|降順)
    is_dead_p1: bool
    is_dead_p2: bool
    near_game_boundary: bool
    pending_p1: int = 0  # D1b用: 1P に向かう pending お邪魔 (npz再計算は常に0)
    pending_p2: int = 0  # D1b用: 2P に向かう pending お邪魔 (npz再計算は常に0)
    room1: int = 0       # D1b用: 1P の空き容量 (board_room、両モードとも実値)
    room2: int = 0       # D1b用: 2P の空き容量 (board_room、両モードとも実値)


@dataclass(frozen=True)
class Suspect:
    """走査器が検出した 1 件の疑わしい判定。"""

    video_id: str
    t_sec: float
    game_idx: int
    detector: str
    severity: str
    evidence: str

    def to_tsv_row(self) -> str:
        """TSV 1 行を返す (video_id, t_sec, detector, severity, evidence,
        game_idx の順。game_idx は指令書の必須列に無いがトリアージ用に末尾
        追加、後方互換上の懸念なし=新規ファイルのため)。"""
        return (
            f"{self.video_id}\t{self.t_sec:.3f}\t{self.detector}\t"
            f"{self.severity}\t{self.evidence}\t{self.game_idx}"
        )


# ============================
# 検出ロジック (純関数 — レコードを受け取って判定するだけ)
# ============================

def detect_d0(record: JudgmentRecord) -> Optional[Suspect]:
    """D0: 主因1位の符号と総合判定 adv の符号が逆なら矛盾とする。

    閾値不要の論理矛盾のため severity は常に CRITICAL。
    主因が無い/符号が定義できない (差分または adv がちょうど 0) 場合は
    「矛盾を主張できる根拠が無い」として None を返す。
    """
    if not record.drivers:
        return None
    top_name, top_diff = record.drivers[0]
    if top_diff == 0.0 or record.adv == 0.0:
        return None
    driver_favors_1p = top_diff > 0.0
    adv_favors_1p = record.adv > 0.0
    if driver_favors_1p == adv_favors_1p:
        return None
    jp = JP_LABEL.get(top_name, top_name)
    driver_side = "1P有利" if driver_favors_1p else "2P有利"
    adv_side = "1P有利" if adv_favors_1p else "2P有利"
    evidence = (
        f"主因1位「{jp}差 {top_diff:+.3f}」({driver_side}) なのに "
        f"総合判定 adv={record.adv:+.1f} ({adv_side})"
    )
    return Suspect(
        video_id=record.video_id, t_sec=record.t_sec, game_idx=record.game_idx,
        detector="D0", severity=SEVERITY_CRITICAL, evidence=evidence,
    )


def detect_d1a(record: JudgmentRecord) -> list[Suspect]:
    """D1a: 確定死 (is_dead=True) の側を有利判定していれば矛盾とする。

    ゲーム境界近傍 (`record.near_game_boundary`) は既知の正当例外
    (試合境界直後の残存フレーム) としてガードし、対象外にする。
    """
    if record.near_game_boundary:
        return []
    suspects: list[Suspect] = []
    checks = (
        ("1P", record.is_dead_p1, record.adv > 0.0, record.p1 > 0.5, record.p1),
        ("2P", record.is_dead_p2, record.adv < 0.0, record.p1 < 0.5, 1.0 - record.p1),
    )
    for side, is_dead, adv_favors_side, p1_favors_side, winprob_side in checks:
        if not is_dead:
            continue
        if not (adv_favors_side or p1_favors_side):
            continue
        evidence = (
            f"{side} は窒息確定 (is_dead=True, DEATH_ROW/DEATH_COL 直読み) "
            f"なのに adv={record.adv:+.1f} / 勝率{side}={winprob_side:.1%} で有利判定"
        )
        suspects.append(Suspect(
            video_id=record.video_id, t_sec=record.t_sec, game_idx=record.game_idx,
            detector="D1a", severity=SEVERITY_CRITICAL, evidence=evidence,
        ))
    return suspects


def detect_d1b(record: JudgmentRecord) -> list[Suspect]:
    """D1b: 致死確定 (pending/room 比が KILL_RATIO_FULL 以上) の無視を検出する。

    D1a (盤面が既に is_dead=True) と異なり、こちらは「まだ盤面上は窒息して
    いないが、これから降る pending お邪魔だけで受け容量を超えることが確定
    している」側を対象にする。`kill_override()` が生存側へ完全に寄せる基準
    (`KILL_RATIO_FULL`) と同じ閾値を再利用するため、本来 kill_override が
    是正しているはずの状況を検出する (=kill_override 自体の抜け漏れ発見用)。
    D1a と同じくゲーム境界近傍は既知の正当例外としてガードする。
    """
    if record.near_game_boundary:
        return []
    suspects: list[Suspect] = []
    checks = (
        ("1P", record.pending_p1, record.room1, record.adv > 0.0, record.p1 > 0.5, record.p1),
        ("2P", record.pending_p2, record.room2, record.adv < 0.0, record.p1 < 0.5, 1.0 - record.p1),
    )
    for side, pending, room, adv_favors_side, p1_favors_side, winprob_side in checks:
        ratio = pending / max(KILL_ROOM_FLOOR, room)
        if ratio < KILL_RATIO_FULL:
            continue
        if not (adv_favors_side or p1_favors_side):
            continue
        evidence = (
            f"{side} は致死確定 (pending={pending} / room={room} = 比{ratio:.2f} "
            f"≥ KILL_RATIO_FULL={KILL_RATIO_FULL}) なのに "
            f"adv={record.adv:+.1f} / 勝率{side}={winprob_side:.1%} で有利判定"
        )
        suspects.append(Suspect(
            video_id=record.video_id, t_sec=record.t_sec, game_idx=record.game_idx,
            detector="D1b", severity=SEVERITY_CRITICAL, evidence=evidence,
        ))
    return suspects


# ============================
# ダミー会計 (npz からは再構築不能な OjamaAccountingTracker の代用、
# モジュール docstring 「計算経路と既知の近似」参照)
# ============================

def _dummy_snapshot(t_sec: float) -> OjamaAccountSnapshot:
    """会計ゼロ固定のスナップショットを返す (net/forecast は常に 0)。"""
    return OjamaAccountSnapshot(
        t_sec=t_sec,
        pending_p1=0, pending_p2=0,
        total_generated_by_p1=0, total_generated_by_p2=0,
        total_offset_by_p1=0, total_offset_by_p2=0,
        total_dropped_to_p1=0, total_dropped_to_p2=0,
        net_ojama_balance=0,
        overflow_risk_p1=False, overflow_risk_p2=False,
        confidence=0.0,
        leftover_p1=0, leftover_p2=0,
        all_clear_pending_p1=False, all_clear_pending_p2=False,
        net_balance_capped=0, forecast_p1=0, forecast_p2=0,
    )


ScoreFn = Callable[[Board, Board], tuple[float, float, list[tuple[str, float]]]]


def make_score_fn(model: object) -> ScoreFn:
    """学習済モデルを束縛した `(b1, b2) -> (adv, p1, drivers)` を返す。

    `_score_advantage` は attribution_exclude 省略時に
    `ATTRIBUTION_EXCLUDED_INDICATORS` (本番の表示除外リスト) を既定で使う
    ため、ここでも明示せず本番と同じ挙動に揃える。
    """
    def _fn(b1: Board, b2: Board) -> tuple[float, float, list[tuple[str, float]]]:
        return _score_advantage(model, b1, b2, _dummy_snapshot(0.0))
    return _fn


# ============================
# npz -> JudgmentRecord (per-side-settled 相当のペアリング)
# ============================

def _build_lookup(
    t_local: np.ndarray, g_local: np.ndarray, orig_idx_local: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """game_idx -> (t_sec昇順配列, 元 npz 行インデックス配列) の辞書を作る。"""
    lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for gidx in np.unique(g_local):
        mask = g_local == gidx
        order = np.argsort(t_local[mask], kind="stable")
        lookup[int(gidx)] = (t_local[mask][order], orig_idx_local[mask][order])
    return lookup


def _latest_before(
    lookup: dict[int, tuple[np.ndarray, np.ndarray]], game_idx: int, t_sec: float,
) -> Optional[int]:
    """同じ game_idx 内で t_sec 以下最大の元 npz 行インデックスを返す (無ければ None)。"""
    entry = lookup.get(int(game_idx))
    if entry is None:
        return None
    times, idxs = entry
    pos = int(np.searchsorted(times, t_sec, side="right")) - 1
    if pos < 0:
        return None
    return int(idxs[pos])


def _game_start_times(t_sec: np.ndarray, game_idx: np.ndarray) -> np.ndarray:
    """各 game_idx の最初の t_sec (両 side 通算) をソート済み配列で返す。"""
    starts: dict[int, float] = {}
    for gi, t in zip(game_idx.tolist(), t_sec.tolist()):
        prev = starts.get(int(gi))
        if prev is None or t < prev:
            starts[int(gi)] = float(t)
    return np.array(sorted(starts.values()), dtype=float)


def _is_near_boundary(
    t_sec: float, boundary_times: np.ndarray,
    guard_sec: float = GAME_BOUNDARY_GUARD_SEC,
) -> bool:
    """t_sec がいずれかのゲーム境界から guard_sec 秒以内かを返す。"""
    if boundary_times.size == 0:
        return False
    pos = int(np.searchsorted(boundary_times, t_sec))
    lo = max(0, pos - 1)
    hi = min(boundary_times.size, pos + 1)
    return bool(np.any(np.abs(boundary_times[lo:hi] - t_sec) <= guard_sec))


def scan_video(
    npz_path: Path, score_fn: ScoreFn, max_records: int = 0,
) -> list[JudgmentRecord]:
    """1 動画分の npz を読み、per-side-settled 相当の JudgmentRecord 列を返す。

    「片側でも STABLE なら再計算する」(採用済み `--per-side-settled` と同じ
    思想、`src/production_config.py` 参照) を、npz に含まれる STABLE
    snapshot だけで近似する: 各 side 自身の snapshot 1件ごとに「もう片方の
    直近既知盤面」と組んで 1 レコードを作る。もう片方がまだ一度も STABLE に
    なっていない (同じ game_idx で t_sec 以下の snapshot が無い) 場合は
    スキップする。

    Args:
        max_records: 0 より大きい場合、時刻順で先頭 N 件のトリガーのみ処理する
            (2026-08-11 追加、スモークテスト用)。学習済モデルが
            expected_fire_k1/k2 (モンテカルロ) 等の高コスト候補指標を含む
            場合 `_score_advantage` 1 回が数百ms〜数秒かかりうるため
            (`scripts/_diag_single_call_2026-08-11.py` 実測、モジュール
            docstring 参照)、全動画を待たずに実データで動作確認する目的。
            既定 0 = 全件処理 (後方互換)。
    """
    video_id = npz_path.stem
    d = np.load(npz_path, allow_pickle=True)
    grids = d["grids"]
    if grids.shape[0] == 0:
        return []
    side = d["side"]
    t_sec = d["t_sec"].astype(float)
    game_idx = d["game_idx"].astype(int)

    idx1 = np.nonzero(side == "1P")[0]
    idx2 = np.nonzero(side == "2P")[0]
    if idx1.size == 0 or idx2.size == 0:
        return []

    lookup1 = _build_lookup(t_sec[idx1], game_idx[idx1], idx1)
    lookup2 = _build_lookup(t_sec[idx2], game_idx[idx2], idx2)
    boundary_times = _game_start_times(t_sec, game_idx)

    board_cache: dict[int, Board] = {}

    def _get_board(i: int) -> Board:
        b = board_cache.get(i)
        if b is None:
            b = Board.from_list(grids[i].tolist())
            board_cache[i] = b
        return b

    triggers: list[tuple[int, str, dict[int, tuple[np.ndarray, np.ndarray]]]] = (
        [(int(i), "1P", lookup2) for i in idx1]
        + [(int(i), "2P", lookup1) for i in idx2]
    )
    triggers.sort(key=lambda tup: float(t_sec[tup[0]]))
    if max_records > 0:
        triggers = triggers[:max_records]

    records: list[JudgmentRecord] = []
    for trigger_idx, trigger_side, other_lookup in triggers:
        gidx = int(game_idx[trigger_idx])
        t = float(t_sec[trigger_idx])
        other_idx = _latest_before(other_lookup, gidx, t)
        if other_idx is None:
            continue
        b_trigger = _get_board(trigger_idx)
        b_other = _get_board(other_idx)
        b1 = b_trigger if trigger_side == "1P" else b_other
        b2 = b_other if trigger_side == "1P" else b_trigger
        adv, p1, drivers = score_fn(b1, b2)
        records.append(JudgmentRecord(
            video_id=video_id, t_sec=t, game_idx=gidx, trigger_side=trigger_side,
            adv=adv, p1=p1, drivers=tuple(drivers),
            is_dead_p1=b1.is_dead(), is_dead_p2=b2.is_dead(),
            near_game_boundary=_is_near_boundary(t, boundary_times),
            # D1b用 (2026-08-11 追加)。会計はダミー (pending は常に0=モジュール
            # docstring 参照) だが room は盤面グリッドの実値 (会計非依存)。
            pending_p1=0, pending_p2=0,
            room1=board_room(b1), room2=board_room(b2),
        ))
    records.sort(key=lambda r: r.t_sec)
    return records


# ============================
# dump 読み出しモード (2026-08-11 追加)
# ============================

def scan_video_from_dump(dump_path: Path) -> list[JudgmentRecord]:
    """1 動画分のタイムラインdump npz を JudgmentRecord 列に変換する。

    `visualize_advantage_overlay.generate(..., dump_timeline_path=...)` が
    settled 更新のたびに書き出した値をそのまま読むだけで、モデル学習・
    `_score_advantage` の再計算が一切不要 (npz 再計算モード `scan_video` の
    148動画で約39日という実測に対し、これは読み出しのみで数分オーダー)。
    `adv`/`p1` の意味論の非対称性はモジュール docstring 参照。
    """
    video_id, rows = load_timeline_dump(dump_path)
    if not rows:
        return []
    t_sec_arr = np.array([row.t_sec for row in rows], dtype=float)
    game_idx_arr = np.array([row.game_idx for row in rows], dtype=int)
    boundary_times = _game_start_times(t_sec_arr, game_idx_arr)
    records: list[JudgmentRecord] = []
    for row in rows:
        drivers = (
            ((row.drivers_top1_name, row.drivers_top1_val),)
            if row.drivers_top1_name else ()
        )
        records.append(JudgmentRecord(
            video_id=video_id, t_sec=row.t_sec, game_idx=row.game_idx,
            trigger_side=TRIGGER_SIDE_UNKNOWN,
            adv=row.adv_raw, p1=row.p1, drivers=drivers,
            is_dead_p1=row.is_dead1, is_dead_p2=row.is_dead2,
            near_game_boundary=_is_near_boundary(row.t_sec, boundary_times),
            pending_p1=row.pending_p1, pending_p2=row.pending_p2,
            room1=row.room1, room2=row.room2,
        ))
    return records


# ============================
# CLI
# ============================

def _resolve_default_npz_dir(base: Path) -> Path:
    """`boards_lean_phase_l_*` のうち npz 件数が最大のディレクトリを選ぶ。

    ディレクトリ名の日付が新しくても regen 走行中で npz が数本しか無い
    ことがある (2026-08-11 実例: 08-11 dir は走行開始直後で 1 本のみ)。
    単純な名前ソート降順だと未完走ディレクトリを誤って選ぶため、
    「npz 件数の最大値」を実質的な完成度の代理指標として採用する
    (同数ならディレクトリ名降順=より新しい方を優先)。
    """
    candidates = sorted(base.glob(_PHASE_L_GLOB))
    if not candidates:
        raise FileNotFoundError(
            f"{base}/{_PHASE_L_GLOB} に一致するディレクトリが見つかりません"
        )
    return max(candidates, key=lambda p: (len(list(p.glob("*.npz"))), p.name))


def _write_suspects_tsv(suspects: list[Suspect], out_path: Path) -> None:
    header = "video_id\tt_sec\tdetector\tseverity\tevidence\tgame_idx"
    lines = [header] + [s.to_tsv_row() for s in suspects]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_from_dump(args: argparse.Namespace) -> int:
    """--from-dump モード: dump npz 群を読むだけで D0/D1a/D1b を検出する。

    モデル学習・盤面再計算が一切不要なため、npz 再計算モード (`main()` の
    既定経路、148動画で約39日と実測) に比べ大幅に高速 (モジュール docstring
    参照)。
    """
    out_dir = args.out_dir or Path(f"data/verify/judgment_scan_{date.today().isoformat()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.from_dump_dir.glob("*.npz"))
    if args.limit_videos > 0:
        files = files[: args.limit_videos]
    print(f"[scan] from-dump dir={args.from_dump_dir} ({len(files)} 動画)", flush=True)

    t0 = time.time()
    suspects: list[Suspect] = []
    counts = {"D0": 0, "D1a": 0, "D1b": 0}
    for i, fpath in enumerate(files):
        vt0 = time.time()
        records = scan_video_from_dump(fpath)
        for rec in records:
            s0 = detect_d0(rec)
            if s0 is not None:
                suspects.append(s0)
                counts["D0"] += 1
            for s1 in detect_d1a(rec):
                suspects.append(s1)
                counts["D1a"] += 1
            for s2 in detect_d1b(rec):
                suspects.append(s2)
                counts["D1b"] += 1
        print(
            f"  [{i + 1}/{len(files)}] {fpath.stem}: {len(records)} records, "
            f"{time.time() - vt0:.2f}s, 累計 D0={counts['D0']} D1a={counts['D1a']} "
            f"D1b={counts['D1b']}",
            flush=True,
        )

    out_path = out_dir / "suspects.tsv"
    _write_suspects_tsv(suspects, out_path)
    print(
        f"[scan] 完了(from-dump): 動画={len(files)} suspects={len(suspects)} "
        f"(D0={counts['D0']} D1a={counts['D1a']} D1b={counts['D1b']}) -> {out_path}\n"
        f"[scan] 所要時間 合計={time.time() - t0:.1f}s (モデル学習不要)",
        flush=True,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="「ありえない判定」走査器 (D0+D1a+D1b)")
    ap.add_argument(
        "--npz-dir", type=Path, default=None,
        help="入力 npz ディレクトリ (既定: data/indicators_v2/boards_lean_phase_l_* "
             "のうち最も npz 件数が多いもの)。--from-dump 指定時は無視される。",
    )
    ap.add_argument(
        "--from-dump", type=Path, default=None, dest="from_dump_dir",
        help="npz 再計算の代わりに、visualize_advantage_overlay.py "
             "--dump-timeline が書き出したタイムラインdump群 (ディレクトリ) を"
             "読むだけで検出する (2026-08-11 追加)。モデル学習・盤面再計算が"
             "不要になり大幅高速化する (148動画で約39日→dump読み出しのみへ、"
             "モジュール docstring 参照)。指定時は --npz-dir/--exclude-video-"
             "for-training/--max-threads/--max-records-per-video は無視される。",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="出力ディレクトリ (既定: data/verify/judgment_scan_<今日の日付>)",
    )
    ap.add_argument(
        "--limit-videos", type=int, default=0,
        help="先頭 N 動画だけ処理する (0=全動画、スモークテスト用)",
    )
    ap.add_argument(
        "--exclude-video-for-training", default=None,
        help="_train_model への学習除外動画 ID (リーク防止、省略時は全動画で学習)",
    )
    ap.add_argument(
        "--max-threads", type=int, default=DEFAULT_MAX_THREADS,
        help=(
            "BLAS/OpenMP スレッド数上限 (既定 2)。他ジョブ (Phase L regen 等) との"
            "併走時のオーバーサブスクリプションを防ぐ (モジュール docstring "
            "「CPU 使用に関する注記」参照)。0 以下を指定すると無制限 (旧挙動)。"
        ),
    )
    ap.add_argument(
        "--max-records-per-video", type=int, default=0,
        help=(
            "1 動画あたり処理するトリガー件数の上限 (時刻順で先頭 N 件、"
            "0=全件)。学習済モデルの高コスト候補指標により `_score_advantage` "
            "1 回が数百ms〜数秒かかりうるため、全域走査前の動作確認・所要時間"
            "見積もりに使う (`scan_video` の同名引数 docstring 参照)。"
        ),
    )
    args = ap.parse_args()

    if args.from_dump_dir is not None:
        return _run_from_dump(args)

    npz_dir = args.npz_dir or _resolve_default_npz_dir(Path("data/indicators_v2"))
    out_dir = args.out_dir or Path(f"data/verify/judgment_scan_{date.today().isoformat()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(npz_dir.glob("*.npz"))
    if args.limit_videos > 0:
        files = files[: args.limit_videos]
    print(f"[scan] npz_dir={npz_dir} ({len(files)} 動画)", flush=True)

    thread_limit = args.max_threads if args.max_threads > 0 else None
    with threadpoolctl.threadpool_limits(limits=thread_limit):
        t0 = time.time()
        model = _train_model(args.exclude_video_for_training)
        print(f"[scan] model trained in {time.time() - t0:.1f}s", flush=True)
        score_fn = make_score_fn(model)

        suspects: list[Suspect] = []
        d0_count = 0
        d1a_count = 0
        d1b_count = 0
        per_video_sec: list[float] = []
        for i, fpath in enumerate(files):
            vt0 = time.time()
            records = scan_video(fpath, score_fn, max_records=args.max_records_per_video)
            for rec in records:
                s0 = detect_d0(rec)
                if s0 is not None:
                    suspects.append(s0)
                    d0_count += 1
                for s1 in detect_d1a(rec):
                    suspects.append(s1)
                    d1a_count += 1
                for s1b in detect_d1b(rec):
                    suspects.append(s1b)
                    d1b_count += 1
            dt = time.time() - vt0
            per_video_sec.append(dt)
            print(
                f"  [{i + 1}/{len(files)}] {fpath.stem}: {len(records)} records, "
                f"{dt:.1f}s, 累計 D0={d0_count} D1a={d1a_count} D1b={d1b_count}",
                flush=True,
            )

    out_path = out_dir / "suspects.tsv"
    _write_suspects_tsv(suspects, out_path)
    total_dt = time.time() - t0
    avg = float(np.mean(per_video_sec)) if per_video_sec else 0.0
    print(
        f"[scan] 完了: 動画={len(files)} suspects={len(suspects)} "
        f"(D0={d0_count} D1a={d1a_count} D1b={d1b_count}) -> {out_path}\n"
        f"[scan] 所要時間 合計={total_dt:.1f}s (学習含む) "
        f"動画あたり平均={avg:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
