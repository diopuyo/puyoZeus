"""お邪魔会計 検証ツール — 連鎖検出完全性・相殺タイミングの自動測定。

背景 (2026-07-05):
    予告 (forecast) の per-moment 正解値は自動取得できない (視覚アイコン検出は
    連鎖発光を誤読するため no-go、memory `project_ojama_forecast_quality_2026-06-22`)。
    代わりに **信頼できる score OCR を基準** に、「連鎖の発生と相殺のタイミングが
    正しく捉えられているか」を自動測定する。

    連鎖 = スコア上昇 なので、
        - スコア上昇イベント (基準から >= CHAIN_FIRE_MIN_SCORE 上昇して落ち着く) を
          「実際に起きた連鎖」の近似とみなし、
        - OjamaAccountingTracker の finalize イベント (連鎖終了確定) と照合すれば、
          検出漏れ (例: 先撃ちされた 2P 副連鎖の取りこぼし) を炙り出せる。

設計方針 (コア不変):
    - RecognitionPipeline.load_default (自動 HSV のみ) で動画を処理し、
      collect_indicators_v2 と同じく on_state_transition / on_tsumo_settled で
      OjamaAccountingTracker を駆動する。
    - finalize / offset / discard イベントは **ログ計装** で観測する
      (src.ojama_accounting の logger に非破壊ハンドラを取り付けて構造化ログを解析)。
      コア (ojama_accounting / recognition) には一切手を入れない。
    - スコア上昇イベントは pipeline が返す score 系列から独立に再構成する。

使い方:
    python -m scripts.verify_chain_accounting \
        --video data/frames/video_124_4min.mp4

出力: サイド別 (1P/2P) に
    - 検出完全性率 = finalize 数 / スコア上昇イベント数
    - chain_total 整合 (finalize の chain_total とスコア上昇量の乖離件数)
    - 破棄率 (discard 数 / (finalize + discard))
    - 相殺タイミング (両サイド近接連鎖ペアで canceled>0 だったか、相殺漏れリスト)
    - 内部サニティ (net 収支範囲、両者同時 forecast>0 フレーム数)
    - 取りこぼし / 相殺漏れの具体リスト (時刻・スコア上昇量)
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2

# プロジェクトルートを import path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import (  # noqa: E402
    CHAIN_FIRE_MIN_SCORE,
    K_SETTLE_FRAMES,
    OjamaAccountingTracker,
    SCORE_RESET_THRESHOLD,
)
from src.recognition_pipeline import RecognitionPipeline, SideResult  # noqa: E402

# ============================
# 定数
# ============================

TARGET_W: int = 1920
TARGET_H: int = 1080
DEFAULT_FPS: float = 30.0

# スコア上昇イベント検出: 基準から >= この点数上昇して落ち着いたら 1 連鎖とみなす。
# CHAIN_FIRE_MIN_SCORE (=40) = 会計コアの最小連鎖スコアと同一基準。
SCORE_RISE_MIN: int = CHAIN_FIRE_MIN_SCORE

# スコア上昇が「落ち着いた」と判定する連続不変フレーム数。
# 会計コアの finalize と「1連鎖=1イベント」を構造的に一致させるため K_SETTLE_FRAMES
# と同一値を使う。旧値10は段間休止(~11-12読取frame)より短く1連鎖を段ごとに分割して
# いた(=完全性率が過小に出る不具合)。段間休止 < K_SETTLE_FRAMES で段分割を防ぐ。
RISE_SETTLE_FRAMES: int = K_SETTLE_FRAMES

# chain_total 整合の許容乖離 (点数)。これを超えたら「乖離」件数に計上。
# score OCR 端数誤読・基準スコアの取り方の差を吸収する余裕を持たせる。
CHAIN_TOTAL_TOLERANCE: int = 200

# 相殺ペア検出の近接ウィンドウ (秒)。両サイドの連鎖がこの範囲内なら相殺候補。
CANCEL_PROXIMITY_SEC: float = 3.0

# ============================
# ログ計装 (finalize / offset / discard 捕捉)
# ============================

# src.ojama_accounting のログ書式 (コード側で確定している文字列) を解析する。
#   finalize[p1]: score_start=1234 score_after=5678 chain_total=4444 ...
#   offset[p1]: gen=10 canceled=3 surplus=7 ...
#   chain_end[p1]: score_at_chain_start=None, discarding t=12.34
#   chain_end[p1]: chain_total=-5 <= 0 ...
#   chain_end[p1]: chain_total=12 < min=40 (OCR端数誤読の疑い), discarding ...
# 注: 末尾タイムスタンプは `\bt=` で捉える。ログ内の `start=`/`self.forecast=` 等の
# 語末 "t" に対しては単語境界が立たないため誤マッチせず、末尾の " t=..." のみ拾える。
_RE_FINALIZE = re.compile(
    r"finalize\[(p1|p2)\]: score_start=(-?\d+) score_after=(-?\d+) "
    r"chain_total=(-?\d+) .*?-> gen=(\d+) .*?\bt=([\d.]+)"
)
_RE_OFFSET = re.compile(
    r"offset\[(p1|p2)\]: gen=(\d+) canceled=(\d+) surplus=(\d+) .*?\bt=([\d.]+)"
)
_RE_DISCARD = re.compile(
    r"chain_end\[(p1|p2)\]: .*?(?:discard|<= 0|< min).*?\bt=([\d.]+)"
)


@dataclass
class FinalizeEvent:
    """finalize (連鎖生成計上) イベント 1 件。"""
    side: str
    t_sec: float
    score_start: int
    score_after: int
    chain_total: int
    gen: int
    canceled: int = 0
    surplus: int = 0


@dataclass
class DiscardEvent:
    """finalize 破棄イベント 1 件。"""
    side: str
    t_sec: float
    reason: str


class _AccountingLogCapture(logging.Handler):
    """src.ojama_accounting の logger から finalize/offset/discard を非破壊捕捉する。"""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.finalizes: list[FinalizeEvent] = []
        self.discards: list[DiscardEvent] = []

    def emit(self, record: logging.LogRecord) -> None:
        # ログ文言はコア改修で変わりうるため、パース失敗は行スキップで頑健化する。
        try:
            self._parse(record.getMessage())
        except Exception:
            pass

    def _parse(self, msg: str) -> None:
        m = _RE_FINALIZE.search(msg)
        if m is not None:
            self.finalizes.append(FinalizeEvent(
                side=m.group(1),
                score_start=int(m.group(2)),
                score_after=int(m.group(3)),
                chain_total=int(m.group(4)),
                gen=int(m.group(5)),
                t_sec=float(m.group(6)),
            ))
            return
        m = _RE_OFFSET.search(msg)
        if m is not None:
            # 直近 finalize に相殺情報を付与 (finalize→offset は同一 _finalize_chain_end 内で連続)
            side = m.group(1)
            for fe in reversed(self.finalizes):
                if fe.side == side and fe.canceled == 0 and fe.surplus == 0:
                    fe.canceled = int(m.group(3))
                    fe.surplus = int(m.group(4))
                    break
            return
        m = _RE_DISCARD.search(msg)
        if m is not None:
            self.discards.append(DiscardEvent(
                side=m.group(1),
                t_sec=float(m.group(2)),
                reason=("score_at_start=None" if "score_at_chain_start=None" in msg
                        else ("<=0" if "<= 0" in msg else "<min")),
            ))


# ============================
# スコア上昇イベント抽出 (連鎖の近似)
# ============================

@dataclass
class ScoreRiseEvent:
    """基準から落ち着くまでのスコア上昇 1 件 (= 実連鎖の近似)。"""
    side: str
    t_start_sec: float   # 上昇開始 (基準観測) 時刻
    t_end_sec: float     # 落ち着いた時刻
    base_score: int      # 基準スコア
    peak_score: int      # 落ち着いた到達スコア
    rise: int            # peak - base


@dataclass
class _RiseState:
    """1 サイドのスコア上昇検出用状態機械。"""
    base_score: int | None = None       # 直近の落ち着いたスコア基準
    peak_score: int | None = None        # 上昇中の最大値
    rise_start_sec: float = 0.0
    settle_consec: int = 0               # peak 不変連続フレーム数
    rising: bool = False                 # 上昇イベント進行中か


class ScoreRiseDetector:
    """score 系列から「基準から >= SCORE_RISE_MIN 上昇して落ち着いた」イベントを抽出する。

    掛け算式 (score None) は上昇途中とみなしスキップ (カウント進めない)。
    試合境界 (score 大幅減少) で基準をリセットする。
    """

    def __init__(self) -> None:
        self._st: dict[str, _RiseState] = {"p1": _RiseState(), "p2": _RiseState()}
        self.events: list[ScoreRiseEvent] = []

    def feed(self, side: str, score: int | None, t_sec: float) -> None:
        st = self._st[side]
        if score is None:
            return  # 掛け算式表示中 = 上昇途中、判定保留
        # 試合境界: 大幅減少 → 全リセット
        if st.base_score is not None and st.base_score - score >= SCORE_RESET_THRESHOLD:
            self._flush(side, t_sec)
            st.base_score = score
            st.rising = False
            st.peak_score = None
            st.settle_consec = 0
            return
        if st.base_score is None:
            st.base_score = score
            return
        if score > (st.peak_score if st.peak_score is not None else st.base_score):
            # 上昇中: peak 更新・settle カウントリセット
            st.peak_score = score
            st.settle_consec = 0
            if not st.rising:
                st.rising = True
                st.rise_start_sec = t_sec
        elif st.rising:
            # peak 到達後の不変/微減 → 落ち着き判定
            if score >= (st.peak_score or 0):
                st.settle_consec += 1
            else:
                st.settle_consec += 1  # 微減も落ち着きとみなす (端数誤読吸収)
            if st.settle_consec >= RISE_SETTLE_FRAMES:
                self._commit(side, t_sec)

    def _commit(self, side: str, t_sec: float) -> None:
        st = self._st[side]
        assert st.base_score is not None and st.peak_score is not None
        rise = st.peak_score - st.base_score
        if rise >= SCORE_RISE_MIN:
            self.events.append(ScoreRiseEvent(
                side=side,
                t_start_sec=round(st.rise_start_sec, 2),
                t_end_sec=round(t_sec, 2),
                base_score=st.base_score,
                peak_score=st.peak_score,
                rise=rise,
            ))
        # 新基準 = 到達スコア
        st.base_score = st.peak_score
        st.peak_score = None
        st.rising = False
        st.settle_consec = 0

    def _flush(self, side: str, t_sec: float) -> None:
        """試合境界前に進行中の上昇イベントを確定する。"""
        st = self._st[side]
        if st.rising and st.peak_score is not None and st.base_score is not None:
            if st.peak_score - st.base_score >= SCORE_RISE_MIN:
                self.events.append(ScoreRiseEvent(
                    side=side,
                    t_start_sec=round(st.rise_start_sec, 2),
                    t_end_sec=round(t_sec, 2),
                    base_score=st.base_score,
                    peak_score=st.peak_score,
                    rise=st.peak_score - st.base_score,
                ))

    def finalize(self, t_sec: float) -> None:
        for side in ("p1", "p2"):
            self._flush(side, t_sec)


# ============================
# tsumo delta drain (collect と同一ロジック)
# ============================

@dataclass
class _SideTsumo:
    prev_tsumo: int = 0


def _drain_by_tsumo_delta(
    tracker: OjamaAccountingTracker,
    pipeline: RecognitionPipeline,
    side_tsumo: _SideTsumo,
    ojama_key: str,
    pipeline_key: str,
    t_sec: float,
) -> None:
    """tsumo_count の増分 delta 回 on_tsumo_settled を呼ぶ (collect_indicators_v2 と同一)。"""
    curr_tsumo = pipeline.tsumo_count(pipeline_key)
    delta = curr_tsumo - side_tsumo.prev_tsumo
    if delta > 0:
        for _ in range(delta):
            tracker.on_tsumo_settled(ojama_key, t_sec)
    side_tsumo.prev_tsumo = curr_tsumo


# ============================
# サニティ収集
# ============================

@dataclass
class _Sanity:
    """内部サニティ集計。"""
    net_min: int = 10**9
    net_max: int = -10**9
    both_forecast_frames: int = 0     # 両者同時 forecast>0 のフレーム数
    total_frames: int = 0
    forecast_p1_max: int = 0
    forecast_p2_max: int = 0


# ============================
# メイン処理
# ============================

@dataclass
class VerifyResult:
    """1 動画の検証結果。"""
    video_id: str
    rises: list[ScoreRiseEvent] = field(default_factory=list)
    finalizes: list[FinalizeEvent] = field(default_factory=list)
    discards: list[DiscardEvent] = field(default_factory=list)
    sanity: _Sanity = field(default_factory=_Sanity)


def run_verify(video_path: Path, max_sec: float = 0.0) -> VerifyResult:
    """1 動画を処理し、スコア上昇イベント・finalize・discard・サニティを収集する。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_sec > 0:
        n_frames = min(n_frames, int(max_sec * fps))
    video_id = video_path.stem

    # ログ計装: src.ojama_accounting logger に非破壊ハンドラを取り付ける
    acc_logger = logging.getLogger("src.ojama_accounting")
    prev_level = acc_logger.level
    prev_propagate = acc_logger.propagate
    acc_logger.setLevel(logging.INFO)  # finalize/offset/discard は INFO で出る
    acc_logger.propagate = False        # 標準出力へのノイズ抑制
    capture = _AccountingLogCapture()
    acc_logger.addHandler(capture)

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )
    _vid_match = re.search(r"(v\d+|video_\d+)", video_path.name)
    if _vid_match and hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(_vid_match.group(1))

    tracker = OjamaAccountingTracker()
    tracker.reset()
    rise_det = ScoreRiseDetector()
    sanity = _Sanity()
    prev_state_p1 = BoardState.MENU
    prev_state_p2 = BoardState.MENU
    ts_p1 = _SideTsumo()
    ts_p2 = _SideTsumo()

    try:
        for fi in range(n_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[:2] != (TARGET_H, TARGET_W):
                frame = cv2.resize(
                    frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
                )
            t_sec = fi / fps
            result = pipeline.update(fi, t_sec, frame)
            _drive(tracker, result.p1, result.p2, prev_state_p1, prev_state_p2,
                   t_sec, pipeline, ts_p1, ts_p2)
            prev_state_p1 = result.p1.state
            prev_state_p2 = result.p2.state
            # スコア上昇イベント
            rise_det.feed("p1", result.p1.score, t_sec)
            rise_det.feed("p2", result.p2.score, t_sec)
            # サニティ収集
            snap = tracker.get_snapshot(t_sec)
            _collect_sanity(sanity, snap)
        rise_det.finalize(t_sec)
    finally:
        cap.release()
        acc_logger.removeHandler(capture)
        acc_logger.setLevel(prev_level)
        acc_logger.propagate = prev_propagate

    return VerifyResult(
        video_id=video_id,
        rises=rise_det.events,
        finalizes=capture.finalizes,
        discards=capture.discards,
        sanity=sanity,
    )


def _drive(
    tracker: OjamaAccountingTracker,
    p1: SideResult,
    p2: SideResult,
    prev_p1: BoardState,
    prev_p2: BoardState,
    t_sec: float,
    pipeline: RecognitionPipeline,
    ts_p1: _SideTsumo,
    ts_p2: _SideTsumo,
) -> None:
    """collect_indicators_v2._drive_ojama と同一の駆動 (on_state_transition + tsumo drain)。"""
    tracker.on_state_transition("p1", prev_p1, p1.state, p1.score, t_sec)
    tracker.on_state_transition("p2", prev_p2, p2.state, p2.score, t_sec)
    _drain_by_tsumo_delta(tracker, pipeline, ts_p1, "p1", "1P", t_sec)
    _drain_by_tsumo_delta(tracker, pipeline, ts_p2, "p2", "2P", t_sec)


def _collect_sanity(sanity: _Sanity, snap: object) -> None:
    """スナップショットからサニティ指標を更新する。"""
    net = snap.net_ojama_balance  # type: ignore[attr-defined]
    f1 = snap.forecast_p1  # type: ignore[attr-defined]
    f2 = snap.forecast_p2  # type: ignore[attr-defined]
    sanity.total_frames += 1
    sanity.net_min = min(sanity.net_min, net)
    sanity.net_max = max(sanity.net_max, net)
    sanity.forecast_p1_max = max(sanity.forecast_p1_max, f1)
    sanity.forecast_p2_max = max(sanity.forecast_p2_max, f2)
    if f1 > 0 and f2 > 0:
        sanity.both_forecast_frames += 1


# ============================
# メトリクス算出 + マッチング
# ============================

@dataclass
class SideMetrics:
    """1 サイドの検出完全性メトリクス。"""
    side: str
    n_rises: int = 0
    n_finalizes: int = 0
    n_discards: int = 0
    completeness: float = 0.0            # finalize / rise
    n_missed: int = 0                    # 対応 finalize なしの rise 数
    missed: list[ScoreRiseEvent] = field(default_factory=list)
    n_total_mismatch: int = 0            # chain_total 乖離件数
    total_mismatches: list[tuple] = field(default_factory=list)
    discard_rate: float = 0.0


def _match_side(
    side: str,
    rises: list[ScoreRiseEvent],
    finalizes: list[FinalizeEvent],
    discards: list[DiscardEvent],
) -> SideMetrics:
    """スコア上昇イベントと finalize を時刻近接でマッチングし、取りこぼしを列挙する。"""
    s_rises = [r for r in rises if r.side == side]
    s_fin = [f for f in finalizes if f.side == side]
    s_disc = [d for d in discards if d.side == side]
    m = SideMetrics(side=side, n_rises=len(s_rises), n_finalizes=len(s_fin),
                    n_discards=len(s_disc))
    m.completeness = (len(s_fin) / len(s_rises)) if s_rises else 0.0
    n_fin_disc = len(s_fin) + len(s_disc)
    m.discard_rate = (len(s_disc) / n_fin_disc) if n_fin_disc else 0.0

    # 各 rise に最も近い finalize を貪欲マッチ (時刻±CANCEL_PROXIMITY_SEC*2 以内)
    used: set[int] = set()
    match_window = CANCEL_PROXIMITY_SEC * 2.0
    for r in s_rises:
        best_j = -1
        best_dt = match_window
        for j, f in enumerate(s_fin):
            if j in used:
                continue
            dt = abs(f.t_sec - r.t_end_sec)
            if dt <= best_dt:
                best_dt = dt
                best_j = j
        if best_j < 0:
            m.missed.append(r)
        else:
            used.add(best_j)
            f = s_fin[best_j]
            if abs(f.chain_total - r.rise) > CHAIN_TOTAL_TOLERANCE:
                m.total_mismatches.append(
                    (round(r.t_end_sec, 2), r.rise, f.chain_total,
                     abs(f.chain_total - r.rise))
                )
    m.n_missed = len(m.missed)
    m.n_total_mismatch = len(m.total_mismatches)
    return m


@dataclass
class CancelIssue:
    """相殺漏れ (近接連鎖ペアなのに canceled=0) 1 件。"""
    t_p1: float
    t_p2: float
    p1_rise: int
    p2_rise: int
    note: str


def _analyze_cancel(res: VerifyResult) -> list[CancelIssue]:
    """両サイドの連鎖 (finalize) が近接時刻に起きたペアで相殺が発火したか検査する。

    近接ペアで両サイドとも canceled=0 の場合、相殺すべきなのに漏れた候補として列挙する。
    (厳密には撃ち合いの向き・タイミング差により片方相殺で正しい場合もあるため、
     「両方 canceled=0」を保守的に相殺漏れ候補とする。)
    """
    fin_p1 = [f for f in res.finalizes if f.side == "p1"]
    fin_p2 = [f for f in res.finalizes if f.side == "p2"]
    issues: list[CancelIssue] = []
    for a in fin_p1:
        for b in fin_p2:
            if abs(a.t_sec - b.t_sec) <= CANCEL_PROXIMITY_SEC:
                if a.canceled == 0 and b.canceled == 0:
                    issues.append(CancelIssue(
                        t_p1=round(a.t_sec, 2), t_p2=round(b.t_sec, 2),
                        p1_rise=a.chain_total, p2_rise=b.chain_total,
                        note="近接連鎖ペアだが両者 canceled=0",
                    ))
    return issues


# ============================
# レポート出力
# ============================

def _print_report(res: VerifyResult) -> None:
    print("=" * 72)
    print(f"お邪魔会計 検証レポート: {res.video_id}")
    print("=" * 72)
    for side, label in (("p1", "1P"), ("p2", "2P")):
        m = _match_side(side, res.rises, res.finalizes, res.discards)
        print(f"\n【{label}】")
        print(f"  スコア上昇イベント数 (実連鎖近似): {m.n_rises}")
        print(f"  finalize 数              : {m.n_finalizes}")
        print(f"  discard 数               : {m.n_discards}")
        print(f"  検出完全性率 (finalize/rise): {m.completeness:.2%}")
        print(f"  chain_total 乖離件数 (>{CHAIN_TOTAL_TOLERANCE}点): {m.n_total_mismatch}")
        print(f"  破棄率 (discard/(fin+disc)): {m.discard_rate:.2%}")
        if m.missed:
            print(f"  -- 取りこぼし (finalize なしの上昇) {len(m.missed)}件 --")
            print(f"     {'t_end':>8} {'base':>8} {'peak':>8} {'rise':>7}")
            for r in m.missed:
                print(f"     {r.t_end_sec:8.2f} {r.base_score:8d} "
                      f"{r.peak_score:8d} {r.rise:7d}")
        if m.total_mismatches:
            print(f"  -- chain_total 乖離 {len(m.total_mismatches)}件 --")
            print(f"     {'t':>8} {'rise':>7} {'chain_total':>12} {'diff':>7}")
            for (t, rise, ct, diff) in m.total_mismatches:
                print(f"     {t:8.2f} {rise:7d} {ct:12d} {diff:7d}")

    # 相殺タイミング
    issues = _analyze_cancel(res)
    print(f"\n【相殺タイミング】(近接ウィンドウ ±{CANCEL_PROXIMITY_SEC}s)")
    print(f"  相殺漏れ候補件数: {len(issues)}")
    if issues:
        print(f"     {'t_p1':>8} {'t_p2':>8} {'p1_ct':>8} {'p2_ct':>8}  note")
        for it in issues:
            print(f"     {it.t_p1:8.2f} {it.t_p2:8.2f} {it.p1_rise:8d} "
                  f"{it.p2_rise:8d}  {it.note}")

    # 内部サニティ
    s = res.sanity
    print("\n【内部サニティ】")
    print(f"  net収支 範囲: [{s.net_min}, {s.net_max}]")
    print(f"  forecast最大: 1P={s.forecast_p1_max}, 2P={s.forecast_p2_max}")
    print(f"  両者同時 forecast>0 フレーム数: {s.both_forecast_frames} "
          f"/ {s.total_frames} ({(s.both_forecast_frames/max(1,s.total_frames)):.2%})")
    print("  (相殺モデルでは両者同時 forecast>0 は 0 が期待)")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="お邪魔会計 連鎖検出完全性・相殺タイミング検証ツール",
    )
    parser.add_argument("--video", type=Path, required=True, help="入力動画")
    parser.add_argument(
        "--max-sec", type=float, default=0.0, help="処理する最大秒数 (0 = 全長)",
    )
    args = parser.parse_args()
    res = run_verify(args.video, args.max_sec)
    _print_report(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
