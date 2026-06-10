"""
時系列解析モジュール (Phase 3 拡張)

動画全体に対して試合区間ごとの有利不利スコアを時系列で算出する統合エンドポイント。

主要機能:
    - 試合区間の自動検出 or boundaries TSV による外部指定
    - 各試合内で 0.6 秒ごとの有利不利スコア (ScorePoint)
    - 各試合内で 0.2 秒ごとの連鎖イベント (ChainEvent) 追跡
    - WIN パネルから開始/終了の勝数を取得 (失敗時は None)
    - JSON シリアライズで実勝敗との合致率測定に供する

設計方針:
    - Analyzer など既存の単一フレーム解析器は変更しない
    - 動画 I/O は cv2.VideoCapture(POS_MSEC) でランダムアクセス
    - 試合区間が短い・解像度違いはスキップして続行する
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.board import Board
from src.chain import ChainSimulator
from src.chain_detector import ChainEvent, VideoChainTracker
from src.image_reader import ImageReader
from src.old.indicators import (
    ALL_INDICATOR_NAMES,
    INDICATOR_NEXT_ACCEPTANCE,
    IndicatorCalculator,
    IndicatorSet,
)
from src.match_state import MatchState, MatchStateDetector
from src.sampling_config import (
    BOARD_INTERVAL_SEC,
    EVAL_INTERVAL_SEC,
)
from src.old.scorer import PhaseAwareScorer, Scorer

# scorer_mode 識別子
SCORER_MODE_DEFAULT: str = "default"
SCORER_MODE_PHASE_AWARE: str = "phase_aware"

# ============================
# 定数定義
# ============================

# 試合区間の最低長 (これ未満は除外)
MIN_MATCH_DURATION_SEC: float = 5.0

# MatchStateDetector ベースで自動検出する際の最低長
MIN_AUTO_MATCH_DURATION_SEC: float = 20.0

# 自動検出時のサンプリング間隔
AUTO_DETECT_SAMPLE_SEC: float = 1.0

# 自動検出時に in_match 連続区間を作る際、許容する欠損長
AUTO_DETECT_GAP_TOLERANCE_SEC: float = 2.0

# 動画が想定する解像度
EXPECTED_FRAME_HEIGHT: int = 1080
EXPECTED_FRAME_WIDTH: int = 1920

# matches.tsv の列名
TSV_COL_IDX: str = "idx"
TSV_COL_START: str = "start_sec"
TSV_COL_END: str = "end_sec"
TSV_COL_DURATION: str = "duration_sec"


# ============================
# データクラス
# ============================


@dataclass(frozen=True)
class ScorePoint:
    """
    1 評価点 (0.6 秒ごと) の有利不利スコア。

    Attributes:
        t_sec: 試合開始からの相対時刻 (秒)。
        score: 有利不利スコア -100〜+100 (正=1P 有利)。
        breakdown: 指標名→1P-2P 寄与差分。
        p1_advantage_score: 1P 側の指標寄与合計。
        p2_advantage_score: 2P 側の指標寄与合計。
        phase: phase_aware モード時に記録する "start"/"mid"/"end" (任意)。
    """
    t_sec: float
    score: float
    breakdown: dict[str, float]
    p1_advantage_score: float
    p2_advantage_score: float
    phase: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON 化用辞書に変換する。"""
        out: dict[str, Any] = {
            "t_sec": self.t_sec,
            "score": self.score,
            "breakdown": dict(self.breakdown),
            "p1_advantage_score": self.p1_advantage_score,
            "p2_advantage_score": self.p2_advantage_score,
        }
        if self.phase is not None:
            out["phase"] = self.phase
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScorePoint":
        """辞書から復元する。"""
        phase_val = data.get("phase")
        return cls(
            t_sec=float(data["t_sec"]),
            score=float(data["score"]),
            breakdown={str(k): float(v) for k, v in data.get("breakdown", {}).items()},
            p1_advantage_score=float(data["p1_advantage_score"]),
            p2_advantage_score=float(data["p2_advantage_score"]),
            phase=str(phase_val) if phase_val is not None else None,
        )


@dataclass(frozen=True)
class ChainEventSummary:
    """
    1 連鎖イベントの JSON 用要約 (Board は除外)。

    Attributes:
        side: "1P" or "2P"。
        trigger_sec: 試合開始から見た発火時刻 (秒)。
        end_sec: 連鎖完了時刻 (秒)。
        chain_count: 連鎖数。
        total_erased: 消去 puyo 数。
        total_score: 連鎖合計得点。
        ojama_sent: 送出おじゃま数。
        is_all_clear: 全消し連鎖か。
    """
    side: str
    trigger_sec: float
    end_sec: float
    chain_count: int
    total_erased: int
    total_score: int
    ojama_sent: int
    is_all_clear: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "trigger_sec": self.trigger_sec,
            "end_sec": self.end_sec,
            "chain_count": self.chain_count,
            "total_erased": self.total_erased,
            "total_score": self.total_score,
            "ojama_sent": self.ojama_sent,
            "is_all_clear": self.is_all_clear,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChainEventSummary":
        return cls(
            side=str(data["side"]),
            trigger_sec=float(data["trigger_sec"]),
            end_sec=float(data["end_sec"]),
            chain_count=int(data["chain_count"]),
            total_erased=int(data["total_erased"]),
            total_score=int(data["total_score"]),
            ojama_sent=int(data["ojama_sent"]),
            is_all_clear=bool(data["is_all_clear"]),
        )

    @classmethod
    def from_event(cls, side: str, event: ChainEvent) -> "ChainEventSummary":
        """ChainEvent から要約を生成する。"""
        return cls(
            side=side,
            trigger_sec=event.trigger_sec,
            end_sec=event.end_sec,
            chain_count=event.chain_count,
            total_erased=event.total_erased,
            total_score=event.total_score,
            ojama_sent=event.ojama_sent,
            is_all_clear=event.is_all_clear,
        )


@dataclass(frozen=True)
class MatchSegment:
    """
    1 試合分のタイムライン情報。

    Attributes:
        match_idx: 試合通番 (1 始まり)。
        start_sec: 動画上の試合開始秒。
        end_sec: 動画上の試合終了秒。
        duration_sec: 試合長 (= end - start)。
        score_timeline: 0.6s ごとの ScorePoint。
        chain_events: 0.2s 解像度で検出した両側の連鎖イベント。
        win_panel_at_start: (1P 勝数, 2P 勝数) - OCR 困難なら None。
        win_panel_at_end: 同上。
        winner: "1P" / "2P" / None (不明)。
    """
    match_idx: int
    start_sec: float
    end_sec: float
    duration_sec: float
    score_timeline: list[ScorePoint]
    chain_events: list[ChainEventSummary]
    win_panel_at_start: tuple[int, int] | None = None
    win_panel_at_end: tuple[int, int] | None = None
    winner: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON 化用辞書に変換する。"""
        return {
            "match_idx": self.match_idx,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration_sec": self.duration_sec,
            "score_timeline": [p.to_dict() for p in self.score_timeline],
            "chain_events": [e.to_dict() for e in self.chain_events],
            "win_panel_at_start": list(self.win_panel_at_start)
            if self.win_panel_at_start is not None else None,
            "win_panel_at_end": list(self.win_panel_at_end)
            if self.win_panel_at_end is not None else None,
            "winner": self.winner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchSegment":
        """辞書から復元する。"""
        wps = data.get("win_panel_at_start")
        wpe = data.get("win_panel_at_end")
        return cls(
            match_idx=int(data["match_idx"]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            duration_sec=float(data["duration_sec"]),
            score_timeline=[
                ScorePoint.from_dict(p) for p in data.get("score_timeline", [])
            ],
            chain_events=[
                ChainEventSummary.from_dict(e) for e in data.get("chain_events", [])
            ],
            win_panel_at_start=(int(wps[0]), int(wps[1])) if wps else None,
            win_panel_at_end=(int(wpe[0]), int(wpe[1])) if wpe else None,
            winner=data.get("winner"),
        )


@dataclass(frozen=True)
class TimelineResult:
    """
    動画 1 本に対するタイムライン解析全体結果。

    Attributes:
        video_path: 入力動画ファイルパス。
        duration_sec: 動画長 (秒)。
        fps: フレームレート。
        match_segments: 試合区間ごとの解析結果。
        meta: cnn_path, calib_path 等のメタ情報。
    """
    video_path: str
    duration_sec: float
    fps: float
    match_segments: list[MatchSegment]
    meta: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON 化用辞書に変換する。"""
        return {
            "video_path": self.video_path,
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "match_segments": [m.to_dict() for m in self.match_segments],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimelineResult":
        """辞書から復元する。"""
        return cls(
            video_path=str(data.get("video_path", "")),
            duration_sec=float(data.get("duration_sec", 0.0)),
            fps=float(data.get("fps", 0.0)),
            match_segments=[
                MatchSegment.from_dict(m) for m in data.get("match_segments", [])
            ],
            meta={str(k): str(v) for k, v in data.get("meta", {}).items()},
        )


# ============================
# JSON シリアライザ
# ============================


def to_json(result: TimelineResult, path: Path) -> None:
    """TimelineResult を JSON ファイルに保存する。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    p.write_text(text, encoding="utf-8")


def from_json(path: Path) -> TimelineResult:
    """JSON ファイルから TimelineResult を復元する。"""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return TimelineResult.from_dict(data)


# ============================
# 補助関数: 試合区間特定
# ============================


def parse_boundaries_tsv(tsv_path: Path) -> list[tuple[int, float, float]]:
    """
    matches.tsv を (idx, start_sec, end_sec) タプル列に変換する。

    Args:
        tsv_path: タブ区切りの matches.tsv パス。

    Returns:
        list of (idx, start, end)。ヘッダ行はスキップする。
    """
    p = Path(tsv_path)
    lines = p.read_text(encoding="utf-8").splitlines()
    out: list[tuple[int, float, float]] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        if cols[0] == TSV_COL_IDX:
            continue  # ヘッダ
        try:
            idx = int(float(cols[0]))
            start = float(cols[1])
            end = float(cols[2])
        except ValueError:
            continue
        out.append((idx, start, end))
    return out


def detect_match_boundaries_auto(
    video_path: Path,
    detector: MatchStateDetector,
    sample_interval_sec: float = AUTO_DETECT_SAMPLE_SEC,
    min_duration_sec: float = MIN_AUTO_MATCH_DURATION_SEC,
    gap_tolerance_sec: float = AUTO_DETECT_GAP_TOLERANCE_SEC,
) -> list[tuple[int, float, float]]:
    """
    MatchStateDetector で in-match 連続区間を抽出する。

    Args:
        video_path: 入力動画パス。
        detector: 試合中判定器。
        sample_interval_sec: サンプリング間隔。
        min_duration_sec: 採用最低長。
        gap_tolerance_sec: in-match 連続判定で許容するギャップ。

    Returns:
        (idx, start_sec, end_sec) タプル列。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0

    samples: list[tuple[float, bool]] = []
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            in_match = _safe_in_match(detector, frame)
            samples.append((t, in_match))
        t += sample_interval_sec
    cap.release()

    return _samples_to_intervals(samples, min_duration_sec, gap_tolerance_sec)


def _safe_in_match(detector: MatchStateDetector, frame: np.ndarray) -> bool:
    """検出器を呼びつつ例外を吸収して bool を返す。"""
    try:
        return detector.detect(frame).state == MatchState.IN_MATCH
    except Exception:
        return False


def _samples_to_intervals(
    samples: list[tuple[float, bool]],
    min_duration_sec: float,
    gap_tolerance_sec: float,
) -> list[tuple[int, float, float]]:
    """in-match=True が連続する区間を [start,end] に集約する。"""
    intervals: list[tuple[float, float]] = []
    cur_start: float | None = None
    last_in: float | None = None
    for t, ok in samples:
        if ok:
            if cur_start is None:
                cur_start = t
            last_in = t
        else:
            if cur_start is not None and last_in is not None:
                if t - last_in > gap_tolerance_sec:
                    intervals.append((cur_start, last_in))
                    cur_start = None
                    last_in = None
    if cur_start is not None and last_in is not None:
        intervals.append((cur_start, last_in))

    out: list[tuple[int, float, float]] = []
    idx = 1
    for s, e in intervals:
        if e - s >= min_duration_sec:
            out.append((idx, s, e))
            idx += 1
    return out


# ============================
# TimelineAnalyzer
# ============================


class TimelineAnalyzer:
    """
    動画を時系列解析する統合エンドポイント。

    Usage:
        ana = TimelineAnalyzer()
        res = ana.analyze_video(Path("video.mp4"),
                                boundaries_tsv=Path("matches.tsv"))
        to_json(res, Path("timeline.json"))
    """

    def __init__(
        self,
        cnn_path: Path | None = None,
        calib_path: Path | None = None,
        scorer_mode: str = SCORER_MODE_DEFAULT,
        phase_aware_interpolate: bool = True,
    ) -> None:
        """
        Args:
            cnn_path: CNN 分類器の重みパス (NextDetector 用)。
                未指定なら NextDetector は使わずネクスト無しで指標計算する。
            calib_path: ImageReader / MatchStateDetector のキャリブレーション JSON。
                未指定ならデフォルト座標を利用する。
            scorer_mode: "default" (既存 Scorer + DEFAULT_WEIGHTS) または
                "phase_aware" (PhaseAwareScorer)。デフォルトは "default" で、
                既存呼び出しの動作を変えない。
            phase_aware_interpolate: scorer_mode="phase_aware" 時に
                PhaseAwareScorer の interpolate 引数として渡される。
        """
        if scorer_mode not in (SCORER_MODE_DEFAULT, SCORER_MODE_PHASE_AWARE):
            raise ValueError(
                f"未知の scorer_mode: {scorer_mode}. "
                f"利用可能: {SCORER_MODE_DEFAULT}, {SCORER_MODE_PHASE_AWARE}",
            )
        self._cnn_path = Path(cnn_path) if cnn_path is not None else None
        self._calib_path = Path(calib_path) if calib_path is not None else None
        self._scorer_mode = scorer_mode
        self._reader = self._build_reader()
        self._calculator = IndicatorCalculator()
        self._scorer = Scorer()
        self._phase_aware_scorer: PhaseAwareScorer | None = (
            PhaseAwareScorer(interpolate=phase_aware_interpolate)
            if scorer_mode == SCORER_MODE_PHASE_AWARE else None
        )
        self._next_detector = self._try_load_next_detector()
        self._match_detector = self._try_load_match_detector()

    # ============================
    # 公開メソッド
    # ============================

    def analyze_video(
        self,
        video_path: Path,
        boundaries_tsv: Path | None = None,
    ) -> TimelineResult:
        """
        動画 1 本を時系列解析する。

        Args:
            video_path: 入力動画パス。
            boundaries_tsv: 試合区間 TSV。未指定なら自動検出を試みる。

        Returns:
            TimelineResult: 全試合分のタイムライン。
        """
        video_path = Path(video_path)
        info = self._get_video_info(video_path)
        boundaries = self._resolve_boundaries(video_path, boundaries_tsv)

        segments: list[MatchSegment] = []
        for idx, start, end in boundaries:
            if end - start < MIN_MATCH_DURATION_SEC:
                continue
            seg = self._analyze_segment(video_path, idx, start, end)
            if seg is not None:
                segments.append(seg)

        return TimelineResult(
            video_path=str(video_path),
            duration_sec=info["duration_sec"],
            fps=info["fps"],
            match_segments=segments,
            meta=self._build_meta(),
        )

    # ============================
    # 内部: 区間決定
    # ============================

    def _resolve_boundaries(
        self,
        video_path: Path,
        boundaries_tsv: Path | None,
    ) -> list[tuple[int, float, float]]:
        """boundaries TSV があれば優先、無ければ自動検出。"""
        if boundaries_tsv is not None:
            tsv_path = Path(boundaries_tsv)
            if tsv_path.exists():
                return parse_boundaries_tsv(tsv_path)
        if self._match_detector is None:
            return []
        return detect_match_boundaries_auto(
            video_path, self._match_detector,
        )

    # ============================
    # 内部: 1 試合の解析
    # ============================

    def _analyze_segment(
        self,
        video_path: Path,
        idx: int,
        start: float,
        end: float,
    ) -> MatchSegment | None:
        """1 試合区間を解析して MatchSegment を返す。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        try:
            score_timeline = self._collect_score_points(cap, start, end)
            chain_events = self._collect_chain_events(cap, start, end)
        finally:
            cap.release()

        return MatchSegment(
            match_idx=idx,
            start_sec=start,
            end_sec=end,
            duration_sec=end - start,
            score_timeline=score_timeline,
            chain_events=chain_events,
            win_panel_at_start=None,
            win_panel_at_end=None,
            winner=None,
        )

    # ============================
    # 内部: 評価点収集 (0.6s)
    # ============================

    def _collect_score_points(
        self,
        cap: cv2.VideoCapture,
        start: float,
        end: float,
    ) -> list[ScorePoint]:
        """0.6s ごとに ScorePoint を計算して返す。"""
        match_duration = max(0.0, end - start)
        points: list[ScorePoint] = []
        t = 0.0
        while start + t < end:
            abs_t = start + t
            frame = self._read_frame_at(cap, abs_t)
            if frame is None:
                t += EVAL_INTERVAL_SEC
                continue
            point = self._evaluate_frame(frame, t, match_duration)
            if point is not None:
                points.append(point)
            t += EVAL_INTERVAL_SEC
        return points

    def _evaluate_frame(
        self,
        frame: np.ndarray,
        t_sec: float,
        match_duration_sec: float = 0.0,
    ) -> ScorePoint | None:
        """1 フレームから ScorePoint を生成する (失敗時 None)。

        scorer_mode="phase_aware" の場合は ``t_sec`` と
        ``match_duration_sec`` から phase を判定して PhaseAwareScorer で
        スコアリングし、ScorePoint.phase に phase 名を記録する。
        """
        try:
            board_1p, board_2p = self._reader.read_both_boards(frame)
        except Exception:
            return None

        next_pairs = self._detect_next_pairs(frame)
        try:
            # 2026-04-27: opponent_board を相互に渡す (Phase J 凝視指標が機能)
            ind1 = self._calculator.compute_all(
                board_1p,
                next_pair=next_pairs.get("p1_next"),
                dnext_pair=next_pairs.get("p1_dnext"),
                opponent_board=board_2p,
            )
            ind2 = self._calculator.compute_all(
                board_2p,
                next_pair=next_pairs.get("p2_next"),
                dnext_pair=next_pairs.get("p2_dnext"),
                opponent_board=board_1p,
            )
            phase: str | None = None
            if self._phase_aware_scorer is not None:
                score = self._phase_aware_scorer.score(
                    ind1, ind2,
                    elapsed_sec=t_sec,
                    match_duration_sec=match_duration_sec,
                )
                phase = self._phase_aware_scorer.current_phase(
                    t_sec, match_duration_sec,
                )
            else:
                score = self._scorer.score(ind1, ind2)
        except Exception:
            return None

        return _build_score_point(t_sec, score, ind1, ind2, phase=phase)

    def _detect_next_pairs(
        self,
        frame: np.ndarray,
    ) -> dict[str, tuple[int, int]]:
        """利用可能なら NextDetector でネクスト検出。失敗は空辞書。"""
        if self._next_detector is None:
            return {}
        try:
            both = self._next_detector.detect_both(frame)
        except Exception:
            return {}
        return {
            "p1_next": both.p1.next_pair,
            "p1_dnext": both.p1.dnext_pair,
            "p2_next": both.p2.next_pair,
            "p2_dnext": both.p2.dnext_pair,
        }

    # ============================
    # 内部: 連鎖イベント収集 (0.2s)
    # ============================

    def _collect_chain_events(
        self,
        cap: cv2.VideoCapture,
        start: float,
        end: float,
    ) -> list[ChainEventSummary]:
        """0.2s 解像度で 1P/2P それぞれ連鎖イベントを追跡する。"""
        tracker_1p = VideoChainTracker(match_start_sec=0.0)
        tracker_2p = VideoChainTracker(match_start_sec=0.0)
        events: list[ChainEventSummary] = []
        t = 0.0
        while start + t < end:
            abs_t = start + t
            frame = self._read_frame_at(cap, abs_t)
            if frame is None:
                t += BOARD_INTERVAL_SEC
                continue
            self._update_trackers(
                frame, t, tracker_1p, tracker_2p, events,
            )
            t += BOARD_INTERVAL_SEC
        return events

    @staticmethod
    def _update_trackers(
        frame: np.ndarray,
        t_sec: float,
        tracker_1p: VideoChainTracker,
        tracker_2p: VideoChainTracker,
        events: list[ChainEventSummary],
    ) -> None:
        """両トラッカーに盤面を投入し、検出されたイベントを追加する。"""
        try:
            reader = _SHARED_READER
            board_1p, board_2p = reader.read_both_boards(frame)
        except Exception:
            return
        try:
            ev1 = tracker_1p.update(t_sec, board_1p)
            if ev1 is not None:
                events.append(ChainEventSummary.from_event("1P", ev1))
        except Exception:
            pass
        try:
            ev2 = tracker_2p.update(t_sec, board_2p)
            if ev2 is not None:
                events.append(ChainEventSummary.from_event("2P", ev2))
        except Exception:
            pass

    # ============================
    # 内部: I/O ヘルパ
    # ============================

    @staticmethod
    def _read_frame_at(
        cap: cv2.VideoCapture,
        t_sec: float,
    ) -> np.ndarray | None:
        """指定秒のフレームを読む。失敗時 None。"""
        cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame

    @staticmethod
    def _get_video_info(video_path: Path) -> dict[str, float]:
        """動画長と fps を返す。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"fps": 0.0, "duration_sec": 0.0}
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = total / fps if fps > 0 else 0.0
        return {"fps": fps, "duration_sec": duration}

    # ============================
    # 内部: 構築ヘルパ
    # ============================

    def _build_reader(self) -> ImageReader:
        """ImageReader を構築。calib があれば BoardRegion を上書きする。"""
        if self._calib_path is not None and self._calib_path.exists():
            try:
                from src.calibration import CalibratedConfig
                config = CalibratedConfig.load(self._calib_path)
                return ImageReader(
                    p1_region=config.p1_region,
                    p2_region=config.p2_region,
                )
            except Exception:
                pass
        return ImageReader()

    def _try_load_next_detector(self):
        """CNN 重みがあれば NextDetector を構築、無理なら None。"""
        if self._cnn_path is None or not self._cnn_path.exists():
            return None
        try:
            from src.next_detector import NextDetector
            return NextDetector.load_default(cnn_path=self._cnn_path)
        except Exception:
            return None

    def _try_load_match_detector(self) -> MatchStateDetector | None:
        """calib があれば MatchStateDetector を構築、無理なら None。"""
        if self._calib_path is None or not self._calib_path.exists():
            return None
        try:
            return MatchStateDetector.load_default(calib_path=self._calib_path)
        except Exception:
            return None

    def _build_meta(self) -> dict[str, str]:
        """meta 辞書を組み立てる。"""
        meta: dict[str, str] = {}
        if self._cnn_path is not None:
            meta["cnn_path"] = str(self._cnn_path)
        if self._calib_path is not None:
            meta["calib_path"] = str(self._calib_path)
        meta["board_interval_sec"] = str(BOARD_INTERVAL_SEC)
        meta["eval_interval_sec"] = str(EVAL_INTERVAL_SEC)
        return meta


# ============================
# 共有 ImageReader (連鎖追跡用)
# ============================


# 連鎖追跡で逐次新規 reader を作るのを避けるためモジュールレベルで共有
_SHARED_READER: ImageReader = ImageReader()


# ============================
# ScorePoint 構築ヘルパ
# ============================


def _build_score_point(
    t_sec: float,
    score_result: Any,
    ind1: IndicatorSet,
    ind2: IndicatorSet,
    phase: str | None = None,
) -> ScorePoint:
    """ScoreResult / IndicatorSet から ScorePoint を組み立てる。"""
    breakdown: dict[str, float] = {}
    p1_break = score_result.player1_breakdown
    p2_break = score_result.player2_breakdown
    for name in ALL_INDICATOR_NAMES:
        breakdown[name] = float(
            p1_break.get(name, 0.0) - p2_break.get(name, 0.0)
        )
    # next_acceptance も別領域に格納
    breakdown[INDICATOR_NEXT_ACCEPTANCE] = float(
        ind1.next_acceptance - ind2.next_acceptance
    )
    return ScorePoint(
        t_sec=t_sec,
        score=float(score_result.total_score),
        breakdown=breakdown,
        p1_advantage_score=float(score_result.player1_raw),
        p2_advantage_score=float(score_result.player2_raw),
        phase=phase,
    )
