"""DirectShow入力の明示選択と内容確認。色較正本体はここでは実行しない。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterator

import cv2
import numpy as np
from src.production_config import FAST_FIRST_JUDGMENT_DEFAULT, LIVE_VERIFY_ASYNC_DEFAULT
from src.live_speed_c import enabled as c_enabled

from .dshow_color import (COLOR_CORRECTION_MODES, DEFAULT_COLOR_CORRECTION, correct_601_as_709,
                          fourcc_to_str, resolve_correction)
from .live_source import CapturedFrame, FrameSource, NATIVE_SIZE, RECOGNITION_HZ
from .live_input_guard import FrameContinuity
from . import live_stage_timing as timing
from .live_border_grace import wrap_verifier, verify_screen
from . import live_verify_reasons as reasons
from . import live_verify_split as split
from . import live_verify_entry as entry

ASPECT_TOLERANCE = 0.01
VERIFY_PERIOD_SEC = 1.0
READ_RETRY_SEC = 0.05
GRID_MARGIN = 20
GRID_POSITION_TOLERANCE = 80
BORDER_POSITION_TOLERANCE = 24
BORDER_COVERAGE = 0.65
# 継続確認の不合格を何回連続で数えたら hold するか。既定 1 = 従来 (1 回で hold)。
DEFAULT_VERIFY_FAIL_STREAK = 1
CALIBRATION_ROOT = Path('config/device_calibration')
# 保存先の切替 (任意)。未設定なら従来の CALIBRATION_ROOT のまま (配布ランチャーが設定から与える)。
CALIBRATION_DIR_ENV = 'PUYO_CALIBRATION_DIR'


@dataclass(frozen=True)
class DeviceConfig:
    name: str
    index: int
    verification_only: bool = True
    # DirectShow の色行列補正 (off / auto / 601to709)。既定 off = 従来と同一 (dshow_color.py)。
    color_correction: str = DEFAULT_COLOR_CORRECTION
    # 画面確認の緩和 (既定 False = 従来と同一): 得点 8 桁 または 掛け算式で得点欄を許容し、外枠照合は上辺を免除。
    relaxed_verify: bool = False
    # 継続確認で連続 K 回不合格になって初めて不合格扱い (既定 1 = 従来)。入口の初回確認は常に 1 回。
    verify_fail_streak: int = DEFAULT_VERIFY_FAIL_STREAK
    fast_first_judgment: bool = FAST_FIRST_JUDGMENT_DEFAULT

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or type(self.index) is not int or self.index < 0:
            raise ValueError('機器名と非負の機器indexを明示してください')
        if type(self.verification_only) is not bool:
            raise ValueError('verification_onlyはboolが必要です')
        if self.color_correction not in COLOR_CORRECTION_MODES:
            raise ValueError(f'color_correctionは{COLOR_CORRECTION_MODES}のいずれかが必要です')
        if type(self.relaxed_verify) is not bool:
            raise ValueError('relaxed_verifyはboolが必要です')
        if type(self.verify_fail_streak) is not int or self.verify_fail_streak < 1:
            raise ValueError('verify_fail_streakは1以上の整数が必要です')
        if type(self.fast_first_judgment) is not bool:
            raise ValueError('fast_first_judgmentはboolが必要です')

    @classmethod
    def load(cls, path: Path) -> DeviceConfig:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        return cls(name=data['name'], index=data['index'], verification_only=data.get('verification_only', True),
                   color_correction=data.get('color_correction', DEFAULT_COLOR_CORRECTION),
                   relaxed_verify=data.get('relaxed_verify', False),
                   verify_fail_streak=data.get('verify_fail_streak', DEFAULT_VERIFY_FAIL_STREAK),
                   fast_first_judgment=data.get('fast_first_judgment', FAST_FIRST_JUDGMENT_DEFAULT))

    @property
    def calibration_path(self) -> Path:
        identity = hashlib.sha256(f'{self.name}:{self.index}'.encode()).hexdigest()
        override = os.environ.get(CALIBRATION_DIR_ENV)
        return (Path(override) if override else CALIBRATION_ROOT) / (identity + '.json')


class PuyoScreenVerifier:
    """固定の盤面枠位置と両側得点OCRを既存検出器で確認する。

    relaxed=True (既定 False = 従来と同一) のとき:
      - 得点欄は「8桁が読める」または「掛け算式が有効」で合格 (連鎖中の式表示を許容)。
      - 外枠フォールバックは上辺を免除し下/左/右の3辺で照合 (高積みで上辺が隠れる局面)。
    """

    relaxed = False
    # live_verify_soft_hold が ON のときだけ、直近の失敗の段を failure に残す (OFF は何も残さず従来と同一)。
    _record_failure = False
    failure: split.VerifyFailure | None = None

    def __init__(self, relaxed: bool = False) -> None:
        from src.board_grid_detector import BoardGridDetector
        from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
        from src.score_ocr import ScoreOcr
        self.grid = BoardGridDetector()
        self.score = ScoreOcr.load_default()
        self.regions = (DEFAULT_P1_REGION, DEFAULT_P2_REGION)
        self.relaxed = relaxed
        self._record_failure = split.enabled()

    def _fail(self, stage: str, side: str = '', failed_edges: int | None = None) -> bool:
        """不合格を返す。ON のときだけ、どの段で落ちたかを残す。"""
        if self._record_failure:
            self.failure = split.VerifyFailure(stage, side, failed_edges)
        return False

    def _border_ok(self, crop: np.ndarray, region: Any, side: str) -> bool:
        """格子が見つからない側の外枠照合。ON のときだけ失敗した辺の数を記録する (OFF は従来と同じ呼び出し)。"""
        if not self._record_failure:
            return frame_border_matches(crop, region.width, region.height, exempt_top=self.relaxed)
        edges: list[int] = []
        matched = frame_border_matches(crop, region.width, region.height,
                                       exempt_top=self.relaxed, failed_edges_out=edges)
        if not matched:
            self._fail(split.STAGE_BORDER, side, edges[0] if edges else None)
        return matched

    def score_ok(self, image: np.ndarray, side: str) -> bool:
        """得点欄が読めるか。relaxed 時は掛け算式が有効でも可 (既存 read_formula_side を再利用)。"""
        score = self.score.read_side(image, side)[0]
        if score is not None:
            reasons.side_detail('score', passed=True, ocr_ok=True, value=score, formula_valid=None)
            return True
        formula = bool(self.score.read_formula_side(image, side).valid) if self.relaxed else None
        passed = self.relaxed and bool(formula)
        reasons.side_detail('score', passed=passed, ocr_ok=False, value=None, formula_valid=formula)
        return passed

    def __call__(self, image: np.ndarray) -> bool:
        from .distribution_input import has_feedback
        if self._record_failure:
            self.failure = None
        self.feedback = has_feedback(image)
        reasons.detail('feedback', passed=not self.feedback, value=bool(self.feedback),
                       **reasons.feedback_values())
        if self.feedback:
            return self._fail(split.STAGE_FEEDBACK)
        for side, region in zip(('1P', '2P'), self.regions):
            reasons.select_side(side)
            x, y = region.x-GRID_MARGIN, region.y-GRID_MARGIN
            crop = image[y:region.y+region.height+GRID_MARGIN,
                         x:region.x+region.width+GRID_MARGIN]
            grid = self.grid.detect(crop)
            reasons.side_detail('grid', detected=grid is not None)
            if not self.score_ok(image, side):
                return self._fail(split.STAGE_SCORE, side)
            if grid is None:
                if not self._border_ok(crop, region, side):
                    return False
                continue
            expected = np.array([GRID_MARGIN, GRID_MARGIN,
                                 region.width+GRID_MARGIN, region.height+GRID_MARGIN])
            detected = np.array([*grid.top_left, *grid.bottom_right])
            error = np.max(np.abs(expected-detected))
            reasons.side_detail('grid_position', passed=bool(error <= GRID_POSITION_TOLERANCE),
                                error_px=float(error), tolerance_px=GRID_POSITION_TOLERANCE)
            if error > GRID_POSITION_TOLERANCE:
                return self._fail(split.STAGE_GRID_POSITION, side)
        return True


def frame_border_matches(crop: np.ndarray, width: int, height: int, exempt_top: bool = False,
                         failed_edges_out: list[int] | None = None) -> bool:
    """内部grid線が隠れる局面は、既存Hough分離器で外枠4辺を照合する (exempt_top 時は上辺を除く3辺)。

    failed_edges_out を渡したときだけ、不合格なら被覆の足りない辺の数を 1 件追加する (判定は変えない)。
    """
    from src.board_grid_detector import (CANNY_LOW, CANNY_HIGH, HOUGH_RHO, HOUGH_THETA,
        HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP, _segments_to_lines)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    segments = cv2.HoughLinesP(edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH, maxLineGap=HOUGH_MAX_LINE_GAP)
    if segments is None:
        reasons.border_details([], [], width, height, exempt_top)
        _note_failed_edges(failed_edges_out, [], [], width, height, exempt_top)
        return False
    horizontal, vertical = _segments_to_lines(segments)
    reasons.border_details(horizontal, vertical, width, height, exempt_top)
    matched = (border_pair(horizontal, 1, width, height, exempt_first=exempt_top)
               and border_pair(vertical, 0, height, width))
    if not matched:
        _note_failed_edges(failed_edges_out, horizontal, vertical, width, height, exempt_top)
    return matched


def failed_edge_count(lines: list[tuple], axis: int, span: int, distance: int,
                      exempt_first: bool = False) -> int:
    """被覆の足りない辺の数。border_pair と同じ辺・同じ基準で数える (border_pair が真なら 0)。"""
    targets = (GRID_MARGIN, GRID_MARGIN+distance)[1 if exempt_first else 0:]
    return sum(1 for target in targets
               if not border_coverage(lines, axis, span, target) >= BORDER_COVERAGE)


def _note_failed_edges(out: list[int] | None, horizontal: list[tuple], vertical: list[tuple],
                       width: int, height: int, exempt_top: bool) -> None:
    """out があるときだけ、失敗した辺の合計 (上下 + 左右) を追加する。None なら計算もしない。"""
    if out is not None:
        out.append(failed_edge_count(horizontal, 1, width, height, exempt_first=exempt_top)
                   + failed_edge_count(vertical, 0, height, width))


def border_pair(lines: list[tuple], axis: int, span: int, distance: int,
                exempt_first: bool = False) -> bool:
    """位置と辺の長さの両方が合う線を、対向する2辺に要求する。exempt_first 時は先頭辺 (上/左) を免除。"""
    targets = (GRID_MARGIN, GRID_MARGIN+distance)[1 if exempt_first else 0:]
    return all(border_coverage(lines, axis, span, target) >= BORDER_COVERAGE
               for target in targets)


def border_coverage(lines: list[tuple], axis: int, span: int, target: int) -> float:
    """ぷよや演出で分断された同じ辺の線分を重複なく合計する。"""
    intervals = []
    for line in lines:
        if abs((line[axis]+line[axis+2])/2-target) > BORDER_POSITION_TOLERANCE:
            continue
        start, end = sorted((line[1-axis], line[3-axis]))
        intervals.append((max(GRID_MARGIN, start), min(GRID_MARGIN+span, end)))
    covered, previous_end = 0.0, float('-inf')
    for start, end in sorted(intervals):
        covered += max(0.0, end-max(start, previous_end))
        previous_end = max(previous_end, end)
    return covered/span


class DirectShowSource(FrameSource):
    """名前だけでは検索しない。未確認映像は認識へ一枚も渡さない。"""

    # live_verify_soft_hold が ON のときだけ、保留の入力診断へ失敗の種別を足す (OFF の診断は従来と同じ)。
    _split_diagnostics = False

    def __init__(self, config: DeviceConfig, duration: float,
                 on_hold: Callable[[], None],
                 capture_factory: Callable[..., Any] = cv2.VideoCapture,
                 verifier: Callable[[np.ndarray], bool] | None = None,
                 clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep,
                 on_status: Callable[[str], None] | None = None) -> None:
        self.config, self.duration, self.on_hold = config, duration, on_hold
        self.capture_factory, self.verifier = capture_factory, verifier
        self.clock, self.sleep, self.dropped = clock, sleep, 0
        self.on_status = on_status
        self.on_format_change: Callable[[], None] | None = None
        self.on_match_end_hold: Callable[[Any, float], None] | None = None
        self._match_end_hold_verifier: PuyoScreenVerifier | None = None
        self._source_size: tuple[int, int] | None = None
        self._correct_color = False  # 開いた機器のサブタイプ確定後に __iter__ が決める
        from .live_color_buffer import configured_corrector
        self._color_corrector = configured_corrector()
        self.input_diagnostics: dict[str, Any] = {}
        self._split_diagnostics = split.enabled()
        from .live_degrade import EventPriority
        self.event_priority = EventPriority()
        self._buffered_capture = self.event_priority.enabled or c_enabled('live_capture_thread')
        from .live_capture_reconnect import make_reconnector
        self._reconnector = make_reconnector()   # 既定 OFF は None (従来どおり 1 回だけ開く)
        self._read_ok = True

    def _capture(self) -> Any:
        if not self._buffered_capture:
            return self.capture_factory(self.config.index, cv2.CAP_DSHOW)
        from .live_capture_buffer import BufferedCapture
        return BufferedCapture(self.capture_factory, (self.config.index, cv2.CAP_DSHOW),
                               self.event_priority, self.clock, expose_metadata=c_enabled('live_capture_thread'))

    def _configure(self, capture: Any) -> None:
        """開いた直後の機器へ解像度・FPS を要求し、色補正の要否を決める (起動時と再接続時で共通)。"""
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE_SIZE[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_SIZE[1])
        capture.set(cv2.CAP_PROP_FPS, RECOGNITION_HZ)
        self._correct_color = self._decide_color_correction(capture)

    def _maintain(self, capture: Any, origin: float, now: float) -> Any:
        """既定 OFF の再接続 (live_capture_reconnect)。読めない状態が続いたら機器を作り直して返す。
        OFF (None) では何もしない。直前の機器は必ず解放してから新しい機器を開く。"""
        reconnector = self._reconnector
        if reconnector is None or not reconnector.due(self._read_ok, self._read_ok or capture.isOpened(), now):
            return capture
        capture.release()
        fresh = self._capture()
        if self._buffered_capture:
            fresh.origin, fresh.duration = origin, self.duration
        self._configure(fresh)
        reconnector.record_attempt(bool(fresh.isOpened()), self.clock())
        return fresh

    def _read(self, capture: Any) -> tuple[float, Any, tuple[int, int] | None]:
        captured = self.clock()
        timing.begin('recognition', captured)
        ok, image = timing.measure('capture', capture.read) if capture.isOpened() else (False, None)
        self._read_ok = bool(ok)
        if self.config.fast_first_judgment and ok and image is not None and image.ndim >= 2:
            self._check_format((image.shape[1], image.shape[0]))
        if self._buffered_capture:
            captured = capture.last_captured_at if ok else captured
            self.dropped = capture.dropped
        normalized = timing.measure('normalize', self._normalize, image) if ok else None
        if normalized is not None and self._correct_color:
            from .live_color_roi import correct_input
            correct = correct_input if c_enabled('live_color_roi') else (self._color_corrector or correct_601_as_709)
            normalized = timing.measure('color_correction', correct, normalized)
        size = (image.shape[1], image.shape[0]) if normalized is not None else None
        return captured, normalized, size

    def _check_format(self, size: tuple[int, int]) -> None:
        """正規化前の形式変更を通知する。画面検証と強制境界の条件は変更しない。"""
        if self._source_size is not None and size != self._source_size and self.on_format_change:
            self.on_format_change()
        self._source_size = size

    def _screen_state(self, verifier: Any) -> Any:
        """既存の非同期設定に従い、検証状態の所有者を作る。"""
        nonblocking = os.environ.get('PUYO_LIVE_VERIFY_ASYNC', str(int(LIVE_VERIFY_ASYNC_DEFAULT))) == '1'
        if nonblocking or os.environ.get('PUYO_LIVE_ASYNC_SCREEN_VERIFY') == '1':
            from .live_screen_worker import AsyncScreenState
            return AsyncScreenState(verifier, self._judge, VERIFY_PERIOD_SEC, wait_initial=not nonblocking)
        return None

    def __iter__(self) -> Iterator[CapturedFrame]:
        capture = self._capture()
        origin, last_verify, verified = self.clock(), float('-inf'), False
        entry_fast_retry = entry.enabled()
        fail_streak = 0  # 検証済み状態での連続不合格回数 (verify_fail_streak 未満なら許容)
        screen_generation = 0
        if self._buffered_capture:
            capture.origin, capture.duration = origin, self.duration
        continuity = FrameContinuity(VERIFY_PERIOD_SEC)
        async_screen = None
        try:
            verifier = wrap_verifier(self.verifier or PuyoScreenVerifier(relaxed=self.config.relaxed_verify))
            async_screen = self._screen_state(verifier)
            self._configure(capture)
            while self.clock()-origin < self.duration:
                captured, normalized, size = self._read(capture)
                capture = self._maintain(capture, origin, captured)
                transport_ready = normalized is not None and timing.measure(
                    'continuity', continuity.ready, normalized, size, captured)
                if async_screen is not None:
                    verified, fail_streak = self._async_check(async_screen, normalized, size,
                                                              captured, transport_ready)
                elif not transport_ready:
                    screen_generation += 1
                    verified, fail_streak = False, 0
                    last_verify = float('-inf')
                elif captured-last_verify >= entry.verification_period(verified, VERIFY_PERIOD_SEC, entry_fast_retry):
                    passed = timing.measure('screen_verify', verify_screen, verifier, normalized,
                                            captured, size, screen_generation)
                    verified, fail_streak = self._judge(passed, verified, fail_streak)
                    last_verify = captured
                if not verified:
                    if self.on_match_end_hold is not None:
                        self._feed_match_end_hold(normalized, captured, transport_ready)
                    self._screen_hold(transport_ready, normalized, continuity, fail_streak,
                                      async_screen or verifier)
                    continue
                elapsed = captured-origin
                yield CapturedFrame(round(elapsed*RECOGNITION_HZ), elapsed,
                                    captured, self.clock(), normalized,
                                    dropped_before=getattr(capture, 'dropped_before', 0), source_size=size)
                self.sleep(max(0.0, captured+1/RECOGNITION_HZ-self.clock()))
        finally:
            try:
                capture.release()
            finally:
                if async_screen is not None:
                    async_screen.close()
            if self._buffered_capture:
                self.dropped = capture.dropped

    def _feed_match_end_hold(self, image: Any, captured: float, ready: bool) -> None:
        """検証workerとOCR等の可変状態を共有せず、最新の画像を独立確認する。"""
        from .live_match_end_feed import eligible
        allowed = False
        if ready:
            if self._match_end_hold_verifier is None:
                self._match_end_hold_verifier = PuyoScreenVerifier(relaxed=self.config.relaxed_verify)
            allowed = eligible(image, self._match_end_hold_verifier)
        self.on_match_end_hold(image if allowed else None, captured)

    def _async_check(self, state: Any, image: Any, size: Any,
                     captured: float, ready: bool) -> tuple[bool, int]:
        """workerの実時間は対象画像の時刻付きで記録し、認識側の待ちと分離する。"""
        from dataclasses import asdict
        result = timing.measure('screen_verify', state.update, image, size, captured, ready)
        if state.latest_result is not None:
            record = asdict(state.latest_result)
            if not self._split_diagnostics:
                record.pop('failure', None)   # OFF の診断出力は従来と同じキー集合のまま
            if not state.wait_initial:
                record.update(applied_at=captured, application_delay_sec=state.application_delay_sec,
                              fail_streak=result[1])
            timing.detail(screen_result=record)
        return result

    def _screen_hold(self, ready: bool, image: Any, continuity: FrameContinuity,
                     streak: int, verifier: Any) -> None:
        """保留通知はworkerではなく、入力所有者から従来どおり発行する。"""
        self.input_diagnostics = dict(transport_ready=ready, normalized=image is not None,
            continuity=dict(continuity.diagnostics) if image is not None else None,
            verifier_fail_streak=streak, verifier_fail_limit=self.config.verify_fail_streak)
        if self._split_diagnostics:
            self.input_diagnostics.update(split.diagnostics(getattr(verifier, 'failure', None), ready))
        if self.on_status:
            self.on_status('verifying' if not ready else
                           'overlay_feedback' if getattr(verifier, 'feedback', False) else 'no_puyo_screen')
        else:
            self.on_hold()
        timing.finish(held=True, input_diagnostics=self.input_diagnostics)
        self.sleep(READ_RETRY_SEC)

    def _judge(self, passed: bool, verified: bool, fail_streak: int) -> tuple[bool, int]:
        """検証結果を継続確認の連続不合格数へ反映する。入口 (未検証) の不合格は即不合格 (K=1)。"""
        if passed:
            return True, 0
        fail_streak += 1
        return verified and fail_streak < self.config.verify_fail_streak, fail_streak

    def _decide_color_correction(self, capture: Any) -> bool:
        """色補正の要否。auto は開いた機器のメディアサブタイプ (BufferedCapture 等 get が無い場合は不明扱い)。"""
        mode = self.config.color_correction
        if mode == DEFAULT_COLOR_CORRECTION:
            return False
        getter = getattr(capture, 'get', None)
        subtype = fourcc_to_str(getter(cv2.CAP_PROP_FOURCC)) if callable(getter) else None
        return resolve_correction(mode, subtype)

    @staticmethod
    def _normalize(image: np.ndarray | None) -> np.ndarray | None:
        if image is None or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            return None
        height, width = image.shape[:2]
        if abs(width/height-NATIVE_SIZE[0]/NATIVE_SIZE[1]) > ASPECT_TOLERANCE:
            return None
        return image if (width, height) == NATIVE_SIZE else cv2.resize(image, NATIVE_SIZE)


class DeviceMetadataCapture:
    """既存動画CLIへFPS/尺のみ渡す。実デバイスは認識processだけが開く。"""

    def __init__(self, duration: float) -> None:
        self.duration = duration

    def isOpened(self) -> bool:
        return True

    def get(self, key: int) -> float:
        return {cv2.CAP_PROP_FPS: RECOGNITION_HZ,
                cv2.CAP_PROP_FRAME_COUNT: self.duration*RECOGNITION_HZ,
                cv2.CAP_PROP_FRAME_WIDTH: NATIVE_SIZE[0],
                cv2.CAP_PROP_FRAME_HEIGHT: NATIVE_SIZE[1]}.get(key, 0.0)

    def set(self, key: int, value: float) -> bool:
        return True

    def release(self) -> None:
        pass
