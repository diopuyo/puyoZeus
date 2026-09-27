"""入力ごとの色ウォームアップ。既存HSV較正の信頼条件と色別readyを再用する。"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.online_hsv_calibrator import OnlineHsvCalibrator, MIN_SAMPLES

PALETTE_SIZE = 4
PUYO_COLORS = (1, 2, 3, 4, 5)
RECHECK_DIVISOR = 4
PROFILE_VERSION = 1
PERCENT = 100
HSV_LIMITS = (180, 255, 255)


class ColorWarmup:
    """起動・入力切替後は新規サンプルが集まるまでreadyにしない。"""

    def __init__(self, device: Any, verification_only: bool = False,
                 calibrator: OnlineHsvCalibrator | None = None) -> None:
        self.device = device
        self.verification_only = verification_only
        self.profile = self._load()
        self.cached = self.profile is not None
        self.calibrator = deepcopy(calibrator) if calibrator is not None else OnlineHsvCalibrator()
        self.calibrator.reset()
        self.full_target = self.calibrator._min_samples
        self.target = max(1, self.full_target // RECHECK_DIVISOR) if self.cached else self.full_target
        self.calibrator._min_samples = self.target
        self.ready, self.progress = False, 0
        self.mode = 'verification' if verification_only else 'correction'
        self.changed_profile = False

    def _load(self) -> dict | None:
        try:
            data = json.loads(self.device.calibration_path.read_text(encoding='utf-8'))
            if (data['version'] == PROFILE_VERSION and data['device'] == self.identity()
                    and isinstance(data['ranges'], dict) and len(data['ranges']) >= PALETTE_SIZE):
                for color, values in data['ranges'].items():
                    if int(color) not in PUYO_COLORS or len(values) != 6:
                        return None
                    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                        return None
                    if any(not 0 <= values[2*i] <= values[2*i+1] <= limit
                           for i, limit in enumerate(HSV_LIMITS)):
                        return None
                return data
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def identity(self) -> dict:
        return dict(name=self.device.name, index=self.device.index)

    def status(self, phase: str | None = None) -> dict:
        return dict(phase=phase or ('ready' if self.ready else 'calibrating'),
                    progress=self.progress, cached=self.cached, mode=self.mode,
                    device_id=self.device.calibration_path.stem)

    def observe(self, frame: Any, result: Any, classifier: Any) -> dict:
        """STABLEの実ぷよだけを集める。空画面では既存profileがあっても終了しない。"""
        from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
        if self.ready:
            return self.status()
        if not result.is_match_active:
            return self.status('no_puyo_screen')
        for side, region in zip((result.p1, result.p2), (DEFAULT_P1_REGION, DEFAULT_P2_REGION)):
            if side.state.name != 'STABLE' or side.confirmed_board is None:
                continue
            proba, hsv = classifier.predict_proba_and_hsv_grid(frame, region)
            self.calibrator.update(frame, region, side.confirmed_board, proba, hsv)
        return self.advance(classifier)

    def advance(self, classifier: Any) -> dict:
        """4色それぞれが既存の必要サンプル数へ到達したら自動終了する。"""
        counts = self.calibrator.get_sample_counts()
        amounts = sorted((min(self.target, counts[c]) for c in PUYO_COLORS), reverse=True)
        self.progress = min(PERCENT-1, int(PERCENT*sum(amounts[:PALETTE_SIZE])/(self.target*PALETTE_SIZE)))
        colors = [c for c in PUYO_COLORS if self.calibrator.is_ready(c)]
        if len(colors) < PALETTE_SIZE:
            return self.status('calibrating' if any(amounts) else 'no_puyo_screen')
        ranges = {c: r for c, r in self.calibrator.get_per_video_ranges().items() if c in PUYO_COLORS}
        if self.cached and not self._compatible(ranges):
            self.cached, self.changed_profile, self.profile = False, True, None
            self.target = self.full_target
            self.calibrator._min_samples = self.target
            return self.advance(classifier)
        if self.cached:
            ranges = {int(c): tuple(r) for c, r in self.profile['ranges'].items()}
        if not self.verification_only:
            classifier._hsv.set_color_ranges_from_simple(ranges)
        self._save(ranges)
        self.ready, self.progress = True, PERCENT
        return self.status()

    def _compatible(self, ranges: dict) -> bool:
        """新規サンプルのHSV範囲と保存範囲が各軸で重なることを確認する。"""
        saved = self.profile['ranges']
        for color, bounds in ranges.items():
            previous = saved.get(str(color))
            if previous is None or any(max(bounds[i], previous[i]) > min(bounds[i+1], previous[i+1])
                                       for i in (0, 2, 4)):
                return False
        return True

    def _save(self, ranges: dict) -> None:
        """完了済みprofileだけを機器ID別に原子的に置き換える。"""
        path: Path = self.device.calibration_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(version=PROFILE_VERSION, device=self.identity(), mode=self.mode,
                    ranges=ranges, sample_counts=self.calibrator.get_sample_counts())
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
