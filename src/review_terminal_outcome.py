"""レビュー表示用に、左右対称な決着ロゴから確定勝敗を検出する。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.match_end_detector import (
    DEFAULT_NCC_THRESHOLD,
    DEFAULT_TEMPLATE_DIR,
    MATCH_END_PREFIX,
    SEARCH_P1,
    SEARCH_P2,
)


SIDES = ("p1", "p2")
REQUIRED_CONSECUTIVE_HITS = 3
EXPECTED_FRAME_SHAPE = (1080, 1920)
P2_TOP_EXTENSION = SEARCH_P2[1] - SEARCH_P1[1]
REVIEW_SEARCH_P2: tuple[int, int, int, int] = (
    SEARCH_P2[0], SEARCH_P1[1], SEARCH_P2[2], SEARCH_P2[3] + P2_TOP_EXTENSION,
)
YATTA_TEMPLATE_SHAPE = (370, 520)
YATTA_LOGO_CROP: tuple[int, int, int, int] = (60, 210, 460, 160)


@dataclass(frozen=True, slots=True)
class ReviewTerminalEvidence:
    """現在フレームだけから得た、死亡側と整合する決着表示の証拠。"""

    loser: str
    template_name: str
    score: float
    consecutive_hits: int
    bilateral: bool = False


class ReviewTerminalOutcomeDetector:
    """「やった」「ばたんきゅー」を両盤面で探索し、片側勝敗を確定する。"""

    def __init__(
        self, templates: dict[str, np.ndarray],
        threshold: float = DEFAULT_NCC_THRESHOLD,
        required_hits: int = REQUIRED_CONSECUTIVE_HITS,
    ) -> None:
        self._templates = templates
        self._threshold = threshold
        self._required_hits = required_hits
        self._expected_loser: str | None = None
        self._hit_count = 0

    @classmethod
    def load_default(
        cls, template_dir: Path = DEFAULT_TEMPLATE_DIR,
        threshold: float = DEFAULT_NCC_THRESHOLD,
        required_hits: int = REQUIRED_CONSECUTIVE_HITS,
        yatta_logo_only: bool = False,
    ) -> "ReviewTerminalOutcomeDetector":
        templates = _load_templates(template_dir, yatta_logo_only=yatta_logo_only)
        return cls(templates, threshold=threshold, required_hits=required_hits)

    def reset(self) -> None:
        self._expected_loser = None
        self._hit_count = 0

    def update(
        self, frame: np.ndarray, expected_loser: str,
    ) -> ReviewTerminalEvidence | None:
        if expected_loser not in SIDES or frame.shape[:2] != EXPECTED_FRAME_SHAPE:
            self.reset()
            return None
        expected, contradiction = _outcome_scores(frame, self._templates, expected_loser)
        if expected[0] < self._threshold or contradiction >= self._threshold:
            self.reset()
            return None
        if expected_loser != self._expected_loser:
            self._expected_loser = expected_loser
            self._hit_count = 0
        self._hit_count += 1
        if self._hit_count < self._required_hits:
            return None
        return ReviewTerminalEvidence(
            loser=expected_loser, template_name=expected[1],
            score=expected[0], consecutive_hits=self._hit_count,
        )

    def update_symmetric(
        self, frame: np.ndarray,
    ) -> ReviewTerminalEvidence | None:
        """勝者・敗者ロゴが左右で一致した場合だけ、死亡候補なしで確定する。"""

        if frame.shape[:2] != EXPECTED_FRAME_SHAPE:
            self.reset()
            return None
        candidate = _symmetric_outcome(frame, self._templates, self._threshold)
        if candidate is None:
            self.reset()
            return None
        loser, score = candidate
        if loser != self._expected_loser:
            self._expected_loser = loser
            self._hit_count = 0
        self._hit_count += 1
        if self._hit_count < self._required_hits:
            return None
        return ReviewTerminalEvidence(
            loser=loser, template_name="match_end_yatta+batan",
            score=score, consecutive_hits=self._hit_count, bilateral=True,
        )


def _load_templates(
    template_dir: Path, *, yatta_logo_only: bool = False,
) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for path in sorted(template_dir.glob(f"{MATCH_END_PREFIX}*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            if path.stem == "match_end_yatta" and yatta_logo_only:
                image = _crop_yatta_logo(image)
            templates[path.stem] = image
    return templates


def _crop_yatta_logo(template: np.ndarray) -> np.ndarray:
    if template.shape != YATTA_TEMPLATE_SHAPE:
        raise ValueError(f"やった画像shapeが固定crop契約と不一致です: {template.shape}")
    x, y, width, height = YATTA_LOGO_CROP
    return np.ascontiguousarray(template[y:y + height, x:x + width])


def _outcome_scores(
    frame: np.ndarray, templates: dict[str, np.ndarray], loser: str,
) -> tuple[tuple[float, str], float]:
    winner = "p2" if loser == "p1" else "p1"
    expected_pairs = (("match_end_batan", loser), ("match_end_yatta", winner))
    contradiction_pairs = (("match_end_batan", winner), ("match_end_yatta", loser))
    expected = max((_template_score(frame, templates, *pair), pair[0]) for pair in expected_pairs)
    contradiction = max(_template_score(frame, templates, *pair) for pair in contradiction_pairs)
    return expected, contradiction


def _symmetric_outcome(
    frame: np.ndarray, templates: dict[str, np.ndarray], threshold: float,
) -> tuple[str, float] | None:
    """反対側と矛盾せず、勝ち・負けロゴが対になる側だけを返す。"""

    candidates: list[tuple[str, float]] = []
    for loser in SIDES:
        winner = "p2" if loser == "p1" else "p1"
        support = min(
            _template_score(frame, templates, "match_end_batan", loser),
            _template_score(frame, templates, "match_end_yatta", winner),
        )
        contradiction = max(
            _template_score(frame, templates, "match_end_batan", winner),
            _template_score(frame, templates, "match_end_yatta", loser),
        )
        if support >= threshold and contradiction < threshold:
            candidates.append((loser, support))
    return candidates[0] if len(candidates) == 1 else None


def _template_score(
    frame: np.ndarray, templates: dict[str, np.ndarray], name: str, side: str,
) -> float:
    template = templates.get(name)
    if template is None:
        return -1.0
    x, y, width, height = SEARCH_P1 if side == "p1" else REVIEW_SEARCH_P2
    roi = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
    if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
        return -1.0
    return float(cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED).max())


__all__ = [
    "EXPECTED_FRAME_SHAPE",
    "REVIEW_SEARCH_P2",
    "REQUIRED_CONSECUTIVE_HITS",
    "ReviewTerminalEvidence",
    "ReviewTerminalOutcomeDetector",
    "YATTA_LOGO_CROP",
    "YATTA_TEMPLATE_SHAPE",
]
