"""試合範囲ゲート — 既存の matches.tsv を使って試合範囲外の記録を落とす。

2026-09-17 制定。Fable 総合レビュー P2。

背景:
    collect_boards_lean は `force_in_match=True` で RecognitionPipeline を生成するため、
    `raw_active` が常時 True になり、試合が始まっていない画面 (キャラ選択・暗転・待機) も
    盤面として記録される。実測 (video_38 先頭240秒・P0-a採用後): 182枚中21枚 = 11.5% が
    試合範囲外だった。user が実画面で誤読と判定した f8638 (キャラ選択) /
    f8754 (暗転) もこの中に含まれる。

    試合境界は 2026-08-18〜08-20 に自動検出が本番採用済みで、
    `data/verify/match_boundaries_v5/<video_id>/matches.tsv` に既に存在する
    (video_38 は43試合、94本ぶんの同種データあり)。にもかかわらず収集側から
    参照されていなかった (grep 0件)。本モジュールはその既存資産へ繋ぐだけで、
    新しい検出器を作らない。

契約:
    - 既定 OFF。フラグ未指定なら従来挙動と完全に同一。
    - matches.tsv が無い動画では何も落とさない (可用性を優先し、落とした件数を0と記録する)。
    - 落とした件数は必ず母数と併記できるよう counts() で返す。0件と未適用を区別する。
    - 認識そのものは変えない。記録するかどうかだけを決める。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

# 既存の境界データの探索順。新しい版を優先する。
BOUNDARY_DIRS: tuple[str, ...] = (
    "data/verify/match_boundaries_v5",
    "data/verify/match_boundaries_v4",
)


@dataclass
class MatchRangeGate:
    """試合範囲の内側かどうかを判定する。範囲が無ければ常に通す。"""

    ranges: tuple[tuple[float, float], ...] = ()
    source: str | None = None
    kept: int = 0
    dropped: int = 0
    _dropped_times: list[float] = field(default_factory=list)

    @property
    def active(self) -> bool:
        """境界データを読めたときだけ有効。"""
        return bool(self.ranges)

    def allows(self, t_sec: float) -> bool:
        """記録してよいかを返す。範囲が無いときは常に True。"""
        if not self.ranges:
            self.kept += 1
            return True
        for start, end in self.ranges:
            if start <= t_sec <= end:
                self.kept += 1
                return True
        self.dropped += 1
        if len(self._dropped_times) < 50:
            self._dropped_times.append(round(t_sec, 2))
        return False

    def counts(self) -> dict:
        """件数は必ず母数と併記する。0件と未適用を区別する。"""
        total = self.kept + self.dropped
        return dict(
            active=self.active,
            source=self.source,
            match_count=len(self.ranges),
            total=total,
            kept=self.kept,
            dropped=self.dropped,
            dropped_rate=(self.dropped / total) if total else None,
            dropped_times_head=list(self._dropped_times),
        )


def load_ranges(video_id: str, root: Path) -> tuple[tuple[tuple[float, float], ...], str | None]:
    """既存の matches.tsv から試合範囲を読む。無ければ空を返す (作らない)。"""
    for relative in BOUNDARY_DIRS:
        path = root / relative / video_id / "matches.tsv"
        if not path.is_file():
            continue
        ranges: list[tuple[float, float]] = []
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                try:
                    start, end = float(row["start_sec"]), float(row["end_sec"])
                except (KeyError, TypeError, ValueError):
                    continue
                if end > start:
                    ranges.append((start, end))
        if ranges:
            return tuple(ranges), str(path)
    return (), None


def build(video_id: str, root: Path, enabled: bool) -> MatchRangeGate:
    """既定 OFF。enabled=False なら範囲を読まず、常に通すゲートを返す。"""
    if not enabled:
        return MatchRangeGate()
    ranges, source = load_ranges(video_id, root)
    return MatchRangeGate(ranges=ranges, source=source)
