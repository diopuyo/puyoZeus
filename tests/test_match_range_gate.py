"""試合範囲ゲートの検査。既定OFFの不変性と、母数つきの件数記録を確認する。"""
from __future__ import annotations

from pathlib import Path

from src import match_range_gate as G


def make_tsv(root: Path, video_id: str, rows: list[tuple[float, float]]) -> Path:
    path = root / "data/verify/match_boundaries_v5" / video_id
    path.mkdir(parents=True, exist_ok=True)
    target = path / "matches.tsv"
    lines = ["idx\tstart_sec\tend_sec\tduration_sec"]
    for index, (start, end) in enumerate(rows, start=1):
        lines.append("%d\t%.1f\t%.1f\t%.1f" % (index, start, end, end - start))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def test_disabled_by_default_passes_everything(tmp_path: Path) -> None:
    """既定OFF。境界データがあっても読まず、全部通す (従来挙動と同一)。"""
    make_tsv(tmp_path, "video_38", [(154.0, 227.0)])
    gate = G.build("video_38", tmp_path, enabled=False)
    assert gate.active is False
    assert all(gate.allows(t) for t in (0.0, 63.5, 154.0, 500.0))
    counts = gate.counts()
    assert counts["dropped"] == 0 and counts["total"] == 4
    assert counts["active"] is False


def test_drops_outside_match_ranges(tmp_path: Path) -> None:
    """有効時は試合範囲外を落とす。実測の値 (試合1は154.0〜227.0秒) を使う。"""
    make_tsv(tmp_path, "video_38", [(154.0, 227.0), (230.0, 269.0)])
    gate = G.build("video_38", tmp_path, enabled=True)
    assert gate.active is True and len(gate.ranges) == 2
    assert gate.allows(160.0) is True
    assert gate.allows(250.0) is True
    assert gate.allows(63.53) is False       # 実測の範囲外の例
    assert gate.allows(143.97) is False      # userがキャラ選択と判定したframe
    assert gate.allows(228.5) is False       # 試合と試合の間
    counts = gate.counts()
    assert counts["kept"] == 2 and counts["dropped"] == 3
    assert counts["total"] == 5
    assert abs(counts["dropped_rate"] - 0.6) < 1e-9


def test_boundaries_are_inclusive(tmp_path: Path) -> None:
    """境界そのものは試合内として扱う。"""
    make_tsv(tmp_path, "video_38", [(154.0, 227.0)])
    gate = G.build("video_38", tmp_path, enabled=True)
    assert gate.allows(154.0) is True
    assert gate.allows(227.0) is True
    assert gate.allows(153.99) is False


def test_missing_tsv_passes_everything(tmp_path: Path) -> None:
    """境界データが無い動画では何も落とさない。可用性を優先する。"""
    gate = G.build("video_unknown", tmp_path, enabled=True)
    assert gate.active is False and gate.source is None
    assert gate.allows(1.0) is True
    assert gate.counts()["dropped"] == 0


def test_counts_distinguish_zero_from_inactive(tmp_path: Path) -> None:
    """0件と未適用を区別する。件数だけを見て合格と読み替えない。"""
    make_tsv(tmp_path, "video_38", [(0.0, 1000.0)])
    active = G.build("video_38", tmp_path, enabled=True)
    active.allows(100.0)
    inactive = G.build("video_38", tmp_path, enabled=False)
    inactive.allows(100.0)
    assert active.counts()["dropped"] == 0 and active.counts()["active"] is True
    assert inactive.counts()["dropped"] == 0 and inactive.counts()["active"] is False


def test_v4_is_used_when_v5_missing(tmp_path: Path) -> None:
    """v5が無ければv4を見る。新しい版を優先する。"""
    path = tmp_path / "data/verify/match_boundaries_v4/video_x"
    path.mkdir(parents=True, exist_ok=True)
    (path / "matches.tsv").write_text("idx\tstart_sec\tend_sec\n1\t10.0\t20.0\n", encoding="utf-8")
    gate = G.build("video_x", tmp_path, enabled=True)
    assert gate.active is True
    assert gate.allows(15.0) is True and gate.allows(5.0) is False
