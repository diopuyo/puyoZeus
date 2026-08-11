"""「ありえない判定」走査器 D0/D1a/D1b + dump 読み出しモードのテスト.

- 純関数 detect_d0/detect_d1a/detect_d1b の陽性/陰性/境界ガード (単体テスト、npz非依存)
- scan_video() を含む end-to-end 1 件 (合成 npz + 決定的な stub score_fn)
- scan_video_from_dump() の end-to-end 1 件 (合成タイムラインdump npz)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.board import DEATH_COL, DEATH_ROW, Board
from scripts.scan_judgment_anomalies import (
    GAME_BOUNDARY_GUARD_SEC,
    JudgmentRecord,
    detect_d0,
    detect_d1a,
    detect_d1b,
    scan_video,
    scan_video_from_dump,
)
from scripts.visualize_advantage_overlay import (
    KILL_RATIO_FULL,
    KILL_ROOM_FLOOR,
    TimelineDumpRow,
    save_timeline_dump,
)


def _record(**overrides: object) -> JudgmentRecord:
    """テスト用デフォルト JudgmentRecord (境界外・生存・矛盾なし・pending/room 安全値)。"""
    base: dict[str, object] = dict(
        video_id="v_test", t_sec=100.0, game_idx=0, trigger_side="1P",
        adv=10.0, p1=0.55, drivers=(("board_color_puyo_total", 0.3),),
        is_dead_p1=False, is_dead_p2=False, near_game_boundary=False,
        pending_p1=0, pending_p2=0, room1=72, room2=72,
    )
    base.update(overrides)
    return JudgmentRecord(**base)  # type: ignore[arg-type]


# ============================
# D0: 主因⇔結論の符号矛盾
# ============================

class TestDetectD0:
    def test_positive_sign_mismatch(self) -> None:
        """主因1位が1P有利(+)なのに adv が2P有利(-) → 検出する。"""
        rec = _record(drivers=(("board_color_puyo_total", 0.67),), adv=-20.0)
        s = detect_d0(rec)
        assert s is not None
        assert s.detector == "D0"
        assert s.severity == "CRITICAL"
        assert "色ぷよ総数差" in s.evidence
        assert "+0.670" in s.evidence

    def test_negative_sign_match(self) -> None:
        """主因1位と adv が同じ向き → 検出しない。"""
        rec = _record(drivers=(("board_color_puyo_total", 0.67),), adv=20.0)
        assert detect_d0(rec) is None

    def test_negative_empty_drivers(self) -> None:
        """主因候補が無い (除外リストで全滅等) → 検出しない。"""
        rec = _record(drivers=())
        assert detect_d0(rec) is None

    def test_boundary_zero_driver_diff(self) -> None:
        """主因の差分がちょうど 0 → 符号を定義できないため検出しない。"""
        rec = _record(drivers=(("board_color_puyo_total", 0.0),), adv=-5.0)
        assert detect_d0(rec) is None

    def test_boundary_zero_adv(self) -> None:
        """adv がちょうど 0 (互角) → 符号を定義できないため検出しない。"""
        rec = _record(drivers=(("board_color_puyo_total", 0.5),), adv=0.0)
        assert detect_d0(rec) is None

    def test_uses_second_driver_ignored(self) -> None:
        """2位以下の符号は無視し、1位のみで判定する。"""
        rec = _record(
            drivers=(("board_color_puyo_total", -0.1), ("max_column_height", 0.9)),
            adv=-5.0,
        )
        # 1位 (-0.1) と adv(-5.0) は同符号 → 矛盾なし
        assert detect_d0(rec) is None


# ============================
# D1a: 確定死の無視
# ============================

class TestDetectD1a:
    def test_positive_dead_1p_favored_by_adv(self) -> None:
        """1P窒息確定なのに adv が1P有利 → 検出する。"""
        rec = _record(is_dead_p1=True, is_dead_p2=False, adv=30.0, p1=0.65)
        suspects = detect_d1a(rec)
        assert len(suspects) == 1
        assert suspects[0].detector == "D1a"
        assert "1P" in suspects[0].evidence

    def test_positive_dead_2p_favored_by_p1_only(self) -> None:
        """2P窒息確定・adv自体は2P有利を示していなくても p1<0.5(2P有利)なら検出する

        (OR条件の p1 分岐を単独で踏むケース。adv/p1 は本来同じ量から出るため
        実運用では乖離しないが、OR ロジック自体の健全性を確認する)。
        """
        rec = _record(is_dead_p1=False, is_dead_p2=True, adv=2.0, p1=0.49)
        suspects = detect_d1a(rec)
        assert len(suspects) == 1
        assert "2P" in suspects[0].evidence

    def test_negative_dead_but_correctly_unfavored(self) -> None:
        """1P窒息確定で adv も1P不利(2P有利) → 正しい判定なので検出しない。"""
        rec = _record(is_dead_p1=True, is_dead_p2=False, adv=-40.0, p1=0.2)
        assert detect_d1a(rec) == []

    def test_negative_near_boundary_suppressed(self) -> None:
        """境界近傍ガード: 矛盾条件を満たしても near_game_boundary=True なら検出しない。"""
        rec = _record(
            is_dead_p1=True, is_dead_p2=False, adv=30.0, p1=0.65,
            near_game_boundary=True,
        )
        assert detect_d1a(rec) == []

    def test_negative_both_alive(self) -> None:
        """両者生存 → 何も検出しない。"""
        rec = _record(is_dead_p1=False, is_dead_p2=False, adv=90.0, p1=0.95)
        assert detect_d1a(rec) == []

    def test_positive_both_dead_both_flagged(self) -> None:
        """両者窒息確定という異常な盤面でも、判定が有利寄りなら両側とも報告する。"""
        rec = _record(is_dead_p1=True, is_dead_p2=True, adv=5.0, p1=0.51)
        suspects = detect_d1a(rec)
        sides = {"1P" if "1P" in s.evidence else "2P" for s in suspects}
        assert "1P" in sides


# ============================
# end-to-end: scan_video (合成 npz + 決定的 stub score_fn)
# ============================

def _dead_grid() -> np.ndarray:
    grid = np.zeros((13, 6), dtype=np.int8)
    grid[DEATH_ROW, DEATH_COL] = 1  # 窒息確定
    return grid


def _alive_grid() -> np.ndarray:
    return np.zeros((13, 6), dtype=np.int8)


def _stub_score_fn(b1: Board, b2: Board):
    """1Pが窒息していれば1P有利、2Pが窒息していれば2P有利を返す決定的スタブ。

    D0(符号矛盾)は起こさず D1a の配線 (npz読込 + per-side ペアリング +
    境界ガード) だけを検証する目的の単純化。
    """
    if b1.is_dead():
        return 60.0, 0.8, [("board_color_puyo_total", 0.3)]
    if b2.is_dead():
        return -60.0, 0.2, [("board_color_puyo_total", -0.3)]
    return 0.0, 0.5, [("board_color_puyo_total", 0.0)]


def _write_synthetic_npz(path: Path) -> None:
    """1P: t=0.0(窒息) / t=10.0(窒息)、2P: t=0.05(生存) の最小 npz を書く。

    ペアリング後の期待レコード:
      - t=0.05 (2P trigger, b1=1P@t0.0(dead), b2=2P@t0.05) → 境界(0.0)から
        0.05秒しか離れておらず GAME_BOUNDARY_GUARD_SEC 以内 → 抑制される
      - t=10.0 (1P trigger, b1=1P@t10.0(dead), b2=2P@t0.05) → 境界から
        10秒離れており抑制されない → D1a 検出 1 件
    """
    grids = np.stack([_dead_grid(), _dead_grid(), _alive_grid()])
    side = np.array(["1P", "1P", "2P"])
    t_sec = np.array([0.0, 10.0, 0.05], dtype=np.float32)
    game_idx = np.array([0, 0, 0], dtype=np.int32)
    np.savez_compressed(
        str(path), grids=grids, side=side, t_sec=t_sec, game_idx=game_idx,
    )


class TestScanVideoEndToEnd:
    def test_boundary_guard_and_detection_wiring(self, tmp_path: Path) -> None:
        assert GAME_BOUNDARY_GUARD_SEC == 2.0  # 前提確認 (指令書の ±2秒)
        npz_path = tmp_path / "v_e2e.npz"
        _write_synthetic_npz(npz_path)

        records = scan_video(npz_path, _stub_score_fn)
        # 2P@0.05 の trigger と 1P@10.0 の trigger の2レコードが作られる
        # (1P@0.0 の trigger は相手2Pの過去snapshotが無くスキップされる)
        assert len(records) == 2

        all_suspects = []
        for rec in records:
            all_suspects.extend(detect_d0(rec) and [detect_d0(rec)] or [])
            all_suspects.extend(detect_d1a(rec))

        d1a_suspects = [s for s in all_suspects if s.detector == "D1a"]
        assert len(d1a_suspects) == 1
        assert d1a_suspects[0].t_sec == pytest.approx(10.0)
        assert d1a_suspects[0].video_id == "v_e2e"
        assert "1P" in d1a_suspects[0].evidence


# ============================
# D1b: 致死確定 (pending/room) の無視
# ============================

class TestDetectD1b:
    def test_positive_1p_certain_death_favored_by_adv(self) -> None:
        """1P: pending/room が KILL_RATIO_FULL 以上 (致死確定) なのに adv が1P有利 → 検出。"""
        rec = _record(
            pending_p1=100, room1=40, pending_p2=0, room2=72,  # 100/40=2.5 >= 1.5
            adv=30.0, p1=0.7,
        )
        suspects = detect_d1b(rec)
        assert len(suspects) == 1
        assert suspects[0].detector == "D1b"
        assert "1P" in suspects[0].evidence

    def test_negative_below_kill_ratio_full_not_flagged(self) -> None:
        """比が KILL_RATIO_FULL 未満 (致死確定ではない) → 検出しない。"""
        rec = _record(
            pending_p1=40, room1=40, pending_p2=0, room2=72,  # 40/40=1.0 < 1.5
            adv=30.0, p1=0.7,
        )
        assert detect_d1b(rec) == []

    def test_negative_correctly_unfavored(self) -> None:
        """致死確定でも adv/p1 が正しく生存側(2P)を favor → 検出しない。"""
        rec = _record(
            pending_p1=100, room1=40, pending_p2=0, room2=72,
            adv=-30.0, p1=0.2,
        )
        assert detect_d1b(rec) == []

    def test_negative_near_boundary_suppressed(self) -> None:
        """D1a と同じ境界ガードが効く。"""
        rec = _record(
            pending_p1=100, room1=40, pending_p2=0, room2=72,
            adv=30.0, p1=0.7, near_game_boundary=True,
        )
        assert detect_d1b(rec) == []

    def test_boundary_ratio_uses_kill_room_floor(self) -> None:
        """room が KILL_ROOM_FLOOR 未満でも 0 除算せず、下限で丸められる
        (kill_override() と同じ規約)。"""
        rec = _record(
            pending_p1=KILL_ROOM_FLOOR * KILL_RATIO_FULL, room1=1, pending_p2=0, room2=72,
            adv=30.0, p1=0.7,
        )
        # room=1 < KILL_ROOM_FLOOR なので分母は KILL_ROOM_FLOOR に丸められ、
        # ratio = KILL_RATIO_FULL ちょうど (>= 判定なので検出される)
        assert len(detect_d1b(rec)) == 1


# ============================
# dump 読み出しモード: scan_video_from_dump (合成タイムラインdump npz)
# ============================

def _dump_row(**overrides: object) -> TimelineDumpRow:
    base: dict[str, object] = dict(
        t_sec=0.0, game_idx=0, adv_raw=0.0, adv_ema=0.0, p1=0.5,
        pending_p1=0, pending_p2=0, room1=72, room2=72,
        is_dead1=False, is_dead2=False,
        drivers_top1_name="board_color_puyo_total", drivers_top1_val=0.0,
        drivers_top3_names=("board_color_puyo_total", "", ""),
        drivers_top3_vals=(0.0, 0.0, 0.0),
        score1=0, score2=0, b1_hash=0, b2_hash=0,
        state1="STABLE", state2="STABLE",
    )
    base.update(overrides)
    return TimelineDumpRow(**base)  # type: ignore[arg-type]


class TestScanVideoFromDumpEndToEnd:
    def test_d0_detected_from_dump_record(self, tmp_path: Path) -> None:
        """dump 由来の adv_raw/drivers から D0 (符号矛盾) を検出できる。"""
        rows = [
            _dump_row(
                t_sec=20.0, game_idx=1,
                drivers_top1_name="board_color_puyo_total", drivers_top1_val=0.67,
                adv_raw=-20.0,
            ),
        ]
        dump_path = tmp_path / "v_dump.npz"
        save_timeline_dump(dump_path, "v_dump", rows)

        records = scan_video_from_dump(dump_path)
        assert len(records) == 1
        assert records[0].video_id == "v_dump"
        suspects = [s for r in records for s in ([detect_d0(r)] if detect_d0(r) else [])]
        assert len(suspects) == 1
        assert suspects[0].detector == "D0"

    def test_d1a_detected_from_dump_record(self, tmp_path: Path) -> None:
        """dump 由来の is_dead1 + p1 (表示用EMA後勝率) から D1a を検出できる
        (scan_video_from_dump は record.p1 <- row.p1 にマッピングする)。
        ゲーム境界ガード (±GAME_BOUNDARY_GUARD_SEC秒) に触れないよう、
        同じ game_idx 内に t=0.0 のアンカー行も加えて境界を離しておく。
        """
        rows = [
            _dump_row(t_sec=0.0, game_idx=2),
            _dump_row(t_sec=50.0, game_idx=2, is_dead1=True, adv_raw=0.0, p1=0.7),
        ]
        dump_path = tmp_path / "v_dump_d1a.npz"
        save_timeline_dump(dump_path, "v_dump_d1a", rows)

        records = scan_video_from_dump(dump_path)
        suspects = [s for r in records for s in detect_d1a(r)]
        assert len(suspects) == 1
        assert "1P" in suspects[0].evidence

    def test_empty_dump_yields_no_records(self, tmp_path: Path) -> None:
        dump_path = tmp_path / "v_dump_empty.npz"
        save_timeline_dump(dump_path, "v_dump_empty", [])
        assert scan_video_from_dump(dump_path) == []
