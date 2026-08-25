"""「ありえない判定」走査器 D0/D1a/D1b + dump 読み出しモードのテスト.

- 純関数 detect_d0/detect_d1a/detect_d1b の陽性/陰性/境界ガード (単体テスト、npz非依存)
- raw/display 分離 (Suspect.stage、2026-08-11 アーキ審査追加) の分岐網羅
- scan_video() を含む end-to-end 1 件 (合成 npz + 決定的な stub score_fn)
- scan_video_from_dump() の end-to-end 1 件 (合成タイムラインdump npz)
- 集計ヘルパー _tally_suspects/_gate_count の単体テスト
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.board import DEATH_COL, DEATH_ROW, Board
from scripts.scan_judgment_anomalies import (
    CHAIN_PHASE_STATE_NAMES,
    GAME_BOUNDARY_GUARD_SEC,
    STAGE_BOTH,
    STAGE_DISPLAY,
    STAGE_RAW_ONLY,
    JudgmentRecord,
    Suspect,
    _gate_count,
    _tally_suspects,
    detect_d0,
    detect_d1a,
    detect_d1b,
    scan_video,
    scan_video_from_dump,
)
from scripts.visualize_advantage_overlay import (
    KILL_MIN_PENDING,
    KILL_RATIO_FULL,
    KILL_ROOM_FLOOR,
    TimelineDumpRow,
    save_timeline_dump,
)


def _record(**overrides: object) -> JudgmentRecord:
    """テスト用デフォルト JudgmentRecord (境界外・生存・矛盾なし・pending/room 安全値)。

    p1_raw/adv_ema を明示指定しなければ p1/adv と同値にする (raw==display の
    「both」ケースとして振る舞う)。2026-08-11 のraw/display分離追加前から
    存在するテストが adv/p1 のみ指定していても結果が変わらないようにするため。
    stage を明示的に検証したいテストは p1_raw/adv_ema を個別に上書きする。
    """
    base: dict[str, object] = dict(
        video_id="v_test", t_sec=100.0, game_idx=0, trigger_side="1P",
        adv=10.0, p1=0.55, drivers=(("board_color_puyo_total", 0.3),),
        is_dead_p1=False, is_dead_p2=False, near_game_boundary=False,
        pending_p1=0, pending_p2=0, room1=72, room2=72,
    )
    base.update(overrides)
    base.setdefault("p1_raw", base["p1"])
    base.setdefault("adv_ema", base["adv"])
    return JudgmentRecord(**base)  # type: ignore[arg-type]


# ============================
# D0: 主因⇔結論の符号矛盾
# ============================

class TestDetectD0:
    def test_positive_sign_mismatch(self) -> None:
        """主因1位が1P有利(+)なのに adv が2P有利(-) → 検出する。stage は D0 固定で raw_only。"""
        rec = _record(drivers=(("board_color_puyo_total", 0.67),), adv=-20.0)
        s = detect_d0(rec)
        assert s is not None
        assert s.detector == "D0"
        assert s.severity == "CRITICAL"
        assert s.stage == STAGE_RAW_ONLY
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

    def test_ignores_display_fields_entirely(self) -> None:
        """D0 は raw (adv/drivers) 固定。adv_ema/p1_raw をどう変えても結果不変
        (kill_override の正当な符号反転を D0 が誤検知しないための設計、
        2026-08-11 アーキ判定)。"""
        rec = _record(
            drivers=(("board_color_puyo_total", 0.67),), adv=-20.0,
            adv_ema=99.0, p1_raw=0.99, p1=0.99,
        )
        s = detect_d0(rec)
        assert s is not None and s.stage == STAGE_RAW_ONLY


# ============================
# D1a: 確定死の無視 (raw/display 分離)
# ============================

class TestDetectD1a:
    def test_positive_dead_1p_favored_by_adv_both_stage(self) -> None:
        """1P窒息確定なのに adv(raw)/adv_ema(display) 双方が1P有利 → both で検出。"""
        rec = _record(is_dead_p1=True, is_dead_p2=False, adv=30.0, p1=0.65)
        suspects = detect_d1a(rec)
        assert len(suspects) == 1
        assert suspects[0].detector == "D1a"
        assert suspects[0].stage == STAGE_BOTH
        assert "1P" in suspects[0].evidence

    def test_positive_dead_2p_favored_by_p1_only(self) -> None:
        """2P窒息確定・adv自体は2P有利を示していなくても p1<0.5(2P有利)なら検出する

        (OR条件の p1 分岐を単独で踏むケース。raw/display とも同値指定のため both)。
        """
        rec = _record(is_dead_p1=False, is_dead_p2=True, adv=2.0, p1=0.49)
        suspects = detect_d1a(rec)
        assert len(suspects) == 1
        assert suspects[0].stage == STAGE_BOTH
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

    def test_negative_chain_phase_own_side_suppressed(self) -> None:
        """1Pが CHAIN 中なら is_dead_p1=True でも保留する (2026-08-23 根治②、
        reference_death_judgment_during_chain_2026-08-22)。"""
        rec = _record(is_dead_p1=True, is_dead_p2=False, adv=30.0, p1=0.65, state1="CHAIN")
        assert detect_d1a(rec) == []

    def test_negative_gravity_settle_own_side_suppressed(self) -> None:
        """GRAVITY_SETTLE も同様に保留対象 (連鎖終了直後の重力settle中)。"""
        rec = _record(
            is_dead_p1=True, is_dead_p2=False, adv=30.0, p1=0.65,
            state1="GRAVITY_SETTLE",
        )
        assert detect_d1a(rec) == []

    def test_negative_chain_guard_is_per_side(self) -> None:
        """連鎖中ガードは判定対象側だけに効く。1Pが連鎖中でも2Pの窒息確定は
        通常通り検出する。"""
        rec = _record(
            is_dead_p1=False, is_dead_p2=True, adv=-2.0, p1=0.51,
            state1="CHAIN", state2="STABLE",
        )
        suspects = detect_d1a(rec)
        assert len(suspects) == 1
        assert "2P" in suspects[0].evidence

    def test_positive_stable_state_not_suppressed(self) -> None:
        """state1="STABLE" (連鎖中でない) は従来通り検出する (回帰防止)。"""
        rec = _record(is_dead_p1=True, is_dead_p2=False, adv=30.0, p1=0.65, state1="STABLE")
        assert len(detect_d1a(rec)) == 1

    def test_positive_default_empty_state_not_suppressed(self) -> None:
        """state 未設定 ("", npz再計算モード相当) はガード非適用=従来通り
        検出する (後方互換、`TestScanVideoEndToEnd` の既存挙動と整合)。"""
        rec = _record(is_dead_p1=True, is_dead_p2=False, adv=30.0, p1=0.65)
        assert rec.state1 == "" and rec.state2 == ""
        assert len(detect_d1a(rec)) == 1

    def test_chain_phase_state_names_content(self) -> None:
        """ガード対象は CHAIN と GRAVITY_SETTLE の2つのみ (それ以外の
        BoardState.name は STABLE と同様に扱われる)。"""
        assert CHAIN_PHASE_STATE_NAMES == frozenset({"CHAIN", "GRAVITY_SETTLE"})

    def test_stage_raw_only_when_display_corrected(self) -> None:
        """raw (adv/p1_raw) は1P窒息を無視して1P有利だが、display (adv_ema/p1) は
        kill_override 等で正しく2P有利に是正されている → raw_only (内部品質バックログ)。
        """
        rec = _record(
            is_dead_p1=True, is_dead_p2=False,
            adv=30.0, p1_raw=0.65,       # raw: 1P有利 (矛盾)
            adv_ema=-30.0, p1=0.35,      # display: 2P有利 (正しい)
        )
        suspects = detect_d1a(rec)
        assert len(suspects) == 1
        assert suspects[0].stage == STAGE_RAW_ONLY
        assert "[raw_only]" in suspects[0].evidence  # stage が evidence にも明記される
        assert "内部品質バックログ" in suspects[0].evidence

    def test_stage_display_only_when_raw_was_fine(self) -> None:
        """raw は正しく2P有利を示すが、display だけが1P有利に矛盾している
        (何らかの後段処理が矛盾を持ち込んだケース) → display (リリースブロッカー)。
        """
        rec = _record(
            is_dead_p1=True, is_dead_p2=False,
            adv=-30.0, p1_raw=0.35,      # raw: 2P有利 (正しい)
            adv_ema=30.0, p1=0.65,       # display: 1P有利 (矛盾)
        )
        suspects = detect_d1a(rec)
        assert len(suspects) == 1
        assert suspects[0].stage == STAGE_DISPLAY
        assert "リリースブロッカー" in suspects[0].evidence


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
        # npz 再計算モードには display 段階が存在しないため raw==display と
        # なり、構造的に必ず both になる (モジュール docstring 参照)。
        assert d1a_suspects[0].stage == STAGE_BOTH


# ============================
# D1b: 致死確定 (pending/room) の無視 (raw/display 分離)
# ============================

class TestDetectD1b:
    def test_positive_1p_certain_death_favored_by_adv_both_stage(self) -> None:
        """1P: pending/room が KILL_RATIO_FULL 以上 (致死確定) なのに adv が1P有利 → 検出。"""
        rec = _record(
            pending_p1=100, room1=40, pending_p2=0, room2=72,  # 100/40=2.5 >= 1.5
            adv=30.0, p1=0.7,
        )
        suspects = detect_d1b(rec)
        assert len(suspects) == 1
        assert suspects[0].detector == "D1b"
        assert suspects[0].stage == STAGE_BOTH
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
        (kill_override() と同じ規約)。pending は KILL_MIN_PENDING (=40) 以上
        (2026-08-23 根治②追加分) を満たす値にする (床下限クランプの検証が
        目的で、下限自体の境界は別テストで確認する)。"""
        rec = _record(
            pending_p1=max(KILL_MIN_PENDING, KILL_ROOM_FLOOR * KILL_RATIO_FULL),
            room1=1, pending_p2=0, room2=72, adv=30.0, p1=0.7,
        )
        # room=1 < KILL_ROOM_FLOOR なので分母は KILL_ROOM_FLOOR に丸められる。
        assert len(detect_d1b(rec)) == 1

    def test_negative_below_kill_min_pending_not_flagged(self) -> None:
        """pending < KILL_MIN_PENDING (=40) は比が KILL_RATIO_FULL 以上でも
        本番 kill_override() 自体が致死扱いしない量のため検出しない
        (2026-08-23 根治②、実測: D1bの96.2%がこのケースだった)。"""
        rec = _record(
            pending_p1=KILL_MIN_PENDING - 1, room1=4, pending_p2=0, room2=72,
            adv=30.0, p1=0.7,  # 比 = 39/4 = 9.75 ≫ KILL_RATIO_FULL
        )
        assert detect_d1b(rec) == []

    def test_positive_at_kill_min_pending_boundary_flagged(self) -> None:
        """pending == KILL_MIN_PENDING ちょうどは境界内 (>= 判定) で検出する。"""
        rec = _record(
            pending_p1=KILL_MIN_PENDING, room1=4, pending_p2=0, room2=72,
            adv=30.0, p1=0.7,
        )
        assert len(detect_d1b(rec)) == 1

    def test_negative_chain_phase_own_side_suppressed(self) -> None:
        """判定対象側 (1P) が CHAIN 中なら、致死確定条件を満たしていても
        検出しない (2026-08-23 根治②、reference_death_judgment_during_chain_
        2026-08-22: 連鎖中は保留)。"""
        rec = _record(
            pending_p1=100, room1=4, pending_p2=0, room2=72,
            adv=30.0, p1=0.7, state1="CHAIN",
        )
        assert detect_d1b(rec) == []

    def test_negative_gravity_settle_own_side_suppressed(self) -> None:
        """GRAVITY_SETTLE も CHAIN と同様に保留対象。"""
        rec = _record(
            pending_p1=100, room1=4, pending_p2=0, room2=72,
            adv=30.0, p1=0.7, state1="GRAVITY_SETTLE",
        )
        assert detect_d1b(rec) == []

    def test_positive_stable_state_not_suppressed(self) -> None:
        """state1="STABLE" (連鎖中でない) は従来通り検出する (回帰防止)。"""
        rec = _record(
            pending_p1=100, room1=4, pending_p2=0, room2=72,
            adv=30.0, p1=0.7, state1="STABLE",
        )
        assert len(detect_d1b(rec)) == 1

    def test_positive_default_empty_state_not_suppressed(self) -> None:
        """state 未設定 ("", npz再計算モード相当) はガード非適用=従来通り
        検出する (後方互換)。"""
        rec = _record(pending_p1=100, room1=4, pending_p2=0, room2=72, adv=30.0, p1=0.7)
        assert rec.state1 == ""  # 前提確認 (既定値)
        assert len(detect_d1b(rec)) == 1

    def test_stage_raw_only_when_kill_override_corrected(self) -> None:
        """raw は致死無視で1P有利だが、display は kill_override で2P有利に
        是正済み → raw_only (内部品質バックログ、kill_override 自体は機能している)。
        """
        rec = _record(
            pending_p1=100, room1=40, pending_p2=0, room2=72,
            adv=30.0, p1_raw=0.7,      # raw: 1P有利 (矛盾)
            adv_ema=-30.0, p1=0.3,     # display: 2P有利 (kill_overrideで是正済み)
        )
        suspects = detect_d1b(rec)
        assert len(suspects) == 1
        assert suspects[0].stage == STAGE_RAW_ONLY

    def test_stage_display_only_when_raw_was_fine(self) -> None:
        """raw は正しいが display だけ矛盾 → display (リリースブロッカー)。"""
        rec = _record(
            pending_p1=100, room1=40, pending_p2=0, room2=72,
            adv=-30.0, p1_raw=0.3,     # raw: 2P有利 (正しい)
            adv_ema=30.0, p1=0.7,      # display: 1P有利 (矛盾)
        )
        suspects = detect_d1b(rec)
        assert len(suspects) == 1
        assert suspects[0].stage == STAGE_DISPLAY


# ============================
# dump 読み出しモード: scan_video_from_dump (合成タイムラインdump npz)
# ============================

def _dump_row(**overrides: object) -> TimelineDumpRow:
    base: dict[str, object] = dict(
        t_sec=0.0, game_idx=0, adv_raw=0.0, adv_ema=0.0, p1=0.5, p1_raw=0.5,
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
        assert suspects[0].stage == STAGE_RAW_ONLY

    def test_d1a_display_only_detected_from_dump_record(self, tmp_path: Path) -> None:
        """dump 由来の is_dead1 + p1 (表示用EMA後勝率) から D1a を検出できる
        (scan_video_from_dump は record.p1 <- row.p1、record.p1_raw <- row.p1_raw
        にマッピングする)。raw (adv_raw/p1_raw の既定値0.0/0.5) は無害寄りだが
        display (p1=0.7) だけが1P有利に矛盾 → stage="display" (リリースブロッカー)。
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
        assert suspects[0].stage == STAGE_DISPLAY

    def test_d1a_raw_only_detected_from_dump_record(self, tmp_path: Path) -> None:
        """raw (p1_raw=0.7) が1P有利に矛盾するが display (p1=0.3) は2P有利に
        正しく是正されている → stage="raw_only" (内部品質バックログ)。
        """
        rows = [
            _dump_row(t_sec=0.0, game_idx=3),
            _dump_row(
                t_sec=50.0, game_idx=3, is_dead1=True,
                adv_raw=30.0, p1_raw=0.7, adv_ema=-30.0, p1=0.3,
            ),
        ]
        dump_path = tmp_path / "v_dump_d1a_raw.npz"
        save_timeline_dump(dump_path, "v_dump_d1a_raw", rows)

        records = scan_video_from_dump(dump_path)
        suspects = [s for r in records for s in detect_d1a(r)]
        assert len(suspects) == 1
        assert suspects[0].stage == STAGE_RAW_ONLY

    def test_empty_dump_yields_no_records(self, tmp_path: Path) -> None:
        dump_path = tmp_path / "v_dump_empty.npz"
        save_timeline_dump(dump_path, "v_dump_empty", [])
        assert scan_video_from_dump(dump_path) == []

    def test_state_wired_into_record_suppresses_d1a_during_chain(
        self, tmp_path: Path,
    ) -> None:
        """(2026-08-23 根治②) dump の state1/state2 が JudgmentRecord に配線
        され、CHAIN 中の is_dead1=True は D1a が保留する。従来は state が
        レコードに存在せず (フィールド自体が無かった)、連鎖中の凍結満杯盤面を
        全部「異常」と数えていた。"""
        rows = [
            _dump_row(t_sec=0.0, game_idx=4),
            _dump_row(
                t_sec=50.0, game_idx=4, is_dead1=True, adv_raw=30.0, p1=0.65,
                state1="CHAIN",
            ),
        ]
        dump_path = tmp_path / "v_dump_chain_guard.npz"
        save_timeline_dump(dump_path, "v_dump_chain_guard", rows)

        records = scan_video_from_dump(dump_path)
        assert records[1].state1 == "CHAIN"
        suspects = [s for r in records for s in detect_d1a(r)]
        assert suspects == []

    def test_kpending_kroom_used_for_d1b_not_raw_values(self, tmp_path: Path) -> None:
        """(2026-08-23 根治①②連携) D1b は kill_override へ実際に渡された
        是正後の値 (kpending_p1/kroom1) を見る。生値だけなら致死確定だが
        是正後は安全な値になっているフレームは検出しない (盲点の根治、
        raw pending/room をそのまま使うと誤検出していたケース)。"""
        rows = [
            _dump_row(t_sec=0.0, game_idx=5),
            _dump_row(
                t_sec=50.0, game_idx=5, adv_raw=30.0, p1=0.7,
                pending_p1=216, room1=5,  # 生値だけなら致死確定 (216/5=43.2)
                kpending_p1=0.0, kroom1=62,  # 是正後は安全 (連鎖完走後の相殺済み)
            ),
        ]
        dump_path = tmp_path / "v_dump_kfields.npz"
        save_timeline_dump(dump_path, "v_dump_kfields", rows)

        records = scan_video_from_dump(dump_path)
        assert records[1].pending_p1 == 0
        assert records[1].room1 == 62
        assert detect_d1b(records[1]) == []


# ============================
# 集計ヘルパー: _tally_suspects / _gate_count
# ============================

def _suspect(detector: str, stage: str) -> Suspect:
    return Suspect(
        video_id="v", t_sec=0.0, game_idx=0, detector=detector,
        severity="CRITICAL", stage=stage, evidence="dummy",
    )


class TestTallyAndGate:
    def test_d0_always_counted_regardless_of_stage(self) -> None:
        """D0 は stage を問わず D0 バケツに数える (D0 は常に raw_only 形式値)。"""
        tally = _tally_suspects([_suspect("D0", STAGE_RAW_ONLY)])
        assert tally["D0"] == 1
        assert _gate_count(tally) == 1

    def test_d1a_raw_only_excluded_from_gate(self) -> None:
        """D1a の raw_only はゲート対象から除外される (内部品質バックログ)。"""
        tally = _tally_suspects([_suspect("D1a", STAGE_RAW_ONLY)])
        assert tally["D1a_raw_only"] == 1
        assert tally["D1a_display"] == 0
        assert _gate_count(tally) == 0

    def test_d1a_display_and_both_counted_in_gate(self) -> None:
        """D1a の display/both はどちらもゲート対象 ("display+both" バケツ)。"""
        tally = _tally_suspects([
            _suspect("D1a", STAGE_DISPLAY), _suspect("D1a", STAGE_BOTH),
        ])
        assert tally["D1a_display"] == 2
        assert _gate_count(tally) == 2

    def test_d1b_raw_only_excluded_from_gate(self) -> None:
        tally = _tally_suspects([_suspect("D1b", STAGE_RAW_ONLY)])
        assert tally["D1b_raw_only"] == 1
        assert _gate_count(tally) == 0

    def test_mixed_tally(self) -> None:
        """D0 1件 + D1a display 1件 + D1a raw_only 1件 + D1b both 1件
        → ゲート対象 = D0(1) + D1a display(1) + D1b display+both(1) = 3。
        raw_only(D1a 1件) はゲートに含まれない。
        """
        suspects = [
            _suspect("D0", STAGE_RAW_ONLY),
            _suspect("D1a", STAGE_DISPLAY),
            _suspect("D1a", STAGE_RAW_ONLY),
            _suspect("D1b", STAGE_BOTH),
        ]
        tally = _tally_suspects(suspects)
        assert tally == {
            "D0": 1, "D1a_display": 1, "D1a_raw_only": 1,
            "D1b_display": 1, "D1b_raw_only": 0,
        }
        assert _gate_count(tally) == 3
