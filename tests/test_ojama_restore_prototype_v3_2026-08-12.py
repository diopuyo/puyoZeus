"""scripts/_ojama_restore_prototype_2026-08-12.py の v3 (tsumo_count 増分ゲート)
のユニットテスト。

v2 (次ネクストペア変化ゲート) は65動画検証で pooled相関0.33〜0.38 と不合格
だった。根本原因は「dedup済み STABLE snapshot は1着地に対応しない」
(video_c11実測)。v3 は RecognitionPipeline.tsumo_count(side) の増分を着地
イベントのゲートに使う (NextDetector 非依存)。

実データでの精度検証は tsumo_count 列入り npz が1本できてから行う
(後続タスク)。本テストはユニットテストレベルの動作確認までを対象とする。
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

# ファイル名にハイフン+日付が含まれるため import 文でなく importlib で読む
# (scripts/collect_boards_lean.py 等の通常モジュールとは異なる制約)。
_MODULE_NAME = "scripts._ojama_restore_prototype_2026-08-12"


def _import_proto():
    """プロトタイプモジュールをインポートして返す。"""
    return importlib.import_module(_MODULE_NAME)


# ============================
# _is_landing_by_tsumo_count のテスト
# ============================


class TestIsLandingByTsumoCount:
    """着地イベント判定ゲートの境界条件を検証する。"""

    def test_first_observation_is_not_landing(self) -> None:
        """初回観測 (last_tsumo_count=None) は着地と判定しない。"""
        mod = _import_proto()
        assert not mod._is_landing_by_tsumo_count(0, None)
        assert not mod._is_landing_by_tsumo_count(3, None)

    def test_increment_is_landing(self) -> None:
        """tsumo_count が増分していれば着地と判定する。"""
        mod = _import_proto()
        assert mod._is_landing_by_tsumo_count(4, 3)

    def test_unchanged_is_not_landing(self) -> None:
        """tsumo_count が不変 (dedup snapshot の巻き込み) なら着地ではない。

        v2 の弱点 (次ネクスト不変でも grid だけ変化する行を誤って着地扱い
        していた) を再発させないための核心テスト。
        """
        mod = _import_proto()
        assert not mod._is_landing_by_tsumo_count(3, 3)

    def test_unknown_sentinel_is_not_landing(self) -> None:
        """TSUMO_COUNT_UNKNOWN (-1) は未取得として着地でないと判定する。"""
        mod = _import_proto()
        assert not mod._is_landing_by_tsumo_count(mod.TSUMO_COUNT_UNKNOWN, 3)

    def test_none_is_not_landing(self) -> None:
        """tsumo_count=None (旧世代npz・列欠損) は着地でないと判定する。"""
        mod = _import_proto()
        assert not mod._is_landing_by_tsumo_count(None, 3)

    def test_decrement_is_not_landing(self) -> None:
        """tsumo_count が減少 (試合境界後の再カウント想定) は着地ではない。"""
        mod = _import_proto()
        assert not mod._is_landing_by_tsumo_count(0, 5)


# ============================
# reconstruct_ojama_sequence_tsumo_count_gate のテスト
# ============================


class TestReconstructOjamaSequenceTsumoCountGate:
    """v3 復元シミュレータの主要挙動を検証する。"""

    def test_dedup_snapshots_without_landing_do_not_drain(self) -> None:
        """tsumo_count 不変の連続 snapshot は drain/chain を発生させない。

        video_c11 実測 (t=277.8-282.1秒、next1不変のまま9行連続で grid だけ
        変化) を模した回帰テスト。v2 はここで誤って毎回 drain していた。
        """
        mod = _import_proto()
        # 1P: tsumo_count=2 で score不変のまま3回 snapshot (dedup巻き込み想定)
        events = [
            (0.0, "1P", 1000, 1),
            (1.0, "1P", 1000, 2),  # 最初の着地 (増分だが直前は初回なので着地扱い外)
            (2.0, "1P", 1000, 2),
            (3.0, "1P", 1000, 2),
            (4.0, "1P", 1000, 2),
        ]
        rows, diag = mod.reconstruct_ojama_sequence_tsumo_count_gate(events)
        # tsumo_count=2 への遷移が唯一の着地 (score変化なし=plain settle 1回のみ)
        assert diag.plain_settle_count == 1
        assert diag.chain_count == 0
        assert len(rows) == 5

    def test_score_jump_on_landing_triggers_chain(self) -> None:
        """着地イベントと同時に score が跳ねたら chain 会計が発火する。"""
        mod = _import_proto()
        events = [
            (0.0, "1P", 1000, 1),
            (1.0, "1P", 1000, 2),  # tsumo_count 不変なので着地でない (最初の観測)
            (60.0, "1P", 5000, 3),  # tsumo_count 増分 かつ score +4000 → chain
        ]
        rows, diag = mod.reconstruct_ojama_sequence_tsumo_count_gate(events)
        assert diag.chain_count == 1
        # 相手 (2P) の forecast が増えているはず (score差分から生成)
        assert rows[-1]["pred_ojama_forecast_raw"] >= 0.0

    def test_score_reset_boundary_resets_side_state(self) -> None:
        """score が SCORE_RESET_THRESHOLD 以上減少したら leftover/forecast をリセットする。"""
        mod = _import_proto()
        events = [
            (0.0, "1P", 1000, 1),
            (1.0, "1P", 1000, 2),
            (60.0, "1P", 5000, 3),  # chain 発生 → leftover/forecast が変化
            (61.0, "1P", 100, 3),   # score 大幅減少 (試合境界) → リセット
        ]
        rows, diag = mod.reconstruct_ojama_sequence_tsumo_count_gate(events)
        assert diag.reset_count == 1
        # リセット直後の行は forecast=0 に戻っているはず
        assert rows[-1]["pred_ojama_forecast_raw"] == 0.0

    def test_negative_delta_without_reset_is_flagged(self) -> None:
        """score が僅かに減少 (リセット未満) しつつ着地した場合は negative_delta として記録する。"""
        mod = _import_proto()
        events = [
            (0.0, "1P", 1000, 1),
            (1.0, "1P", 1000, 2),
            (2.0, "1P", 990, 3),  # 減少幅10 < SCORE_RESET_THRESHOLD、着地あり
        ]
        rows, diag = mod.reconstruct_ojama_sequence_tsumo_count_gate(events)
        assert diag.negative_delta_count == 1
        assert diag.reset_count == 0
        assert len(rows) == 3

    def test_unknown_tsumo_count_rows_never_trigger_landing(self) -> None:
        """tsumo_count が全行 None (未取得) の場合は着地イベントが1件も出ない。

        load_npz_events_tsumo_count_gate が旧世代npz (列欠損) を読んだ場合の
        安全側フォールバックを模す。
        """
        mod = _import_proto()
        events = [
            (0.0, "1P", 1000, None),
            (1.0, "1P", 5000, None),
            (2.0, "1P", 9000, None),
        ]
        rows, diag = mod.reconstruct_ojama_sequence_tsumo_count_gate(events)
        assert diag.chain_count == 0
        assert diag.plain_settle_count == 0
        assert diag.negative_delta_count == 0
        assert len(rows) == 3

    def test_two_sides_independent_state(self) -> None:
        """1P/2P の状態が独立に管理されること (相手側への繰越以外は混ざらない)。"""
        mod = _import_proto()
        events = [
            (0.0, "1P", 1000, 1),
            (0.0, "2P", 1000, 1),
            (1.0, "1P", 1000, 2),
            (1.0, "2P", 1000, 2),
            (60.0, "1P", 6000, 3),  # 1P だけ chain 発生
        ]
        rows, diag = mod.reconstruct_ojama_sequence_tsumo_count_gate(events)
        assert diag.chain_count == 1
        p1_rows = [r for r in rows if r["side"] == "1P"]
        p2_rows = [r for r in rows if r["side"] == "2P"]
        assert len(p1_rows) == 3
        assert len(p2_rows) == 2


# ============================
# load_npz_events_tsumo_count_gate のテスト
# ============================


class TestLoadNpzEventsTsumoCountGate:
    """npz → イベント列 読み込みの後方互換・欠損時挙動を検証する。"""

    def test_reads_tsumo_count_column(self, tmp_path: Path) -> None:
        """tsumo_count 列を正しく読み出せること。"""
        mod = _import_proto()
        out = tmp_path / "v3_test.npz"
        np.savez_compressed(
            str(out),
            video_id=np.array(["v29", "v29"]),
            t_sec=np.array([0.0, 1.0], dtype=np.float32),
            side=np.array(["1P", "1P"]),
            score=np.array([1000, 5000], dtype=np.int32),
            tsumo_count=np.array([1, 2], dtype=np.int32),
        )
        by_video = mod.load_npz_events_tsumo_count_gate(out)
        events = by_video["v29"]
        assert events[0] == (0.0, "1P", 1000, 1)
        assert events[1] == (1.0, "1P", 5000, 2)

    def test_unknown_sentinel_becomes_none(self, tmp_path: Path) -> None:
        """tsumo_count=-1 (TSUMO_COUNT_UNKNOWN) は None に変換されること。"""
        mod = _import_proto()
        out = tmp_path / "v3_unknown.npz"
        np.savez_compressed(
            str(out),
            video_id=np.array(["v29"]),
            t_sec=np.array([0.0], dtype=np.float32),
            side=np.array(["1P"]),
            score=np.array([1000], dtype=np.int32),
            tsumo_count=np.array([-1], dtype=np.int32),
        )
        by_video = mod.load_npz_events_tsumo_count_gate(out)
        assert by_video["v29"][0][3] is None

    def test_missing_tsumo_count_column_falls_back_to_none(
        self, tmp_path: Path,
    ) -> None:
        """tsumo_count 列が存在しない (旧世代収集) npz でも全行 None で読める
        こと (後方互換: 例外を出さない)。
        """
        mod = _import_proto()
        out = tmp_path / "v3_legacy.npz"
        np.savez_compressed(
            str(out),
            video_id=np.array(["v29"]),
            t_sec=np.array([0.0], dtype=np.float32),
            side=np.array(["1P"]),
            score=np.array([1000], dtype=np.int32),
        )
        by_video = mod.load_npz_events_tsumo_count_gate(out)
        assert by_video["v29"][0][3] is None

    def test_score_none_sentinel_becomes_none(self, tmp_path: Path) -> None:
        """score=-1 (SCORE_NONE_SENTINEL) は None に変換されること
        (load_npz_events と同じ挙動)。
        """
        mod = _import_proto()
        out = tmp_path / "v3_score_none.npz"
        np.savez_compressed(
            str(out),
            video_id=np.array(["v29"]),
            t_sec=np.array([0.0], dtype=np.float32),
            side=np.array(["1P"]),
            score=np.array([-1], dtype=np.int32),
            tsumo_count=np.array([0], dtype=np.int32),
        )
        by_video = mod.load_npz_events_tsumo_count_gate(out)
        assert by_video["v29"][0][2] is None

    def test_events_sorted_by_t_sec(self, tmp_path: Path) -> None:
        """入力が t_sec 順でなくても出力は昇順ソートされること。"""
        mod = _import_proto()
        out = tmp_path / "v3_unsorted.npz"
        np.savez_compressed(
            str(out),
            video_id=np.array(["v29", "v29"]),
            t_sec=np.array([5.0, 1.0], dtype=np.float32),
            side=np.array(["1P", "1P"]),
            score=np.array([2000, 1000], dtype=np.int32),
            tsumo_count=np.array([2, 1], dtype=np.int32),
        )
        by_video = mod.load_npz_events_tsumo_count_gate(out)
        t_secs = [e[0] for e in by_video["v29"]]
        assert t_secs == sorted(t_secs)


# ============================
# collect_boards_lean.py で追加された tsumo_count 列との結合確認
# ============================


class TestIntegrationWithCollectBoardsLeanOutput:
    """collect_boards_lean.py が出力する npz (tsumo_count 列あり) を
    v3 ローダで読めることを確認する (2026-08-12 追加列との結合テスト)。
    """

    def test_lean_npz_with_tsumo_count_loadable_by_v3_loader(
        self, tmp_path: Path,
    ) -> None:
        """collect_boards_lean._LeanNpzAccumulator の出力を v3 ローダで
        読み込めること。
        """
        import scripts.collect_boards_lean as lean_mod
        from src.board import BOARD_COLS, BOARD_ROWS, COLOR_RED
        from src.board import Board

        mod = _import_proto()

        def _make_board() -> Board:
            g = [[0] * BOARD_COLS for _ in range(BOARD_ROWS)]
            for col in range(BOARD_COLS):
                g[BOARD_ROWS - 1][col] = COLOR_RED
            return Board.from_list(g)

        acc = lean_mod._LeanNpzAccumulator()
        acc.append(
            _make_board()._grid, "v99", "1P", 1.0, 0, 10,
            score=1000, tsumo_count=1,
        )
        acc.append(
            _make_board()._grid, "v99", "1P", 2.0, 0, 20,
            score=5000, tsumo_count=2,
        )
        out = tmp_path / "lean_with_tsumo_count.npz"
        acc.save(out)

        by_video = mod.load_npz_events_tsumo_count_gate(out)
        events = by_video["v99"]
        assert events[0] == (1.0, "1P", 1000, 1)
        assert events[1] == (2.0, "1P", 5000, 2)
