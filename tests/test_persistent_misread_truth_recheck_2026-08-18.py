"""W24 (測定器事故10件目) 対策: 持続誤認測定器の真値再検証ロジックのテスト。

`scripts/_diag_persistent_misread_truth_recheck_2026-08-18.py` の
`truth_stable_bounds` / `_confidence_tier` / `_recheck_entry` / `recheck_tag`
を検証する。

対象データ (`data/indicators_v2/`, `data/verify/`) は `.gitignore` の
`data/*` により git管理外 — 新規clone環境ではファイルが存在しない。
そのためコア判定ロジックは合成データのみで検証し (fail-silent警戒:
実データ依存だと新規環境で黙ってテストが無意味化する)、実データが手元に
存在する場合の回帰確認は既知事案 (c21 r4c0、W24事故の実例) をピンする
統合テストとして別途 skip 可能な形で用意する。
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

_MODULE_NAME = "scripts._diag_persistent_misread_truth_recheck_2026-08-18"
_ROOT = Path(__file__).resolve().parent.parent


def _mod():
    """テスト対象モジュールをインポートして返す (ファイル名にハイフン+日付を
    含むため import 文でなく importlib 経由、既存流儀に合わせる)。"""
    return importlib.import_module(_MODULE_NAME)


def _row(
    frame_idx: int, t_sec: float, tsumo_count: int, grid_val: int, r: int = 0, c: int = 0,
) -> dict:
    """`_find_run` / `truth_stable_bounds` が要求する最小限のrow辞書を作る。"""
    grid = np.zeros((13, 6), dtype=np.int8)
    grid[r, c] = grid_val
    return {"frame_idx": frame_idx, "t_sec": t_sec, "grid": grid, "tsumo_count": tsumo_count}


# ============================
# truth_stable_bounds のテスト (物理時計 tsumo_count による区間再検証)
# ============================


class TestTruthStableBounds:
    """区間 [lo, hi] のうち tsumo_count がアンカーと一致する部分を抽出する
    コアロジックの検証。"""

    def test_no_transition_keeps_full_range(self) -> None:
        """tsumo_countが区間中一切変化しなければ、旧測定器と完全に同じ
        区間を返す (bit-identical、退行なし)。"""
        m = _mod()
        rows = [_row(i, i * 0.1, tsumo_count=7, grid_val=9) for i in range(10)]
        stable_lo, stable_hi, n = m.truth_stable_bounds(rows, lo=0, hi=9, anchor_idx=4)
        assert (stable_lo, stable_hi, n) == (0, 9, 0)

    def test_single_transition_after_anchor_truncates_forward(self) -> None:
        """アンカー直後にツモ設置が起きた場合、それ以降を信用区間から除く。"""
        m = _mod()
        rows = [_row(i, i * 0.1, tsumo_count=(7 if i <= 2 else 8), grid_val=9) for i in range(10)]
        stable_lo, stable_hi, n = m.truth_stable_bounds(rows, lo=0, hi=9, anchor_idx=2)
        assert stable_hi == 2
        assert stable_lo == 0
        assert n == 1

    def test_transition_before_anchor_truncates_backward(self) -> None:
        """アンカー手前で設置が起きた場合、その手前を信用区間から除く。"""
        m = _mod()
        rows = [_row(i, i * 0.1, tsumo_count=(3 if i < 4 else 4), grid_val=9) for i in range(10)]
        stable_lo, stable_hi, n = m.truth_stable_bounds(rows, lo=0, hi=9, anchor_idx=6)
        assert stable_lo == 4
        assert stable_hi == 9
        assert n == 1

    def test_multiple_transitions_counted(self) -> None:
        """複数回の設置イベントは全て遷移回数として数える (c21実測: 8〜19回)。"""
        m = _mod()
        counts = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        rows = [_row(i, i * 0.1, tsumo_count=counts[i], grid_val=9) for i in range(10)]
        _, _, n = m.truth_stable_bounds(rows, lo=0, hi=9, anchor_idx=4)
        assert n == 4

    def test_anchor_at_boundary_does_not_crash(self) -> None:
        """アンカーが区間の端 (lo または hi) にある境界条件を確認する。"""
        m = _mod()
        rows = [_row(i, i * 0.1, tsumo_count=7, grid_val=9) for i in range(5)]
        stable_lo, stable_hi, n = m.truth_stable_bounds(rows, lo=0, hi=4, anchor_idx=0)
        assert (stable_lo, stable_hi, n) == (0, 4, 0)
        stable_lo, stable_hi, n = m.truth_stable_bounds(rows, lo=0, hi=4, anchor_idx=4)
        assert (stable_lo, stable_hi, n) == (0, 4, 0)


# ============================
# _confidence_tier のテスト
# ============================


class TestConfidenceTier:
    """遷移回数から確信度タグへの写像 (除外可否とは独立な付加情報)。"""

    def test_zero_transitions_is_no_transition(self) -> None:
        m = _mod()
        assert m._confidence_tier(0) == "no_transition"

    def test_below_high_confidence_threshold_is_ambiguous(self) -> None:
        m = _mod()
        assert m._confidence_tier(1) == "ambiguous"
        assert m._confidence_tier(m.HIGH_CONFIDENCE_TRANSITION_COUNT - 1) == "ambiguous"

    def test_at_or_above_threshold_is_high_confidence(self) -> None:
        m = _mod()
        assert m._confidence_tier(m.HIGH_CONFIDENCE_TRANSITION_COUNT) == "high_confidence_artifact"
        assert m._confidence_tier(m.HIGH_CONFIDENCE_TRANSITION_COUNT + 10) == "high_confidence_artifact"


# ============================
# _recheck_entry のテスト (1セル分の再判定)
# ============================


class TestRecheckEntry:
    """旧エントリ1件に truth-recheck 情報を付与する関数の検証。"""

    def test_no_transition_is_bit_identical_to_old(self) -> None:
        """物理イベントが一切なければ、新測定器は旧測定器と同じ値を返す。"""
        m = _mod()
        rows = [_row(i, i * 0.1, tsumo_count=7, grid_val=9) for i in range(10)]
        run = m._persist._find_run(rows, anchor_idx=4, r=0, c=0)
        entry = {
            "sheet_id": "s", "video_id": "v", "side": "1P", "r": 0, "c": 0,
            "wrong_value": 9, "correct_value": 0,
            "frames_equiv": run["frames_equiv"], "duration_sec": run["duration_sec"],
            "boundary_censored": run["boundary_censored"],
            "prev_val": run["prev_val"], "next_val": run["next_val"],
        }
        out = m._recheck_entry(rows, anchor_idx=4, r=0, c=0, entry=entry)
        assert out["n_tsumo_transitions_in_original_run"] == 0
        assert out["truth_may_have_changed"] is False
        assert out["reclassified_non_persistent"] is False
        assert out["truth_stable_frames_equiv"] == out["original_frames_equiv"]
        assert out["confidence_tier"] == "no_transition"

    def test_transition_at_run_start_forces_reclassification(self) -> None:
        """アンカー直後にツモ設置イベントがある区間は、信用できる幅が
        ほぼ0になり persistent 判定から除外される (W24型の再現)。"""
        m = _mod()
        rows = [_row(i, i * 0.1, tsumo_count=(7 if i == 0 else 8), grid_val=9) for i in range(10)]
        entry = {
            "sheet_id": "s", "video_id": "v", "side": "1P", "r": 0, "c": 0,
            "wrong_value": 9, "correct_value": 0,
            "frames_equiv": 270.0, "duration_sec": 0.9,
            "boundary_censored": False, "prev_val": None, "next_val": None,
        }
        out = m._recheck_entry(rows, anchor_idx=0, r=0, c=0, entry=entry)
        assert out["n_tsumo_transitions_in_original_run"] == 1
        assert out["truth_may_have_changed"] is True
        assert out["truth_stable_frames_equiv"] == 0.0
        assert out["reclassified_non_persistent"] is True
        assert out["confidence_tier"] == "ambiguous"
        # 旧値は保持され、比較可能である (新旧を並べられる形にする、の要件)
        assert out["original_frames_equiv"] == 270.0
        assert out["original_duration_sec"] == 0.9


# ============================
# recheck_tag のend-to-endテスト (npz + score.json + 旧集計jsonを合成データで再現)
# ============================


class TestRecheckTagEndToEnd:
    """`recheck_tag` が npz/score.json/旧集計jsonを正しく突合し、旧集計を
    一切書き換えずに新集計を返すことを検証する。"""

    def _build_synthetic_inputs(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        npz_dir = tmp_path / "npz"
        scoring_dir = tmp_path / "scoring"
        out_dir = tmp_path / "out"
        npz_dir.mkdir()
        scoring_dir.mkdir()
        out_dir.mkdir()

        n = 10
        t_sec = (np.arange(n, dtype=np.float32)) * 0.1
        frame_idx = np.arange(n, dtype=np.int32)

        # series A ("1P"): tsumo_count不変 -> 再分類されないはず
        tsumo_a = np.full(n, 7, dtype=np.int32)
        # series B ("2P"): idx1以降でツモ設置イベント -> 再分類されるはず
        tsumo_b = np.array([7] + [8] * (n - 1), dtype=np.int32)

        grids = np.zeros((2 * n, 13, 6), dtype=np.int8)
        grids[:n, 1, 1] = 9  # series A の対象セル、区間全体で同じ誤り値
        grids[n:, 2, 2] = 9  # series B の対象セル、区間全体で同じ誤り値

        np.savez(
            npz_dir / "chunk0.npz",
            grids=grids,
            video_id=np.array(["vX"] * (2 * n)),
            side=np.array(["1P"] * n + ["2P"] * n),
            frame_idx=np.concatenate([frame_idx, frame_idx]),
            t_sec=np.concatenate([t_sec, t_sec]),
            tsumo_count=np.concatenate([tsumo_a, tsumo_b]),
        )

        import json

        score_rows = [
            {
                "sheet_id": "sheet_A", "video_id": "vX", "side": "1P",
                "match_method": "exact", "npz": "chunk0.npz", "frame_idx": 4, "t_sec": 0.4,
            },
            {
                "sheet_id": "sheet_B", "video_id": "vX", "side": "2P",
                "match_method": "exact", "npz": "chunk0.npz", "frame_idx": 0, "t_sec": 0.0,
            },
        ]
        (scoring_dir / "score_a.json").write_text(
            json.dumps(score_rows, ensure_ascii=False), encoding="utf-8",
        )

        old_result = {
            "tag": "a", "n_wrong_cells_total": 2, "n_wrong_cells_analyzable": 2,
            "n_persistent": 2, "n_reflection_delay": 0,
            "persistent_cells": [
                {
                    "sheet_id": "sheet_A", "video_id": "vX", "side": "1P", "r": 1, "c": 1,
                    "wrong_value": 9, "correct_value": 0,
                    "frames_equiv": 27.0, "duration_sec": 0.9,
                    "boundary_censored": False, "prev_val": None, "next_val": None,
                },
                {
                    "sheet_id": "sheet_B", "video_id": "vX", "side": "2P", "r": 2, "c": 2,
                    "wrong_value": 9, "correct_value": 0,
                    "frames_equiv": 27.0, "duration_sec": 0.9,
                    "boundary_censored": False, "prev_val": None, "next_val": None,
                },
            ],
            "reflection_delay_cells": [],
        }
        (out_dir / "persistent_misread_a.json").write_text(
            json.dumps(old_result, ensure_ascii=False), encoding="utf-8",
        )
        return npz_dir, scoring_dir, out_dir

    def test_no_transition_cell_stays_persistent_transition_cell_reclassified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        m = _mod()
        npz_dir, scoring_dir, out_dir = self._build_synthetic_inputs(tmp_path)
        monkeypatch.setattr(m, "OUT_DIR", out_dir)
        monkeypatch.setattr(m._persist, "SCORING_DIR", scoring_dir)
        monkeypatch.setitem(m._persist.NPZ_DIRS, "a", npz_dir)

        old_bytes_before = (out_dir / "persistent_misread_a.json").read_bytes()
        result = m.recheck_tag("a")
        old_bytes_after = (out_dir / "persistent_misread_a.json").read_bytes()

        # 旧集計ファイルは一切変更されない (新旧比較のための必須要件)
        assert old_bytes_before == old_bytes_after

        assert result["n_persistent_original"] == 2
        assert result["n_persistent_truth_verified"] == 1
        assert result["n_reclassified_non_persistent"] == 1

        by_sheet = {e["sheet_id"]: e for e in result["cells"]}
        cell_a = by_sheet["sheet_A"]
        cell_b = by_sheet["sheet_B"]

        assert cell_a["reclassified_non_persistent"] is False
        assert cell_a["confidence_tier"] == "no_transition"
        assert cell_a["truth_stable_frames_equiv"] == cell_a["original_frames_equiv"]

        assert cell_b["reclassified_non_persistent"] is True
        assert cell_b["n_tsumo_transitions_in_original_run"] == 1
        assert cell_b["truth_stable_frames_equiv"] == 0.0

    def test_missing_old_result_raises_clear_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """旧集計ファイルが無い場合はfail-silentにせず例外で明示する。"""
        m = _mod()
        monkeypatch.setattr(m, "OUT_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            m.recheck_tag("a")


# ============================
# 実データがある場合の回帰確認 (既知事案 004_c21_1P_f144486 r4c0 のピン)
# ============================


_REAL_NPZ_DIR = _ROOT / "data" / "indicators_v2" / "yardstick_v2_boards_f_2026-08-17"
_REAL_OLD_JSON = _ROOT / "data" / "verify" / "recognition_unified_2026-08-17" / "persistent_misread_f.json"
_REAL_SCORE_JSON = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14" / "scoring_ablation" / "score_f.json"


@pytest.mark.skipif(
    not (_REAL_NPZ_DIR.exists() and _REAL_OLD_JSON.exists() and _REAL_SCORE_JSON.exists()),
    reason="実データ (data/配下、git管理外) がこの環境に無いためskip",
)
class TestRealDataRegressionC21R4C0:
    """W24の実例 (004_c21_1P_f144486 r4c0) が、真値再検証で高確信の除外候補に
    なることを確認する (docs/KNOWN_WEAKNESSES.md W24節に記載の事案そのもの)。
    """

    def test_c21_r4c0_is_reclassified_as_high_confidence_artifact(self) -> None:
        m = _mod()
        result = m.recheck_tag("f")
        target = next(
            (e for e in result["cells"] if e["sheet_id"] == "004_c21_1P_f144486" and e["r"] == 4 and e["c"] == 0),
            None,
        )
        assert target is not None, "既知事案のセルが persistent_misread_f.json に見当たらない"
        assert target["reclassified_non_persistent"] is True
        assert target["confidence_tier"] == "high_confidence_artifact"
        assert target["n_tsumo_transitions_in_original_run"] >= m.HIGH_CONFIDENCE_TRANSITION_COUNT
