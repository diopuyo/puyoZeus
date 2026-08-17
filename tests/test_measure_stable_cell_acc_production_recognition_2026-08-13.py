"""scripts/measure_stable_cell_acc.py (物差し) の本番構成配線是正 (2026-08-13) テスト。

横展開監査 (docs/CROSS_CUTTING_AUDIT_2026-08-13.md P1) の発見:
物差し (99.5%基準の測定器) がバーストガード等6フラグ
(src.production_config.RECOGNITION_ADOPTED) を手渡し前提にしており、
明示的にフラグを渡さない限り本番より劣化した認識で精度を測っていた。

本テストは2層構成:
  1. `resolve_production_recognition_flags` の純関数テスト (全分岐)。
  2. `main()` レベルの転送値検証 (`_collect_results` をスタブ化し、
     実際に渡された kwargs を検証する。
     tests/test_advantage_overlay_production_recognition_2026-08-13.py と
     同じ「文字列があるかでなく実際に渡された値を検証する」パターン)。

**物差しの継続性**: --no-production-recognition を明示指定すると、過去の
測定 (production_recognition 概念が存在しなかった頃) と bit-identical な
挙動 (各フラグを明示指定しない限り全て無効) を再現できることも検証する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.measure_stable_cell_acc as msca  # noqa: E402
from src.production_config import recognition_load_default_kwargs  # noqa: E402


class _NS:
    """argparse.Namespace の軽量代替 (テスト専用、属性を自由に生やせる)。"""


def _bare_args() -> _NS:
    """6フラグ・no_production_recognition 属性を持たない「素の args」を作る。"""
    a = _NS()
    a.enable_effect_gate = False
    a.enable_burst_guard_v2 = False
    a.enable_transition_merge_guard = False
    a.enable_hidden_row_burst_guard = False
    a.enable_match_transition_debounce = False
    a.burst_gate_open_threshold = None
    return a


class TestResolveProductionRecognitionFlagsUnit:
    """resolve_production_recognition_flags の純関数テスト (全分岐)。"""

    def test_production_on_applies_all_recognition_adopted(self) -> None:
        resolved = msca.resolve_production_recognition_flags(_bare_args(), True)
        expected = recognition_load_default_kwargs()
        assert expected, "RECOGNITION_ADOPTED が空では検証にならない"
        for key, value in expected.items():
            assert resolved[key] == value, f"{key} が本番値に解決されていない"

    def test_production_off_reproduces_legacy_all_disabled(self) -> None:
        """--no-production-recognition 相当: 全フラグが明示指定しない限り無効。"""
        resolved = msca.resolve_production_recognition_flags(_bare_args(), False)
        assert resolved["enable_effect_gate"] is False
        assert resolved["enable_burst_guard_v2"] is False
        assert resolved["enable_transition_merge_guard"] is False
        assert resolved["enable_hidden_row_burst_guard"] is False
        assert resolved["enable_match_transition_debounce"] is False
        assert resolved["burst_gate_open_threshold"] is None

    def test_explicit_cli_flag_wins_even_when_production_off(self) -> None:
        """個別 CLI 明示指定 (--enable-effect-gate) は production OFF でも維持される。"""
        args = _bare_args()
        args.enable_effect_gate = True
        resolved = msca.resolve_production_recognition_flags(args, False)
        assert resolved["enable_effect_gate"] is True
        assert resolved["enable_burst_guard_v2"] is False

    def test_explicit_threshold_overrides_production_value(self) -> None:
        """明示 --burst-gate-open-threshold は production 既定値 (0.954) より優先される。"""
        args = _bare_args()
        args.burst_gate_open_threshold = 0.5
        resolved = msca.resolve_production_recognition_flags(args, True)
        assert resolved["burst_gate_open_threshold"] == 0.5


class TestMainForwardsResolvedFlagsToCollectResults:
    """main() が実際に _collect_results へ渡す kwargs を検証する
    (test_advantage_overlay_production_recognition_2026-08-13.py と同パターン)。"""

    def _run_main_capturing_collect_results(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str],
    ) -> dict:
        captured: list[dict] = []

        def _fake_collect_results(*_args: object, **kwargs: object) -> list:
            captured.append(kwargs)
            return []  # 空リスト = 「処理した動画がゼロ件」経路 (早期return)

        monkeypatch.setattr(msca, "_collect_results", _fake_collect_results)
        monkeypatch.setattr(sys, "argv", argv)
        rc = msca.main()
        assert rc == 2, "動画ゼロ件時の終了コードが変化した (テスト前提が崩れている)"
        assert len(captured) == 1
        return captured[0]

    def test_default_applies_production_recognition(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        kwargs = self._run_main_capturing_collect_results(
            monkeypatch,
            ["measure_stable_cell_acc.py", "--videos", "v99",
             "--output", str(tmp_path / "out.json")],
        )
        expected = recognition_load_default_kwargs()
        for key, value in expected.items():
            assert kwargs[key] == value, f"{key} が既定で本番値に解決されていない"

    def test_no_production_recognition_reproduces_legacy_all_disabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """--no-production-recognition で過去測定と bit-identical な旧構成を再現する。"""
        kwargs = self._run_main_capturing_collect_results(
            monkeypatch,
            ["measure_stable_cell_acc.py", "--videos", "v99",
             "--output", str(tmp_path / "out.json"),
             "--no-production-recognition"],
        )
        assert kwargs["enable_effect_gate"] is False
        assert kwargs["enable_burst_guard_v2"] is False
        assert kwargs["enable_transition_merge_guard"] is False
        assert kwargs["enable_hidden_row_burst_guard"] is False
        assert kwargs["enable_match_transition_debounce"] is False
        assert kwargs["burst_gate_open_threshold"] is None
