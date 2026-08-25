"""本番構成の一元管理を守るための回帰テスト (2026-08-08)。

## なぜこのテストが必要か
2026-08-08 に「有利不利が以前より劣化した」という退行が起きた。 真因は
`--early-fire-reaction` (2026-07-29 の user レビュー指摘に対応して実装済み)
を **デモ生成時に付け忘れた**こと。 改善が「フラグ追加 + 既定 OFF」で入る
規約のため、 付け忘れると機能が存在しないのと同じになる。

このテストは **採用済みフラグが実在すること** と
**デモ/本番の生成スクリプトが採用済みフラグを漏らしていないこと** を機械的に
確認し、 同じ退行の再発を防ぐ。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.production_config import (
    ADVANTAGE_ADOPTED,
    COLLECT_ONLY_ADOPTED,
    COUNTER_REACH_ENABLED_BY_DEFAULT,
    INDICATOR_REORG_DECISIONS,
    LEARNING_DATA_BUILD_ADOPTED,
    RECOGNITION_ADOPTED,
    VISUALIZATION_ADOPTED,
    advantage_overlay_flags,
    describe,
    recognition_flags,
    visualization_flags,
)

_ROOT = Path(__file__).resolve().parent.parent


def _flag_name(flag: str) -> str:
    """'--name value' から '--name' だけ取り出す。"""
    return flag.split()[0]


def _script_text(rel: str) -> str:
    p = _ROOT / rel
    return p.read_text(encoding="utf-8") if p.exists() else ""


class TestAdoptedFlagsAreReal:
    """採用済みとして登録したフラグが、実際に受け付けられること。"""

    @pytest.mark.parametrize("flag", [f.flag for f in ADVANTAGE_ADOPTED])
    def test_advantage_flag_exists_in_script(self, flag: str) -> None:
        """有利不利オーバーレイが実際にそのフラグを定義していること。"""
        text = _script_text("scripts/visualize_advantage_overlay.py")
        assert text, "visualize_advantage_overlay.py が見つからない"
        assert _flag_name(flag) in text, f"{flag} がスクリプトに存在しない"

    @pytest.mark.parametrize("flag", [f.flag for f in VISUALIZATION_ADOPTED])
    def test_visualization_flag_exists_in_script(self, flag: str) -> None:
        """認識オーバーレイが実際にそのフラグを定義していること。"""
        text = _script_text("scripts/visualize_recognition.py")
        assert text, "visualize_recognition.py が見つからない"
        assert _flag_name(flag) in text, f"{flag} がスクリプトに存在しない"

    @pytest.mark.parametrize(
        "flag", [f.flag for f in RECOGNITION_ADOPTED + COLLECT_ONLY_ADOPTED])
    def test_recognition_flag_exists_in_collect(self, flag: str) -> None:
        """収集スクリプトが実際にそのフラグを定義していること。"""
        text = _script_text("scripts/collect_boards_lean.py")
        assert text, "collect_boards_lean.py が見つからない"
        assert _flag_name(flag) in text, f"{flag} がスクリプトに存在しない"

    @pytest.mark.parametrize("flag", [f.flag for f in RECOGNITION_ADOPTED])
    def test_common_flag_also_exists_in_visualizer(self, flag: str) -> None:
        """共通フラグは表示スクリプトでも受け付けられること。

        ここが分かれていないと、収集専用フラグを表示側に渡して
        「unrecognized arguments」で落ちる (2026-08-08 に実際に踏んだ)。
        """
        text = _script_text("scripts/visualize_recognition.py")
        assert _flag_name(flag) in text, f"{flag} が表示側に存在しない"


class TestAdoptedFlagsHaveProvenance:
    """採用の根拠が必ず記録されていること (後から辿れるようにするため)。"""

    @pytest.mark.parametrize(
        "flag",
        list(RECOGNITION_ADOPTED) + list(COLLECT_ONLY_ADOPTED)
        + list(ADVANTAGE_ADOPTED) + list(VISUALIZATION_ADOPTED),
        ids=lambda f: f.flag,
    )
    def test_has_date_and_reason(self, flag) -> None:
        """採用日 (YYYY-MM-DD) と根拠が空でないこと。"""
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", flag.adopted), flag.flag
        assert len(flag.reason) >= 10, f"{flag.flag} の根拠が薄い"


class TestGeneratorsUseProductionFlags:
    """デモ生成スクリプトが採用済みフラグを漏らしていないこと。

    ここが退行の実際の入口だった。 生成スクリプトを新しく作るたびに手で
    フラグを並べると必ず抜けるため、 機械的に突き合わせる。
    """

    def test_demo_generator_pulls_flags_from_production_config(self) -> None:
        """最終デモの生成スクリプトが production_config からフラグを取得すること。

        フラグを直書きすると採用漏れが起きる (これが 2026-08-08 の退行の
        入口だった)。 よって「直書きしていないこと」自体をテストする。
        """
        for rel in (
            "scripts/_gen_demo_final_2026-08-08.sh",
            "scripts/_gen_demo_final_cd_2026-08-08.sh",
        ):
            text = _script_text(rel)
            assert text, f"{rel} が見つからない"
            assert "production_config" in text, (
                f"{rel} が production_config を参照していない (直書きの疑い)"
            )

    def test_helpers_return_all_flags(self) -> None:
        """ヘルパ関数が登録済みフラグを全て返すこと。"""
        for flags, fn in (
            (RECOGNITION_ADOPTED, recognition_flags),
            (ADVANTAGE_ADOPTED, advantage_overlay_flags),
            (VISUALIZATION_ADOPTED, visualization_flags),
        ):
            s = fn()
            for f in flags:
                assert _flag_name(f.flag) in s

    def test_describe_lists_everything(self) -> None:
        """describe() が全フラグを列挙すること (生成物への記録用)。"""
        text = describe()
        for f in (
            list(RECOGNITION_ADOPTED) + list(COLLECT_ONLY_ADOPTED)
            + list(ADVANTAGE_ADOPTED) + list(VISUALIZATION_ADOPTED)
        ):
            assert _flag_name(f.flag) in text


class TestCounterReachAdoption:
    """0-4 (指標大整理提案書): 反撃計算の正式登録・既定ON化の回帰テスト。

    過去に「採用」とコードコメントのみ先行し、本ファイル未登録・CLI既定値も
    OFF のままという食い違いがあった (2026-08-12 発覚)。この不整合の再発を
    機械的に防ぐ。
    """

    def test_counter_reach_registered_in_advantage_adopted(self) -> None:
        """--counter-reach が ADVANTAGE_ADOPTED に登録されていること。"""
        flags = [_flag_name(f.flag) for f in ADVANTAGE_ADOPTED]
        assert "--counter-reach" in flags

    def test_counter_reach_enabled_by_default_flag_is_true(self) -> None:
        """単一情報源の真偽値が True (ON) であること。"""
        assert COUNTER_REACH_ENABLED_BY_DEFAULT is True

    def test_visualize_advantage_overlay_generate_default_matches(self) -> None:
        """generate() の enable_counter_reach 既定値が production_config と
        一致すること (関数既定値と CLI 既定値の食い違い再発防止)。"""
        import inspect

        import scripts.visualize_advantage_overlay as vao

        sig = inspect.signature(vao.generate)
        default = sig.parameters["enable_counter_reach"].default
        assert default == COUNTER_REACH_ENABLED_BY_DEFAULT

    def test_cli_default_sourced_from_production_config(self) -> None:
        """--counter-reach の CLI 既定値が production_config 定数の直接参照
        になっていること (直書き default=False の再発防止、正規表現で
        --counter-reach 引数定義ブロック内の default= を確認する)。"""
        text = _script_text("scripts/visualize_advantage_overlay.py")
        m = re.search(
            r'"--counter-reach",\s*action="store_true",\s*'
            r"default=(\w+)", text,
        )
        assert m is not None, "--counter-reach の定義が見つからない"
        assert m.group(1) == "COUNTER_REACH_ENABLED_BY_DEFAULT"


class TestIndicatorReorgDecisions:
    """指標大整理 (2026-08-12 決定記録) の記録が production_config に残って
    いることの回帰テスト。"""

    def test_has_date_and_reason(self) -> None:
        """各決定記録に採用日と根拠が記録されていること。"""
        for f in INDICATOR_REORG_DECISIONS:
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", f.adopted), f.flag
            assert len(f.reason) >= 10, f"{f.flag} の根拠が薄い"

    def test_describe_includes_reorg_section(self) -> None:
        """describe() に指標大整理セクションが出ること。"""
        text = describe()
        assert "指標大整理" in text
        for f in INDICATOR_REORG_DECISIONS:
            assert f.flag in text


class TestLearningDataBuildAdopted:
    """学習データビルダーの標準採用オプション (2026-08-18 新設) の回帰テスト。"""

    def test_has_date_and_reason(self) -> None:
        """各エントリに採用日と根拠が記録されていること。"""
        assert len(LEARNING_DATA_BUILD_ADOPTED) >= 1
        for f in LEARNING_DATA_BUILD_ADOPTED:
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", f.adopted), f.flag
            assert len(f.reason) >= 10, f"{f.flag} の根拠が薄い"

    def test_describe_includes_learning_data_build_section(self) -> None:
        """describe() に学習データビルダーセクションが出ること。"""
        text = describe()
        assert "学習データビルダー" in text
        for f in LEARNING_DATA_BUILD_ADOPTED:
            assert f.flag in text

    @pytest.mark.parametrize(
        "flag", [f.flag for f in LEARNING_DATA_BUILD_ADOPTED])
    def test_flag_exists_in_build_labeled_win_from_npz(self, flag: str) -> None:
        """採用済みフラグが実際に build_labeled_win_from_npz.py に存在すること。"""
        text = _script_text("scripts/build_labeled_win_from_npz.py")
        assert text, "scripts/build_labeled_win_from_npz.py が見つからない"
        assert _flag_name(flag) in text, f"{flag} がスクリプトに存在しない"


# 親→子依存の台帳 (visualize_advantage_overlay.py の各 help 文
# 「--xxx 無効時は無視される」記載と一致させること)。子は親が OFF だと
# 完全に無視されるため、「子だけ登録・親未登録」は
# 「単一情報源からフラグを取ると何も効かない」状態を意味する。
_RESOLVED_FLAG_PARENT: dict[str, str] = {
    "--resolved-decisive-amplify": "--resolved-exchange-eval",
    "--resolved-live-defender": "--resolved-exchange-eval",
    "--resolved-live-defender-strict": "--resolved-live-defender",
    "--resolved-kill-override": "--resolved-exchange-eval",
    "--resolved-kill-override-counter-aware": "--resolved-kill-override",
    "--resolved-victim-gen-live": "--resolved-exchange-eval",
    "--resolved-pending-landing-gate": "--resolved-exchange-eval",
    "--resolved-counter-placement-reuse": "--resolved-exchange-eval",
    "--resolved-counter-budget-quantize": "--resolved-exchange-eval",
}


class TestResolvedParentChildConsistency:
    """決着先読み (resolved-*) 一族の親子整合の回帰テスト (2026-08-24)。

    2026-08-15 に子フラグ2つ (--resolved-live-defender-strict/
    --resolved-kill-override) だけが ADVANTAGE_ADOPTED に登録され、親3つ
    (--resolved-exchange-eval/--resolved-decisive-amplify/
    --resolved-live-defender) が 2026-08-24 まで未登録という不整合があった。
    親が OFF だと子は全部無視される実装のため、単一情報源から取ったフラグを
    そのまま渡すと「子だけON = 何も効かない」。この不整合そのものを検出して
    再発を構造的に防ぐ。
    """

    def _registered(self) -> set[str]:
        return {_flag_name(f.flag) for f in ADVANTAGE_ADOPTED}

    def test_resolved_parent_flags_registered(self) -> None:
        """親3フラグが登録されていること (2026-08-24 user承認の正式登録)。"""
        registered = self._registered()
        for flag in (
            "--resolved-exchange-eval",
            "--resolved-decisive-amplify",
            "--resolved-live-defender",
        ):
            assert flag in registered, f"{flag} が ADVANTAGE_ADOPTED に無い"

    def test_registered_child_requires_registered_ancestors(self) -> None:
        """登録済みの子フラグは、依存する祖先が全段登録されていること。

        例: --resolved-live-defender-strict が登録されているなら、親
        --resolved-live-defender とその親 --resolved-exchange-eval も
        登録されていなければならない (どれか1つでも外れると子は死ぬ)。
        """
        registered = self._registered()
        for child in registered & set(_RESOLVED_FLAG_PARENT):
            cur = child
            while cur in _RESOLVED_FLAG_PARENT:
                cur = _RESOLVED_FLAG_PARENT[cur]
                assert cur in registered, (
                    f"{child} が登録されているのに祖先 {cur} が未登録 "
                    "(親OFFで子は全部無視される = 単一情報源から取ると"
                    "何も効かない不整合)"
                )

    def test_dependency_ledger_matches_overlay_help_text(self) -> None:
        """依存台帳が実装の help 文と食い違っていないこと (台帳の陳腐化防止)。

        visualize_advantage_overlay.py の argparse ブロックから
        「--xxx 無効時 (は無視)」記載を自動抽出し、
        (1) 抽出できた依存は必ず台帳に同内容で載っていること
        (2) 既知の4依存が抽出できること (陽性対照 = 抽出器自体の健全性確認、
            fail-silent 警戒)
        を確認する。文字列連結で行またぎになった flag 名 (例: counter-aware の
        help 内 \"--resolved-kill-\" + \"override\") は抽出不能のため、台帳側が
        superset であることのみ要求する。
        """
        text = _script_text("scripts/visualize_advantage_overlay.py")
        assert text, "visualize_advantage_overlay.py が見つからない"
        found: dict[str, str] = {}
        for block in text.split("ap.add_argument(")[1:]:
            m_self = re.search(r'"(--[a-z0-9-]+)"', block)
            if m_self is None:
                continue
            m_dep = re.search(r'(--[a-z0-9-]+) ?無効時', block)
            if m_dep is not None and m_dep.group(1) != m_self.group(1):
                found[m_self.group(1)] = m_dep.group(1)
        # (2) 陽性対照: 既知の4依存を抽出器が拾えていること
        for child, parent in (
            ("--resolved-decisive-amplify", "--resolved-exchange-eval"),
            ("--resolved-live-defender", "--resolved-exchange-eval"),
            ("--resolved-live-defender-strict", "--resolved-live-defender"),
            ("--resolved-kill-override", "--resolved-exchange-eval"),
        ):
            assert found.get(child) == parent, (
                f"抽出器が既知の依存 {child} -> {parent} を拾えていない "
                f"(実際: {found.get(child)})。help 文の書式が変わったなら"
                "抽出正規表現と台帳を更新すること"
            )
        # (1) 抽出できた依存は台帳に同内容で載っていること
        for child, parent in found.items():
            assert _RESOLVED_FLAG_PARENT.get(child) == parent, (
                f"help 文の依存 {child} -> {parent} が台帳 "
                f"_RESOLVED_FLAG_PARENT と食い違う "
                f"(台帳: {_RESOLVED_FLAG_PARENT.get(child)})"
            )
