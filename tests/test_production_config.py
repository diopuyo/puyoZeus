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
