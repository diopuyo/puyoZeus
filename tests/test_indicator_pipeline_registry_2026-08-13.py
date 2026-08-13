"""指標パイプラインの台帳整合 恒久ガードテスト (2026-08-13、横展開監査対応)。

docs/CROSS_CUTTING_AUDIT_2026-08-13.md の3提案を実装する:

1. **レジストリ整合テスト (P1/P2)**: `scripts/collect_indicators_v2.py`
   (旧収集) の `INDICATOR_COLUMNS` と `scripts/build_labeled_win_from_npz.py`
   (新変換) の全出力列の差分が、`src.production_config.KNOWN_PIPELINE_GAPS`
   (意図的ギャップ許容リスト) に一致することを検査する。**現状の乖離11列
   (main_linked_pair_count 等、docs/INDICATOR_PROPOSAL_ROUND2_2026-08-13.md
   A-1、user採否待ち) は許容リストに含めない** — 含めると A-1 の問題が
   テストから見えなくなり「直したことにする」事故を再生産するため。
   本テストは「11列が現在も未接続のまま検出されること」を確認する形にし
   (expectedFailure で隠さない)、A-1 が実施されて乖離が縮まったときは
   このテスト自身が **失敗して** 期待値の更新を要求する
   (xfail(strict) 相当の「直したらテストを更新せよ」効果を、xfail機構を
   使わない素の assert で実現する — CI の green/red がそのまま可視化される
   ため xfail バナーの裏に事故を隠さない)。

2. **削除台帳整合テスト (P2)**: `src.production_config.REORG_REMOVED_
   INDICATORS` (saturated_chain_count/absorption_capacity + 死亡確定3件
   honsen_output/taiou_capacity/disturbance_rejection) と
   `scripts/visualize_advantage_overlay.FEATURE_CANDIDATES` /
   `scripts/model_indicator_win.REDUNDANT_COLS` の整合を検査する。

3. **DIFF_5分類 完全分割テスト (P2「健全と確認」項目の恒久化)**:
   `build_labeled_win_from_npz.py` の grid-only レジストリ (light+heavy+
   connectivity) の全列が DIFF_* 定数群5分類のいずれか **正確に1つ** に
   属することを検査する (新指標追加時に分類漏れがあれば本テストが落ちる)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.build_labeled_win_from_npz as blw  # noqa: E402
import scripts.collect_indicators_v2 as civ2  # noqa: E402
import scripts.model_indicator_win as miw  # noqa: E402
import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.production_config import (  # noqa: E402
    KNOWN_PIPELINE_GAPS,
    REORG_REMOVED_INDICATORS,
    known_pipeline_gap_columns,
    reorg_removed_indicator_names,
)

# collect_indicators_v2.INDICATOR_COLUMNS のうち、指標本体でなく付随メタ列
# (連鎖数のソース種別・確定した max_chain 等) のため比較対象から除外する。
_META_ONLY_COLUMNS: frozenset[str] = frozenset({
    "reach_fire_power_source", "reach_fire_power_max_chain",
    "chain_duration_source",
})

# 2026-08-13 時点で実際に検出される「意図的でない脱落」列 (A-1、
# docs/INDICATOR_PROPOSAL_ROUND2_2026-08-13.md、user採否待ち)。
# **この集合を KNOWN_PIPELINE_GAPS に足して本テストを黙らせてはいけない**
# (脱落の可視化そのものが本テストの目的)。A-1 が実施されたら、実際に
# 再接続された列をこの集合から手動で削除し、コメントで反映日を明記すること。
#
# 【2026-08-13 A-1実装反映】immediate_fire_power/chain_efficiency/
# min_puyos_to_ignite/second_chain_potential/main_linked_pair_count/
# isolated_pair_count/main_linked_ratio/ignition_point_count/
# multi_color_ignition/simultaneous_pop_richness の10列を
# GRID_ONLY_HEAVY_INDICATORS へ再接続済み (scripts/build_labeled_win_
# from_npz.py)。reach_fire_power はこの10列と異なり next_pair/dnext_pair
# 必須で grid-onlyレジストリの型と非互換なため対象外のまま
# (production_config.KNOWN_PIPELINE_GAPS に既に文書化済み、gaps側で除外
# されるため本集合には現れない)。残る ojama_disruption は A-1 の対象外
# (別件、未対応のまま)。
EXPECTED_UNDOCUMENTED_GAP_COLUMNS: frozenset[str] = frozenset({
    "ojama_disruption",
})


def _old_base_columns() -> list[str]:
    """collect_indicators_v2.INDICATOR_COLUMNS から比較対象の base 列名を返す。

    *_raw 列 (score の定数倍で完全重複、新パイプラインでは a-1決定により
    そもそも出力しない設計) とメタ列は比較対象から除外する。
    """
    return [
        c for c in civ2.INDICATOR_COLUMNS
        if not c.endswith("_raw") and c not in _META_ONLY_COLUMNS
    ]


def _new_pipeline_full_fields() -> set[str]:
    """build_labeled_win_from_npz.py の full profile 出力列集合を返す。"""
    return set(blw._final_fieldnames("full"))


def _is_covered(column: str, full_fields: set[str]) -> bool:
    """new パイプラインが当該概念を own または diff_ のどちらかで出力しているか。

    b-2 決定 (DIFF_REPLACE_OWN_COLUMNS) により own → diff_ へ完全置換された
    列があるため、own 名がなくても diff_ 名があれば「概念としては存続」と
    みなす (レジストリ整合テストの対象は「概念の欠落」であって「列名の
    完全一致」ではない)。
    """
    return column in full_fields or f"diff_{column}" in full_fields


class TestRegistryGapAllowList:
    """collect_indicators_v2 → build_labeled_win_from_npz の列欠落が
    「既知の意図的ギャップ」+「現状把握済みの11列」以外に存在しないこと。"""

    def test_undocumented_gap_matches_known_11_columns(self) -> None:
        """未接続なのに許容リストに無い列が、現状把握済みの11列と完全一致すること。

        この assert が失敗する2パターン:
          (a) A-1 が実施され一部/全部が再接続された → 期待値集合を縮小して更新。
          (b) 新たな指標が追加されたのに新パイプラインへの接続もKNOWN_
              PIPELINE_GAPSへの登録も忘れた → 型B事故の再発、実装漏れを直す。
        どちらのケースでも「テストを見て初めて気づく」を「テストが落ちて
        気づく」に変えることが本テストの目的。
        """
        full_fields = _new_pipeline_full_fields()
        gaps = known_pipeline_gap_columns()
        undocumented = {
            c for c in _old_base_columns()
            if not _is_covered(c, full_fields) and c not in gaps
        }
        assert undocumented == EXPECTED_UNDOCUMENTED_GAP_COLUMNS, (
            f"未接続列の集合が変化した (現在={sorted(undocumented)}, "
            f"期待={sorted(EXPECTED_UNDOCUMENTED_GAP_COLUMNS)})。A-1 対応で"
            "再接続した場合はテストの期待値集合を更新し、意図しない新規"
            "脱落の場合は接続漏れを修正すること"
        )

    def test_known_pipeline_gaps_does_not_swallow_the_11_columns(self) -> None:
        """KNOWN_PIPELINE_GAPS に11列を紛れ込ませて本テストを無力化していないこと。"""
        assert not (
            known_pipeline_gap_columns() & EXPECTED_UNDOCUMENTED_GAP_COLUMNS
        ), "現状把握済み11列 (A-1 対象) を KNOWN_PIPELINE_GAPS に含めてはいけない"

    def test_known_pipeline_gaps_entries_are_actually_absent(self) -> None:
        """KNOWN_PIPELINE_GAPS に登録された列が実際に新パイプラインで未接続であること

        (許容リスト自体が古くなり、実は再接続済みなのに残り続ける事故を防ぐ)。
        """
        full_fields = _new_pipeline_full_fields()
        for gap in KNOWN_PIPELINE_GAPS:
            assert not _is_covered(gap.column, full_fields), (
                f"{gap.column!r} は KNOWN_PIPELINE_GAPS に登録されているが、"
                "実際には新パイプラインで既に出力されている (許容リストが"
                "陳腐化、エントリを削除すること)"
            )

    def test_known_pipeline_gaps_columns_are_real_old_indicators(self) -> None:
        """KNOWN_PIPELINE_GAPS の各列名が旧 INDICATOR_COLUMNS に実在すること (typo guard)。"""
        old_base = set(_old_base_columns())
        for gap in KNOWN_PIPELINE_GAPS:
            assert gap.column in old_base, (
                f"{gap.column!r} は collect_indicators_v2.INDICATOR_COLUMNS "
                "に存在しない (KNOWN_PIPELINE_GAPS のタイプミスの疑い)"
            )


class TestRemovedIndicatorLedgerConsistency:
    """REORG_REMOVED_INDICATORS (削除台帳) と実際の除外リスト2箇所の整合。"""

    def test_feature_candidates_excludes_all_removed_indicators(self) -> None:
        """visualize_advantage_overlay.FEATURE_CANDIDATES に削除確定指標が残っていないこと。"""
        leaked = set(vao.FEATURE_CANDIDATES) & reorg_removed_indicator_names()
        assert not leaked, (
            f"FEATURE_CANDIDATES に削除確定指標が残っている: {sorted(leaked)} "
            "(REORG_REMOVED_INDICATORS への反映漏れ)"
        )

    def test_redundant_cols_includes_all_removed_indicators(self) -> None:
        """model_indicator_win.REDUNDANT_COLS が削除確定指標を全て含むこと (*_raw込み)。"""
        names = reorg_removed_indicator_names()
        missing = {n for n in names if n not in miw.REDUNDANT_COLS}
        missing_raw = {
            f"{n}_raw" for n in names if f"{n}_raw" not in miw.REDUNDANT_COLS
        }
        assert not missing, f"REDUNDANT_COLS に未反映の削除確定指標: {sorted(missing)}"
        assert not missing_raw, (
            f"REDUNDANT_COLS に未反映の削除確定指標 (_raw版): {sorted(missing_raw)}"
        )

    def test_removed_indicators_ledger_has_expected_five_entries(self) -> None:
        """台帳の想定5件 (saturated_chain_count/absorption_capacity + 死亡確定3件) を確認。

        件数・名前が変わった場合は台帳の意図的な更新のはずなので、
        このテストを見て気づけるようにする (サイレントな増減を防ぐ)。
        """
        expected_names = {
            "saturated_chain_count", "absorption_capacity",
            "honsen_output", "taiou_capacity", "disturbance_rejection",
        }
        assert reorg_removed_indicator_names() == expected_names
        assert len(REORG_REMOVED_INDICATORS) == 5


class TestDiffClassificationCompleteness:
    """build_labeled_win_from_npz.py の DIFF_* 5分類が grid-only レジストリの
    全列を過不足なく分類していること (新指標追加時の分類漏れガード)。"""

    def _all_diff_classified_columns(self) -> list[str]:
        """DIFF_* 5分類タプルを結合した列一覧 (重複チェックは別途行う)。"""
        return (
            list(blw.DIFF_REPLACE_OWN_COLUMNS)
            + list(blw.DIFF_KEEP_OWN_HEAVY_COLUMNS)
            + list(blw.DIFF_KEEP_OWN_PAIR_COLUMNS)
            + list(blw.DIFF_KEEP_OWN_NEW_COLUMNS)
            + list(blw.DIFF_EXEMPT_OWN_ONLY_COLUMNS)
        )

    def _all_registry_columns(self) -> set[str]:
        """light+heavy レジストリ + connectivity 固定3列の全キー集合。"""
        return (
            set(blw.GRID_ONLY_INDICATORS)
            | set(blw.GRID_ONLY_HEAVY_INDICATORS)
            | set(blw.CONN_ALWAYS_PRESENT_COLUMNS)
        )

    def test_every_registry_column_is_classified_exactly_once(self) -> None:
        """レジストリの全列が DIFF_* 5分類のいずれか正確に1つに属すること。"""
        classified = self._all_diff_classified_columns()
        registry_cols = self._all_registry_columns()
        # 未分類 (新指標追加時に分類し忘れた列)
        unclassified = registry_cols - set(classified)
        assert not unclassified, (
            f"DIFF_* 5分類のどれにも属さない列: {sorted(unclassified)} "
            "(新指標を追加したら DIFF_REPLACE_OWN_COLUMNS 等いずれかに分類すること)"
        )
        # 二重分類 (同じ列が複数の分類タプルに重複登録)
        duplicates = {c for c in classified if classified.count(c) > 1}
        assert not duplicates, f"DIFF_* 5分類で重複登録された列: {sorted(duplicates)}"

    def test_no_diff_classification_references_unknown_column(self) -> None:
        """DIFF_* 5分類にレジストリ非存在の列名 (typo) が紛れていないこと。"""
        registry_cols = self._all_registry_columns()
        classified = set(self._all_diff_classified_columns())
        unknown = classified - registry_cols
        assert not unknown, (
            f"DIFF_* 5分類にレジストリ非存在の列名が含まれる (typoの疑い): "
            f"{sorted(unknown)}"
        )
