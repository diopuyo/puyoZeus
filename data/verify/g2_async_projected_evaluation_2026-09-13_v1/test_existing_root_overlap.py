"""実盤面の重複区間を人工familyに入れ、既存投影器の責務境界を確認する。"""
from dataclasses import replace
import json
from pathlib import Path
from src.chain import ChainSimulator
from src.event_root_family_v1 import PrefireAnchor, ChildChainSnapshot, RootFamilyKey, RootFamilySnapshot
from src.projected_state_observation_adapter_v1 import estimate_family, unique_post_chain_anchor

ROOT = Path(__file__).resolve().parent
RATE = 70  # この反例専用の人工rate。実動画のrate検収には使わない。


def family() -> RootFamilySnapshot:
    source = json.loads((ROOT / 'OPPONENT_ROOT_RELATION_v2.json').read_bytes())
    children = []
    for index, row in enumerate(source['origins'][:2]):
        anchor = PrefireAnchor(f'artificial-anchor-{index}', row['frame'], row['frame'] * 1000 // 60,
            tuple(map(tuple, row['source_grid'])), False, 'artificial-source', 'artificial-build',
            'artificial-attempt', 0)
        children.append(ChildChainSnapshot(f'artificial-child-{index}', anchor, 0, True,
            RATE, f'artificial-event-{index}', (), 0, 0))
    return RootFamilySnapshot(RootFamilyKey('p2', 1, 'artificial-common-root'), tuple(children), True, ())


def test_existing_amount_does_not_detect_overlapping_distinct_anchors() -> None:
    value = family()
    simulator = ChainSimulator(exclude_hidden_row_from_pop=True)
    original = estimate_family(value, simulator)
    whole = estimate_family(replace(value, children=value.children[:1]), simulator)
    assert original.trusted and whole.trusted
    assert original.projected_amount > whole.projected_amount
    assert original.projected_amount == 79080 // RATE + 52500 // RATE


def test_existing_board_guard_rejects_distinct_anchor_family() -> None:
    anchor, reasons = unique_post_chain_anchor(family(), 'p2')
    assert anchor is None and reasons == ('recipient_post_chain_anchor_ambiguous',)


def test_existing_duplicate_anchor_guard_is_not_weakened() -> None:
    value = family()
    duplicate = replace(value.children[1], anchor=value.children[0].anchor)
    result = estimate_family(replace(value, children=(value.children[0], duplicate)),
                             ChainSimulator(exclude_hidden_row_from_pop=True))
    assert not result.trusted and result.projected_amount is None
    assert result.reason_codes == ('duplicate_prefire_anchor_within_family',)
