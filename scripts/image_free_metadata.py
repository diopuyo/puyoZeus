"""分割比較の監査列だけを、通し実行の識別子へ戻す。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

META_COLUMNS = frozenset(('lean_row_p1', 'lean_row_p2', 'game_idx', 'row_key'))


def leaf_differences(expected: Any, actual: Any, prefix: str = '') -> list[dict]:
    """欠落・型差も含めて列/添字単位で記録する。"""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(expected.keys() | actual.keys()):
            path = f'{prefix}.{key}' if prefix else key
            if key not in expected or key not in actual:
                result.append(dict(column=path, old=expected.get(key), split=actual.get(key),
                                   missing='old' if key not in expected else 'split'))
            else:
                result.extend(leaf_differences(expected[key], actual[key], path))
        return result
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        return [difference for i, (a, b) in enumerate(zip(expected, actual))
                for difference in leaf_differences(a, b, f'{prefix}[{i}]')]
    return [] if expected == actual else [dict(column=prefix, old=expected, split=actual)]


@dataclass(frozen=True)
class MetadataOffset:
    """NEXT行番号は両側共通のappendカウンター。欠測値-1は動かさない。"""
    lean_rows: int = 0
    games: int = 0
    adopted_games: int = 0
    previous_adopted: int | None = None

    def apply(self, value: dict) -> dict:
        result = deepcopy(value)
        if 'game' in result:
            result['game']['index'] += self.games
            adopted = result['game'].get('adopted')
            result['game']['adopted'] = (self.previous_adopted if adopted is None
                                         else adopted+self.adopted_games)
        row = result.get('b3_features')
        if row is None:
            return result
        for name in ('lean_row_p1', 'lean_row_p2'):
            if row.get(name, -1) >= 0:
                row[name] += self.lean_rows
        if 'game_idx' in row:
            row['game_idx'] += self.games
            row['row_key'] = (f"{row['video_id']}|g{row['game_idx']}|"
                              f"f{row['frame_idx']}|o{row['orientation']}")
        return result


def metadata_only(differences: list[dict]) -> bool:
    allowed = {f'b3_features.{name}' for name in META_COLUMNS} | {'game.index', 'game.adopted'}
    return all(d['column'] in allowed
               for d in differences)
