"""B3 の元動画 group と決定的 fold 割当 (事前登録 §2.2〜§2.3)。

手順 (§2.3): group キー → 試合数 n_g → sha256(塩+キー) 昇順 → 試合数合計が最小の fold へ順に入れる
(同点は fold 番号の小さい方)。fold 割当は seed に依存しない。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from src.b3_train_config import (GROUP_JOIN, INNER_SALT_FMT, N_INNER_FOLDS, N_OUTER_FOLDS, OUTER_SALT,
                                 PLAYER_CTRL_SALT)


class FoldLeakError(RuntimeError):
    """同一 group が学習側と検証側に跨った。"""


def group_key(youtube_ids: Iterable[str]) -> str:
    """単一動画は YouTube ID、複数動画 (再投稿等) は昇順に '+' 連結。"""
    ids = sorted(set(youtube_ids))
    if not ids:
        raise ValueError("YouTube ID が空")
    return GROUP_JOIN.join(ids)


def video_to_group(video_youtube: Mapping[str, str], merges: Sequence[Sequence[str]] = ()) -> dict[str, str]:
    """video_id -> group キー。merges は同一 group にする YouTube ID の組 (例 c110/c112 の再投稿)。

    分割動画 (video_c96s1 等) は同じ YouTube ID なので自動的に同一 group になる。
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for yt in video_youtube.values():
        find(yt)
    for pair in merges:
        base = find(pair[0])
        for other in pair[1:]:
            parent[find(other)] = base
    members: dict[str, set[str]] = {}
    for yt in list(parent):
        members.setdefault(find(yt), set()).add(yt)
    key_of = {yt: group_key(m) for m in members.values() for yt in m}
    return {vid: key_of[yt] for vid, yt in video_youtube.items()}


def _hash_order(key: str, salt: str) -> str:
    return hashlib.sha256((salt + key).encode("utf-8")).hexdigest()


def assign_folds(games_per_group: Mapping[str, int], n_folds: int, salt: str) -> dict[str, int]:
    """試合数均衡の決定的な group -> fold 割当。"""
    if n_folds < 2:
        raise ValueError("n_folds >= 2")
    if len(games_per_group) < n_folds:
        raise ValueError(f"group 数 {len(games_per_group)} < fold 数 {n_folds}")
    totals = [0] * n_folds
    out: dict[str, int] = {}
    for key in sorted(games_per_group, key=lambda k: _hash_order(k, salt)):
        fold = min(range(n_folds), key=lambda f: (totals[f], f))
        out[key] = fold
        totals[fold] += int(games_per_group[key])
    return out


def outer_assignment(games_per_group: Mapping[str, int]) -> dict[str, int]:
    return assign_folds(games_per_group, N_OUTER_FOLDS, OUTER_SALT)


def inner_assignment(games_per_group: Mapping[str, int], outer_fold: int,
                     outer: Mapping[str, int]) -> dict[str, int]:
    """外側 fold k の train 側 group だけに同じ手順を適用する (塩を変える)。"""
    train = {g: n for g, n in games_per_group.items() if outer[g] != outer_fold}
    return assign_folds(train, N_INNER_FOLDS, INNER_SALT_FMT.format(k=outer_fold))


def assert_no_group_leak(train_groups: Iterable[str], test_groups: Iterable[str]) -> None:
    both = set(train_groups) & set(test_groups)
    if both:
        raise FoldLeakError(f"group が学習と検証に跨る: {sorted(both)[:5]}")


def games_per_group_from_rows(rows: pd.DataFrame, group_of_video: Mapping[str, str]) -> dict[str, int]:
    """行から (動画, 試合) の一意数を group ごとに数える (完了票の試合数が無い場合の代替)。"""
    uniq = rows[["video_id", "game_idx"]].drop_duplicates()
    counts: dict[str, int] = {}
    for vid, n in uniq.groupby("video_id").size().items():
        key = group_of_video[str(vid)]
        counts[key] = counts.get(key, 0) + int(n)
    return counts


def write_assignment_tsv(path: Path, games_per_group: Mapping[str, int], assignment: Mapping[str, int],
                         group_videos: Mapping[str, Sequence[str]]) -> str:
    """割当表 TSV を保存し SHA256 を返す (事前登録 §2.3-5)。"""
    lines = ["group\tvideos\tn_games\tfold"]
    for key in sorted(assignment, key=lambda k: (assignment[k], k)):
        lines.append(f"{key}\t{','.join(sorted(group_videos.get(key, ())))}\t{games_per_group[key]}\t{assignment[key]}")
    data = ("\n".join(lines) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def fold_summary(games_per_group: Mapping[str, int], assignment: Mapping[str, int]) -> dict:
    out: dict[int, dict[str, int]] = {}
    for g, f in assignment.items():
        d = out.setdefault(int(f), {"groups": 0, "games": 0})
        d["groups"] += 1
        d["games"] += int(games_per_group[g])
    return {str(k): v for k, v in sorted(out.items())}


def check_group_minimum(n_groups: int, minimum: int) -> None:
    """§2.1: 最終 group 数が下限未満なら学習を始めない。"""
    if n_groups < minimum:
        raise ValueError(f"group 数 {n_groups} < {minimum}: 学習を始めず統括へ戻す")


# --- 副次 CV (§3.3) の補助 ---
def normalize_player(name: str) -> str:
    """対戦者名の正規化キー: NFKC → 前後空白除去 → 空白削除 → casefold → カタカナをひらがなへ。"""
    import unicodedata
    s = unicodedata.normalize("NFKC", name).strip()
    s = "".join(s.split()).casefold()
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def control_drop_order(player: str, candidate_videos: Sequence[str]) -> list[str]:
    """対照モデルで train から除く動画の順 (sha256(塩+選手) 順ではなく、動画別に塩+選手+動画で決定的に並べる)。"""
    return sorted(candidate_videos, key=lambda v: _hash_order(f"{player}|{v}", PLAYER_CTRL_SALT))


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
