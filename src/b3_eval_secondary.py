"""副次 CV: 対戦者単位の leave-out (事前登録 §3.3) の分割・d 指標。学習はしない (分割と集計だけ)。

対戦者名の抽出規則 (§3.3):
  1. title を NFKC 正規化、2. 「「A vs B」」形式があれば括弧内をすべて取る、
  3. 無ければ先頭の【…】を除き `^\\s*(.+?)\\s+vs\\s+(.+?)\\s*\\d+先` で A/B、4. 名前の正規化キーは normalize_player。
抽出できない動画があれば停止する。別名の自動統合はしない (別名表は学習前に事前登録へ追記)。

!! 事前登録の「対照」の文面 (test の動画は同じ・train から P と無関係な動画を P 除外で減った本数だけ除く) は、
   test = P の全動画のとき P 除外モデルの train と一致して縮退する。本モジュールは文面どおりの対照が作れる場合
   (P 除外で train から外れる動画が test 以外にもある場合) のみ作り、縮退する場合は control=None と理由を返す。
   解釈の確定は統括 (報告に記載)。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Mapping, Optional, Sequence

from src.b3_train_config import SECONDARY_MIN_PLAYER_VIDEOS
from src.b3_train_folds import control_drop_order, normalize_player

_QUOTED = re.compile(r"「\s*(.+?)\s+vs\s+(.+?)\s*」", re.IGNORECASE)
_PLAIN = re.compile(r"^\s*(.+?)\s+vs\s+(.+?)\s*\d+先", re.IGNORECASE)
_LEADING_BRACKETS = re.compile(r"^(?:【[^】]*】)+")


class PlayerExtractionError(ValueError):
    """対戦者を抽出できない動画がある (停止条件)。"""


def extract_players(title: str) -> list[tuple[str, str]]:
    t = unicodedata.normalize("NFKC", title)
    pairs = [(a.strip(), b.strip()) for a, b in _QUOTED.findall(t)]
    if pairs:
        return pairs
    m = _PLAIN.match(_LEADING_BRACKETS.sub("", t))
    if m:
        return [(m.group(1).strip(), m.group(2).strip())]
    raise PlayerExtractionError(f"対戦者を抽出できない: {title!r}")


def video_players(titles: Mapping[str, str]) -> dict[str, set[str]]:
    """video_id -> 正規化キーの選手集合。1 本でも抽出できなければ停止。"""
    out: dict[str, set[str]] = {}
    for vid, title in titles.items():
        out[vid] = {normalize_player(n) for pair in extract_players(title) for n in pair}
    return out


def eligible_players(players: Mapping[str, set[str]], min_videos: int = SECONDARY_MIN_PLAYER_VIDEOS) -> dict[str, list[str]]:
    """3 本以上の動画に出る選手 -> その動画 (昇順)。"""
    by: dict[str, list[str]] = {}
    for vid, ps in players.items():
        for p in ps:
            by.setdefault(p, []).append(vid)
    return {p: sorted(v) for p, v in sorted(by.items()) if len(v) >= min_videos}


def leave_player_split(player: str, players: Mapping[str, set[str]], group_of_video: Mapping[str, str]) -> dict:
    """P の出る動画 (と同一 group の動画) を test、それ以外を train にする。group は train/test で跨がない。"""
    test = {v for v, ps in players.items() if player in ps}
    test_groups = {group_of_video[v] for v in test}
    test |= {v for v, g in group_of_video.items() if g in test_groups and v in players}
    train = sorted(v for v in players if v not in test)
    if {group_of_video[v] for v in train} & test_groups:
        raise AssertionError("train と test で group が跨った")
    return {"player": player, "test": sorted(test), "train": train, "n_removed_from_pool": len(test)}


def control_split(player: str, split: Mapping, players: Mapping[str, set[str]], group_of_video: Mapping[str, str]) -> dict:
    """事前登録の対照。train を P 除外モデルと同じ本数にそろえるための除外 k 本を決定的に選ぶ。

    k = (test 以外で P 除外により train から外れた本数)。P の全動画が test のとき k=0 で縮退する (control=None)。
    """
    pool = [v for v in players if v not in set(split["test"])]
    not_train = [v for v in pool if v not in set(split["train"])]
    k = len(not_train)
    if k == 0:
        return {"control": None, "reason": "k=0: P 除外の train と一致して縮退 (対照の解釈は統括確認)"}
    unrelated = [v for v in split["train"] if player not in players[v]]
    drop = control_drop_order(player, unrelated)[:k]
    return {"control": sorted(set(pool) - set(drop)), "dropped": sorted(drop), "k": k}


def d_player(ll_excl: float, ll_control: float) -> float:
    """d_P = LL(P 除外) − LL(対照)。"""
    return float(ll_excl - ll_control)


def d_mean(per_player: Mapping[str, Mapping]) -> float:
    """全選手を合わせた d̄ (選手の test 試合数で重み付け)。per_player[p] = {d, test_games}。"""
    num = sum(v["d"] * v["test_games"] for v in per_player.values())
    den = sum(v["test_games"] for v in per_player.values())
    return float(num / den) if den else float("nan")
