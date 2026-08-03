"""「終局連鎖の欠落」全域定量化 (2026-08-03 main発注、修正はまだ・報告のみ)。

match_02の1P (t=3003.67->3007.10、色ぷよ44->3) で実際に起きたはずの2回目の
発火が exchange_delta_winprob 側に一切記録されていなかった (=_diag_match02_
underclamp_2026-08-03.py で確認済み)。本スクリプトはこの現象が66動画
(boards_lean_regen) + olRyxDGacbg 全体でどれだけ起きているかを定量化する。

## 操作的定義 (推測でなく計算可能な形に落とす)
1試合1サイド分について:
    - 既存の発火検出器 (_detect_fire_events + _merge_fire_event_clusters、
      再実装しない) を素の npz score 列にそのまま適用し、そのサイド・その
      試合内で検出された最後のクラスタの fire_index / score を求める
      (無ければ試合最初の有効スコアを基準0とする)。
    - その試合・そのサイドの score 列で最後に有効 (>=0) だったインデックス
      (=試合末尾側で読み取れた最後のスコア) の score 値を取る。
    - 「終局スコア差」 = 試合末尾側の最後の有効スコア - 最後に検出された
      発火クラスタの score (または基準0)。
    - 終局スコア差 >= SCORE_DELTA_FIRE (=80、既存定数) かつ、その最後の
      有効インデックスが最後の発火クラスタより後ろにある場合を
      「終局連鎖の欠落」と判定する (=発火閾値相当のスコア増分が起きたのに
      対応する発火イベントが記録されていない)。

## 「勝敗決定連鎖の欠落率」
欠落が検出された (試合, サイド) のうち、その試合の won フラグ (npz由来、
そのサイドの最終行 won=1) がそのサイドの勝利を示すものの割合。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.label_exchange_outcome import (
    SCORE_DELTA_FIRE,
    NpzRecord,
    _detect_fire_events,
    _load_npz,
    _merge_fire_event_clusters,
)

REGEN_NPZ_DIR = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
OLRYX_NPZ_PATH = Path("data/indicators_v2/boards_lean_olRyxDGacbg_2026-08-03/olRyxDGacbg.npz")
AUG_CSV = Path("data/indicators_v2/exchange_labels_regen_step3_aug_2026-08-02.csv")


@dataclass(frozen=True)
class TerminalGapResult:
    """1試合1サイド分の終局スコア差判定結果。"""
    video_id: str
    side: str
    game_idx: int
    last_event_score: int
    last_valid_score: int
    gap: int
    is_missing: bool
    had_prior_event: bool  # False=試合中一度も検出発火なし (小刻み蓄積との混同注意)
    won_this_side: "bool | None"  # 判定不能(有効wonなし)ならNone


def _last_valid_score_index(score: np.ndarray) -> "int | None":
    """score配列内で最後に有効 (>=0) だったインデックスを返す (無ければNone)。"""
    valid_idx = np.where(score >= 0)[0]
    return int(valid_idx[-1]) if len(valid_idx) > 0 else None


def evaluate_terminal_gap_for_side(
    video_id: str, side: str, game_idx: int,
    t_sec: np.ndarray, score: np.ndarray, grids: np.ndarray, won: np.ndarray,
) -> "TerminalGapResult | None":
    """1試合1サイド分の終局スコア差を判定する (データ不足ならNone)。"""
    last_valid_idx = _last_valid_score_index(score)
    if last_valid_idx is None:
        return None
    fire_indices = _detect_fire_events(t_sec, score)
    clusters = _merge_fire_event_clusters(t_sec, score, grids, fire_indices)
    had_prior_event = bool(clusters)
    if clusters:
        last_event_idx = clusters[-1].fire_index
        last_event_score = int(score[last_event_idx])
    else:
        last_event_idx = -1
        first_valid = np.where(score >= 0)[0]
        last_event_score = int(score[first_valid[0]]) if len(first_valid) > 0 else 0
    last_valid_score = int(score[last_valid_idx])
    gap = last_valid_score - last_event_score
    is_missing = gap >= SCORE_DELTA_FIRE and last_valid_idx > last_event_idx
    won_this_side = bool(won[last_valid_idx] >= 0.5) if won[last_valid_idx] == won[last_valid_idx] else None
    return TerminalGapResult(
        video_id=video_id, side=side, game_idx=int(game_idx),
        last_event_score=last_event_score, last_valid_score=last_valid_score,
        gap=gap, is_missing=is_missing, had_prior_event=had_prior_event,
        won_this_side=won_this_side,
    )


def evaluate_all_games_for_record(rec: NpzRecord) -> list[TerminalGapResult]:
    """1動画1サイド分、game_idx別に終局スコア差を判定する。"""
    results: list[TerminalGapResult] = []
    for game_idx in np.unique(rec.game_idx):
        mask = rec.game_idx == game_idx
        order = np.argsort(rec.t_sec[mask])
        t = rec.t_sec[mask][order]
        score = rec.score[mask][order]
        grids = rec.grids[mask][order]
        won = rec.won[mask][order]
        result = evaluate_terminal_gap_for_side(
            rec.video_id, rec.side, int(game_idx), t, score, grids, won)
        if result is not None:
            results.append(result)
    return results


def collect_all_npz_paths() -> list[Path]:
    paths = sorted(REGEN_NPZ_DIR.glob("*.npz"))
    if OLRYX_NPZ_PATH.exists():
        paths.append(OLRYX_NPZ_PATH)
    return paths


def main() -> None:
    all_results: list[TerminalGapResult] = []
    for path in collect_all_npz_paths():
        for rec in _load_npz(path):
            all_results.extend(evaluate_all_games_for_record(rec))

    df = pd.DataFrame([r.__dict__ for r in all_results])
    total_games_sides = len(df)
    print(f"=== 終局連鎖の欠落 全域定量化 (SCORE_DELTA_FIRE={SCORE_DELTA_FIRE}) ===")
    print(f"対象 (試合×サイド) 総数: {total_games_sides}"
          f"  (動画数={df['video_id'].nunique()}, 試合数={df.groupby(['video_id','game_idx']).ngroups})")

    missing = df[df["is_missing"]]
    print(f"\n欠落あり (試合×サイド): {len(missing)} / {total_games_sides}"
          f" ({len(missing) / total_games_sides:.1%})")
    for side in ("1P", "2P"):
        side_total = len(df[df["side"] == side])
        side_missing = len(missing[missing["side"] == side])
        print(f"  {side}: {side_missing} / {side_total} ({side_missing / side_total:.1%})")

    games_with_any_missing = missing.groupby(["video_id", "game_idx"]).ngroups
    total_games = df.groupby(["video_id", "game_idx"]).ngroups
    print(f"\n試合単位 (いずれかのサイドで欠落): {games_with_any_missing} / {total_games}"
          f" ({games_with_any_missing / total_games:.1%})")

    # 正直な注記: had_prior_event=False (試合中一度も検出発火なし) の欠落は
    # 「単発の大型連鎖漏れ」でなく「小刻みな加点の蓄積」と混同している可能性が
    # あるため、had_prior_event=True (=match_02と同型、既検出イベントの後に
    # さらに閾値相当のスコア増分がある明確な終局漏れ) のみで再集計する。
    unambiguous = missing[missing["had_prior_event"]]
    print(f"\n=== 内訳: had_prior_event別 (誠実性チェック) ===")
    print(f"  had_prior_event=True (=既検出発火の後に閾値相当の増分、match_02と同型・明確な終局漏れ):"
          f" {len(unambiguous)} / {len(missing)} ({len(unambiguous) / len(missing):.1%})")
    print(f"  had_prior_event=False (試合中一度も検出発火なし、小刻み蓄積の可能性あり・要注意):"
          f" {len(missing) - len(unambiguous)} / {len(missing)}")
    print(f"  -> 明確な終局漏れのみで見た割合: {len(unambiguous)} / {total_games_sides}"
          f" ({len(unambiguous) / total_games_sides:.1%})")

    decided = missing[missing["won_this_side"].notna()]
    decisive = decided[decided["won_this_side"]]
    print(f"\n=== 勝敗決定連鎖の欠落率 (全欠落ベース) ===")
    print(f"  won判定可能な欠落: {len(decided)} / {len(missing)}")
    if len(decided) > 0:
        print(f"  そのうち欠落側が勝者: {len(decisive)} / {len(decided)} ({len(decisive) / len(decided):.1%})")

    decided_ub = unambiguous[unambiguous["won_this_side"].notna()]
    decisive_ub = decided_ub[decided_ub["won_this_side"]]
    print(f"\n=== 勝敗決定連鎖の欠落率 (had_prior_event=True の明確な欠落のみ) ===")
    print(f"  won判定可能な欠落: {len(decided_ub)} / {len(unambiguous)}")
    if len(decided_ub) > 0:
        print(f"  そのうち欠落側が勝者: {len(decisive_ub)} / {len(decided_ub)}"
              f" ({len(decisive_ub) / len(decided_ub):.1%})")

    print(f"\n=== 欠落の gap 分布 (欠落判定分のみ) ===")
    print(missing["gap"].describe().to_string())

    print(f"\n=== 欠落上位20件 (gap降順) ===")
    print(missing.sort_values("gap", ascending=False).head(20)[
        ["video_id", "game_idx", "side", "last_event_score", "last_valid_score", "gap", "won_this_side"]
    ].to_string())

    # exchange_labels_regen_step3 aug CSV への影響 (該当試合が既存ラベルに何行含まれるか)
    if AUG_CSV.exists():
        aug_df = pd.read_csv(AUG_CSV)
        missing_keys = set(zip(missing["video_id"], missing["game_idx"]))
        aug_keys = set(zip(aug_df["video_id"], aug_df["game_idx"]))
        overlap = missing_keys & aug_keys
        print(f"\n=== exchange_labels_regen_step3への影響 ===")
        print(f"  欠落検出試合数: {len(missing_keys)}")
        print(f"  そのうち aug CSV に行がある試合数: {len(overlap)} ({len(overlap) / max(1,len(missing_keys)):.1%})")
        print(f"  (=欠落試合の大半で「その試合の一部の発火は捕捉されているが終局分だけ漏れている」ことを示す)")

    out_path = Path("data/verify/terminal_chain_gap_2026-08-03.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[保存] 全結果: {out_path}")


if __name__ == "__main__":
    main()
