"""真の打ち合い (両者同時 chain_event) の発生頻度を測る (2026-08-21)。

指摘#9〜#19 の一連の対応 (ResolvedExchangeTracker) は `self._result`
(=両者同時発火の打ち合いを `_resolve()` で解決した実績) を前提にしている。
打ち合いが稀なら、この前提そのものが成立せず対応が全編で無効になる恐れがある
(coordinator 指示、2026-08-21)。本スクリプトは学習データ62本 (per-side 記録
npz、`chain_mechanism`/`chain_trigger_sec` 列) から、両者が時間的に重なって
連鎖している「エピソード」を検出し、頻度・持続時間を層別 (動画ごと) で測る。

軽量な走査のみ (盤面認識・レンダリングは行わない、npz の列を読むだけ)。
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NPZ_DIR = "data/indicators_v2/boards_lean_model50v2_2026-08-20"


@dataclass
class ChainEpisode:
    """1つの連鎖イベントの観測区間 ([start_sec, end_sec])。

    start_sec は `chain_trigger_sec` (検出側が確定した発火時刻)、end_sec は
    同一 trigger を共有する観測行の最大 t_sec (観測できた最後の時刻)。
    実際のアニメーション終了時刻はこれより後ろにずれ得る (観測密度に依存する
    下限値、過小評価方向のバイアスがある点に注意)。
    """
    start_sec: float
    end_sec: float


@dataclass
class GameStats:
    video_id: str
    game_idx: int
    n_rows: int = 0
    n_rows_1p_active: int = 0
    n_rows_2p_active: int = 0
    n_rows_both_active: int = 0
    n_episodes_1p: int = 0
    n_episodes_2p: int = 0
    overlap_durations_sec: list = field(default_factory=list)
    has_any_overlap: bool = False


def _build_episodes(t_sec: np.ndarray, trigger_sec: np.ndarray,
                     mechanism: np.ndarray) -> list[ChainEpisode]:
    """1 side・1試合分の行から連鎖エピソード一覧を作る (`chain_trigger_sec` で
    グルーピング、既知の食い違い率0.0% [別コーダ実測] のため `chain_mechanism`
    の非空判定のみで active 行を選ぶ)。"""
    active = mechanism != ""
    if not np.any(active):
        return []
    groups: dict[float, list[float]] = {}
    for t, tr in zip(t_sec[active], trigger_sec[active]):
        groups.setdefault(float(tr), []).append(float(t))
    return [ChainEpisode(start_sec=tr, end_sec=max(ts + [tr]))
            for tr, ts in groups.items()]


def _intervals_overlap(a: ChainEpisode, b: ChainEpisode) -> "float | None":
    """区間が重なるなら重なり秒数を返す (重ならなければ None)。"""
    lo = max(a.start_sec, b.start_sec)
    hi = min(a.end_sec, b.end_sec)
    return (hi - lo) if hi >= lo else None


def _analyze_game(video_id: str, game_idx: int, rows_mask: np.ndarray,
                   d: dict) -> GameStats:
    """1試合分の行 (rows_mask で絞り込み済み) を解析する。"""
    side = d["side"][rows_mask]
    t_sec = d["t_sec"][rows_mask]
    trigger = d["chain_trigger_sec"][rows_mask]
    mech = d["chain_mechanism"][rows_mask]
    stats = GameStats(video_id=video_id, game_idx=game_idx, n_rows=int(rows_mask.sum()))

    ep1 = _build_episodes(t_sec[side == "1P"], trigger[side == "1P"], mech[side == "1P"])
    ep2 = _build_episodes(t_sec[side == "2P"], trigger[side == "2P"], mech[side == "2P"])
    stats.n_episodes_1p, stats.n_episodes_2p = len(ep1), len(ep2)

    def _active_at(t: float, eps: list[ChainEpisode]) -> bool:
        return any(e.start_sec <= t <= e.end_sec for e in eps)

    for t in t_sec:
        a1 = _active_at(t, ep1)
        a2 = _active_at(t, ep2)
        stats.n_rows_1p_active += int(a1)
        stats.n_rows_2p_active += int(a2)
        stats.n_rows_both_active += int(a1 and a2)

    for e1 in ep1:
        for e2 in ep2:
            ov = _intervals_overlap(e1, e2)
            if ov is not None:
                stats.has_any_overlap = True
                stats.overlap_durations_sec.append(ov)
    return stats


def _positive_control() -> bool:
    """合成データで「両者同時発火なら検出される」ことを確認する自己検収。

    1P: trigger=10.0, 観測 t=[10.0,10.3,10.6] / 2P: trigger=10.2,
    観測 t=[10.2,10.5] → 区間 [10.0,10.6] と [10.2,10.5] は重なる (0.3秒)。
    """
    ep1 = _build_episodes(
        np.array([10.0, 10.3, 10.6]), np.array([10.0, 10.0, 10.0]),
        np.array(["baseline", "baseline", "baseline"]))
    ep2 = _build_episodes(
        np.array([10.2, 10.5]), np.array([10.2, 10.2]),
        np.array(["landing", "landing"]))
    ov = _intervals_overlap(ep1[0], ep2[0])
    ok = ov is not None and abs(ov - 0.3) < 1e-6
    print(f"[positive-control] 合成同時発火の検出: {'OK' if ok else 'NG'} (重なり={ov})")
    # 陰性対照: 重ならないケース (2P が 1P 終了後に開始) も確認する。
    ep2_no = _build_episodes(np.array([20.0]), np.array([20.0]), np.array(["baseline"]))
    ov_no = _intervals_overlap(ep1[0], ep2_no[0])
    ok_no = ov_no is None
    print(f"[negative-control] 非重複ケースで非検出: {'OK' if ok_no else 'NG'} (重なり={ov_no})")
    return ok and ok_no


def main() -> None:
    assert _positive_control(), "検出ロジックの陽性/陰性対照に失敗 (実装バグの疑い)"
    npz_paths = sorted(glob.glob(f"{NPZ_DIR}/*.npz"))
    print(f"[scan] 対象npz数: {len(npz_paths)}")
    per_video: dict[str, list[GameStats]] = {}
    all_games: list[GameStats] = []
    for path in npz_paths:
        d = np.load(path, allow_pickle=True)
        video_ids = np.unique(d["video_id"])
        for vid in video_ids:
            vid_mask = d["video_id"] == vid
            for g in np.unique(d["game_idx"][vid_mask]):
                rows_mask = vid_mask & (d["game_idx"] == g)
                stats = _analyze_game(str(vid), int(g), rows_mask, d)
                per_video.setdefault(str(vid), []).append(stats)
                all_games.append(stats)
    _report(per_video, all_games)


def _report(per_video: dict[str, list[GameStats]], all_games: list[GameStats]) -> None:
    total_rows = sum(g.n_rows for g in all_games)
    total_both = sum(g.n_rows_both_active for g in all_games)
    total_either = sum(
        g.n_rows_1p_active + g.n_rows_2p_active - g.n_rows_both_active for g in all_games)
    n_games = len(all_games)
    n_games_with_overlap = sum(1 for g in all_games if g.has_any_overlap)
    all_durations = [d for g in all_games for d in g.overlap_durations_sec]

    print("\n=== 62本全体 (プール) ===")
    print(f"総行数: {total_rows}")
    print(f"両者同時 chain 行の割合: {total_both / total_rows * 100:.3f}%")
    print(f"片側以上 chain 行の割合: {total_either / total_rows * 100:.3f}%")
    if total_either:
        print(f"  -> 「片側以上」のうち両者同時が占める割合: "
              f"{total_both / total_either * 100:.2f}%")
    print(f"試合数(video×game_idx): {n_games}")
    print(f"同時発火が一度も無い試合の割合: "
          f"{(n_games - n_games_with_overlap) / n_games * 100:.1f}% "
          f"({n_games - n_games_with_overlap}/{n_games})")
    print(f"同時発火が1回以上ある試合の割合: "
          f"{n_games_with_overlap / n_games * 100:.1f}% ({n_games_with_overlap}/{n_games})")
    total_overlap_events = sum(len(g.overlap_durations_sec) for g in all_games)
    print(f"同時発火エピソードの総数: {total_overlap_events} "
          f"(試合あたり平均 {total_overlap_events / n_games:.3f} 回)")
    if all_durations:
        arr = np.array(all_durations)
        print(f"同時発火の持続時間 (観測下限値、秒): "
              f"min={arr.min():.2f} median={np.median(arr):.2f} "
              f"p90={np.percentile(arr, 90):.2f} max={arr.max():.2f}")
    else:
        print("同時発火の持続時間: 該当エピソードなし")

    print("\n=== 動画別 (層別、feedback_stratify_before_pooling) ===")
    rows_out = []
    for vid, games in sorted(per_video.items()):
        v_rows = sum(g.n_rows for g in games)
        v_both = sum(g.n_rows_both_active for g in games)
        v_n_games = len(games)
        v_overlap_games = sum(1 for g in games if g.has_any_overlap)
        v_events = sum(len(g.overlap_durations_sec) for g in games)
        pct_both = v_both / v_rows * 100 if v_rows else float("nan")
        rows_out.append({
            "video_id": vid, "n_rows": v_rows, "n_games": v_n_games,
            "pct_both_rows": round(pct_both, 3),
            "n_games_with_overlap": v_overlap_games,
            "pct_games_with_overlap": round(
                v_overlap_games / v_n_games * 100, 1) if v_n_games else float("nan"),
            "n_overlap_events": v_events,
        })
    for r in rows_out:
        print(f"  video={r['video_id']:>6} games={r['n_games']:>2} "
              f"同時行%={r['pct_both_rows']:>6.3f}% "
              f"同時発火あり試合={r['n_games_with_overlap']}/{r['n_games']} "
              f"({r['pct_games_with_overlap']:>5.1f}%) "
              f"エピソード数={r['n_overlap_events']}")

    out_path = Path("data/verify/mutual_exchange_frequency_2026-08-21.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "pooled": {
            "total_rows": total_rows, "total_both_rows": total_both,
            "total_either_rows": total_either, "n_games": n_games,
            "n_games_with_overlap": n_games_with_overlap,
            "total_overlap_events": total_overlap_events,
            "overlap_durations_sec": all_durations,
        },
        "per_video": rows_out,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
