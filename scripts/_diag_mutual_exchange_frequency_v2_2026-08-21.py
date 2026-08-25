"""真の打ち合い (両者同時 chain_event) の発生頻度 再測定 (2026-08-21 v2)。

【前回 (`_diag_mutual_exchange_frequency_2026-08-21.py`) の誤り (user指摘)】
連鎖エピソードの終端を「観測できた最後の t_sec」としていたため、盤面が
STABLE で記録される瞬間 (=連鎖アニメが止まった後) しか捉えられず、
持続時間が発火直後の一瞬 (中央値0.13秒) に縮んでいた。

【本ファイルの修正】
`src/recognition_pipeline.py` の `chain_event` 保持ロジック (:4479-4566) を
確認したところ、`r_p1.chain_event`/`r_p2.chain_event` (=ResolvedExchangeTracker
が読む ev1/ev2) は **発火直後の1フレームだけでなく、保持タイマーが切れるまで
非None を維持し続ける** ことがコードで確認できた
(`self._active_chain_1p = ev` を毎フレーム signals に乗せ続ける設計)。

保持タイマーの式 (本番既定・generate()/collect_boards_lean.py はいずれも
override しないため bit-identical にこの既定式が使われている):
    hold_sec(N) = min(CHAIN_HOLD_BASE_SEC + CHAIN_HOLD_PER_STEP_SEC * N,
                       CHAIN_MAX_HOLD_SEC)
    = min(0.0 + 0.3 * N, 5.0)   (src/recognition_pipeline.py:729,737,749)

参考 (user が触れた「n=418」の較正値、23動画418イベント実測。ただし
**src の既定値ではなく呼び出し側 opt-in の較正候補**、
src/recognition_pipeline.py:731-736 参照):
    hold_sec_calibrated(N) = 2.61 + 1.17 * N   (未採用・sensitivity 用のみ)

npz (`boards_lean_model50v2_2026-08-20`) には連鎖数の列が無い
(`chain_mechanism`/`chain_trigger_sec` のみ、確認済み)。score からの逆算は
既知に信頼できない (memory project_chain_count_both_untrustworthy) ため
採用しない。**選択: 代表連鎖数 N を複数点 (1,4,6,8) でスイープし、感度を
そのまま報告する** (どの N でも同じ結論なら選択の影響は小さい、という形で
判断材料にする)。再フィット・自作の新規較正はしていない
(既存の2式のみ使用)。
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NPZ_DIR = "data/indicators_v2/boards_lean_model50v2_2026-08-20"

# src/recognition_pipeline.py の既定値そのまま (再フィットしない)。
CHAIN_HOLD_BASE_SEC_DEFAULT = 0.0       # :737
CHAIN_HOLD_PER_STEP_SEC_DEFAULT = 0.3   # :729
CHAIN_MAX_HOLD_SEC_DEFAULT = 5.0        # :749
# 参考較正 (n=418、opt-in、既定では使われない) :731-736
CHAIN_HOLD_BASE_SEC_CALIBRATED = 2.61
CHAIN_HOLD_PER_STEP_SEC_CALIBRATED = 1.17

FORMULAS = {
    "default_capped5s": lambda n: min(
        CHAIN_HOLD_BASE_SEC_DEFAULT + CHAIN_HOLD_PER_STEP_SEC_DEFAULT * n,
        CHAIN_MAX_HOLD_SEC_DEFAULT),
    "calibrated_n418_uncapped": lambda n: (
        CHAIN_HOLD_BASE_SEC_CALIBRATED + CHAIN_HOLD_PER_STEP_SEC_CALIBRATED * n),
}
CHAIN_COUNT_SWEEP = (1, 4, 6, 8)


@dataclass
class ChainEpisode:
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
    overlap_durations_sec: list = field(default_factory=list)
    has_any_overlap: bool = False


def _build_episodes(
    t_sec: np.ndarray, trigger_sec: np.ndarray, mechanism: np.ndarray,
    hold_sec: float,
) -> list[ChainEpisode]:
    """`chain_trigger_sec` でグルーピングし、区間 = [trigger, trigger+hold_sec]
    とする (hold_sec は仮定した連鎖数から式で求めた保持時間)。"""
    active = mechanism != ""
    if not np.any(active):
        return []
    triggers = sorted(set(float(tr) for tr in trigger_sec[active]))
    return [ChainEpisode(start_sec=tr, end_sec=tr + hold_sec) for tr in triggers]


def _intervals_overlap(a: ChainEpisode, b: ChainEpisode) -> "float | None":
    lo = max(a.start_sec, b.start_sec)
    hi = min(a.end_sec, b.end_sec)
    return (hi - lo) if hi >= lo else None


def _analyze_game(video_id: str, game_idx: int, rows_mask: np.ndarray,
                   d: dict, hold_sec: float) -> GameStats:
    side = d["side"][rows_mask]
    t_sec = d["t_sec"][rows_mask]
    trigger = d["chain_trigger_sec"][rows_mask]
    mech = d["chain_mechanism"][rows_mask]
    stats = GameStats(video_id=video_id, game_idx=game_idx, n_rows=int(rows_mask.sum()))

    ep1 = _build_episodes(t_sec[side == "1P"], trigger[side == "1P"],
                           mech[side == "1P"], hold_sec)
    ep2 = _build_episodes(t_sec[side == "2P"], trigger[side == "2P"],
                           mech[side == "2P"], hold_sec)

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


def _positive_control(hold_sec: float) -> bool:
    """合成データで、保持時間モデルを使った区間でも同時発火が検出されることを
    確認する (前回と同じ合成シナリオ、区間の作り方だけ新方式に更新)。"""
    ep1 = _build_episodes(
        np.array([10.0]), np.array([10.0]), np.array(["baseline"]), hold_sec)
    ep2 = _build_episodes(
        np.array([10.0 + hold_sec * 0.5]), np.array([10.0 + hold_sec * 0.5]),
        np.array(["landing"]), hold_sec)
    ov = _intervals_overlap(ep1[0], ep2[0])
    ok = ov is not None and ov > 0.0
    ep2_no = _build_episodes(
        np.array([10.0 + hold_sec * 3]), np.array([10.0 + hold_sec * 3]),
        np.array(["baseline"]), hold_sec)
    ov_no = _intervals_overlap(ep1[0], ep2_no[0])
    ok_no = ov_no is None
    print(f"  [control] hold_sec={hold_sec:.2f}: 陽性={'OK' if ok else 'NG'}(重なり={ov}) "
          f"陰性={'OK' if ok_no else 'NG'}(重なり={ov_no})")
    return ok and ok_no


def _scan(hold_sec: float) -> tuple[dict, list]:
    npz_paths = sorted(glob.glob(f"{NPZ_DIR}/*.npz"))
    per_video: dict[str, list[GameStats]] = {}
    all_games: list[GameStats] = []
    for path in npz_paths:
        d = np.load(path, allow_pickle=True)
        video_ids = np.unique(d["video_id"])
        for vid in video_ids:
            vid_mask = d["video_id"] == vid
            for g in np.unique(d["game_idx"][vid_mask]):
                rows_mask = vid_mask & (d["game_idx"] == g)
                stats = _analyze_game(str(vid), int(g), rows_mask, d, hold_sec)
                per_video.setdefault(str(vid), []).append(stats)
                all_games.append(stats)
    return per_video, all_games


def _pooled_summary(all_games: list) -> dict:
    total_rows = sum(g.n_rows for g in all_games)
    total_both = sum(g.n_rows_both_active for g in all_games)
    total_either = sum(
        g.n_rows_1p_active + g.n_rows_2p_active - g.n_rows_both_active for g in all_games)
    n_games = len(all_games)
    n_games_with_overlap = sum(1 for g in all_games if g.has_any_overlap)
    all_durations = [d for g in all_games for d in g.overlap_durations_sec]
    return {
        "total_rows": total_rows,
        "pct_both_rows": total_both / total_rows * 100 if total_rows else float("nan"),
        "pct_either_rows": total_either / total_rows * 100 if total_rows else float("nan"),
        "pct_both_of_either": (
            total_both / total_either * 100 if total_either else float("nan")),
        "n_games": n_games,
        "n_games_with_overlap": n_games_with_overlap,
        "pct_games_never_overlap": (
            (n_games - n_games_with_overlap) / n_games * 100 if n_games else float("nan")),
        "pct_games_with_overlap": (
            n_games_with_overlap / n_games * 100 if n_games else float("nan")),
        "n_overlap_events": len(all_durations),
        "dur_min": float(np.min(all_durations)) if all_durations else None,
        "dur_median": float(np.median(all_durations)) if all_durations else None,
        "dur_p90": float(np.percentile(all_durations, 90)) if all_durations else None,
        "dur_max": float(np.max(all_durations)) if all_durations else None,
    }


def main() -> None:
    print("=== 保持時間モデルのスイープ (formula × 仮定連鎖数) ===")
    results = {}
    for formula_name, formula in FORMULAS.items():
        for n in CHAIN_COUNT_SWEEP:
            hold_sec = formula(n)
            key = f"{formula_name}_N{n}"
            print(f"\n--- {key} (hold_sec={hold_sec:.2f}秒) ---")
            assert _positive_control(hold_sec), f"{key}: 陽性/陰性対照に失敗"
            per_video, all_games = _scan(hold_sec)
            summary = _pooled_summary(all_games)
            results[key] = {"hold_sec": hold_sec, "summary": summary}
            print(f"  両者同時行%={summary['pct_both_rows']:.3f}% "
                  f"片側以上行%={summary['pct_either_rows']:.3f}% "
                  f"同時無し試合%={summary['pct_games_never_overlap']:.1f}% "
                  f"エピソード数={summary['n_overlap_events']} "
                  f"持続(中央値/p90/最大)="
                  f"{summary['dur_median']}/{summary['dur_p90']}/{summary['dur_max']}")
    # 層別 (代表として default_capped5s, N=4 のみ動画別を出す)
    per_video, all_games = _scan(FORMULAS["default_capped5s"](4))
    print("\n=== 動画別 (層別、default_capped5s N=4 hold_sec=1.2秒) ===")
    for vid, games in sorted(per_video.items()):
        v_rows = sum(g.n_rows for g in games)
        v_both = sum(g.n_rows_both_active for g in games)
        v_overlap_games = sum(1 for g in games if g.has_any_overlap)
        pct_both = v_both / v_rows * 100 if v_rows else float("nan")
        print(f"  video={vid:>10} games={len(games):>2} 同時行%={pct_both:>6.3f}% "
              f"同時発火あり試合={v_overlap_games}/{len(games)}")

    out_path = Path("data/verify/mutual_exchange_frequency_v2_2026-08-21.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
