"""窒息(is_dead)長時間持続だが実際はプレイ継続している事象の定量計装。

読み込み専用の診断スクリプト (本体コード変更なし)。
data/verify/zenchi_render_slide_exit_guard_v2_2026-08-22/*.npz (8区間、
TimelineDumpRow ダンプ) を読み、is_dead1/is_dead2 が True の連続区間を
抽出し、その区間中に:
  - state (state1/state2) が CHAIN/GRAVITY_SETTLE/TSUMO_FALL/OJAMA_FALL
    (=STABLE でない = 生きて動いている) を含むか
  - game_idx が遷移しているか (試合終了があったか)
を判定する。

注意: dump 行は「settled 更新の瞬間」にのみ記録される
(scripts/visualize_advantage_overlay.py:5192 `if dump_timeline_path is not None:`
は line 5048 の `if b1 is not None and b2 is not None and settled:` の
内側)。--per-side-settled 下では settled は「片側だけ STABLE でも True」
(OR条件、同ファイル5029-5033行) になるため、記録される state1/state2 は
b1/b2 (凍結盤面) の更新条件とは非同期でありうる。b1 は
`r.p1.state == STABLE` の瞬間にのみ更新される (同ファイル4979-4980行付近)。
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np

SEG_DIR = Path(
    "data/verify/zenchi_render_slide_exit_guard_v2_2026-08-22"
)
OUT_DIR = Path("logs/is_dead_persist_2026-08-23")

ALIVE_STATES = {"CHAIN", "GRAVITY_SETTLE", "TSUMO_FALL", "OJAMA_FALL"}


@dataclass
class Run:
    seg: str
    side: str  # "1P" or "2P"
    t_start: float
    t_end: float
    duration: float
    n_rows: int
    game_idx_start: int
    game_idx_end: int
    game_idx_changed: bool
    states_seen: tuple  # own side states during run
    alive_state_frac: float  # own side non-STABLE 割合 (行数ベース)
    other_state_alive_frac: float  # 相手側 non-STABLE 割合 (連鎖している可能性)
    t_mid: float


def extract_runs(npz_path: Path) -> list[Run]:
    d = np.load(npz_path, allow_pickle=True)
    t = d["t_sec"]
    game_idx = d["game_idx"]
    seg = npz_path.stem
    runs: list[Run] = []
    for side, is_dead_key, state_key, other_state_key in (
        ("1P", "is_dead1", "state1", "state2"),
        ("2P", "is_dead2", "state2", "state1"),
    ):
        is_dead = d[is_dead_key]
        state = d[state_key]
        other_state = d[other_state_key]
        n = len(is_dead)
        i = 0
        while i < n:
            if not bool(is_dead[i]):
                i += 1
                continue
            j = i
            while j < n and bool(is_dead[j]):
                j += 1
            # run は [i, j)
            seg_states = state[i:j]
            seg_other_states = other_state[i:j]
            alive_own = sum(1 for s in seg_states if s in ALIVE_STATES)
            alive_other = sum(1 for s in seg_other_states if s in ALIVE_STATES)
            duration = float(t[j - 1] - t[i]) if j - 1 > i else 0.0
            runs.append(Run(
                seg=seg, side=side,
                t_start=float(t[i]), t_end=float(t[j - 1]),
                duration=duration, n_rows=j - i,
                game_idx_start=int(game_idx[i]), game_idx_end=int(game_idx[j - 1]),
                game_idx_changed=int(game_idx[i]) != int(game_idx[j - 1]),
                states_seen=tuple(sorted(set(seg_states.tolist()))),
                alive_state_frac=alive_own / (j - i),
                other_state_alive_frac=alive_other / (j - i),
                t_mid=float((t[i] + t[j - 1]) / 2.0),
            ))
            i = j
    return runs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(SEG_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] npz が見つかりません: {SEG_DIR}", file=sys.stderr)
        sys.exit(1)

    all_runs: list[Run] = []
    for p in npz_files:
        all_runs.extend(extract_runs(p))

    print(f"=== 全 {len(npz_files)} 区間、is_dead True 連続区間 総数: {len(all_runs)} ===")
    total_dead_time = sum(r.duration for r in all_runs)
    print(f"is_dead True 区間の合計時間(両側合算): {total_dead_time:.1f} 秒")

    durations = np.array([r.duration for r in all_runs])
    if len(durations) > 0:
        print(f"持続時間分布: median={np.median(durations):.2f}s "
              f"p90={np.percentile(durations, 90):.2f}s "
              f"max={durations.max():.2f}s mean={durations.mean():.2f}s")

    # 「誤判定」候補: 持続が一定以上長く(閾値1.0秒以上、真の窒息はそもそも
    # 試合が終わるので長続きしないはず)、かつ own state に alive state を含む
    # (=連鎖/設置動作が観測された)、かつ game_idx が変わっていない(試合が
    # 終わっていない)。
    THRESH = 1.0
    suspects = [
        r for r in all_runs
        if r.duration >= THRESH and not r.game_idx_changed and r.alive_state_frac > 0.0
    ]
    print(f"\n=== 疑わしい区間 (持続>={THRESH}s かつ own-state に生存状態あり "
          f"かつ試合未終了): {len(suspects)} 件 ===")
    sus_durations = np.array([r.duration for r in suspects]) if suspects else np.array([])
    if len(sus_durations) > 0:
        print(f"  合計時間: {sus_durations.sum():.1f}秒")
        print(f"  分布: median={np.median(sus_durations):.2f}s "
              f"p90={np.percentile(sus_durations, 90):.2f}s "
              f"max={sus_durations.max():.2f}s")

    # 動画全長 (npz t_sec レンジの合計、8区間おおよそ連結)
    total_video_span = 0.0
    for p in npz_files:
        d = np.load(p, allow_pickle=True)
        t = d["t_sec"]
        if len(t) > 1:
            total_video_span += float(t[-1] - t[0])
    print(f"\n動画全長 (8区間 t_sec レンジ合計、概算): {total_video_span:.1f}秒 "
          f"({total_video_span/60:.1f}分)")
    if total_video_span > 0:
        print(f"  疑わしい区間の合計時間 / 動画全長 = "
              f"{sus_durations.sum()/total_video_span*100:.3f}%" if len(sus_durations) else "  0%")
        print(f"  is_dead True 区間(全部)の合計時間 / 動画全長 = "
              f"{total_dead_time/total_video_span*100:.3f}%")

    # 上位 (長い順) を一覧表示
    suspects_sorted = sorted(suspects, key=lambda r: -r.duration)
    print(f"\n=== 疑わしい区間 上位20件 (長い順) ===")
    for r in suspects_sorted[:20]:
        print(f"  {r.seg} side={r.side} t=[{r.t_start:.2f},{r.t_end:.2f}] "
              f"dur={r.duration:.2f}s n_rows={r.n_rows} "
              f"game_idx={r.game_idx_start}->{r.game_idx_end} "
              f"own_states={r.states_seen} alive_own_frac={r.alive_state_frac:.2f} "
              f"alive_other_frac={r.other_state_alive_frac:.2f}")

    # 全区間 (誤判定候補に限らず) の一覧も参考までに件数だけ
    all_sorted = sorted(all_runs, key=lambda r: -r.duration)
    print(f"\n=== 全 is_dead=True 連続区間 上位10件 (フィルタなし、長い順) ===")
    for r in all_sorted[:10]:
        print(f"  {r.seg} side={r.side} t=[{r.t_start:.2f},{r.t_end:.2f}] "
              f"dur={r.duration:.2f}s game_idx={r.game_idx_start}->{r.game_idx_end} "
              f"own_states={r.states_seen}")

    # 結果を npz に保存 (後続の実画面切り出しスクリプトで使う)
    out_npz = OUT_DIR / "runs.npz"
    if all_runs:
        np.savez(
            out_npz,
            seg=np.array([r.seg for r in all_runs], dtype=object),
            side=np.array([r.side for r in all_runs], dtype=object),
            t_start=np.array([r.t_start for r in all_runs]),
            t_end=np.array([r.t_end for r in all_runs]),
            duration=np.array([r.duration for r in all_runs]),
            game_idx_changed=np.array([r.game_idx_changed for r in all_runs]),
            alive_state_frac=np.array([r.alive_state_frac for r in all_runs]),
            other_state_alive_frac=np.array([r.other_state_alive_frac for r in all_runs]),
            states_seen=np.array([",".join(r.states_seen) for r in all_runs], dtype=object),
        )
        print(f"\n[保存] {out_npz}")


if __name__ == "__main__":
    main()
