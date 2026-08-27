"""60fps全フレーム版 (A) と stride2/実効30fps版 (B) の npz を突き合わせる。

2026-08-12 user指示「60フレーム全部読む必要はない、間引きでNGが出るケースを
洗い出す」への回答用。既存 scripts/_diag_fps_normalize_ab_2026-07-30.py と同じ
「直近有効snapshot対応」方式を流用し、score/tsumo_count/all_clear_pending の
突き合わせも追加する (2026-08-12 タスク要件)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

STALE_LIMIT_SEC: float = 0.05  # t_sec±0.05s許容
SIDES = ("1P", "2P")


def load(path: Path) -> dict:
    """npz から比較列を取り出す。"""
    z = np.load(path, allow_pickle=True)
    keys = [
        "side", "t_sec", "grids", "next1_a", "next1_b",
        "chain_trigger_sec", "score", "tsumo_count", "all_clear_pending",
        "frame_idx",
    ]
    return {k: np.asarray(z[k]) for k in keys if k in z}


def compare(full_path: Path, stride_path: Path) -> None:
    """full60fps を基準に stride2 の乖離を表示する。"""
    a = load(full_path)
    s = load(stride_path)
    print(f"snapshot数: full60fps={len(a['t_sec'])} stride2={len(s['t_sec'])}")

    exact_match = 0
    tol_match_board_diff = 0
    tol_match_board_same = 0
    no_match = 0
    score_mismatch = 0
    tsumo_mismatch = 0
    acp_mismatch = 0
    cell_diffs: list[int] = []
    mismatch_events: list[str] = []

    for side in SIDES:
        ma = a["side"].astype(str) == side
        ms = s["side"].astype(str) == side
        ta = a["t_sec"][ma].astype(float)
        ga = a["grids"][ma]
        sca = a["score"][ma]
        tca = a["tsumo_count"][ma]
        aca = a["all_clear_pending"][ma]
        ts = s["t_sec"][ms].astype(float)
        gs = s["grids"][ms]
        scs = s["score"][ms]
        tcs = s["tsumo_count"][ms]
        acs = s["all_clear_pending"][ms]
        if len(ta) == 0 or len(ts) == 0:
            continue
        order = np.argsort(ta)
        ta_o, ga_o = ta[order], ga[order]
        sca_o, tca_o, aca_o = sca[order], tca[order], aca[order]

        for i, t0 in enumerate(ts):
            # 完全一致 (同一t_sec) を先に試す
            exact_idx = np.where(np.abs(ta_o - t0) < 1e-6)[0]
            if len(exact_idx) > 0:
                k = int(exact_idx[0])
                exact_match += 1
            else:
                k = int(np.searchsorted(ta_o, t0, side="right")) - 1
                if k < 0 or abs(t0 - ta_o[k]) > STALE_LIMIT_SEC:
                    # ±0.05s 許容でも対応が無い
                    k2 = int(np.argmin(np.abs(ta_o - t0)))
                    if abs(t0 - ta_o[k2]) <= STALE_LIMIT_SEC:
                        k = k2
                    else:
                        no_match += 1
                        continue

            g1, g2 = gs[i], ga_o[k]
            if np.array_equal(g1, g2):
                tol_match_board_same += 1
            else:
                tol_match_board_diff += 1
                d = int((g1 != g2).sum())
                cell_diffs.append(d)
                if len(mismatch_events) < 30:
                    mismatch_events.append(
                        f"  side={side} t_stride2={t0:.3f}s "
                        f"t_full60={ta_o[k]:.3f}s cell_diff={d}"
                    )
            if int(scs[i]) != int(sca_o[k]):
                score_mismatch += 1
            if int(tcs[i]) != int(tca_o[k]):
                tsumo_mismatch += 1
            if int(acs[i]) != int(aca_o[k]):
                acp_mismatch += 1

    total = exact_match + tol_match_board_same + tol_match_board_diff
    print(f"完全一致(同一t_sec): {exact_match}")
    print(f"許容一致(±{STALE_LIMIT_SEC}s)・盤面も一致: {tol_match_board_same}")
    print(f"許容一致・盤面不一致: {tol_match_board_diff}")
    print(f"対応なし(取りこぼし疑い): {no_match}")
    print(f"score不一致: {score_mismatch}")
    print(f"tsumo_count不一致: {tsumo_mismatch}")
    print(f"all_clear_pending不一致: {acp_mismatch}")
    if cell_diffs:
        arr = np.array(cell_diffs)
        print(
            f"盤面不一致時のセル差: 中央値{np.median(arr):.0f} "
            f"最大{arr.max()} (78セル中)"
        )
    if mismatch_events:
        print("不一致イベント (最大30件):")
        for line in mismatch_events:
            print(line)

    a_chain = a.get("chain_trigger_sec")
    s_chain = s.get("chain_trigger_sec")
    if a_chain is not None and s_chain is not None:
        a_rate = float(np.mean(~np.isnan(a_chain))) if len(a_chain) else 0.0
        s_rate = float(np.mean(~np.isnan(s_chain))) if len(s_chain) else 0.0
        print(
            f"chain_trigger_sec 非NaN率: full60fps={a_rate*100:.2f}% "
            f"stride2={s_rate*100:.2f}%"
        )


def main() -> None:
    """CLI 引数 (full_npz, stride_npz) で比較を実行する。"""
    if len(sys.argv) != 3:
        print("usage: _compare_fps_stride_ab_2026-08-12.py <full60fps.npz> <stride2.npz>")
        return
    compare(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
