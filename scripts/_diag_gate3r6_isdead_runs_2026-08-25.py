"""Gate 3R-6: タイムライン dump から is_dead 誤判定 run と STABLE凍結痕跡を集計する。

読み込み専用。任意の timeline dump npz (旧 8/22 OFF 構成 seg01 / 新 8/25 ON 構成)
に対して同一ロジックで走査し、対象2 (W32 is_dead 誤判定) の出現量を
「件数 / 合計秒数 / 発生試合数 / 母数試合数」で出す。

run の定義 (8/24 の `_diag_residual_isdead_classify_2026-08-24.py` と同一):
  - is_dead の連続 True 区間
  - 区間内で game_idx が変わらない (試合終了の真の死亡を除外)
  - own 側 state に活動状態 (CHAIN/GRAVITY_SETTLE/TSUMO_FALL/OJAMA_FALL) を含む
    (= 凍結盤面への誤判定が疑われる)

加えて対象1の痕跡として「score が伸びている (連鎖中の証拠) のに state が
STABLE のまま」の区間を dump 粒度 (settled 更新 ~0.15s) で抽出する:
  - score{side} が前行より増加した行を「連鎖進行の証拠」とみなし、
    証拠行から state が CHAIN になるまでの遅延を測る。

使い方:
  python scripts/_diag_gate3r6_isdead_runs_2026-08-25.py <npz...> \
      --t-lo 18.0 --t-hi 336.0 --label off_baseline \
      --out-dir data/verify/gate3r6_diag_2026-08-25
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

ALIVE_STATES = {"CHAIN", "GRAVITY_SETTLE", "TSUMO_FALL", "OJAMA_FALL"}


def load_arrays(paths: list[Path]) -> dict:
    """npz 群を連結して列辞書を返す (t_sec 昇順を前提にファイル名順)。"""
    cols = ["t_sec", "game_idx", "is_dead1", "is_dead2",
            "state1", "state2", "score1", "score2", "b1_hash", "b2_hash"]
    acc: dict[str, list] = {c: [] for c in cols}
    for p in sorted(paths):
        f = np.load(p, allow_pickle=True)
        for c in cols:
            acc[c].append(f[c])
    return {c: np.concatenate(acc[c]) for c in cols}


def extract_isdead_runs(d: dict, side: str, t_lo: float, t_hi: float) -> list[dict]:
    """is_dead 誤判定疑い run を抽出する (定義は module docstring)。"""
    m = (d["t_sec"] >= t_lo) & (d["t_sec"] <= t_hi)
    t = d["t_sec"][m]
    gi = d["game_idx"][m]
    dead = d[f"is_dead{side}"][m]
    own = d[f"state{side}"][m]
    other = d[f"state{'2' if side == '1' else '1'}"][m]
    bh = d[f"b{side}_hash"][m]
    runs: list[dict] = []
    n = len(t)
    i = 0
    while i < n:
        if not dead[i]:
            i += 1
            continue
        j = i
        while j < n and dead[j] and gi[j] == gi[i]:
            j += 1
        own_states = set(str(s) for s in own[i:j])
        other_states = set(str(s) for s in other[i:j])
        alive_own = bool(own_states & ALIVE_STATES)
        gi_terminal = (j < n and gi[min(j, n - 1)] != gi[i]) or (j == n)
        # run が試合境界で終わる (= 真の死亡の可能性) かの目印
        runs.append(dict(
            side=side,
            game_idx=int(gi[i]),
            t_start=round(float(t[i]), 3), t_end=round(float(t[j - 1]), 3),
            duration=round(float(t[j - 1] - t[i]), 3),
            n_rows=int(j - i),
            own_states=",".join(sorted(own_states)),
            other_states=",".join(sorted(other_states)),
            suspicious=bool(alive_own),
            ends_at_game_boundary=bool(gi_terminal),
            frozen_hash_unchanged=bool(len(set(bh[i:j].tolist())) == 1),
        ))
        i = j
    return runs


def extract_stable_lag(d: dict, side: str, t_lo: float, t_hi: float) -> list[dict]:
    """score 増加 (連鎖進行の証拠) から state=CHAIN 到達までの遅延を測る。

    dump は settled 更新イベント (~0.15s) 粒度。score{side} が前行比 +40 以上
    (最小の 4 個消し素点 40 = 物理量由来、シーン逆算ではない) 増えた行を
    証拠行とし、その時点の state が STABLE なら「連鎖しているのに STABLE」
    候補。以後 state が CHAIN になるまでの時間を lag とする。
    """
    m = (d["t_sec"] >= t_lo) & (d["t_sec"] <= t_hi)
    t = d["t_sec"][m]
    gi = d["game_idx"][m]
    sc = d[f"score{side}"][m].astype(np.int64)
    own = d[f"state{side}"][m]
    n = len(t)
    out: list[dict] = []
    i = 1
    while i < n:
        if gi[i] == gi[i - 1] and sc[i] - sc[i - 1] >= 40 and str(own[i]) == "STABLE":
            # state が CHAIN になる行を探す (同一試合内)
            j = i
            reached = None
            while j < n and gi[j] == gi[i]:
                if str(own[j]) == "CHAIN":
                    reached = j
                    break
                j += 1
            lag = (float(t[reached] - t[i]) if reached is not None else None)
            out.append(dict(
                side=side, game_idx=int(gi[i]),
                t_evidence=round(float(t[i]), 3),
                score_jump=int(sc[i] - sc[i - 1]),
                lag_to_chain=(round(lag, 3) if lag is not None else None),
            ))
            # 同一連鎖の連続 score 増を重複計上しない: CHAIN 到達 or score 増が
            # 途切れるまでスキップ
            k = i + 1
            while k < n and gi[k] == gi[i] and sc[k] > sc[k - 1] and str(own[k]) == "STABLE":
                k += 1
            i = max(k, (reached + 1) if reached is not None else k)
        else:
            i += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", type=Path)
    ap.add_argument("--t-lo", type=float, default=0.0)
    ap.add_argument("--t-hi", type=float, default=1e18)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/verify/gate3r6_diag_2026-08-25"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    d = load_arrays(a.npz)
    all_runs: list[dict] = []
    all_lags: list[dict] = []
    for side in ("1", "2"):
        all_runs.extend(extract_isdead_runs(d, side, a.t_lo, a.t_hi))
        all_lags.extend(extract_stable_lag(d, side, a.t_lo, a.t_hi))

    runs_csv = a.out_dir / f"isdead_runs_{a.label}.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_runs[0].keys()) if all_runs
                           else ["side"])
        w.writeheader()
        w.writerows(all_runs)
    lags_csv = a.out_dir / f"stable_lag_{a.label}.csv"
    with lags_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_lags[0].keys()) if all_lags
                           else ["side"])
        w.writeheader()
        w.writerows(all_lags)

    # 集計
    m = (d["t_sec"] >= a.t_lo) & (d["t_sec"] <= a.t_hi)
    games = sorted(set(int(g) for g in d["game_idx"][m]))
    sus = [r for r in all_runs if r["suspicious"] and not r["ends_at_game_boundary"]]
    sus_games = sorted(set(r["game_idx"] for r in sus))
    lag_vals = [r["lag_to_chain"] for r in all_lags if r["lag_to_chain"] is not None]
    summary = dict(
        label=a.label,
        window=[a.t_lo, a.t_hi],
        n_rows=int(m.sum()),
        games_in_window=games,
        n_games=len(games),
        isdead_runs_total=len(all_runs),
        isdead_runs_suspicious=len(sus),
        isdead_suspicious_total_sec=round(sum(r["duration"] for r in sus), 2),
        isdead_suspicious_games=sus_games,
        isdead_suspicious_by_side=dict(Counter(r["side"] for r in sus)),
        stable_lag_events=len(all_lags),
        stable_lag_ge_1s=len([v for v in lag_vals if v >= 1.0]),
        stable_lag_never_chain=len([r for r in all_lags
                                    if r["lag_to_chain"] is None]),
        stable_lag_median=(round(float(np.median(lag_vals)), 3)
                           if lag_vals else None),
        stable_lag_max=(round(float(max(lag_vals)), 3) if lag_vals else None),
    )
    out_json = a.out_dir / f"summary_{a.label}.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[保存] {runs_csv}")
    print(f"[保存] {lags_csv}")
    print(f"[保存] {out_json}")


if __name__ == "__main__":
    main()
