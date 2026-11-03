"""案A (正味の予告) の再現検証: 計装 ledger_chain_diag_20261025 の評価行と 5 場面で、net が確定後の値と
一致する割合を再計算し、RESULT.md の数字 (符号一致 98.2%・完全一致 4316 行など) を再現する。読むだけ。

使い方: python -m scripts.verify_b3_net_ledger_diag [--diag-dir DIR] [--json OUT.json]
評価行 CSV (rows_24b.csv / rows_23y_*.csv) は計装の extract.py が作ったもの。製品コードの純関数
`src.b3_net_ledger.derive_net_columns` を通した値で数える (計装側の独自計算は使わない)。
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.b3_column_spec import Mask
from src.b3_net_ledger import SOURCE_COLUMNS, derive_net_columns

DEFAULT_DIAG_DIR = Path(_public_dev_text('<work>/ledger_chain_diag_20261025') if os.name == "nt"
                        else _public_dev_text('<work>/ledger_chain_diag_20261025'))
SETTLE_WINDOW_SEC = 4.0        # 計装 future_truth.py と同じ: 両側 inflight=0 になるまで見る上限
ICON_LOOKAHEAD_SEC = 2.0       # T2: 2 秒後のアイコン差 (±ICON_LOOKAHEAD_TOL_SEC)
ICON_LOOKAHEAD_TOL_SEC = 0.5
PCT_DIGITS = 1
# RESULT.md / summary.json に書かれた再現目標 (符号一致 %, 母数)。
EXPECTED = {"T1_net": (98.2, 5336), "T1_exact_net": (100.0, 4243), "T2_net": (98.9, 4811),
            "T1_icon_now": (85.1, 4185), "T1_pending": (87.8, 4775), "T1_naive": (53.9, 5276)}
EXPECTED_EXACT = {"exact_rows": 4316, "single_sided": 3885, "both_sided": 431}
# RESULT.md の場面の時刻 (記録秒) と、そこでの (受け側向き差, naive, 現アイコン差)。
SCENE_POINTS = {"S1_single_big_chain": (699.47, -173.0, 95.0, -15.0),
                "S2_both_chain_counter": (2234.15, 17.0, -3.0, None),
                "S4_interrupt": (948.82, -186.0, 14.0, -37.0)}
POINT_TOL_SEC = 0.1


def load_rows(diag_dir: Path) -> list[tuple[str, pd.DataFrame]]:
    """評価行 CSV を (ファイル名, DataFrame) で返す。試合キーは epoch_game、時刻順。"""
    files = [diag_dir / "rows_24b.csv", *map(Path, sorted(glob.glob(str(diag_dir / "rows_23y_*.csv"))))]
    out = []
    for path in files:
        df = pd.read_csv(path)
        df = pd.concat([df, (df["epoch"].astype(str) + "_" + df["game"].astype(str)).rename("key")], axis=1)
        out.append((path.name, df.sort_values(["key", "t_sec"]).reset_index(drop=True)))
    return out


def add_net(df: pd.DataFrame) -> pd.DataFrame:
    """製品の導出関数で net 列を足す。CSV に無い時間軸の入力 (ta_q_*・mk_ta) は欠測・非該当で埋める。"""
    src: dict[str, Any] = {k: df[k].to_numpy(dtype=np.float64) for k in SOURCE_COLUMNS if k in df.columns}
    for side in ("p1", "p2"):
        src.update({f"ta_q_med_{side}": np.nan, f"ta_q_fast_{side}": np.nan, f"mk_ta_{side}": float(Mask.NOT_APPLICABLE)})
        src.setdefault(f"ta_i_sl_{side}", np.nan)
    src = {k: (np.full(len(df), v) if np.isscalar(v) else v) for k, v in src.items()}
    cols = derive_net_columns(src)
    added = pd.DataFrame({"net_p1": cols["net_pending_p1"], "net_p2": cols["net_pending_p2"],
                          "df_net": cols["df_sl_net"], "net_mask": cols["mk_sl_net_p1"],
                          "ta_i_slnet_p1": cols["ta_i_slnet_p1"], "ta_i_slnet_p2": cols["ta_i_slnet_p2"]}, index=df.index)
    return pd.concat([df.copy(), added], axis=1)


def _settle_index(g: pd.DataFrame, r: int) -> int | None:
    """行 r の後で最初に両側 inflight=0 になる行 (SETTLE_WINDOW_SEC 以内)。無ければ None。"""
    t, i1, i2 = g["t_sec"].to_numpy(), g["sl_inflight_p1"].to_numpy(), g["sl_inflight_p2"].to_numpy()
    for j in range(r + 1, len(g)):
        if t[j] - t[r] > SETTLE_WINDOW_SEC:
            return None
        if np.isfinite(i1[j]) and np.isfinite(i2[j]) and i1[j] == 0 and i2[j] == 0:
            return j
    return None


def _icon_after(g: pd.DataFrame, r: int) -> float:
    """2 秒後 (±0.5 秒) のアイコン差。無ければ NaN。"""
    t, di = g["t_sec"].to_numpy(), g["df_icon_corr_total"].to_numpy()
    j = int(np.searchsorted(t, t[r] + ICON_LOOKAHEAD_SEC))
    cand = [x for x in (j - 1, j) if 0 <= x < len(g) and abs(t[x] - (t[r] + ICON_LOOKAHEAD_SEC)) <= ICON_LOOKAHEAD_TOL_SEC
            and np.isfinite(di[x])]
    return float(di[min(cand, key=lambda x: abs(t[x] - (t[r] + ICON_LOOKAHEAD_SEC)))]) if cand else float("nan")


def collect_pairs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """inflight>0 の各行について、候補 (pending / naive / 旧 recv / net / アイコン) と確定後の値 T1・T2 を集める。"""
    recs: list[dict[str, Any]] = []
    for _key, g in df.groupby("key"):
        g = g.reset_index(drop=True)
        p1, p2, i1, i2 = (g[f"sl_{s}_{n}"].to_numpy() for s in ("pending", "inflight") for n in ("p1", "p2"))
        gen = (g["sl_generated_p1"].to_numpy(), g["sl_generated_p2"].to_numpy())
        drop = (g["sl_dropped_p1"].to_numpy(), g["sl_dropped_p2"].to_numpy())
        for r in range(len(g)):
            if not (np.isfinite(i1[r]) and np.isfinite(i2[r]) and np.isfinite(p1[r]) and np.isfinite(p2[r])
                    and (i1[r] > 0 or i2[r] > 0)):
                continue
            s = _settle_index(g, r)
            same_drop = s is not None and drop[0][s] == drop[0][r] and drop[1][s] == drop[1][r]
            exact = bool(same_drop and gen[0][s] - gen[0][r] == i1[r] and gen[1][s] - gen[1][r] == i2[r])
            recs.append(dict(pending=p1[r] - p2[r], naive=p1[r] - p2[r] + i1[r] - i2[r],
                             recv=p1[r] - p2[r] + i2[r] - i1[r], net=g["df_net"][r], icon=g["df_icon_corr_total"][r],
                             T1=(p1[s] - p2[s]) if same_drop else np.nan, T2=_icon_after(g, r), exact=exact,
                             both=int(i1[r] > 0 and i2[r] > 0),
                             net_hit=bool(exact and g["net_p1"][r] == p1[s] and g["net_p2"][r] == p2[s])))
    return recs


def sign_agree(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> tuple[float | None, int]:
    """符号一致 % と母数 (両方が有限かつ非 0 の組だけ)。"""
    m = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    if mask is not None:
        m &= mask
    n = int(m.sum())
    return (round(100 * float((np.sign(a[m]) == np.sign(b[m])).mean()), PCT_DIGITS) if n else None, n)


def summarize(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """RESULT.md の表と同じ指標。完全一致は exact (後続段なし) の行で数える。"""
    R = pd.DataFrame(recs)
    col = {c: R[c].to_numpy(dtype=np.float64) for c in ("pending", "naive", "recv", "net", "icon", "T1", "T2")}
    out: dict[str, Any] = {"inflight_rows": len(R)}
    for cand in ("pending", "naive", "net", "icon"):
        out["T1_" + ("icon_now" if cand == "icon" else cand)] = sign_agree(col[cand], col["T1"])
    out["T1_exact_net"] = sign_agree(col["net"], col["T1"], R["exact"].to_numpy())
    out["T2_net"] = sign_agree(col["net"], col["T2"])
    out["net_equals_old_recv"] = bool(np.array_equal(col["net"], col["recv"]))
    ex = R[R["exact"]]
    out.update(exact_rows=len(ex), single_sided=int((ex["both"] == 0).sum()), both_sided=int((ex["both"] == 1).sum()),
               exact_net_hit=int(ex["net_hit"].sum()),
               both_net_hit=int(ex.loc[ex["both"] == 1, "net_hit"].sum()))
    return out


def ta_orientation(frames: list[pd.DataFrame]) -> dict[str, int]:
    """実データで ta_i_sl の向き (攻撃側 s の I = 反対側の sl_pending) と ta_i_slnet の定義を確かめる。

    ta_i_sl が有限の行 (n) で、ta_i_sl==反対側の pending、ta_i_slnet==反対側の net、
    両側 inflight=0 なら ta_i_slnet==ta_i_sl、が全行で成り立つ。成り立たない行数 (bad_*) は 0 が期待値。
    """
    out = dict(n=0, bad_sl_side=0, bad_net_side=0, n_idle=0, bad_idle=0, net_differs=0)
    for df in frames:
        for side, other in (("p1", "p2"), ("p2", "p1")):
            m = np.isfinite(df[f"ta_i_sl_{side}"].to_numpy())
            d = df[m]
            idle = (d["sl_inflight_p1"] == 0) & (d["sl_inflight_p2"] == 0)
            out["n"] += len(d)
            out["bad_sl_side"] += int((d[f"ta_i_sl_{side}"] != d[f"sl_pending_{other}"]).sum())
            out["bad_net_side"] += int((d[f"ta_i_slnet_{side}"] != d[f"net_{other}"]).sum())
            out["n_idle"] += int(idle.sum())
            out["bad_idle"] += int((d.loc[idle, f"ta_i_slnet_{side}"] != d.loc[idle, f"ta_i_sl_{side}"]).sum())
            out["net_differs"] += int((d[f"ta_i_slnet_{side}"] != d[f"ta_i_sl_{side}"]).sum())
    return out


def scene_report(diag_dir: Path) -> list[dict[str, Any]]:
    """5 場面: 全行で df_sl_net が計装の dsum_recv と一致するか、RESULT.md の時刻の値が出るか。"""
    scenes = json.loads((diag_dir / "scenes.json").read_text(encoding="utf-8"))
    report = []
    for sc in scenes:
        df = pd.DataFrame(sc["rows"])
        for side in ("p1", "p2"):
            df[f"ta_i_sl_{side}"], df[f"mk_sl_{side}"], df[f"mk_sl_inflight_{side}"] = np.nan, 0.0, 0.0
        net = add_net(df)
        diff = float(np.nanmax(np.abs(net["df_net"].to_numpy() - df["dsum_recv"].to_numpy())))
        item: dict[str, Any] = {"id": sc["id"], "rows": len(df), "max_abs_diff_vs_dsum_recv": diff}
        if sc["id"] in SCENE_POINTS:
            t, want_net, want_naive, want_icon = SCENE_POINTS[sc["id"]]
            k = int((df["t_sec"] - t).abs().idxmin())
            item.update(t_sec=float(df["t_sec"][k]), t_ok=bool(abs(df["t_sec"][k] - t) <= POINT_TOL_SEC),
                        df_sl_net=float(net["df_net"][k]), want_net=want_net, dsum_naive=float(df["dsum_naive"][k]),
                        want_naive=want_naive, dicon=float(df["dicon"][k]), want_icon=want_icon)
        report.append(item)
    return report


def run(diag_dir: Path) -> dict[str, Any]:
    recs: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for _name, df in load_rows(diag_dir):
        frames.append(add_net(df))
        recs += collect_pairs(frames[-1])
    return {"summary": summarize(recs), "ta_orientation": ta_orientation(frames), "scenes": scene_report(diag_dir)}


def reproduced(result: dict[str, Any]) -> dict[str, bool]:
    """目標値との一致 (符号一致%・母数・完全一致の行数)。"""
    s = result["summary"]
    ok = {k: tuple(s[k]) == v for k, v in EXPECTED.items()}
    ok.update({k: s[k] == v for k, v in EXPECTED_EXACT.items()})
    ok["exact_net_all_hit"] = s["exact_net_hit"] == s["exact_rows"] and s["both_net_hit"] == s["both_sided"]
    ok["net_equals_old_recv"] = s["net_equals_old_recv"]
    t = result["ta_orientation"]
    ok["ta_orientation"] = t["n"] > 0 and t["bad_sl_side"] == t["bad_net_side"] == t["bad_idle"] == 0
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diag-dir", type=Path, default=DEFAULT_DIAG_DIR)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    result = run(a.diag_dir)
    result["reproduced"] = reproduced(result)
    print(json.dumps(result, ensure_ascii=False, indent=1, default=float))
    if a.json:
        a.json.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
