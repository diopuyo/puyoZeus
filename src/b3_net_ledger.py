"""案A: 台帳の既存列から作る『受ける側から見た正味の予告』 (純関数・stateless・既定OFF)。

## 何を直すか
`sl_pending_pN` は「側 N が**受ける**予告」、`sl_inflight_pN` は「側 N が**生成している** (相手へ行く) 未反映量」で、
同じ `_p1/_p2` の綴りで向きが逆。両側の未反映量どうしは相殺もされない。この向きの混在のまま p1-p2 を取ると
予告アイコンの差と逆符号になる (逆符号 16.0% のうち 79% が向きの混在。
<work>/ledger_chain_diag_20261025/RESULT.md)。

## 定義 (入力は側 i の P_i=sl_pending_pi、G_i=sl_inflight_pi。いずれも 0 以上の個数)
ゲームの規則 (台帳 src/b3_score_ledger.py の `_convert`/`_offset_own`): 連鎖の得点で生じた n 個は**先に自分の受ける予告を
相殺**し、余りを相手の予告へ積む。両側の未反映量を順に決済した後の「受ける側の量」が net:

    x = P1 - G1,  y = P2 - G2
    net_pending_p1 = max(0, x, x - y)      net_pending_p2 = max(0, y, y - x)
    df_sl_net      = net_pending_p1 - net_pending_p2 = x - y   (= (P1+G2) - (P2+G1)、受ける側向きの差)

導出: P1 側を先に決済すると a1=max(0,x)、a2=P2+max(0,-x)、次に P2 側で p2''=max(0,a2-G2)=max(0,y,y-x)、
p1''=a1+max(0,G2-a2)=max(0,x,x-y)。P2 側を先にしても同じ (試験で両順序を 0..12 の全組と乱数で照合)。
つまり未来の得点反映順序が未観測でも net は一意。割り込み (新しい予告から相殺) は「どの塊が残るか」だけを変え、
総量は変えないので、個数の列である net には影響しない。この式は両側の inflight が 0 なら pending に一致する。

## 注意 (そのまま扱う)
全消しの +30 は式の積に含まれず inflight に出ない (確定時に載る。RESULT.md の S5)。net も確定まで 30 少ない。
着弾認識の遅れ・幻・取り逃し (台帳側の狂い) は直さない (案B/C)。inflight の期限切れ (FORMULA_EXPIRE_SEC) もそのまま。

## 側交換
net_pending_* と ta_*_slnet は『側の量』(SIDE_ABS)、df_sl_net は『差』(DIFF_RAW)。式は p1<->p2 の入れ替えで対称
(x<->y) なので、入力を側交換してから導出 = 導出してから側交換、が成り立つ (tests/test_b3_net_ledger.py)。

## 配信と学習の一致
行 1 つぶんの `derive_net_row` と、DataFrame/parquet 用の `derive_frame` は同じ `derive_net_columns` を通る。
`sl_pending` など既存の列だけが入力で、学習行を作り直さずに足せる (scripts/derive_b3_net_ledger_cols.py)。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import numpy as np

from src.b3_column_spec import (NET_LEDGER_COLUMN_NAMES, NET_TA_STEMS, NET_TA_VARIANTS, SIDES, Mask,
                                net_ledger_specs, nonsat, signed_nonsat)
from src.b3_timeaxis import UNCANCELLABLE_OJAMA

if TYPE_CHECKING:
    import pandas as pd

NAN = float("nan")
MASK_COLUMN_PREFIX = "mk_"
# 列仕様 (SPEC_BY_NAME) にこの列があるときだけ案A がON。行生成と仕様が食い違わないよう仕様を唯一の入口にする。
NET_SWITCH_COLUMN: str = NET_LEDGER_COLUMN_NAMES[0]
# 既存の to_frame (src/b3_rows.py の _DTYPES) と同じ dtype 対応。試験で一致を固定する。
FRAME_DTYPES: Mapping[str, str] = {"f8": "float64", "f4": "float32", "i1": "Int8"}


def _both(stem: str) -> tuple[str, ...]:
    return tuple(f"{stem}_{side}" for side in SIDES)


# 導出に必要な既存列 (これだけを読む)。足りない列は黙って落とさず例外にする。
SOURCE_COLUMNS: tuple[str, ...] = (
    *_both("sl_pending"), *_both("sl_inflight"), *_both("mk_sl"), *_both("mk_sl_inflight"),
    *_both("ta_i_sl"), *_both("ta_q_med"), *_both("ta_q_fast"), *_both("mk_ta"))


def _valid_count(*values: np.ndarray) -> np.ndarray:
    """有限で 0 以上のときだけ真 (負の個数は定義域外。伝播させず欠測にする)。"""
    ok = np.ones(np.shape(values[0]), dtype=bool)
    for v in values:
        ok &= np.isfinite(v) & (v >= 0.0)
    return ok


def settle_net(p1: object, p2: object, g1: object, g2: object) -> tuple[np.ndarray, np.ndarray]:
    """(net_pending_p1, net_pending_p2)。入力が有限でない/負なら両側とも NaN。配列・スカラーどちらも可。"""
    a, b, c, d = (np.asarray(v, dtype=np.float64) for v in (p1, p2, g1, g2))
    valid = _valid_count(a, b, c, d)
    x, y = a - c, b - d
    n1 = np.maximum(np.maximum(x, x - y), 0.0) + 0.0   # +0.0: -0.0 を 0.0 に揃える
    n2 = np.maximum(np.maximum(y, y - x), 0.0) + 0.0
    return np.where(valid, n1, np.nan), np.where(valid, n2, np.nan)


def settle_net_sequential(p1: int, p2: int, g1: int, g2: int, first: int = 0) -> tuple[int, int]:
    """ゲーム規則を文字どおり順に適用した参照実装 (試験専用)。first=0 なら P1 の未反映量から先に決済する。"""
    pend, gen = [p1, p2], [g1, g2]
    for side in ((0, 1) if first == 0 else (1, 0)):
        used = min(pend[side], gen[side])
        pend[side] -= used
        pend[1 - side] += gen[side] - used
    return pend[0], pend[1]


def margin_columns(q: object, i: object) -> dict[str, np.ndarray]:
    """余地 M=Q-I とその派生。b3_timeaxis._fill_margins と同じ式の配列版 (試験で一致を固定)。"""
    qa, ia = np.asarray(q, dtype=np.float64), np.asarray(i, dtype=np.float64)
    ok = np.isfinite(qa) & np.isfinite(ia)
    m = np.where(ok, qa - ia, np.nan)
    recv = np.maximum(-m, 0.0)
    return {"ta_m": m, "ta_mg": signed_nonsat(m), "ta_recv": recv, "ta_send": np.maximum(m, 0.0),
            "ta_unc72": np.where(ok, (recv >= UNCANCELLABLE_OJAMA).astype(np.float64), np.nan)}


def _mask_code(*masks: np.ndarray) -> np.ndarray:
    """欠測理由の最大値 (mk_df_* と同じ規約)。読めない (NaN) 理由は UNOBSERVABLE。"""
    code = np.maximum.reduce(masks)
    return np.where(np.isnan(code), float(Mask.UNOBSERVABLE), code)


def _net_mask(src: Mapping[str, np.ndarray], valid: np.ndarray) -> np.ndarray:
    """net の欠測理由。上流 4 つの mask の最大値。値が欠測なのに mask=0 になる穴は UNOBSERVABLE で塞ぐ。"""
    code = _mask_code(*(src[k] for k in (*_both("mk_sl"), *_both("mk_sl_inflight"))))
    return np.where((code == 0) & ~valid, float(Mask.UNOBSERVABLE), code)


def _ta_net(side: str, receiver_net: np.ndarray, net_mask: np.ndarray, src: Mapping[str, np.ndarray],
            out: dict[str, np.ndarray]) -> None:
    """攻撃側 side の時間軸 net 列。受け側 (反対側) の net を I にして ta_q_* から余地を作り直す。"""
    i_sl = src[f"ta_i_sl_{side}"]
    i_net = np.where(np.isfinite(i_sl) & np.isfinite(receiver_net), receiver_net, np.nan)
    out[f"ta_i_slnet_{side}"] = i_net
    q_all_finite = np.isfinite(i_net)
    for variant in NET_TA_VARIANTS:
        q = src[f"ta_q_{variant.split('_')[1]}_{side}"]
        q_all_finite = q_all_finite & np.isfinite(q)
        for stem, values in margin_columns(q, i_net).items():
            out[f"{stem}_{variant}_{side}"] = values
    # 時間軸側の理由 (mk_ta) を先に引き継ぎ、それが 0 で値が欠測のときだけ net 側の理由 (無ければ UNOBSERVABLE)。
    ta_code = _mask_code(src[f"mk_ta_{side}"])
    fallback = np.where(net_mask != 0, net_mask, float(Mask.UNOBSERVABLE))
    out[f"mk_ta_slnet_{side}"] = np.where(ta_code != 0, ta_code, np.where(q_all_finite, 0.0, fallback))


def derive_net_columns(src: Mapping[str, object]) -> dict[str, np.ndarray]:
    """既存列 (SOURCE_COLUMNS) から案A の全列を作る。{列名: 1 次元配列}。値は float64 (mask は int8)。

    src の値はスカラーでも 1 次元配列でもよい (スカラーは長さ 1 に揃える)。順序は列仕様 (NET_LEDGER_COLUMN_NAMES)。
    """
    missing = [k for k in SOURCE_COLUMNS if k not in src]
    if missing:
        raise KeyError(f"正味の予告の導出元の列が無い: {missing} (sl_inflight 列を含む行が必要)")
    f = {k: np.atleast_1d(np.asarray(src[k], dtype=np.float64)) for k in SOURCE_COLUMNS}
    n1, n2 = settle_net(f["sl_pending_p1"], f["sl_pending_p2"], f["sl_inflight_p1"], f["sl_inflight_p2"])
    net_mask = _net_mask(f, np.isfinite(n1))
    out: dict[str, np.ndarray] = {"net_pending_p1": n1, "net_pending_p2": n2,
                                  "net_pending_nf_p1": nonsat(n1), "net_pending_nf_p2": nonsat(n2),
                                  "df_sl_net": n1 - n2, "mk_sl_net_p1": net_mask, "mk_sl_net_p2": net_mask,
                                  "mk_df_sl_net": net_mask}
    _ta_net("p1", n2, net_mask, f, out)   # 攻撃側 p1 の受け側は p2
    _ta_net("p2", n1, net_mask, f, out)
    unknown = set(out) ^ set(NET_LEDGER_COLUMN_NAMES)
    if unknown:
        raise AssertionError(f"導出した列と列仕様が不一致: {sorted(unknown)}")
    return {name: (out[name].astype(np.int8) if name.startswith(MASK_COLUMN_PREFIX) else out[name])
            for name in NET_LEDGER_COLUMN_NAMES}


def derive_net_row(row: Mapping[str, object]) -> dict[str, float | int]:
    """1 行 (build_features の辞書) 用。実数列は float、mask は int (既存の行と同じ型)。"""
    cols = derive_net_columns(row)
    return {k: (int(v[0]) if k.startswith(MASK_COLUMN_PREFIX) else float(v[0])) for k, v in cols.items()}


def derive_frame(df: "pd.DataFrame") -> "pd.DataFrame":
    """学習行の DataFrame (M3 行) から案A の列だけを作る。dtype は to_frame (b3_rows) と同じ。

    入力に案A の列が既にあれば例外 (二重に足さない)。index は入力と同じ。
    """
    import pandas as pd
    present = sorted(set(NET_LEDGER_COLUMN_NAMES) & set(df.columns))
    if present:
        raise ValueError(f"案A の列が既にある: {present[:4]} (計{len(present)})")
    missing = [k for k in SOURCE_COLUMNS if k not in df.columns]
    if missing:
        raise KeyError(f"正味の予告の導出元の列が無い: {missing}")
    src = {k: df[k].to_numpy(dtype=np.float64, na_value=np.nan) for k in SOURCE_COLUMNS}
    cols = derive_net_columns(src)
    out: dict[str, object] = {}
    for spec in net_ledger_specs():
        series = pd.Series(cols[spec.name], index=df.index)
        dtype = FRAME_DTYPES[spec.dtype]
        out[spec.name] = (pd.array(series.where(series.notna(), pd.NA), dtype=dtype)
                          if spec.dtype == "i1" else series.astype(dtype))
    return pd.DataFrame(out, index=df.index)
