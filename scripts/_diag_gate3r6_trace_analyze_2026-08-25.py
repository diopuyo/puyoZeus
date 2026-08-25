"""Gate 3R-6: フレームトレース (trace.jsonl) から対象1の出現量を集計する。

対象1 = 「画面で連鎖している (掛け算式が実読できている) のに state が STABLE」。

「掛け算式セッション」の定義 (定数は物理量由来、シーン逆算ではない):
  - fv{side}=True (FormulaValueRead.valid) のフレームをシードにする。
  - 段の表示は 27〜29 フレーム、段間の幕間は 12〜19 フレーム
    (production_config --enable-formula-step-interlude の実測記録)。
    30fps では最大 (29+19)/30 = 1.6 秒なので、valid が 1.6 秒を超えて
    出なくなったらセッション終了とする。
  - セッション中に state が STABLE だった実時間と、セッション開始から
    state=CHAIN に初到達するまでの遅延を測る。

出力: 標準出力 (json) + --out-json 指定先。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SESSION_GAP_SEC = 1.6  # (29+19)フレーム/30fps、物理量由来


def load_trace(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def sessions_for_side(rows: list[dict], side: int) -> list[dict]:
    fv_key, st_key = f"fv{side}", f"s{side}"
    out: list[dict] = []
    cur: dict | None = None
    last_valid_t = None
    for r in rows:
        t = r["t"]
        if cur is not None and last_valid_t is not None and \
                t - last_valid_t > SESSION_GAP_SEC:
            out.append(cur)
            cur = None
            last_valid_t = None
        if r.get(fv_key):
            if cur is None:
                cur = dict(side=side, t_start=t, t_end=t,
                           stable_sec=0.0, n_frames=0,
                           chain_reached_t=None, states=set(),
                           prev_t=t)
            last_valid_t = t
        if cur is not None:
            dt = t - cur["prev_t"]
            cur["prev_t"] = t
            cur["t_end"] = t
            cur["n_frames"] += 1
            st = str(r[st_key]).upper()
            cur["states"].add(st)
            if st == "STABLE":
                cur["stable_sec"] += dt
            if st == "CHAIN" and cur["chain_reached_t"] is None:
                cur["chain_reached_t"] = t
    if cur is not None:
        out.append(cur)
    for s in out:
        s["states"] = ",".join(sorted(s["states"]))
        s["duration"] = round(s["t_end"] - s["t_start"], 3)
        s["stable_sec"] = round(s["stable_sec"], 3)
        s["lag_to_chain"] = (
            round(s["chain_reached_t"] - s["t_start"], 3)
            if s["chain_reached_t"] is not None else None)
        s.pop("prev_t", None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--min-session-sec", type=float, default=0.5,
                    help="ノイズ除去: 単発誤読 (1〜2フレーム) を捨てる下限。"
                         "掛け算式の1段は最低27フレーム=0.9秒表示されるので、"
                         "0.5秒未満のセッションは実連鎖ではあり得ない (物理量由来)")
    a = ap.parse_args()
    rows = load_trace(a.trace)
    result: dict = {"trace": str(a.trace), "n_rows": len(rows)}
    all_sessions = []
    for side in (1, 2):
        ss = [s for s in sessions_for_side(rows, side)
              if s["duration"] >= a.min_session_sec]
        all_sessions.extend(ss)
        lag = [s["lag_to_chain"] for s in ss if s["lag_to_chain"] is not None]
        never = [s for s in ss if s["lag_to_chain"] is None]
        result[f"side{side}"] = dict(
            sessions=len(ss),
            stable_sec_total=round(sum(s["stable_sec"] for s in ss), 2),
            stable_sec_ge_1s_sessions=len(
                [s for s in ss if s["stable_sec"] >= 1.0]),
            lag_max=(max(lag) if lag else None),
            lag_ge_1s=len([v for v in lag if v >= 1.0]),
            never_reached_chain=len(never),
            never_reached_chain_sec=round(
                sum(s["stable_sec"] for s in never), 2),
        )
    result["sessions_detail"] = all_sessions
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.out_json:
        a.out_json.parent.mkdir(parents=True, exist_ok=True)
        a.out_json.write_text(text, encoding="utf-8")
        print(f"[保存] {a.out_json}")
    # detail は保存のみ、標準出力は要約
    result.pop("sessions_detail")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
