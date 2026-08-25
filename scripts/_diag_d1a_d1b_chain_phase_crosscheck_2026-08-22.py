"""D1a/D1b 5,904件の検出時刻が「連鎖中」かを state1/state2 で集計する計装スクリプト。

user指令: 走査器が検出した5,904件 (D1a display+both=3,879 / D1b display+both=2,025)
が実際には連鎖中 (CHAIN/GRAVITY_SETTLE) の「消えるので窒息でない」盤面を
誤検出しているのではないかを、タイムラインdump (state1/state2) を突合して
数値確定する。コードは変更しない (計装のみ)。

代替ローダー: 検収エージェント報告のとおり `load_timeline_dump()` は
`d["field"][i]` をループ内で毎回呼び都度フル解凍する性能バグがあるため、
ここでは同じ npz を素朴に全フィールド一括ロード (スライスなし) する
代替ローダーを使う (判定ロジック自体は本体 detect_d1a/detect_d1b を
scripts.scan_judgment_anomalies から import して共用)。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
SUSPECTS_TSV = BASE / "data/verify/judgment_scan_zenchi_recheck_2026-08-22_fast/suspects.tsv"
DUMP_DIR = BASE / "data/verify/zenchi_render_2026-08-21"

# 「連鎖中」の定義 (狭義): CHAIN (消去+重力) / GRAVITY_SETTLE (連鎖終了直後の
# 重力settle)。この2つはNON_STABLE_STATESのうち「盤面がまだ消える途中/
# 着地中」を意味する (src/board_state_machine.py)。
# dump の state1/state2 は `r.p1.state.name` (Enum.name、大文字) で
# 書き出されている (visualize_advantage_overlay.py 4791行目付近) ため
# 大文字で定義する。
CHAIN_NARROW = {"CHAIN", "GRAVITY_SETTLE"}
# 広義 (NON_STABLE_STATES全体): TSUMO_FALL/OJAMA_FALL/EFFECT も含む。
CHAIN_BROAD = {"CHAIN", "GRAVITY_SETTLE", "EFFECT", "OJAMA_FALL", "TSUMO_FALL"}

# 突合の許容誤差 (秒)。同一ソースの再計算のため理論上 0 だが、浮動小数点
# 丸め誤差を許容する。
MATCH_TOL_SEC = 0.02


def load_dump_fast(path: Path) -> dict[str, np.ndarray]:
    """savez_compressed の npz を全フィールド一括ロードする代替ローダー。

    `load_timeline_dump()` の `d["field"][i]` ループ (要素ごとに再解凍) を
    避け、各フィールドを1回だけ丸ごと配列として取り出す。
    """
    d = np.load(str(path), allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_all_dumps() -> dict[str, np.ndarray]:
    """8セグメント分の dump を結合した辞書を返す (t_sec でソート済み)。"""
    files = sorted(DUMP_DIR.glob("seg*.npz"))
    parts = [load_dump_fast(f) for f in files]
    keys = parts[0].keys()
    combined: dict[str, np.ndarray] = {}
    for k in keys:
        if k == "video_id":
            continue
        combined[k] = np.concatenate([p[k] for p in parts])
    order = np.argsort(combined["t_sec"], kind="stable")
    for k in combined:
        combined[k] = combined[k][order]
    return combined


def parse_side_from_evidence(evidence: str) -> str | None:
    m = re.match(r"^(1P|2P) は", evidence)
    return m.group(1) if m else None


def nearest_match(dump: dict[str, np.ndarray], t_sec: float) -> tuple[int, float]:
    """t_sec に最も近い dump 行の (index, |dt|) を返す。"""
    t_arr = dump["t_sec"]
    pos = int(np.searchsorted(t_arr, t_sec))
    candidates = [i for i in (pos - 1, pos, pos + 1) if 0 <= i < t_arr.shape[0]]
    best_i = min(candidates, key=lambda i: abs(float(t_arr[i]) - t_sec))
    return best_i, abs(float(t_arr[best_i]) - t_sec)


def main() -> None:
    print("[1/3] suspects.tsv 読み込み中...")
    lines = SUSPECTS_TSV.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    rows = [l.split("\t") for l in lines[1:]]
    print(f"  総行数={len(rows)}")

    print("[2/3] dump (8セグメント) 読み込み中...")
    dump = load_all_dumps()
    print(f"  結合後行数={dump['t_sec'].shape[0]}")

    print("[3/3] 突合中...")
    target_rows = []
    for r in rows:
        video_id, t_sec_s, detector, severity, stage, evidence, game_idx = r[:7]
        if detector not in ("D1a", "D1b"):
            continue
        if stage not in ("display", "both"):
            continue
        target_rows.append((float(t_sec_s), detector, stage, evidence, int(game_idx)))

    print(f"  対象行 (D1a/D1b, display+both) = {len(target_rows)} 件")

    unmatched = 0
    max_dt = 0.0
    results = []
    for t_sec, detector, stage, evidence, game_idx in target_rows:
        side = parse_side_from_evidence(evidence)
        idx, dt = nearest_match(dump, t_sec)
        max_dt = max(max_dt, dt)
        if dt > MATCH_TOL_SEC:
            unmatched += 1
            continue
        state1 = str(dump["state1"][idx])
        state2 = str(dump["state2"][idx])
        side_state = state1 if side == "1P" else (state2 if side == "2P" else None)
        other_state = state2 if side == "1P" else (state1 if side == "2P" else None)
        results.append({
            "t_sec": t_sec, "detector": detector, "stage": stage, "side": side,
            "side_state": side_state, "other_state": other_state,
            "state1": state1, "state2": state2, "game_idx": game_idx,
            "is_dead1": bool(dump["is_dead1"][idx]), "is_dead2": bool(dump["is_dead2"][idx]),
            "pending_p1": int(dump["pending_p1"][idx]), "pending_p2": int(dump["pending_p2"][idx]),
            "room1": int(dump["room1"][idx]), "room2": int(dump["room2"][idx]),
        })

    print(f"  突合失敗 (|dt|>{MATCH_TOL_SEC}s) = {unmatched} / {len(target_rows)}, 最大|dt|={max_dt:.4f}s")

    out_path = BASE / "data/verify/judgment_scan_zenchi_recheck_2026-08-22_fast/d1a_d1b_chain_crosscheck.tsv"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("t_sec\tdetector\tstage\tside\tside_state\tother_state\tstate1\tstate2\t"
                "is_dead1\tis_dead2\tpending_p1\tpending_p2\troom1\troom2\tgame_idx\n")
        for r in results:
            f.write(
                f"{r['t_sec']:.3f}\t{r['detector']}\t{r['stage']}\t{r['side']}\t"
                f"{r['side_state']}\t{r['other_state']}\t{r['state1']}\t{r['state2']}\t"
                f"{r['is_dead1']}\t{r['is_dead2']}\t{r['pending_p1']}\t{r['pending_p2']}\t"
                f"{r['room1']}\t{r['room2']}\t{r['game_idx']}\n"
            )
    print(f"  出力: {out_path} ({len(results)}件)")

    # ---- 集計 ----
    def _tally(detector: str) -> None:
        sub = [r for r in results if r["detector"] == detector]
        n = len(sub)
        if n == 0:
            print(f"{detector}: 0件")
            return
        narrow_side = sum(1 for r in sub if r["side_state"] in CHAIN_NARROW)
        broad_side = sum(1 for r in sub if r["side_state"] in CHAIN_BROAD)
        narrow_either = sum(
            1 for r in sub
            if r["side_state"] in CHAIN_NARROW or r["other_state"] in CHAIN_NARROW
        )
        broad_either = sum(
            1 for r in sub
            if r["side_state"] in CHAIN_BROAD or r["other_state"] in CHAIN_BROAD
        )
        both_stable = sum(
            1 for r in sub if r["side_state"] == "STABLE" and r["other_state"] == "STABLE"
        )
        print(f"\n=== {detector} (n={n}) ===")
        print(f"  当該side が CHAIN/GRAVITY_SETTLE (狭義連鎖中)        = {narrow_side} ({narrow_side/n:.1%})")
        print(f"  当該side が NON_STABLE全体 (広義)                    = {broad_side} ({broad_side/n:.1%})")
        print(f"  どちらかのsideが CHAIN/GRAVITY_SETTLE (狭義)          = {narrow_either} ({narrow_either/n:.1%})")
        print(f"  どちらかのsideが NON_STABLE全体 (広義)                = {broad_either} ({broad_either/n:.1%})")
        print(f"  両側とも STABLE (連鎖と無関係)                        = {both_stable} ({both_stable/n:.1%})")
        # side_state の内訳
        from collections import Counter
        c = Counter(r["side_state"] for r in sub)
        print(f"  side_state 内訳: {dict(c)}")

    _tally("D1a")
    _tally("D1b")

    # ---- エピソード化 (連続する suspect をまとめる) ----
    print("\n=== エピソード化 (同一 detector 内、連続時刻をグルーピング、gap<=0.5s) ===")
    for detector in ("D1a", "D1b"):
        sub = sorted([r for r in results if r["detector"] == detector], key=lambda r: r["t_sec"])
        episodes = []
        cur = []
        for r in sub:
            if cur and r["t_sec"] - cur[-1]["t_sec"] > 0.5:
                episodes.append(cur)
                cur = []
            cur.append(r)
        if cur:
            episodes.append(cur)
        long_eps = [(e[0]["t_sec"], e[-1]["t_sec"], e[-1]["t_sec"] - e[0]["t_sec"], len(e)) for e in episodes if e[-1]["t_sec"] - e[0]["t_sec"] > 5.0]
        print(f"{detector}: 総エピソード数={len(episodes)}, 5秒超={len(long_eps)}")
        for start, end, dur, n in sorted(long_eps, key=lambda x: -x[2]):
            # このエピソード内の state 分布
            ep = [e for e in episodes if abs(e[0]["t_sec"] - start) < 1e-6][0]
            states = [e["side_state"] for e in ep]
            from collections import Counter
            print(f"  t={start:.1f}~{end:.1f} (dur={dur:.1f}s, n={n}) side_state内訳={dict(Counter(states))}")


if __name__ == "__main__":
    main()
