"""D1a/D1b 5,904件を「両方連鎖中/片方だけ連鎖中/両方STABLE」で再集計する。

coordinator指示 (2026-08-22 14:30): 前回集計は「片側でも連鎖中」で一括り
にしていたが、userの判断基準は「両方連鎖中=保留、片方だけ連鎖中=連鎖して
いない側は確定しているので評価すべき」。片方だけ連鎖中のケースは、
「フラグが立った側 (窒息/致死確定と判定された側) 自身が連鎖中か、それとも
非連鎖 (確定) 側か」でさらに分けないと、保留が正しいかどうか判定できない。

GRAVITY_SETTLE の扱い: docs/KNOWN_WEAKNESSES.md W30 (2026-08-21) の実測により
GRAVITY_SETTLE は「連鎖の消去→落下待ちの中間状態」であり同一連鎖の途中
(chain_event が一時的に None になるだけで連鎖自体は続いている) と確定して
いるため、本スクリプトでは GRAVITY_SETTLE を CHAIN と同じ「連鎖中」バケット
に含める。
"""
from __future__ import annotations

import importlib.util
import csv
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parent / "_diag_d1a_d1b_chain_phase_crosscheck_2026-08-22.py"
_spec = importlib.util.spec_from_file_location("_diag_chain_crosscheck_base", _MOD_PATH)
_base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_base)

BASE = Path(__file__).resolve().parent.parent
CROSSCHECK_TSV = BASE / "data/verify/judgment_scan_zenchi_recheck_2026-08-22_fast/d1a_d1b_chain_crosscheck.tsv"

CHAIN_NARROW = _base.CHAIN_NARROW  # {"CHAIN", "GRAVITY_SETTLE"}


def is_chain(state: str) -> bool:
    return state in CHAIN_NARROW


def adv_bucket(mag: float) -> str:
    if mag < 30.0:
        return "<30"
    if mag < 80.0:
        return "30-80"
    return ">=80"


def main() -> None:
    rows = list(csv.DictReader(CROSSCHECK_TSV.open(encoding="utf-8"), delimiter="\t"))
    print(f"総行数(D1a+D1b display+both)={len(rows)}")

    both_chain, one_chain, both_stable, other_neither = [], [], [], []
    for r in rows:
        s1, s2 = r["state1"], r["state2"]
        c1, c2 = is_chain(s1), is_chain(s2)
        if c1 and c2:
            both_chain.append(r)
        elif c1 != c2:
            one_chain.append(r)
        elif s1 == "STABLE" and s2 == "STABLE":
            both_stable.append(r)
        else:
            other_neither.append(r)  # 連鎖ではない他状態同士 (OJAMA_FALL同士等)

    n = len(rows)
    print("\n=== 大分類 (state1/state2 の組み合わせ) ===")
    print(f"  両方が連鎖中 (CHAIN/GRAVITY_SETTLE)        = {len(both_chain)} ({len(both_chain)/n:.1%})  ← 保留が妥当")
    print(f"  片方だけ連鎖中                              = {len(one_chain)} ({len(one_chain)/n:.1%})  ← 要個別判定")
    print(f"  両方 STABLE                                 = {len(both_stable)} ({len(both_stable)/n:.1%})")
    print(f"  その他 (連鎖ではない非STABLE同士等)         = {len(other_neither)} ({len(other_neither)/n:.1%})")

    # ---- 片方だけ連鎖中の内訳: フラグが立った側 (side_state) は連鎖側か確定側か ----
    print("\n=== 片方だけ連鎖中 の内訳 (フラグが立った側=side_state で判定) ===")
    flagged_is_chain = [r for r in one_chain if is_chain(r["side_state"])]
    flagged_is_confirmed = [r for r in one_chain if not is_chain(r["side_state"])]
    m = len(one_chain)
    print(f"  フラグ側 自身が連鎖中 (その側の窒息/pending 読みは連鎖前の凍結値=不確定)"
          f" = {len(flagged_is_chain)} ({len(flagged_is_chain)/m:.1%})")
    print(f"  フラグ側 は非連鎖 (確定/STABLE等、もう片方が連鎖中なだけ)"
          f" = {len(flagged_is_confirmed)} ({len(flagged_is_confirmed)/m:.1%})  ← 評価すべき場面・異常候補")

    # flagged_is_confirmed の side_state 内訳 (STABLE以外にTSUMO_FALL/OJAMA_FALL等が
    # 混じりうる。これらも b1/b2 は直近STABLE凍結値のため実質「確定」相当だが
    # 明示的に内訳を出す)。
    from collections import Counter
    print(f"  (flagged_is_confirmed の side_state 内訳: "
          f"{dict(Counter(r['side_state'] for r in flagged_is_confirmed))})")

    # ---- flagged_is_confirmed (異常候補) の詳細分析 ----
    print("\n=== 異常候補 (片方だけ連鎖中・フラグ側は確定) の詳細 ===")
    # crosscheck tsv には adv_ema/p1 が無いため、元 suspects.tsv の evidence から
    # 数値を正規表現で抽出する (計装のみ・コード変更なしの制約下での再利用)。
    import re
    suspects_path = BASE / "data/verify/judgment_scan_zenchi_recheck_2026-08-22_fast/suspects.tsv"
    lines = suspects_path.read_text(encoding="utf-8").splitlines()[1:]
    # (t_sec, detector) -> evidence 文字列 (複数一致しうるが t_sec 3桁精度+detector で
    # ほぼ一意。念のため最初の一致を使う)
    ev_lookup: dict[tuple[str, str], str] = {}
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        key = (parts[1], parts[2])
        if key not in ev_lookup:
            ev_lookup[key] = parts[5]

    pat = re.compile(
        r"adv_raw=([+-]?[\d.]+)/勝率\S+\(raw\)=([\d.]+)%,\s*"
        r"adv_disp=([+-]?[\d.]+)/勝率\S+\(表示\)=([\d.]+)%"
    )
    strong_reversal = 0  # adv・p1 とも明確に致死/致命側を favor
    partial_reversal = 0  # どちらか一方のみ favor 判定に該当 (境界的)
    parse_fail = 0
    mags = []
    episodes_t = []
    for r in flagged_is_confirmed:
        t_key = f"{float(r['t_sec']):.3f}"
        ev = ev_lookup.get((t_key, r["detector"]))
        if ev is None:
            parse_fail += 1
            continue
        m2 = pat.search(ev)
        if not m2:
            parse_fail += 1
            continue
        adv_raw, winp_raw, adv_disp, winp_disp = (float(x) for x in m2.groups())
        mags.append(abs(adv_disp))
        episodes_t.append(float(r["t_sec"]))
        side = r["side"]
        # winp_disp/winp_raw は「フラグが立った側 (死/致死確定側) の表示上の
        # 勝率」(detect_d1a/d1b の evidence 組み立てに合わせた値)。
        # 50%超は表示が明確にその死側を favor、50%ちょうど付近は境界的。
        disp_wrong = winp_disp > 50.0
        raw_wrong = winp_raw > 50.0
        if disp_wrong and raw_wrong:
            strong_reversal += 1
        else:
            partial_reversal += 1

    print(f"  evidence 数値抽出成功 = {len(mags)} / {len(flagged_is_confirmed)} (失敗={parse_fail})")
    print(f"  向きが明確に逆 (adv_disp/adv_raw とも死側favor)      = {strong_reversal}")
    print(f"  向きが部分的 (raw/dispどちらかのみ死側favor、境界的) = {partial_reversal}")
    if mags:
        b = Counter(adv_bucket(x) for x in mags)
        print(f"  |adv_disp| 分布: <30={b.get('<30',0)}, 30-80={b.get('30-80',0)}, "
              f">=80={b.get('>=80',0)} (n={len(mags)})")

    # ---- エピソード化 ----
    episodes_t.sort()
    episodes = []
    cur = []
    for t in episodes_t:
        if cur and t - cur[-1] > 0.5:
            episodes.append(cur)
            cur = []
        cur.append(t)
    if cur:
        episodes.append(cur)
    long_eps = [(e[0], e[-1], e[-1] - e[0], len(e)) for e in episodes if e[-1] - e[0] > 0.0]
    print(f"\n  エピソード数(連続時刻グルーピング, gap<=0.5s) = {len(episodes)}")
    print(f"  うち 1秒超 = {sum(1 for e in long_eps if e[2] > 1.0)}, "
          f"5秒超 = {sum(1 for e in long_eps if e[2] > 5.0)}")
    for s, e, d, n2 in sorted(long_eps, key=lambda x: -x[2])[:15]:
        print(f"    t={s:.1f}~{e:.1f} dur={d:.1f}s n={n2}")


if __name__ == "__main__":
    main()
