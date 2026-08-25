"""Gate 3-0: 掛け算式ベースの安定 chain_id が実データで取れるかを検証する (2026-08-24)。

## 何を計装するか (本体コードは変更しない)

既存の実測ログ (`logs/_probe_formula_interlude_2026-08-24/probe_{w1,w2}_{off,on}.log`)
は `_probe_formula_interlude_2026-08-24.py` が `RecognitionPipeline.update` に
モンキーパッチを当てて `_active_chain_1p/2p` (ChainEvent) の変化を
`[ev] t=... ev{1,2} (trigger_sec, mechanism, chain_count, total_score)` として
既に記録済み。本スクリプトは **その既存テキストログを再パースするだけ**で、
動画の再処理は一切行わない (指示の「既存資産を先に探す」を優先)。

## 測定方法

1. OFF ログ・ON ログそれぞれについて、side ごとに [ev] の時系列を復元する。
2. **ON ログを「物理連鎖の正解境界」の代理指標として使う** — Q-01 修正により
   ON は同一 trigger_sec を段の識別子として保持したまま chain_count が
   1→2→…→N と単調に進み、最後に mechanism='baseline' で真の合計得点に
   着地することを個別に screen-frame 付きで確認済み (W34, 本スクリプトでも再確認)。
   ON の 1 セッション (None…None で区切られた区間) を「物理連鎖 1 本」とみなし、
   その [t_start, t_end] 区間を用いて OFF 側のイベントを対応付ける。
3. OFF 側で、ON の物理連鎖区間に時間的に重なる trigger_sec が何種類出たかを数える
   → 測定1・測定2 (断片化の実数)。
4. OFF 側で、ON がまだ chain_count 進行中と判定している時刻に OFF がすでに
   `None` (セッション終端) を報告していた回数を数える → 測定3 (早すぎる打ち切り)。

## 制約遵守

- 本体コード (`src/*.py`) は一切変更しない。
- 既存ログ・成果物は読むだけで上書きしない。
- 出力先は新規ディレクトリ `data/verify/gate3_chainid_2026-08-24/` のみ。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs/_probe_formula_interlude_2026-08-24"
OUT_DIR = PROJECT_ROOT / "data/verify/gate3_chainid_2026-08-24"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EV_RE = re.compile(
    r"^\[ev\] t=(?P<t>[\d.]+) (?P<key>ev1|ev2) "
    r"(?:None|\((?P<trig>[\d.]+), '(?P<mech>[a-z_]+)', (?P<cc>\d+), (?P<score>\d+)\))$"
)


@dataclass
class Event:
    t: float
    key: str  # ev1 (1P) / ev2 (2P)
    trig: float | None
    mech: str | None
    cc: int | None
    score: int | None


def parse_log(path: Path) -> list[Event]:
    """既存の probe ログから [ev] 行を復元する (計装ログの再パースのみ)。"""
    events: list[Event] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = EV_RE.match(line)
        if not m:
            continue
        if m.group("trig") is None:
            events.append(Event(float(m.group("t")), m.group("key"), None, None, None, None))
        else:
            events.append(Event(
                float(m.group("t")), m.group("key"),
                float(m.group("trig")), m.group("mech"),
                int(m.group("cc")), int(m.group("score")),
            ))
    return events


@dataclass
class Session:
    """物理連鎖1本の候補区間 (側ごと)。

    None...None の区切りではなく、mechanism='baseline' (score>0) の着地確定
    イベントを境界に使う。理由: baseline は掛け算式の段累積が終わり通常スコアの
    差分確認まで済んだ「真の着地」であり、ChainEvent オブジェクトが着地直前に
    一瞬 None を経由する (interlude 反映の遅延) ケースがあり、素朴な
    None 区切りだと同一物理連鎖が2分割されてしまうため (実測で1件確認)。
    """
    key: str
    t_start: float
    t_end: float
    events: list[Event] = field(default_factory=list)

    @property
    def trig_values(self) -> list[float]:
        seen: list[float] = []
        for e in self.events:
            if e.trig is not None and e.trig not in seen:
                seen.append(e.trig)
        return seen

    @property
    def final(self) -> Event | None:
        return self.events[-1] if self.events else None


POST_LANDING_MERGE_GAP_SEC = 1.5
"""None を経由する着地確認の合流しきい値。

実測 (w1, ev1, t=792.633 None -> t=793.3 baseline cc=11 score=52160) で、
真の着地確認イベント (baseline) が直前の formula 進行セッションから
**0.667 秒だけ None を挟んで**独立した1件セッションとして現れるケースを確認した。
段間の幕間実測 (W34: 0.433〜0.634秒) の上限 0.634 秒よりわずかに大きいが、
同じ現象 (着地確認までの読み取りラグ) の範囲内とみなせる。安全側に
1.5 秒 (幕間実測上限の約2.4倍) を採用し、無関係な次の物理連鎖 (実測最小ギャップ
は本測定データで3.1秒以上) を誤って合流させない。
"""


def build_sessions(events: list[Event]) -> list[Session]:
    """side ごとに None...None の連続区間 (raw session) へ分割したうえで、
    「formula 進行中のセッション」→(短い None)→「baseline 単独着地セッション」
    という並びを1本の物理連鎖として合流する。

    合流条件 (機械的、閾値は上記 POST_LANDING_MERGE_GAP_SEC のみ):
      - 次の raw session が baseline (score>0) の1件のみで構成される
      - 直前の raw session の最終 chain_count 以上である
      - None を挟んだギャップが POST_LANDING_MERGE_GAP_SEC 以下
    それ以外の None 区切りはそのまま別セッションとして扱う
    (=無関係の次の物理連鎖、または未確定の尻切れ区間)。
    """
    raw: list[Session] = []
    cur: dict[str, Session | None] = {"ev1": None, "ev2": None}
    for e in events:
        s = cur[e.key]
        if e.trig is None:
            if s is not None:
                s.t_end = e.t
                raw.append(s)
                cur[e.key] = None
            continue
        if s is None:
            cur[e.key] = Session(e.key, e.t, e.t, [e])
        else:
            s.events.append(e)
            s.t_end = e.t
    for s in cur.values():
        if s is not None:
            raw.append(s)

    by_key: dict[str, list[Session]] = {}
    for s in raw:
        by_key.setdefault(s.key, []).append(s)

    merged: list[Session] = []
    for key, ss in by_key.items():
        ss.sort(key=lambda s: s.t_start)
        i = 0
        while i < len(ss):
            base = ss[i]
            j = i + 1
            while (
                j < len(ss)
                and len(ss[j].events) == 1
                and ss[j].events[0].mech == "baseline"
                and (ss[j].events[0].score or 0) > 0
                and base.events
                and base.events[-1].cc is not None
                and ss[j].events[0].cc >= base.events[-1].cc
                and (ss[j].t_start - base.t_end) <= POST_LANDING_MERGE_GAP_SEC
            ):
                base.events.extend(ss[j].events)
                base.t_end = ss[j].t_end
                j += 1
            merged.append(base)
            i = j

    # baseline (score>0) で終わっていないセッション (=未確定の尻切れ) は
    # 「真の物理連鎖境界」の代理指標として使えないため除外する。
    return [
        s for s in merged
        if s.final is not None and s.final.mech == "baseline" and (s.final.score or 0) > 0
    ]


def monotonic_chain_count(session: Session) -> bool:
    """段数 (chain_count) が単調非減少かどうか (formula/landing/baselineの混在含む)。"""
    ccs = [e.cc for e in session.events if e.cc is not None]
    return all(a <= b for a, b in zip(ccs, ccs[1:]))


def analyze_window(tag: str, off_path: Path, on_path: Path) -> dict:
    off_events = parse_log(off_path)
    on_events = parse_log(on_path)
    off_sessions = build_sessions(off_events)
    on_sessions = build_sessions(on_events)

    # build_sessions は baseline (score>0) 着地で必ず終端するので、
    # on_sessions は全件が「真の着地を確認できた物理連鎖候補」。
    on_confirmed = on_sessions

    # 別々の連鎖が1つの trigger_sec に癒着していないか (測定1後半):
    # 同じ side で連続する2つの物理連鎖候補が同じ trigger_sec を共有していたら癒着疑い。
    fusion_suspects = 0
    by_key_sessions: dict[str, list[Session]] = {}
    for s in on_confirmed:
        by_key_sessions.setdefault(s.key, []).append(s)
    for key, ss in by_key_sessions.items():
        ss.sort(key=lambda s: s.t_start)
        for a, b in zip(ss, ss[1:]):
            a_trigs = set(a.trig_values)
            b_trigs = set(b.trig_values)
            if a_trigs & b_trigs:
                fusion_suspects += 1

    per_chain: list[dict] = []
    for s in on_confirmed:
        key = s.key
        # ON の chain_id (trigger_sec) が単一かどうか (測定1: 別連鎖の癒着チェックも兼ねる)。
        on_trig_count = len(s.trig_values)
        on_cc_monotonic = monotonic_chain_count(s)
        true_cc = s.final.cc
        true_score = s.final.score

        # OFF 側で、この ON 物理連鎖区間 [t_start, t_end] に時間的に重なる
        # trigger_sec の集合を数える (測定1・測定2)。
        overlapping_off_trigs: list[float] = []
        off_events_in_window: list[Event] = []
        for e in off_events:
            if e.key != key or e.trig is None:
                continue
            if s.t_start - 0.05 <= e.t <= s.t_end + 0.05:
                off_events_in_window.append(e)
                if e.trig not in overlapping_off_trigs:
                    overlapping_off_trigs.append(e.trig)

        # 測定3: OFF がこの区間内で「終端 (None)」を報告した回数
        # (= ON がまだ進行中と判定している間に OFF がセッションを打ち切った回数)。
        # 単発 (true_cc=1) は「1段だけ表示して None に戻る」のが正常動作なので
        # 早すぎる打ち切りの定義に使えない。true_cc>=2 かつ、その時点で
        # OFF 自身がまだ真の段数に到達していない (=本当にまだ連鎖の途中) 場合のみ
        # 「早すぎる打ち切り」としてカウントする。
        off_early_none_count = 0
        off_running_max_cc = 0
        for e in off_events:
            if e.key != key:
                continue
            if e.cc is not None and s.t_start - 0.05 <= e.t <= s.t_end + 0.05:
                off_running_max_cc = max(off_running_max_cc, e.cc)
            if (
                true_cc >= 2
                and s.t_start < e.t < s.t_end
                and e.trig is None
                and off_running_max_cc < true_cc
            ):
                off_early_none_count += 1

        per_chain.append({
            "side": key,
            "t_start": s.t_start,
            "t_end": s.t_end,
            "true_chain_count": true_cc,
            "true_score": true_score,
            "on_trigger_count": on_trig_count,
            "on_cc_monotonic": on_cc_monotonic,
            "off_trigger_count": len(overlapping_off_trigs),
            "off_trigger_values": overlapping_off_trigs,
            "off_early_none_count": off_early_none_count,
            "off_events_in_window_n": len(off_events_in_window),
        })

    return {
        "tag": tag,
        "off_log": str(off_path),
        "on_log": str(on_path),
        "off_session_count_total": len(off_sessions),
        "on_session_count_total": len(on_sessions),
        "on_confirmed_physical_chain_count": len(on_confirmed),
        "on_fusion_suspects": fusion_suspects,
        "per_chain": per_chain,
    }


def main() -> None:
    windows = [
        ("w1", LOG_DIR / "probe_w1_off.log", LOG_DIR / "probe_w1_on.log"),
        ("w2", LOG_DIR / "probe_w2_off.log", LOG_DIR / "probe_w2_on.log"),
    ]
    all_results = []
    for tag, off_path, on_path in windows:
        if not off_path.exists() or not on_path.exists():
            print(f"[skip] {tag}: ログ不在")
            continue
        result = analyze_window(tag, off_path, on_path)
        all_results.append(result)
        out_path = OUT_DIR / f"result_{tag}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[保存] {out_path}")

    # 集計
    total_chains = 0
    total_off_trig_sum = 0
    off_frag_ge2 = 0
    on_internal_frag = 0  # ON の1物理連鎖内に複数trigger_secが同居 (内部断片化)
    total_on_fusion = 0   # ON で隣接する2連鎖が同じtrigger_secを共有 (癒着)
    cc_mismatch = 0
    total_early_none = 0
    off_only_zero = 0  # OFF側がこの連鎖区間内に一切triggerを出していない(見逃し)件数
    for r in all_results:
        total_on_fusion += r["on_fusion_suspects"]
        for c in r["per_chain"]:
            total_chains += 1
            total_off_trig_sum += c["off_trigger_count"]
            if c["off_trigger_count"] >= 2:
                off_frag_ge2 += 1
            if c["off_trigger_count"] == 0:
                off_only_zero += 1
            if c["on_trigger_count"] >= 2:
                on_internal_frag += 1
            if not c["on_cc_monotonic"]:
                cc_mismatch += 1
            total_early_none += c["off_early_none_count"]

    # 段数>=2 (真に複数段の掛け算式が表示された連鎖) だけに絞った層別
    # (段数1は「そもそも断片化しようがない」ため主張の母数から分離する)。
    all_chains = [c for r in all_results for c in r["per_chain"]]
    multi = [c for c in all_chains if c["true_chain_count"] >= 2]
    off_worse_than_on = sum(
        1 for c in multi if c["off_trigger_count"] > c["on_trigger_count"])
    off_equal_on = sum(
        1 for c in multi if c["off_trigger_count"] == c["on_trigger_count"])
    off_better_than_on = sum(
        1 for c in multi if c["off_trigger_count"] < c["on_trigger_count"])
    off_max_trig_multi = max((c["off_trigger_count"] for c in multi), default=0)
    on_max_trig_multi = max((c["on_trigger_count"] for c in multi), default=0)

    summary = {
        "windows_analyzed": [r["tag"] for r in all_results],
        "true_physical_chains_n_confirmed_by_on_baseline_landing": total_chains,
        "measurement1_on_internal_fragmentation": f"{on_internal_frag}/{total_chains}",
        "measurement1_on_cross_chain_fusion_suspect": f"{total_on_fusion}/{total_chains}",
        "measurement1_on_cc_monotonic_violation": f"{cc_mismatch}/{total_chains}",
        "measurement1_2_off_fragment_ge2_of_chains": f"{off_frag_ge2}/{total_chains}",
        "measurement1_2_off_total_trigger_ids_over_all_chains": total_off_trig_sum,
        "measurement1_2_off_zero_trigger_overlap_count": off_only_zero,
        "measurement2_multi_step_chains_n": len(multi),
        "measurement2_off_worse_than_on_count": f"{off_worse_than_on}/{len(multi)}",
        "measurement2_off_equal_on_count": f"{off_equal_on}/{len(multi)}",
        "measurement2_off_better_than_on_count": f"{off_better_than_on}/{len(multi)}",
        "measurement2_off_max_trigger_ids_multi_step": off_max_trig_multi,
        "measurement2_on_max_trigger_ids_multi_step": on_max_trig_multi,
        "measurement3_off_early_none_during_on_active_window_total": total_early_none,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
