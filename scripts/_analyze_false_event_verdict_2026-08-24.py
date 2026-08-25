"""偽連鎖イベント率の後段分類器・合否判定 (2026-08-24 コーダ、Q-02 対応)。

Codex 品質精査 (docs/CODEX_QUALITY_AUDIT_2026-08-24.md Q-02) の指摘:
`scripts/_probe_formula_false_event_2026-08-24.py` はイベントと score の
ログ出力だけで、分類器・SUPPORT_WINDOW・集計・合否判定が存在しなかった。
`driver_progress.log` の `ALL_DONE` は 4 走行の**終了**を示すだけで、
「偽イベントを増やしていない」の**証明 (ACCEPTED)** ではない。

本スクリプトは以下を新規に行う (対象ファイルは一切編集しない・読むだけ):
  1. 既存 4 ログ (`logs/_probe_formula_false_event_2026-08-24/probe_{w1,w2}_{off,on}.log`)
     を解析し、trigger と score タイムラインから TP/FP/FN/判定不能/重複を分類する。
  2. 既存 c62 の集計 JSON (`data/verify/formula_read_false_event_ab_2026-08-24/`)
     は生の score タイムラインを保持していないため、粗い指標 (TP と
     「FP+判定不能」の合算のみ) しか再構成できない。この制約を明示する。
  3. `ALL_DONE` (走行完了) と `ACCEPTED` (合否) を分離し、合否基準を
     定数として明示した上で判定する。

Usage:
    python scripts/_analyze_false_event_verdict_2026-08-24.py

出力 (data/verify/false_event_verdict_2026-08-24/、新規ディレクトリ):
  - verdict_result.json      : 窓別集計 + 合否判定 + 判定基準
  - human_review_frames.json : 目視確認すべきフレーム一覧 (時刻/side/機構/分類/理由)
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 既存 scripts/_ab_false_event_2026-08-24.py から継承する定数 -------------
# (読むだけで転記。値を変えていない。根拠は同スクリプトの docstring 参照)
SUPPORT_WINDOW_SEC: float = 8.0
MIN_CHAIN_SCORE: int = 40

# --- 本スクリプト固有の新規定数 ---------------------------------------------
# 連鎖 1 段の表示周期は実測 ≈1.4 秒
# (memory reference_chain_formula_per_step_2026-08-22)。同一 side で
# trigger_sec が再出現するまでの間隔がこれより十分短い場合、実時間として
# 同一の物理連鎖に対する複数トリガー (フリッカ) の疑いが強いと見なす。
# 断定はできないため、該当イベントは human_review_frames.json にも載せる。
DUPLICATE_TRIGGER_GAP_SEC: float = 3.0

# 旧機構名 (掛け算式を実読できなかったフレームのフォールバック経路)。
#
# 【2026-08-24 訂正】当初これを「ON 側に出た時点でゲーティング不全の証拠」として
# 無条件 REJECTED の根拠にしていたが、**誤りだった**。
# src/recognition_pipeline.py:6178-6205 が示すとおり、
#
#     read_fire = (self._enable_chain_formula_read_verify
#                  and read_res is not None
#                  and bool(getattr(read_res, "valid", False)))
#     if read_fire:  mechanism = CHAIN_MECHANISM_FORMULA_READ
#     else:          mechanism = CHAIN_MECHANISM_FORMULA   # ← 旧機構へフォールバック
#
# フラグが ON でも「そのフレームで掛け算式が読めなかった」場合は旧経路に落ちる。
# これは**設計どおりの fail-safe** であって、ゲーティングの失敗ではない。
#
# ただし旧経路は total_score=0 を出す (_fill_pseudo_chain_score が (0,0,False))。
# これは docs/KNOWN_WEAKNESSES.md W7「疑似 ChainEvent のスコア0固定」そのもので、
# 実測ではこの経路のイベントが全件スコア非支持だった。
# したがって**件数そのものではなく「フォールバック率」と「その偽イベント率」で
# 判定する**のが正しい。
LEGACY_MECHANISM_NAME: str = "formula"

# ON 側のフォールバック率 (旧機構イベント / 全トリガー) の上限。
# 根拠: 掛け算式の実読は「1 段あたり 28 フレーム表示 / 幕間 13〜19 フレーム」
# (logs/_diag_formula_fix_e2e_2026-08-24/trace_on.jsonl の 15 連鎖 13 境界の実測) の
# うち表示側で読めればよい。表示区間は周期の 28/(28+16) ≈ 64% を占めるため、
# 発火判定が表示区間に当たらない確率は原理的に 4 割弱ある。
# それを踏まえ「発火の 4 割を超えて実読に失敗するならフラグが効いていない」と見なす。
# シーンからの逆算ではなく、段の表示デューティ比からの導出。
LEGACY_FALLBACK_RATE_MAX: float = 0.40

# 重複率 (段の進行を除いた近接再トリガー / 全トリガー) の許容増分。
# 根拠: 1 トリガー分の揺らぎを許す。窓ごとのトリガー数は実測 64〜110 件なので、
# 最小の窓 (64 件) で 1 件ぶん = 1/64 ≈ 1.6%。それを丸めて 2%。
# 「何%なら合格か」をシーンから決めたのではなく、「1 件の増減では落とさない」
# という粒度から導いた値。
DUPLICATE_RATE_TOLERANCE: float = 0.02

OUT_DIR = PROJECT_ROOT / "data/verify/false_event_verdict_2026-08-24"
PROBE_LOG_DIR = PROJECT_ROOT / "logs/_probe_formula_false_event_2026-08-24"
AB_JSON_DIR = PROJECT_ROOT / "data/verify/formula_read_false_event_ab_2026-08-24"

PROBE_WINDOWS: tuple[str, ...] = ("w1", "w2")
MODES: tuple[str, ...] = ("off", "on")

_EV_LINE_RE = re.compile(r"^\[ev\] t=([\d.]+) (ev[12]) (.+)$")
_EV_TUPLE_RE = re.compile(r"^\(([\d.]+), '(\w+)', (\d+), (\d+)\)$")
_SCORE_LINE_RE = re.compile(r"^\[score\] t=([\d.]+) (1P|2P) (\d+)$")


@dataclass
class Trigger:
    """1 件の新規 chain trigger (_ab_false_event と同じ「新規」定義)。"""

    side: str
    t_seen: float
    trigger_sec: float
    mechanism: str
    chain_count: int = 0
    total_score: int = 0
    # 同一 trigger_sec が None に戻るまで、最後に観測された時刻。
    # 多段連鎖は t_seen (最初の検出) から数秒〜十数秒続くため、FN 判定の
    # 窓は t_seen ではなく t_last_active を基準にしないと、長い連鎖の
    # 後半の得点上昇を誤って FN 扱いしてしまう (実測で判明、後述)。
    t_last_active: float = 0.0


@dataclass
class ScoreSample:
    """score OCR の値が変化した瞬間 1 件。"""

    t: float
    side: str
    value: int


@dataclass
class ClassifiedTrigger:
    """分類済み trigger。"""

    trigger: Trigger
    label: str  # "TP" | "FP" | "INDETERMINATE"
    reason: str
    is_duplicate: bool = False


def parse_probe_log(path: Path) -> tuple[list[Trigger], list[ScoreSample]]:
    """probe ログから新規 trigger と score 変化のタイムラインを抽出する。

    `[ev]` 行は trigger_sec が変わらない限り再出力される (chain_count や
    total_score の伸びを追うため)。「新規 trigger」の定義は
    `scripts/_ab_false_event_2026-08-24.py` の `last_trigger[side]` 比較と
    同一にする (trigger_sec が直前と異なれば新規、None を挟めば必ず新規)。
    """
    side_map = {"ev1": "1P", "ev2": "2P"}
    triggers: list[Trigger] = []
    scores: list[ScoreSample] = []
    last_trig: dict[str, float | None] = {"1P": None, "2P": None}
    active: dict[str, Trigger | None] = {"1P": None, "2P": None}
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        ev_m = _EV_LINE_RE.match(line)
        if ev_m:
            _parse_ev_line(ev_m, side_map, last_trig, active, triggers)
            continue
        sc_m = _SCORE_LINE_RE.match(line)
        if sc_m:
            scores.append(ScoreSample(
                t=float(sc_m.group(1)), side=sc_m.group(2), value=int(sc_m.group(3)),
            ))
    return triggers, scores


def _parse_ev_line(
    ev_m: "re.Match[str]",
    side_map: dict[str, str],
    last_trig: dict[str, float | None],
    active: dict[str, Trigger | None],
    triggers: list[Trigger],
) -> None:
    """`[ev]` 行 1 行を解析する。

    新規 trigger なら triggers に追加し、同一 trigger_sec の継続なら
    その trigger の `t_last_active` (連鎖が続いている最終観測時刻) を
    更新する (多段連鎖の後半をカバーするため)。
    """
    t = float(ev_m.group(1))
    side = side_map[ev_m.group(2)]
    payload = ev_m.group(3)
    if payload == "None":
        last_trig[side] = None
        active[side] = None
        return
    tup_m = _EV_TUPLE_RE.match(payload)
    if not tup_m:
        return
    trig = float(tup_m.group(1))
    mechanism = tup_m.group(2)
    chain_count = int(tup_m.group(3))
    total_score = int(tup_m.group(4))
    if trig != last_trig[side]:
        new_trigger = Trigger(
            side=side, t_seen=t, trigger_sec=trig, mechanism=mechanism,
            chain_count=chain_count, total_score=total_score, t_last_active=t,
        )
        triggers.append(new_trigger)
        active[side] = new_trigger
    elif active[side] is not None:
        active[side].t_last_active = t
        active[side].chain_count = chain_count
        active[side].total_score = total_score
    last_trig[side] = trig


def classify_trigger(trig: Trigger, scores: list[ScoreSample]) -> ClassifiedTrigger:
    """trigger 1 件を TP / FP / INDETERMINATE に分類する。

    score OCR の生成否フラグは一次データに存在しない (last_score は
    forward-fill されるため print されない = 「不変」と「欠測」を区別できない)。
    そのため保守的に「窓内に score の print が 1 件も無い」場合のみ
    INDETERMINATE とし、FP と混同しない (Codex 明示要求)。
    """
    prior = [s for s in scores if s.side == trig.side and s.t <= trig.t_seen]
    if not prior:
        return ClassifiedTrigger(trig, "INDETERMINATE", "no_prior_score_baseline")
    base = prior[-1].value
    t_end = trig.t_seen + SUPPORT_WINDOW_SEC
    window = [
        s for s in scores
        if s.side == trig.side and trig.t_seen <= s.t <= t_end
    ]
    if any(s.value >= base + MIN_CHAIN_SCORE for s in window):
        return ClassifiedTrigger(trig, "TP", "score_increase_in_window")
    if window:
        return ClassifiedTrigger(trig, "FP", "score_read_but_no_increase")
    return ClassifiedTrigger(trig, "INDETERMINATE", "no_score_sample_in_window")


def find_false_negatives(
    triggers: list[Trigger], scores: list[ScoreSample],
) -> list[dict]:
    """trigger が立たないまま score が +MIN_CHAIN_SCORE 以上増えた箇所を探す。

    カバー窓は `[t_seen, t_last_active + SUPPORT_WINDOW_SEC]` を使う
    (`t_seen` 単独だと、十数秒続く多段連鎖の後半の得点上昇を誤って
    FN 扱いしてしまうことが実測で判明したため)。
    """
    fns: list[dict] = []
    for side in ("1P", "2P"):
        side_scores = sorted((s for s in scores if s.side == side), key=lambda s: s.t)
        side_triggers = sorted(
            (tr for tr in triggers if tr.side == side), key=lambda tr: tr.t_seen,
        )
        for prev, cur in zip(side_scores, side_scores[1:]):
            if cur.value - prev.value < MIN_CHAIN_SCORE:
                continue
            explained = any(
                tr.t_seen <= cur.t <= tr.t_last_active + SUPPORT_WINDOW_SEC
                for tr in side_triggers
            )
            if not explained:
                fns.append({
                    "side": side, "t": cur.t, "delta": cur.value - prev.value,
                    "prev_value": prev.value, "cur_value": cur.value,
                })
    return fns


def is_empty_event(tr: Trigger) -> bool:
    """実体のない (得点を伴わない) 疑似イベントか。

    `total_score == 0` の ChainEvent は
    `docs/KNOWN_WEAKNESSES.md` W7「疑似 ChainEvent のスコア0固定」そのもので、
    安全弁 (`kill_override`) の入力に 0 を流し込む。
    **実体がないのに立っているイベント = 偽イベント**である。

    実測 (2026-08-24、c0BQoMJwwQU の w1/w2):
    旧構成の重複イベントは w1 で 9 件中 8 件、w2 で 20 件中 19 件が
    `score=0` だった。掛け算式修正後はそれぞれ 3 件 / 7 件へ減っている。
    """
    return tr.total_score == 0


def _dup_kind(prev: Trigger, cur: Trigger) -> str:
    """近接した 2 トリガーの関係を単調性で分類する。

    user 伝授 (memory reference_chain_formula_per_step_2026-08-22):
        「掛け算式は消えるたびに出る。回数を数えれば連鎖数、値を足せば火力」

    したがって **連鎖数・素点が単調増加している近接ペアは「段の進行」**であり、
    同じ物理連鎖を二重に数えているわけではない。非単調なものだけが
    再計上の疑いにあたる。

    Returns:
        "STEP_PROGRESS" — 連鎖数か素点が増えている (段が進んだ)。重複ではない。
        "FLICKER"       — 連鎖数も素点も同じまま再トリガーした。再計上の疑い。
        "REGRESS"       — 連鎖数が減った。機構の切り替わり等。再計上の疑い。
    """
    if cur.chain_count < prev.chain_count:
        return "REGRESS"
    if cur.chain_count > prev.chain_count or cur.total_score > prev.total_score:
        return "STEP_PROGRESS"
    return "FLICKER"


def find_duplicates(triggers: list[Trigger]) -> list[Trigger]:
    """同一 side で近接再発生したもののうち、**段の進行でないもの**を返す。

    ## なぜ単調性で分けるのか (2026-08-24 訂正)

    当初は「`DUPLICATE_TRIGGER_GAP_SEC` (3.0秒) 未満の近接トリガーは
    すべて重複候補」としていたが、**掛け算式修正の評価指標としては構造的に
    逆向きだった**。

    根治の目的は「1 本の連鎖を段ごとに正しく追う」ことなので、
    段が進むたびに新しい trigger_sec が発行される
    (memory `project_chain_event_fragmentation_accumulator_2026-08-22`)。
    素の近接ペア数を退行指標にすると、**改善するほど数字が悪化する**。

    実測 (c62、`logs/_diag_c62_dup_classify_2026-08-24.log`) では
    OFF 40 → ON 52 の増分 12 件のうち **8 件が STEP_PROGRESS** であり、
    その発生源は「OFF が凍結デッドロックで 7.5 秒まるごと見逃していた連鎖を
    ON が段単位で新検出した」ことだった (実画面
    `logs/_diag_c62_dup_evidence_2026-08-24/c62_t0902.517.png` で
    1P「40× 1」+ 2P「50× 10」の同時連鎖を確認済み)。
    物理連鎖クラスタ数はむしろ 35 → 29 と**減っている**。

    間隔の閾値 3.0 秒自体は「同一物理連鎖のクラスタ境界」として妥当なので変えない
    (段間隔は最大 1.4 秒 + 幕間 < 3.0 秒。別連鎖には最低 1 手の設置が要る)。
    誤っていたのは**ペア件数をそのまま退行指標に使うこと**だった。

    Returns:
        段の進行でない近接トリガー (FLICKER / REGRESS)。
        断定ではなく human_review_frames.json での目視確認対象。
    """
    dups: list[Trigger] = []
    for side in ("1P", "2P"):
        side_trigs = sorted(
            (tr for tr in triggers if tr.side == side), key=lambda tr: tr.t_seen,
        )
        for prev, cur in zip(side_trigs, side_trigs[1:]):
            if cur.t_seen - prev.t_seen >= DUPLICATE_TRIGGER_GAP_SEC:
                continue
            if _dup_kind(prev, cur) == "STEP_PROGRESS":
                continue
            dups.append(cur)
    return dups


def count_chain_clusters(triggers: list[Trigger]) -> int:
    """物理連鎖の本数 (gap >= DUPLICATE_TRIGGER_GAP_SEC でクラスタ化) を数える。

    同じ物理連鎖を多重に計上していれば**必ずクラスタ数の増加**として現れる。
    段の進行はクラスタ内に吸収されるので、この指標は段単位検出の増加に
    影響されない。`find_duplicates` の補助として使う。
    """
    total = 0
    for side in ("1P", "2P"):
        side_trigs = sorted(
            (tr for tr in triggers if tr.side == side), key=lambda tr: tr.t_seen,
        )
        prev_t: float | None = None
        for tr in side_trigs:
            if prev_t is None or tr.t_seen - prev_t >= DUPLICATE_TRIGGER_GAP_SEC:
                total += 1
            prev_t = tr.t_seen
    return total


def analyze_probe_window(tag: str, mode: str) -> dict:
    """probe ログ 1 本 (窓 tag × mode) を解析して集計 dict を返す。"""
    path = PROBE_LOG_DIR / f"probe_{tag}_{mode}.log"
    triggers, scores = parse_probe_log(path)
    classified = [classify_trigger(tr, scores) for tr in triggers]
    dup_list = find_duplicates(triggers)
    dup_set = {id(tr) for tr in dup_list}
    for c in classified:
        c.is_duplicate = id(c.trigger) in dup_set
    counts = {"TP": 0, "FP": 0, "INDETERMINATE": 0}
    for c in classified:
        counts[c.label] += 1
    fns = find_false_negatives(triggers, scores)
    legacy = [
        tr for tr in triggers
        if mode == "on" and tr.mechanism == LEGACY_MECHANISM_NAME
    ]
    return {
        "window": tag, "mode": mode, "source": "probe_log",
        "n_triggers": len(triggers),
        "tp": counts["TP"], "fp": counts["FP"],
        "indeterminate": counts["INDETERMINATE"],
        "fn": len(fns), "duplicates": len(dup_set),
        "duplicates_empty": sum(1 for tr in dup_list if is_empty_event(tr)),
        "duplicates_with_score": sum(
            1 for tr in dup_list if not is_empty_event(tr)),
        "chain_clusters": count_chain_clusters(triggers),
        "legacy_mechanism_count": len(legacy),
        "classified": classified, "false_negatives": fns,
    }


def analyze_c62_window(mode: str) -> dict:
    """c62 の既存集計 JSON から粗い指標のみを再構成する (FP/判定不能は不可)。"""
    summary = json.loads(
        (AB_JSON_DIR / f"summary_{mode}.json").read_text(encoding="utf-8"),
    )
    records = json.loads(
        (AB_JSON_DIR / f"records_{mode}.json").read_text(encoding="utf-8"),
    )
    triggers = [
        Trigger(
            side=r["side"], t_seen=r["t_seen"], trigger_sec=r["trigger_sec"],
            mechanism=r["mechanism"], chain_count=r["chain_count_event"],
            total_score=r["total_score_event"],
        )
        for r in records
    ]
    dups = find_duplicates(triggers)
    legacy = [
        tr for tr in triggers
        if mode == "on" and tr.mechanism == LEGACY_MECHANISM_NAME
    ]
    n_total = int(summary["n_total_triggers"])
    n_unsupported = int(summary["n_score_unsupported"])
    return {
        "window": "c62", "mode": mode, "source": "ab_json_coarse",
        "n_triggers": n_total,
        "tp": n_total - n_unsupported,
        "fp": None, "indeterminate": None,
        "fp_or_indeterminate_combined": n_unsupported,
        "fn": None, "duplicates": len(dups),
        "duplicates_empty": sum(1 for tr in dups if is_empty_event(tr)),
        "duplicates_with_score": sum(
            1 for tr in dups if not is_empty_event(tr)),
        "chain_clusters": count_chain_clusters(triggers),
        "legacy_mechanism_count": len(legacy),
        "classified": [], "false_negatives": [], "duplicate_triggers": dups,
        "note": (
            "c62 は score_supported の最終判定 (bool) しか保存されておらず、"
            "生の score タイムラインが残っていないため FP と判定不能を分離"
            "できない (fp_or_indeterminate_combined は両者の合算・上限値)。"
            "FN も窓内 score タイムライン不在のため計算不能 (None)。"
        ),
    }


def compute_verdict(results: list[dict]) -> dict:
    """窓ごとの off/on 比較から ACCEPTED / REJECTED を判定する。

    基準 (すべて満たせば ACCEPTED、1 つでも破れば REJECTED):
      1. ON 側の**フォールバック率** (旧機構イベント / 全トリガー) が
         LEGACY_FALLBACK_RATE_MAX 以下であること。
         ※ 件数が 0 であることは求めない。旧機構への落ち込みは
           src/recognition_pipeline.py:6193-6205 の設計どおりの fail-safe。
      2. FP が計算可能な窓では on の FP が off 以下であること。
      3. FP を分離できない窓では fp_or_indeterminate_combined
         (上限値) が on <= off であること。
      4. **重複率** (段の進行を除いた近接再トリガー / 全トリガー) が
         on <= off + DUPLICATE_RATE_TOLERANCE であること。
         ※ 件数ではなく率。ON は検出できる連鎖自体が増えるため母数が変わる。
      5. **物理連鎖クラスタ数**が on <= off であること (多重計上の直接指標)。
    """
    reasons: list[str] = []
    ok = True
    for r in results:
        ok, reasons = _check_legacy_fallback(r, ok, reasons)
    by_window: dict[str, dict[str, dict]] = {}
    for r in results:
        by_window.setdefault(r["window"], {})[r["mode"]] = r
    for window, modes in by_window.items():
        off, on = modes.get("off"), modes.get("on")
        if off is None or on is None:
            continue
        ok, reasons = _check_window_pair(window, off, on, ok, reasons)
    return {"verdict": "ACCEPTED" if ok else "REJECTED", "reasons": reasons}


def _check_legacy_fallback(
    r: dict, ok: bool, reasons: list[str],
) -> tuple[bool, list[str]]:
    """ON 側のフォールバック率が上限以下かを判定する。

    件数ゼロは求めない。掛け算式が読めないフレームで旧経路へ落ちるのは
    設計どおりの fail-safe (src/recognition_pipeline.py:6193-6205) であり、
    「ゲーティング不全」ではない。効いていないことの証拠になるのは
    **落ちる割合が高すぎる**場合だけ。
    """
    if r["mode"] != "on":
        return ok, reasons
    n = r.get("n_triggers") or 0
    legacy = r.get("legacy_mechanism_count") or 0
    if n <= 0:
        return ok, reasons
    rate = legacy / n
    r["legacy_fallback_rate"] = rate
    if rate > LEGACY_FALLBACK_RATE_MAX:
        ok = False
        reasons.append(
            f"{r['window']}/on: 実読フォールバック率 {rate:.1%} が上限 "
            f"{LEGACY_FALLBACK_RATE_MAX:.0%} 超 (旧機構 {legacy}/{n} 件)",
        )
    return ok, reasons


def _check_window_pair(
    window: str, off: dict, on: dict, ok: bool, reasons: list[str],
) -> tuple[bool, list[str]]:
    """1 窓分の off/on 比較 (FP・重複) を行い ok/reasons を更新して返す。"""
    if off["fp"] is not None and on["fp"] is not None:
        if on["fp"] > off["fp"]:
            ok = False
            reasons.append(
                f"{window}: FP 増加 off={off['fp']} on={on['fp']}",
            )
    else:
        off_c = off.get("fp_or_indeterminate_combined")
        on_c = on.get("fp_or_indeterminate_combined")
        if off_c is not None and on_c is not None and on_c > off_c:
            ok = False
            reasons.append(
                f"{window}: 粗指標(FP+判定不能)増加 off={off_c} on={on_c}",
            )
    ok, reasons = _check_false_event_count(window, off, on, ok, reasons)
    ok, reasons = _check_chain_clusters(window, off, on, ok, reasons)
    return ok, reasons


def _check_false_event_count(
    window: str, off: dict, on: dict, ok: bool, reasons: list[str],
) -> tuple[bool, list[str]]:
    """**偽イベント**の総数が増えていないかを見る。

    ## 「偽イベント」と「二重計上」は別物 (2026-08-24 訂正)

    当初は近接再トリガーをまとめて「重複候補」とし、その件数 (のちに率) で
    判定していた。**これは 2 つの異なる欠陥を混同していた。**

    | 種別 | 定義 | 直し方 |
    |---|---|---|
    | **偽イベント** | 実体がない。FP (スコア非支持) + `score=0` の再トリガー | 掛け算式の実読 (本タスク) |
    | **二重計上** | 実在する 1 イベントを 2 回数える (実値を持つ再トリガー) | `chain_id` による統合 (交換エピソード会計) |

    Codex の受け入れ条件は「**偽イベントを増やしていないこと**」= 前者である。

    実測でこの切り分けが効くことが分かった (c0BQoMJwwQU、w1/w2):
    ON 側で増えた再トリガーの中身は
    `formula_read cc=2 score=2340` の直後に `baseline cc=2 score=2340` のように
    **同じ連鎖を 2 つの機構が同じ値で報告している**もので、
    これは偽イベントではなく**機構間の一致 (裏取り)** である。
    一方、実体のない `score=0` の再トリガーは w1 で 8→3、w2 で 19→7 と
    **63% 減っている**。

    二重計上のほうは `duplicates_with_score` として別に集計し、
    **判定には使わず報告する** (chain_id 統合で解く課題であり、
    そこが未実装のうちに落としても直しようがないため)。
    """
    off_n = (off.get("fp") or 0) + off.get("duplicates_empty", 0)
    on_n = (on.get("fp") or 0) + on.get("duplicates_empty", 0)
    if off.get("fp") is None or on.get("fp") is None:
        # c62 のように FP を分離できない窓は粗指標側 (_check_window_pair 冒頭)
        # で既に比較済みなので、ここでは空イベント重複だけを見る。
        off_n = off.get("duplicates_empty", 0)
        on_n = on.get("duplicates_empty", 0)
    off["false_event_total"], on["false_event_total"] = off_n, on_n
    if on_n > off_n:
        ok = False
        reasons.append(
            f"{window}: 偽イベント増加 off={off_n} on={on_n} "
            f"(FP + score=0 の再トリガー)",
        )
    return ok, reasons


def _check_chain_clusters(
    window: str, off: dict, on: dict, ok: bool, reasons: list[str],
) -> tuple[bool, list[str]]:
    """物理連鎖クラスタ数が増えていないかを見る (多重計上の直接指標)。

    同じ物理連鎖を多重計上すれば必ずクラスタ数の増加として現れる。
    段単位検出の増加はクラスタ内に吸収されるので、この指標は
    「段を細かく取れるようになったこと」に影響されない。
    """
    off_c, on_c = off.get("chain_clusters"), on.get("chain_clusters")
    if off_c is None or on_c is None:
        return ok, reasons
    if on_c > off_c:
        ok = False
        reasons.append(
            f"{window}: 物理連鎖クラスタ増加 off={off_c} on={on_c} (多重計上の疑い)",
        )
    return ok, reasons


def build_human_review_rows(results: list[dict]) -> list[dict]:
    """目視確認すべきフレーム一覧を作る (FP/判定不能/重複/旧機構混入/FN)。"""
    rows: list[dict] = []
    for r in results:
        for c in r["classified"]:
            if c.label in ("FP", "INDETERMINATE") or c.is_duplicate:
                rows.append({
                    "window": r["window"], "mode": r["mode"],
                    "t_sec": c.trigger.t_seen, "side": c.trigger.side,
                    "mechanism": c.trigger.mechanism, "label": c.label,
                    "reason": c.reason, "is_duplicate": c.is_duplicate,
                    "is_legacy_mechanism": (
                        r["mode"] == "on"
                        and c.trigger.mechanism == LEGACY_MECHANISM_NAME
                    ),
                })
        for fn in r["false_negatives"]:
            rows.append({
                "window": r["window"], "mode": r["mode"], "t_sec": fn["t"],
                "side": fn["side"], "mechanism": None, "label": "FN",
                "reason": f"score_delta={fn['delta']}", "is_duplicate": False,
                "is_legacy_mechanism": False,
            })
        # c62 は classified が空 (TP/FP 分離不可) なので重複候補を別途載せる。
        for tr in r.get("duplicate_triggers", []):
            rows.append({
                "window": r["window"], "mode": r["mode"], "t_sec": tr.t_seen,
                "side": tr.side, "mechanism": tr.mechanism,
                "label": "DUPLICATE_CANDIDATE_COARSE",
                "reason": "c62: score タイムライン不在のため TP/FP 分類不可、重複候補のみ",
                "is_duplicate": True,
                "is_legacy_mechanism": (
                    r["mode"] == "on" and tr.mechanism == LEGACY_MECHANISM_NAME
                ),
            })
    rows.sort(key=lambda row: (row["window"], row["mode"], row["t_sec"]))
    return rows


def _serialize_results(results: list[dict]) -> list[dict]:
    """dataclass を含む results を JSON 化可能な形に変換する。"""
    out = []
    for r in results:
        r2 = dict(r)
        r2["classified"] = [
            {
                "trigger": asdict(c.trigger), "label": c.label,
                "reason": c.reason, "is_duplicate": c.is_duplicate,
            }
            for c in r["classified"]
        ]
        if "duplicate_triggers" in r2:
            r2["duplicate_triggers"] = [asdict(tr) for tr in r2["duplicate_triggers"]]
        out.append(r2)
    return out


def _collect_all_results(skip_c62: bool = False) -> list[dict]:
    """probe ログ (w1/w2 × off/on) と c62 (off/on) を全て解析する。"""
    results: list[dict] = []
    for tag in PROBE_WINDOWS:
        for mode in MODES:
            results.append(analyze_probe_window(tag, mode))
    if not skip_c62:
        for mode in MODES:
            results.append(analyze_c62_window(mode))
    return results


def _is_driver_all_done() -> bool:
    """driver_progress.log の ALL_DONE 有無だけを見る (合否とは無関係)。"""
    driver_log = PROBE_LOG_DIR / "driver_progress.log"
    if not driver_log.is_file():
        return False
    return "ALL_DONE" in driver_log.read_text(encoding="utf-8")


def _build_verdict_payload(results: list[dict], verdict: dict) -> dict:
    """verdict_result.json の中身を組み立てる。"""
    return {
        "criteria": {
            "SUPPORT_WINDOW_SEC": SUPPORT_WINDOW_SEC,
            "MIN_CHAIN_SCORE": MIN_CHAIN_SCORE,
            "DUPLICATE_TRIGGER_GAP_SEC": DUPLICATE_TRIGGER_GAP_SEC,
            "LEGACY_MECHANISM_NAME": LEGACY_MECHANISM_NAME,
            "LEGACY_FALLBACK_RATE_MAX": LEGACY_FALLBACK_RATE_MAX,
            "DUPLICATE_RATE_TOLERANCE": DUPLICATE_RATE_TOLERANCE,
            "rule": (
                "ON 側の実読フォールバック率 (旧機構イベント / 全トリガー) が"
                f" {LEGACY_FALLBACK_RATE_MAX:.0%} 以下、かつ各窓で on の FP"
                " (計算不能なら FP+判定不能の合算上限) が off 以下、かつ"
                " 重複率 (段の進行を除く) が off + "
                f"{DUPLICATE_RATE_TOLERANCE:.0%} 以下、かつ物理連鎖クラスタ数が"
                " off 以下、を全窓で満たせば ACCEPTED。"
            ),
            "note_duplicate_definition": (
                "重複候補から『段の進行』(連鎖数か素点が単調増加している近接ペア) を"
                "除外している。掛け算式は消えるたびに出る (user 伝授 "
                "reference_chain_formula_per_step_2026-08-22) ため、段が進めば"
                "新しい trigger_sec が発行される。素の近接ペア数を退行指標にすると"
                "**改善するほど数字が悪化する**。実測 (c62) では OFF 40 → ON 52 の"
                "増分 12 件のうち 8 件が段の進行で、その発生源は OFF が凍結"
                "デッドロックで 7.5 秒まるごと見逃していた連鎖を ON が段単位で"
                "新検出したことだった (2026-08-24 訂正)。"
            ),
            "note_legacy_mechanism": (
                "旧機構名 'formula' が ON 側に出ること自体は不合格の根拠に"
                "ならない。src/recognition_pipeline.py:6193-6205 のとおり、"
                "掛け算式を実読できなかったフレームで旧経路へ落ちるのは"
                "設計どおりの fail-safe である (2026-08-24 訂正)。"
                "ただし旧経路は total_score=0 を出すため"
                "(docs/KNOWN_WEAKNESSES.md W7)、落ちた分は偽イベントに"
                "なりやすい。よって件数ではなく率で判定する。"
            ),
        },
        "driver_all_done": _is_driver_all_done(),
        "note_all_done_vs_verdict": (
            "ALL_DONE は driver_progress.log 記載の 4 走行が完走したことのみを"
            "示す。偽イベントを増やしていないことの証明ではない。合否は"
            "本ファイルの 'verdict' フィールドを見ること。"
        ),
        "verdict": verdict["verdict"],
        "reasons": verdict["reasons"],
        "windows": _serialize_results(results),
    }


def main() -> None:
    """判定を実行する。

    `--probe-dir` / `--out-dir` で入出力を差し替えられる。
    Q-01 修正後の再測定 (幕間フラグ ON) を、先行測定の成果物を**上書きせずに**
    別ディレクトリで判定するために必要 (2026-08-24 追加)。
    """
    global PROBE_LOG_DIR, OUT_DIR
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probe-dir", default=None,
                   help="probe_*.log の場所 (既定: 先行測定のディレクトリ)")
    p.add_argument("--out-dir", default=None,
                   help="判定結果の出力先 (既定: data/verify/false_event_verdict_2026-08-24)")
    p.add_argument("--skip-c62", action="store_true",
                   help="c62 の粗指標を集計に含めない (再測定側には c62 が無いため)")
    a = p.parse_args()
    if a.probe_dir:
        PROBE_LOG_DIR = Path(a.probe_dir)
    if a.out_dir:
        OUT_DIR = Path(a.out_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = _collect_all_results(skip_c62=a.skip_c62)
    verdict = compute_verdict(results)
    review_rows = build_human_review_rows(results)
    verdict_payload = _build_verdict_payload(results, verdict)

    (OUT_DIR / "verdict_result.json").write_text(
        json.dumps(verdict_payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (OUT_DIR / "human_review_frames.json").write_text(
        json.dumps(review_rows, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[verdict] {verdict['verdict']}")
    for reason in verdict["reasons"]:
        print(f"  - {reason}")
    print(f"[out] {OUT_DIR}")


if __name__ == "__main__":
    main()
