"""kill_override 修正① (fctracker.inc1/inc2 -> snap.pending_p1/pending_p2) の
自己検収用 短区間 新旧比較 (2026-08-22)。

scripts/_diag_kill_override_wiring_2026-08-22.py (根因調査時の計装) と同じ
monkeypatch方式を再利用し、3アンカー (t=886.5/200.3/3228) を含む短窓だけを
処理する (フルレンダは行わない、指示厳守)。

## 検証項目 (受け入れ条件)
  1. アンカー3点で kill_override の出力符号が「致死側が不利」になっているか
  2. 陽性対照: 呼出側の実引数を意図的に旧経路 (fctracker.inc1/inc2) に
     戻すと、同じアンカーで異常 (方向逆転/見落とし) が再現するか
  3. 安全弁の発火回数 (fired_full/partial) が新旧でどう変わるか (過剰発火確認)

## 新旧の切替方法
ソースファイルを書き換えず (レビュー対象の本修正コミットをそのまま使う)、
generate() 内部で `kill_override(adv, snap.pending_p1, snap.pending_p2, ...)`
を直接 monkeypatch するのは呼び出し側の引数式そのものを差し替えられない
(ソース内リテラル式のため)。そこで `vao.kill_override` 自体を差し替え、
"旧経路互換ラッパ" (_STATE に保持した fctracker.inc1/inc2 を使って再計算する
関数) に挿げ替えることで、修正コミットのソースを一切変更せずに
「もし旧経路の値を使っていたら」を再現する (陽性対照専用、本番コードには
一切影響しない)。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
cv2.setNumThreads(1)

import scripts.visualize_advantage_overlay as vao  # noqa: E402

VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
MODEL_DIR = PROJECT_ROOT / "data/verify/retrain_model62_2026-08-21"
OUT_DIR = PROJECT_ROOT / "logs/_verify_killoverride_fix_2026-08-22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# アンカー: (窓の名前, start_sec, end_sec, warmup_sec, 監視窓(下限,上限), 表示用アンカー秒)
ANCHORS = [
    ("A_t886", 848.0, 905.0, 28.0, (878.0, 902.0), 886.533),
    ("B_t200", 165.0, 215.0, 30.0, (195.0, 212.0), 200.3),
    ("C_t3228", 3190.0, 3260.0, 30.0, (3218.0, 3238.0), 3228.0),
]


def _run_one(
    window_name: str, start_sec: float, end_sec: float, warmup_sec: float,
    watch: tuple[float, float], use_legacy_inc: bool,
) -> dict:
    """1アンカー分を実行し、kill_override呼出ログと集計を返す。

    use_legacy_inc=True: kill_override を「fctracker.inc1/inc2 を使う旧経路
    互換」ラッパに差し替える (陽性対照専用)。False なら本修正コミットの
    ソースをそのまま実行する (=snap.pending_p1/p2、新経路)。
    """
    lo, hi = watch
    detail: list[str] = []
    counters = {"calls": 0, "fired_full": 0, "fired_partial": 0}

    orig_kill_override = vao.kill_override
    orig_drive_ojama = vao._drive_ojama
    orig_fc_update = vao.RealtimeForecastTracker.update
    state: dict = {"t": None, "pending_p1": None, "pending_p2": None,
                   "inc1": None, "inc2": None}

    def patched_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw):
        snap = orig_drive_ojama(tracker, rp1, rp2, ps1, ps2, t, **kw)
        state["t"] = t
        state["pending_p1"] = snap.pending_p1
        state["pending_p2"] = snap.pending_p2
        return snap

    def patched_fc_update(self, score1, score2, tsumo1, tsumo2, rate=70.0):
        result = orig_fc_update(self, score1, score2, tsumo1, tsumo2, rate=rate)
        state["inc1"], state["inc2"] = self.inc1, self.inc2
        return result

    def patched_kill_override(adv, arg_inc1, arg_inc2, room1, room2):
        # use_legacy_inc=True の場合だけ、実引数 (=snap.pending、新経路) を
        # 無視して fctracker.inc1/inc2 (旧経路の値) に差し替える。
        # False の場合は実引数をそのまま使う (=本修正コミットの実際の挙動)。
        inc1 = state["inc1"] if use_legacy_inc and state["inc1"] is not None else arg_inc1
        inc2 = state["inc2"] if use_legacy_inc and state["inc2"] is not None else arg_inc2
        counters["calls"] += 1
        result = orig_kill_override(adv, inc1, inc2, room1, room2)
        fired_full = abs(result) == 100.0 and result != adv
        fired_partial = result != adv and not fired_full
        if fired_full:
            counters["fired_full"] += 1
        elif fired_partial:
            counters["fired_partial"] += 1
        t = state["t"]
        if t is not None and lo <= t <= hi:
            detail.append(
                f"t={t:.3f} adv_in={adv:+.2f} used_inc1={inc1:.2f} used_inc2={inc2:.2f} "
                f"true_pending_p1={state['pending_p1']} true_pending_p2={state['pending_p2']} "
                f"room1={room1} room2={room2} adv_out={result:+.2f}"
            )
        return result

    vao.kill_override = patched_kill_override
    vao._drive_ojama = patched_drive_ojama
    vao.RealtimeForecastTracker.update = patched_fc_update
    try:
        vao.generate(
            video=VIDEO,
            out=OUT_DIR / f"_dummy_{window_name}.mp4",
            max_sec=0.0,
            sample_interval=0.15,
            start_sec=start_sec,
            end_sec=end_sec,
            warmup_sec=warmup_sec,
            model_dir=MODEL_DIR,
            force_in_match=False,
            enable_early_fire_reaction=True,
            enable_resolved_exchange_eval=True,
            enable_resolved_decisive_amplify=True,
            enable_resolved_live_defender=True,
            enable_resolved_live_defender_strict=True,
            layout="panel",
            panel_subtitle_h=0,
            render=False,
            dump_timeline_path=None,
        )
    finally:
        vao.kill_override = orig_kill_override
        vao._drive_ojama = orig_drive_ojama
        vao.RealtimeForecastTracker.update = orig_fc_update

    label = "LEGACY(fctracker.inc)" if use_legacy_inc else "FIXED(snap.pending)"
    log_path = OUT_DIR / f"{window_name}_{'legacy' if use_legacy_inc else 'fixed'}.log"
    log_path.write_text(
        f"# {window_name} {label}\n" + "\n".join(detail), encoding="utf-8")
    print(f"[{window_name}] {label}: calls={counters['calls']} "
          f"fired_full={counters['fired_full']} fired_partial={counters['fired_partial']} "
          f"-> {log_path}")
    return counters


def main() -> None:
    for name, s, e, w, watch, anchor_t in ANCHORS:
        print(f"=== {name} (アンカー t={anchor_t}s, 窓 {s}-{e}s, 暖機{w}s) ===")
        _run_one(name, s, e, w, watch, use_legacy_inc=False)
        _run_one(name, s, e, w, watch, use_legacy_inc=True)


if __name__ == "__main__":
    main()
