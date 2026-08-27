"""残る6%不一致の原因切り分け (2026-08-21、coordinator依頼)。

既存の dump (data/verify/zenchi_warmup_2026-08-21/{ref,w5,w26}.npz、
2026-08-21作成済み・再生成不要) を読み、不一致行について
b1_hash/b2_hash (盤面自体) が一致しているかで
「認識側 (盤面がずれている)」か「判定側内部状態 (盤面は同じだが adv がずれる)」
かを一発で切り分ける。

さらに online_hsv 較正注入ログとの時刻関係、pending/score/state 列の一致も見る。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualize_advantage_overlay import load_timeline_dump  # noqa: E402

DUMP_DIR = PROJECT_ROOT / "data" / "verify" / "zenchi_warmup_2026-08-21"
COMPARE_FROM = 3330.0
COMPARE_TO = 3360.0

# online_hsv injected ログの時刻 (frame -> t_sec = frame/60)
ONLINE_HSV_FRAMES = {
    "ref": 198008, "w0": 199806, "w5": 199268, "w15": 198668, "w26": 198008,
}


def _load(name: str):
    path = DUMP_DIR / f"{name}.npz"
    _, rows = load_timeline_dump(path)
    return [r for r in rows if COMPARE_FROM <= r.t_sec <= COMPARE_TO]


def analyze(ref_rows, test_rows, label: str) -> None:
    n = min(len(ref_rows), len(test_rows))
    board_mismatch = []
    judgment_only_mismatch = []
    input_mismatch = []  # pending/score/state/game_idx が違う行
    for i in range(n):
        rr, tr = ref_rows[i], test_rows[i]
        if abs(rr.t_sec - tr.t_sec) > 0.05:
            continue  # 時刻がずれている行は別途報告済み (w0系)
        board_same = (rr.b1_hash == tr.b1_hash and rr.b2_hash == tr.b2_hash)
        adv_same = abs(rr.adv_raw - tr.adv_raw) < 1e-6
        input_same = (
            rr.pending_p1 == tr.pending_p1 and rr.pending_p2 == tr.pending_p2
            and rr.score1 == tr.score1 and rr.score2 == tr.score2
            and rr.game_idx == tr.game_idx
            and rr.state1 == tr.state1 and rr.state2 == tr.state2
        )
        if not board_same:
            board_mismatch.append((rr.t_sec, rr, tr))
        elif not adv_same:
            judgment_only_mismatch.append((rr.t_sec, rr, tr))
        if not input_same:
            input_mismatch.append((rr.t_sec, rr, tr))

    print(f"\n{'=' * 70}\n[{label}] 比較行数={n}\n{'=' * 70}")
    print(f"  盤面ハッシュ不一致 (b1_hash/b2_hash) = {len(board_mismatch)}件 "
          f"({len(board_mismatch)/n*100:.1f}%) ← 認識側のずれ")
    print(f"  盤面一致だが判定値(adv_raw)不一致    = {len(judgment_only_mismatch)}件 "
          f"({len(judgment_only_mismatch)/n*100:.1f}%) ← 判定側の内部状態のずれ")
    print(f"  入力列(pending/score/state/game_idx)不一致 = {len(input_mismatch)}件 "
          f"({len(input_mismatch)/n*100:.1f}%)")

    if board_mismatch:
        t0 = board_mismatch[0][0]
        t1 = board_mismatch[-1][0]
        print(f"  盤面不一致の時間範囲: {t0:.2f}s 〜 {t1:.2f}s "
              f"({t1-t0:.1f}秒間、以後全て不一致かどうかは下記件数で判断)")
        rr, tr = board_mismatch[0][1], board_mismatch[0][2]
        print(f"  最初の盤面不一致 詳細: t={t0:.2f}s "
              f"ref(b1={rr.b1_hash},b2={rr.b2_hash}) "
              f"test(b1={tr.b1_hash},b2={tr.b2_hash}) "
              f"state1={rr.state1}/{tr.state1} state2={rr.state2}/{tr.state2}")
    if judgment_only_mismatch:
        t0 = judgment_only_mismatch[0][0]
        rr, tr = judgment_only_mismatch[0][1], judgment_only_mismatch[0][2]
        print(f"  最初の「盤面同一だが判定差」: t={t0:.2f}s "
              f"adv_raw ref={rr.adv_raw:.3f} test={tr.adv_raw:.3f} "
              f"adv_ema ref={rr.adv_ema:.3f} test={tr.adv_ema:.3f}")

    # online_hsv injected 時刻との関係
    ref_frame = ONLINE_HSV_FRAMES.get("ref")
    test_frame = ONLINE_HSV_FRAMES.get(label)
    if ref_frame and test_frame:
        ref_t = ref_frame / 60.0
        test_t = test_frame / 60.0
        print(f"  online_hsv注入時刻: ref={ref_t:.1f}s test({label})={test_t:.1f}s "
              f"(比較窓[{COMPARE_FROM},{COMPARE_TO}]の{'内側' if COMPARE_FROM<=ref_t<=COMPARE_TO or COMPARE_FROM<=test_t<=COMPARE_TO else '外側 (窓より前)'})")


def main() -> None:
    ref = _load("ref")
    print(f"ref行数={len(ref)}")
    for name in ("w5", "w26"):
        test = _load(name)
        analyze(ref, test, name)


if __name__ == "__main__":
    main()
