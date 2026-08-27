"""Gate 3R-6 P1 是正 (境界の正式受理ラッチ、2026-08-26) の再検収。

Codex 第26報レビュー要件5・6 に対応する。既存の検収器
`scripts/_verify_gate3r6_boundary_confirm_selfcheck_2026-08-25.py` は
上書きせず別ファイルにする (明示パスのみ・glob 不使用)。

追加した検査 (前版に無いもの):
  - 【要件5】`total_boundaries` が実際の境界数と一致するか。実際の境界数は
    ON dump の `game_idx` の遷移回数から独立に数える (ログの自己申告と
    突き合わせる。カウンタの値だけを見て合格としない)。
  - 【要件6】P1 是正の前後で OFF 出力が bit-identical か。是正は
    `resolve_boundary_confirmations` の呼出回数を変えるが、OFF では death 列が
    dump に出ないため一致するはず。「はず」で済ませず旧 OFF と直接突合する。
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np

D = Path("data/verify/gate3r6_boundary_confirm_2026-08-25")
LOG = Path("logs/gate3r6_p1v2_2026-08-26/recheck.log")

# P1 是正後 (今回取り直した分)
ON = D / "first5games_boundaryconfirm_on_p1v2.npz"
OFFS = [D / f"first5games_boundaryconfirm_off_p1v2_run{i}.npz" for i in (1, 2, 3)]
GROSS = D / "first5games_boundaryconfirm_gross_p1v2.npz"
ON_GROSS = D / "first5games_boundaryconfirm_on_gross_p1v2.npz"
# P1 是正前 (第26報の成果物、上書きせず読むだけ)
OLD_OFFS = [D / f"first5games_boundaryconfirm_off_run{i}.npz" for i in (1, 2, 3)]
# P1 是正 第1版 (game_idx を正式境界と分離したまま) の OFF。第2版との差分確認用。
OLD_P1FIX_OFF = D / "first5games_boundaryconfirm_off_p1fix_run1.npz"
OLD_ON = D / "first5games_boundaryconfirm_on.npz"

# 第26報で確定した3区間 (実際の敗北側との突合は Codex が PASS 判定済み)。
EXPECTED_CONFIRM_T = (232.467, 278.100, 335.967)


def _pct(num: int, den: int) -> str:
    return f"{num}/{den} (母数0)" if den == 0 else f"{num}/{den} ({100.0 * num / den:.2f}%)"


def _eq(x, y) -> bool:
    if x.shape != y.shape:
        return False
    if x.dtype.kind in "fc":
        return np.array_equal(np.nan_to_num(x, nan=-9e99), np.nan_to_num(y, nan=-9e99))
    return np.array_equal(x, y)


def _cmp(label: str, a_path: Path, b_path: Path,
         window: tuple[float, float] | None = None) -> None:
    if not a_path.exists() or not b_path.exists():
        print(f"{label}: ファイル不足 ({a_path.name} / {b_path.name})")
        return
    a, b = np.load(a_path, allow_pickle=True), np.load(b_path, allow_pickle=True)
    shared = sorted(set(a.files) & set(b.files))
    if window is not None and "t_sec" in shared:
        lo, hi = window
        ta, tb = a["t_sec"], b["t_sec"]
        ma, mb = (ta >= lo) & (ta <= hi), (tb >= lo) & (tb <= hi)
        print(f"  [窓確認] {label}: a側 {int(ma.sum())}行 / b側 {int(mb.sum())}行")
        bad = [k for k in shared if not _eq(
            a[k][ma] if a[k].ndim > 0 else a[k],
            b[k][mb] if b[k].ndim > 0 else b[k])]
    else:
        bad = [k for k in shared if not _eq(a[k], b[k])]
    print(f"{label}: 不一致 {len(bad)}/{len(shared)} (母数={len(shared)}共通キー)"
          + (f" -> {bad}" if bad else ""))


def _confirmed_runs(d, side: int) -> list[tuple[float, float]]:
    """`is_dead{side}_confirmed` が True の連続区間を返す。"""
    k = f"is_dead{side}_confirmed"
    if k not in d.files:
        return []
    t, v = d["t_sec"], d[k].astype(bool)
    idx = np.flatnonzero(v)
    if len(idx) == 0:
        return []
    runs, s = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b != a + 1:
            runs.append((float(t[s]), float(t[a])))
            s = b
    runs.append((float(t[s]), float(t[idx[-1]])))
    return runs


def main() -> int:
    if not ON.exists():
        print(f"=== 中止: ON dump が無い ({ON}) — 再検収がまだ完走していない ===")
        return 1
    d = np.load(ON, allow_pickle=True)
    t = d["t_sec"]
    print(f"ON: 行数={len(t)} t={t.min():.2f}〜{t.max():.2f}")

    print("\n--- 検収3項目 (母数併記) ---")
    for name, lo, hi, side, want in (
        ("既知の誤判定(1P t=164.03-164.73)", 163.5, 165.5, 1, False),
        ("真の窒息(2P t=223)", 220.0, 235.0, 2, True),
        ("まちうけ画面(2P t=18-90.5)", 18.0, 90.5, 2, False),
    ):
        k = f"is_dead{side}_confirmed"
        if k not in d.files:
            print(f"  {k}: 列なし")
            continue
        m = (t >= lo) & (t <= hi)
        n = int(d[k].astype(bool)[m].sum())
        ok = (n > 0) == want
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: True {_pct(n, int(m.sum()))} "
              f"(期待={'あり' if want else 'なし'})")

    print("\n--- 境界確定3件が維持されているか (第26報と同じ時刻か) ---")
    runs = _confirmed_runs(d, 2)
    starts = [r[0] for r in runs]
    print(f"  is_dead2_confirmed の確定区間 {len(runs)}本: "
          + ", ".join(f"t={a:.3f}〜{b:.3f}" for a, b in runs))
    for want_t in EXPECTED_CONFIRM_T:
        hit = any(abs(s - want_t) < 0.5 for s in starts)
        print(f"  [{'PASS' if hit else 'FAIL'}] t={want_t} の確定が残っている")

    print("\n--- 【要件5】total_boundaries が実際の境界数と一致するか ---")
    real_boundaries = int(np.count_nonzero(np.diff(d["game_idx"]) != 0))
    print(f"  ON dump の game_idx 遷移から数えた実境界数: {real_boundaries} "
          f"(game_idx の異なり数 {len(np.unique(d['game_idx']))})")
    if LOG.exists():
        txt = LOG.read_text(encoding="utf-8", errors="replace")
        hits = re.findall(r"境界イベント総数\s+(\d+)", txt)
        print(f"  ログ中の『境界イベント総数』の出現: {hits if hits else '見つからない'}")
        for h in hits:
            ok = int(h) == real_boundaries
            print(f"  [{'PASS' if ok else 'FAIL'}] total_boundaries={h} "
                  f"vs 実境界数={real_boundaries}")
    else:
        print(f"  ログが無い ({LOG}) — total_boundaries を突合できない")

    print("\n--- OFF 3run bit-identical (P1是正後、動画全体) ---")
    for i, j in itertools.combinations(range(3), 2):
        _cmp(f"OFF run{i+1} vs run{j+1}", OFFS[i], OFFS[j])

    print("\n--- P1是正 第1版 vs 第2版 の OFF 差分 ---")
    print("  第2版は game_idx 加算を正式境界へ統合したため、OFF 出力の game_idx 列は")
    print("  **意図して変わる** (Codex 対応5: 古い dump との bit-identical 維持を")
    print("  理由に欠陥を残さない)。どの列が変わったかを明示して確認する。")
    _cmp("第1版OFF run1 vs 第2版OFF run1", OLD_P1FIX_OFF, OFFS[0])

    print("\n--- ON vs OFF 共通キー (同一窓 t=0〜420) ---")
    _cmp("ON vs OFF run1", ON, OFFS[0], window=(0.0, 420.0))

    print("\n--- gross のみ vs gross+death (同一窓) ---")
    _cmp("gross vs gross+death", GROSS, ON_GROSS, window=(0.0, 420.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
