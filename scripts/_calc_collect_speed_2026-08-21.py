"""実際の収集ログから「動画1分あたり何秒かかるか」を算出する (2026-08-21)。

user 質問「今10分の動画を読み込むのに何秒かかるの?」への回答用。

## なぜログから出すか

ベンチスクリプトの数字は短いクリップで測るので、実際の動画1本を通した
ときの速度とずれる (試合外の演出・場面転換・盤面が満杯の重い区間などが
平均に入らない)。実際の本番収集は「1本ごとの所要秒」をログに残しているので、
それと npz に記録された動画の長さ (t_sec の最大値) を突き合わせれば
**本番構成そのままの実効速度**が出る。

## 注意

本番収集は 14並列。速度の数字は並列数とセットでないと意味がない
(memory feedback_speed_claims_need_parallelism_2026-08-20)。
単独実行の値は別途ベンチで測る必要がある。
"""
from __future__ import annotations

import glob
import os
import re
import statistics
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 収集所要秒を記録しているログ (複数waveぶん)
_LOGS = (
    "logs/regen_add14_2026-08-21.log",
    "logs/regen_model50v2_2026-08-20.log",
)
_NPZ_DIR = "data/indicators_v2/boards_lean_model50v2_2026-08-20"
_MIN_LEN_SEC = 60.0  # 極端に短い動画は平均を歪めるので除外
_PARALLEL = 14  # 本番収集の並列数 (数字の意味づけに必須)


def _collect_seconds() -> dict[str, float]:
    """ログから 動画ID -> 所要秒 を拾う。"""
    txt = ""
    for lg in _LOGS:
        p = PROJECT_ROOT / lg
        if p.exists():
            txt += p.read_text(encoding="utf-8", errors="ignore")
    pairs = re.findall(r"\[collect\]\[(\S+)\] status=OK rows=\d+ sec=([0-9.]+)", txt)
    return {vid: float(s) for vid, s in pairs}


def main() -> int:
    """実収集ログから実効速度を出す。"""
    secs = _collect_seconds()
    rows: list[tuple[str, float, float, float]] = []
    for f in sorted(glob.glob(str(PROJECT_ROOT / _NPZ_DIR / "*.npz"))):
        vid = os.path.basename(f).replace(".npz", "")
        if vid not in secs:
            continue
        try:
            z = np.load(f, allow_pickle=True)
            length = float(np.max(z["t_sec"]))
        except Exception:
            continue
        if length < _MIN_LEN_SEC:
            continue
        s = secs[vid]
        rows.append((vid, length, s, s / (length / 60.0)))

    if not rows:
        print("[error] 突合できる動画が無い")
        return 1

    rows.sort(key=lambda r: r[3])
    print(f"{'動画':>8} {'長さ(分)':>9} {'所要(秒)':>9} {'1分あたり秒':>11}")
    for v, t, s, r in rows[:6]:
        print(f"{v:>8} {t / 60:9.1f} {s:9.0f} {r:11.1f}")
    print(f"{'...':>8}")
    for v, t, s, r in rows[-3:]:
        print(f"{v:>8} {t / 60:9.1f} {s:9.0f} {r:11.1f}")

    med = statistics.median(r[3] for r in rows)
    print(f"\n本数={len(rows)}  中央値: 動画1分あたり {med:.1f}秒 ({_PARALLEL}並列)")
    print(f"→ 10分の動画:  {med * 10:.0f}秒 = {med * 10 / 60:.1f}分 ({_PARALLEL}並列)")
    print(f"→ 117分の動画: {med * 117 / 3600:.1f}時間 ({_PARALLEL}並列の1枠)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
