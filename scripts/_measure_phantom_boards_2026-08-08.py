"""非試合画面の「幻満杯盤面」混入量を全動画で測る (診断専用、2026-08-08).

品質ゲート FAIL トリアージ ([[project-quality-gate-fail-triage-2026-08-08]]) で、
対戦カード紹介の白背景演出やロビー画面で盤面が誤認識され、 STABLE
スナップショットとして npz に記録されていることが実画面で確定した
(c12 t=133.4s / c27 t=1.20s)。 本スクリプトはその混入量をライブラリ全体で
測り、 「一部動画の問題か、 全体の問題か」 を切り分ける。

## 検知の物理的根拠 (シーン逆算禁止則に従い、 定数は物理から導く)
実戦では盤面 (可視 12 段 × 6 列) がおじゃまでほぼ埋まった時点で窒息死する。
死ねば試合が終わり盤面はリセットされるので、 **おじゃま比率が極端に高い
満杯盤面が STABLE として何フレームも記録されること自体があり得ない**。
よって以下を「幻盤面」の署名とする:
  - 非空セル数 >= PHANTOM_MIN_NONEMPTY (盤面の大半が埋まっている)
  - かつ おじゃま比率 >= PHANTOM_MIN_OJAMA_RATIO (色ぷよでなくおじゃま)
特定動画を狙った値ではなく、 「即死盤面が安定継続するのは物理的に不可能」
という一点から決めた閾値である。

出力: data/verify/phase_l_quality_gate_2026-08-07/phantom_boards_2026-08-08.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import COLOR_EMPTY, COLOR_OJAMA  # noqa: E402

NPZ_DIR = _ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-07"
OUT_TSV = (
    _ROOT / "data" / "verify" / "phase_l_quality_gate_2026-08-07"
    / "phantom_boards_2026-08-08.tsv"
)

# 可視段 (隠し段 row0 を除く 12 段) × 6 列 = 72 セル。
VISIBLE_ROW_LO: int = 1
# 「盤面の大半が埋まっている」= 可視 72 セル中 48 セル (2/3) 以上。
# 実戦でここまで埋まると窒息が目前で、 長く安定継続しない。
PHANTOM_MIN_NONEMPTY: int = 48
# 非空セルのうちおじゃまが占める比率。 相手の連鎖でおじゃまが降り切っても
# 自分の色ぷよが土台に残るため、 実戦で 0.7 を超える盤面は事実上死んでいる。
PHANTOM_MIN_OJAMA_RATIO: float = 0.7


def _measure(npz_path: Path) -> tuple[int, int, float, list[tuple[str, float, int]]]:
    """1 動画の幻盤面スナップショット数・総数・率・代表を返す."""
    d = np.load(npz_path, allow_pickle=True)
    grids = d["grids"][:, VISIBLE_ROW_LO:, :]
    nonempty = (grids != COLOR_EMPTY).sum(axis=(1, 2))
    ojama = (grids == COLOR_OJAMA).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(nonempty > 0, ojama / np.maximum(nonempty, 1), 0.0)
    hit = (nonempty >= PHANTOM_MIN_NONEMPTY) & (ratio >= PHANTOM_MIN_OJAMA_RATIO)
    n_hit = int(hit.sum())
    n_all = int(grids.shape[0])
    reps: list[tuple[str, float, int]] = []
    if n_hit:
        idx = np.argwhere(hit).ravel()[:3]
        for i in idx:
            reps.append((str(d["side"][i]), float(d["t_sec"][i]), int(nonempty[i])))
    return n_hit, n_all, (n_hit / n_all if n_all else 0.0), reps


def main() -> int:
    npzs = sorted(NPZ_DIR.glob("*.npz"))
    if not npzs:
        print(f"npz が無い: {NPZ_DIR}")
        return 1
    rows: list[str] = ["video_id\tphantom_snaps\ttotal_snaps\tphantom_rate\trepresentatives"]
    results: list[tuple[float, str, int]] = []
    for p in npzs:
        try:
            n_hit, n_all, rate, reps = _measure(p)
        except Exception as e:  # 破損 npz は行を残して継続
            rows.append(f"{p.stem}\tERROR\tERROR\tERROR\t{e}")
            continue
        rep_s = "; ".join(f"{s} t={t:.2f} cells={c}" for s, t, c in reps) or "-"
        rows.append(f"{p.stem}\t{n_hit}\t{n_all}\t{rate:.6f}\t{rep_s}")
        results.append((rate, p.stem, n_hit))
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_TSV.write_text("\n".join(rows) + "\n", encoding="utf-8")
    results.sort(reverse=True)
    n_affected = sum(1 for r, _, _ in results if r > 0)
    print(f"動画数={len(results)} 幻盤面ありの動画={n_affected}")
    print("--- 混入率 上位 15 ---")
    for rate, vid, n_hit in results[:15]:
        print(f"  {vid:10s} rate={rate:.4%} snaps={n_hit}")
    print(f"出力: {OUT_TSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
