"""掛け算式 全編スキャンのセッション解析 (2026-08-24 コーダ)。

logs/_scan_formula_3digit_left_2026-08-24/all_valid.tsv (0.2 秒間引きの
有効読取り全件) を FormulaStepAccumulator (本番と同一ロジック) に通して
連鎖セッションへ集約し、以下を出力する:

  - セッション一覧 (side, 開始/終了, 段数, 素点合計, 最終右辺)
  - ルール整合チェック: 最終段の右辺 = 連鎖ボーナス表[最終連鎖数] + 残差
    (残差は 連結ボーナス max10 + 色ボーナス max12 の範囲 [0,22] に入るはず。
    段数がテーブル位置と一致するかで読み落とし/数え過ぎを独立検定する)
  - 大連鎖 10 ケース窓との突合表
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.score_ocr import FormulaReadResult, FormulaStepAccumulator  # noqa: E402
from src.scoring import chain_power  # noqa: E402

SCAN = Path("logs/_scan_formula_3digit_left_2026-08-24/all_valid.tsv")

# 大連鎖10ケース (logs/_analyst_telop_check_2026-08-24/_extract.py と同一)
CASES = [
    ("c01", "1P", 6702.5, 6718.9),
    ("c02", "2P", 6488.1, 6510.4),
    ("c03", "2P", 4249.9, 4262.4),
    ("c04", "2P", 5229.0, 5250.1),
    ("c05", "2P", 1476.0, 1495.9),
    ("c06", "1P", 5455.5, 5477.6),
    ("c07", "2P", 874.3, 885.6),
    ("c08", "2P", 792.7, 808.7),
    ("c09", "1P", 5570.9, 5585.9),
    ("c10", "1P", 3304.5, 3323.6),
]

# 右辺残差の許容範囲: 連結ボーナス最大10 + 色ボーナス最大12 (4色試合)
RESIDUAL_MAX = 22


def infer_chain_from_bonus(right: int) -> tuple[int, int] | None:
    """右辺 (最終段ボーナス) から連鎖数を逆引きする。(連鎖数, 残差)。"""
    best = None
    for k in range(1, 25):
        cp = chain_power(k)
        resid = right - max(cp, 1 if cp == 0 else cp) if cp > 0 else right - 1
        # 1連鎖目はボーナス0→表示は max(1, 0+連結+色) なので特別扱い
        if k == 1:
            resid = right - 1
        if 0 <= resid <= RESIDUAL_MAX:
            best = (k, resid)
    return best


def main() -> None:
    rows = []
    for line in SCAN.read_text().splitlines():
        p = line.split("\t")
        if len(p) < 5:
            continue
        rows.append((float(p[0]), p[1], int(p[2]), int(p[3])))
    rows.sort(key=lambda r: r[0])

    sessions: dict[str, list[dict]] = {"1P": [], "2P": []}
    accs = {"1P": FormulaStepAccumulator(), "2P": FormulaStepAccumulator()}
    last_steps = {"1P": 0, "2P": 0}
    for t, side, left, right in rows:
        acc = accs[side]
        prev_count = acc.step_count
        res = FormulaReadResult(True, left, right, left * right, 0.9)
        step = acc.update(t, res)
        if acc.step_count < prev_count or (
            acc.step_count == 1 and prev_count > 1
        ):
            pass  # セッション切替は下の steps スナップショットで扱う
        if step is not None:
            cur = sessions[side]
            # 直前セッションの続きか、新セッションか
            if acc.step_count == 1:
                cur.append({"t0": step.t_sec, "t1": step.t_sec,
                            "steps": [], "side": side})
            if not cur:
                cur.append({"t0": step.t_sec, "t1": step.t_sec,
                            "steps": [], "side": side})
            cur[-1]["steps"].append((step.t_sec, step.left, step.right))
            cur[-1]["t1"] = step.t_sec
        last_steps[side] = acc.step_count

    print("==== セッション一覧 (段数>=5 のみ表示) ====")
    all_sessions = sessions["1P"] + sessions["2P"]
    all_sessions.sort(key=lambda s: s["t0"])
    for s in all_sessions:
        n = len(s["steps"])
        if n < 5:
            continue
        power = sum(l * r for _, l, r in s["steps"])
        last_right = s["steps"][-1][2]
        inf = infer_chain_from_bonus(last_right)
        print(
            f"{s['side']} t={s['t0']:.1f}-{s['t1']:.1f} steps={n} "
            f"power={power} last_right={last_right} "
            f"rule_chain={inf[0] if inf else '?'} resid={inf[1] if inf else '?'}"
        )

    print("\n==== 10ケース突合 ====")
    for cid, side, t0, t1 in CASES:
        hits = [
            s for s in sessions[side]
            if s["steps"] and not (s["t1"] < t0 - 6 or s["t0"] > t1 + 1)
        ]
        for s in hits:
            n = len(s["steps"])
            power = sum(l * r for _, l, r in s["steps"])
            last_right = s["steps"][-1][2]
            inf = infer_chain_from_bonus(last_right)
            flag = ""
            if inf and inf[0] != n:
                flag = f"  <-- 段数 {n} != ルール逆引き {inf[0]} (読み落とし疑い)"
            print(
                f"{cid} {side} session t={s['t0']:.1f}-{s['t1']:.1f} "
                f"steps={n} power={power} last_right={last_right} "
                f"rule_chain={inf[0] if inf else '?'}{flag}"
            )
            for st in s["steps"]:
                print(f"    t={st[0]:.1f} {st[1]}x{st[2]}={st[1]*st[2]}")


if __name__ == "__main__":
    main()
