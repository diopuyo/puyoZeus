"""再判定の門 (exev DECISIONS 2026-10-01 事前登録 (再判定)) をそのまま適用する。

構成: 旧 rs0 = 現本番の再生 R0 (rs0 の S′・G_fe が v3・v1 と予測/係数一致、REUSE_CHECK.json)、
旧 rs1..4 = J_orig_rs<k>、補正 rs0..4 = J_T_rs<k> (整列ON)。採点は score_<構成>.txt (report_e36b)。
CV: S1′_light の LL (同じ行 253,335 = 3分割seed×84,445) の差 補正−旧 を random_state ごとに出す。
出力: logs/next_shift_rejudge/GATE.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

TRAIN = Path("logs/next_shift_train")
REJUDGE = Path("logs/next_shift_rejudge")
SEEDS = range(5)
Q_MAX_DIFF, SCENE_MAX = .002, 2766.0
CONFIG = {("orig", 0): "R0", **{("orig", k): f"J_orig_rs{k}" for k in range(1, 5)},
          **{("T", k): f"J_T_rs{k}" for k in SEEDS}}


def score(name: str) -> dict:
    """score_<構成>.txt の最後の JSON 行。"""
    lines = [line for line in (TRAIN / f"score_{name}.txt").read_text().splitlines() if line.startswith("{\"q\"")]
    return json.loads(lines[-1])


def s1_ll(kind: str, seed: int) -> pd.DataFrame:
    """S1′_light の CV 予測 (seed0 は初回の s1cv_*、それ以外は再判定の s1cv_*_rs<k>)。"""
    root = TRAIN / f"s1cv_{kind}" if seed == 0 else REJUDGE / f"s1cv_{kind}_rs{seed}"
    files = sorted(root.glob("seed_*/fold_*/predictions.csv"))
    assert len(files) == 15, (root, len(files))
    return pd.concat([pd.read_csv(p, usecols=["seed", "row_id", "label", "S1_prime_light"]) for p in files])


def cv_diff(seed: int) -> dict:
    """同じ行で LL(補正) − LL(旧)。"""
    merged = s1_ll("orig", seed).merge(s1_ll("T", seed), on=["seed", "row_id"], suffixes=("_o", "_t"),
                                       validate="one_to_one")
    y = merged.label_o.to_numpy()
    old, new = log_loss(y, merged.S1_prime_light_o), log_loss(y, merged.S1_prime_light_t)
    return dict(rows=len(merged), ll_orig=old, ll_T=new, diff=new - old)


def main() -> None:
    """4条件を判定して保存する。"""
    scores = {key: score(name) for key, name in CONFIG.items()}
    q = {kind: [scores[(kind, k)]["q"]["log_loss"] for k in SEEDS] for kind in ("orig", "T")}
    z = {kind: [scores[(kind, k)]["zenchi"]["hits"] for k in SEEDS] for kind in ("orig", "T")}
    cv = [cv_diff(k) for k in SEEDS]
    safety = {CONFIG[key]: dict(false=s["deaths"]["false"], total=s["deaths"]["total"],
              scene=s["scene_first_sec"], game14=s["game14_false_times"]) for key, s in scores.items()}
    gates = dict(
        q=float(np.mean(q["T"]) - np.mean(q["orig"])) <= Q_MAX_DIFF,
        zenchi=bool(np.mean(z["T"]) > np.mean(z["orig"]) and sum(t > o for t, o in zip(z["T"], z["orig"])) >= 4),
        cv=float(np.mean([c["diff"] for c in cv])) <= 0,
        safety=all(v["false"] == 0 and v["scene"] is not None and v["scene"] <= SCENE_MAX and not v["game14"]
                   for v in safety.values()))
    result = dict(q=q, q_mean_diff=float(np.mean(q["T"]) - np.mean(q["orig"])), zenchi=z,
                  zenchi_pairs_T_better=sum(t > o for t, o in zip(z["T"], z["orig"])),
                  cv_s1_light=cv, cv_mean_diff=float(np.mean([c["diff"] for c in cv])),
                  safety=safety, gates=gates, passed=all(gates.values()))
    (REJUDGE / "GATE.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("q_mean_diff", "zenchi", "zenchi_pairs_T_better",
                                            "cv_mean_diff", "gates", "passed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
