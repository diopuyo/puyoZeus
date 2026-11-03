"""B3段2実行器。--dry-runは合成モデル専用、本物の学習は行わない。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import pandas as pd

from src.b3_eval_adapter import (CERTAIN_SOURCES, EvaluatorTrace, run_false_death_scan,
                                 score_m_time_agree, write_replay_record)
from src.b3_eval_stage2 import evaluate_stage2
from src.b3_stage2_score import (confirm_changes, metric_intervals, probability_intervals,
                               raw_game_scores, read_json, safety_receipt)
from src.b3_stage2_trace import make_display, predict_rows, review_rows
from src.b3_train_model import HgbConfig, feature_columns, fit_two_state, load_model, save_model

R_COMMIT = "ccc9623e84fd013c64f161ecb23805c5b4568032"
PREREG_COMMIT = "a77664731dba31c5312a885310539fc4ad0490f0"
PARTS = {1: [f"c{i}" for i in range(1, 7)] + ["zenchi"], 2: [f"s{i}" for i in range(7)]}
EXPECTED_GAMES = {1: 57, 2: 58}
SMOKE_CONFIG = HgbConfig("stage2_dummy", 3, .2, 3, 5)
SMOKE_SEED = 1003
EXTRA_SAFETY_PARTS = ("review", "q_7gc4TgFig", "fcXG83vInDY", "mia8KCjr52g")


def sha256(path: Path) -> str:
    """入力原票を固定する。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    """NaNをJSONへ混ぜない。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """実行対象はzenchi固定。封印セット用の汎用入口は設けない。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exev-root", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True, help="wt_a1/logs/a1b/combo")
    ap.add_argument("--labels-root", type=Path, required=True, help="wt_evalset/logs/eval_set")
    ap.add_argument("--calibration", type=Path, required=True, help="Rのwinprob_calib.json")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=Path, help="保存済みTwoStateModel")
    group.add_argument("--dry-run", action="store_true", help="合成データだけで仮モデルを作る")
    ap.add_argument("--review-csv", action="store_true")
    return ap.parse_args()


def verify_sources(exev: Path) -> dict:
    """F2で固定した採点器・平滑化のソースを照合する。"""
    catalog = pd.read_csv(Path(__file__).parents[1] / "docs/b3_prereg_fill/r_source.sha256.tsv", sep="\t")
    required = ["scripts/eval_set_score_20261001.py", "scripts/_diag_set2_falsedeath_audit.py",
                "src/exchange_display_smoothing.py"]
    hashes = {}
    for name in required:
        expected = catalog.loc[catalog.path == name, "sha256"].item()
        actual = hashlib.sha256((exev / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError(f"固定ソースと不一致: {name}")
        hashes[name] = actual
    local = Path(__file__).parents[1] / required[-1]
    if hashlib.sha256(local.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != hashes[required[-1]]:
        raise ValueError("ローカル平滑化がRと異なる")
    return hashes


def prepare_model(a: argparse.Namespace) -> Path:
    """仮モデルの学習入力はmake_rowsのみ。評価用zenchiは一切学習しない。"""
    if not a.dry_run:
        return a.model
    from src.b3_train_synth import make_rows
    synth, _ = make_rows(n_videos=3, games_per_video=2, rows_per_game=20, seed=SMOKE_SEED)
    model = fit_two_state(synth, feature_columns(synth.columns), SMOKE_CONFIG, SMOKE_SEED)
    path = a.out / "dummy_model.joblib"
    save_model(model, path)
    return path


def load_reference(a: argparse.Namespace, part: str) -> tuple[dict, list[dict], dict]:
    """A1b+A2の実フラグと完了票を検査し、表示・証明原票だけを読む。"""
    base = a.reference_root / "replay/a1" / part
    if read_json(base / "status.json")["state"] != "completed":
        raise ValueError(f"Rの未完了記録: {part}")
    command = read_json(base / "COMMAND.json")
    if not all(flag in command for flag in ("--completion-score-max-chain", "--a2-safety-review")):
        raise ValueError("RにA1b+A2がない")
    with np.load(base / "display.npz") as data:
        reference = {key: data[key] for key in ("t_sec", "display_p1", "display_adv", "source")}
    events = [json.loads(line) for line in (base / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    hashes = {str(base / name): sha256(base / name) for name in ("display.npz", "events.jsonl", "COMMAND.json")}
    return reference, events, hashes


def write_part(a: argparse.Namespace, part: str, df: pd.DataFrame, games: list[dict], k: float | None) -> dict:
    """原票と同じ時刻でアダプタ入力を作り、確率と死亡証明を別々に保存する。"""
    reference, events, hashes = load_reference(a, part)
    # このパートに表示フレームを持つ試合だけを渡す。
    t = reference["t_sec"]
    covered = [g for g in games if np.any((t >= g["start"]) & (t < g["end"]))]
    display = make_display(df, reference, covered, events, k)
    death = np.isin(reference["source"], CERTAIN_SOURCES)
    np.testing.assert_array_equal(display["source"][death], reference["source"][death])
    np.testing.assert_array_equal(display["display_p1"][death], reference["display_p1"][death])
    # イベントは独立証明だけを転写。旧評価器の確率イベントをB3予測と偽らない。
    proofs = [dict(game_idx=e["game_idx"], values=[v for v in e["values"]
              if v["source"] in CERTAIN_SOURCES]) for e in events]
    dest = a.out / "replay" / part
    receipt = write_replay_record(dest, EvaluatorTrace(t, display["display_p1"]), [], proofs,
                                  display_arrays=display)
    return dict(display=display, reference=reference, input_hashes=hashes, output=receipt)


def join_arrays(parts: list[dict], key: str) -> dict[str, np.ndarray]:
    """重複のない表示パートを連結する。"""
    joined = {name: np.concatenate([p[key][name] for p in parts]) for name in parts[0][key]}
    order = np.argsort(joined["t_sec"])
    joined = {name: value[order] for name, value in joined.items()}
    if np.any(np.diff(joined["t_sec"]) <= 0):
        raise ValueError("Rパートの表示時刻が重複")
    return joined


def attach_review(df: pd.DataFrame, display: dict, reference: dict) -> pd.DataFrame:
    """30Hz表示の直前フレームを行へ結び、レビュー用の死亡状態も付ける。"""
    out = df.copy()
    idx = np.searchsorted(display["t_sec"], out.t_source_sec.to_numpy(float), side="right") - 1
    valid = (idx >= 0) & (idx < len(display["t_sec"]))
    idx = np.clip(idx, 0, len(display["t_sec"]) - 1)
    out["p_display"] = np.where(valid, display["p_display"][idx], np.nan)
    out["p_ref"] = np.where(valid, reference["display_p1"][idx], np.nan)
    # Rの表示確率は保存されていない。advからの逆変換で捏造しない。
    out["p_ref_display"] = np.nan
    for side, losing in (("p1", display["display_p1"][idx] < .5),
                         ("p2", display["display_p1"][idx] > .5)):
        dead = valid & losing & np.isin(display["source"][idx], CERTAIN_SOURCES)
        out[side + "_death_state"] = np.where(dead, "proven_dead", "none")
        out[side + "_death_basis"] = np.where(dead, display["source"][idx], "")
    return out


def inherited_safety_evidence(a: argparse.Namespace) -> dict:
    """115試合と重ならない入力原票も固定し、継承する死亡フレームを検収可能にする。"""
    out = {}
    for name in EXTRA_SAFETY_PARTS:
        reference, events, hashes = load_reference(a, name)
        death = np.isin(reference["source"], CERTAIN_SOURCES)
        dest = a.out / "inherited_safety" / name
        dest.mkdir(parents=True)
        # 外部3動画の勝率をB3予測であるかのように出さず、独立証明だけを保存する。
        np.savez_compressed(dest / "r_death_frames.npz", **{k: v[death] for k, v in reference.items()})
        out[name] = dict(input_hashes=hashes, display_frames=len(death), certain_frames=int(death.sum()),
                         proof_sha256=sha256(dest / "r_death_frames.npz"),
                         method="F10により死亡証明はRと同一。安全回帰のみ継承、B3勝率の評価対象外")
    return out


def evaluate_set(a: argparse.Namespace, number: int, df: pd.DataFrame, k: float | None) -> dict:
    """57/58試合の全数を独立に採点する。"""
    label_path = a.labels_root / ("labels.json" if number == 1 else "set2/labels.json")
    games = read_json(label_path)["games"]
    if len(games) != EXPECTED_GAMES[number] or set(df.zenchi_game) != {g["game"] for g in games}:
        raise ValueError("zenchiの試合母数・識別が不一致")
    for game in games:
        labels = df.loc[df.zenchi_game == game["game"], "won_p1"].dropna().unique()
        if len(labels) > 1 or (len(labels) == 1 and labels[0] != int(game["winner"] == "1P")):
            raise ValueError("行と固定ラベルの勝者不一致")
    parts = [write_part(a, name, df, games, k) for name in PARTS[number]]
    paths = [a.out / "replay" / name for name in PARTS[number]]
    refs = [a.reference_root / "replay/a1" / name for name in PARTS[number]]
    cand = score_m_time_agree(a.exev_root, [p / "display.npz" for p in paths], label_path)
    ref = score_m_time_agree(a.exev_root, [p / "display.npz" for p in refs], label_path)
    scans = [run_false_death_scan(a.exev_root, pp, games) for pp in (refs, paths)]
    display, reference = join_arrays(parts, "display"), join_arrays(parts, "reference")
    raw = raw_game_scores(display, reference, games)
    pred = attach_review(df, display, reference)
    pred.to_parquet(a.out / f"predictions_set{number}.parquet", index=False)
    if a.review_csv:
        from src.b3_review_input import save_review_input
        save_review_input(a.out / f"review_set{number}.csv", review_rows(df, pred))
    return dict(candidate=cand, reference=ref, raw_rows=raw,
                metrics={**metric_intervals(cand["rows"], ref["rows"]), **probability_intervals(raw)},
                scan=scans[1], confirms=confirm_changes(*scans),
                input_hashes={str(label_path): sha256(label_path), **{key: value for part in parts
                              for key, value in part["input_hashes"].items()}},
                replay_receipts=[part["output"] for part in parts])


def finish(a: argparse.Namespace, sets: dict, model_path: Path, source_hashes: dict, started: float) -> None:
    """数値門と未実施のレビュー・正式時間門を分けて受領票へ残す。"""
    report = read_json(a.reference_root / "RESULT.json")
    cand = [r for s in sets.values() for r in s["candidate"]["rows"]]
    ref = [r for s in sets.values() for r in s["reference"]["rows"]]
    raw = [r for s in sets.values() for r in s["raw_rows"]]
    safety = safety_receipt(report, [s["scan"] for s in sets.values()], [s["confirms"] for s in sets.values()])
    safety_evidence = inherited_safety_evidence(a)
    inp = {**metric_intervals(cand, ref), **probability_intervals(raw), **safety}
    # F10: 第1版ではforecast72を実装しない。レビュー/正式時間門は未評価のまま。
    inp["forecast72"] = dict(sole_confirms=0, false_confirms=0, applicable=False, reason="F10対象外")
    gate = evaluate_stage2(inp)
    save_json(a.out / "scores.json", dict(sets=sets, total=inp))
    receipt = dict(dry_run=a.dry_run, interpretation="検証用の仮モデル。数値・合否に意味はない" if a.dry_run
                   else "段2数値評価。レビュー・正式時間門は別途必要", elapsed_sec=time.perf_counter() - started,
                   r_commit=R_COMMIT, prereg_commit=PREREG_COMMIT, model=str(model_path),
                   model_sha256=sha256(model_path), rows_sha256=sha256(a.rows), source_hashes=source_hashes,
                   reference_report=str(a.reference_root / "RESULT.json"),
                   reference_report_sha256=sha256(a.reference_root / "RESULT.json"),
                   calibration_sha256=sha256(a.calibration), population=dict(games=len(raw),
                   frames=sum(r["frames"] for r in raw), null_frames=sum(r["null_frames"] for r in raw)),
                   inherited_safety=safety_evidence,
                   reference_method="A1b+A2 combo原票から既存採点器で115試合再計算。短5/第14/3:00は同一証明のR原票継承",
                   contract="display_p1=死亡証明適用後の平滑前p、display_adv=平滑後adv。LL/BrierはB3の死亡証明適用前p_selected対R原票display_p1（R死亡証明由来確率を含む）",
                   limitations=["レビュー7窓・OBS連続30分の正式時間門は未実施", "Rのp_ref_displayは原票になく欠測"],
                   inputs=inp, gate=gate)
    save_json(a.out / "RECEIPT.json", receipt)
    print(json.dumps(dict(dry_run=a.dry_run, games=len(raw), elapsed_sec=receipt["elapsed_sec"],
                         receipt=str(a.out / "RECEIPT.json")), ensure_ascii=False))


def main() -> None:
    """指定されたモデルを読み、zenchiだけを評価する。"""
    a, started = parse_args(), time.perf_counter()
    if a.rows.name != "b3_zenchi_merged.parquet":
        raise ValueError("この実行器はzenchiの登録済み行専用")
    a.out.mkdir(parents=True, exist_ok=False)
    source_hashes = verify_sources(a.exev_root)
    model_path = prepare_model(a)
    df = predict_rows(pd.read_parquet(a.rows), load_model(model_path))
    if set(df.zenchi_set) != set(PARTS):
        raise ValueError("zenchiセット識別が不一致")
    calibration = read_json(a.calibration)
    k = float(calibration["k"]) if calibration.get("kind") == "logistic_symmetric" else None
    sets = {str(n): evaluate_set(a, n, df[df.zenchi_set == n], k) for n in PARTS}
    finish(a, sets, model_path, source_hashes, started)


if __name__ == "__main__":
    main()
