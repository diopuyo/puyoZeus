"""E1: 固定15動画foldを再現し、全採用行でイベント評価モデルを保存する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.exchange_event_evaluator import MODEL_COLUMNS, MODEL_VERSION, file_sha256
from src.exchange_event_features import D_COLUMNS, PHASE_BOUNDS, g_features, score_features

ROOT = Path(__file__).resolve().parents[1]
SHARED = Path("/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer")
REFERENCE = Path("/mnt/c/Users/ryouj/.codex/worktrees/56dc/puyo_analyzer")
RUN_NAMES = dict(base="r2_baselines_148_20260926", g="r2_causal_phase_fe_148_20260926",
                 s1="r2_exchange_timeline_148_20260926", s3="r2_exchange_timeline_s3_148_20260926",
                 m0="r2_grouped_cv_148_20260925")
SEEDS, FOLDS, MAX_WORKERS = range(3), range(5), 3
LR_PARAMS = dict(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-4, random_state=0)
LIGHT_PARAMS = dict(max_iter=80, max_depth=8, max_leaf_nodes=31)
TARGETS = {"G_fe_overall": .6514, "G_fe_middle": .5608, "S1": .6383, "S3_reference": .7084}
TOLERANCE, EXPECTED_VIDEOS, MIDDLE = .005, 148, 1
# T03/T09の既存採用行は148原票中144動画。除外行を復活させない。
EXPECTED_LABELED_VIDEOS = 144
ALLOWED_TIERS = ("A級", "マスター", "チャレンジャー", "S級")


def save_json(path: Path, value: Any) -> None:
    """完了済みJSONだけを読めるよう原子的に保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_path(reference: Path, kind: str) -> Path:
    """外部資産は常に読取り専用の入力パスとして扱う。"""
    return reference / "logs" / RUN_NAMES[kind]


def validate_rows(reference: Path, tier_path: Path) -> dict[str, Any]:
    """148動画のタイトル・ティアを確認し、未確認動画を黙って学習しない。"""
    rows = pd.read_csv(run_path(reference, "base") / "rows.csv")
    tiers = pd.read_csv(tier_path, sep="\t").set_index("video_name")
    videos = sorted(rows.video_id.unique())
    invalid = [v for v in videos if v not in tiers.index or
               not any(t in str(tiers.loc[v, "tier"]) for t in ALLOWED_TIERS)]
    if invalid or len(videos) != EXPECTED_LABELED_VIDEOS:
        raise ValueError(f"148動画のティア確認に失敗: {invalid}, count={len(videos)}")
    if not np.array_equal(rows.row_id, np.arange(len(rows))):
        raise ValueError("D行順が連番と一致しない")
    columns = json.loads((run_path(reference, "base") / "columns.json").read_text())
    if tuple(columns) != D_COLUMNS:
        raise ValueError("D列順が正式特徴定義と一致しない")
    event_rows = pd.read_csv(run_path(reference, "s1") / "rows.csv")
    s3_rows = pd.read_csv(run_path(reference, "s3") / "rows.csv")
    if not event_rows.equals(s3_rows):
        raise ValueError("S1/S3の行対応が不一致")
    return dict(videos=videos, rows=len(rows), event_rows=len(event_rows),
                event_videos=int(event_rows.video_id.nunique()),
                tier_titles={v: str(tiers.loc[v, "title"]) for v in videos})


def score_video(reference: Path, raw_root: Path, video: str) -> tuple[np.ndarray, np.ndarray]:
    """保存済み区間の得点を読み、正式なマージン対応関数で再計算する。"""
    with np.load(raw_root / (video.removeprefix("video_") + ".npz")) as raw:
        with np.load(run_path(reference, "s3") / "features" / (video + ".npz")) as ids:
            before_ids, after_ids, row_ids = ids["pre_ids"], ids["post_ids"], ids["row_id"]
        with np.load(run_path(reference, "s1") / "features" / (video + ".npz")) as pre:
            triggers = pre["trigger"]
            np.testing.assert_array_equal(pre["row_id"], row_ids)
        rows = pd.read_csv(run_path(reference, "s1") / "rows.csv", usecols=["row_id", "sign"])
        signs = rows.set_index("row_id").loc[row_ids, "sign"].to_numpy()
        scores, times, games = raw["score"], raw["t_sec"], raw["game_idx"]
        starts = {g: times[games == g].min() for g in np.unique(games)}
        values, cache = [], {}
        for before, after, trigger, sign in zip(before_ids, after_ids, triggers, signs):
            key = (*before, *after, float(trigger), float(sign))
            if key not in cache:
                pre = np.array([scores[i] if i >= 0 else np.nan for i in before])
                elapsed = float(trigger - starts[games[after[0]]])
                cache[key] = score_features(pre, scores[after], elapsed, int(sign < 0))
            values.append(cache[key])
    return row_ids, np.asarray(values, dtype=np.float32)


def prepare_scores(reference: Path, raw_root: Path, output: Path) -> None:
    """T11行集合を維持して、マージン対応S3行列を作る。"""
    rows = pd.read_csv(run_path(reference, "s1") / "rows.csv")
    chunks = [score_video(reference, raw_root, video) for video in sorted(rows.video_id.unique())]
    ids = np.concatenate([part[0] for part in chunks])
    order = np.argsort(ids)
    np.testing.assert_array_equal(ids[order], rows.row_id)
    scores = np.concatenate([part[1] for part in chunks])[order]
    np.save(output / "score_features_margin.npy", scores)


def datasets(reference: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    """整列済みDと因果位相を読み込む。外部検証スクリプトはimportしない。"""
    rows = pd.read_csv(run_path(reference, "base") / "rows.csv")
    design = np.load(run_path(reference, "base") / "design.npy", mmap_mode="r")
    with np.load(run_path(reference, "g") / "phase_inputs.npz") as saved:
        phases = {name: saved[name] for name in saved.files}
    return rows, design, phases


def fit_g_fold(reference: Path, seed: int, fold: int) -> pd.DataFrame:
    """T03の内側OOFと外側予測を区別し、T09を再学習する。"""
    rows, design, phases = datasets(reference)
    root = run_path(reference, "m0") / f"seed_{seed}/fold_{fold}"
    outer = pd.read_csv(root / "predictions.csv")
    test = outer.row_id.to_numpy()
    train = np.flatnonzero(~rows.video_id.isin(outer.video_id.unique()))
    oof = np.full(len(rows), np.nan)
    for inner in FOLDS:
        with np.load(root / f"inner_{inner}_m0.npz") as saved:
            assert not np.isfinite(oof[saved["indices"]]).any()
            oof[saved["indices"]] = saved["probability"]
    assert np.isfinite(oof[train]).all() and np.isnan(oof[test]).all()
    thresholds = np.quantile(phases["elapsed"][train], PHASE_BOUNDS)
    phase = np.column_stack((phases["fill"], np.searchsorted(thresholds, phases["elapsed"], side="left")))
    sign, labels = rows.sign.to_numpy(), rows.label.to_numpy()
    x = g_features(design[train], oof[train], sign[train], phase[train])
    model = LogisticRegression(**LR_PARAMS).fit(x, np.where(sign[train] > 0, labels[train], 1-labels[train]))
    del x
    x = g_features(design[test], outer.A.to_numpy(), sign[test], phase[test])
    p = model.predict_proba(x)[:, 1]
    result = rows.iloc[test][["row_id", "video_id", "label", "phase"]].copy()
    result["G_fe"] = np.where(sign[test] > 0, p, 1-p)
    return result


def event_designs(reference: Path, output: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """再現用固定70版と本番候補マージン版を明示的に分ける。"""
    rows = pd.read_csv(run_path(reference, "s1") / "rows.csv")
    original = np.load(run_path(reference, "base") / "design.npy", mmap_mode="r")
    arrival = np.load(run_path(reference, "s1") / "s1_features.npy")
    s1 = np.column_stack((original[rows.row_id.to_numpy()], arrival))
    scores = np.load(output / "score_features_margin.npy")
    reference_scores = np.load(run_path(reference, "s3") / "score_features.npy")
    return rows, dict(S1=s1, S3=np.column_stack((s1, scores)),
                      S3_reference=np.column_stack((s1, reference_scores)))


def fit_event_fold(reference: Path, output: Path, seed: int, fold: int) -> pd.DataFrame:
    """同じ動画holdoutで通常版と事前固定の軽量版を比較する。"""
    rows, designs = event_designs(reference, output)
    ref = pd.read_csv(run_path(reference, "s1") / f"seed_{seed}/fold_{fold}/predictions.csv")
    test = rows.video_id.isin(ref.video_id.unique()).to_numpy()
    sign, labels = rows.sign.to_numpy(), rows.label.to_numpy()
    target = np.where(sign > 0, labels, 1-labels)
    result = rows.loc[test, ["row_id", "video_id", "label", "phase"]].copy()
    for name in ("S1", "S3_reference", "S3", "S1_light", "S3_light"):
        params = LIGHT_PARAMS if name.endswith("_light") else {}
        x = designs[name.removesuffix("_light")]
        model = HistGradientBoostingClassifier(random_state=0, **params).fit(x[~test], target[~test])
        p = model.predict_proba(x[test])[:, 1]
        result[name] = np.where(sign[test] > 0, p, 1-p)
    return result


def fold_job(job: tuple[str, str, int, int]) -> str:
    """一つのfoldを低優先度で保存する。既存入力への書込みは禁止。"""
    reference_text, output_text, seed, fold = job
    reference, output = Path(reference_text), Path(output_text)
    os.nice(max(0, 19 - os.nice(0)))
    destination = output / f"seed_{seed}/fold_{fold}"
    destination.mkdir(parents=True, exist_ok=True)
    for name, fit in (("g", fit_g_fold), ("events", fit_event_fold)):
        path = destination / (name + ".csv")
        if not path.exists():
            frame = fit(reference, seed, fold) if name == "g" else fit(reference, output, seed, fold)
            frame.to_csv(path.with_suffix(".tmp"), index=False)
            path.with_suffix(".tmp").replace(path)
        print(f"完了: seed={seed} fold={fold} {name}", flush=True)
    return str(destination)


def cv_report(output: Path) -> dict[str, Any]:
    """15foldのOOFをまとめ、丸め前のAUCで合否を判定する。"""
    frames = {kind: pd.concat([pd.read_csv(output / f"seed_{s}/fold_{f}/{kind}.csv")
              for s in SEEDS for f in FOLDS], ignore_index=True) for kind in ("g", "events")}
    g, events = frames["g"], frames["events"]
    middle = g[g.phase == MIDDLE]
    auc = dict(G_fe_overall=roc_auc_score(g.label, g.G_fe),
               G_fe_middle=roc_auc_score(middle.label, middle.G_fe))
    for name in ("S1", "S3_reference", "S3", "S1_light", "S3_light"):
        auc[name] = roc_auc_score(events.label, events[name])
    delta = {name: auc[name] - target for name, target in TARGETS.items()}
    light = {name: auc[name + "_light"] - auc[name] for name in ("S1", "S3")}
    return dict(auc=auc, target_differences=delta, lightweight_differences=light,
                reproduction_pass=all(abs(v) <= TOLERANCE for v in delta.values()),
                lightweight_pass=all(v >= -TOLERANCE for v in light.values()),
                folds=len(SEEDS)*len(FOLDS), g_predictions=len(g), event_predictions=len(events))


def save_model(directory: Path, name: str, model: Any) -> dict[str, Any]:
    """分類器・列順・ハッシュを保存し、LRは人が読める係数も出す。"""
    path = directory / (name + ".joblib")
    joblib.dump(model, path)
    columns = MODEL_COLUMNS[name.removesuffix("_light")]
    if hasattr(model, "coef_"):
        save_json(directory / (name + ".coefficients.json"), dict(columns=columns,
                  coef=model.coef_.tolist(), intercept=model.intercept_.tolist(), classes=model.classes_.tolist()))
    return dict(file=path.name, sha256=file_sha256(path), columns=columns, parameters=model.get_params())


def final_fit(reference: Path, output: Path, directory: Path) -> dict[str, Any]:
    """Gはseed0の外側OOF、S1/S3は採用行全体で最終学習する。"""
    directory.mkdir(parents=True, exist_ok=True)
    rows, design, phases = datasets(reference)
    thresholds = np.quantile(phases["elapsed"], PHASE_BOUNDS)
    phase = np.column_stack((phases["fill"], np.searchsorted(thresholds, phases["elapsed"], side="left")))
    sign, label = rows.sign.to_numpy(), rows.label.to_numpy()
    x = g_features(design, rows.A.to_numpy(), sign, phase)
    model = LogisticRegression(**LR_PARAMS).fit(x, np.where(sign > 0, label, 1-label))
    models = {"G_fe": save_model(directory, "G_fe", model)}
    del x, model, design
    events, designs = event_designs(reference, output)
    y = np.where(events.sign > 0, events.label, 1-events.label)
    for name in ("S1", "S3", "S1_light", "S3_light"):
        params = LIGHT_PARAMS if name.endswith("_light") else {}
        model = HistGradientBoostingClassifier(random_state=0, **params).fit(designs[name.removesuffix("_light")], y)
        models[name] = save_model(directory, name, model)
        print(f"最終学習: {name}", flush=True)
    return dict(version=MODEL_VERSION, elapsed_thresholds=thresholds.tolist(), models=models,
                sklearn_version=sklearn.__version__, numpy_version=np.__version__, python_version=platform.python_version(),
                m0_training="T03 seed0 outer OOF (rows.A); M0推論モデルは呼出側が供給",
                orientation="source-side D and target; output converted to 1P; no blanket negation",
                score_conversion="発火時刻−試合内最初の保存時刻; compute_effective_rate; fractional delta/rate",
                s1_definition="T11再現: D + 発火前特徴35列（M0/位相なし）")


def input_hashes(reference: Path, raw_root: Path, tier_path: Path) -> dict[str, str]:
    """学習行列・行キー・位相・M0 OOF・原票・ティアの内容を固定する。"""
    paths = [tier_path]
    for kind, names in (("base", ("rows.csv", "columns.json", "design.npy")),
                        ("g", ("phase_inputs.npz",)), ("s1", ("rows.csv", "s1_features.npy")),
                        ("s3", ("rows.csv", "score_features.npy"))):
        paths.extend(run_path(reference, kind) / name for name in names)
    paths.extend(run_path(reference, "m0").glob("seed_*/fold_*/*m0.npz"))
    for kind in ("s1", "s3"):
        paths.extend(run_path(reference, kind).glob("features/*.npz"))
    paths.extend(raw_root.glob("*.npz"))
    return {str(path): file_sha256(path) for path in sorted(set(paths))}


def execute(args: argparse.Namespace) -> None:
    """特徴準備→15fold→最終学習の順に実行し、採用判断は行わない。"""
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = validate_rows(args.reference, args.tier_index)
    raw_videos = sorted("video_" + path.stem for path in args.raw_root.glob("*.npz"))
    if len(raw_videos) != EXPECTED_VIDEOS or not set(counts["videos"]).issubset(raw_videos):
        raise ValueError("148動画原票と採用動画集合が不一致")
    counts.update(raw_videos=raw_videos, excluded_by_reference=sorted(set(raw_videos)-set(counts["videos"])))
    hashes = input_hashes(args.reference, args.raw_root, args.tier_index)
    protocol = dict(counts=counts, input_sha256=hashes, light_parameters=LIGHT_PARAMS, targets=TARGETS)
    previous = args.output / "PROTOCOL.json"
    if previous.exists() and json.loads(previous.read_text()) != protocol:
        raise ValueError("再開元の入力または設定が変わっているため別出力先が必要")
    save_json(previous, protocol)
    print(f"入力確認: {counts['rows']}行 / {len(counts['videos'])}動画", flush=True)
    if not (args.output / "score_features_margin.npy").exists():
        prepare_scores(args.reference, args.raw_root, args.output)
    jobs = [(str(args.reference), str(args.output), s, f) for s in SEEDS for f in FOLDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fold_job, jobs))
    report = cv_report(args.output)
    save_json(args.output / "CV_METRICS.json", report)
    manifest = final_fit(args.reference, args.output, args.models)
    manifest.update(training_data=protocol, validation=report, elapsed_seconds=time.monotonic()-started)
    save_json(args.models / "manifest.json", manifest)
    save_json(args.output / "COMPLETE.json", dict(seconds=time.monotonic()-started, validation=report))
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main() -> None:
    """CPU限定・nice19の実行引数を受け取る。"""
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--raw-root", type=Path, default=SHARED / "data/indicators_v2/boards_lean_phase_l_2026-08-11")
    parser.add_argument("--tier-index", type=Path, default=SHARED / "data/video_tier_index_2026-08-07.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/exchange_event_v1")
    parser.add_argument("--models", type=Path, default=ROOT / "models/exchange_event_v1")
    parser.add_argument("--workers", type=int, choices=range(1, MAX_WORKERS+1), default=MAX_WORKERS)
    args = parser.parse_args()
    os.nice(max(0, 19-os.nice(0)))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)


if __name__ == "__main__":
    main()
