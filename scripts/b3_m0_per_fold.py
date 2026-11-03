"""B3の事前登録foldでM0を再学習する。採点は行わない。"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from src.b3_m0_inputs import DEFAULT_FOLDS, FOLD_SHA256, INPUT_COLUMNS, checked_record, fold_maps, recover_inputs
from src.b3_m0_training import DEFAULT_ARGS, DEFAULT_SEED, canonical, nested_fold
from src.b3_source import B2_ROOT_DEFAULT, sha256_file
from src.b3_train_data import list_complete_parquets, load_training_rows
from scripts.train_advantage_m1_causal_ledger_v1 import CanonicalSamples

CPU_THREADS = 2
GPU_FRACTION = 0.85


def load_inputs(paths: dict[str, Path], b2_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """動画単位で記録を復元し、入力資産ハッシュを保存する。"""
    frames, boards, queues, sources = [], [], [], {}
    for video, path in sorted(paths.items()):
        frame = pd.read_parquet(path, columns=list(INPUT_COLUMNS))
        if not (frame.video_id == video).all():
            raise ValueError("動画名と行のvideo_idが不一致")
        record = checked_record(path, b2_root)
        board, queue = recover_inputs(frame, record)
        frames.append(frame)
        boards.append(board)
        queues.append(queue)
        sources[video] = {"rows_sha256": sha256_file(path),
                          "manifest_sha256": sha256_file(path.with_suffix(".manifest.json"))}
        print(f"入力復元: {video} rows={len(frame)}", flush=True)
    return pd.concat(frames, ignore_index=True), np.concatenate(boards), np.concatenate(queues), sources


def check_rcv(paths: dict[str, Path], logits: pd.DataFrame, fold: int, seed: int) -> dict:
    """静止G_feまでの接続を確認する。成績を計算・保存しない。"""
    from src.b3_eval_cv import TrainContext
    from src.b3_train_rcv import _m0_attach, rcv_trainer
    df = _m0_attach(load_training_rows(paths, []), logits.set_index("row_key"))
    groups, _ = fold_maps()
    held = df.m0_role == "outer_test"
    started = time.monotonic()
    predict, info = rcv_trainer(m0_column="m0_logit")(df[~held], TrainContext(fold, seed, groups, {}, []))
    values = predict(df[held])
    if not np.isfinite(values).all() or info["m0_scope"] != "static_only":
        raise ValueError("M0→R_cvの動作確認に失敗")
    return {"seconds": time.monotonic() - started, "predicted_rows": len(values), "all_finite": True}


def run_fold(args: argparse.Namespace, df: pd.DataFrame, samples: CanonicalSamples, fold_rows: np.ndarray,
             fold: int, paths: dict[str, Path], sources: dict, device: torch.device) -> dict:
    """fold単位でlogit・checkpoint・完了票を公開する。"""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    values, records = nested_fold(samples, fold_rows, fold, args.out, args, device, args.seed, args.smoke)
    logits = df[["row_key", "video_id", "game_idx"]].copy()
    logits["m0_logit"], logits["m0_outer_fold"] = values, fold
    logits["m0_role"] = np.where(fold_rows == fold, "outer_test", "inner_train_oof")
    path = args.out / f"fold{fold}.parquet"
    logits.to_parquet(path, index=False)
    logits[logits.m0_role == "outer_test"].to_parquet(args.out / f"oof_fold{fold}.parquet", index=False)
    result = dict(fold=fold, fold_sha256=FOLD_SHA256, seed=args.seed, smoke=args.smoke,
                  hyperparameters={key: getattr(args, key) for key in DEFAULT_ARGS}, records=records,
                  seconds=time.monotonic() - started, rows=len(df), sources=sources,
                  logits_sha256=sha256_file(path), torch_version=torch.__version__, device=str(device),
                  gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0)
    if args.check_rcv:
        result["rcv_check"] = check_rcv(paths, logits, fold, args.seed)
    result["checkpoint_hashes"] = {record["checkpoint"]: sha256_file(args.out / record["checkpoint"])
                                    for record in records}
    result["code_sha256"] = code_hashes()
    (args.out / f"fold{fold}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"fold={fold} 完了 seconds={result['seconds']:.1f}", flush=True)
    return result


def code_hashes() -> dict[str, str]:
    """再現に必要な実行器・モデル・既存レシピを記録する。"""
    root = Path(__file__).resolve().parents[1]
    names = ("scripts/b3_m0_per_fold.py", "src/b3_m0_inputs.py", "src/b3_m0_training.py",
             "src/advantage_m0_current_cnn_v1.py", "scripts/train_advantage_m1_causal_ledger_v1.py",
             "scripts/train_advantage_m0_current_cnn_v1.py", "src/b3_train_rcv.py")
    return {name: sha256_file(root / name) for name in names}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-dir", type=Path, required=True)
    parser.add_argument("--b2-root", type=Path, default=B2_ROOT_DEFAULT)
    parser.add_argument("--fold-table", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", help="試走限定。通常は正本の全動画を必須にする")
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--smoke", action="store_true", help="動作確認用。少数groupの内側tuneを試合分離にする")
    parser.add_argument("--check-rcv", action="store_true", help="R_cvの予測まで確認（採点なし）")
    for key, value in DEFAULT_ARGS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    if args.videos and not args.smoke:
        parser.error("動画の部分指定には --smoke が必要")
    if any(getattr(args, key) <= 0 for key in DEFAULT_ARGS):
        parser.error("ハイパーパラメータは正値が必要")
    if not args.smoke and any(getattr(args, key) != value for key, value in DEFAULT_ARGS.items()):
        parser.error("通常実行は本番の固定ハイパーパラメータを使用する")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.monotonic()
    groups, folds = fold_maps(args.fold_table)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAが利用できない。GPU対応torch環境で実行してください")
        torch.cuda.set_per_process_memory_fraction(GPU_FRACTION, device)
    torch.set_num_threads(CPU_THREADS)
    paths = list_complete_parquets(args.rows_dir, args.videos or sorted(groups))
    if set(paths) - groups.keys():
        raise ValueError("事前登録に無い動画")
    args.out.mkdir(parents=True, exist_ok=False)
    df, boards, queues, sources = load_inputs(paths, args.b2_root)
    samples = canonical(df, boards, queues, groups)
    fold_rows = df.video_id.map(folds).to_numpy(dtype=int)
    preparation_seconds = time.monotonic() - started
    results = [run_fold(args, df, samples, fold_rows, fold, paths, sources, device) for fold in args.folds]
    manifest = dict(preparation_seconds=preparation_seconds, total_seconds=time.monotonic() - started,
                    folds=args.folds, rows=len(df), videos=len(paths), smoke=args.smoke,
                    fold_seconds=[result["seconds"] for result in results])
    (args.out / "COMPLETE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
