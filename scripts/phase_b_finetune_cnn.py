"""Phase B-7: CNN を menu_truth hard negative で fine-tune.

入力:
    - data/training_phase_u/manual_labels.npz (既存 manual GT、主データ)
    - data/training/menu_truth/v??_menu.npz (新規 hard negative、メニュー画面誤認識)

データ統合:
    - patches/labels を concat
    - クラス均衡化 (max_per_class) で EMPTY が大多数になるのを防ぐ
    - holdout 10% で精度測定

初期重み: models/cnn_phase_u_v16.pt
出力: models/cnn_phase_b_v1.pt

新方針 (project_recognition_strategy_pivot) の B-7 実装。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# CPU 強制 (GPU を学習で使うと他作業に影響、保守的選択)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from src.console_init import init_console  # noqa: E402

init_console()

import numpy as np  # noqa: E402

from src.patch_classifier import CnnPatchClassifier, PatchSample  # noqa: E402

DEFAULT_PHASE_U: Path = (
    _ROOT / "data" / "training_phase_u" / "manual_labels.npz"
)
DEFAULT_MENU_DIR: Path = _ROOT / "data" / "training" / "menu_truth"
DEFAULT_DRIFT_DIR: Path = _ROOT / "data" / "training" / "drift_truth"
DEFAULT_INIT_MODEL: Path = _ROOT / "models" / "cnn_phase_u_v16.pt"
DEFAULT_OUT_MODEL: Path = _ROOT / "models" / "cnn_phase_b_v1.pt"


def load_phase_u(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["patches"], d["labels"].astype(np.int32)


def load_menu_truth(menu_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if not menu_dir.is_dir():
        return np.empty((0, 16, 16, 3), dtype=np.uint8), np.empty(0, dtype=np.int32)
    patches_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for npz_path in sorted(menu_dir.glob("v*_menu.npz")):
        d = np.load(npz_path)
        patches_list.append(d["images"])
        labels_list.append(d["labels"].astype(np.int32))
    if not patches_list:
        return np.empty((0, 16, 16, 3), dtype=np.uint8), np.empty(0, dtype=np.int32)
    return np.concatenate(patches_list, axis=0), np.concatenate(labels_list)


def load_drift_truth(
    drift_dir: Path, empty_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """drift_truth dataset をロード.

    Args:
        empty_only: True なら真値=EMPTY のみ採用 (= menu_truth と同類の
            「CNN が puyo と誤認した cell」だけ集める安全策)。
            False なら全採用するが、真値が古い盤面情報の可能性ありで
            訓練データ汚染リスクが高い。
    """
    if not drift_dir.is_dir():
        return np.empty((0, 16, 16, 3), dtype=np.uint8), np.empty(0, dtype=np.int32)
    patches_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for npz_path in sorted(drift_dir.glob("v*_drift.npz")):
        d = np.load(npz_path)
        imgs = d["images"]
        lbls = d["labels"].astype(np.int32)
        if empty_only:
            mask = (lbls == 0)
            imgs = imgs[mask]
            lbls = lbls[mask]
        patches_list.append(imgs)
        labels_list.append(lbls)
    if not patches_list:
        return np.empty((0, 16, 16, 3), dtype=np.uint8), np.empty(0, dtype=np.int32)
    return np.concatenate(patches_list, axis=0), np.concatenate(labels_list)


def balance_classes(
    patches: np.ndarray, labels: np.ndarray,
    max_per_class: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_per_class <= 0:
        return patches, labels
    keep: list[int] = []
    for c in np.unique(labels):
        idxs = np.where(labels == c)[0]
        if len(idxs) > max_per_class:
            idxs = rng.choice(idxs, size=max_per_class, replace=False)
        keep.extend(idxs.tolist())
    keep_arr = np.array(sorted(keep))
    return patches[keep_arr], labels[keep_arr]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase-u", type=Path, default=DEFAULT_PHASE_U,
        help="既存 phase_u manual_labels.npz",
    )
    parser.add_argument(
        "--menu-dir", type=Path, default=DEFAULT_MENU_DIR,
        help="menu_truth 集合ディレクトリ",
    )
    parser.add_argument(
        "--drift-dir", type=Path, default=DEFAULT_DRIFT_DIR,
        help="drift_truth 集合ディレクトリ (空ならスキップ)",
    )
    parser.add_argument(
        "--init-model", type=Path, default=DEFAULT_INIT_MODEL,
    )
    parser.add_argument(
        "--out-model", type=Path, default=DEFAULT_OUT_MODEL,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)  # fine-tune 用に低め
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-per-class", type=int, default=600,
    )
    parser.add_argument("--holdout-ratio", type=float, default=0.10)
    parser.add_argument(
        "--menu-sample-cap", type=int, default=2000,
        help="menu_truth EMPTY サンプル数上限 (上限超なら subsample)。"
             " EMPTY だけ大量に増えて class imbalance 悪化するのを防ぐ。",
    )
    parser.add_argument(
        "--drift-sample-cap", type=int, default=4000,
        help="drift_truth サンプル数上限 (class 別に balance はここで取らない)",
    )
    parser.add_argument(
        "--drift-include-puyo", action="store_true",
        help="drift_truth で真値=puyo (色) のサンプルも採用。"
             " 真値が古い盤面の可能性があるため default で False (EMPTY のみ)。",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    # 1. phase_u 既存データ
    phase_u_patches, phase_u_labels = load_phase_u(args.phase_u)
    print(
        f"[phase_u] patches={phase_u_patches.shape} "
        f"labels.dist={dict(zip(*np.unique(phase_u_labels, return_counts=True)))}"
    )

    # 2. menu_truth 追加データ (label は全部 0 = EMPTY)
    menu_patches, menu_labels = load_menu_truth(args.menu_dir)
    print(
        f"[menu] patches={menu_patches.shape} "
        f"labels.dist={dict(zip(*np.unique(menu_labels, return_counts=True))) if len(menu_labels) else 'empty'}"
    )

    # 3. menu_truth subsample (EMPTY 過剰防止)
    if len(menu_patches) > args.menu_sample_cap:
        idx = rng.choice(
            len(menu_patches), size=args.menu_sample_cap, replace=False,
        )
        menu_patches = menu_patches[idx]
        menu_labels = menu_labels[idx]
        print(f"[menu] subsampled to {len(menu_patches)}")

    # 3b. drift_truth 追加データ (default: 真値=EMPTY のみ、汚染回避)
    drift_patches, drift_labels = load_drift_truth(
        args.drift_dir, empty_only=not args.drift_include_puyo,
    )
    print(
        f"[drift] (empty_only={not args.drift_include_puyo}) "
        f"patches={drift_patches.shape} "
        f"labels.dist={dict(zip(*np.unique(drift_labels, return_counts=True))) if len(drift_labels) else 'empty'}"
    )
    if len(drift_patches) > args.drift_sample_cap:
        idx = rng.choice(
            len(drift_patches), size=args.drift_sample_cap, replace=False,
        )
        drift_patches = drift_patches[idx]
        drift_labels = drift_labels[idx]
        print(f"[drift] subsampled to {len(drift_patches)}")

    # 4. concat
    arrays_p = [phase_u_patches]
    arrays_l = [phase_u_labels]
    if len(menu_patches) > 0:
        arrays_p.append(menu_patches)
        arrays_l.append(menu_labels)
    if len(drift_patches) > 0:
        arrays_p.append(drift_patches)
        arrays_l.append(drift_labels)
    patches = np.concatenate(arrays_p)
    labels = np.concatenate(arrays_l)
    print(f"[merged] total {len(patches)}")

    # 5. class balance
    patches, labels = balance_classes(
        patches, labels, args.max_per_class, rng,
    )
    print(
        f"[balanced] {len(patches)} "
        f"dist={dict(zip(*np.unique(labels, return_counts=True)))}"
    )

    # 6. holdout split
    n = len(patches)
    perm = rng.permutation(n)
    n_holdout = int(n * args.holdout_ratio)
    test_idx = perm[:n_holdout]
    train_idx = perm[n_holdout:]
    train_patches = patches[train_idx]
    train_labels = labels[train_idx]
    test_patches = patches[test_idx]
    test_labels = labels[test_idx]
    print(f"[split] train={len(train_patches)} holdout={len(test_patches)}")

    # 7. classifier 初期化
    classifier = CnnPatchClassifier()
    if args.init_model and args.init_model.exists():
        try:
            import torch
            state = torch.load(
                str(args.init_model), map_location="cpu", weights_only=True,
            )
            classifier._model.load_state_dict(state)
            print(f"[init] loaded {args.init_model.name}")
        except Exception as e:
            print(f"[init] failed ({e}), random initialization")
    else:
        print("[init] random initialization")

    # 8. 訓練
    samples = [
        PatchSample(patch=train_patches[i], color=int(train_labels[i]))
        for i in range(len(train_patches))
    ]
    print(
        f"[train] epochs={args.epochs} lr={args.lr} batch={args.batch_size}"
    )
    losses = classifier.fit(
        samples, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, class_weighted=True,
    )
    print(f"[train] final_loss={losses[-1]:.4f}")

    # 9. holdout 評価 (全体 + menu部分)
    correct_total = 0
    correct_menu = 0
    n_menu_in_holdout = 0
    classifier._model.eval()
    for i in range(len(test_patches)):
        pred = classifier.classify(test_patches[i])
        if pred == int(test_labels[i]):
            correct_total += 1
        # ラベルが 0 (= EMPTY) かつ 元 phase_u になかった候補は menu の hard negative
        # 厳密には判別しづらいが、EMPTY 全体の精度を見るだけで十分
        if int(test_labels[i]) == 0:
            n_menu_in_holdout += 1
            if pred == 0:
                correct_menu += 1
    holdout_acc = correct_total / max(1, len(test_patches))
    empty_acc = (
        correct_menu / n_menu_in_holdout if n_menu_in_holdout else 0.0
    )
    print(
        f"[eval] holdout_acc={correct_total}/{len(test_patches)} "
        f"({holdout_acc:.4f})"
    )
    print(
        f"[eval] EMPTY_acc={correct_menu}/{n_menu_in_holdout} "
        f"({empty_acc:.4f})"
    )

    # 10. 保存
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(classifier._model.state_dict(), str(args.out_model))
    print(f"[save] {args.out_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
