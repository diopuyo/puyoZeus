"""CellColorFineTuner: cell 色 CNN を擬似ラベルで AdamW fine-tune.

base model: cnn_phase_b_v1.pt (cnn_phase_b 系列の自己学習基底)
output:     cnn_phase_b_finetuned.pt (Phase I.b 自己学習後)

入力: list[PseudoLabelSample] (component=COMPONENT_CELL)
    input_data["patch"] = BGR cell patch (uint8)
    label = settled_color (int, COLOR_*)

実装方針:
    - CnnPatchClassifier 経由で base model を load
    - validation split (10%) を切って fine-tune 前後の loss/accuracy を測定
    - AdamW (lr=1e-4, epochs=5, batch_size=64) で学習
    - rollback() で初期 state_dict 復元
    - 学習後 model を output_path に保存
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.board import Board
from src.cell_augment import (
    DEFAULT_PERMUTABLE_COLORS,
    apply_color_permutation_to_label,
    permute_colors_in_patch,
    random_color_permutation,
)
from src.board import COLOR_OJAMA
from src.patch_classifier import (
    COLOR_TO_CLASS_INDEX,
    CLASS_INDEX_TO_COLOR,
    CnnPatchClassifier,
)

# cycle 57 (2026-05-23): ojama 層凍結 fine-tune 用. 最終 Linear 層の
# OJAMA class row だけ gradient zero にして ojama 認識を保護する。
# cycle 56_v2 で ojama -99.6% 退行した教訓に対応。
OJAMA_CLASS_INDEX: int = COLOR_TO_CLASS_INDEX[COLOR_OJAMA]
from src.self_supervised.online_fine_tuner import OnlineFineTuner
from src.self_supervised.physical_consistency import (
    filter_pseudo_labels_by_consistency,
)
from src.self_supervised.pseudo_label import (
    COMPONENT_CELL,
    PseudoLabelSample,
)
from src.self_supervised.topo_filter import (
    DEFAULT_MIN_AGREEMENT,
    DEFAULT_N_CLUSTERS,
    topo_filter_with_color_symmetry,
)


# board_lookup_fn の型 (timestamp, side -> Board | None)
BoardLookupFn = "Callable[[float, str], Optional[Board]]"


# ============================
# 定数 (ハイパラ)
# ============================

DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_EPOCHS: int = 5
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_BASE_MODEL: str = "models/cnn_phase_b_v1.pt"
DEFAULT_OUTPUT_PATH: str = "models/cnn_phase_b_finetuned.pt"

# fine-tune を実行する最低サンプル数
MIN_TOTAL_SAMPLES: int = 10

# validation split 比率
VALIDATION_RATIO: float = 0.10

# rng seed (再現性)
DEFAULT_SEED: int = 42


# ============================
# Metrics
# ============================


@dataclass
class CellColorFineTuneMetrics:
    """fine_tune の戻り値."""

    n_samples: int
    n_train: int
    n_val: int
    n_epochs: int
    loss_before: float
    loss_after: float
    accuracy_before: float
    accuracy_after: float
    samples_per_label: dict[int, int] = field(default_factory=dict)
    saved_to: str = ""


# ============================
# Fine-Tuner 本体
# ============================


class CellColorFineTuner(OnlineFineTuner):
    """cell 色 CNN を擬似ラベルで AdamW fine-tune."""

    def __init__(
        self,
        base_model_path: str | Path = DEFAULT_BASE_MODEL,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
        lr: float = DEFAULT_LEARNING_RATE,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        validation_ratio: float = VALIDATION_RATIO,
        seed: int = DEFAULT_SEED,
        cnn: CnnPatchClassifier | None = None,
        board_lookup_fn: Optional[
            Callable[[float, str], Optional[Board]]
        ] = None,
        augment: bool = False,
        enable_topo_filter: bool = False,
        topo_n_clusters: int = DEFAULT_N_CLUSTERS,
        topo_min_agreement: float = DEFAULT_MIN_AGREEMENT,
        topo_n_permutations: Optional[int] = None,
        topo_max_samples: Optional[int] = None,
        topo_use_minibatch: bool = True,
        class_balance: bool = False,
        focal_gamma: float = 0.0,
        logit_adjustment_tau: float = 0.0,
        oversample_alpha: float = 0.0,
        freeze_ojama_logit: bool = False,
    ) -> None:
        if lr <= 0 or epochs <= 0 or batch_size <= 0:
            raise ValueError("lr/epochs/batch_size must be positive")
        if not (0.0 <= validation_ratio < 1.0):
            raise ValueError("validation_ratio must be in [0, 1)")
        self._base_model_path = Path(base_model_path)
        self._output_path = Path(output_path)
        self._lr = float(lr)
        self._epochs = int(epochs)
        self._batch_size = int(batch_size)
        self._val_ratio = float(validation_ratio)
        self._seed = int(seed)
        self._cnn: CnnPatchClassifier | None = cnn  # 既存 inst を渡せばそれを使う
        self._backup_state: Any | None = None
        # 物理整合性 filter で使う board lookup (None = filter 無効, 後方互換)
        self._board_lookup_fn: Optional[
            Callable[[float, str], Optional[Board]]
        ] = board_lookup_fn
        # 直近 fine_tune の filter stats (デバッグ用に保持)
        self._last_filter_stats: dict[str, int] = {}
        # 4 色 permutation augmentation (default off で後方互換)
        self._augment: bool = bool(augment)
        # S-7 TopoFilter (default off で後方互換)
        self._enable_topo_filter: bool = bool(enable_topo_filter)
        self._topo_n_clusters: int = int(topo_n_clusters)
        self._topo_min_agreement: float = float(topo_min_agreement)
        self._topo_n_permutations: Optional[int] = (
            int(topo_n_permutations)
            if topo_n_permutations is not None
            else None
        )
        # OOM ガード: 巨大データで使う sub-sample 上限と MiniBatchKMeans 切替
        self._topo_max_samples: Optional[int] = (
            int(topo_max_samples)
            if topo_max_samples is not None
            else None
        )
        self._topo_use_minibatch: bool = bool(topo_use_minibatch)
        # クラスバランス: 不均衡 dataset (例 empty 43% / others 各 5-10%) で
        # mode collapse を防ぐため、cross_entropy に class weight を適用。
        # weight = (median_count / class_count) で頻出 class を抑える。
        self._class_balance: bool = bool(class_balance)
        # Focal loss (gamma>0) / Logit Adjustment (tau>0)。
        # focal_gamma = 0.0 で従来の class-weighted CE。
        # logit_adjustment_tau > 0.0 で train 時 logits から log(prior)*tau を減算
        # し、長尾 dataset で minority class が学べるよう補正 (Menon et al.
        # ICLR 2021)。
        self._focal_gamma: float = float(focal_gamma)
        self._la_tau: float = float(logit_adjustment_tau)
        # CReST 風 minority oversample: class 別 sample 数を
        # (median / N_c)^alpha で複製。alpha=0 で no-op、alpha=1 で完全均等。
        self._oversample_alpha: float = float(oversample_alpha)
        # cycle 57 (2026-05-23): ojama 層凍結。 最終 Linear 層の OJAMA class
        # row の gradient を zero にする (= ojama 認識を保護)。 5 色のみ学習。
        self._freeze_ojama_logit: bool = bool(freeze_ojama_logit)
        # 直近 topo filter stats
        self._last_topo_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fine_tune(
        self, samples: list[PseudoLabelSample],
    ) -> dict[str, Any]:
        """擬似ラベルで model を fine-tune.

        board_lookup_fn が与えられていれば、物理整合性 (R-3) で
        擬似ラベルを cross-validate してから学習する。

        Returns metrics dict (CellColorFineTuneMetrics の dict 化)。
        """
        # R-3 物理整合性 filter (board_lookup_fn=None なら no-op)
        samples = self._apply_consistency_filter(samples)
        x_list, y_list = self._collect_pairs(samples)
        if len(x_list) < MIN_TOTAL_SAMPLES:
            return CellColorFineTuneMetrics(
                n_samples=len(x_list), n_train=0, n_val=0, n_epochs=0,
                loss_before=0.0, loss_after=0.0,
                accuracy_before=0.0, accuracy_after=0.0,
                samples_per_label=self._count_per_label(y_list),
            ).__dict__
        # CNN ロード (lazy: cnn が外部注入されていなければ初期化)
        if self._cnn is None:
            self._cnn = self._load_base_cnn()
        if self._backup_state is None:
            self._save_backup()
        # split
        x_train, y_train, x_val, y_val = self._split_train_val(
            x_list, y_list,
        )
        loss_before = self._compute_loss(x_val, y_val)
        acc_before = self._compute_accuracy(x_val, y_val)
        self._train_loop(x_train, y_train)
        loss_after = self._compute_loss(x_val, y_val)
        acc_after = self._compute_accuracy(x_val, y_val)
        # save
        save_str = self._save_output()
        return CellColorFineTuneMetrics(
            n_samples=len(x_list),
            n_train=len(x_train),
            n_val=len(x_val),
            n_epochs=self._epochs,
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            accuracy_before=float(acc_before),
            accuracy_after=float(acc_after),
            samples_per_label=self._count_per_label(y_list),
            saved_to=save_str,
        ).__dict__

    def rollback(self) -> None:
        """fine-tune 前の state_dict に巻き戻し."""
        if self._backup_state is None or self._cnn is None:
            return
        self._cnn._model.load_state_dict(self._backup_state)
        self._backup_state = None

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _apply_consistency_filter(
        self, samples: list[PseudoLabelSample],
    ) -> list[PseudoLabelSample]:
        """物理整合性 filter (R-3) を適用. board_lookup_fn=None なら no-op.

        ``enable_topo_filter=True`` の場合、R-3 の後に S-7 TopoFilter
        (24 通り color permutation + KMeans + 多数決) も chain で適用する。
        """
        # ---- R-3 物理整合性 ----
        if self._board_lookup_fn is None or not samples:
            self._last_filter_stats = {
                "n_in": len(samples), "n_out": len(samples),
                "filter_applied": 0,
            }
            current = list(samples)
        else:
            filtered, stats = filter_pseudo_labels_by_consistency(
                samples, self._board_lookup_fn,
            )
            stats["filter_applied"] = 1
            self._last_filter_stats = stats
            current = filtered
        # ---- S-7 TopoFilter (default off で後方互換) ----
        if self._enable_topo_filter and current:
            current = self._apply_topo_filter(current)
        else:
            self._last_topo_stats = {
                "n_in": len(current), "n_out": len(current),
                "filter_applied": 0,
            }
        return current

    def _apply_topo_filter(
        self, samples: list[PseudoLabelSample],
    ) -> list[PseudoLabelSample]:
        """S-7 TopoFilter を適用 (color symmetry 24 通り集約)."""
        kwargs: dict[str, Any] = dict(
            n_clusters=self._topo_n_clusters,
            min_agreement=self._topo_min_agreement,
            n_permutations=self._topo_n_permutations,
            use_minibatch=self._topo_use_minibatch,
        )
        if self._topo_max_samples is not None:
            kwargs["max_samples"] = self._topo_max_samples
        filtered, stats = topo_filter_with_color_symmetry(samples, **kwargs)
        stats["filter_applied"] = 1
        self._last_topo_stats = stats
        return filtered

    def _load_base_cnn(self) -> CnnPatchClassifier:
        """base_model_path から CnnPatchClassifier をロード."""
        cnn = CnnPatchClassifier()
        if self._base_model_path.is_file():
            import torch
            state = torch.load(
                str(self._base_model_path),
                map_location="cpu", weights_only=True,
            )
            cnn._model.load_state_dict(state)
        # GPU 切替 (CUDA があれば)
        try:
            import os as _os
            import torch as _torch
            if (
                _os.environ.get("CUDA_VISIBLE_DEVICES", "all") != ""
                and _torch.cuda.is_available()
            ):
                cnn.to_device("cuda")
        except Exception:
            pass
        return cnn

    @staticmethod
    def _collect_pairs(
        samples: list[PseudoLabelSample],
    ) -> tuple[list[np.ndarray], list[int]]:
        """擬似ラベルから (patch, color_label) を抽出."""
        color_to_idx: dict[int, int] = dict(COLOR_TO_CLASS_INDEX)
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for s in samples:
            if s.component != COMPONENT_CELL:
                continue
            if not isinstance(s.input_data, dict):
                continue
            patch = s.input_data.get("patch")
            label = s.label
            if patch is None or label is None:
                continue
            if not isinstance(patch, np.ndarray):
                continue
            try:
                color = int(label)
            except (TypeError, ValueError):
                continue
            idx = color_to_idx.get(color)
            if idx is None:
                continue
            xs.append(patch)
            ys.append(int(idx))
        return xs, ys

    def _split_train_val(
        self, xs: list[np.ndarray], ys: list[int],
    ) -> tuple[list[np.ndarray], list[int], list[np.ndarray], list[int]]:
        """train/val 分割."""
        n = len(xs)
        rng = np.random.default_rng(seed=self._seed)
        idx = rng.permutation(n)
        n_val = int(n * self._val_ratio)
        if n_val == 0 and n >= 4:
            n_val = max(1, n // 10)
        val_idx = set(int(i) for i in idx[:n_val])
        x_train: list[np.ndarray] = []
        y_train: list[int] = []
        x_val: list[np.ndarray] = []
        y_val: list[int] = []
        for i in range(n):
            if i in val_idx:
                x_val.append(xs[i])
                y_val.append(ys[i])
            else:
                x_train.append(xs[i])
                y_train.append(ys[i])
        return x_train, y_train, x_val, y_val

    @staticmethod
    def _count_per_label(ys: list[int]) -> dict[int, int]:
        out: dict[int, int] = {}
        for y in ys:
            out[y] = out.get(y, 0) + 1
        return out

    def _save_backup(self) -> None:
        """現 state_dict を backup."""
        assert self._cnn is not None
        self._backup_state = {
            k: v.detach().clone()
            for k, v in self._cnn._model.state_dict().items()
        }

    def _save_output(self) -> str:
        """fine-tune 後 model を output_path に保存."""
        assert self._cnn is not None
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        torch = self._cnn._torch
        torch.save(self._cnn._model.state_dict(), str(self._output_path))
        return str(self._output_path)

    def _compute_loss(
        self, xs: list[np.ndarray], ys: list[int],
    ) -> float:
        """全サンプルでの mean cross-entropy loss を計算."""
        if not xs or self._cnn is None:
            return 0.0
        torch = self._cnn._torch
        self._cnn._model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for i in range(0, len(xs), self._batch_size):
                bx = xs[i:i + self._batch_size]
                by = ys[i:i + self._batch_size]
                tensors = [self._cnn._patch_to_tensor(p)[0] for p in bx]
                batch = torch.stack(tensors).to(self._cnn._device)
                target = torch.tensor(by, dtype=torch.long).to(
                    self._cnn._device,
                )
                logits = self._cnn._model(batch)
                loss = torch.nn.functional.cross_entropy(
                    logits, target, reduction="sum",
                )
                total += float(loss.item())
                n += len(bx)
        return total / max(1, n)

    def _compute_accuracy(
        self, xs: list[np.ndarray], ys: list[int],
    ) -> float:
        """全サンプルでの accuracy を計算."""
        if not xs or self._cnn is None:
            return 0.0
        torch = self._cnn._torch
        self._cnn._model.eval()
        correct = 0
        n = 0
        with torch.no_grad():
            for i in range(0, len(xs), self._batch_size):
                bx = xs[i:i + self._batch_size]
                by = ys[i:i + self._batch_size]
                tensors = [self._cnn._patch_to_tensor(p)[0] for p in bx]
                batch = torch.stack(tensors).to(self._cnn._device)
                target = torch.tensor(by, dtype=torch.long).to(
                    self._cnn._device,
                )
                logits = self._cnn._model(batch)
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == target).sum().item())
                n += len(bx)
        return correct / max(1, n)

    def _train_loop(
        self, xs: list[np.ndarray], ys: list[int],
    ) -> None:
        """AdamW で N epoch 学習.

        ``augment=True`` の場合、batch ごとに 4 色 permutation を抽選し、
        対象 cell (RED/BLUE/GREEN/YELLOW のみ) の patch hue と label を
        対応付けて変換する。EMPTY/OJAMA/UNKNOWN/PURPLE は不変。
        """
        if not xs or self._cnn is None:
            return
        torch = self._cnn._torch
        self._cnn._model.train()
        opt = torch.optim.AdamW(
            self._cnn._model.parameters(), lr=self._lr,
        )
        # cycle 57: ojama 層凍結 hook (= 最終 Linear 層の OJAMA row だけ
        # gradient を zero にする)。 module は nn.Sequential の最終要素。
        ojama_hooks: list[Any] = []
        if self._freeze_ojama_logit:
            final_layer = self._cnn._model[-1]
            ojama_idx = OJAMA_CLASS_INDEX

            def _mask_ojama_row(grad):  # noqa: ANN001
                g = grad.clone()
                g[ojama_idx] = 0
                return g

            ojama_hooks.append(
                final_layer.weight.register_hook(_mask_ojama_row),
            )
            ojama_hooks.append(
                final_layer.bias.register_hook(_mask_ojama_row),
            )
        # class balance / logit adjustment: 共通で class 分布を計算
        counts: dict[int, int] = {}
        for y in ys:
            counts[int(y)] = counts.get(int(y), 0) + 1
        n_cls = max(int(max(counts.keys())) + 1, 7) if counts else 7
        # class weight (median / count)
        class_weight_t = None
        if self._class_balance and counts:
            med = float(np.median(list(counts.values())))
            w = np.ones(n_cls, dtype=np.float32)
            for cls_idx, cnt in counts.items():
                w[int(cls_idx)] = med / max(1.0, float(cnt))
            class_weight_t = torch.tensor(w).to(self._cnn._device)
        # logit adjustment bias: log(prior) * tau (Menon et al. ICLR 2021)
        la_bias_t = None
        if self._la_tau > 0.0 and counts:
            total = float(sum(counts.values()))
            prior = np.full(n_cls, 1e-6, dtype=np.float32)
            for cls_idx, cnt in counts.items():
                prior[int(cls_idx)] = float(cnt) / total
            la_bias = (np.log(prior) * self._la_tau).astype(np.float32)
            la_bias_t = torch.tensor(la_bias).to(self._cnn._device)
        rng = np.random.default_rng(seed=self._seed)
        # oversample: class 別 indices を生成し、minority を repeat してから shuffle
        oversample_indices: np.ndarray | None = None
        if self._oversample_alpha > 0.0 and counts:
            class_to_indices: dict[int, list[int]] = {}
            for i, y in enumerate(ys):
                class_to_indices.setdefault(int(y), []).append(i)
            n_per_cls = list(counts.values())
            med = float(np.median(n_per_cls))
            chunks: list[np.ndarray] = []
            for cls_idx, indices in class_to_indices.items():
                base_n = len(indices)
                if base_n == 0:
                    continue
                # repeat 倍率 (alpha=1 で均等、alpha=0 で repeat 1)
                target_ratio = (med / float(base_n)) ** self._oversample_alpha
                target_n = int(round(base_n * max(1.0, target_ratio)))
                if target_n <= base_n:
                    chunks.append(np.array(indices, dtype=np.int64))
                else:
                    # repeat + 余り random sample
                    full_reps = target_n // base_n
                    extra = target_n - full_reps * base_n
                    arr = np.tile(np.array(indices, dtype=np.int64), full_reps)
                    if extra:
                        extra_pick = rng.choice(
                            np.array(indices, dtype=np.int64),
                            size=extra, replace=False,
                        )
                        arr = np.concatenate([arr, extra_pick])
                    chunks.append(arr)
            if chunks:
                oversample_indices = np.concatenate(chunks)
        for _ in range(self._epochs):
            if oversample_indices is not None:
                idx = oversample_indices.copy()
                rng.shuffle(idx)
            else:
                idx = rng.permutation(len(xs))
            for i in range(0, len(idx), self._batch_size):
                batch_idx = idx[i:i + self._batch_size]
                bx = [xs[int(j)] for j in batch_idx]
                by = [ys[int(j)] for j in batch_idx]
                if self._augment:
                    bx, by = self._augment_batch(bx, by, rng)
                tensors = [self._cnn._patch_to_tensor(p)[0] for p in bx]
                batch = torch.stack(tensors).to(self._cnn._device)
                target = torch.tensor(by, dtype=torch.long).to(
                    self._cnn._device,
                )
                logits = self._cnn._model(batch)
                # Logit Adjustment: train 時のみ logits から log(prior)*tau を減算
                if la_bias_t is not None:
                    adj_logits = logits - la_bias_t.unsqueeze(0)
                else:
                    adj_logits = logits
                if self._focal_gamma > 0.0:
                    # Focal loss: (1-p_t)^gamma * CE
                    log_probs = torch.nn.functional.log_softmax(
                        adj_logits, dim=1,
                    )
                    nll = torch.nn.functional.nll_loss(
                        log_probs, target,
                        weight=class_weight_t, reduction="none",
                    )
                    p_t = torch.exp(-nll)
                    focal_factor = (1.0 - p_t).clamp(min=1e-8) ** self._focal_gamma
                    loss = (focal_factor * nll).mean()
                else:
                    loss = torch.nn.functional.cross_entropy(
                        adj_logits, target, weight=class_weight_t,
                    )
                opt.zero_grad()
                loss.backward()
                opt.step()
        # cycle 57: ojama hook を解除 (= 後続の学習に影響残さない)
        for h in ojama_hooks:
            h.remove()
        self._cnn._model.eval()

    @staticmethod
    def _augment_batch(
        bx: list[np.ndarray],
        by: list[int],
        rng: np.random.Generator,
    ) -> tuple[list[np.ndarray], list[int]]:
        """batch に 1 つの color permutation を適用 (patch + label 同時変換).

        実装シンプル化のため batch 全体に同一 permutation を適用する。
        epoch / iteration ごとに rng で異なる permutation を引くため、
        十分な多様性が得られる。

        Args:
            bx: BGR patch list (uint8)。
            by: class index list。
            rng: numpy.random.Generator (seed 制御)。

        Returns:
            tuple: 変換後 (patch list, class index list)。
        """
        cmap = random_color_permutation(rng, DEFAULT_PERMUTABLE_COLORS)
        new_bx: list[np.ndarray] = []
        new_by: list[int] = []
        for patch, cls_idx in zip(bx, by):
            color = int(CLASS_INDEX_TO_COLOR[int(cls_idx)])
            new_patch = permute_colors_in_patch(patch, color, cmap)
            new_color = apply_color_permutation_to_label(color, cmap)
            new_cls = int(COLOR_TO_CLASS_INDEX[new_color])
            new_bx.append(new_patch)
            new_by.append(new_cls)
        return new_bx, new_by


__all__ = [
    "BoardLookupFn",
    "CellColorFineTuneMetrics",
    "CellColorFineTuner",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_OUTPUT_PATH",
    "MIN_TOTAL_SAMPLES",
    "VALIDATION_RATIO",
]
