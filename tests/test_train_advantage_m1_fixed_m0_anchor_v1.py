from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from scripts import merge_advantage_m1_zero_counterfactual_v3 as merger
from scripts import train_advantage_m0_current_cnn_v1 as base
from scripts import train_advantage_m1_frozen_residual_v1 as frozen
from scripts import train_advantage_m1_fixed_m0_anchor_v1 as subject
from scripts import train_advantage_m1_zero_counterfactual_v3 as v3
from src.advantage_m0_current_cnn_v1 import AdvantageM0CurrentCNNV2
from src.event_provisional_oof_v1 import FoldPlan


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _anchor_plan() -> dict[str, Any]:
    return {
        "format_version": merger.MERGE_PLAN_VERSION,
        "source_plan_format_version": v3.PLAN_VERSION,
        "source_results_format_version": v3.TRAINING_VERSION,
        "source_complete_format_version": v3.COMPLETE_VERSION,
        "dataset_format_version": v3.builder.FORMAT_VERSION,
        "input_schema_version": v3.M1_INPUT_SCHEMA_VERSION,
        "model_version": v3.ZERO_COUNTERFACTUAL_MODEL_VERSION,
        "source_count": 46, "dataset_root": "/unused/anchor",
        "dataset_sha256": "dataset", "dataset_manifest_sha256": "manifest",
        "dataset_complete_sha256": "complete",
        "source_group_fold_mapping_sha256": "fold-map",
    }


def _m0_record(path: Path, seed: int, fold: int) -> dict[str, Any]:
    digest = base.file_sha256(path)
    plan = FoldPlan(fold, fold % 6 + 1, tuple(
        value for value in range(1, 7) if value not in {fold, fold % 6 + 1}
    ))
    return {
        "eval_fold": fold, "tune_fold": plan.tune_fold,
        "train_folds": list(plan.train_folds), "best_epoch": 1,
        "symmetric_platt_slope": 0.4,
        "tune_raw": {"game_equal_log_loss": 0.7},
        "tune_calibrated": {"game_equal_log_loss": 0.6},
        "model": path.name, "model_sha256": digest,
        "m0_state_sha256": "a" * 64,
        "checkpoint_reference": {"path": str(path), "sha256": digest},
    }


def _make_anchor_root(
    root: Path, *, missing: tuple[int, int] | None = None,
) -> Path:
    root.mkdir()
    results: dict[str, Any] = {}
    for seed in subject.PUBLIC_SEEDS:
        records = []
        for fold in range(1, 7):
            if (seed, fold) == missing:
                continue
            checkpoint = root / f"m0__seed_{seed}__fold_{fold}.pt"
            checkpoint.write_bytes(f"{seed}/{fold}".encode("ascii"))
            records.append(_m0_record(checkpoint, seed, fold))
        results[f"m0__seed_{seed}"] = {"records": records}
    plan = _anchor_plan()
    _write_json(root / "PLAN.json", plan)
    _write_json(root / "results.json", {
        "format_version": merger.MERGE_RESULTS_VERSION,
        "not_production": True, "plan": plan, "results": results,
    })
    np.savez(root / "oof_predictions.npz", labels=np.asarray([0], dtype=np.float32))
    _write_json(root / "COMPLETE", {
        "format_version": merger.MERGE_COMPLETE_VERSION,
        "plan_sha256": base.file_sha256(root / "PLAN.json"),
        "results_sha256": base.file_sha256(root / "results.json"),
        "predictions_sha256": base.file_sha256(root / "oof_predictions.npz"),
    })
    return root


def _reference(path: Path, *, state_sha256: str = "a" * 64) -> subject.M0AnchorReference:
    return subject.M0AnchorReference(
        20260904, 1, path, base.file_sha256(path), state_sha256, 0.4, 1,
        {"game_equal_log_loss": 0.7}, {"game_equal_log_loss": 0.6},
        path.parent, "b" * 64, path.parent / "oof_predictions.npz", "c" * 64,
    )


def _samples(
    state_ids: list[str], groups: list[str], games: list[str], folds: list[int],
) -> v3.CanonicalSamplesV3:
    count = len(state_ids)
    return v3.CanonicalSamplesV3(
        np.zeros((count, 2, 13, 6), dtype=np.int8),
        np.zeros((count, 2, 4), dtype=np.int8),
        np.zeros((count, 2, 6), dtype=np.float32),
        np.zeros((count, 2, 6, 5), dtype=np.float32),
        np.asarray([index % 2 for index in range(count)], dtype=np.float32),
        np.asarray(groups), np.asarray(games), np.ones(count, dtype=np.float32),
        np.asarray(folds, dtype=np.int8), np.asarray(state_ids),
        np.arange(count, dtype=np.int64), np.zeros(count, dtype=np.int32),
        np.ones(count, dtype=np.bool_), np.asarray(["" for _ in state_ids]),
        np.asarray(["not_joined" for _ in state_ids]),
    )


def _manifests() -> tuple[dict[str, Any], dict[str, Any]]:
    groups = [f"old-{index}" for index in range(46)]
    sources = [
        {"target_id": f"v{index}", "partition_fold": index % 6 + 1}
        for index in range(46)
    ]
    anchor = {"source_count": 46, "source_group_ids": groups, "sources": sources}
    added = list(subject.EXPECTED_ADDED_GROUPS)
    target_sources = [dict(value) for value in sources] + [
        {"target_id": subject.EXPECTED_ADDED_GROUPS[group][0],
         "partition_fold": subject.EXPECTED_ADDED_GROUPS[group][1]}
        for group in added
    ]
    target = {
        "source_count": 48, "source_group_ids": groups + added,
        "sources": target_sources,
    }
    return anchor, target


def _relation_fixture() -> tuple[
    v3.CanonicalSamplesV3, dict[str, Any], v3.CanonicalSamplesV3, dict[str, Any]
]:
    anchor_manifest, target_manifest = _manifests()
    anchor = _samples(["s0", "s1"], ["old-0", "old-1"], ["g0", "g1"], [1, 2])
    added = list(subject.EXPECTED_ADDED_GROUPS)
    target = _samples(
        ["s0", "s1", "s2", "s3"], ["old-0", "old-1", *added],
        ["g0", "g1", "g2", "g3"], [1, 2, 1, 3],
    )
    return anchor, anchor_manifest, target, target_manifest


def test_registry_builds_exact_public_seed_eval_fold_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_anchor_root(tmp_path / "anchor")
    monkeypatch.setattr(subject, "load_m0_anchor", lambda *_: object())
    registry, receipts = subject.load_anchor_registry([root], enforce_preregistered=False)
    assert set(registry) == {
        (seed, fold) for seed in subject.PUBLIC_SEEDS for fold in range(1, 7)
    }
    assert registry[(20260905, 4)].seed == 20260905
    assert registry[(20260905, 4)].fold == 4
    assert len(receipts) == 1


def test_registry_rejects_checkpoint_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_anchor_root(tmp_path / "anchor")
    (root / "m0__seed_20260904__fold_1.pt").write_bytes(b"tampered")
    monkeypatch.setattr(subject, "load_m0_anchor", lambda *_: object())
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="checkpoint SHA"):
        subject.load_anchor_registry([root], enforce_preregistered=False)


def test_registry_rejects_missing_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_anchor_root(tmp_path / "anchor", missing=(20260906, 6))
    monkeypatch.setattr(subject, "load_m0_anchor", lambda *_: object())
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="coverage"):
        subject.load_anchor_registry([root], enforce_preregistered=False)


def test_registry_rejects_duplicate_seed_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _make_anchor_root(tmp_path / "first")
    second = _make_anchor_root(tmp_path / "second")
    monkeypatch.setattr(subject, "load_m0_anchor", lambda *_: object())
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="重複"):
        subject.load_anchor_registry([first, second], enforce_preregistered=False)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("variant", "values_and_masks"), ("seed", 9), ("fold", 2),
        ("model_version", "wrong"), ("m0_state_sha256", "b" * 64),
        ("symmetric_platt_slope", 0.8),
    ],
)
def test_checkpoint_metadata_mismatch_fails(
    tmp_path: Path, key: str, value: Any,
) -> None:
    path = tmp_path / "model.pt"
    path.write_bytes(b"placeholder")
    reference = _reference(path)
    checkpoint = {
        "variant": "m0", "seed": reference.seed, "fold": reference.fold,
        "model_version": AdvantageM0CurrentCNNV2.model_version,
        "m0_state_sha256": reference.state_sha256,
        "symmetric_platt_slope": reference.slope, "state_dict": {},
    }
    checkpoint[key] = value
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="metadata"):
        subject._validate_checkpoint_payload(checkpoint, reference)


def test_load_m0_anchor_and_state_tamper(tmp_path: Path) -> None:
    model = AdvantageM0CurrentCNNV2()
    state_sha = frozen.model_state_sha256(model)
    path = tmp_path / "model.pt"
    torch.save({
        "variant": "m0", "seed": 20260904, "fold": 1,
        "model_version": model.model_version, "m0_state_sha256": state_sha,
        "symmetric_platt_slope": 0.4, "state_dict": model.state_dict(),
    }, path)
    loaded = subject.load_m0_anchor(_reference(path, state_sha256=state_sha), torch.device("cpu"))
    assert loaded.state_sha256 == state_sha
    bad = _reference(path, state_sha256="0" * 64)
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="metadata"):
        subject.load_m0_anchor(bad, torch.device("cpu"))


def test_load_m0_anchor_rejects_model_state_shape(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save({
        "variant": "m0", "seed": 20260904, "fold": 1,
        "model_version": AdvantageM0CurrentCNNV2.model_version,
        "m0_state_sha256": "a" * 64, "symmetric_platt_slope": 0.4,
        "state_dict": {},
    }, path)
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="state/model"):
        subject.load_m0_anchor(_reference(path), torch.device("cpu"))


def test_dataset_relation_accepts_exact_clean46_subset() -> None:
    anchor, anchor_manifest, target, target_manifest = _relation_fixture()
    relation = subject.validate_dataset_relation(
        anchor, anchor_manifest, target, target_manifest,
    )
    assert relation.receipt["exact_common_state_count"] == 2
    assert relation.receipt["added_state_count"] == 2
    assert relation.receipt["game_collision_count"] == 0


def test_dataset_relation_rejects_changed_common_state() -> None:
    anchor, anchor_manifest, target, target_manifest = _relation_fixture()
    changed = target.labels.copy()
    changed[0] = 1.0 - changed[0]
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="labels"):
        subject.validate_dataset_relation(
            anchor, anchor_manifest, replace(target, labels=changed), target_manifest,
        )


def test_dataset_relation_allows_only_uniform_weight_renormalization() -> None:
    anchor, anchor_manifest, target, target_manifest = _relation_fixture()
    uniform = target.weights.copy()
    uniform[:2] *= np.float32(0.9)
    subject.validate_dataset_relation(
        anchor, anchor_manifest, replace(target, weights=uniform), target_manifest,
    )
    nonuniform = uniform.copy()
    nonuniform[0] *= np.float32(0.9)
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="一様な再正規化"):
        subject.validate_dataset_relation(
            anchor, anchor_manifest, replace(target, weights=nonuniform), target_manifest,
        )


def test_increment_weights_restore_clean46_game_equal_scale() -> None:
    anchor_manifest, target_manifest = _manifests()
    anchor = _samples(["s0", "s1"], ["old-0", "old-1"], ["g0", "g1"], [1, 2])
    added = list(subject.EXPECTED_ADDED_GROUPS)
    target = _samples(
        ["s0", "s1", "s2", "s3", "s4"],
        ["old-0", "old-1", added[0], added[0], added[1]],
        ["g0", "g1", "g2", "g2", "g3"], [1, 2, 1, 1, 3],
    )
    target_weights = np.asarray([0.9, 0.9, 0.45, 0.45, 0.9], dtype=np.float32)
    target = replace(target, weights=target_weights)
    relation = subject.validate_dataset_relation(
        anchor, anchor_manifest, target, target_manifest,
    )
    stabilized, receipt = subject.stabilize_increment_weights(target, relation)
    assert np.array_equal(stabilized.weights[:2], anchor.weights)
    assert np.array_equal(
        stabilized.weights[2:], np.asarray([0.5, 0.5, 1.0], dtype=np.float32),
    )
    assert receipt["common_weight_bytes_exact"] is True
    assert len(receipt["added_game_totals"]) == 2
    assert receipt["all_game_totals"]["max_abs_deviation"] == 0.0


def test_increment_weights_reject_unequal_added_game_total() -> None:
    anchor, anchor_manifest, target, target_manifest = _relation_fixture()
    relation = subject.validate_dataset_relation(anchor, anchor_manifest, target, target_manifest)
    bad = target.weights.copy()
    bad[-1] = np.float32(0.5)
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="全game総weight"):
        subject.stabilize_increment_weights(replace(target, weights=bad), relation)


def test_dataset_relation_rejects_wrong_added_source() -> None:
    anchor, anchor_manifest, target, target_manifest = _relation_fixture()
    groups = target.source_groups.copy()
    groups[-1] = "old-2"
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="source所属"):
        subject.validate_dataset_relation(
            anchor, anchor_manifest, replace(target, source_groups=groups), target_manifest,
        )


def test_dataset_relation_rejects_game_collision() -> None:
    anchor, anchor_manifest, target, target_manifest = _relation_fixture()
    games = target.game_keys.copy()
    games[-1] = "g0"
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="game.*衝突"):
        subject.validate_dataset_relation(
            anchor, anchor_manifest, replace(target, game_keys=games), target_manifest,
        )


def test_copy_checkpoint_preserves_exact_bytes(tmp_path: Path) -> None:
    source, output = tmp_path / "source.pt", tmp_path / "output"
    source.write_bytes(bytes(range(255)))
    output.mkdir()
    reference = _reference(source)
    name, digest = subject._copy_checkpoint(reference, output)
    assert name == source.name
    assert digest == base.file_sha256(source) == base.file_sha256(output / name)
    assert (output / name).read_bytes() == source.read_bytes()


def test_common_oof_prediction_requires_bit_exact(tmp_path: Path) -> None:
    anchor, anchor_manifest, target, target_manifest = _relation_fixture()
    relation = subject.validate_dataset_relation(anchor, anchor_manifest, target, target_manifest)
    path = tmp_path / "model.pt"
    path.write_bytes(b"model")
    predictions = tmp_path / "oof_predictions.npz"
    np.savez(predictions, **{
        "labels": anchor.labels, "folds": anchor.folds,
        "m0__seed_20260904__raw": np.asarray([0.2, 0.3], dtype=np.float32),
        "m0__seed_20260904__calibrated": np.asarray([0.4, 0.5], dtype=np.float32),
    })
    reference = replace(
        _reference(path), predictions_path=predictions,
        predictions_sha256=base.file_sha256(predictions),
    )
    subject._assert_common_prediction_exact(
        np.asarray([0.2, 0.3], dtype=np.float32),
        np.asarray([0.4, 0.5], dtype=np.float64), anchor,
        relation, reference, {},
    )
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="raw予測"):
        subject._assert_common_prediction_exact(
            np.asarray([0.21, 0.3], dtype=np.float32),
            np.asarray([0.4, 0.5], dtype=np.float32), anchor,
            relation, reference, {},
        )


def test_plan_keeps_v3_merge_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_plan = {
        "format_version": v3.PLAN_VERSION, "not_production": True,
        "dataset_format_version": v3.builder.FORMAT_VERSION,
        "input_schema_version": v3.M1_INPUT_SCHEMA_VERSION,
        "model_version": v3.ZERO_COUNTERFACTUAL_MODEL_VERSION,
        "projected_inputs_used": False, "source_count": 48,
        "variants": list(subject.DEFAULT_VARIANTS), "seeds": list(subject.PUBLIC_SEEDS),
    }
    monkeypatch.setattr(v3, "_plan", lambda *_: dict(base_plan))
    monkeypatch.setattr(subject, "_code_hashes", lambda: {
        "production_config": subject.EXPECTED_PRODUCTION_CONFIG_SHA256,
        "fixed_m0_anchor_trainer": "a" * 64,
    })
    args, samples = SimpleNamespace(), _samples(["s"], ["g"], ["x"], [1])
    relation = subject.DatasetRelation(samples, {}, {"s": 0}, {"added_state_count": 2})
    path = tmp_path / "m0.pt"
    path.write_bytes(b"m0")
    source = _reference(path)
    registry = {
        (seed, fold): replace(source, seed=seed, fold=fold)
        for seed in subject.PUBLIC_SEEDS for fold in range(1, 7)
    }
    oof = [
        {"seed": seed, "eval_fold": fold, "raw_bit_exact": True}
        for seed in subject.PUBLIC_SEEDS for fold in range(1, 7)
    ]
    plan = subject._build_plan(
        args, samples, {}, [], relation, registry, oof,
        {"seed": 20260904, "eval_fold": 1, "raw_bit_exact": True},
        {"policy": "fixed"}, [],
    )
    report = {"dataset_source_count": 48}
    merger._validate_plan_contract(plan, report, Path("partial"))
    assert merger._result_keys(plan) == tuple(
        f"{name}__seed_{seed}"
        for seed in subject.PUBLIC_SEEDS
        for name in ("m0", "m1_zero_values_and_masks", "m1_zero_random_control")
    )
    assert plan["m0_anchor_policy"] == "external_clean46_fixed"
    assert len(plan["m0_anchor_checkpoint_receipts"]) == 18
    assert len(plan["clean46_m0_oof_validation_receipts"]) == 18


def test_run_fold_uses_public_seed_not_fold_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor, anchor_manifest, target, target_manifest = _relation_fixture()
    relation = subject.validate_dataset_relation(anchor, anchor_manifest, target, target_manifest)
    path = tmp_path / "m0.pt"
    path.write_bytes(b"m0")
    reference = _reference(path)
    loaded = frozen.M0Anchor(
        AdvantageM0CurrentCNNV2(), 0.4, 1, {"x": 1.0}, {"x": 1.0}, "a" * 64,
    )
    observed: dict[str, Any] = {}
    def load_anchor(ref: subject.M0AnchorReference, _device: torch.device) -> frozen.M0Anchor:
        observed["ref"] = ref
        return loaded
    monkeypatch.setattr(subject, "load_m0_anchor", load_anchor)
    monkeypatch.setattr(
        subject.frozen, "_predict_m0",
        lambda _model, rows, *_: np.full(len(rows.labels), 0.2, dtype=np.float32),
    )
    monkeypatch.setattr(subject, "_assert_common_prediction_exact", lambda *_: None)
    monkeypatch.setattr(subject, "_copy_checkpoint", lambda *_: (path.name, reference.sha256))
    monkeypatch.setattr(v3, "_fit_residual", lambda *args: SimpleNamespace())
    monkeypatch.setattr(v3, "_prepare_split", lambda samples, *_: SimpleNamespace(samples=samples))
    monkeypatch.setattr(v3, "_residual_output", lambda *_: v3.FoldOutputV3(
        np.asarray([0.2, 0.3]), np.asarray([0.4, 0.5]), {"fallback_to_m0": True},
    ))
    args = SimpleNamespace(variants=("values_and_masks",), output_root=tmp_path)
    outputs = subject._run_fold(
        target, FoldPlan(1, 2, (3, 4, 5, 6)), 20260904,
        {(20260904, 1): reference}, relation, {}, args, torch.device("cpu"),
    )
    assert observed["ref"] is reference
    assert "m0" in outputs and "m1_zero_values_and_masks" in outputs


def test_parse_args_requires_anchor_roots_and_public_seed(tmp_path: Path) -> None:
    common = ["--dataset-root", str(tmp_path), "--output-root", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        subject.parse_args(common)
    with pytest.raises(SystemExit):
        subject.parse_args([*common, "--anchor-roots", str(tmp_path), "--seeds", "9"])


@pytest.mark.parametrize(
    "override",
    [
        ["--seeds", "20260904,20260905"],
        ["--folds", "1,2"],
        ["--variants", "values_and_masks"], ["--epochs", "11"],
        ["--patience", "2"], ["--batch-size", "256"],
        ["--learning-rate", "0.001"], ["--weight-decay", "0.0"],
    ],
)
def test_parse_args_rejects_preregistered_override(
    tmp_path: Path, override: list[str],
) -> None:
    common = [
        "--dataset-root", str(tmp_path), "--output-root", str(tmp_path / "out"),
        "--anchor-roots", str(tmp_path),
    ]
    with pytest.raises(SystemExit):
        subject.parse_args([*common, *override])


def test_check_only_does_not_require_output_root(tmp_path: Path) -> None:
    args = subject.parse_args([
        "--dataset-root", str(tmp_path), "--anchor-roots", str(tmp_path),
        "--check-only", "--device", "cpu",
    ])
    assert args.check_only is True
    assert args.output_root is None


def test_check_output_is_check_only_and_must_be_new(tmp_path: Path) -> None:
    common = [
        "--dataset-root", str(tmp_path), "--anchor-roots", str(tmp_path),
        "--check-output", str(tmp_path / "receipt.json"),
    ]
    with pytest.raises(SystemExit):
        subject.parse_args(common)
    args = subject.parse_args([*common, "--check-only", "--device", "cpu"])
    assert args.check_output == tmp_path / "receipt.json"
    args.check_output.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        subject.parse_args([*common, "--check-only", "--device", "cpu"])


def test_environment_fingerprint_records_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject.runtime, "_gpu_information", lambda: (["gpu"], ["8.9"]))
    monkeypatch.setattr(subject.runtime, "_cuda_driver_version", lambda: "driver")
    result = subject._environment_fingerprint(torch.device("cpu"))
    assert result["selected_device"] == "cpu"
    assert result["gpu_names"] == ["gpu"]
    assert "deterministic_algorithms" in result


def test_common_weight_scale_receipt_explains_rtol() -> None:
    anchor, anchor_manifest, target, target_manifest = _relation_fixture()
    relation = subject.validate_dataset_relation(
        anchor, anchor_manifest, target, target_manifest,
    )
    assert relation.receipt["common_weight_scale_rtol"] == subject.COMMON_WEIGHT_SCALE_RTOL
    assert "float32" in relation.receipt["common_weight_scale_rtol_rationale"]


def test_input_asset_mutation_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"before")
    receipts = subject._capture_input_assets([path.resolve()])
    subject._assert_input_assets_unchanged(receipts)
    path.write_bytes(b"after")
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="入力資産"):
        subject._assert_input_assets_unchanged(receipts)


def test_input_asset_inventory_includes_plan_oof_and_datasets(tmp_path: Path) -> None:
    result_root = tmp_path / "result"
    anchor_dataset, target_dataset = tmp_path / "anchor-data", tmp_path / "target-data"
    result_root.mkdir()
    _write_json(result_root / "PLAN.json", {"dataset_root": str(anchor_dataset)})
    checkpoint = result_root / "m0.pt"
    checkpoint.write_bytes(b"m0")
    reference = _reference(checkpoint)
    champion_path = result_root / "m1.pt"
    champion_path.write_bytes(b"m1")
    champion = subject.M1ChampionReference(
        champion_path, base.file_sha256(champion_path), "d" * 64,
        result_root, result_root / "oof_predictions.npz", "c" * 64, {},
    )
    paths = set(subject._input_asset_paths(
        SimpleNamespace(dataset_root=target_dataset),
        {(20260904, 1): reference}, champion,
    ))
    required = {
        result_root / "PLAN.json", result_root / "oof_predictions.npz",
        anchor_dataset / "manifest.json", anchor_dataset / "dataset.npz",
        target_dataset / "manifest.json", target_dataset / "dataset.npz", checkpoint,
        champion_path,
    }
    assert {path.resolve() for path in required} <= paths


def test_complete_is_not_written_after_input_mutation(tmp_path: Path) -> None:
    asset, output = tmp_path / "input.bin", tmp_path / "output"
    asset.write_bytes(b"before")
    output.mkdir()
    plan_path = output / "PLAN.json"
    _write_json(plan_path, {"format_version": v3.PLAN_VERSION})
    samples = _samples(["s"], ["g"], ["x"], [1])
    checkpoint = tmp_path / "m1.pt"
    checkpoint.write_bytes(b"m1")
    champion = subject.M1ChampionReference(
        checkpoint, base.file_sha256(checkpoint), "d" * 64,
        tmp_path, tmp_path / "oof.npz", "e" * 64, {},
    )
    inputs = subject.ValidatedInputs(
        samples, {"source_count": 48}, {}, champion, [],
        subject.DatasetRelation(samples, {}, {"s": 0}, {}),
        {"policy": "fixed"},
        subject._capture_input_assets([asset.resolve()]),
    )
    asset.write_bytes(b"after")
    results = {"m0__seed_20260904": {
        "raw": np.asarray([0.2], dtype=np.float32),
        "calibrated": np.asarray([0.3], dtype=np.float32), "records": [],
    }}
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="入力資産"):
        subject._write_results(output, inputs, {}, plan_path, results)
    assert not (output / "COMPLETE").exists()


def test_evidence_coverage_requires_all_18() -> None:
    complete = [
        {"seed": seed, "eval_fold": fold}
        for seed in subject.PUBLIC_SEEDS for fold in range(1, 7)
    ]
    subject._validate_evidence_coverage(complete, complete)
    with pytest.raises(subject.FixedM0AnchorTrainingError, match="OOF receipt"):
        subject._validate_evidence_coverage(complete, complete[:-1])


def test_functions_are_at_most_50_lines() -> None:
    path = Path(subject.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    oversized = {
        node.name: node.end_lineno - node.lineno + 1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.end_lineno - node.lineno + 1 > 50
    }
    assert oversized == {}
