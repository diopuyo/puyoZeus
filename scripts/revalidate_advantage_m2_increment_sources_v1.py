"""収集済みM2純増2 sourceを現行quarantine契約で再検証する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from scripts import build_advantage_m2_dataset_v1 as receipt
from scripts import run_advantage_m2_increment_sources_v1 as campaign
from scripts import run_event_pilot_48_v1 as event_campaign


FORMAT_VERSION = "advantage-m2-increment-quarantine-revalidation/v1"
DEFAULT_SOURCE_ROOT = campaign.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    "data/verify/event_source_v1_m2_increment_s1_s2_quarantine_2026-09-06_v1"
)


class IncrementRevalidationError(RuntimeError):
    """収集済み純増sourceの再検証契約違反。"""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IncrementRevalidationError(f"JSONを読めません: {path}") from error
    if not isinstance(value, dict):
        raise IncrementRevalidationError(f"JSON objectではありません: {path}")
    return value


def _retry_paths(
    source_root: Path, row: campaign.IncrementSourceV1,
) -> event_campaign.RetryPaths:
    attempt = f"{campaign.ATTEMPT_NAMESPACE}-{row.target_id}"
    root = source_root / "targets" / f"target={row.target_id}"
    root = root / f"attempt={attempt}" / "retry=0000"
    return event_campaign.RetryPaths(
        0, attempt, root, root / "work/collection.npz",
        root / "logs/collection.log", root / "results/run.json",
        root / "event-runs", root / "validation",
    )


def _revalidate_one(
    source_root: Path, row: campaign.IncrementSourceV1,
    command_runner: campaign.CommandRunner,
) -> dict[str, Any]:
    paths = _retry_paths(source_root, row)
    metadata = event_campaign.probe_video((campaign.REPO_ROOT / row.video_path).resolve())
    run = campaign._load_run_reference(paths, metadata)
    validation = event_campaign.run_validations(run, command_runner)
    cross = validation.summary.get("cross")
    if not validation.all_pass or not isinstance(cross, dict):
        raise IncrementRevalidationError(f"再検証gate不合格です: {row.target_id}")
    return {
        "target_id": row.target_id, "youtube_id": row.youtube_id,
        "fold": row.fold, "run_dir": str(run.run_dir),
        "validation_round_root": str(validation.round_root.resolve()),
        "physical_gate_pass": cross.get("physical_gate_pass"),
        "quarantine_contract_pass": cross.get("quarantine_contract_pass"),
        "unsupported_game_count": cross.get("unsupported_game_count"),
        "high_confidence_candidate_count": cross.get(
            "high_confidence_candidate_count"
        ),
        "high_confidence_unsupported_count": cross.get(
            "high_confidence_unsupported_count"
        ),
    }


def _safe_revalidate(
    source_root: Path, row: campaign.IncrementSourceV1,
    command_runner: campaign.CommandRunner,
) -> dict[str, Any]:
    try:
        return {"state": "completed", **_revalidate_one(source_root, row, command_runner)}
    except Exception as error:  # noqa: BLE001 - 2件両方の結果を証跡化する
        return {
            "state": "failed", "target_id": row.target_id,
            "youtube_id": row.youtube_id, "error": repr(error),
        }


def run_revalidation(
    args: argparse.Namespace,
    command_runner: campaign.CommandRunner = event_campaign._execute_command,
) -> dict[str, Any]:
    """旧成果物を上書きせず、完成runだけを新契約で再検証する。"""

    if args.output_root.exists():
        raise IncrementRevalidationError(f"出力先は新規必須です: {args.output_root}")
    original_summary = args.source_root / "SUMMARY.json"
    original = _read_json(original_summary)
    rows = campaign.load_manifest(args.manifest)
    preflight = campaign.validate_inputs(rows, args.parent_manifest)
    if original.get("manifest_sha256") != receipt.file_sha256(args.manifest):
        raise IncrementRevalidationError("旧campaignと投入manifestが一致しません")
    args.output_root.mkdir(parents=True, exist_ok=False)
    results = [_safe_revalidate(args.source_root, row, command_runner) for row in rows]
    all_pass = all(row["state"] == "completed" for row in results)
    report = {
        "format_version": FORMAT_VERSION, "all_pass": all_pass,
        "not_production": True, "source_root": str(args.source_root.resolve()),
        "source_summary_sha256": receipt.file_sha256(original_summary),
        "manifest_sha256": receipt.file_sha256(args.manifest),
        "preflight": preflight, "results": results,
        "production_config_changed": False, "video_deleted": False,
    }
    report_path = args.output_root / "REPORT.json"
    campaign._write_json_exclusive(report_path, report)
    if all_pass:
        campaign._write_json_exclusive(
            args.output_root / "COMPLETE",
            {"format_version": FORMAT_VERSION,
             "report_sha256": receipt.file_sha256(report_path)},
        )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=campaign.DEFAULT_MANIFEST)
    parser.add_argument(
        "--parent-manifest", type=Path, default=campaign.DEFAULT_PARENT_MANIFEST,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    report = run_revalidation(parse_args(argv))
    print(json.dumps({"all_pass": report["all_pass"]}, ensure_ascii=False))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
