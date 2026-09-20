"""保存済み先頭30本を再認識せず、固定版の完成原本へ3並列で再出力する。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class ReexportInput:
    target_id: str
    video: Path
    npz: Path
    observation: Path
    accounting: Path
    physical: Path
    base_config: Path


def _load_target_ids(marker: Path) -> tuple[str, ...]:
    value = json.loads(marker.read_text(encoding="utf-8"))
    targets = value.get("completed_target_ids")
    if value.get("completed_raw_count") != 30 or not isinstance(targets, list):
        raise ValueError("30本完了印が不正です")
    result = tuple(str(item) for item in targets)
    if len(result) != 30 or len(set(result)) != 30:
        raise ValueError("30本完了印の対象数または一意性が不正です")
    return result


def _target_input(project_root: Path, raw_root: Path, target_id: str) -> ReexportInput:
    target_root = raw_root / "targets" / f"target={target_id}"
    candidates = sorted(target_root.glob("attempt=*/retry=*/work/collection.npz"))
    complete = [path for path in candidates if _has_all_sidecars(path)]
    if len(complete) != 1:
        raise ValueError(f"{target_id}の保存済み収集結果を一意に選べません")
    npz = complete[0]
    retry_root = npz.parent.parent
    configs = sorted(retry_root.glob("event-runs/**/recognition-config.json"))
    if len(configs) != 1:
        raise ValueError(f"{target_id}の基準設定を一意に選べません")
    config = json.loads(configs[0].read_text(encoding="utf-8"))
    source_id = str(config.get("source_video_id", ""))
    video = project_root / "data" / "frames" / f"{source_id}.mp4"
    if source_id != f"video_{target_id}" or not video.is_file():
        raise ValueError(f"{target_id}の元映像が一致しません")
    return ReexportInput(
        target_id, video, npz,
        npz.with_name(f"{npz.stem}_event_observations_v1.json"),
        npz.with_name(f"{npz.stem}_event_accounting_v1.json"),
        npz.with_name(f"{npz.stem}_event_physical_v1.json"), configs[0],
    )


def _has_all_sidecars(npz: Path) -> bool:
    suffixes = (
        "_event_observations_v1.json",
        "_event_accounting_v1.json",
        "_event_physical_v1.json",
    )
    return npz.is_file() and all(
        npz.with_name(f"{npz.stem}{suffix}").is_file() for suffix in suffixes
    )


def _run_one(item: ReexportInput, output_root: Path) -> dict[str, Any]:
    result_path = output_root / "results" / f"target={item.target_id}.json"
    log_path = output_root / "logs" / f"target={item.target_id}.log"
    if result_path.exists() or log_path.exists():
        raise FileExistsError(f"{item.target_id}の再出力先が既に存在します")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _command(item, output_root, result_path)
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode != 0 or not result_path.is_file():
        raise RuntimeError(f"{item.target_id}の再出力に失敗しました: {log_path}")
    value = json.loads(result_path.read_text(encoding="utf-8"))
    return {"target_id": item.target_id, **value}


def _command(
    item: ReexportInput, output_root: Path, result_path: Path,
) -> list[str]:
    return [
        sys.executable, "-u", "-m", "scripts.reexport_event_snapshot_pilot_v1",
        "--base-config", str(item.base_config), "--video", str(item.video),
        "--npz", str(item.npz), "--observation-sidecar", str(item.observation),
        "--accounting-sidecar", str(item.accounting),
        "--physical-sidecar", str(item.physical),
        "--output-root", str(output_root / "runs"),
        "--attempt-id", f"first30-order-ambiguity-v1-{item.target_id}",
        "--result-json", str(result_path),
    ]


def execute(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent
    target_ids = _load_target_ids(args.marker)
    items = [_target_input(project_root, args.raw_root, item) for item in target_ids]
    if args.check_only:
        return {
            "format": "event-first30-reexport/1",
            "target_count": len(items),
            "workers": args.workers,
            "target_ids": list(target_ids),
            "runs": [],
        }
    if args.output_root.exists():
        raise FileExistsError("30本再出力の新規ルートが既に存在します")
    args.output_root.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_one, item, args.output_root): item.target_id
            for item in items
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[first30-reexport] completed {result['target_id']} "
                f"({len(results)}/30)", flush=True,
            )
    results.sort(key=lambda item: target_ids.index(str(item["target_id"])))
    summary = {
        "format": "event-first30-reexport/1",
        "target_count": len(results),
        "workers": args.workers,
        "target_ids": list(target_ids),
        "runs": results,
    }
    _write_exclusive(args.output_root / "SUMMARY.json", summary)
    return summary


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 3:
        parser.error("workersは1〜3です")
    return args


def main() -> int:
    summary = execute(parse_args())
    print(json.dumps({
        "target_count": summary["target_count"],
        "workers": summary["workers"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
