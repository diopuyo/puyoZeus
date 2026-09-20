"""今回の修復対象だけを、起動条件・版・終了コード・JUnitと共に記録する。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

TEST_NAMES = (
    "production_dependency_contract", "analyze_advantage_m2_projected_safe_v1",
    "build_advantage_m1_clean46_manifest_unseen_review_v1", "compare_advantage_m1_46v_48v_common_v1",
    "sanitize_advantage_m1_training_quarantine_v1", "train_advantage_m1_fixed_m0_anchor_v1",
    "train_advantage_m2_projected_safe_v1", "analyze_advantage_m2_old308_anchor_ledger_v1",
    "train_advantage_m2_old308_anchor_ledger_v1", "advantage_overlay_fps_normalize",
    "advantage_overlay_production_recognition_2026-08-13", "advantage_overlay_timeline_dump",
    "measure_stable_cell_acc_production_recognition_2026-08-13",
    "visualize_recognition_production_config_2026-08-13", "measure_worker_arg_order_2026_08_24",
)
ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/mnt/d/puyo_analyzer/verify/pytest_repair_2026-09-20_v1/final_cpu_v1")


def code_hashes() -> dict[str, str]:
    """対象テストの依存を含め、実行前後のコード版を確認する。"""
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("src", "scripts", "tests") for p in sorted((ROOT/folder).rglob("*.py"))}


def save(name: str, value: dict) -> None:
    """先行結果を上書きしない。"""
    with (OUT/name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main() -> int:
    """通常のpytestを一度起動し、部品検証の範囲を明記する。"""
    OUT.mkdir(parents=True, exist_ok=False)
    files = [f"tests/test_{name}.py" for name in TEST_NAMES]
    argv = [sys.executable, "-m", "pytest", *files, "-q", "--tb=short",
            f"--junitxml={OUT/'junit.xml'}"]
    before = code_hashes()
    save("START.json", dict(argv=argv, cwd=str(ROOT), files=files, code_sha256=before,
                           owner_pid=os.getpid(), quality_scope="修復対象CPU。全pytest/G3品質合格ではない"))
    with (OUT/"pytest.log").open("xb") as stream:
        result = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    after = code_hashes()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    suites = ET.parse(OUT/"junit.xml").getroot().iter("testsuite")
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.get(key, "0"))
    save("RESULT.json", dict(exit_code=result.returncode, counts=counts, changed_code=changed,
        source_unchanged=not changed, passed=result.returncode == 0 and not changed))
    print(json.dumps(dict(exit_code=result.returncode, counts=counts, changed_code=changed)))
    return result.returncode if not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
