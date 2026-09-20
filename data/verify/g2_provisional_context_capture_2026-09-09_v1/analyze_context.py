"""全保存joinと暫定context受理を解析。人工CPUモデルは数値結線の検査のみ。"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import numpy as np
import saved_context as S

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "video38_current_scope_live_2026-09-09_v1"


def write(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)


def analyze(output: Path, registration: dict[str, Any], model: Any) -> dict[str, Any]:
    index = S.indices(output)
    holds, faults, accepted = Counter(), Counter(), []
    count = 0
    for row in S.rows(output / "provisional_context.jsonl"):
        S.saved_join(row, index)
        count += 1
        try:
            bound = S.B.bind_provisional_context(row, registration)
        except S.B.ContextHold as error:
            holds[str(error)] += 1
            continue
        except S.B.ContextFault as error:
            faults[str(error)] += 1
            continue
        result = S.B.evaluate_bound(bound, model)
        inputs = {key: S.B.digest(getattr(bound.inputs, key).tolist()) for key in
                  ("boards", "queues", "ledger_values", "ledger_availability")}
        accepted.append({"frame": row["frame_idx"], "context_digest": bound.context_digest,
            "input_array_sha256": inputs, "supported": bound.supported, "integrity_valid": bound.integrity_valid,
            "reason": bound.reason, "artificial_untrained_model_result": asdict(result)})
    return {"update_count": count, "accepted_count": len(accepted), "accepted": accepted,
        "holds": dict(holds), "faults": dict(faults), "exact_same_run_join": True,
        "trained_model_quality_measured": False, "m1_residual_enabled": False, "quality_gate_clear": False}


def source_guards() -> dict[str, str]:
    paths = [ROOT / name for name in ("analyze_context.py", "saved_context.py", "binding.py")]
    paths += [S.B.BRIDGE, Path(S.B.C.__file__), Path(S.B.C.R.__file__), Path(S.B.C.S.__file__),
              S.B.C.S.P.__file__]
    paths += [Path(module.__file__) for name, module in sys.modules.items()
              if name.startswith("src.") and getattr(module, "__file__", None)]
    return {str(path): S.sha(Path(path)) for path in paths}


def main() -> int:
    input_root, output = Path(sys.argv[1]).resolve(), ROOT / sys.argv[2]
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    import torch
    from src.advantage_m0_current_cnn_v1 import AdvantageM0CurrentCNNV2
    from src.advantage_m1_zero_counterfactual_v3 import AdvantageM1ZeroCounterfactualV3
    before, receipt = S.verify_complete(input_root)
    before.update(source_guards())
    before[str(OLD / "frames.jsonl")] = S.sha(OLD / "frames.jsonl")
    unchanged = S.sha(input_root / "frames.jsonl") == before[str(OLD / "frames.jsonl")]
    S.B.require(unchanged, "original_frames_changed")
    torch.manual_seed(930)
    model = AdvantageM1ZeroCounterfactualV3(AdvantageM0CurrentCNNV2(), "values_and_masks").eval()
    model_hash = S.B.digest({key: value.detach().numpy().tolist() for key, value in model.state_dict().items()})
    report = analyze(input_root, S.registration(input_root, receipt), model)
    report.update(original_frames_sha_equal=unchanged, model_state_sha256=model_hash,
                  model_seed=930, artificial_untrained_model=True, current_publication_unchanged=True)
    write(output / "BINDING_REPORT.json", report)
    after = {path: S.sha(Path(path)) for path in before}
    S.B.require(before == after and not torch.cuda.is_initialized(), "analysis_source_or_device_changed")
    result = {"pid": os.getpid(), "actual_exit": 0, "seconds": time.perf_counter() - started,
        "before": before, "after": after, "saved_run_complete": True, "quality_gate_clear": False}
    write(output / "RESULT.json", result)
    write(output / "ANALYSIS_COMPLETE.json", {"sha256": {name: S.sha(output / name)
        for name in ("BINDING_REPORT.json", "RESULT.json")}})
    print({"actual_exit": 0, "seconds": result["seconds"], "bound": report["accepted_count"],
           "holds": report["holds"], "faults": report["faults"]}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
