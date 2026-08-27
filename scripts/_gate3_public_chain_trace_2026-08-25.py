"""既存Q-01本番プローブを変更せず、内部/公開ChainEventを同時採取する。"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT_ROOT / "scripts/_probe_formula_interlude_2026-08-24.py"
OUT = PROJECT_ROOT / "data/verify/gate3_chain_public_trace_2026-08-25/zenchi_w1.json"
T0: float = 780.0
T1: float = 1080.0


def _load_source() -> ModuleType:
    sys.argv = [str(SOURCE), "on", str(T0), str(T1), "codex_public_trace"]
    spec = importlib.util.spec_from_file_location("gate3_public_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"プローブをロードできません: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event_dict(event: Any) -> dict[str, Any] | None:
    if event is None:
        return None
    return {
        "trigger_sec": round(float(event.trigger_sec), 3),
        "mechanism": event.mechanism,
        "chain_count": int(event.chain_count),
        "total_score": int(event.total_score),
    }


def _install_trace(module: ModuleType, rows: list[dict[str, Any]]) -> None:
    original_update = module._orig_update
    previous: dict[str, tuple[Any, Any] | None] = {"1P": None, "2P": None}

    def audited_update(self: Any, frame_idx: int, time_sec: float, frame: Any) -> Any:
        result = original_update(self, frame_idx, time_sec, frame)
        pairs = (
            ("1P", result.p1.chain_event, self._active_chain_1p),
            ("2P", result.p2.chain_event, self._active_chain_2p),
        )
        for side, public, active in pairs:
            key = (_event_dict(public), _event_dict(active))
            comparable = (repr(key[0]), repr(key[1]))
            if previous[side] != comparable:
                rows.append({
                    "t_sec": round(time_sec, 3), "side": side,
                    "public": key[0], "active": key[1],
                })
                previous[side] = comparable
        return result

    module._orig_update = audited_update


def main() -> None:
    module = _load_source()
    rows: list[dict[str, Any]] = []
    _install_trace(module, rows)
    module.main()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {OUT} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
