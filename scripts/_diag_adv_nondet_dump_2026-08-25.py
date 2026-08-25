"""有利不利スコア非決定性の計装付き dump 生成 (本体コード無変更・monkeypatchのみ)。

`scripts/_diag_gate3r6_planA_dump_2026-08-25.py` の OFF モードと**完全同一の
動画・区間・フラグ構成** (zenchi 先頭5試合 t=0-420、model62、本番採用フラグ) で
timeline dump を生成しつつ、HeavyAdvCache のリフレッシュ1回ごとに:

  - _score_advantage への入力 (b1/b2 の grid crc32、会計 snap の全フィールド)
  - _side_feats_full_base の戻り dict (全指標値、float.hex() で完全精度)
  - 出力 (adv, p1, drivers)

を JSONL に記録する。2回走らせて JSONL を突合すれば「入力が同じで出力が違う
リフレッシュ」と「どの指標列が揺れたか」を1回分で特定できる。

使い方:
  python scripts/_diag_adv_nondet_dump_2026-08-25.py --tag r1
    -> data/verify/adv_nondeterminism_2026-08-25/dump_r1.npz / trace_r1.jsonl
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402

OUT_DIR = Path("data/verify/adv_nondeterminism_2026-08-25")


def _hexf(v: object) -> str:
    """float を完全精度の hex 表現へ (str/int はそのまま文字列化)。"""
    try:
        return float(v).hex()
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--end-sec", default="420",
                    help="解析終了秒 (既定420=元diagと同一。不一致窓はt=112-260"
                         "に集中するため280で短縮可、因果的に前方の行は不変)")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump_path = OUT_DIR / f"dump_{a.tag}.npz"
    trace_path = OUT_DIR / f"trace_{a.tag}.jsonl"
    log_f = open(trace_path, "w", encoding="utf-8")

    # ---- 計装: monkeypatch (本体コードは無変更) ----
    orig_score = vao._score_advantage
    orig_base = vao._side_feats_full_base
    call = {"n": 0, "side": 0}

    def base_wrap(board):
        row = orig_base(board)
        call["side"] += 1
        rec = {
            "kind": "base", "n": call["n"], "seq": call["side"],
            "crc": zlib.crc32(board._grid.tobytes()),
            "row": {k: _hexf(v) for k, v in row.items()},
        }
        log_f.write(json.dumps(rec) + "\n")
        return row

    def score_wrap(model, b1, b2, snap, *args, **kw):
        call["n"] += 1
        adv, p1, drivers = orig_score(model, b1, b2, snap, *args, **kw)
        # snap は dataclass / NamedTuple / 通常オブジェクトのどれでも拾う
        if hasattr(snap, "__dataclass_fields__"):
            fields = list(snap.__dataclass_fields__)
        elif hasattr(snap, "_asdict"):
            fields = list(snap._asdict())
        else:
            fields = [k for k in vars(snap) if not k.startswith("__")]
        snap_d = {f: _hexf(getattr(snap, f)) for f in fields}
        rec = {
            "kind": "score", "n": call["n"],
            "crc1": zlib.crc32(b1._grid.tobytes()),
            "crc2": zlib.crc32(b2._grid.tobytes()),
            # 単体再現用: 盤面グリッド本体 (base64、6x13 int8/int64 いずれでも可)
            "grid1": base64.b64encode(b1._grid.tobytes()).decode(),
            "grid2": base64.b64encode(b2._grid.tobytes()).decode(),
            "grid_dtype": str(b1._grid.dtype), "grid_shape": list(b1._grid.shape),
            "snap": snap_d,
            "adv": _hexf(adv), "p1": _hexf(p1),
            "drivers": [[name, _hexf(v)] for name, v in drivers],
        }
        log_f.write(json.dumps(rec) + "\n")
        return adv, p1, drivers

    vao._score_advantage = score_wrap
    vao._side_feats_full_base = base_wrap

    # 実行環境の記録 (PYTHONHASHSEED 等)
    env_rec = {
        "kind": "env", "tag": a.tag,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "hash_of_test_str": hash("determinism_probe"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    log_f.write(json.dumps(env_rec) + "\n")

    argv_backup = sys.argv[:]
    try:
        # _diag_gate3r6_planA_dump_2026-08-25.py --mode off と完全同一の引数列
        sys.argv = [
            "visualize_advantage_overlay.py",
            "--video", "data/frames/video_zenchi_c0BQoMJwwQU.mp4",
            "--start-sec", "0",
            "--end-sec", str(a.end_sec),
            "--layout", "panel", "--panel-subtitle-h", "0",
            "--no-force-in-match", "--no-render",
            "--dump-timeline", str(dump_path),
            "--model-dir", "data/verify/retrain_model62_2026-08-21",
            "--warmup-sec", "0",
            "--kill-override-chain-completion",
            "--enable-slide-exit-min-display-guard",
        ]
        import src.production_config as pc
        adopted = pc.advantage_overlay_flags()
        if adopted:
            sys.argv.extend(adopted.split())
        sys.argv.append("--no-counter-reach")
        vao.main()
    finally:
        sys.argv = argv_backup
        log_f.close()

    print(f"[保存] {dump_path} / {trace_path}")


if __name__ == "__main__":
    main()
