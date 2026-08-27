"""Gate 3R-5: --gross-ledger-dump 既定OFFのbit-identical検証 (2026-08-25)。

既存 `scripts/_diag_adv_nondet_dump_2026-08-25.py` と同一の実行構成
(zenchi先頭区間・model62・本番採用フラグ) で `vao.main()` を3回実行し、
生成された3本の timeline dump npz が全キー・全値で完全一致することを
確認する。フラグを渡さない (既定 False) ため、Gate 3R-5 で追加した
gross_* 列は一切保存されない (`save_timeline_dump` の分岐、
bit-identical要件そのものを直接検査する)。

元スクリプトは書き換えず、本スクリプトを独立コピーとして新規作成した
(タスク指示「元を書き換えず、必要ならコピー」に従う)。

使い方: python scripts/_gate3r5_off_bitidentical_2026-08-25.py
出力: data/verify/gate3r5_off_bitidentical_2026-08-25/dump_r{1,2,3}.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402

OUT_DIR = Path("data/verify/gate3r5_off_bitidentical_2026-08-25")
# 3試合先頭のstate機械・会計を十分駆動できる短窓 (元diagの420秒より短縮、
# 因果的に前方の行は不変なので既定OFFのbit-identical検証には十分)。
END_SEC = "90"


def _run(tag: str) -> Path:
    """既定OFF (--gross-ledger-dump を渡さない) で1本の timeline dump を作る。"""
    dump_path = OUT_DIR / f"dump_{tag}.npz"
    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "visualize_advantage_overlay.py",
            "--video", "data/frames/video_zenchi_c0BQoMJwwQU.mp4",
            "--start-sec", "0", "--end-sec", END_SEC,
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
    return dump_path


def _compare(paths: list[Path]) -> None:
    """3本のnpzが全キー・全値で完全一致するかを母数付きで検証する。"""
    ds = [np.load(str(p), allow_pickle=True) for p in paths]
    key_sets = [set(d.files) for d in ds]
    assert all(ks == key_sets[0] for ks in key_sets), f"キー集合が不一致: {key_sets}"
    gross_keys_present = key_sets[0] & set(vao._TIMELINE_GROSS_KEYS)
    assert not gross_keys_present, (
        f"既定OFFなのにgross列が保存されている: {gross_keys_present}")
    n_keys = len(key_sets[0])
    n_rows = int(ds[0]["t_sec"].shape[0])
    mismatched_keys: list[str] = []
    for key in sorted(key_sets[0]):
        base = ds[0][key]
        for other in ds[1:]:
            if not np.array_equal(base, other[key]):
                mismatched_keys.append(key)
                break
    print(
        f"[Gate 3R-5 OFF bit-identical] 行数={n_rows} "
        f"不一致キー={len(mismatched_keys)}/{n_keys} {mismatched_keys}"
    )
    assert not mismatched_keys, f"{len(mismatched_keys)}/{n_keys} 行不一致: {mismatched_keys}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [_run(tag) for tag in ("r1", "r2", "r3")]
    _compare(paths)
    print("[PASS] 3run 全キー完全一致・gross列は一切保存されない (既定OFF)")


if __name__ == "__main__":
    main()
