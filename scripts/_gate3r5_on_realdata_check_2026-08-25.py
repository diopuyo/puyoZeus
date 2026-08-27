"""Gate 3R-5: --gross-ledger-dump 有効時の実データ配線チェック (2026-08-25)。

`_gate3r5_off_bitidentical_2026-08-25.py` と同一の動画・区間・本番採用フラグに
`--gross-ledger-dump` **だけ**を追加して1回実行し、生成された timeline dump に
gross_* 列が正しく現れること・保存則残差が母数付きで確認できることを見る
(P1-3 のcap前累積カウンタ自体の健全性は Codex が9,000 frame/18,000 sideで
既に検収済みなので、本スクリプトは再検証ではなく**このdump配線**が
壊れていないかの実データ smoke)。

使い方: python scripts/_gate3r5_on_realdata_check_2026-08-25.py
出力: data/verify/gate3r5_on_realdata_check_2026-08-25/dump_on.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as vao  # noqa: E402

OUT_DIR = Path("data/verify/gate3r5_on_realdata_check_2026-08-25")
END_SEC = "90"
_RESIDUAL_EPS = 1e-6  # 浮動小数点比較の許容誤差 (シーン逆算ではない一般的な値)


def _run() -> Path:
    dump_path = OUT_DIR / "dump_on.npz"
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
            "--gross-ledger-dump",
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


def _check(path: Path) -> None:
    d = np.load(str(path), allow_pickle=True)
    assert "gross_inspected_sides" in d.files, "gross_* 列が保存されていない"
    n_rows = int(d["t_sec"].shape[0])
    inspected_sides = d["gross_inspected_sides"].astype(np.int64)
    sides_total = int(inspected_sides.sum())
    residual = np.concatenate([d["gross_residual_p1"], d["gross_residual_p2"]])
    # inspected_sides は行ごとに0か2 (両side同時検査) なので、残差配列
    # (p1全行+p2全行を連結) にも同じ行単位マスクを2倍に伸ばして揃える。
    inspected_mask = inspected_sides > 0
    residual_mask = np.concatenate([inspected_mask, inspected_mask])
    nonzero = int((np.abs(residual[residual_mask]) > _RESIDUAL_EPS).sum())
    wiped_total = int(d["gross_wiped_p1"].sum() + d["gross_wiped_p2"].sum())
    clamp_total = int(d["gross_clamp_loss_p1"].sum() + d["gross_clamp_loss_p2"].sum())
    print(
        f"[Gate 3R-5 ON 実データ] 行数={n_rows} 検査side数={sides_total} "
        f"保存則残差非0={nonzero}/{sides_total} 境界ワイプ量合計={wiped_total} "
        f"clamp_loss合計={clamp_total}"
    )
    assert sides_total > 0, "検査side数が0=一度も検査していない (測定器事故の疑い)"
    assert nonzero == 0, f"保存則残差が非0のsideが{nonzero}/{sides_total}件ある"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _run()
    _check(path)
    print("[PASS] gross_ledger_dump配線は実データで保存則残差0を再現した")


if __name__ == "__main__":
    main()
