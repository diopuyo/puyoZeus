"""設置確定レイテンシ A/B 実験 追試: 非対称案 (2026-08-08、read-only診断)。

コーディネーター追加依頼 (2026-08-08): 既存実装済・default OFF の
`enable_asymmetric_recovery_min_frames` + `recovery_add_min_frames` を使い、
「空→色のみ3フレーム、色→空/色→色は8維持」の水準を同一80秒クリップで測定する。
`_ab_recovery_min_frames_2026-08-08.py` の一律3水準・現行8水準と横並び比較する。

報告対象は A (発火直前配置の反映成否) と B (STABLE中セル書換数) のみ。
C (ライブラリゲート検査) / D・E (訂正レイテンシ層別・パッチ抽出サンプル拡大) は
今回スコープ外 (コーディネーター指示)。

読み取り専用診断。src/ は一切変更しない。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

cv2.setNumThreads(1)

# 既存 A/B スクリプトをモジュールとして読み込み、pipeline構築・イベント抽出・
# 汚染量計測のロジックを再利用する (ファイル名にハイフンがあるため importlib)。
import importlib.util  # noqa: E402

_AB_PATH = PROJ_ROOT / "scripts" / "_ab_recovery_min_frames_2026-08-08.py"
_spec = importlib.util.spec_from_file_location("_ab_rmf", _AB_PATH)
ab = importlib.util.module_from_spec(_spec)
sys.modules["_ab_rmf"] = ab
_spec.loader.exec_module(ab)  # type: ignore[union-attr]

OUT_PATH = PROJ_ROOT / "data" / "verify" / "recovery_min_frames_ab_2026-08-08" / "result_asymmetric.json"

# 非対称水準: 方向1(空→色)のみ 3 フレームに短縮、方向2/3(色→空/色→色) は
# STABLE_RECOVERY_MIN_FRAMES=8 を維持 (recovery_min_frames は上書きしない=None)。
ASYMMETRIC_EXTRA_KWARGS: dict = dict(
    enable_asymmetric_recovery_min_frames=True,
    recovery_add_min_frames=3,
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    _log("[clip] level=asymmetric_3_8 (空→色=3, 色→空/色→色=8) 走査開始")
    t0 = time.time()
    recs_1p, recs_2p, fps, _pipe = ab._collect_records_standard(
        ab.CLIP_PATH, "m01clip", 0.0, 80.0,
        recovery_min_frames=None,  # 色→空/色→色 は既定8を維持 (無変更)
        extra_kwargs=ASYMMETRIC_EXTRA_KWARGS,
    )
    _log(f"[clip] level=asymmetric_3_8 走査完了 ({time.time() - t0:.1f}s)")

    result: dict = {}
    for side, records in (("1P", recs_1p), ("2P", recs_2p)):
        fire = ab._analyze_fire_detection(records, "m01clip", side, fps)
        pollution = ab._measure_pollution(records)
        result[side] = {"fire": fire, "pollution": pollution}
        _log(
            f"[clip] asymmetric_3_8 {side}: "
            f"instant_chain={fire['n_instant_chain_cases']} "
            f"captured={fire['n_captured_before_chain']} "
            f"pollution_flips={pollution['total_cell_flips']} "
            f"(n_stable_frames={pollution['n_stable_frames']})",
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    _log(f"[DONE] 出力: {OUT_PATH}")


if __name__ == "__main__":
    main()
