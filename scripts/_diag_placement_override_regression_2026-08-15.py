"""placement_override(c1)単体が持ち込む新規劣化セルを列挙する診断スクリプト (計装のみ、修正なし)。

score_a.json (baseline) と score_c1.json (a+placement_override) を突合し、
「aでは正解だったがc1では誤り」になったセルを全部出す。
sheet_id -> video_id/side/frame相当のメタも一緒に出す (実画面切り出しの入力用)。
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
SCORING_DIR = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14" / "scoring_ablation"
MANIFEST_PATH = _ROOT / "data" / "verify" / "yardstick_v2_2026-08-14" / "manifest.json"


def main() -> None:
    a_rows = json.loads((SCORING_DIR / "score_a.json").read_text(encoding="utf-8"))
    c1_rows = json.loads((SCORING_DIR / "score_c1.json").read_text(encoding="utf-8"))
    manifest = {e["sheet_id"]: e for e in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))}

    a_by_sheet = {r["sheet_id"]: r for r in a_rows}
    c1_by_sheet = {r["sheet_id"]: r for r in c1_rows}
    common = set(a_by_sheet) & set(c1_by_sheet)

    regressions = []
    for sid in sorted(common):
        a_cells = {(c["r"], c["c"]): c for c in a_by_sheet[sid].get("cells", [])}
        c1_cells = {(c["r"], c["c"]): c for c in c1_by_sheet[sid].get("cells", [])}
        for key, ac in a_cells.items():
            cc = c1_cells.get(key)
            if cc is None:
                continue
            if ac["is_correct"] and not cc["is_correct"]:
                ent = manifest.get(sid, {})
                regressions.append({
                    "sheet_id": sid,
                    "video_id": a_by_sheet[sid]["video_id"],
                    "side": a_by_sheet[sid]["side"],
                    "phase": a_by_sheet[sid]["phase"],
                    "frame_idx": ent.get("frame_idx"),
                    "t_sec": ent.get("t_sec"),
                    "r": key[0], "c": key[1],
                    "correct": ac["correct"],
                    "a_pred": ac["pred"],
                    "c1_pred": cc["pred"],
                })

    print(f"[info] 新規劣化セル総数: {len(regressions)}")
    by_video: dict[str, int] = {}
    for reg in regressions:
        by_video[reg["video_id"]] = by_video.get(reg["video_id"], 0) + 1
    print(f"[info] 動画別内訳: {by_video}")
    print()
    for reg in regressions:
        print(f"  sheet={reg['sheet_id']:<28} video={reg['video_id']:<8} side={reg['side']} "
              f"phase={reg['phase']:<6} frame_idx={reg['frame_idx']} t_sec={reg['t_sec']:.2f} "
              f"r{reg['r']}c{reg['c']}: correct={reg['correct']} a_pred={reg['a_pred']} c1_pred={reg['c1_pred']}")

    out_path = SCORING_DIR / "_diag_c1_new_regressions_2026-08-15.json"
    out_path.write_text(json.dumps(regressions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] -> {out_path}")


if __name__ == "__main__":
    main()
