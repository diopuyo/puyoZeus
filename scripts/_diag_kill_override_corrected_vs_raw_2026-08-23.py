"""dump に保存される pending_p1/room1 (raw/是正前) と、実際に
kill_override() へ渡された是正後の kpending1/kroom1 (`_kill_override_
chain_completion_inputs` の出力) を突き合わせる計装 (2026-08-23)。

目的: scripts/_compare_kill_override_fix_episodes_2026-08-22.py の
kill_g() 再計算が dump の raw pending_p1/room1 を使っているため、
enable_kill_override_chain_completion=True (根治①フラグ) 時に
本番が実際に使った是正後の値と乖離している疑いを検証する
(visualize_advantage_overlay.py:5202-5203 は snap.pending_p1 / 生の
room1,room2 を dump に書き、kill_override() 自体には別変数 kpending1/
kpending2/kroom1/kroom2 が渡る。両者は enable_kill_override_chain_
completion=True の間フレームごとに乖離しうる)。

コードは変更しない。scripts/visualize_advantage_overlay.py の
kill_override / _kill_override_chain_completion_inputs をモンキー
パッチして入出力を記録するだけの外部計装ラッパー。

使い方:
  python scripts/_diag_kill_override_corrected_vs_raw_2026-08-23.py \
    --start-sec 4379.5 --end-sec 4625 --out logs/xxx.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/frames/video_zenchi_c0BQoMJwwQU.mp4")
    ap.add_argument("--model-dir", default="data/verify/retrain_model62_2026-08-21")
    ap.add_argument("--start-sec", type=float, required=True)
    ap.add_argument("--end-sec", type=float, required=True)
    ap.add_argument("--warmup-sec", type=float, default=30.0)
    ap.add_argument("--out-tsv", type=Path, required=True)
    ap.add_argument("--out-npz", type=Path, required=True)
    a = ap.parse_args()

    import scripts.visualize_advantage_overlay as vao  # noqa: E402

    # kill_override は dump 行の生成経路以外 (例: 決着先読み hold 系) からも
    # 呼ばれており、呼び出し回数と dump 行数が一致しない (実測: 5922 vs 4845)。
    # そのため「dump 行を書く直前の最後の kill_override 呼び出し」を捕まえる
    # ため、_build_timeline_dump_row 呼び出し時点での最新値をスナップショット
    # する方式にする (同一フレーム内で kill_override -> dump row 構築が
    # 直列に呼ばれる実装、visualize_advantage_overlay.py:5159,5199-5206 に依存)。
    latest_call: dict = {}
    records: list[dict] = []
    orig_kill_override = vao.kill_override
    orig_build_row = vao._build_timeline_dump_row

    def patched_kill_override(adv, inc1, inc2, room1, room2):
        ret = orig_kill_override(adv, inc1, inc2, room1, room2)
        latest_call.update(
            adv_before=adv, kpending1=inc1, kpending2=inc2,
            kroom1=room1, kroom2=room2, adv_after=ret,
        )
        return ret

    def patched_build_row(*args, **kwargs):
        records.append(dict(latest_call))
        return orig_build_row(*args, **kwargs)

    vao.kill_override = patched_kill_override
    vao._build_timeline_dump_row = patched_build_row

    # main() が argparse するための sys.argv を、本番と全く同じフラグ構成
    # (scripts/_rescan_zenchi_slide_exit_guard_v2_2026-08-22.sh の $FLAGS) で
    # 組み立てる。時間窓だけ縮小して高速化する (因果的な処理のため過去フレーム
    # の結果は end_sec を縮めても不変)。
    adopted = [
        "--early-fire-reaction", "--per-side-settled", "--no-score-lead-bias",
        "--no-pressure", "--sample-interval", "0", "--counter-reach",
        "--normalize-fps-30", "--production-recognition", "--resize-1080p",
        "--resolved-live-defender-strict", "--resolved-kill-override",
    ]
    argv = [
        "visualize_advantage_overlay.py",
        "--video", a.video,
        "--start-sec", str(a.start_sec),
        "--end-sec", str(a.end_sec),
        "--layout", "panel", "--panel-subtitle-h", "0",
        "--no-force-in-match", "--no-render",
        "--model-dir", a.model_dir,
        "--warmup-sec", str(a.warmup_sec),
        "--resolved-exchange-eval", "--resolved-decisive-amplify",
        "--resolved-live-defender",
        "--kill-override-chain-completion",
        "--enable-slide-exit-min-display-guard",
        *adopted,
        "--dump-timeline", str(a.out_npz),
        "--out", str(a.out_npz.with_suffix(".mp4")),
    ]
    print("[argv]", " ".join(argv))
    sys.argv = argv
    vao.main()

    print(f"[記録件数] kill_override 呼び出し回数={len(records)}")

    import numpy as np
    d = np.load(a.out_npz, allow_pickle=True)
    n = len(d["t_sec"])
    print(f"[dump行数] {n}")
    if len(records) != n:
        print(f"[警告] kill_override呼び出し回数({len(records)}) != dump行数({n})"
              " -- 1:1対応でない可能性 (要確認)")

    with open(a.out_tsv, "w", encoding="utf-8") as f:
        header = ("t_sec\traw_pending_p1\traw_pending_p2\traw_room1\traw_room2\t"
                   "corrected_kpending1\tcorrected_kpending2\t"
                   "corrected_kroom1\tcorrected_kroom2\t"
                   "adv_before_override\tadv_after_override\tadv_raw_dump\tadv_ema_dump\t"
                   "state1\tstate2\n")
        f.write(header)
        m = min(len(records), n)
        for i in range(m):
            rec = records[i]
            f.write(
                f"{float(d['t_sec'][i]):.3f}\t{int(d['pending_p1'][i])}\t"
                f"{int(d['pending_p2'][i])}\t{int(d['room1'][i])}\t{int(d['room2'][i])}\t"
                f"{rec['kpending1']:.1f}\t{rec['kpending2']:.1f}\t"
                f"{rec['kroom1']:.1f}\t{rec['kroom2']:.1f}\t"
                f"{rec['adv_before']:.2f}\t{rec['adv_after']:.2f}\t"
                f"{float(d['adv_raw'][i]):.2f}\t{float(d['adv_ema'][i]):.2f}\t"
                f"{d['state1'][i]}\t{d['state2'][i]}\n"
            )
    print(f"[出力] {a.out_tsv}")


if __name__ == "__main__":
    main()
