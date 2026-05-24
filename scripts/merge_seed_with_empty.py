"""HSV anchor seed に empty/ojama を pseudo store から追加する.

v6 model が empty を忘却したので、 anchor (5 色) に empty/ojama サンプルを
追加して fine-tune の anchor + base 維持 を両立させる。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.self_supervised.label_store import LabelStore
from src.self_supervised.pseudo_label import (
    COMPONENT_CELL,
    PseudoLabelSample,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--anchor-root", type=Path, required=True)
    p.add_argument("--pseudo-root", type=Path,
                    default=Path("data/pseudo_labels"))
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--empty-per-video", type=int, default=2000,
                    help="動画あたり empty サンプル追加数")
    p.add_argument("--ojama-per-video", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    for vid in args.videos:
        anc = LabelStore(video_id=vid, root=args.anchor_root)
        out = LabelStore(video_id=vid, root=args.out_root)
        # 1. anchor 全部コピー
        anc_samples = list(anc.load(COMPONENT_CELL))
        out.append(anc_samples)
        # 2. pseudo から empty/ojama 抽出 (中央付近のみ)
        pseudo = LabelStore(video_id=vid, root=args.pseudo_root)
        empty_buf: list[PseudoLabelSample] = []
        ojama_buf: list[PseudoLabelSample] = []
        for s in pseudo.load(COMPONENT_CELL):
            try: c = int(s.label)
            except: continue
            if c == 0 and len(empty_buf) < args.empty_per_video:
                empty_buf.append(s)
            elif c == 9 and len(ojama_buf) < args.ojama_per_video:
                ojama_buf.append(s)
            if (len(empty_buf) >= args.empty_per_video
                    and len(ojama_buf) >= args.ojama_per_video):
                break
        out.append(empty_buf + ojama_buf)
        print(
            f"[merge] {vid}: anchor={len(anc_samples)} +empty={len(empty_buf)} "
            f"+ojama={len(ojama_buf)} total={len(anc_samples)+len(empty_buf)+len(ojama_buf)}",
        )


if __name__ == "__main__":
    main()
