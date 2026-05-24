"""viz log から CHAIN state 開始/終了 frame を抽出して画像保存."""
from __future__ import annotations
import argparse, re
from pathlib import Path
import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--viz", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fps", type=float, default=60.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(
        r"\[progress\] (\d+)/\d+ \(.*?\) 1P=(\w+) 2P=(\w+)",
    )
    states: list[tuple[int, str, str]] = []
    with args.log.open() as f:
        for line in f:
            m = pattern.search(line)
            if m:
                states.append((int(m.group(1)), m.group(2), m.group(3)))
    # CHAIN transitions
    chain_starts: list[tuple[int, str]] = []
    prev_p1, prev_p2 = "", ""
    for fi, p1, p2 in states:
        if (p1 == "chain" and prev_p1 != "chain"):
            chain_starts.append((fi, "1P"))
        if (p2 == "chain" and prev_p2 != "chain"):
            chain_starts.append((fi, "2P"))
        prev_p1, prev_p2 = p1, p2
    cap = cv2.VideoCapture(str(args.viz))
    for fi, side in chain_starts[:10]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if ok:
            sec = fi / args.fps
            out_path = args.out_dir / f"chain_{side}_f{fi:05d}_{sec:.1f}s.png"
            cv2.imwrite(str(out_path), fr)
            print(f"saved {out_path}")
    cap.release()


if __name__ == "__main__":
    main()
