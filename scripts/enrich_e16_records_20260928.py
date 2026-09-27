"""既存記録の認識結果を変えず、元フレームから窒息確認信号だけを追記する。"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import cv2
from src.exchange_event_record import read_records, encode
from src.exchange_event_terminal import ObservedDeathDetector
from scripts.run_e3_exchange_eval_20260926 import SOURCES, VIDEO_ROOT, save_json

OUT = Path("logs/e16/records")
ZENCHI_VIDEO = Path("/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/frames/video_zenchi_c0BQoMJwwQU.mp4")
WORKERS, PROGRESS_FRAMES = 3, 3000
FRAME_SIZE = (1920, 1080)


def enrich(job: tuple[str, Path, Path]) -> str:
    """動画単位の完了マーカーがある場合だけチェックポイントを再利用する。"""
    name, record, video = job
    dest, status = OUT / f"{name}.jsonl.gz", OUT / f"{name}.json"
    if dest.exists() and status.exists():
        return name
    cv2.setNumThreads(1)
    detector = ObservedDeathDetector()
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    assert fps > 0
    position, frames, detections = None, 0, []
    with gzip.open(dest.with_suffix(".tmp"), "wt", encoding="utf-8") as stream:
        for row in read_records(record):
            if row["kind"] == "update":
                result, stamp = row["args"][0], row["args"][3]
                index = round(stamp*fps)
                if position is None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    position = index
                while position < index:
                    assert capture.grab()
                    position += 1
                success, frame = capture.read()
                assert success
                position += 1
                if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
                    frame = cv2.resize(frame, FRAME_SIZE)
                result.confirmed_dead_sides = detector.update(frame)
                result.terminal_evidence_available = True
                if result.confirmed_dead_sides:
                    detections.append(dict(t_sec=stamp, sides=result.confirmed_dead_sides))
                frames += 1
                if frames % PROGRESS_FRAMES == 0:
                    print(f"死亡信号 {name} {frames}フレーム", flush=True)
            stream.write(json.dumps(encode(row), ensure_ascii=False, separators=(",", ":"))+"\n")
    capture.release()
    dest.with_suffix(".tmp").replace(dest)
    save_json(status, dict(frames=frames, detections=detections, source=str(record), video=str(video)))
    return name


def main() -> None:
    """既存3動画とzenchi・レビューを同じ検出条件で補完する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, Path("logs/e8/renders")/s/"on/inputs.jsonl.gz",
             VIDEO_ROOT/f"{s}_first_0_900_20260925_v1.mp4") for s in SOURCES]
    jobs += [("zenchi", Path("logs/review_zenchi_part3/on_e10c/inputs.jsonl.gz"), ZENCHI_VIDEO),
             ("review", Path("logs/review_zenchi_g41_43_e14/inputs.jsonl.gz"), ZENCHI_VIDEO)]
    with ProcessPoolExecutor(WORKERS) as pool:
        list(pool.map(enrich, jobs))
    save_json(OUT / "STATUS.json", dict(stage="COMPLETE", sources=[j[0] for j in jobs]))


if __name__ == "__main__":
    main()
