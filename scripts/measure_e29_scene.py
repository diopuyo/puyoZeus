"""E29指定場面の表示初到達を、E27と同じEMAで測定する。"""
from pathlib import Path
from scripts.measure_e23_display_20260928 import displayed, first
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.report_e29 import scene


def main() -> None:
    """生確率の前倒しを表示の前倒しと混同しない。"""
    stamp = first(*displayed(Path('logs/e29/on/review/display.npz')))
    value = dict(first_sec=stamp, baseline_first_sec=2771.25, deadline_sec=2766.,
        scene_gate=stamp is not None and stamp <= 2766., scenes=1)
    save_json(Path('logs/e29/SCENE_DISPLAY.json'), value)
    scene()
    print(value, flush=True)


if __name__ == '__main__':
    main()
