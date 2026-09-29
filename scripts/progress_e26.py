"""完了した単独条件だけの測定進捗を表示する。採否は変更しない。"""
from pathlib import Path
from scripts import report_e7_exchange_20260927 as metrics
from scripts.run_e26_ablation import VARIANTS, OUT


def main() -> None:
    """q動画の固定ラベル・同一行を使い、中間結果を読む。"""
    source = 'q_7gc4TgFig'
    windows, _ = metrics.e3.outcomes(source)
    for name in VARIANTS:
        path = OUT/name/'renders'/source/'on/display.npz'
        if path.exists():
            result = metrics.e3.m3_scores(metrics.load_display(path), windows)
            print(name, result, flush=True)


if __name__ == '__main__':
    main()
