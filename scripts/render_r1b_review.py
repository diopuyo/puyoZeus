"""全条件合格したR1bだけを、音声付き指定サイズでレビュー化する。"""
from __future__ import annotations
from pathlib import Path
from scripts import render_r1_review as render


def main() -> None:
    """描画・再生一致・音声・容量検収はR1の共通経路を流用する。"""
    render.DEST = render.video.prior.ROOT/'logs/review_zenchi_g41_43_r1b'
    render.MOBILE = Path('/mnt/d/puyo_analyzer/videos/review/zenchi_g41-43_r1b_mobile.mp4')
    render.main(Path('logs/r1b'))


if __name__ == '__main__':
    main()
