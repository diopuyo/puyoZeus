"""
cli.py のテスト

サブコマンドが正しく dispatch され、ファイル入出力が期待通り動作することを検証する。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.cli import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_USAGE,
    main,
)


# ============================
# ヘルパー
# ============================


@pytest.fixture
def black_image(tmp_path: Path) -> Path:
    p = tmp_path / "frame.png"
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.imwrite(str(p), frame)
    return p


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    p = tmp_path / "input.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(p), fourcc, 10.0, (320, 180))
    for _ in range(10):
        writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
    writer.release()
    return p


# ============================
# main() - help / usage
# ============================


class TestMainUsage:
    def test_no_args_prints_help_and_returns_usage(self, capsys):
        code = main([])
        assert code == EXIT_USAGE
        captured = capsys.readouterr()
        assert "puyo-analyzer" in captured.out or "usage" in captured.out.lower()


# ============================
# analyze-frame
# ============================


class TestAnalyzeFrame:
    def test_analyze_frame_prints_json(self, black_image, capsys):
        code = main(["analyze-frame", str(black_image)])
        assert code == EXIT_OK
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "score" in data
        assert "player1" in data

    def test_analyze_frame_writes_output(self, black_image, tmp_path):
        out = tmp_path / "result.json"
        code = main(["analyze-frame", str(black_image), "--output", str(out)])
        assert code == EXIT_OK
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "score" in data

    def test_analyze_frame_missing_file(self, tmp_path, capsys):
        code = main(["analyze-frame", str(tmp_path / "nope.png")])
        assert code == EXIT_ERROR

    def test_analyze_frame_invalid_image(self, tmp_path, capsys):
        bad = tmp_path / "bad.png"
        bad.write_text("this is not an image", encoding="utf-8")
        code = main(["analyze-frame", str(bad)])
        assert code == EXIT_ERROR


# ============================
# composite
# ============================


class TestComposite:
    def test_composite_runs(self, synthetic_video, tmp_path, capsys):
        out = tmp_path / "out.mp4"
        code = main([
            "composite",
            str(synthetic_video), str(out),
            "--interval", "0.5",
            "--no-audio",
        ])
        assert code == EXIT_OK
        assert out.exists()

    def test_composite_missing_input(self, tmp_path, capsys):
        code = main([
            "composite",
            str(tmp_path / "nope.mp4"),
            str(tmp_path / "out.mp4"),
            "--no-audio",
        ])
        assert code == EXIT_ERROR


# ============================
# stream (起動のみ)
# ============================


class TestStream:
    def test_stream_start_and_keyboard_interrupt(
        self, monkeypatch, capsys, tmp_path,
    ):
        """stream サブコマンドが KeyboardInterrupt で綺麗に終了すること。"""
        import src.cli as cli_mod

        # sleep を一発で KeyboardInterrupt させる
        call_count = [0]

        def fake_sleep(sec):
            call_count[0] += 1
            raise KeyboardInterrupt

        monkeypatch.setattr(cli_mod.time, "sleep", fake_sleep)
        # 空きポートを利用
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        code = main(["stream", "--port", str(port)])
        assert code == EXIT_OK
        out = capsys.readouterr().out
        assert "http://" in out
        assert call_count[0] == 1
