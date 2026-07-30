"""src/fps_normalize.py の単体テスト (2026-07-30)。

resolve_normalize_fps_30_stride は stateless な純粋関数のため、
入出力の組合せのみを網羅的に確認する。
"""
from __future__ import annotations

from src.fps_normalize import (
    MIN_NORMALIZE_STRIDE,
    NORMALIZE_FPS_30_TARGET,
    resolve_normalize_fps_30_stride,
)


def test_30fps_returns_stride_1() -> None:
    """30fps ちょうどは間引きなし (stride=1)。"""
    assert resolve_normalize_fps_30_stride(30.0) == 1


def test_60fps_returns_stride_2() -> None:
    """60fps は 2フレームおき (実効30fps)。"""
    assert resolve_normalize_fps_30_stride(60.0) == 2


def test_ntsc_59_94fps_returns_stride_2() -> None:
    """59.94fps (NTSC ドロップフレーム相当) も四捨五入で stride=2。"""
    assert resolve_normalize_fps_30_stride(59.94) == 2


def test_120fps_returns_stride_4() -> None:
    """120fps は stride=4。"""
    assert resolve_normalize_fps_30_stride(120.0) == 4


def test_below_30fps_clamped_to_min_stride() -> None:
    """24fps 等 30fps 未満は間引かない (stride=1、安全側に丸める)。"""
    assert resolve_normalize_fps_30_stride(24.0) == MIN_NORMALIZE_STRIDE


def test_zero_fps_fallback() -> None:
    """fps=0 (取得失敗) は MIN_NORMALIZE_STRIDE にフォールバックする。"""
    assert resolve_normalize_fps_30_stride(0.0) == MIN_NORMALIZE_STRIDE


def test_negative_fps_fallback() -> None:
    """負の fps (異常値) も安全側にフォールバックする。"""
    assert resolve_normalize_fps_30_stride(-10.0) == MIN_NORMALIZE_STRIDE


def test_target_constant_is_30() -> None:
    """正規化目標が 30fps であること (定数の意図保護、マジックナンバー回帰防止)。"""
    assert NORMALIZE_FPS_30_TARGET == 30.0
