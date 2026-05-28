"""案 R3 改: PuyoColorProfileDB のテスト群。

テスト対象:
  - ColorProfile.normalized_distance (H 循環, std=0 ガード, 重み付き距離)
  - PuyoColorProfileDB.is_puyo_like (保守的 True, プロファイル合致/不一致)
  - PuyoColorProfileDB.from_online_hsv_state (v29.json の stats 形式)
  - PuyoColorProfileDB.save/load roundtrip
  - PuyoColorProfileDB.load_for_video (per-video / global fallback / None)
  - ImageReader + profile_db 統合テスト (フィルタ動作 / None 時の既存挙動)
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.board import (
    COLOR_BLUE,
    COLOR_EMPTY,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_UNKNOWN,
    COLOR_YELLOW,
)
from src.puyo_color_profile import (
    PUYO_PROFILE_MIN_SAMPLES,
    PUYO_PROFILE_REJECT_THRESHOLD,
    PUYO_PROFILE_STD_FLOOR,
    ColorProfile,
    PuyoColorProfileDB,
)


# ============================
# ヘルパー: テスト用プロファイル生成
# ============================

def _make_profile(
    color: int = COLOR_BLUE,
    h_mean: float = 108.0,
    h_std: float = 2.0,
    s_mean: float = 173.0,
    s_std: float = 5.0,
    v_mean: float = 153.0,
    v_std: float = 10.0,
    n_samples: int = 1000,
) -> ColorProfile:
    """テスト用 ColorProfile を生成するヘルパー。"""
    return ColorProfile(
        color=color,
        h_mean=h_mean, h_std=h_std,
        s_mean=s_mean, s_std=s_std,
        v_mean=v_mean, v_std=v_std,
        n_samples=n_samples,
    )


def _make_db(
    profiles: dict[int, ColorProfile] | None = None,
    video_id: str | None = None,
) -> PuyoColorProfileDB:
    """テスト用 PuyoColorProfileDB を生成するヘルパー。"""
    return PuyoColorProfileDB(
        profiles=profiles or {},
        video_id=video_id,
    )


# ============================
# ColorProfile.normalized_distance のテスト
# ============================

class TestColorProfileDistance:
    """ColorProfile.normalized_distance のテスト群。"""

    def test_distance_zero_at_mean(self) -> None:
        """平均値の距離は 0。"""
        p = _make_profile(h_mean=108.0, s_mean=173.0, v_mean=153.0)
        dist = p.normalized_distance(h=108.0, s=173.0, v=153.0)
        assert dist == pytest.approx(0.0, abs=1e-9)

    def test_h_circular_distance_zero_180(self) -> None:
        """H=0 と H=180 は循環対称: |0 - 180| = 180, min(180, 0) = 0 → 距離 0。"""
        # h_mean=0, 入力 h=180: dh = min(180, 180-180) = min(180,0) = 0
        p = _make_profile(
            h_mean=0.0, h_std=1.0,
            s_mean=200.0, s_std=1.0,
            v_mean=200.0, v_std=1.0,
        )
        dist = p.normalized_distance(h=180.0, s=200.0, v=200.0)
        assert dist == pytest.approx(0.0, abs=1e-9)

    def test_h_circular_distance_small_diff(self) -> None:
        """H=2 と H=178 の循環距離は 4 (直線 176 より短い)。"""
        p = _make_profile(
            h_mean=2.0, h_std=2.0,
            s_mean=200.0, s_std=1.0,
            v_mean=200.0, v_std=1.0,
        )
        # dh = min(|178-2|, 180-176) = min(176, 4) = 4
        # 正規化 dh = 4/2 = 2.0 → H 寄与のみ
        dist = p.normalized_distance(h=178.0, s=200.0, v=200.0)
        expected_h_contrib = (4.0 / 2.0) ** 2  # = 4.0
        expected = math.sqrt(expected_h_contrib)
        assert dist == pytest.approx(expected, rel=1e-6)

    def test_normalized_distance_within_std(self) -> None:
        """std 範囲内の値は距離 < sqrt(H_W + S_W + V_W) (= sqrt(2.5) ≈ 1.58)。"""
        p = _make_profile(
            h_mean=108.0, h_std=2.0,
            s_mean=173.0, s_std=5.0,
            v_mean=153.0, v_std=10.0,
        )
        # 各軸 1std 以内: 距離 < sqrt(1+1+0.25) = sqrt(2.25) = 1.5
        dist = p.normalized_distance(h=109.0, s=175.0, v=158.0)
        assert dist < math.sqrt(2.5)

    def test_normalized_distance_far(self) -> None:
        """数倍 std 離れた値は REJECT_THRESHOLD を超える。"""
        p = _make_profile(
            h_mean=108.0, h_std=2.0,
            s_mean=173.0, s_std=5.0,
            v_mean=153.0, v_std=10.0,
        )
        # H が 20std 離れた場合
        dist = p.normalized_distance(h=150.0, s=173.0, v=153.0)
        assert dist > PUYO_PROFILE_REJECT_THRESHOLD

    def test_std_zero_guard_no_division_by_zero(self) -> None:
        """std=0 でも division by zero が起きないこと。"""
        p = _make_profile(h_std=0.0, s_std=0.0, v_std=0.0)
        # 例外が起きなければ OK
        dist = p.normalized_distance(h=108.0, s=173.0, v=153.0)
        assert isinstance(dist, float)
        assert math.isfinite(dist)

    def test_std_zero_uses_floor(self) -> None:
        """std=0 のとき PUYO_PROFILE_STD_FLOOR (=1.0) を使って距離を計算する。"""
        p = _make_profile(
            h_mean=108.0, h_std=0.0,
            s_mean=173.0, s_std=0.0,
            v_mean=153.0, v_std=0.0,
        )
        # dh = |110 - 108| = 2, floor=1.0 → 正規化 dh = 2.0
        dist_actual = p.normalized_distance(h=110.0, s=173.0, v=153.0)
        expected = math.sqrt(
            1.0 * (2.0 / PUYO_PROFILE_STD_FLOOR) ** 2
            + 1.0 * (0.0 / PUYO_PROFILE_STD_FLOOR) ** 2
            + 0.5 * (0.0 / PUYO_PROFILE_STD_FLOOR) ** 2
        )
        assert dist_actual == pytest.approx(expected, rel=1e-6)


# ============================
# PuyoColorProfileDB.is_puyo_like のテスト
# ============================

class TestIsPuyoLike:
    """PuyoColorProfileDB.is_puyo_like のテスト群。"""

    def test_no_profile_returns_true(self) -> None:
        """プロファイルが存在しない色 → 保守的 True。"""
        db = _make_db(profiles={})
        assert db.is_puyo_like(COLOR_RED, h=7.0, s=176.0, v=173.0) is True

    def test_low_samples_returns_true(self) -> None:
        """n_samples < PUYO_PROFILE_MIN_SAMPLES → 保守的 True。"""
        p = _make_profile(n_samples=PUYO_PROFILE_MIN_SAMPLES - 1)
        db = _make_db(profiles={COLOR_BLUE: p})
        # サンプル不足 → True (EMPTY 化しない)
        assert db.is_puyo_like(COLOR_BLUE, h=200.0, s=1.0, v=1.0) is True

    def test_profile_match_returns_true(self) -> None:
        """プロファイル内の値 → True。"""
        p = _make_profile(
            h_mean=108.0, h_std=2.0,
            s_mean=173.0, s_std=5.0,
            v_mean=153.0, v_std=10.0,
        )
        db = _make_db(profiles={COLOR_BLUE: p})
        assert db.is_puyo_like(COLOR_BLUE, h=108.0, s=173.0, v=153.0) is True

    def test_profile_mismatch_returns_false(self) -> None:
        """プロファイルと大きく離れた値 → False (EMPTY 化)。"""
        p = _make_profile(
            color=COLOR_BLUE,
            h_mean=108.0, h_std=2.0,
            s_mean=173.0, s_std=5.0,
            v_mean=153.0, v_std=10.0,
            n_samples=500,
        )
        db = _make_db(profiles={COLOR_BLUE: p})
        # H=50 は青プロファイル (H=108) から大きく離れている
        assert db.is_puyo_like(COLOR_BLUE, h=50.0, s=173.0, v=153.0) is False

    def test_empty_color_not_in_db_returns_true(self) -> None:
        """db に COLOR_EMPTY がない → 保守的 True (EMPTY はそもそも通過させない前提)。"""
        p = _make_profile(color=COLOR_BLUE)
        db = _make_db(profiles={COLOR_BLUE: p})
        assert db.is_puyo_like(COLOR_EMPTY, h=0.0, s=0.0, v=0.0) is True


# ============================
# PuyoColorProfileDB.from_online_hsv_state のテスト
# ============================

class TestFromOnlineHsvState:
    """PuyoColorProfileDB.from_online_hsv_state のテスト群。"""

    # v29.json の online_hsv_state.stats から抜粋した実数値
    V29_STATS = {
        "1": {
            "h_mean": 7.378333112743375, "h_var": 21.76768643690564,
            "s_mean": 175.9002132129021, "s_var": 39.37314103475506,
            "v_mean": 173.3579373949467, "v_var": 36.484082593395314,
            "n": 3316,
        },
        "2": {
            "h_mean": 109.24921794288899, "h_var": 0.4864731959337504,
            "s_mean": 172.99859325868837, "s_var": 989.8609941660284,
            "v_mean": 153.5074357399815, "v_var": 3836.281814541579,
            "n": 13383,
        },
        "3": {
            "h_mean": 57.56136254670861, "h_var": 0.7386133331679986,
            "s_mean": 197.49947134946805, "s_var": 3.5023293657980297,
            "v_mean": 194.37905823285752, "v_var": 16.12639132957506,
            "n": 3067,
        },
    }

    def test_from_online_hsv_state_v29_blue(self) -> None:
        """v29.json の stats から青 h_mean ≈ 109.25 で構築できる。"""
        db = PuyoColorProfileDB.from_online_hsv_state(
            self.V29_STATS, video_id="v29",
        )
        assert COLOR_BLUE in db.profiles
        assert db.profiles[COLOR_BLUE].h_mean == pytest.approx(109.249, rel=1e-3)
        assert db.video_id == "v29"

    def test_from_online_hsv_state_h_std_from_var(self) -> None:
        """h_std = sqrt(h_var) が正しく計算される。"""
        db = PuyoColorProfileDB.from_online_hsv_state(
            self.V29_STATS, video_id="v29",
        )
        blue = db.profiles[COLOR_BLUE]
        expected_h_std = math.sqrt(0.4864731959337504)
        assert blue.h_std == pytest.approx(expected_h_std, rel=1e-6)

    def test_from_online_hsv_state_skips_low_samples(self) -> None:
        """n < PUYO_PROFILE_MIN_SAMPLES の色は除外される。"""
        stats = {
            "2": {
                "h_mean": 109.0, "h_var": 1.0,
                "s_mean": 173.0, "s_var": 1.0,
                "v_mean": 153.0, "v_var": 1.0,
                "n": PUYO_PROFILE_MIN_SAMPLES - 1,  # 不足
            },
            "1": {
                "h_mean": 7.0, "h_var": 20.0,
                "s_mean": 175.0, "s_var": 40.0,
                "v_mean": 173.0, "v_var": 36.0,
                "n": PUYO_PROFILE_MIN_SAMPLES,  # 丁度 OK
            },
        }
        db = PuyoColorProfileDB.from_online_hsv_state(stats)
        assert COLOR_BLUE not in db.profiles  # 除外
        assert COLOR_RED in db.profiles       # 採用

    def test_from_online_hsv_state_all_skipped_empty_db(self) -> None:
        """全色サンプル不足の場合は空 DB。"""
        stats = {
            "2": {
                "h_mean": 109.0, "h_var": 1.0,
                "s_mean": 173.0, "s_var": 1.0,
                "v_mean": 153.0, "v_var": 1.0,
                "n": 0,
            },
        }
        db = PuyoColorProfileDB.from_online_hsv_state(stats)
        assert db.profiles == {}

    def test_from_online_hsv_state_n_samples_correct(self) -> None:
        """n_samples が stats の n と一致する。"""
        db = PuyoColorProfileDB.from_online_hsv_state(self.V29_STATS)
        assert db.profiles[COLOR_RED].n_samples == 3316
        assert db.profiles[COLOR_BLUE].n_samples == 13383


# ============================
# PuyoColorProfileDB.save / load roundtrip のテスト
# ============================

class TestSaveLoadRoundtrip:
    """save → load で値が一致することを検証するテスト群。"""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """save → load で ColorProfile の全フィールドが一致する。"""
        profiles = {
            COLOR_BLUE: _make_profile(color=COLOR_BLUE, h_mean=108.5),
            COLOR_RED: _make_profile(color=COLOR_RED, h_mean=7.4),
        }
        db = PuyoColorProfileDB(profiles=profiles, video_id="v29")
        npz_path = tmp_path / "test.npz"
        db.save(npz_path)
        loaded = PuyoColorProfileDB.load(npz_path)
        assert loaded.video_id == "v29"
        assert set(loaded.profiles.keys()) == {COLOR_BLUE, COLOR_RED}
        bp = loaded.profiles[COLOR_BLUE]
        assert bp.h_mean == pytest.approx(108.5, rel=1e-6)
        assert bp.n_samples == 1000

    def test_save_empty_db_and_load(self, tmp_path: Path) -> None:
        """空 DB を save → load しても profiles が空。"""
        db = PuyoColorProfileDB(profiles={}, video_id=None)
        npz_path = tmp_path / "empty.npz"
        db.save(npz_path)
        loaded = PuyoColorProfileDB.load(npz_path)
        assert loaded.profiles == {}

    def test_save_video_id_none_loads_none(self, tmp_path: Path) -> None:
        """video_id=None を保存 → load で None (空文字列は None に変換)。"""
        db = PuyoColorProfileDB(profiles={}, video_id=None)
        npz_path = tmp_path / "none_vid.npz"
        db.save(npz_path)
        loaded = PuyoColorProfileDB.load(npz_path)
        assert loaded.video_id is None


# ============================
# PuyoColorProfileDB.load_for_video のテスト
# ============================

class TestLoadForVideo:
    """load_for_video の優先順位テスト群。"""

    def _make_and_save(
        self, path: Path, video_id: str | None = None,
    ) -> PuyoColorProfileDB:
        """テスト用 DB を作成して保存する。"""
        p = _make_profile(color=COLOR_BLUE)
        db = PuyoColorProfileDB(
            profiles={COLOR_BLUE: p}, video_id=video_id,
        )
        db.save(path)
        return db

    def test_per_video_exists_loads_it(self, tmp_path: Path) -> None:
        """per-video npz がある場合はそれをロードする。"""
        per_video_path = tmp_path / "v29_profile.npz"
        self._make_and_save(per_video_path, video_id="v29")
        db = PuyoColorProfileDB.load_for_video("v29", profile_dir=tmp_path)
        assert db.video_id == "v29"
        assert COLOR_BLUE in db.profiles

    def test_fallback_global_when_no_per_video(self, tmp_path: Path) -> None:
        """per-video npz がない場合は global_profile.npz にフォールバック。"""
        global_path = tmp_path / "global_profile.npz"
        self._make_and_save(global_path, video_id=None)
        db = PuyoColorProfileDB.load_for_video("v99", profile_dir=tmp_path)
        assert COLOR_BLUE in db.profiles

    def test_returns_empty_db_when_no_files(self, tmp_path: Path) -> None:
        """per-video も global もない場合は空 DB (保守的動作)。"""
        db = PuyoColorProfileDB.load_for_video("v29", profile_dir=tmp_path)
        assert db.profiles == {}

    def test_video_id_none_uses_global(self, tmp_path: Path) -> None:
        """video_id=None の場合は global_profile.npz を直接ロード。"""
        global_path = tmp_path / "global_profile.npz"
        self._make_and_save(global_path, video_id=None)
        db = PuyoColorProfileDB.load_for_video(None, profile_dir=tmp_path)
        assert COLOR_BLUE in db.profiles

    def test_per_video_takes_priority_over_global(self, tmp_path: Path) -> None:
        """per-video と global 両方ある場合は per-video 優先。"""
        per_video_path = tmp_path / "v29_profile.npz"
        global_path = tmp_path / "global_profile.npz"
        # per-video: h_mean=108 / global: h_mean=50 (違う値)
        p_pv = _make_profile(color=COLOR_BLUE, h_mean=108.0)
        db_pv = PuyoColorProfileDB(profiles={COLOR_BLUE: p_pv}, video_id="v29")
        db_pv.save(per_video_path)
        p_gl = _make_profile(color=COLOR_BLUE, h_mean=50.0)
        db_gl = PuyoColorProfileDB(profiles={COLOR_BLUE: p_gl}, video_id=None)
        db_gl.save(global_path)
        loaded = PuyoColorProfileDB.load_for_video("v29", profile_dir=tmp_path)
        # per-video の h_mean=108 が採用される
        assert loaded.profiles[COLOR_BLUE].h_mean == pytest.approx(108.0, rel=1e-6)


# ============================
# ImageReader + profile_db 統合テスト
# ============================

class TestImageReaderWithProfileDB:
    """ImageReader に profile_db を注入した場合の統合テスト群。"""

    def _make_bgr_patch(
        self, h: int, s: int, v: int, size: int = 16,
    ) -> np.ndarray:
        """HSV を BGRパッチに変換する (ImageReader に渡すテスト用)。"""
        hsv = np.full((size, size, 3), [h, s, v], dtype=np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def test_profile_db_filters_classify_to_empty(self) -> None:
        """profile_db が青と判定したが実際に全く違う HSV → EMPTY 化される。"""
        from src.image_reader import ImageReader
        from src.board import COLOR_BLUE, COLOR_EMPTY

        # 青プロファイル: h_mean=108, h_std=2 (非常に tight)
        p = ColorProfile(
            color=COLOR_BLUE,
            h_mean=108.0, h_std=2.0,
            s_mean=173.0, s_std=5.0,
            v_mean=153.0, v_std=10.0,
            n_samples=500,
        )
        db = PuyoColorProfileDB(profiles={COLOR_BLUE: p}, video_id="test")
        reader = ImageReader(puyo_profile_db=db)

        # H=50 (緑寄り) のパッチ → _apply_profile_filter が EMPTY 化する
        bgr_patch = self._make_bgr_patch(h=50, s=200, v=200)
        # is_puyo_like(COLOR_BLUE, h=50, ...) は False (距離 >> 3.5)
        result = reader._apply_profile_filter(COLOR_BLUE, None, bgr_patch)
        assert result == COLOR_EMPTY

    def test_profile_db_passes_matching_color(self) -> None:
        """プロファイル合致色は EMPTY 化されない。"""
        from src.image_reader import ImageReader
        from src.board import COLOR_BLUE

        p = ColorProfile(
            color=COLOR_BLUE,
            h_mean=108.0, h_std=3.0,
            s_mean=173.0, s_std=10.0,
            v_mean=153.0, v_std=20.0,
            n_samples=500,
        )
        db = PuyoColorProfileDB(profiles={COLOR_BLUE: p}, video_id="test")
        reader = ImageReader(puyo_profile_db=db)

        # H=108 (青) パッチ → プロファイル合致
        bgr_patch = self._make_bgr_patch(h=108, s=173, v=153)
        result = reader._apply_profile_filter(COLOR_BLUE, None, bgr_patch)
        assert result == COLOR_BLUE

    def test_profile_db_none_unchanged(self) -> None:
        """profile_db=None (デフォルト) なら既存挙動と同一 (色変更なし)。"""
        from src.image_reader import ImageReader
        from src.board import COLOR_BLUE

        reader = ImageReader()  # profile_db=None がデフォルト
        bgr_patch = self._make_bgr_patch(h=50, s=200, v=200)
        # DB なし → 変更なし
        result = reader._apply_profile_filter(COLOR_BLUE, None, bgr_patch)
        assert result == COLOR_BLUE

    def test_empty_color_not_filtered(self) -> None:
        """COLOR_EMPTY はプロファイルチェックをスキップして通過する。"""
        from src.image_reader import ImageReader
        from src.board import COLOR_EMPTY

        p = _make_profile(color=COLOR_BLUE)
        db = PuyoColorProfileDB(profiles={COLOR_BLUE: p}, video_id="test")
        reader = ImageReader(puyo_profile_db=db)
        # EMPTY は変更しない
        bgr_patch = self._make_bgr_patch(h=108, s=173, v=153)
        result = reader._apply_profile_filter(COLOR_EMPTY, None, bgr_patch)
        assert result == COLOR_EMPTY

    def test_unknown_color_not_filtered(self) -> None:
        """COLOR_UNKNOWN もプロファイルチェックをスキップして通過する。"""
        from src.image_reader import ImageReader
        from src.board import COLOR_UNKNOWN

        p = _make_profile(color=COLOR_BLUE)
        db = PuyoColorProfileDB(profiles={COLOR_BLUE: p}, video_id="test")
        reader = ImageReader(puyo_profile_db=db)
        bgr_patch = self._make_bgr_patch(h=108, s=173, v=153)
        result = reader._apply_profile_filter(COLOR_UNKNOWN, None, bgr_patch)
        assert result == COLOR_UNKNOWN

    def test_set_puyo_profile_db_updates_db(self) -> None:
        """set_puyo_profile_db で DB を更新できる。"""
        from src.image_reader import ImageReader
        from src.board import COLOR_BLUE

        reader = ImageReader()
        assert reader._puyo_profile_db is None
        p = _make_profile(color=COLOR_BLUE)
        db = PuyoColorProfileDB(profiles={COLOR_BLUE: p}, video_id="v29")
        reader.set_puyo_profile_db(db)
        assert reader._puyo_profile_db is db

    def test_set_puyo_profile_db_to_none(self) -> None:
        """set_puyo_profile_db(None) で無効化できる。"""
        from src.image_reader import ImageReader

        p = _make_profile(color=COLOR_BLUE)
        db = PuyoColorProfileDB(profiles={COLOR_BLUE: p}, video_id="v29")
        reader = ImageReader(puyo_profile_db=db)
        assert reader._puyo_profile_db is not None
        reader.set_puyo_profile_db(None)
        assert reader._puyo_profile_db is None

    def test_use_highlight_override_default_is_false(self) -> None:
        """案 P2 撤回: use_highlight_override のデフォルトが False。"""
        from src.image_reader import ImageReader

        reader = ImageReader()
        assert reader._use_highlight_override is False

    def test_apply_profile_filter_with_hsv_patch(self) -> None:
        """hsv_patch が渡された場合に再計算せず hsv_patch を使う。"""
        from src.image_reader import ImageReader
        from src.board import COLOR_BLUE

        # 青プロファイル tight
        p = ColorProfile(
            color=COLOR_BLUE,
            h_mean=108.0, h_std=1.0,
            s_mean=173.0, s_std=1.0,
            v_mean=153.0, v_std=1.0,
            n_samples=500,
        )
        db = PuyoColorProfileDB(profiles={COLOR_BLUE: p}, video_id="test")
        reader = ImageReader(puyo_profile_db=db)

        # hsv_patch (H=108) を渡す → プロファイル合致 → COLOR_BLUE のまま
        hsv_patch = np.full((8, 8, 3), [108, 173, 153], dtype=np.uint8)
        bgr_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)
        result = reader._apply_profile_filter(COLOR_BLUE, hsv_patch, bgr_patch)
        assert result == COLOR_BLUE
