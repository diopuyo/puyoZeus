"""案 R3 改: ぷよ色プロファイル DB (per-video 動画別 × 色別 H/S/V 統計)。

online_hsv_state.stats (= OnlineHsvCalibrator の学習済統計) を元に
各色の H/S/V 平均・標準偏差を保持し、classify 後の色に対して
「本当にその色のぷよに見えるか」を判定する。

設計方針 (= アーキ確定 2026-05-28):
  - 下段方向のみ: classify が色を返したが profile に合致しない場合 EMPTY 化
  - 上段救済禁止: tier1 EMPTY 判定の覆しは行わない (= 案 P2 失敗教訓)
  - データ汚染ガード: n_samples < MIN_SAMPLES の色は使用しない
  - 不足時は保守的 True (= ぷよあり扱い、EMPTY 化しない)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# ============================
# 定数定義
# ============================

# 距離がこの閾値を超えたらプロファイルと不一致 → EMPTY 化
PUYO_PROFILE_REJECT_THRESHOLD: float = 3.5

# プロファイルを信用するための最低サンプル数
PUYO_PROFILE_MIN_SAMPLES: int = 50

# HSV 各軸の重み (V は照明変動が大きいため低め)
PUYO_PROFILE_H_WEIGHT: float = 1.0
PUYO_PROFILE_S_WEIGHT: float = 1.0
PUYO_PROFILE_V_WEIGHT: float = 0.5

# std=0 のとき division by zero を防ぐフォールバック値
PUYO_PROFILE_STD_FLOOR: float = 1.0

# グローバルプロファイルを信用するための最低動画本数相当サンプル数
# MIN_SAMPLES × 3 (= 動画 3 本以上の根拠が必要)
PUYO_PROFILE_GLOBAL_MIN_SAMPLES_FACTOR: int = 3

# npz 保存時のキー名定数
_NPZ_KEY_COLORS = "colors"
_NPZ_KEY_H_MEAN = "h_mean"
_NPZ_KEY_H_STD = "h_std"
_NPZ_KEY_S_MEAN = "s_mean"
_NPZ_KEY_S_STD = "s_std"
_NPZ_KEY_V_MEAN = "v_mean"
_NPZ_KEY_V_STD = "v_std"
_NPZ_KEY_N_SAMPLES = "n_samples"
_NPZ_KEY_VIDEO_ID = "video_id"

# per-video プロファイル保存先ディレクトリ (デフォルト)
DEFAULT_PROFILE_DIR: str = "data/puyo_profiles"

# グローバルプロファイルファイル名
GLOBAL_PROFILE_FILENAME: str = "global_profile.npz"


@dataclass(frozen=True)
class ColorProfile:
    """1 色分の HSV 統計プロファイル。

    Attributes:
        color: ぷよ色コード (src.board の COLOR_* 定数)
        h_mean: H (色相) 平均 (0–180)
        h_std: H 標準偏差
        s_mean: S (彩度) 平均 (0–255)
        s_std: S 標準偏差
        v_mean: V (明度) 平均 (0–255)
        v_std: V 標準偏差
        n_samples: 統計の元サンプル数
    """
    color: int
    h_mean: float
    h_std: float
    s_mean: float
    s_std: float
    v_mean: float
    v_std: float
    n_samples: int

    def normalized_distance(self, h: float, s: float, v: float) -> float:
        """HSV 値からプロファイルまでの重み付き正規化距離を返す。

        H は 0–180 循環 (例: H=0 と H=180 は同じ赤) に対応。
        各軸の std で正規化し、重み付きユークリッド距離を計算する。
        std=0 の軸は PUYO_PROFILE_STD_FLOOR でフォールバックして
        division by zero を防ぐ。

        Args:
            h: Hue 値 (0–180)
            s: Saturation 値 (0–255)
            v: Value 値 (0–255)

        Returns:
            float: 正規化された距離 (0 に近いほど一致)
        """
        # H の循環距離 (0–180 の折り返し)
        dh = abs(h - self.h_mean)
        dh = min(dh, 180.0 - dh)
        ds = abs(s - self.s_mean)
        dv = abs(v - self.v_mean)

        # std=0 を STD_FLOOR に置換して division by zero を回避
        h_std = max(self.h_std, PUYO_PROFILE_STD_FLOOR)
        s_std = max(self.s_std, PUYO_PROFILE_STD_FLOOR)
        v_std = max(self.v_std, PUYO_PROFILE_STD_FLOOR)

        # 重み付き正規化ユークリッド距離
        dist = math.sqrt(
            PUYO_PROFILE_H_WEIGHT * (dh / h_std) ** 2
            + PUYO_PROFILE_S_WEIGHT * (ds / s_std) ** 2
            + PUYO_PROFILE_V_WEIGHT * (dv / v_std) ** 2
        )
        return dist


@dataclass
class PuyoColorProfileDB:
    """per-video ぷよ色プロファイル DB。

    OnlineHsvCalibrator が学習した stats から構築し、
    classify 後の色が本当にその色のぷよに見えるかを判定する。

    Attributes:
        profiles: 色コード → ColorProfile マッピング
        video_id: 対応動画 ID (None = グローバル)
    """
    profiles: dict[int, ColorProfile] = field(default_factory=dict)
    video_id: str | None = None

    def is_puyo_like(
        self, color: int, h: float, s: float, v: float,
    ) -> bool:
        """指定 HSV 値が color のプロファイルに一致するか判定する。

        保守的 True 条件 (= ぷよあり扱い、EMPTY 化しない):
          - プロファイルが存在しない色
          - n_samples < PUYO_PROFILE_MIN_SAMPLES のプロファイル

        プロファイルが十分なサンプルを持つ場合のみ距離チェックを行う。

        Args:
            color: 分類された色コード
            h: セル中央 H 中央値 (0–180)
            s: セル中央 S 中央値 (0–255)
            v: セル中央 V 中央値 (0–255)

        Returns:
            True = ぷよらしい (色を保持)、False = プロファイル不一致 (EMPTY 化対象)
        """
        profile = self.profiles.get(color)
        # プロファイルなし or サンプル不足 → 保守的 True
        if profile is None or profile.n_samples < PUYO_PROFILE_MIN_SAMPLES:
            return True
        dist = profile.normalized_distance(h, s, v)
        return dist <= PUYO_PROFILE_REJECT_THRESHOLD

    @classmethod
    def from_online_hsv_state(
        cls,
        stats: dict,
        video_id: str | None = None,
    ) -> "PuyoColorProfileDB":
        """OnlineHsvCalibrator の stats から PuyoColorProfileDB を構築する。

        stats 形式:
            {"<color>": {
                "h_mean": float, "h_var": float,
                "s_mean": float, "s_var": float,
                "v_mean": float, "v_var": float,
                "n": int
            }, ...}

        n_samples < PUYO_PROFILE_MIN_SAMPLES の色は除外する。

        Args:
            stats: online_hsv_state["stats"] の dict
            video_id: 動画 ID (省略可)

        Returns:
            PuyoColorProfileDB インスタンス
        """
        profiles: dict[int, ColorProfile] = {}
        for color_str, stat in stats.items():
            n = int(stat.get("n", 0))
            if n < PUYO_PROFILE_MIN_SAMPLES:
                # サンプル不足の色は除外 (データ汚染ガード)
                continue
            color = int(color_str)
            h_std = math.sqrt(max(0.0, float(stat.get("h_var", 0.0))))
            s_std = math.sqrt(max(0.0, float(stat.get("s_var", 0.0))))
            v_std = math.sqrt(max(0.0, float(stat.get("v_var", 0.0))))
            profiles[color] = ColorProfile(
                color=color,
                h_mean=float(stat.get("h_mean", 0.0)),
                h_std=h_std,
                s_mean=float(stat.get("s_mean", 0.0)),
                s_std=s_std,
                v_mean=float(stat.get("v_mean", 0.0)),
                v_std=v_std,
                n_samples=n,
            )
        return cls(profiles=profiles, video_id=video_id)

    def save(self, npz_path: str | Path) -> None:
        """プロファイル DB を npz ファイルに保存する。

        Args:
            npz_path: 保存先パス (.npz)
        """
        npz_path = Path(npz_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        colors = list(self.profiles.keys())
        if not colors:
            # プロファイルなし → 空 npz を保存
            np.savez(
                str(npz_path),
                **{
                    _NPZ_KEY_COLORS: np.array([], dtype=np.int32),
                    _NPZ_KEY_H_MEAN: np.array([], dtype=np.float64),
                    _NPZ_KEY_H_STD: np.array([], dtype=np.float64),
                    _NPZ_KEY_S_MEAN: np.array([], dtype=np.float64),
                    _NPZ_KEY_S_STD: np.array([], dtype=np.float64),
                    _NPZ_KEY_V_MEAN: np.array([], dtype=np.float64),
                    _NPZ_KEY_V_STD: np.array([], dtype=np.float64),
                    _NPZ_KEY_N_SAMPLES: np.array([], dtype=np.int64),
                    _NPZ_KEY_VIDEO_ID: np.array(
                        [self.video_id or ""], dtype=object,
                    ),
                },
            )
            return
        profiles_list = [self.profiles[c] for c in colors]
        np.savez(
            str(npz_path),
            **{
                _NPZ_KEY_COLORS: np.array(colors, dtype=np.int32),
                _NPZ_KEY_H_MEAN: np.array(
                    [p.h_mean for p in profiles_list], dtype=np.float64,
                ),
                _NPZ_KEY_H_STD: np.array(
                    [p.h_std for p in profiles_list], dtype=np.float64,
                ),
                _NPZ_KEY_S_MEAN: np.array(
                    [p.s_mean for p in profiles_list], dtype=np.float64,
                ),
                _NPZ_KEY_S_STD: np.array(
                    [p.s_std for p in profiles_list], dtype=np.float64,
                ),
                _NPZ_KEY_V_MEAN: np.array(
                    [p.v_mean for p in profiles_list], dtype=np.float64,
                ),
                _NPZ_KEY_V_STD: np.array(
                    [p.v_std for p in profiles_list], dtype=np.float64,
                ),
                _NPZ_KEY_N_SAMPLES: np.array(
                    [p.n_samples for p in profiles_list], dtype=np.int64,
                ),
                _NPZ_KEY_VIDEO_ID: np.array(
                    [self.video_id or ""], dtype=object,
                ),
            },
        )

    @classmethod
    def load(cls, npz_path: str | Path) -> "PuyoColorProfileDB":
        """npz ファイルから PuyoColorProfileDB をロードする。

        Args:
            npz_path: ロード元パス (.npz)

        Returns:
            PuyoColorProfileDB インスタンス
        """
        npz_path = Path(npz_path)
        data = np.load(str(npz_path), allow_pickle=True)
        colors = data[_NPZ_KEY_COLORS].tolist()
        h_means = data[_NPZ_KEY_H_MEAN].tolist()
        h_stds = data[_NPZ_KEY_H_STD].tolist()
        s_means = data[_NPZ_KEY_S_MEAN].tolist()
        s_stds = data[_NPZ_KEY_S_STD].tolist()
        v_means = data[_NPZ_KEY_V_MEAN].tolist()
        v_stds = data[_NPZ_KEY_V_STD].tolist()
        n_samples = data[_NPZ_KEY_N_SAMPLES].tolist()
        video_id_arr = data.get(_NPZ_KEY_VIDEO_ID, np.array([""]))[0]
        video_id = str(video_id_arr) if video_id_arr else None
        profiles: dict[int, ColorProfile] = {}
        for i, color in enumerate(colors):
            profiles[int(color)] = ColorProfile(
                color=int(color),
                h_mean=float(h_means[i]),
                h_std=float(h_stds[i]),
                s_mean=float(s_means[i]),
                s_std=float(s_stds[i]),
                v_mean=float(v_means[i]),
                v_std=float(v_stds[i]),
                n_samples=int(n_samples[i]),
            )
        return cls(profiles=profiles, video_id=video_id or None)

    @classmethod
    def load_for_video(
        cls,
        video_id: str | None,
        profile_dir: str | Path = DEFAULT_PROFILE_DIR,
    ) -> "PuyoColorProfileDB":
        """動画 ID に対応する per-video npz をロードする。

        優先順位:
          1. {profile_dir}/{video_id}_profile.npz が存在すればロード
          2. {profile_dir}/global_profile.npz が存在すれば fallback
          3. 両方不在なら空 PuyoColorProfileDB を返す (保守的動作)

        Args:
            video_id: 動画 ID (例: "v29")。None の場合は global 直接。
            profile_dir: プロファイル保存ディレクトリ

        Returns:
            PuyoColorProfileDB インスタンス
        """
        profile_dir = Path(profile_dir)
        # per-video npz を試みる
        if video_id is not None:
            per_video_path = profile_dir / f"{video_id}_profile.npz"
            if per_video_path.exists():
                return cls.load(per_video_path)
        # global fallback
        global_path = profile_dir / GLOBAL_PROFILE_FILENAME
        if global_path.exists():
            return cls.load(global_path)
        # 両方不在 → 空 DB (全色保守的 True)
        return cls(profiles={}, video_id=video_id)
