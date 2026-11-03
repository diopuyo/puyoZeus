"""dl1k 撃ち合い特徴 (EX) の固定 70 列台帳 (研究専用・既定 OFF)。

検索側 (e7777d7d 起点) と M3 側 (9dedfe88 起点) の両ブランチへ同じバイト列で置く。
既存 M3 列は変更しない。新しい列は固定順で末尾にだけ足す:
生値 20 → 正規化 20 → 比率/状態 20 → mask 8 → available_at 2 (計 70、学習は正規化と比率/状態の 40)。
左右交換は全列 SIDE_ABS (p1<->p2 の入れ替えだけ)。絶対量に負号や 1-x をかけない。
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

DEFAULT_ENABLED = False
COLUMN_VERSION = 'dl1k_ex_v1'
SIDES = ('p1', 'p2')
# 非負の得点・個数・手数の正規化 z/(A+z), z=log1p(x)。B3 の nonsat と同じ形で A は固定。
NORMALIZE_A = 3.0
# 欠測理由コード。B3 の src.b3_column_spec.Mask と同じ値 (M3 側の試験で一致を検査する)。
MASK_OK = 0
MASK_NOT_YET = 3
MASK_UNPREDICTABLE = 6
MASK_TRUNCATED = 7
MASK_NOT_APPLICABLE = 8
MASK_CODES = (MASK_OK, MASK_NOT_YET, MASK_UNPREDICTABLE, MASK_TRUNCATED, MASK_NOT_APPLICABLE)

Y_STEMS = ('ex_y_score', 'ex_y_generated', 'ex_y_sent', 'ex_y_pending', 'ex_hY')
Z_STEMS = ('ex_z_score', 'ex_z_generated', 'ex_z_sent', 'ex_z_pending', 'ex_hZ')
RAW_STEMS = Y_STEMS + Z_STEMS
AGREE_STEMS = ('ex_y_power_agree', 'ex_z_power_agree', 'ex_direction_agree',
               'ex_y_fired_fraction', 'ex_z_fired_fraction')
STATUS_STEMS = ('ex_search_complete', 'ex_search_cutoff', 'ex_completed_tapes_nf')
STATE_STEMS = AGREE_STEMS + STATUS_STEMS + ('ex_proof_dead', 'ex_hyp_dead_fraction')
MASK_STEMS = ('mk_ex_y', 'mk_ex_z', 'mk_ex_pair', 'mk_ex_proof')

RAW_COLUMNS = tuple(f'{s}_{p}' for s in RAW_STEMS for p in SIDES)
NORMAL_COLUMNS = tuple(f'{s}_nf_{p}' for s in RAW_STEMS for p in SIDES)
STATE_COLUMNS = tuple(f'{s}_{p}' for s in STATE_STEMS for p in SIDES)
MASK_COLUMNS = tuple(f'{s}_{p}' for s in MASK_STEMS for p in SIDES)
AVAILABLE_COLUMNS = ('av_ex_p1', 'av_ex_p2')
COLUMNS = RAW_COLUMNS + NORMAL_COLUMNS + STATE_COLUMNS + MASK_COLUMNS + AVAILABLE_COLUMNS
LEARNING_COLUMNS = NORMAL_COLUMNS + STATE_COLUMNS
AUDIT_COLUMNS = RAW_COLUMNS + MASK_COLUMNS + AVAILABLE_COLUMNS
EXPECTED_COUNTS = dict(columns=70, learning=40, audit=30)
# 正規化列と生値の再計算差の許容 (float32 保存の丸めより十分大きく、計算違いより十分小さい)。
NORMAL_TOLERANCE = 1e-6
# 列ごとの欠測理由列の語幹。Y/Z 生値・正規化は役割別、一致度・状態・仮想死亡は pair、証明は proof。
_MASK_STEM_OF = {**{s: 'mk_ex_y' for s in Y_STEMS}, **{s: 'mk_ex_z' for s in Z_STEMS},
                 **{s: 'mk_ex_pair' for s in AGREE_STEMS + STATUS_STEMS + ('ex_hyp_dead_fraction',)},
                 'ex_proof_dead': 'mk_ex_proof'}


def side_of(column: str) -> str:
    """列名末尾の物理側 (p1/p2)。"""
    side = column.rsplit('_', 1)[-1]
    if side not in SIDES:
        raise ValueError(f'側の無い列: {column}')
    return side


def stem_of(column: str) -> str:
    """列名から側と正規化接尾辞を外した語幹。"""
    stem = column.rsplit('_', 1)[0]
    return stem[:-3] if stem.endswith('_nf') and stem[:-3] in RAW_STEMS else stem


def mask_of(column: str) -> str | None:
    """値列が欠測のとき理由を持つ mask 列名。mask・時刻列は None。"""
    if column in MASK_COLUMNS or column in AVAILABLE_COLUMNS:
        return None
    return f'{_MASK_STEM_OF[stem_of(column)]}_{side_of(column)}'


def normalize(value: float) -> float:
    """非負量の z/(3+z)。欠測は欠測のまま。負は計算の誤りとして止める。"""
    if not math.isfinite(value):
        return math.nan
    if value < 0:
        raise ValueError('非負値のみ正規化する')
    z = math.log1p(value)
    return z/(NORMALIZE_A+z)


def blank(reason: int) -> dict[str, Any]:
    """全値を欠測、mask を同じ理由にした 70 列。"""
    if reason == MASK_OK:
        raise ValueError('欠測行の理由に OK は使えない')
    return {c: (reason if c in MASK_COLUMNS else math.nan) for c in COLUMNS}


def swap(extra: Mapping[str, Any]) -> dict[str, Any]:
    """全列 SIDE_ABS の左右交換。2 回で元に戻る。"""
    def other(name: str) -> str:
        return name[:-2]+('p2' if name.endswith('p1') else 'p1')
    return {name: extra[other(name)] for name in COLUMNS}


def append(row: Mapping[str, Any], extra: Mapping[str, Any], enabled: bool = DEFAULT_ENABLED) -> dict[str, Any]:
    """既定 OFF は元の行をそのまま返す。ON は固定順の 70 列を末尾へ足す。"""
    if not enabled:
        return dict(row)
    if set(row) & set(COLUMNS):
        raise ValueError('既存列との衝突')
    if tuple(extra) != COLUMNS:
        raise ValueError('70 列の順序不一致')
    return dict(row, **extra)


def _value_violations(values: Mapping[str, Any], column: str, finite: Any) -> dict[str, Any]:
    """1 値列の範囲違反の真偽配列 (欠測は範囲検査から外す)。"""
    import numpy as np
    v = np.asarray(values[column], dtype=np.float64)
    if column in RAW_COLUMNS:
        return dict(raw_negative=finite & (v < 0))
    if column in NORMAL_COLUMNS:
        raw = np.asarray(values[column.replace('_nf_', '_')], dtype=np.float64)
        expected = np.where(finite, np.log1p(np.where(finite, np.maximum(raw, 0.), 0.)), 0.)
        expected = expected/(NORMALIZE_A+expected)
        return dict(normal_range=finite & ((v < 0) | (v >= 1)),
                    normal_mismatch=finite & (np.abs(v-expected) > NORMAL_TOLERANCE))
    return dict(state_range=finite & ((v < 0) | (v > 1)))


def array_violations(values: Mapping[str, Any]) -> dict[str, int]:
    """列名→配列の表で契約違反を数える。全 0 が合格 (母数は呼出側が併記する)。"""
    import numpy as np
    counts: dict[str, int] = {}

    def add(name: str, flags: Any) -> None:
        counts[name] = counts.get(name, 0)+int(np.count_nonzero(flags))
    for column in MASK_COLUMNS:
        add('mask_code', ~np.isin(np.asarray(values[column]), MASK_CODES))
    for column in AVAILABLE_COLUMNS:
        add('available_infinite', np.isinf(np.asarray(values[column], dtype=np.float64)))
    for column in RAW_COLUMNS+NORMAL_COLUMNS+STATE_COLUMNS:
        v = np.asarray(values[column], dtype=np.float64)
        mask = np.asarray(values[mask_of(column)])
        missing = np.isnan(v)
        add('missing_with_ok_mask', missing & (mask == MASK_OK))
        add('value_with_absent_mask', ~missing & np.isin(mask, (MASK_NOT_YET, MASK_NOT_APPLICABLE)))
        for name, flags in _value_violations(values, column, ~missing).items():
            add(name, flags)
    return counts


def column_digest() -> str:
    """列名と順番の SHA256。両ブランチの台帳が同じであることの照合に使う。"""
    return hashlib.sha256('\n'.join(COLUMNS).encode()).hexdigest()


def ledger() -> dict[str, Any]:
    """列仕様 JSON。"""
    counts = dict(columns=len(COLUMNS), learning=len(LEARNING_COLUMNS), audit=len(AUDIT_COLUMNS))
    if counts != EXPECTED_COUNTS or len(set(COLUMNS)) != len(COLUMNS):
        raise AssertionError(f'列数の不一致: {counts}')
    return dict(version=COLUMN_VERSION, default_enabled=DEFAULT_ENABLED, columns=list(COLUMNS),
                learning=list(LEARNING_COLUMNS), audit=list(AUDIT_COLUMNS),
                symmetry={c: 'SIDE_ABS' for c in COLUMNS}, mask_of={c: mask_of(c) for c in COLUMNS},
                mask_codes=dict(OK=MASK_OK, NOT_YET_AVAILABLE=MASK_NOT_YET, UNPREDICTABLE=MASK_UNPREDICTABLE,
                                SEARCH_TRUNCATED=MASK_TRUNCATED, NOT_APPLICABLE=MASK_NOT_APPLICABLE),
                normalization=f'z/({NORMALIZE_A:g}+z), z=log1p(x)', column_sha256=column_digest())
