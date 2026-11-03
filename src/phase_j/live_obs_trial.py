"""2026-10-20本人用試験の明示設定。製品既定はすべてOFF。

2026-10-25 b3_c2 (既定OFF) を追加: 勝率モデルを M3 から C2 へ替え、予告アイコン CNN を読まない構成。
C2 = M3 − アイコン由来 13 列 + 台帳の正味 17 列 (案A, src/b3_net_ledger.py)。前提は b3_judgment・b3_async・b3_m3・
b3_skip_icon_reading。本番の既定は変えない (src/production_config.py に記録だけ)。

2026-10-25 b3_ensemble (既定OFF) を追加: C2 の seed 違いを複数並べて確率を平均する (src/b3_ensemble.py)。前提は b3_c2。
ON のとき b3_model はアンサンブルのマニフェスト (JSON) で、C2_ENSEMBLE_SHA256 (マニフェストの SHA256) と各モデルの
SHA256 で固定する。OFF の動き・記録・設定は従来と同一。
"""
from __future__ import annotations
import argparse
from copy import copy
from hashlib import sha256
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.production_config import B3_ENSEMBLE_DEFAULT

M3_SHA256 = '11f7b32a26a9b70dbfda6544d64bc83695a6218531deaf25204d518704d26788'
# C2 の本番用モデル (全データ・large・seed1003、PC2 m3net_final.py の最終 fit) の SHA256。差し替えはここ 1 か所だけ。
# 照合: PC1 が 2026-10-25 07:46 に LOCK_C2.json で確認 (static 222・exchange 295・特徴 299 が PREREG の C2 と名前・順序一致、
# 入力 60 動画の SHA が一致、正味の列の導出コードが a7dcfdf6 と一致)。最終 fit は判定に使っていない。
# 空 = 未登録。b3_c2 を ON にすると起動を拒否する (どのモデルでも通る状態にはしない)。
C2_SHA256 = '79d7cb0772936a5f4eda9ba1733c2274ea303f39859af3d6825f138bfa0694d6'
C2_SHA256_LENGTH = 64
# 2026-10-25 b3_ensemble (既定OFF): C2 の seed 違いを複数並べて確率を平均する構成 (src/b3_ensemble.py)。
# 登録するのはアンサンブルのマニフェスト (JSON) の SHA256 で、差し替えはここ 1 か所だけ。マニフェストが
# 平均の仕方と各モデルの SHA256 を持つので、モデルの中身・本数・平均の仕方まで固定される。
# 空 = 未登録。b3_ensemble を ON にすると起動を拒否する (最終学習のモデルが届いてから登録する)。
C2_ENSEMBLE_SHA256 = '1377ddf70f4c5d57bd073de0b1dda30acc5d70a4c59f852b2357935c10e6ab6a'

# 子 process へ届ける環境変数。値は '0' / '1'。
C2_ENV = 'PUYO_B3_C2'
# src.b3_ensemble.ENSEMBLE_ENV と同値 (importを軽く保つためここでは複製し、試験で一致を固定)。
ENSEMBLE_ENV = 'PUYO_B3_ENSEMBLE'
NET_LEDGER_ENV = 'PUYO_B3_NET_LEDGER_COLS'
SKIP_ICON_ENV = 'PUYO_B3_SKIP_ICON_READING'   # live_speed.observation_options が読む名前と同じ (試験で固定)
TRIAL_KEYS = ('live_ama_plain', 'b3_m3', 'b3_time_gate', 'live_score_cadence', 'live_static_exact')

# 表示名と記録 (b3_config.json の trial_model)
M3_LABEL = '実戦勝率: B3 (M3)'
C2_LABEL = '実戦勝率: B3 (C2)'
C2_TRIAL_MODEL = 'C2'
# C2 の特徴列数 = M3 の 295 − アイコン由来 13 + 正味 17 (M3 の記録 feature_columns=295 と同じ数え方)。
C2_FEATURE_COLUMNS = 299
# C2 で外すアイコン由来 13 列 (PC2 m3_net PREREG の icon_columns_removed_in_C2 と同一)。
# land_icon_drop_p1/p2 は外さない (着地量は台帳の観測着弾で決まり、アイコンを読まなくても値が変わらない)。
C2_REMOVED_ICON_COLUMNS = (
    'ta_i_icon_p1', 'ta_i_icon_p2', 'ta_m_icon_med_p1', 'ta_m_icon_med_p2',
    'ta_mg_icon_med_p1', 'ta_mg_icon_med_p2', 'ta_recv_icon_med_p1', 'ta_recv_icon_med_p2',
    'ta_send_icon_med_p1', 'ta_send_icon_med_p2', 'ta_unc72_icon_med_p1', 'ta_unc72_icon_med_p2',
    'df_icon_corr_total')


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """--b3-c2 を登録する (既定OFF)。配布の run_live_pipeline は変えられない (基準点に固定) ので live_config から呼ぶ。"""
    if 'b3_c2' not in {action.dest for action in parser._actions}:
        parser.add_argument('--b3-c2', action=argparse.BooleanOptionalAction, default=False,
                            help='勝率モデルを C2 (M3−アイコン13列+台帳の正味17列) に替え、予告アイコンを読まない試験用フラグ')
    if 'b3_ensemble' not in {action.dest for action in parser._actions}:
        parser.add_argument('--b3-ensemble', action=argparse.BooleanOptionalAction, default=B3_ENSEMBLE_DEFAULT,
                            help='C2 の seed 違いを複数並べて確率を平均する試験用フラグ (b3_model はアンサンブルのマニフェスト JSON)')


def c2_enabled() -> bool:
    """この process が C2 構成か (configure が環境変数で子 process へ配送する)。"""
    return os.environ.get(C2_ENV) == '1'


def trial_label() -> str:
    """B3 欄の表示名。C2 のときだけ C2、それ以外 (M3) は従来どおり。"""
    return C2_LABEL if c2_enabled() else M3_LABEL


def icon_reading_enabled() -> bool:
    """予告アイコンを読んでいるか。b3_skip_icon_reading が ON なら読まない。"""
    return os.environ.get(SKIP_ICON_ENV) != '1'


def c2_record(model: Any) -> dict[str, Any]:
    """b3_config.json へ残す C2 の実効構成。列数は M3 の記録と同じ数え方 + 実モデルの列数 (食い違えば見て分かる)。"""
    return dict(trial_model=C2_TRIAL_MODEL, feature_columns=C2_FEATURE_COLUMNS,
                static_columns=len(model.static.columns), exchange_columns=len(model.exchange.columns),
                net_ledger_cols=os.environ.get(NET_LEDGER_ENV) == '1', icon_reading=icon_reading_enabled())


def check_c2_model(model: Any) -> None:
    """C2 に要る列の最低限の照合。SHA 固定の保険 (SHA の登録間違いで別のモデルが通っても、ここで止める)。"""
    from src.b3_column_spec import NET_LEDGER_FEATURE_NAMES
    for name, evaluator in (('static', model.static), ('exchange', model.exchange)):
        left = sorted(set(evaluator.columns) & set(C2_REMOVED_ICON_COLUMNS))
        if left:
            raise ValueError(f'C2モデルの{name}評価器にアイコン由来列が残っています: {left}')
    if not set(model.exchange.columns) & set(NET_LEDGER_FEATURE_NAMES):
        raise ValueError('C2モデルのexchange評価器に台帳の正味の列がありません')


def _c2_requested(options: Any, parser: Any) -> bool:
    """b3_c2 の型と前提 (b3_judgment・b3_async・b3_m3・b3_skip_icon_reading) を検査する。"""
    value = getattr(options, 'b3_c2', False)
    if type(value) is not bool:
        parser.error('b3_c2はboolです')
    if value and not (options.b3_judgment and options.b3_async and getattr(options, 'b3_m3', False)):
        parser.error('b3_c2はtrueにはb3_judgment・b3_async・b3_m3が必要です')
    if value and getattr(options, 'b3_skip_icon_reading', False) is not True:
        parser.error('b3_c2は予告アイコンを読まない構成です。b3_skip_icon_reading=trueが必要です')
    return value


def _ensemble_requested(options: Any, parser: Any, c2: bool) -> bool:
    """b3_ensemble の型と前提 (b3_c2) を検査する。ON のとき b3_model はアンサンブルのマニフェスト (JSON)。"""
    value = getattr(options, 'b3_ensemble', False)
    if type(value) is not bool:
        parser.error('b3_ensembleはboolです')
    if value and not c2:
        parser.error('b3_ensembleはtrueにはb3_c2=trueが必要です')
    return value


def _check_ensemble_sha256(options: Any, parser: Any) -> None:
    """アンサンブルのマニフェストを登録値と照合し、各モデルの SHA256 も名前つきで照合する。未登録 (空) なら起動しない。"""
    from src.b3_ensemble import EnsembleError, read_manifest, verify_member_files
    if len(C2_ENSEMBLE_SHA256) != C2_SHA256_LENGTH:
        parser.error('C2アンサンブルのマニフェストのSHA256が未登録です (src/phase_j/live_obs_trial.py の C2_ENSEMBLE_SHA256)')
    if sha256(Path(options.b3_model).read_bytes()).hexdigest() != C2_ENSEMBLE_SHA256:
        parser.error('C2アンサンブルのマニフェストの固定SHA256が不一致です')
    try:
        verify_member_files(read_manifest(Path(options.b3_model)))
    except EnsembleError as error:
        parser.error(str(error))


def _check_model_sha256(options: Any, parser: Any, c2: bool, ensemble: bool = False) -> None:
    """M3 / C2 のモデルを SHA256 で固定する。C2 は未登録 (空) なら起動しない。ensemble はマニフェストと各モデルを固定する。"""
    if ensemble:
        _check_ensemble_sha256(options, parser)
        return
    expected = C2_SHA256 if c2 else M3_SHA256
    if c2 and len(expected) != C2_SHA256_LENGTH:
        parser.error('C2モデルのSHA256が未登録です (src/phase_j/live_obs_trial.py の C2_SHA256)')
    if sha256(Path(options.b3_model).read_bytes()).hexdigest() != expected:
        parser.error(('C2' if c2 else 'M3') + 'モデルの固定SHA256が不一致です')
    if not c2 and getattr(options, 'b3_skip_icon_reading', False):
        parser.error('M3の学習構成ではアイコン読取りが必要です')


def configure(options: Any, parser: Any) -> None:
    from .live_obs_cpu import configure_main
    configure_main()
    for key in TRIAL_KEYS:
        value = getattr(options, key, False)
        if type(value) is not bool or (value and not (options.b3_judgment and options.b3_async)):
            parser.error(f'{key}はboolで、trueにはb3_judgmentとb3_asyncが必要です')
        os.environ['PUYO_'+key.upper()] = str(int(value))
    c2 = _c2_requested(options, parser)
    ensemble = _ensemble_requested(options, parser, c2)
    # 正味の列は M3 の inflight 列が前提。ON/OFF とも明示し、親の環境の持ち込みで列構成が変わらないようにする。
    os.environ[C2_ENV] = os.environ[NET_LEDGER_ENV] = str(int(c2))
    os.environ[ENSEMBLE_ENV] = str(int(ensemble))   # 同じく ON/OFF とも明示 (親の環境の持ち込みでモデルの読み方が変わらないように)
    if getattr(options, 'b3_m3', False):
        _check_model_sha256(options, parser, c2, ensemble)


def create_baseline(table: Path, output: Path, fast_ema: bool) -> Any:
    """v4.9fの混合元を10/19aのv6へ固定する。起動時にだけ別processの環境を指定。

    v6 は M3 の列を持たない。正味の列は inflight 列 (M3=1) が前提で、M3=0 のまま ON だと列仕様の組み立てが
    例外で止まるので、NET_LEDGER_COLS も '0' を明示する (spawn する AsyncB3 の子も、この環境を継ぐ)。
    """
    from .live_b3 import LiveB3
    baseline_output = output/'v49f_b3_v6'
    baseline_output.mkdir(exist_ok=False)
    with patch.dict(os.environ, {'PUYO_B3_M3': '0', 'PUYO_B3_LEDGER_MODE': '1',
                                'PUYO_B3_LEDGER_LANDING': '1', 'PUYO_B3_TIME_GATE': '0',
                                NET_LEDGER_ENV: '0', C2_ENV: '0', ENSEMBLE_ENV: '0'}):
        return LiveB3(Path('models/v49f/b3_v6.joblib'), table, baseline_output,
                      async_enabled=True, fast_ema=fast_ema, trial_helpers=False)


def baseline_notice(notice: Any) -> Any:
    """旧配布のアイコン省略設定と同じ観測値をv6所有者へ渡す。"""
    frame, events, queues, epoch = notice.b3
    baseline = {k: v for k, v in frame.items() if k not in ('_v49f', '_ama')}
    baseline['sides'] = [dict(s, icon=None) for s in frame['sides']]
    result = copy(notice)
    object.__setattr__(result, 'b3', (baseline, events, queues, epoch))
    return result
