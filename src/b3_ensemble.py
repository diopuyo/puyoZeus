"""B3 の複数モデル平均 (既定 OFF の入れ物)。

同じ行を複数の TwoStateModel に通し、確率を平均して 1 つの確率として返す。TwoStateModel と同じ口
(predict_row / predict / static / exchange) を持つので、配信の LiveB3・b3_async はモデルの読込口以外を変えない。

- 平均の既定は logit の平均 (mean_logit)。確率の算術平均 (mean_p) へも設定で切り替えられる。
  CV (b3_m3b_cv) は確率を平均しておらず、seed ごとの OOF を別々に採点して成績を平均する。そこで PC2 が保存済みの OOF
  (ref 3 本・約 200 万行・60 動画) で測った (c2_seed_ensemble_eval_20261025、diagnostic_not_quality_pass):
  1 本に対する LL の差は logit 平均 -0.000366 [-0.000376, -0.000354]、確率平均 -0.000349 (どちらも区間が 0 より下)。
  ECE は全体で +0.0004〜+0.0005、層の最大で +0.0012〜+0.0013 増える (平均で予測が中央へ寄るため)。
  logit の計算 (確率を [LOGIT_CLIP, 1-LOGIT_CLIP] へ切る・log(p/(1-p)) の平均・sigmoid で戻す) は PC2 の
  make_ens_oof.py と同じ式・同じ切り幅 (1e-15)。
- 全メンバーの値が同じなら、平均せずにその値をそのまま返す。素朴な平均は、同じ値 3 つでも最後の桁がずれる
  (p を 200 万個で試すと mean_p で 16.6%、logit 経由で 32% が元の p と一致しない)。1 本だけの指定と、
  同じモデルを並べた指定が、1 本の出力とビット一致するための規則。
- 特徴列の順序・欠測の扱いは TwoStateModel (RowPredictor) にそのまま任せる。メンバーの列が食い違えば起動しない。
- モデルは配信の外で学習し固定する。マニフェスト (JSON) が平均の仕方と各モデルの SHA256 を持ち、
  マニフェスト自体の SHA256 は src/phase_j/live_obs_trial.py の C2_ENSEMBLE_SHA256 が単一の登録場所。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from src.b3_train_model import TwoStateModel

# 子 process (b3_async の計算 process) へ届ける環境変数。値は '0' / '1'。launcher の旗 b3_ensemble が配送する。
ENSEMBLE_ENV = 'PUYO_B3_ENSEMBLE'
AGG_MEAN_P = 'mean_p'            # 確率の算術平均
AGG_MEAN_LOGIT = 'mean_logit'    # logit の平均 (既定。PC2 の保存 OOF での測定で LL がわずかに良い)
AGGREGATES = (AGG_MEAN_P, AGG_MEAN_LOGIT)
DEFAULT_AGGREGATE = AGG_MEAN_LOGIT
MANIFEST_FORMAT = 'b3_ensemble_v1'
# logit を有限にするため、確率をこの幅だけ内側へ寄せてから変換する (確率 0 / 1 そのものの logit は無限大)。
# PC2 の make_ens_oof.py (OOF の平均) の eps と同じ値。依頼文の 1e-6 ではなく、測定に使った台本の値に合わせた。
LOGIT_CLIP = 1e-15
SHA256_HEX_LENGTH = 64
HASH_CHUNK_BYTES = 1 << 20


class EnsembleError(ValueError):
    """アンサンブルの設定・マニフェスト・モデルの食い違い。メッセージはそのまま画面に出す。"""


def ensemble_enabled() -> bool:
    """この process がアンサンブル構成か (live_obs_trial.configure が環境変数で子 process へ配送する)。"""
    return os.environ.get(ENSEMBLE_ENV) == '1'


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_probabilities(stack: np.ndarray, aggregate: str = DEFAULT_AGGREGATE) -> np.ndarray:
    """(メンバー数, 行数) の確率を行ごとに 1 つへ平均する。全メンバーが同じ値の行は、その値をそのまま返す。"""
    if aggregate not in AGGREGATES:
        raise EnsembleError(f'平均の仕方が不正です: {aggregate!r} ({list(AGGREGATES)} のいずれか)')
    stack = np.asarray(stack, dtype=np.float64)
    first = stack[0]
    same = (stack == first).all(axis=0)
    if aggregate == AGG_MEAN_LOGIT:
        inner = np.clip(stack, LOGIT_CLIP, 1.0 - LOGIT_CLIP)
        mean_logit = np.log(inner / (1.0 - inner)).mean(axis=0)
        mixed = 1.0 / (1.0 + np.exp(-mean_logit))
    else:
        mixed = stack.mean(axis=0)
    return np.where(same, first, mixed)


def _evaluator_columns(model: Any) -> tuple[Optional[list[str]], ...]:
    """(static の列, exchange の列)。評価器が無い状態は None。"""
    return tuple(None if ev is None else list(ev.columns) for ev in (model.static, model.exchange))


def check_same_columns(members: Sequence[Any]) -> None:
    """全メンバーの評価器の有無と特徴列 (名前と順序) が先頭と同じことを確かめる。違えば番号つきで止める。"""
    reference = _evaluator_columns(members[0])
    for index, member in enumerate(members[1:], start=1):
        if _evaluator_columns(member) != reference:
            raise EnsembleError(f'アンサンブルのモデル {index} 番目の特徴列 (名前・順序) か評価器の有無が、'
                                '0 番目と違います (同じ学習の seed 違いだけを並べてください)')


class EnsembleTwoStateModel:
    """複数の TwoStateModel の確率を平均する。TwoStateModel と同じ口を持つ (1 本だけの指定も同じ入れ物)。"""

    def __init__(self, members: Sequence[TwoStateModel], aggregate: str = DEFAULT_AGGREGATE,
                 manifest: Optional['EnsembleManifest'] = None) -> None:
        members = tuple(members)
        if not members:
            raise EnsembleError('アンサンブルのモデルが 0 本です')
        if aggregate not in AGGREGATES:
            raise EnsembleError(f'平均の仕方が不正です: {aggregate!r} ({list(AGGREGATES)} のいずれか)')
        check_same_columns(members)
        self.members, self.aggregate, self.manifest = members, aggregate, manifest
        # LiveB3 の列の照合と c2_record が読む口。全メンバーの列は同じと上で確かめてある。
        self.static, self.exchange = members[0].static, members[0].exchange

    def predict_row(self, row: Mapping[str, object]) -> float:
        """固定評価器の 1 行用経路。各メンバーの predict_row (列順・欠測の扱いを含む) を呼んで平均する。"""
        stack = np.array([[member.predict_row(row)] for member in self.members], dtype=np.float64)
        return float(aggregate_probabilities(stack, self.aggregate)[0])

    def predict(self, df: pd.DataFrame, extra: Optional[Mapping[str, np.ndarray]] = None) -> np.ndarray:
        """一括の経路 (再生・検証用)。TwoStateModel.predict と同じ引数で、メンバーの出力を同じ平均で束ねる。"""
        stack = np.stack([member.predict(df, extra) for member in self.members])
        return aggregate_probabilities(stack, self.aggregate)


# ------------------------------------------------------------------------------------ マニフェスト

@dataclass(frozen=True)
class MemberSpec:
    """マニフェストの 1 モデル。name はマニフェストと同じフォルダの中のファイル名。"""
    name: str
    sha256: str


@dataclass(frozen=True)
class EnsembleManifest:
    path: Path
    aggregate: str
    members: tuple[MemberSpec, ...]

    def member_path(self, spec: MemberSpec) -> Path:
        return self.path.parent / spec.name


def _is_plain_file_name(name: object) -> bool:
    return (isinstance(name, str) and bool(name.strip()) and name == Path(name).name
            and name not in ('.', '..') and '\\' not in name)


def _member_spec(index: int, item: object) -> MemberSpec:
    if not isinstance(item, dict) or set(item) != {'name', 'sha256'}:
        raise EnsembleError(f'マニフェストの members[{index}] は {{"name", "sha256"}} だけの形にしてください')
    name, digest = item['name'], item['sha256']
    if not _is_plain_file_name(name):
        raise EnsembleError(f'マニフェストの members[{index}].name はフォルダを含まないファイル名にしてください: {name!r}')
    if not (isinstance(digest, str) and len(digest) == SHA256_HEX_LENGTH
            and all(c in '0123456789abcdef' for c in digest)):
        raise EnsembleError(f'マニフェストの members[{index}].sha256 は小文字 16 進 64 桁です ({name})')
    return MemberSpec(name, digest)


def read_manifest(path: Path) -> EnsembleManifest:
    """マニフェスト JSON を読んで形を検査する (モデルのファイルは開かない)。"""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EnsembleError(f'アンサンブルのマニフェストが JSON として読めません: {path} ({error})') from error
    if not isinstance(data, dict) or set(data) != {'format', 'aggregate', 'members'}:
        raise EnsembleError('マニフェストの鍵は format・aggregate・members だけにしてください')
    if data['format'] != MANIFEST_FORMAT:
        raise EnsembleError(f'マニフェストの format が {MANIFEST_FORMAT!r} ではありません: {data["format"]!r}')
    if data['aggregate'] not in AGGREGATES:
        raise EnsembleError(f'マニフェストの aggregate が不正です: {data["aggregate"]!r} ({list(AGGREGATES)})')
    items = data['members']
    if not isinstance(items, list) or not items:
        raise EnsembleError('マニフェストの members は 1 本以上のリストにしてください')
    members = tuple(_member_spec(i, item) for i, item in enumerate(items))
    if len({spec.name for spec in members}) != len(members):
        raise EnsembleError('マニフェストの members.name に重複があります')
    return EnsembleManifest(path, data['aggregate'], members)


def verify_member_files(manifest: EnsembleManifest) -> None:
    """各モデルのファイルの SHA256 がマニフェストと一致することを、モデルの名前つきで確かめる。"""
    for spec in manifest.members:
        member = manifest.member_path(spec)
        if not member.is_file():
            raise EnsembleError(f'アンサンブルのモデルがありません: {spec.name} ({member})')
        if sha256_of_file(member) != spec.sha256:
            raise EnsembleError(f'アンサンブルのモデルの SHA256 が不一致です: {spec.name}')


def load_ensemble_model(manifest_path: Path) -> EnsembleTwoStateModel:
    """マニフェストを読み、全モデルの SHA256 を照合してから読み込んで 1 つの入れ物にする。"""
    from src.b3_train_model import load_model
    manifest = read_manifest(manifest_path)
    verify_member_files(manifest)
    members = [load_model(manifest.member_path(spec)) for spec in manifest.members]
    return EnsembleTwoStateModel(members, manifest.aggregate, manifest)


def ensemble_record(model: EnsembleTwoStateModel) -> dict[str, Any]:
    """b3_config.json へ残すアンサンブルの実効構成 (ON のときだけ書く。OFF の記録は従来のまま)。"""
    manifest = model.manifest
    members = [] if manifest is None else [dict(name=s.name, sha256=s.sha256) for s in manifest.members]
    return dict(ensemble=dict(aggregate=model.aggregate, n_members=len(model.members), members=members))


def build_manifest(model_paths: Sequence[Path], aggregate: str = DEFAULT_AGGREGATE) -> dict[str, Any]:
    """モデルのファイルからマニフェストの中身を作る (SHA256 を計算して並べるだけ。登録は別)。"""
    if aggregate not in AGGREGATES:
        raise EnsembleError(f'平均の仕方が不正です: {aggregate!r} ({list(AGGREGATES)} のいずれか)')
    paths = [Path(p) for p in model_paths]
    if not paths:
        raise EnsembleError('モデルが 0 本です')
    if len({p.parent.resolve() for p in paths}) != 1:
        raise EnsembleError('マニフェストと同じフォルダに全モデルを置いてください')
    members = [dict(name=p.name, sha256=sha256_of_file(p)) for p in paths]
    return dict(format=MANIFEST_FORMAT, aggregate=aggregate, members=members)
