"""開発台本の入力場所を環境変数かリポジトリ相対で解決する。配布既定には使わない。"""
from pathlib import Path
import os


def resolve_text(value: str) -> str:
    """個別確認した開発用文字列の記号だけを置き換える。"""
    repo = Path(__file__).resolve().parents[1]
    roots = {
        '<repo>': os.environ.get('PUYO_DEV_REPO', repo.as_posix()),
        '<work>': os.environ.get('PUYO_DEV_WORK', (repo / '.local').as_posix()),
        '<home>': os.environ.get('PUYO_DEV_HOME', Path.home().as_posix()),
    }
    for marker, root in roots.items():
        value = value.replace(marker, Path(root).as_posix())
    return value
