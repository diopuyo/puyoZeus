"""同梱パッケージのライセンス表記を dist-info の METADATA から一覧にする (自動抽出。法務確認の代わりではない)。

    python packaging/list_licenses.py <PuyoLive/python/Lib/site-packages>

出力: パッケージ名 / 版 / License-Expression (無ければ License、無ければ Classifier) / 同梱された
ライセンス関連ファイル数。wheel 内に第三者ライブラリ (delvewheel/auditwheel の *.libs) を含むものは別掲する。
"""
from __future__ import annotations

from email.parser import Parser
from pathlib import Path
import sys

LICENSE_FILE_PATTERNS = ('LICEN*', 'COPYING*', 'NOTICE*')
LICENSE_TEXT_LIMIT = 80  # License 欄に本文が丸ごと入っている wheel があるため先頭だけ使う


def license_of(metadata: dict[str, object]) -> str:
    """表記の優先順: License-Expression > License > Classifier (License ::)。"""
    expression = (metadata.get('License-Expression') or '').strip()
    if expression:
        return expression
    text = (metadata.get('License') or '').strip().replace('\n', ' ')
    if text and text.upper() != 'UNKNOWN':
        return text[:LICENSE_TEXT_LIMIT] + ('…' if len(text) > LICENSE_TEXT_LIMIT else '')
    classifiers = [c.split('::')[-1].strip() for c in metadata.get_all('Classifier') or [] if c.startswith('License ::')]
    return ' / '.join(classifiers) or '(表記なし)'


def rows(site: Path) -> list[tuple[str, str, str, int, list[str]]]:
    result = []
    for info in sorted(site.glob('*.dist-info')):
        meta_path = info / 'METADATA'
        if not meta_path.is_file():
            continue
        metadata = Parser().parsestr(meta_path.read_text(encoding='utf-8', errors='replace'))
        files = [p for pattern in LICENSE_FILE_PATTERNS for p in info.rglob(pattern) if p.is_file()]
        libs = sorted({p.name for d in (site / f'{metadata["Name"].lower().replace("-", "_")}.libs',) if d.is_dir()
                       for p in d.glob('*.dll')})
        result.append((metadata['Name'], metadata['Version'], license_of(metadata), len(files), libs))
    return result


def main() -> None:
    site = Path(sys.argv[1])
    print('| パッケージ | 版 | ライセンス表記 | 同梱ライセンスファイル数 | 同梱ネイティブ DLL (*.libs) |')
    print('|---|---|---|---|---|')
    for name, version, lic, count, libs in rows(site):
        print(f'| {name} | {version} | {lic} | {count} | {", ".join(libs)} |')


if __name__ == '__main__':
    main()
