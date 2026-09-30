"""実行時に open されたファイルを記録する監査フック (sitecustomize として一時配置する)。

配布物へ入れる資産を「推測でなく実測」で決めるための道具。
使い方: このファイルを python/Lib/site-packages/sitecustomize.py へコピーし、環境変数
PUYO_AUDIT_LOG=<出力ベース名> を設定して起動する。spawn した子 process も同じフックが
入り、<ベース名>.<pid> に 1 行 1 パスで追記される。集計は packaging/audit_opens.py。
配布物には入れない (build_bundle.py は組み立てのたびにこのファイルを含めない)。
"""
import os
import sys

_LOG = os.environ.get('PUYO_AUDIT_LOG')
_BUSY = False
_SEEN: set[str] = set()


def _hook(event: str, args: tuple) -> None:
    global _BUSY
    if event != 'open' or _BUSY:
        return
    path = args[0]
    if not isinstance(path, (str, bytes)):
        return
    mode = args[1] if len(args) > 1 else None
    if isinstance(mode, str) and any(flag in mode for flag in ('w', 'a', '+', 'x')):
        return
    text = os.fsdecode(path)
    if text in _SEEN:
        return
    _SEEN.add(text)
    _BUSY = True
    try:
        with open(f'{_LOG}.{os.getpid()}', 'a', encoding='utf-8') as stream:
            stream.write(os.path.abspath(text) + '\n')
    finally:
        _BUSY = False


if _LOG:
    sys.addaudithook(_hook)
