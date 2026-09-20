"""実Consumerの比較本体を変えず、保存先だけ所有scopeに接続する。"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from scripts.g3_context_rows import digest, require
from scripts.g3_postcommit_rows import SavedRows

SOURCE_SHA = '2ec22c2ca387906b3314d7717a115f914ef32f1cdf438978a563f2e69cdf3f9c'
ROWS_NAME = 'POSTCOMMIT_CONSUMER_ROWS.json'
SIDECAR_NAME = 'G3_POSTCOMMIT_ROWS.jsonl'


def cleanup(rows: SavedRows, error: BaseException | None) -> None:
    """元失敗を終了保存の二次例外で置き換えず、取得できた原票を残す。"""
    failure: BaseException | None = None
    try:
        rows.close()
    except BaseException as caught:
        failure = caught
        rows.failure = rows.failure or repr(caught)
    receipt = dict(count=len(rows), stream_closed=rows.stream.closed, failure=rows.failure,
                   original_error=None if error is None else repr(error),
                   export=None if rows.exported is None else str(rows.exported[0]),
                   quality_gate_clear=False)
    try:
        with rows.path.with_name('G3_POSTCOMMIT_END.json').open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, ensure_ascii=False)
    except BaseException as caught:
        failure = failure or caught
    if failure is not None and error is None:
        raise failure


def bind(stack: Any, consumer: Any, module: Any, replace: Any,
         *, source: Path, source_sha: str, verify_root: Path) -> SavedRows:
    """元Consumer.close callbackより外側のstackで保存hookを保持する。"""
    require(Path(module.__file__).resolve() == source.resolve()
            and digest(source) == source_sha, 'postcommit_source')
    require(type(consumer) is module.Consumer and module.ROWS == ROWS_NAME,
            'postcommit_actual_consumer')
    require(type(consumer.rows) is list and not consumer.rows and not consumer.closed
            and not consumer.active and not consumer.errors, 'postcommit_install_late')
    require(consumer.output.resolve().is_relative_to(verify_root.resolve()), 'postcommit_storage')
    rows = SavedRows(consumer.output / SIDECAR_NAME)
    def end(kind: Any, error: Any, trace: Any) -> bool:
        cleanup(rows, error)
        return False
    stack.push(end)
    original = module.write
    def write(path: Path, value: Any) -> Any:
        if value is rows:
            require(path.resolve() == (consumer.output / ROWS_NAME).resolve(), 'postcommit_export_path')
            return rows.export(path)
        return original(path, value)
    replace(stack, module, 'write', write)
    consumer.rows = rows
    return rows


def reader(original: Any, rows: SavedRows) -> Any:
    """明示的な原readの一つのROWSパスだけを保存照合付きで置き換える。"""
    target = rows.path.with_name(ROWS_NAME).resolve()
    def read(path: Path, *args: Any, **kwargs: Any) -> Any:
        if Path(path).resolve() == target:
            require(not args and not kwargs, 'postcommit_reader_arguments')
            return rows.saved(target)
        return original(path, *args, **kwargs)
    return read
