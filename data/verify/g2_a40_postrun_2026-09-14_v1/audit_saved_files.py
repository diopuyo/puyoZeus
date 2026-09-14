"""保存済みJSON/JSONLの完全性を読取検査。欠測や末尾破損を修復済みへしない。"""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent/'video38_second_prefix_candidate_v40'
DEST = Path('/mnt/d/puyo_analyzer/verify/g2_a40_recovery_2026-09-14_v1')
REQUIRED = ('PRIVATE_CONSUMER_STATUS.json','PRIVATE_CONSUMER_COMPARISON.json',
            'POSTCOMMIT_CONSUMER_STATUS.json','TERMINAL_SAVED_REVIEW.json')


def inspect(path: Path) -> dict:
    count, error, summary = None, None, {}
    try:
        if path.suffix=='.jsonl':
            count = 0
            with path.open() as stream:
                for line in stream:
                    json.loads(line)
                    count += 1
        else:
            with path.open() as stream: value=json.load(stream)
            if path.name=='POSTCOMMIT_CONSUMER_COMPARISON.json':
                summary=dict(equal=value.get('equal'),death_equal=value['lanes']['held']['death']==value['lanes']['released']['death'])
    except (ValueError,OSError) as caught:
        error=f'{type(caught).__name__}: {caught}'
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(name=path.name,bytes=path.stat().st_size,sha256=digest,
                complete_records=count,error=error,summary=summary)


def main() -> None:
    start=time.perf_counter()
    rows=[inspect(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.suffix in ('.json','.jsonl')]
    result=dict(original_run=str(ROOT), files=rows,
        missing=[name for name in REQUIRED if not (ROOT/name).exists()],
        corrupt=[r['name'] for r in rows if r['error']], original_files_modified=False,
        full_storage_audit_pass=False, seconds=time.perf_counter()-start)
    with (DEST/'FILE_INTEGRITY.json').open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({key:value for key,value in result.items() if key!='files'}))


if __name__=='__main__': main()
