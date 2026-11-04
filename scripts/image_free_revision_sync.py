"""共有のr2確定票を受け取り、新候補だけを別出力へ展開する。git不要。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

SHARE = r'\\192.0.2.2\example_share\b3_data\codex_tasks\image_free_speed_stream_v1'
POLL_SECONDS = 30


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path: Path, record: dict) -> None:
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    temp.replace(path)


def remote_receipt() -> dict | None:
    command = "$p='"+SHARE+"\\candidate_update.json'; if(Test-Path -LiteralPath $p){Get-Content -Raw -LiteralPath $p -Encoding UTF8}"
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                            check=True, capture_output=True, text=True, timeout=25)
    return json.loads(result.stdout.lstrip('\ufeff')) if result.stdout.strip() else None


def download(record: dict, root: Path, local: Path | None) -> Path:
    destination = root/record['archive_name']
    if not destination.exists() or digest(destination) != record['archive_sha256']:
        if local:
            shutil.copyfile(local/record['archive_name'], destination)
        else:
            target = subprocess.check_output(['wslpath', '-w', str(destination)], text=True).strip()
            source = SHARE+'\\'+record['archive_name']
            command = "Copy-Item -LiteralPath '"+source.replace("'", "''")+"' -Destination '"+target.replace("'", "''")+"' -Force"
            subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], check=True, timeout=300)
    if digest(destination) != record['archive_sha256']:
        raise ValueError('r2一式SHA不一致')
    return destination


def install(root: Path, receipt: dict, archive: Path) -> dict:
    generation = root/'candidate_generations'/receipt['revision_id']
    if not generation.exists():
        generation.mkdir(parents=True)
        with tarfile.open(archive) as stream:
            stream.extractall(generation, filter='data')
    config = copy.deepcopy(read(root/'config.json'))
    output = generation/'results'
    output.mkdir(exist_ok=True)
    for name in receipt['model_sha256']:
        shutil.copyfile(generation/name, output/name)
    config.update(output=str(output), videos=[j for j in config['videos'] if j['route'] == 'new'])
    for job in config['videos']:
        job.update(start_frame=0, owned_start=0)
        job.pop('seed', None)
        job.pop('seed_sha256', None)
    path = generation/'config.json'
    save(path, config)
    record = dict(receipt, config=str(path), source=str(generation/'source'))
    save(root/'candidate_revision.json', record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--local-publication', type=Path)
    args = parser.parse_args()
    while True:
        try:
            path = args.local_publication/'candidate_update.json' if args.local_publication else None
            receipt = read(path) if path and path.exists() else (None if path else remote_receipt())
            if receipt:
                if receipt['changed']:
                    install(args.root, receipt, download(receipt, args.root, args.local_publication))
                save(args.root/'r2_resolution.json', receipt)
                return
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            save(args.root/'revision_sync_error.json', dict(error=str(error), at=time.time()))
        time.sleep(POLL_SECONDS)


if __name__ == '__main__':
    main()
