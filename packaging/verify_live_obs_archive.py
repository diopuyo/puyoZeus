"""完成ZIPの固定資産・最終ソース・入口・除外規則を検収する。"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path('D:/puyo_analyzer/packaging/build_20261020a_release')
ENTRY = Path('D:/PuyoLive_OBS_20261020_USER_ONLY')


def check_cmd(data: bytes) -> None:
    data.decode('ascii')
    assert b'\n' not in data.replace(b'\r\n',b'')


def source_checks(archive: zipfile.ZipFile) -> list[str]:
    names = subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).splitlines()
    checked = []
    for name in names:
        if not name.startswith(('src/','scripts/')):
            continue
        assert archive.read('PuyoLive/app/'+name) == (ROOT/name).read_bytes(), name
        checked.append(name)
    return checked


def main() -> None:
    receipt = json.loads((RELEASE/'archive_receipt.json').read_text())
    inventory = json.loads((RELEASE/'files_sha256.json').read_text())
    old = json.loads((RELEASE.parent/'build_20261019a_release/files_sha256.json').read_text())
    runtime = {n:h for n,h in inventory.items() if n.startswith('PuyoLive/python/')}
    previous = {n:h for n,h in old.items() if n.startswith('PuyoLive/python/')}
    changes = [n for n in runtime.keys() | previous.keys() if runtime.get(n) != previous.get(n)]
    assert changes == ['PuyoLive/python/Lib/site-packages/puyo_core/__pycache__/__init__.cpython-312.pyc']
    with zipfile.ZipFile(receipt['zip']) as archive:
        checked = source_checks(archive)
        for name in ('Start_PuyoLive_USER_ONLY.cmd','List_Cameras.cmd'):
            check_cmd(archive.read('PuyoLive/'+name))
            check_cmd((ENTRY/name).read_bytes())
        for name in ('README_ja.txt','USER_ONLY_DO_NOT_DISTRIBUTE.txt','OBS_TEST_README_ja.txt'):
            assert '配布禁止' in archive.read('PuyoLive/'+name).decode('utf-8-sig')
        name = 'summarize_live_obs_time_gate.py'
        assert archive.read('PuyoLive/'+name) == (ROOT/'packaging'/name).read_bytes() == (ENTRY/name).read_bytes()
        settings = json.loads(archive.read('PuyoLive/puyo_live.json'))
        assert all(settings[k] for k in ('b3_m3','live_v49f','live_ama_plain','b3_time_gate'))
        assert inventory['PuyoLive/app/models/b3/model.joblib'] == '11f7b32a26a9b70dbfda6544d64bc83695a6218531deaf25204d518704d26788'
    value = dict(passed=True,source_files=checked,unchanged_python_files=len(runtime)-len(changes),
                 python_bytecode_regenerated=changes,
                 cmd_ascii_crlf=True,private_warning=True,final_aggregator=True,zip_sha256=receipt['sha256'])
    path = RELEASE/'archive_checks.json'
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
    print(value,flush=True)


if __name__ == '__main__':
    main()
