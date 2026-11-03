"""同値確認したソースだけが21cへ入り、配布設定・モデル・実行系が不変か検収する。"""
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path('D:/puyo_analyzer/packaging')
WORK = ROOT/'build_20261021c_work'


def main() -> None:
    inventories = [json.loads((ROOT/f'build_20261021{v}_release/files_sha256.json').read_text())
                   for v in ('b', 'c')]
    before, after = inventories
    source = json.loads((WORK/'candidate_source_receipt.json').read_text())['source_sha256']
    allowed = {'PuyoLive/app/'+name for name in source}
    allowed |= {'PuyoLive/app/'+str(Path(name).parent/'__pycache__'/
                (Path(name).stem+'.cpython-312.pyc')).replace('\\', '/') for name in source}
    changed = {name for name, digest in after.items() if before.get(name) != digest}
    app_changed = {name for name in changed if name.startswith('PuyoLive/app/')}
    unexpected = app_changed-allowed-{'PuyoLive/app/MANIFEST.json'}
    assert not unexpected, sorted(unexpected)
    assert not before.keys()-after.keys(), '旧配布のファイルが欠落'
    for name, digest in source.items():
        assert after['PuyoLive/app/'+name] == digest, name
    bundles = [ROOT/f'build_20261021{v}_stage/PuyoLive' for v in ('b', 'c')]
    configs = [(b/'puyo_live.json').read_bytes() for b in bundles]
    assert configs[0] == configs[1], '配布設定が異なる'
    immutable = [n for n in before if n.startswith(('PuyoLive/python/', 'PuyoLive/LICENSES/',
                 'PuyoLive/app/models/')) or n.endswith(('.pyd', '.dll', '.exe'))]
    assert all(before[n] == after[n] for n in immutable)
    value = dict(passed=True, configuration_sha256=sha256(configs[0]).hexdigest(),
        unchanged_runtime_model_license_files=len(immutable), changed=sorted(changed),
        source_sha256=source)
    (WORK/'bundle_verification.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
