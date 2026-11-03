"""封印前の本人用ステージだけ、ソース差分を反映してmanifestを確定する。"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_live_obs_20261020 import ROOT, BUNDLE, OUT, ASSETS, OLD, settings, sha
from build_bundle import collect_app_files,source_of,precompile,write_manifest


def main() -> None:
    assert BUNDLE.resolve() == Path('D:/puyo_analyzer/packaging/build_20261020a_work/PuyoLive').resolve()
    assert not (OUT/'archive_receipt.json').exists(), '封印後のステージを変更しない'
    changed, compiled = [], []
    for relative in collect_app_files(ASSETS/'baseline_b/app'):
        if Path(relative).suffix not in ('.py','.js','.css','.html'):
            continue
        source, target = source_of(relative,ASSETS/'baseline_b/app'), BUNDLE/'app'/relative
        if not target.exists() or source.read_bytes()!=target.read_bytes():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
            changed.append(relative)
            if target.suffix=='.py':
                compiled.append(target)
    if compiled:
        precompile(BUNDLE/'python',*compiled)
    (BUNDLE/'puyo_live.json').write_text(json.dumps(settings(),ensure_ascii=False,indent=2),encoding='utf-8')
    files = [p for p in (BUNDLE/'app').rglob('*') if p.is_file() and p.name!='MANIFEST.json']
    write_manifest(BUNDLE,files)
    native = list((BUNDLE/'python/Lib/site-packages/puyo_core').glob('*.pyd'))
    assert len(native)==1
    assert sha(native[0])==sha(next((OLD/'python/Lib/site-packages/puyo_core').glob('*.pyd')))
    receipt = dict(changed=changed,settings=settings(),native_sha256=sha(native[0]),
        m3_sha256=sha(BUNDLE/'app/models/b3/model.joblib'),
        ama_sha256=sha(BUNDLE/'app/models/ama_plain/ama_reference.exe'),
        exact_dll_sha256=sha(BUNDLE/'app/src/phase_j/puyo_live_exact.dll'))
    (OUT/'final_stage_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(receipt,flush=True)


if __name__ == '__main__':
    main()
