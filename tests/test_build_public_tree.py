"""公開用の変換範囲・未確認パスの拒否・入力不変を検証する。"""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('public_tree_builder', ROOT/'packaging/build_public_tree.py')
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)
USER = 'private_example'
WORK = 'private_work_example'
HOME = 'C:/Users/' + USER


def policy() -> dict:
    return dict(exclude_parts=['logs','review_frames','local_frames'], artifact_prefixes=['data/verify/'], retain_artifacts={'data/verify/input.txt':{'reason':'試験入力'}}, preserve_bytes={})


def test_python_only_comments_and_docstrings() -> None:
    text = f'"""{HOME}"""\n# {HOME}\nVALUE = "{HOME}"\n'
    clean = B.sanitize('tool.py', text.encode(), B.patterns(USER, WORK)).decode()
    assert clean == f'"""<home>"""\n# <home>\nVALUE = "{HOME}"\n'
    result, report = B.plan({'tool.py': text.encode()}, policy(), B.patterns(USER, WORK))
    assert report['unreviewed'] == [{'path':'tool.py','matches':1}]
    compile(result['tool.py'], 'tool.py', 'exec')


def test_json_explanations_only() -> None:
    raw = json.dumps({'description':HOME,'path':HOME,'nested':{'notes':[HOME]}}).encode()
    clean = json.loads(B.sanitize('config.json',raw,B.patterns(USER,WORK)))
    assert clean == {'description':'<home>','path':HOME,'nested':{'notes':['<home>']}}


def test_reviewed_model_provenance_preserves_inference_fields() -> None:
    value={'models':{'file':'model.pt','sha256':'abc'},'counts':[1,2], 'training_data':{'input_sha256':{HOME+'/data':'def'}}}
    clean=json.loads(B.sanitize('model.json',json.dumps(value).encode(),B.patterns(USER,WORK),['/training_data/input_sha256']))
    assert clean['models']==value['models'] and clean['counts']==[1,2]
    assert clean['training_data']['input_sha256']=={'<home>/data':'def'}


def test_provenance_collision_is_rejected() -> None:
    value={HOME:'one','<home>':'two'}
    with pytest.raises(ValueError,match='衝突'):
        B.sanitize_provenance(value,B.patterns(USER,WORK),[''])


def test_xml_report_remains_parseable() -> None:
    raw=f'<testsuite tests="1"><failure>{HOME}</failure></testsuite>'.encode()
    clean=B.sanitize('docs/report.xml',raw,B.patterns(USER,WORK))
    import xml.etree.ElementTree as ET
    node=ET.fromstring(clean)
    assert node.attrib['tests']=='1' and node.find('failure').text=='<home>'


def test_nested_output_exclusion_and_required_input() -> None:
    files = {p:b'x' for p in ['logs/a.txt','x/review_frames/a.py','local_frames/a.json','data/verify/drop.txt','data/verify/input.txt','src/main.py']}
    result, report = B.plan(files,policy(),B.patterns(USER,WORK))
    assert set(result) == {'data/verify/input.txt','src/main.py'}
    assert len(report['excluded']) == 4


def test_preserved_fixture_requires_matching_hash() -> None:
    raw = HOME.encode()
    selected = policy()
    selected['preserve_bytes']['fixture.txt'] = {'sha256':hashlib.sha256(raw).hexdigest(),'reason':'検証fixture'}
    result, report = B.plan({'fixture.txt':raw},selected,B.patterns(USER,WORK))
    assert result['fixture.txt'] == raw
    assert report['exceptions'] == [{'path':'fixture.txt','matches':1}]
    with pytest.raises(ValueError,match='再確認'):
        B.plan({'fixture.txt':raw+b'x'},selected,B.patterns(USER,WORK))


@pytest.mark.parametrize('value', [HOME, HOME.replace('/','\\'), HOME.replace('/','\\\\'), '/home/'+USER, '/mnt/c/Users/'+USER])
def test_home_notation_and_idempotence(value: str) -> None:
    rules = B.patterns(USER,WORK)
    assert B.replace_text(value+'/file',rules) == '<home>/file'
    assert B.replace_text(B.replace_text(value,rules),rules) == '<home>'


def test_work_root_and_neighbor_user() -> None:
    rules = B.patterns(USER,WORK)
    assert B.replace_text('C:/'+WORK+'/data',rules) == '<work>/data'
    assert B.replace_text(HOME+'2/data',rules) == HOME+'2/data'


def commit(repo: Path) -> None:
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'add','.'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Test','-c','user.email=test@example.invalid','-c','commit.gpgsign=false','commit','-qm','fixture'],check=True)


def test_export_uses_commit_and_preserves_source(tmp_path: Path) -> None:
    repo = tmp_path/'repo'
    repo.mkdir()
    (repo/'README.md').write_text(HOME,encoding='utf-8')
    commit(repo)
    (repo/'README.md').write_text('uncommitted',encoding='utf-8')
    manifest = tmp_path/'policy.json'
    manifest.write_text(json.dumps(policy()),encoding='utf-8')
    output, report = tmp_path/'public',tmp_path/'report.json'
    B.export_tree(repo,'HEAD',output,report,manifest,USER,WORK)
    assert (output/'README.md').read_text() == '<home>'
    assert (repo/'README.md').read_text() == 'uncommitted'
    with pytest.raises(ValueError,match='未使用'):
        B.export_tree(repo,'HEAD',output,tmp_path/'second.json',manifest,USER,WORK)


def test_unreviewed_executable_stops_before_output(tmp_path: Path) -> None:
    repo = tmp_path/'repo'
    repo.mkdir()
    (repo/'tool.py').write_text(f'VALUE = "{HOME}"',encoding='utf-8')
    commit(repo)
    manifest = tmp_path/'policy.json'
    manifest.write_text(json.dumps(policy()),encoding='utf-8')
    output, report = tmp_path/'public',tmp_path/'report.json'
    with pytest.raises(ValueError,match='未確認'):
        B.export_tree(repo,'HEAD',output,report,manifest,USER,WORK)
    assert not output.exists()
    assert json.loads(report.read_text())['unreviewed'][0]['path'] == 'tool.py'


def test_development_paths_are_explicitly_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.public_dev_paths import resolve_text
    monkeypatch.setenv('PUYO_DEV_WORK','D:/chosen')
    monkeypatch.setenv('PUYO_DEV_HOME','D:/home')
    assert resolve_text('<work>/model') == 'D:/chosen/model'
    assert resolve_text('<home>/tool') == 'D:/home/tool'
    assert resolve_text('literal') == 'literal'


def test_pages_branch_is_orphan_and_contains_only_site_files(tmp_path: Path) -> None:
    spec=importlib.util.spec_from_file_location('pages_builder',ROOT/'packaging/build_pages_branch.py')
    pages=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pages)
    repo=tmp_path/'repo'
    (repo/'site').mkdir(parents=True)
    for name in pages.SITE_FILES:
        (repo/'site'/name).write_bytes(b'site fixture')
    (repo/'private.txt').write_text('must not publish')
    commit(repo)
    result=pages.create_branch(repo)
    git=['git','-C',str(repo)]
    assert subprocess.check_output(git+['show','-s','--format=%P',result]).strip()==b''
    assert set(subprocess.check_output(git+['ls-tree','-r','--name-only',result]).decode().splitlines())==set(pages.SITE_FILES)
    assert subprocess.check_output(git+['show','-s','--format=%ae%n%ce',result]).decode().splitlines()==[pages.PUBLIC_EMAIL]*2
    before=subprocess.check_output(git+['rev-parse','gh-pages'])
    with pytest.raises(ValueError,match='既存'):
        pages.create_branch(repo)
    assert subprocess.check_output(git+['rev-parse','gh-pages'])==before
