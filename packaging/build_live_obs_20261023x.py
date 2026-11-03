"""21cを保存し、配信の X (段0・1・3、既定OFF) のソース差分だけを足した OBS 試験版を新規封印する。

組み立ての手順・設定・入口は 21c (build_live_obs_20261021c.py) をそのまま継承する。
21c からの差は、ソース (76fd2146) の中身の差だけにする。改行だけが違うファイルは 21c の bytes を保つ。
  extract: 21c ZIP を新しいステージへ展開 (21c の組み立て物は読むだけ)
  refresh: 変わったソースだけを差し替え、pyc・MANIFEST・入口を作り直す
  seal:    ZIP を封印し、全 member の bytes 一致を確かめ、起動用の入口 (OFF と X ON) を作る
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
import build_live_obs_20261021c as previous  # 21c の設定・環境変数・手順書の追記を継承する

build = previous.build
OUTPUT = Path('D:/puyo_analyzer/packaging/build_20261023_obs_x')
WORK = OUTPUT/'work'
PRESERVED = WORK/'preserved_sources'
SOURCE_COMMIT = '76fd2146'
BASE_COMMIT = 'a1e2321d'        # 21c 配布 app と同じソース (改行以外) の起点
OLD_ZIP_SHA256 = '2a51c0db480e3cc67f916a592289481d5216b116ab58b35970d35569eac01c18'
X_ON_DIR = 'x_on'
X_FLAGS = ('live_x_direct_publish', 'live_x_prefire_shadow', 'live_x_prefire')
OFF_START = 'Start_PuyoLive_USER_ONLY.cmd'
ON_START = 'Start_PuyoLive_X_ON_USER_ONLY.cmd'
CAMERA_RESOLVER = 'resolve_camera.ps1'
CR = b'\r'

build.BUILD = 'build_20261023_obs_x'
build.OUT = OUTPUT/'release'
build.BUNDLE = OUTPUT/'stage/PuyoLive'
build.FAST = build.BUNDLE
build.WORK = WORK
build.OLD_ZIP = Path('D:/puyo_analyzer/packaging/build_20261021c_release/'
    'PuyoLive_20261021c_USER_ONLY_DO_NOT_DISTRIBUTE.zip')
build.ENTRY = OUTPUT/'PuyoLive_OBS_20261023x_USER_ONLY'
original_source_of = build.source_of
guide_before, populate_before = build.launchers.guide, build.launchers.populate
_old_archive: list[zipfile.ZipFile] = []
eol_preserved: list[str] = []
blob_sourced: list[str] = []


def old_member(relative: str) -> bytes | None:
    """21c ZIP の app member。ネットワーク上の ZIP へ細かな seek をしないよう 1 回だけ読み込む。"""
    if not _old_archive:
        _old_archive.append(zipfile.ZipFile(BytesIO(build.OLD_ZIP.read_bytes())))
    name = 'PuyoLive/app/'+relative
    archive = _old_archive[0]
    return archive.read(name) if name in archive.NameToInfo else None


def git_blob(relative: str) -> bytes | None:
    """commit に記録された bytes (改行の変換前)。git 管理外なら None。"""
    result = subprocess.run(['git', '-C', str(ROOT), 'show', f'{SOURCE_COMMIT}:{relative}'],
                            capture_output=True)
    return result.stdout if result.returncode == 0 else None


def canonical_bytes(relative: str, asset_root: Path) -> tuple[Path, bytes]:
    """配布に入れる bytes。git 管理のソースは commit の blob、改行だけの差は 21c の bytes を保つ。"""
    source = original_source_of(relative, asset_root)
    data, old = source.read_bytes(), old_member(relative)
    if data == old:
        return source, data
    if source.is_relative_to(ROOT) and (blob := git_blob(relative)) is not None and blob != data:
        data = blob
        blob_sourced.append(relative)
    if old is not None and old != data and old.replace(CR, b'') == data.replace(CR, b''):
        eol_preserved.append(relative)
        data = old
    return source, data


def preserving_source_of(relative: str, asset_root: Path) -> Path:
    source, data = canonical_bytes(relative, asset_root)
    if source.read_bytes() == data:
        return source
    target = PRESERVED/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.read_bytes() != data:
        target.write_bytes(data)
    return target


X_NOTE = '''
23x: 撃った連鎖の火力 X を早く出す試験版です。既定はすべて OFF で、21c と同じ動作です。
ON にするのは puyo_live.json の次の 3 項目です (true / false)。
  "live_x_direct_publish": 段1 X が初めて載った結果を、次の通知を待たずにすぐ公開します。
  "live_x_prefire_shadow": 段3 撃つ直前の暫定 X を記録だけします (output/日時/live_x_prefire.jsonl)。
  "live_x_prefire": 段3 暫定 X を v4.9f の枠に文字 1 行で出します (記録も行います)。
段3 の 2 項目は live_v49f が true のときだけ使えます。
B3・v4.9f・ama の勝率の値は変えません。暫定 X は別欄 x_provisional で、v4.9f の p1 には入れません。
環境変数 PUYO_LIVE_X_* は起動時に puyo_live.json の値で上書きされます。切り替えは puyo_live.json で行います。
'''


def entry_guide(bundle: Path) -> str:
    """D: の入口に置く本人試験の手順書 (21c の入口の形を踏襲し、OFF/ON の ABBA を足す)。"""
    python = str(bundle/'python/python.exe')
    archive = build.OUT/f'PuyoLive_{build.BUILD.removeprefix("build_")}_USER_ONLY_DO_NOT_DISTRIBUTE.zip'
    return ENTRY_GUIDE.replace('{BUILD}', build.BUILD).replace('{BUNDLE}', str(bundle)).replace(
        '{ZIP}', str(archive)).replace('{PYTHON}', python).replace('{ON_DIR}', X_ON_DIR).replace(
        '{OFF}', OFF_START).replace('{ON}', ON_START)+X_NOTE


ENTRY_GUIDE = _public_dev_text('OBS 本人試験専用 {BUILD} — 配布禁止\npuyoai-private 由来の ama を含みます。user 本人の試験専用です。第三者へ渡さないでください。\nこの入口が起動する検収済み本体: {BUNDLE}\n本体は {ZIP} を展開したもので、全ファイルが ZIP と一致することを確認済みです。展開し直す必要はありません。\n\n今回の試験の目的\n  撃った連鎖の火力 X を v4.9f の枠へ早く出す仕組み (段1) と、撃つ直前の暫定 X の 1 行 (段3) を、\n  実際の OBS で OFF と ON を交互に流して比べます。B3・v4.9f・ama の勝率の値の出し方は変えていません。\n  OFF は前回 21c と同じ動作です。\n\n起動口は 2 つです。\n  {OFF}       … X OFF (前回 21c と同じ)\n  {ON}  … X ON (段1 直接公開 + 段3 暫定 X の記録と 1 行表示)\n  記録は OFF がこのフォルダの output\\日時、ON が {ON_DIR}\\output\\日時 に分かれて残ります。\n\n準備\n1. 始める前に、PC1 の Claude に「OBS 試験を始める」と伝えてください。\n   Claude が PC1 の重い計算 (M4) を一時停止し、空きメモリを確かめてから「どうぞ」と返します。\n   別の PuyoLive は終了してください。\n2. OBS を起動し、1920×1080 のゲーム映像を映します。\n   仮想カメラはゲーム映像のソースを選び、分析表示を映り込ませないでください。\n   前回と同じく OBS の録画も回してください (後で同じ映像を流し直して比べるためです)。\n3. OBS の「仮想カメラ開始」を押します。\n4. カメラ番号を実際の映像で確かめます (試験の前と、OBS を再起動したときは毎回)。\n   List_Cameras.cmd をダブルクリックすると、最初に機器の一覧 (index 番号: 名前) が出ます。\n   「OBS Virtual Camera」の番号を入力し、ゲーム映像が映ることを確かめて ESC で閉じます。\n   起動口は毎回、名前から番号を自動で合わせます (黒い画面の最初に「OBS Virtual Camera -> device_index N」)。\n   一覧で確かめた番号と N が同じことも見てください。見つからないときは止まります。\n   起動順は OBS → 仮想カメラ → PuyoLive です。\n\n試験の順番 (ABBA、合計 約 2 時間)\n  1 本目 OFF 20 分 → 2 本目 ON 20 分 → 3 本目 ON 55 分 → 4 本目 OFF 20 分\n  OBS と仮想カメラと録画は最後まで付けたままにし、PuyoLive だけを止めて起動し直します。\n  時間は表示が出てからの時間です。20 分を過ぎたら、その試合が終わったところで止めてかまいません。\n5. 各回の始め方: 起動口をダブルクリックします。黒い画面は開いたままにします。\n   初回はファイル検証とモデル読込みに時間がかかります。黒い画面を閉じず、起動完了を待ちます。\n   http://127.0.0.1:8765/ をブラウザで開きます。OBS ブラウザソースも同じ URL です。\n6. 見るところは前回と同じ三つの枠です。\n   実戦勝率: B3 (M3) / 撃ち合い勝率 v4.9f / ama 想定の勝率。\n   ON の回だけ、v4.9f の枠に「暫定 X (1P): 7連鎖 15,360点 確定待ち」のような 1 行が出ることがあります。\n   撃つ直前に見込んだ火力で、勝率には入れていません。\n   撃たなかったのに出た・大きく外れた・出なかった、と気づいたら、時刻 (時計か録画の時刻) を控えてください。\n7. 各回の終わり方: 黒い画面で Ctrl+C を押し、終了を待ちます。\n   次の回は、そのまま次の起動口をダブルクリックします。\n8. 4 本終わったら、仮想カメラを停止し、録画を止めます。OBS の終了は最後です。\n9. 終わったら PC1 の Claude に「OBS 試験の記録を集計して」と伝え、次を知らせてください。\n   - 録画ファイルの場所 (例 <home>\\Videos\\日時.mp4)\n   - 暫定 X の 1 行で気づいたこと (外れ・出なかった・早すぎた等) と、その時刻\n   - 体感 (表示の早さ・カクつき)\n   記録 (output と {ON_DIR}\\output) は D: にあるので、送る必要はありません。\n   時間が取れないときの短い版 (約 80 分): OFF 20 → ON 20 → ON 20 → OFF 20。\n\n集計 (PC1 Claude 向け。PowerShell で、この手順書のあるフォルダを作業場所にする)\n0. 物差しの健全性を先に示す: wt_live_x_obsprep で\n   & "{PYTHON}" -B scripts\\measure_live_x_timing.py known\n   21c の既知値 M3 P50 0.7039574 / P95 0.9593628 秒・n=51、M1=4・M2=30 を小数まで再現してから使う。\n1. 各回の時間門と実遅延 (前回と同じ。ON は {ON_DIR}\\output\\日時):\n   & "{PYTHON}" summarize_live_obs_time_gate.py output\\日時 --out summary_日時.json\n   & "{PYTHON}" summarize_live_obs_actual_latency.py output\\日時\n   処理時間・fps・P50/P95 は各回の先頭 20 分、P99・最大・後半の悪化・メモリは 3 本目 (ON 55 分) の全長。\n2. 段1 の M3 (発火通知 → 当該 X の公開): measure_live_x_timing.py score --run 各回の記録 --placement wall\n   ON の線は P50 650ms 以下かつ P95 900ms 以下 (設計 live_x_timing_20261022 §5)。\n   場面表 (--actual) は録画ごとに要る。既定の表は 21c の録画専用。\n3. 起床の記録: ON の v49f_display_status.json の x_fresh・x_fresh_fires・x_wakes。\n   増える公開は発火 1 回あたり 1 件以内 (同 §9)。\n4. 段3: ON の live_x_prefire.jsonl・live_x_prefire_summary.json。公開に載った誤発火 2 件以下/45 試合、\n   取り下げまで P95 1.0 秒以下 (同 §5)。measure_live_x_effects.py misfire の実連鎖の表も 21c 専用。\n集計は30分未満を合格扱いしません。正式な時間門の採否はPC1 Claudeが判断します。\n本体を移動・削除すると D: の入口から起動できなくなります。\n本線の production_config と既定は変更していません。採否・push は PC1 Claude に委ねています。\n')


def guide(*args: object, **kwargs: object) -> str:
    bundle = args[0] if args else Path('.')
    external = args[1] if len(args) > 1 else kwargs.get('external', False)
    if external:
        return entry_guide(Path(str(bundle)))
    return guide_before(*args, **kwargs)+X_NOTE


def start_lines(bundle: Path, entry: str, resolver: str, label: str) -> list[str]:
    """21c の入口の最終形 (名前でカメラ番号を合わせ、ネットワークドライブでも実行方針で拒否されない)。"""
    return ['@echo off', 'setlocal', 'echo USER ONLY - DO NOT DISTRIBUTE', f'echo {label}',
            f'powershell -NoProfile -ExecutionPolicy Bypass -File "{resolver}"',
            'if errorlevel 1 (', '  pause', '  exit /b 1', ')',
            f'powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_obs.ps1" -Bundle "{bundle}" '
            f'-Entry "{entry}"', 'if errorlevel 1 pause']


def add_entry_launchers(folder: Path, bundle: Path) -> None:
    """D: の入口だけに、X OFF (21c と同じ) と X ON の起動口、名前でのカメラ番号合わせを置く。"""
    resolver = ROOT/'packaging'/CAMERA_RESOLVER
    shutil.copy2(resolver, folder/CAMERA_RESOLVER)
    launch = build.launchers.cmd
    launch(folder/OFF_START, start_lines(bundle, '%~dp0.', f'%~dp0{CAMERA_RESOLVER}', 'X OFF (same as 21c)'))
    launch(folder/ON_START, start_lines(bundle, f'%~dp0{X_ON_DIR}', f'%~dp0{X_ON_DIR}\\{CAMERA_RESOLVER}',
                                        'X ON (live_x_direct_publish + live_x_prefire_shadow + live_x_prefire)'))
    launch(folder/'List_Cameras.cmd', ['@echo off', 'setlocal', 'echo USER ONLY - DO NOT DISTRIBUTE',
        f'powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0{CAMERA_RESOLVER}" -ListOnly',
        f'powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_obs.ps1" -Bundle "{bundle}" '
        '-Entry "%~dp0." -Camera', 'if errorlevel 1 pause'])
    on = folder/X_ON_DIR
    on.mkdir()
    values = json.loads((folder/'puyo_live.json').read_text(encoding='utf-8'))
    assert not any(values.get(key) for key in X_FLAGS), 'OFF の設定に X のフラグが入っている'
    values.update({key: True for key in X_FLAGS})
    (on/'puyo_live.json').write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(resolver, on/CAMERA_RESOLVER)


def populate(folder: Path, bundle: Path, external: bool) -> None:
    populate_before(folder, bundle, external)
    if external:
        add_entry_launchers(folder, bundle)


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)


def check_source() -> dict:
    """配布に入る src・scripts が SOURCE_COMMIT と同じで、worktree に未記録の変更が無いことを確かめる。"""
    head = git('rev-parse', 'HEAD').stdout.strip()
    assert git('merge-base', '--is-ancestor', SOURCE_COMMIT, head).returncode == 0, (SOURCE_COMMIT, head)
    assert git('diff', '--quiet', SOURCE_COMMIT, head, '--', 'src', 'scripts').returncode == 0, 'src・scripts が commit と違う'
    dirty = git('status', '--porcelain', '--', 'src', 'scripts', 'packaging').stdout.strip()
    assert not dirty, dirty
    assert sha256(build.OLD_ZIP.read_bytes()).hexdigest() == OLD_ZIP_SHA256, '21c ZIP の SHA が違う'
    return dict(head=head, source_commit=SOURCE_COMMIT, base_commit=BASE_COMMIT, old_zip_sha256=OLD_ZIP_SHA256)


def refresh() -> None:
    receipt = check_source()
    previous.previous.refresh()   # 21c の main と同じく 21b の refresh (HELPERS の複写込み) を呼ぶ
    WORK.mkdir(parents=True, exist_ok=True)
    receipt.update(eol_preserved=sorted(set(eol_preserved)), blob_sourced=sorted(set(blob_sourced)))
    (WORK/'source_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


build.source_of = preserving_source_of
build.launchers.guide = guide
build.launchers.populate = populate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('extract', 'refresh', 'seal'))
    action = parser.parse_args().action
    if action == 'extract':
        from bundle_archive import extract_smoke_copy
        assert sha256(build.OLD_ZIP.read_bytes()).hexdigest() == OLD_ZIP_SHA256
        extract_smoke_copy(build.OLD_ZIP, build.BUNDLE.parent, preload=True)
    elif action == 'refresh':
        refresh()
    else:
        build.seal()
