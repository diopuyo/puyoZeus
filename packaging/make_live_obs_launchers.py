"""本人試験専用のASCII/CRLF入口と、毎回独立した記録先を作る。"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
from make_live_v49f_launchers import CAMERA_SCRIPT

BUNDLE = Path('D:/puyo_analyzer/packaging/build_20261020a_work/PuyoLive')
OUT = Path('D:/PuyoLive_OBS_20261020_USER_ONLY')
WRAPPER = '''from datetime import datetime
import ctypes
import json
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
    handle = kernel.GetCurrentProcess()
    assert kernel.SetProcessAffinityMask(handle,255)
    assert kernel.SetPriorityClass(handle,0x4000)
    template = Path(sys.argv[1]).resolve()
    bundle = Path(sys.argv[2]).resolve()
    values = json.loads(template.read_text(encoding='utf-8-sig'))
    run = template.parent/'output'/datetime.now().strftime('%Y%m%dT%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    for key, name in [('b3_model','model.joblib'),('b3_timing_table','timing_table.json')]:
        values[key] = str(bundle/'app/models/b3'/name)
    values['output_dir'] = str(run)
    config = run/'puyo_live.json'
    config.write_text(json.dumps(values,indent=2),encoding='utf-8')
    print('USER ONLY - DO NOT DISTRIBUTE', flush=True)
    print('Records: '+str(run), flush=True)
    sys.argv = ['launcher','--config',str(config),'--require-manifest']
    runpy.run_module('src.phase_j.launcher',run_name='__main__')
'''


def cmd(path: Path, lines: list[str]) -> None:
    value = ('\r\n'.join(lines)+'\r\n').encode('ascii')
    path.write_bytes(value)
    assert b'\n' not in value.replace(b'\r\n',b'')


def guide(bundle: Path = Path('.'), external: bool = False) -> str:
    python = str(bundle/'python/python.exe') if external else 'python\\python.exe'
    return '''OBS 本人試験専用 build_20261020a — 配布禁止
puyoai-private 由来の ama を含みます。user 本人の試験専用です。第三者へ渡さないでください。
{INSTALL_NOTE}

1. 試験前にPC1の重い処理を止めてください。別のPuyoLiveも終了してください。
2. OBSを起動し、1920×1080のゲーム映像を映します。
   仮想カメラはゲーム映像のソースを選び、分析表示を映り込ませないでください。
3. OBSの「仮想カメラ開始」を押します。
4. List_Cameras.cmdを開き、候補番号を入力して映像を確認します。ESCでプレビューを閉じます。
   初期値は3ですが、番号はPCの接続状況で変わります。解像度だけで判別しないでください。
   このフォルダのpuyo_live.jsonのdevice_indexを確認した番号へ変更します。
   device_nameは名札です。機器の選択にはdevice_indexを使います。
5. Start_PuyoLive_USER_ONLY.cmdをダブルクリックします。
   起動順は OBS → 仮想カメラ → PuyoLive です。黒い画面は開いたままにします。
   初回はファイル検証とモデル読込みに時間がかかります。黒い画面を閉じず、起動完了を待ちます。
   http://127.0.0.1:8765/ をブラウザで開きます。OBSブラウザソースも同じURLです。
6. 三つの枠を確認してください。
   実戦勝率: B3 (M3) — 実戦データを学習したモデルです。
   撃ち合い勝率 v4.9f — 10/19aと同じ定義・旧v6を混合元に残した独立表示です。
   ama 想定の勝率 — 各側1回のama読みと固定の較正による別枠です。
   amaは評価できない間も直前値を保持します。「秒保持」と理由を一緒に見てください。
   「打てない」は安全な配置なし、「本線なし」は読みで本線を選べなかった状態です。
   「—」はその試合でまだ有効な値がありません。未評価を50%とは表示しません。
7. 30分以上流してください。開始直後から試合の区切りを含め、普段のOBS構成で試します。
   CPU・メモリ・更新時間・待ち行列・null・保持などを自動記録します。
8. 止めるときは黒い画面でCtrl+Cを押し、終了を待ちます。
   次にOBSの仮想カメラを停止します。OBSの終了は最後です。
9. 終わったら「OBS試験の記録を集計して」とPC1のClaudeに頼んでください。
   記録はこのフォルダのoutput/日時です。複数回の試験を上書きしません。

集計例（PC1 Claude向け。PowerShellで、この手順書のあるフォルダを作業場所にする）:
& "{PYTHON}" summarize_live_obs_time_gate.py output/日時 --out summary_日時.json
集計は30分未満を合格扱いしません。正式な時間門の採否はPC1 Claudeが判断します。
ブラウザpaintはサーバー公開時刻に含まれません。原票を残して解析してください。
本体を移動・削除するとD:の入口から起動できなくなります。
本線のproduction_configと既定は変更していません。採否・pushはPC1 Claudeに委ねています。
'''.replace('{PYTHON}',python).replace('{INSTALL_NOTE}',
    'この入口が起動する検収済み本体: '+str(bundle) if external else '')


def populate(folder: Path, bundle: Path, external: bool) -> None:
    folder.mkdir(exist_ok=not external)
    common = ['@echo off','setlocal','set PYTHONUTF8=1',f'cd /d "{bundle}"']
    command = '"python\\python.exe" "%~dp0start_obs_trial.py" "%~dp0puyo_live.json" "'+str(bundle)+'"'
    cmd(folder/'Start_PuyoLive_USER_ONLY.cmd',common+[
        'echo USER ONLY - DO NOT DISTRIBUTE','echo Start OBS Virtual Camera first.',
        'echo Overlay: http://127.0.0.1:8765/','echo Press Ctrl+C to stop.',command,'if errorlevel 1 pause'])
    cmd(folder/'List_Cameras.cmd',common+['"python\\python.exe" "%~dp0camera_indices.py"','pause'])
    (folder/'start_obs_trial.py').write_text(WRAPPER,encoding='ascii')
    (folder/'camera_indices.py').write_text(CAMERA_SCRIPT,encoding='ascii')
    (folder/'OBS_TEST_README_ja.txt').write_bytes(b'\xef\xbb\xbf'+guide(bundle,external).replace('\n','\r\n').encode('utf-8'))
    shutil.copy2(Path(__file__).with_name('summarize_live_obs_time_gate.py'),folder/'summarize_live_obs_time_gate.py')
    if external:
        shutil.copy2(BUNDLE/'puyo_live.json',folder/'puyo_live.json')


def main() -> None:
    from build_live_obs_20261020 import settings
    values = settings()
    (BUNDLE/'puyo_live.json').write_text(json.dumps(values,ensure_ascii=False,indent=2),encoding='utf-8')
    populate(BUNDLE,Path('%~dp0.'),False)
    populate(OUT,BUNDLE,True)
    # ZIPの通常入口も、毎回独立した本人用の記録先へ通す。
    cmd(BUNDLE/'PuyoLive.bat',['@echo off','call "%~dp0Start_PuyoLive_USER_ONLY.cmd"'])
    print(OUT,flush=True)


if __name__ == '__main__':
    main()
