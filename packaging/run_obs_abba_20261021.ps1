param([int]$StartAt=0, [string]$Work='D:/puyo_analyzer/packaging/abba_20261021_default_r2')
$ErrorActionPreference='Stop'
$work=$Work
$env:TEMP="$work/temp"
$env:TMP=$env:TEMP
$env:PYTHONIOENCODING='utf-8'
Set-Location -LiteralPath '<work>/wt_live_v49f'
$python='<work>/build_20261019a_stage/PuyoLive/python/python.exe'
$f='D:/puyo_analyzer/packaging/build_20261021b_stage/PuyoLive'
$c='D:/puyo_analyzer/packaging/abba_20261021/c_bundle/PuyoLive'
$utf8=[Text.UTF8Encoding]::new($false)
$current=Get-Process -Id $PID
[IO.File]::WriteAllText("$work/controller_environment.json",
    (@{at=(Get-Date).ToString('o');affinity=$current.ProcessorAffinity.ToInt64();
       priority=[string]$current.PriorityClass;processors=[Environment]::ProcessorCount} | ConvertTo-Json),$utf8)
$runs=@(@{name='A1_f_fixed';bundle=$f;fast=$true},@{name='B1_c';bundle=$c;fast=$false},
    @{name='B2_c';bundle=$c;fast=$false},@{name='A2_f_fixed';bundle=$f;fast=$true})
foreach ($run in ($runs | Select-Object -Skip $StartAt)) {
    [IO.File]::WriteAllText("$work/monitor.stop.run",$run.name,$utf8)
    $argsRun=@('-B','-u','packaging/run_obs_abba_trial.py','--bundle',$run.bundle,
        '--out',"$work/$($run.name)",'--duration','1200','--start-sec','0',
        '--startup-timeout','1800','--benchmark','--port','18851',
        '--video','C:/puyo1_tools/soak/media/video_d04.mp4',
        '--ffmpeg-dll-dir','<work>/media_tmp/b2_cv_20261020/cv2',
        '--verified-runtime','<work>/build_20261019a_stage/PuyoLive/python')
    if ($run.fast) { $argsRun+=@('--fast-audit','--exact-seed','--stdlib-cache') }
    $ErrorActionPreference='Continue'
    & $python @argsRun *> "$work/$($run.name).log"
    $result=$LASTEXITCODE
    $ErrorActionPreference='Stop'
    [IO.File]::WriteAllText("$work/$($run.name).exit.json",
        (@{exit_code=$result;at=(Get-Date).ToString('o')} | ConvertTo-Json),$utf8)
    # 表示異常は完走した原票として残す。開始できなかった場合は停止する。
    if (-not (Test-Path "$work/$($run.name)/receipt.json") -or
        (Get-Content "$work/$($run.name)/receipt.json" -Raw | ConvertFrom-Json).exit_code -ne 0) {
        throw "ABBA trial failed before receipt: $($run.name) exit=$result"
    }
}
[IO.File]::WriteAllText("$work/monitor.stop",(Get-Date).ToString('o'),$utf8)
