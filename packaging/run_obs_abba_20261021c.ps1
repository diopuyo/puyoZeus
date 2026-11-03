param([int]$StartAt=0)
$ErrorActionPreference='Stop'
$work='D:/puyo_analyzer/packaging/abba_20261021_bc'
$env:TEMP="$work/temp"
$env:TMP=$env:TEMP
$env:PYTHONIOENCODING='utf-8'
$utf8=[Text.UTF8Encoding]::new($false)
Set-Location -LiteralPath '<work>/wt_live_v49f'
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
$current=Get-Process -Id $PID
[IO.File]::WriteAllText("$work/controller_environment.json",
    (@{at=(Get-Date).ToString('o');affinity=$current.ProcessorAffinity.ToInt64();
       priority=[string]$current.PriorityClass;processors=[Environment]::ProcessorCount} | ConvertTo-Json),$utf8)
$runs=@(@{name='A1_b';version='b'},@{name='B1_c';version='c'},
    @{name='B2_c';version='c'},@{name='A2_b';version='b'})
$deadline=[DateTime]::Parse('2026-10-21T19:30:00')
foreach ($run in ($runs | Select-Object -Skip $StartAt)) {
    # 重い他処理は止めず、shellのsleepで5分ごとに空きを確認する。
    while ($true) {
        & powershell -NoProfile -File packaging/wait_live_obs_idle.ps1 -Output "$work/idle_checks.jsonl" -SingleCheck
        if ($LASTEXITCODE -ne 0) { throw '空き資源の確認に失敗' }
        $last=Get-Content "$work/idle_checks.jsonl" -Tail 1 | ConvertFrom-Json
        if ($last.busy -eq 0) { break }
        if ((Get-Date) -ge $deadline) {
            [IO.File]::WriteAllText("$work/measurement_deferred.json",($last|ConvertTo-Json -Depth 8),$utf8)
            [IO.File]::WriteAllText("$work/monitor.stop",(Get-Date).ToString('o'),$utf8)
            exit 0
        }
        Start-Sleep -Seconds 300
    }
    [IO.File]::WriteAllText("$work/monitor.stop.run",$run.name,$utf8)
    $bundle="D:/puyo_analyzer/packaging/build_20261021$($run.version)_stage/PuyoLive"
    $argsRun=@('-B','-u','packaging/run_obs_abba_trial.py','--bundle',$bundle,
        '--out',"$work/$($run.name)",'--duration','1200','--start-sec','0',
        '--startup-timeout','1800','--benchmark','--quiet-throughout','--port','18855',
        '--video','C:/puyo1_tools/soak/media/video_d04.mp4',
        '--ffmpeg-dll-dir','<work>/media_tmp/b2_cv_20261020/cv2',
        '--verified-runtime','<work>/build_20261019a_stage/PuyoLive/python',
        '--fast-audit','--exact-seed','--stdlib-cache')
    if ($run.version -eq 'c') { $argsRun+=@('--prepare-exact','--deep-timing') }
    $ErrorActionPreference='Continue'
    & <work>/build_20261019a_stage/PuyoLive/python/python.exe @argsRun *> "$work/$($run.name).log"
    $result=$LASTEXITCODE
    $ErrorActionPreference='Stop'
    [IO.File]::WriteAllText("$work/$($run.name).exit.json",
        (@{exit_code=$result;at=(Get-Date).ToString('o')} | ConvertTo-Json),$utf8)
    if (-not (Test-Path "$work/$($run.name)/receipt.json") -or
        (Get-Content "$work/$($run.name)/receipt.json" -Raw | ConvertFrom-Json).exit_code -ne 0) {
        throw "ABBA未完了: $($run.name) exit=$result"
    }
}
[IO.File]::WriteAllText("$work/monitor.stop",(Get-Date).ToString('o'),$utf8)
