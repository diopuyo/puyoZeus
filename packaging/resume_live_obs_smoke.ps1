param([Parameter(Mandatory=$true)][string]$RunName, [int]$Port=18838)
trap {
    [System.IO.File]::WriteAllText("D:/puyo_analyzer/packaging/build_20261020d_work/$RunName.launch_error.txt", ($_ | Out-String))
    exit 1
}
$ErrorActionPreference='Stop'
$self=Get-Process -Id $PID
$self.ProcessorAffinity=[IntPtr]61440
$self.PriorityClass='BelowNormal'
$OutputEncoding=[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$taskRoot='<work>/wt_live_v49f'
$taskWork='D:/puyo_analyzer/packaging/build_20261020d_work'
$env:TEMP="$taskWork/temp"
$env:TMP=$env:TEMP
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:PUYO_OBS_TRACE_DIRECTORY='1'
Set-Location -LiteralPath $taskRoot
$runPath=Join-Path $taskWork $RunName
if (Test-Path -LiteralPath $runPath) { throw '既存の検収原票は上書きしません' }
$python='<work>/build_20261019a_stage/PuyoLive/python/python.exe'
$ErrorActionPreference='Continue'
& $python -u packaging/smoke_live_obs_20261020.py `
    --bundle <work>/build_20261020c_validation/PuyoLive `
    --out $runPath --source-overlay $taskRoot --source-plan "$taskWork/reuse_source_plan.json" `
    --duration 240 --startup-timeout 900 --video C:/puyo1_tools/soak/media/video_d04.mp4 `
    --start-sec 600 --ffmpeg-dll-dir <work>/media_tmp/b2_cv_20261020/cv2 `
    --port $Port --fast-audit --exact-seed *> "$taskWork/$RunName.log"
$result=$LASTEXITCODE
$ErrorActionPreference='Stop'
[System.IO.File]::WriteAllText("$taskWork/$RunName.exit.json",
    (@{exit_code=$result;at=(Get-Date).ToString('o')} | ConvertTo-Json))
exit $result
