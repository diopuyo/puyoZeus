param([string]$SmokeName='resume_null_check_r7', [string]$RunName='comparison_after_r7')
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
Remove-Item Env:PUYO_OBS_TRACE_DIRECTORY -ErrorAction SilentlyContinue
Set-Location -LiteralPath $taskRoot
$done="$taskWork/$SmokeName/receipt.json"
$deadline=(Get-Date).AddMinutes(30)
# セッション停止と分離したsmokeの正常終了を待ち、途中原票を速度測定へ流用しない。
while (-not (Test-Path -LiteralPath $done)) {
    if ((Get-Date) -gt $deadline) {
        [System.IO.File]::WriteAllText("$taskWork/$RunName.blocked.txt", '先行smokeの終了票なし。比較は開始していません。')
        exit 1
    }
    Start-Sleep -Seconds 10
}
$smoke=Get-Content -LiteralPath $done -Raw | ConvertFrom-Json
if ($smoke.exit_code -ne 0 -or $smoke.all_three -le 0 -or $smoke.evaluation_errors -ne 0 -or $smoke.ama_worker_fatal) {
    [System.IO.File]::WriteAllText("$taskWork/$RunName.blocked.txt", '先行smoke不合格。比較は開始していません。')
    exit 1
}
$runPath=Join-Path $taskWork $RunName
if (Test-Path -LiteralPath $runPath) { throw '既存の比較原票は上書きしません' }
New-Item -ItemType Directory -Path $runPath | Out-Null
[System.IO.File]::WriteAllText("$runPath/wait_started_at.txt", '2026-10-20T12:06:32+09:00')
$python='<work>/build_20261019a_stage/PuyoLive/python/python.exe'
$ErrorActionPreference='Continue'
& $python -u packaging/run_live_obs_20261020d.py --formal `
    --bundle <work>/build_20261020c_validation/PuyoLive --work $runPath `
    --source-overlay $taskRoot --source-plan "$taskWork/reuse_source_plan.json" `
    --startup-timeout 900 --ffmpeg-dll-dir <work>/media_tmp/b2_cv_20261020/cv2 `
    --cases fast_off seed seed_isolated *> "$taskWork/$RunName.log"
$result=$LASTEXITCODE
$ErrorActionPreference='Stop'
[System.IO.File]::WriteAllText("$taskWork/$RunName.exit.json",
    (@{exit_code=$result;at=(Get-Date).ToString('o')} | ConvertTo-Json))
exit $result
