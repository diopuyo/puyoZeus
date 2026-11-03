param([string]$RunName='delivery_r1', [switch]$CpuIsolation)
trap {
    [System.IO.File]::WriteAllText("D:/puyo_analyzer/packaging/build_20261020d_work/$RunName.launch_error.txt", ($_ | Out-String))
    exit 1
}
$ErrorActionPreference='Stop'
$self=Get-Process -Id $PID
$self.ProcessorAffinity=[IntPtr]61440
$self.PriorityClass='BelowNormal'
$OutputEncoding=[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false)
$taskWork='D:/puyo_analyzer/packaging/build_20261020d_work'
$env:TEMP="$taskWork/temp"
$env:TMP=$env:TEMP
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
Remove-Item Env:PUYO_OBS_TRACE_DIRECTORY -ErrorAction SilentlyContinue
Set-Location -LiteralPath '<work>/wt_live_v49f'
$python='<work>/build_20261019a_stage/PuyoLive/python/python.exe'
$arguments=@('-u', 'packaging/finish_live_obs_delivery.py', '--name', $RunName)
if ($CpuIsolation) { $arguments += '--cpu-isolation' }
$ErrorActionPreference='Continue'
& $python @arguments *> "$taskWork/$RunName.log"
$result=$LASTEXITCODE
$ErrorActionPreference='Stop'
[System.IO.File]::WriteAllText("$taskWork/$RunName.exit.json",
    (@{exit_code=$result;at=(Get-Date).ToString('o')} | ConvertTo-Json))
exit $result
