$ErrorActionPreference='Stop'
$self=Get-Process -Id $PID
$self.ProcessorAffinity=[IntPtr]61440
$self.PriorityClass='BelowNormal'
$taskWork='D:/puyo_analyzer/packaging/build_20261020e_work'
$env:TEMP="$taskWork/temp"
$env:TMP=$env:TEMP
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
Remove-Item Env:PUYO_OBS_TRACE_DIRECTORY -ErrorAction SilentlyContinue
Set-Location -LiteralPath '<work>/wt_live_v49f'
$python='<work>/build_20261019a_stage/PuyoLive/python/python.exe'
$ErrorActionPreference='Continue'
& $python -B -u packaging/finish_live_obs_20261020e.py *> "$taskWork/delivery_r1.log"
$result=$LASTEXITCODE
[System.IO.File]::WriteAllText("$taskWork/delivery_r1.exit.json",
    (@{exit_code=$result;at=(Get-Date).ToString('o')} | ConvertTo-Json))
exit $result