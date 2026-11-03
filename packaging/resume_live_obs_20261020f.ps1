param([string]$RunName='delivery_r1', [switch]$ResumeSeal)
$ErrorActionPreference='Stop'
$self=Get-Process -Id $PID
$self.ProcessorAffinity=[IntPtr]61440
$self.PriorityClass='BelowNormal'
$taskWork='D:/puyo_analyzer/packaging/build_20261020f_work'
$env:TEMP="$taskWork/temp"
$env:TMP=$env:TEMP
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
Remove-Item Env:PUYO_OBS_TRACE_DIRECTORY -ErrorAction SilentlyContinue
Set-Location -LiteralPath '<work>/wt_live_v49f'
$python='<work>/build_20261019a_stage/PuyoLive/python/python.exe'
$arguments=@('-B','-u','packaging/finish_live_obs_20261020f.py','--name',$RunName)
if ($ResumeSeal) { $arguments+='--resume-seal' }
$ErrorActionPreference='Continue'
& $python @arguments *> "$taskWork/$RunName.log"
$result=$LASTEXITCODE
[System.IO.File]::WriteAllText("$taskWork/$RunName.exit.json",
    (@{exit_code=$result;at=(Get-Date).ToString('o')} | ConvertTo-Json))
exit $result