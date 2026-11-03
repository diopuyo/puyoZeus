param([string]$Comparison='comparison_after_r7', [string]$RunName='parity_delivery')
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
Set-Location -LiteralPath $taskRoot
$done="$taskWork/$Comparison.exit.json"
$deadline=(Get-Date).AddHours(2)
# 速度比較に同値検査のCPU負荷を重ねない。読取り途中の終了票では開始しない。
$comparisonResult=$null
while ($null -eq $comparisonResult) {
    if ((Get-Date) -gt $deadline) { throw '比較の終了待ちが2時間を超えました' }
    if (Test-Path -LiteralPath $done) {
        try { $comparisonResult=Get-Content -LiteralPath $done -Raw | ConvertFrom-Json } catch {}
    }
    if ($null -eq $comparisonResult) { Start-Sleep -Seconds 10 }
}
if ($comparisonResult.exit_code -ne 0) { throw '比較処理が中断したため、同値確認の自動開始を保留します' }
$python='<work>/build_20261019a_stage/PuyoLive/python/python.exe'
$ErrorActionPreference='Continue'
& $python -u packaging/verify_live_obs_20261020d.py --out "$taskWork/$RunName" --skip-tests *> "$taskWork/$RunName.log"
$result=$LASTEXITCODE
$ErrorActionPreference='Stop'
[System.IO.File]::WriteAllText("$taskWork/$RunName.exit.json",
    (@{exit_code=$result;at=(Get-Date).ToString('o')} | ConvertTo-Json))
exit $result
