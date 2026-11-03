param([Parameter(Mandatory=$true)][string]$Bundle,
      [Parameter(Mandatory=$true)][string]$Entry, [switch]$Camera)
$ErrorActionPreference='Stop'
$bundlePath=(Resolve-Path -LiteralPath $Bundle).Path
$entryPath=(Resolve-Path -LiteralPath $Entry).Path
if ((Get-PSDrive C).Free -lt 10GB) { throw 'C: free space is below 10 GiB' }
if ((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory*1KB -lt 8.5GB) {
    throw 'Free memory is below the startup reserve of 8.5 GiB'
}
# 実行系は読取りだけ。大きな出力と一時ファイルはD:の入口に置く。
$runPath=Join-Path $entryPath ('output/startup_'+(Get-Date).ToString('yyyyMMddTHHmmss_fffffff'))
$temporary=Join-Path $runPath 'temp'
New-Item -ItemType Directory -Path $temporary -Force | Out-Null
$env:TEMP=$temporary
$env:TMP=$temporary
$env:PUYO_OBS_STDLIB_CACHE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$runtime='<work>/build_20261019a_stage/PuyoLive/python'
if (-not (Test-Path -LiteralPath (Join-Path $runtime 'python.exe'))) {
    $runtime=Join-Path $bundlePath 'python'
}
$python=Join-Path $runtime 'python.exe'
$arguments=@('-B','-u',(Join-Path $bundlePath 'obs_runtime_bootstrap.py'),
    '--bundle',$bundlePath,'--runtime',$runtime,
    '--receipt',(Join-Path $runPath 'runtime_receipt.json'))
if ($Camera) { $arguments+='--camera' } else { $arguments+=@('--entry',$entryPath) }
Set-Location -LiteralPath $bundlePath
$ErrorActionPreference='Continue'
& $python @arguments
exit $LASTEXITCODE
