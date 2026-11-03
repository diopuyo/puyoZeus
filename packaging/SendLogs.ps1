param([string]$LogPath = "$PSScriptRoot/install.log", [string]$Kind = 'installation_failure',
      [string]$RelayUrl = 'https://puyolive-reports.example.invalid/api/reports',
      [string]$AppVersion = 'unknown', [string]$CodeSha = 'unknown', [string]$ModelSha = 'unknown')
# Manual cleanup/reporting only; existing behavior is unchanged.
$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'puyoZeus Live - Send logs'
$reportRoot = Join-Path $env:LOCALAPPDATA 'PuyoLiveManualReports'
if (Test-Path -LiteralPath "$PSScriptRoot/app") { $reportRoot = Join-Path $PSScriptRoot 'logs/reports' }
New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
$idPath = Join-Path $reportRoot 'installation_id.txt'
if (!(Test-Path -LiteralPath $idPath)) { [guid]::NewGuid().ToString() | Set-Content -LiteralPath $idPath }
function Hide-Identity([string]$Value) {
    $Value = [regex]::Replace($Value, '(?i)([\\/]+Users[\\/]+|[\\/]+home[\\/]+)[^\\/\s"'']+', '$1[redacted]')
    foreach ($identity in @($env:USERNAME, $env:COMPUTERNAME)) {
        if ($identity) { $Value = [regex]::Replace($Value, [regex]::Escape($identity), '[redacted]', 'IgnoreCase') }
    }
    return $Value
}
$settings = @{}
$configPath = Join-Path $PSScriptRoot 'distribution.json'
if (Test-Path -LiteralPath $configPath) {
    $settings = Get-Content -Raw -Encoding UTF8 -LiteralPath $configPath | ConvertFrom-Json
    if ($settings.relay_url) { $RelayUrl = $settings.relay_url }
}
$version = @{app=$AppVersion; model='v6'; code_sha=$CodeSha; model_sha256=$ModelSha}
$versionPath = Join-Path $PSScriptRoot 'app/RELEASE.json'
if (Test-Path -LiteralPath $versionPath) { $version = Get-Content -Raw -LiteralPath $versionPath | ConvertFrom-Json }
$lines = @()
if (Test-Path -LiteralPath $LogPath) { $lines = @(Get-Content -LiteralPath $LogPath -Tail 100 | ForEach-Object { $_.ToString() }) }
$hash = [System.Security.Cryptography.SHA256]::Create()
$fingerprint = [BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes("${Kind}:installer"))).Replace('-', '').ToLowerInvariant()
$environment = @{os=[Environment]::OSVersion.VersionString; cpu=$env:PROCESSOR_IDENTIFIER; gpu='unknown'; memory_bytes=$null}
try {
    $environment.gpu = @((Get-CimInstance Win32_VideoController -OperationTimeoutSec 2).Name)
    $environment.memory_bytes = (Get-CimInstance Win32_ComputerSystem -OperationTimeoutSec 2).TotalPhysicalMemory
} catch {}
$report = @{schema_version=1; report_id=[guid]::NewGuid().ToString(); created_at=[DateTime]::UtcNow.ToString('o');
    installation_id=(Get-Content -LiteralPath $idPath).Trim(); version=$version; environment=$environment;
    settings=$settings; error=@{type=$Kind; location='installer'; stack_trace=($lines -join "`n")};
    fingerprint=$fingerprint; summary="${Kind}:installer"; recent_logs=@(); installation_logs=$lines;
    self_diagnostics=@{python_available=(Test-Path "$PSScriptRoot/python/python.exe"); input='unknown'; fps='unknown'; calibration='unknown'; process_delay='unknown'}}
$reports = @()
$pending = @(Get-ChildItem -LiteralPath $reportRoot -Filter 'pending-*.json')
foreach ($path in $pending) { $reports += Get-Content -Raw -LiteralPath $path.FullName | ConvertFrom-Json }
$reports += $report
$payload = Hide-Identity (@{schema_version=1; reports=$reports} | ConvertTo-Json -Depth 20)
Write-Host 'puyoZeus Live - Send logs'
Write-Host 'No automatic sending.'
Write-Host "Destination: $RelayUrl"
Write-Host $payload
$localOnly = ([Uri]$RelayUrl).Host.EndsWith('.invalid', [StringComparison]::OrdinalIgnoreCase)
if ($localOnly) { Write-Host 'The destination is being prepared for 0.6.0. Reviewed reports are saved locally (logs/reports).' }
$answer = Read-Host 'Type SEND only after reviewing'
if ($answer -cne 'SEND') { exit 0 }
try {
    $uri = [Uri]$RelayUrl
    if ($uri.Scheme -ne 'https' -or $uri.UserInfo) { throw 'HTTPS relay required' }
    if ($localOnly) { throw 'Local storage only' }
    Invoke-WebRequest -Uri $RelayUrl -Method Post -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($payload)) -MaximumRedirection 0 -TimeoutSec 10 -UseBasicParsing | Out-Null
    foreach ($path in $pending) { Remove-Item -LiteralPath $path.FullName }
    Write-Host 'Sent'
} catch {
    $safeReports = ($payload | ConvertFrom-Json).reports
    foreach ($item in $safeReports) {
        $target = Join-Path $reportRoot ("pending-" + $item.report_id + '.json')
        $item | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 -LiteralPath $target
    }
    Get-ChildItem -LiteralPath $reportRoot -Filter 'pending-*.json' | Sort-Object LastWriteTime -Descending | Select-Object -Skip 20 | Remove-Item
    Write-Host 'Saved locally for the next manual review.'
}
Read-Host 'Press Enter to close'
