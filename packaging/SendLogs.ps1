param([string]$LogPath = '', [string]$Kind = 'installation_failure',
      [string]$RelayUrl = 'https://puyolive-reports.example.invalid/api/reports',
      [string]$AppVersion = 'unknown', [string]$CodeSha = 'unknown', [string]$ModelSha = 'unknown')
# Manual cleanup/reporting only; existing behavior is unchanged.
$ErrorActionPreference = 'Stop'
$minimumIdentityLength = 3
$maximumReportIdLength = 1024
$Host.UI.RawUI.WindowTitle = 'puyoZeus Live - Send logs'
$appRoot = $PSScriptRoot
if ((Split-Path -Leaf $PSScriptRoot) -eq 'tools') { $appRoot = Split-Path -Parent $PSScriptRoot }
if (!$LogPath) { $LogPath = Join-Path $appRoot 'install.log' }
$reportRoot = Join-Path $env:LOCALAPPDATA 'PuyoLiveManualReports'
if (Test-Path -LiteralPath "$appRoot/app") { $reportRoot = Join-Path $appRoot 'logs/reports' }
New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
$idPath = Join-Path $reportRoot 'installation_id.txt'
if (!(Test-Path -LiteralPath $idPath)) { [guid]::NewGuid().ToString('N') | Set-Content -LiteralPath $idPath }
function Hide-Identity([string]$Value) {
    $parts = [regex]::Split($appRoot.TrimEnd('\', '/'), '[\\/]+') | ForEach-Object { [regex]::Escape($_) }
    $Value = [regex]::Replace($Value, ($parts -join '[\\/]+') + '(?=$|[\\/\s"''])', '<install>', 'IgnoreCase')
    $Value = [regex]::Replace($Value, '(?i)([\\/]+Users[\\/]+|[\\/]+home[\\/]+)[^\\/\s"'']+', '$1[redacted]')
    $Value = [regex]::Replace($Value, '(?i)OneDrive - [^\\/"''\r\n]+', 'OneDrive - [redacted]')
    if ($Value.Contains('@')) { $Value = [regex]::Replace($Value, '(?i)[\w.%+-]{1,64}@[\w.-]{1,253}\.[a-z]{2,63}', '[redacted]') }
    $Value = [regex]::Replace($Value, '(?<![\w:])[\\/]{2,}[^\\/\s"'']+[\\/]+[^\\/\s"'']+', '<network>')
    $Value = [regex]::Replace($Value, '(?i)\b(password|passwd|secret|token|api[_-]?key)(\s*[:=]\s*)(?:"[^"]*"|''[^'']*''|[^\s,;]+)', '$1$2[redacted]')
    foreach ($identity in @($env:USERNAME, $env:COMPUTERNAME)) {
        if ($identity.Length -ge $minimumIdentityLength) { $Value = [regex]::Replace($Value, '(?<!\w)' + [regex]::Escape($identity) + '(?!\w)', '[redacted]', 'IgnoreCase') }
    }
    return $Value
}
function Remove-PrivateSettings($Value) {
    if ($null -eq $Value) { return @{} }
    foreach ($key in @('obs_password', 'relay_url', 'releases_url')) {
        $Value.PSObject.Properties.Remove($key)
    }
    return $Value
}
$settings = @{}
$configPath = Join-Path $appRoot 'distribution.json'
if (Test-Path -LiteralPath $configPath) {
    $settings = Get-Content -Raw -Encoding UTF8 -LiteralPath $configPath | ConvertFrom-Json
    if ($settings.relay_url) { $RelayUrl = $settings.relay_url }
}
$version = @{app=$AppVersion; model='v6'; code_sha=$CodeSha; model_sha256=$ModelSha}
$versionPath = Join-Path $appRoot 'app/RELEASE.json'
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
$report = @{schema_version=1; report_id=[guid]::NewGuid().ToString('N'); created_at=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds();
    installation_id=(Get-Content -LiteralPath $idPath).Trim().Replace('-', '').ToLowerInvariant(); version=$version; environment=$environment;
    settings=$settings; error=@{type=$Kind; location='installer'; stack_trace=($lines -join "`n")};
    fingerprint=$fingerprint; summary="${Kind}:installer"; recent_logs=@(); installation_logs=$lines;
    self_diagnostics=@{python_available=(Test-Path "$PSScriptRoot/python/python.exe"); input='unknown'; fps='unknown'; calibration='unknown'; process_delay='unknown'}}
function Test-Report($Item) {
    return (($Item -is [pscustomobject] -or $Item -is [System.Collections.IDictionary]) -and
        $Item.schema_version -eq 1 -and $Item.report_id -is [string] -and
        $Item.report_id.Length -gt 0 -and $Item.report_id.Length -le $maximumReportIdLength -and
        ($null -eq $Item.settings -or $Item.settings -is [pscustomobject] -or
            $Item.settings -is [System.Collections.IDictionary]))
}
function Get-ReportPath($Item) {
    if (!(Test-Report $Item)) { throw 'Invalid report' }
    $digest = [BitConverter]::ToString($hash.ComputeHash([Text.Encoding]::UTF8.GetBytes($Item.report_id))).Replace('-', '').ToLowerInvariant()
    $directory = [IO.Path]::GetFullPath($reportRoot).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $target = [IO.Path]::GetFullPath((Join-Path $directory ('pending-' + $digest + '.json')))
    if (!$target.StartsWith($directory, [StringComparison]::OrdinalIgnoreCase)) { throw 'Report path outside reports' }
    return $target
}
$reports = @()
$pending = @()
foreach ($path in @(Get-ChildItem -LiteralPath $reportRoot -Filter 'pending-*.json')) {
    try {
        $raw = Get-Content -Raw -Encoding UTF8 -LiteralPath $path.FullName
        if (!$raw.TrimStart().StartsWith('{')) { throw 'Pending report must be an object' }
        $item = $raw | ConvertFrom-Json
        if (!(Test-Report $item)) { throw 'Invalid pending report' }
        $reports += $item
        $pending += $path
    } catch {
        $invalid = Join-Path $reportRoot ('invalid-' + [guid]::NewGuid().ToString('N') + '.json')
        Move-Item -LiteralPath $path.FullName -Destination $invalid
    }
}
$reports += $report
foreach ($item in $reports) { $item.settings = Remove-PrivateSettings $item.settings }
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
        $target = Get-ReportPath $item
        $temporary = $target + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
        [IO.File]::WriteAllText($temporary, ($item | ConvertTo-Json -Depth 20), (New-Object Text.UTF8Encoding($false)))
        Move-Item -LiteralPath $temporary -Destination $target -Force
    }
    Get-ChildItem -LiteralPath $reportRoot -Filter 'pending-*.json' | Sort-Object LastWriteTime -Descending | Select-Object -Skip 20 | Remove-Item
    Write-Host 'Saved locally for the next manual review.'
}
Read-Host 'Press Enter to close'
