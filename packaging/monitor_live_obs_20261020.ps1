param([string]$Output, [string]$StopFile)
$ErrorActionPreference = 'Stop'
$self = Get-Process -Id $PID
$self.ProcessorAffinity = 61440
$self.PriorityClass = 'BelowNormal'
$lastPidCheck = [DateTime]::MinValue
$targetRunning = $true
while (-not (Test-Path -LiteralPath $StopFile)) {
    $now = Get-Date
    $checked = ($now - $lastPidCheck).TotalSeconds -ge 300
    if ($checked) {
        $targetRunning = $null -ne (Get-Process -Id 31372 -ErrorAction SilentlyContinue)
        $lastPidCheck = $now
    }
    $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $row = @{at=$now.ToString('o'); free_gib=$free; target_31372_running=$targetRunning;
             target_checked=$checked; affinity=61440; priority='BelowNormal'; memory_ok=($free -ge 6.5)}
    Add-Content -LiteralPath $Output -Value ($row | ConvertTo-Json -Compress) -Encoding UTF8
    Start-Sleep -Seconds 10
}
