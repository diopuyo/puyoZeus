# Windowsホストの空きメモリ・CPU負荷を5秒周期で記録。他プロセスは変更しない。
param([string]$OutputRoot = 'D:/puyo_analyzer/b3_data/time_gate')
$ErrorActionPreference = 'Stop'
$periodSec = 5
$logicalCpu = 16
$previous = @{}
$last = [DateTime]::UtcNow
$dest = Join-Path $OutputRoot 'host_resources.jsonl'
while (-not (Test-Path (Join-Path $OutputRoot 'all_complete'))) {
    $now = [DateTime]::UtcNow
    $interval = ($now - $last).TotalSeconds
    $processes = @(Get-Process | Where-Object { $null -ne $_.CPU })
    $usage = @($processes | ForEach-Object {
        $cpuPercent = 0.0
        if ($previous.ContainsKey($_.Id) -and $interval -gt 0) {
            $cpuPercent = [Math]::Max(0, ($_.CPU - $previous[$_.Id]) / $interval / $logicalCpu * 100)
        }
        [pscustomobject]@{ name = $_.ProcessName; pid = $_.Id; cpu_percent = $cpuPercent }
        $previous[$_.Id] = $_.CPU
    })
    $osInfo = Get-CimInstance Win32_OperatingSystem
    $row = @{ utc = $now.ToString('o'); available_gib = $osInfo.FreePhysicalMemory * 1024 / 1GB;
        total_gib = $osInfo.TotalVisibleMemorySize * 1024 / 1GB;
        cpu_percent = ($usage | Measure-Object -Property cpu_percent -Sum).Sum;
        top = @($usage | Sort-Object cpu_percent -Descending | Select-Object -First 8) }
    $row | ConvertTo-Json -Compress -Depth 4 | Add-Content -LiteralPath $dest -Encoding utf8
    $last = $now
    Start-Sleep -Seconds $periodSec
}
