param([string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$log = Join-Path $OutputDirectory 'environment.jsonl'
$stopFile = Join-Path $OutputDirectory 'monitor.stop'
while (-not (Test-Path -LiteralPath $stopFile)) {
    $os = Get-CimInstance Win32_OperatingSystem
    $cpu = Get-CimInstance Win32_Processor
    $processes = Get-CimInstance Win32_PerfFormattedData_PerfProc_Process
    $row = @{
        time = (Get-Date).ToString('o')
        free_gib = $os.FreePhysicalMemory / 1MB
        cpu = @($cpu | Select-Object Name,CurrentClockSpeed,MaxClockSpeed,LoadPercentage)
        heavy = @($processes | Where-Object { $_.Name -notmatch '^(_Total|Idle)$' } | Sort-Object PercentProcessorTime -Descending | Select-Object -First 12 Name,IDProcess,PercentProcessorTime,WorkingSetPrivate)
        obs = @($processes | Where-Object { $_.Name -like 'obs64*' } | Select-Object Name,IDProcess,PercentProcessorTime,WorkingSetPrivate)
        ama = @($processes | Where-Object { $_.Name -like '*ama_plain_benchmark*' } | Select-Object Name,IDProcess)
    }
    $row | ConvertTo-Json -Depth 6 -Compress | Add-Content -LiteralPath $log -Encoding UTF8
    if ($row.free_gib -lt 6) { 'Memory below 6 GiB' | Set-Content (Join-Path $OutputDirectory 'RESOURCE_STOP.txt'); break }
    Start-Sleep -Seconds 5
}
