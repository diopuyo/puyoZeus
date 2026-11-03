param([string]$Output, [string]$StopFile)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Output) { throw 'Existing monitor output' }
while (-not (Test-Path -LiteralPath $StopFile)) {
    $osInfo = Get-CimInstance Win32_OperatingSystem
    $cpuInfo = Get-CimInstance Win32_Processor
    $freq = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation |
        Where-Object Name -eq '_Total' | Select-Object ProcessorFrequency,PercentProcessorPerformance,PercentProcessorTime
    $row = [ordered]@{
        utc = (Get-Date).ToUniversalTime().ToString('o')
        available_gib = $osInfo.FreePhysicalMemory / 1MB
        clock_mhz = $cpuInfo.CurrentClockSpeed
        performance = $freq
        processes = @(Get-Process | Where-Object CPU -gt 0 |
            Select-Object Id,ProcessName,CPU,WorkingSet64)
    }
    $row | ConvertTo-Json -Depth 5 -Compress | Add-Content -LiteralPath $Output -Encoding UTF8
    Start-Sleep -Seconds 2
}
