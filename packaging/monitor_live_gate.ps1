param([Parameter(Mandatory=$true)][string]$Output, [Parameter(Mandatory=$true)][string]$StopFile)
$ErrorActionPreference='Stop'
$utf8=[Text.UTF8Encoding]::new($false)
# 測定専用。周波数・性能・同居負荷・メモリ・電源プランを全長で記録する。
while (-not (Test-Path -LiteralPath $StopFile)) {
    $record=@{at=(Get-Date).ToString('o');monitor_pid=$PID}
    try {
        $record.clocks=@(Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation |
            Select-Object Name,ProcessorFrequency,PercentProcessorPerformance,PercentofMaximumFrequency,PercentProcessorUtility)
        $record.processes=@(Get-CimInstance Win32_PerfFormattedData_PerfProc_Process |
            Where-Object {$_.Name -ne '_Total' -and $_.Name -ne 'Idle'} |
            Select-Object Name,IDProcess,PercentProcessorTime,WorkingSetPrivate)
        $record.free_gib=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
        $record.c_free_gib=(Get-PSDrive C).Free/1GB
        $record.battery=@(Get-CimInstance Win32_Battery | Select-Object BatteryStatus)
        $record.active_power_plan=((& powercfg /getactivescheme) | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) { throw 'Cannot query active power plan' }
        $record.active_run=if (Test-Path -LiteralPath ($StopFile+'.run')) {
            [IO.File]::ReadAllText($StopFile+'.run')
        } else {'preparation'}
    } catch { $record.error=$_.Exception.Message }
    [IO.File]::AppendAllText($Output,($record | ConvertTo-Json -Depth 6 -Compress)+"`n",$utf8)
    Start-Sleep -Seconds 10
}
