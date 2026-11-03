param([Parameter(Mandatory=$true)][string]$Output, [int]$ExcludePid=0, [string]$WaitStartedAt='', [switch]$SingleCheck)
$ErrorActionPreference='Stop'
$start=if ($WaitStartedAt) {[DateTime]::Parse($WaitStartedAt)} else {Get-Date}
while ($true) {
    $tree=@(Get-CimInstance Win32_Process)
    $parents=@{}
    foreach ($p in $tree) { $parents[[int]$p.ProcessId]=$p }
    $sample=@{}
    foreach ($p in (Get-Process -ErrorAction SilentlyContinue)) {
        if ($p.Id -ne $ExcludePid) { $sample[$p.Id]=$p.TotalProcessorTime.TotalSeconds }
    }
    Start-Sleep -Seconds 5
    $rows=@()
    foreach ($p in (Get-Process -ErrorAction SilentlyContinue)) {
        if ($p.Id -eq $ExcludePid) { continue }
        if ($p.Id -eq $PID) { continue }
        if ($p.ProcessName -match '^obs') { continue }
        $cursor=$p.Id
        $codex=$false
        $seen=@{}
        while ($parents.ContainsKey($cursor) -and -not $seen.ContainsKey($cursor)) {
            $seen[$cursor]=$true
            $node=$parents[$cursor]
            if ($node.Name -eq 'codex.exe') { $codex=$true; break }
            $cursor=[int]$node.ParentProcessId
        }
        if ($p.ProcessName -notmatch '^(python|pythonw|ama_reference|vmmemWSL)$' -and
            (-not $codex -or $p.ProcessName -match '^(codex|node|powershell|cmd)')) { continue }
        $delta=if ($sample.ContainsKey($p.Id)) { $p.TotalProcessorTime.TotalSeconds-$sample[$p.Id] } else { 1.0 }
        $rows+=@{pid=$p.Id; name=$p.ProcessName; codex_descendant=$codex; cpu_sec_5s=$delta; affinity=[long]$p.ProcessorAffinity; priority=[string]$p.PriorityClass}
    }
    # Include orphaned Python processes and finish our own validation first.
    $busy=@($rows | Where-Object {$_.cpu_sec_5s -gt 0.1})
    $vm=@(Get-CimInstance Win32_PerfFormattedData_PerfProc_Process |
        Where-Object Name -Match '^vmmem' | Select-Object Name,IDProcess,PercentProcessorTime)
    $vmBusy=@($vm | Where-Object {$_.PercentProcessorTime -gt 10}).Count
    $elapsed=((Get-Date)-$start).TotalSeconds
    $clocks=@(Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation |
        Select-Object Name,ProcessorFrequency,PercentProcessorPerformance,PercentofMaximumFrequency,PercentProcessorUtility)
    $free=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB
    $record=@{at=(Get-Date).ToString('o');elapsed_sec=$elapsed;python=$rows;busy=($busy.Count+$vmBusy);vm=$vm;
        timed_out=($elapsed -ge 28800);interval_sec=600;free_gib=$free;clocks=$clocks;
        obs=@(Get-Process -Name obs64 -ErrorAction SilentlyContinue | Select-Object Id,CPU)}
    Add-Content -LiteralPath $Output -Value ($record | ConvertTo-Json -Depth 5 -Compress) -Encoding UTF8
    if ($free -lt 6.5) { throw 'Free memory below 6.5 GiB; this measurement will not start.' }
    if ($SingleCheck -or ($busy.Count+$vmBusy) -eq 0 -or $elapsed -ge 28800) { break }
    Start-Sleep -Seconds 600
}
