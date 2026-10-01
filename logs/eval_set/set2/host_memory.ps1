$ErrorActionPreference = 'Stop'
$output = 'D:\puyo_analyzer\wt_evalset\logs\eval_set\set2'
$encoding = New-Object System.Text.UTF8Encoding($false)
while (-not (Test-Path -LiteralPath "$output\finish.done")) {
    $available = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory
    [IO.File]::WriteAllText("$output\host_available_kib.txt", [string]$available, $encoding)
    [IO.File]::AppendAllText("$output\host_resources.tsv", "$(Get-Date -Format o)`t$available`n", $encoding)
    Start-Sleep -Seconds 20
}
