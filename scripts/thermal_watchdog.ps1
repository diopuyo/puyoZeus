# Thermal watchdog (2026-07-16). ASCII-only comments (PS5.1 misparses UTF-8-no-BOM w/ CJK).
# Samples ACPI thermal zone (\_TZ.THRM via perf counter, no admin) periodically.
# On TRIP: SIGSTOP the heavy jobs (pause). On RESUME temp: SIGCONT (resume). Logs temps.
param(
  [double]$Trip = 95.0,
  [double]$Resume = 89.0,
  [int]$IntervalSec = 15,
  [string]$Pattern = "[c]ollect_chain_stats",
  [string]$Log = "logs\thermal_watch.log"
)
$ErrorActionPreference = 'SilentlyContinue'
function Get-TZ {
  (Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation |
    Select-Object -First 1).HighPrecisionTemperature/10 - 273.15
}
$paused = $false
Add-Content $Log ("=== watchdog start {0} Trip={1} Resume={2} ===" -f (Get-Date), $Trip, $Resume)
while ($true) {
  $t = Get-TZ
  $cpu = (Get-CimInstance Win32_Processor).LoadPercentage
  Add-Content $Log ("{0} TZ={1:N1}C CPU={2}% paused={3}" -f (Get-Date -Format HH:mm:ss), $t, $cpu, $paused)
  if ((-not $paused) -and $t -ge $Trip) {
    Add-Content $Log ("!!! TRIP {0:N1}C >= {1} -> SIGSTOP pause" -f $t, $Trip)
    wsl -d Ubuntu -- bash -c "pkill -STOP -f '$Pattern'" 2>$null | Out-Null
    $paused = $true
  }
  elseif ($paused -and $t -le $Resume) {
    Add-Content $Log ("--- COOL {0:N1}C <= {1} -> SIGCONT resume" -f $t, $Resume)
    wsl -d Ubuntu -- bash -c "pkill -CONT -f '$Pattern'" 2>$null | Out-Null
    $paused = $false
  }
  Start-Sleep -Seconds $IntervalSec
}
