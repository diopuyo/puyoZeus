# Bootstrap wrapper: ensures only one watchdog runs.
# Registered in HKCU\Software\Microsoft\Windows\CurrentVersion\Run so Windows launches it at user logon.
# If an existing watchdog pid is alive, exits quietly. Otherwise spawns a fresh watchdog.

$ErrorActionPreference = 'Continue'
$PROJECT = 'C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer'
$PIDFILE = Join-Path $PROJECT 'data\watchdog.pid'
$SCRIPT  = Join-Path $PROJECT 'scripts\watchdog.ps1'
$BOOTLOG = Join-Path $PROJECT 'data\watchdog_bootstrap.log'

function Add-BootLog([string]$m) {
    try { Add-Content -Path $BOOTLOG -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" -Encoding UTF8 } catch {}
}

Add-BootLog "bootstrap invoked pid=$PID"

$alreadyRunning = $false
if (Test-Path $PIDFILE) {
    try {
        $wdpid = [int](Get-Content $PIDFILE -Raw).Trim()
        $p = Get-Process -Id $wdpid -ErrorAction SilentlyContinue
        if ($p -and $p.ProcessName -match 'powershell') {
            $alreadyRunning = $true
            Add-BootLog "watchdog already running pid=$wdpid — no action"
        } else {
            Add-BootLog "stale pidfile (pid=$wdpid not alive) — respawning"
        }
    } catch {
        Add-BootLog "pidfile unreadable — respawning"
    }
}

if (-not $alreadyRunning) {
    try {
        Start-Process -FilePath 'powershell.exe' `
            -ArgumentList @('-ExecutionPolicy','Bypass','-NoProfile','-WindowStyle','Hidden','-File', $SCRIPT) `
            -WindowStyle Hidden
        Add-BootLog "spawned watchdog from $SCRIPT"
    } catch {
        Add-BootLog "spawn failed: $_"
    }
}
