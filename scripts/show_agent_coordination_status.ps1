$ErrorActionPreference = "Continue"
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

Write-Output "=== agent coordination status ==="
Get-Date -Format "yyyy-MM-dd HH:mm:ss K"
Get-Content -LiteralPath "docs/agent_coordination/CURRENT.md" -Encoding UTF8

Write-Output "=== active verification processes ==="
wsl -d Ubuntu -- bash -lc "ps -eo pid,ppid,ni,stat,etimes,pcpu,cmd | grep -E '[_]driver_formula_fix|[_]diag_formula_fix|[_]probe_formula|[m]easure_stable_cell_acc|[p]ytest tests/ -q'"

Write-Output "=== completion markers ==="
Get-Content "logs/_probe_formula_fix_cases_2026-08-24/driver_progress.log" -Encoding UTF8 -ErrorAction SilentlyContinue
Get-Content "logs/_probe_formula_false_event_2026-08-24/driver_progress.log" -Encoding UTF8 -ErrorAction SilentlyContinue
Get-Content "data/verify/formula_read_backtest_2026-08-24/md5sums.txt" -Encoding UTF8 -ErrorAction SilentlyContinue
Get-Content "logs/pm100_fix_2026-08-24/pytest_full.log" -Encoding UTF8 -Tail 12 -ErrorAction SilentlyContinue

Write-Output "=== worktree summary ==="
git diff --stat
$untrackedCount = (git status --short | Where-Object { $_ -like "??*" }).Count
Write-Output "untracked_count=$untrackedCount"
