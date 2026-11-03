param([Parameter(Mandatory=$true)][int]$UninstallerPid)
$ErrorActionPreference = 'Stop'
# Manual cleanup/reporting only; existing behavior is unchanged.
$cleanupRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
$reparsePoint = [IO.FileAttributes]::ReparsePoint
$cleanupAttempts = 40
$retryDelayMs = 250
if ([IO.Path]::GetDirectoryName($cleanupRoot) -ne $tempRoot -or
    [IO.Path]::GetFileName($cleanupRoot) -notmatch '^is-[a-z0-9]+-uninstall\.tmp$') {
    throw 'Unexpected uninstaller temporary directory'
}
if ((Get-Item -LiteralPath $cleanupRoot).Attributes -band $reparsePoint) {
    throw 'Linked temporary directory is not allowed'
}
$uninstaller = Get-Process -Id $UninstallerPid -ErrorAction SilentlyContinue
if ($uninstaller -and -not $uninstaller.HasExited) {
    if ($uninstaller.Path -ne (Join-Path $cleanupRoot '_unins.tmp')) {
        throw 'Unexpected uninstaller process'
    }
    $uninstaller.WaitForExit()
}
# Manual cleanup/reporting only; existing behavior is unchanged.
for ($attempt = 0; $attempt -lt $cleanupAttempts; $attempt++) {
    try {
        foreach ($name in @('_unins.tmp', '_unins-done.tmp', 'PuyoLiveUninstallCleanup.ps1')) {
            $target = Join-Path $cleanupRoot $name
            if (Test-Path -LiteralPath $target) {
                if ((Get-Item -LiteralPath $target).Attributes -band $reparsePoint) {
                    throw 'Linked temporary file is not allowed'
                }
                Remove-Item -LiteralPath $target -Force
            }
        }
        if (@(Get-ChildItem -LiteralPath $cleanupRoot -Force).Count -eq 0) {
            Remove-Item -LiteralPath $cleanupRoot
        }
        exit 0
    } catch {
        if ($attempt -eq $cleanupAttempts - 1) { throw }
        Start-Sleep -Milliseconds $retryDelayMs
    }
}
