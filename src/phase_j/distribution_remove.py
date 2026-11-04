"""ファイルを作らず、同じPowerShell内で待機・再検査・削除する。"""
REMOVAL_SCRIPT = r"""
$Root=$env:PUYO_REMOVE_ROOT
$WaitPid=[int]$env:PUYO_REMOVE_PID
$LauncherPid=[int]$env:PUYO_REMOVE_PARENT
$ErrorActionPreference = 'Stop'
function Assert-PlainPath([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked path' }
}
function Assert-ProductRoot {
    if (-not [IO.Path]::IsPathRooted($Root)) { throw 'Relative root' }
    $script:Target = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    if ($Target -ne $Root.TrimEnd('\')) { throw 'Root must be an absolute normalized path' }
    if ($Target -eq [IO.Path]::GetPathRoot($Target).TrimEnd('\') -or
        $Target -eq [Environment]::GetFolderPath('UserProfile')) { throw 'Unsafe root' }
    $part = Get-Item -LiteralPath $Target -Force
    while ($null -ne $part) {
        Assert-PlainPath $part.FullName
        $part = $part.Parent
    }
    foreach ($marker in @('app/RELEASE.json','app/MANIFEST.json','app/src/phase_j/distribution_app.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Target $marker) -PathType Leaf)) { throw 'Missing product marker' }
    }
    # リンクへ降りる前に検査する。再帰列挙にリンクの追跡を任せない。
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($Target)
    while ($pending.Count -gt 0) {
        foreach ($child in Get-ChildItem -LiteralPath $pending.Pop() -Force) {
            Assert-PlainPath $child.FullName
            if ($child.PSIsContainer) { $pending.Push($child.FullName) }
        }
    }
}
foreach ($processId in @($WaitPid, $LauncherPid)) {
    if ($processId -gt 0) { Wait-Process -Id $processId -ErrorAction SilentlyContinue }
}
Assert-ProductRoot
Remove-Item -LiteralPath $Target -Recurse -Force
"""
