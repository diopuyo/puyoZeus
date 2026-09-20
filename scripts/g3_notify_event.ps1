param([Parameter(Mandatory=$true)][string]$RequestPath)
# 当該G3イベントの通知だけを行い、表示イベントと解放を保存する。
$ErrorActionPreference = 'Stop'
$g3Resolved = [System.IO.Path]::GetFullPath($RequestPath)
if (-not $g3Resolved.StartsWith('D:\puyo_analyzer\verify\',[System.StringComparison]::OrdinalIgnoreCase)) {throw 'g3_notify_storage'}
$g3Request = Get-Content -LiteralPath $g3Resolved -Raw -Encoding UTF8 | ConvertFrom-Json
if ($g3Request.schema -ne 'g3-local-notification/v1') {throw 'g3_notify_schema'}
$g3Receipt = Join-Path ([System.IO.Path]::GetDirectoryName($g3Resolved)) 'NOTIFY_RESULT.json'
if (Test-Path -LiteralPath $g3Receipt) {throw 'g3_notify_duplicate'}
# 結果保存前の中断も再送せず、開始票を残す。
$g3ClaimPath = Join-Path ([System.IO.Path]::GetDirectoryName($g3Resolved)) 'NOTIFY_STARTED.json'
$g3Claim = [System.IO.File]::Open($g3ClaimPath,[System.IO.FileMode]::CreateNew)
try {$g3ClaimBytes=[System.Text.Encoding]::UTF8.GetBytes(('{"pid":'+$PID+'}'));$g3Claim.Write($g3ClaimBytes,0,$g3ClaimBytes.Length);$g3Claim.Flush($true)} finally {$g3Claim.Dispose()}
$g3Icon = $null
$script:g3Shown = $false
$g3Error = $null
$g3Disposed = $false
try {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    if (-not [Environment]::UserInteractive) {throw 'g3_notify_noninteractive_session'}
    $g3Icon = New-Object System.Windows.Forms.NotifyIcon
    $g3Icon.Icon = [System.Drawing.SystemIcons]::Information
    $g3Icon.Text = 'puyo_analyzer G3'
    $g3Icon.BalloonTipTitle = [string]$g3Request.title
    $g3Icon.BalloonTipText = [string]$g3Request.body
    $g3Icon.add_BalloonTipShown({$script:g3Shown = $true})
    $g3Icon.Visible = $true
    $g3Icon.ShowBalloonTip(5000)
    $g3Watch = [System.Diagnostics.Stopwatch]::StartNew()
    while (-not $script:g3Shown -and $g3Watch.Elapsed.TotalSeconds -lt 15) {
        [System.Windows.Forms.Application]::DoEvents()
        Start-Sleep -Milliseconds 100
    }
    if ($script:g3Shown) {Start-Sleep -Seconds 3}
} catch {
    $g3Error = $_.Exception.Message
} finally {
    if ($null -ne $g3Icon) {
        $g3Icon.Visible = $false
        $g3Icon.Dispose()
        $g3Disposed = $true
    }
}
$g3Value = @{schema='g3-local-notification-result/v1';shown_event=$script:g3Shown;disposed=$g3Disposed;error=$g3Error;user_read=$null;root_thread_resumed=$false;request_path=$g3Resolved;quality_gate_clear=$false}
$g3Json = $g3Value | ConvertTo-Json -Compress
$g3Output = [System.IO.File]::Open($g3Receipt,[System.IO.FileMode]::CreateNew)
try {$g3Bytes=[System.Text.Encoding]::UTF8.GetBytes($g3Json);$g3Output.Write($g3Bytes,0,$g3Bytes.Length);$g3Output.Flush($true)} finally {$g3Output.Dispose()}
$g3Json
if (-not $script:g3Shown -or $g3Error -or -not $g3Disposed) {exit 1}

