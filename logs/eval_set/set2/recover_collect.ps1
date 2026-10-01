# 起動後のシェル修正で失われた終了通知を、収集器自身の完了票で復元する。
# 元プロセスの終了コードは推測せずnullとして記録する。
$ErrorActionPreference = 'Stop'
$output = 'D:\puyo_analyzer\wt_evalset\logs\eval_set\set2'
$encoding = New-Object System.Text.UTF8Encoding($false)
$active = ((& wsl -d Ubuntu -- pgrep -af scripts.collect_eval_set_20261001) -join "`n")
foreach ($part in @('s0','s1','s2','s3','s4','s5','s6','check')) {
    $done = "$output\jobs\collect_$part.done"
    $receiptPath = "$output\collect\completed\$part.json"
    if ((Test-Path -LiteralPath $done) -or -not (Test-Path -LiteralPath $receiptPath)) { continue }
    if ($active.Contains("--part $part ")) { continue }
    $receipt = Get-Content -Encoding utf8 -Raw $receiptPath | ConvertFrom-Json
    $meta = Get-Content -Encoding utf8 -Raw "$output\collect\records\$part.jsonl.json" | ConvertFrom-Json
    $record = "$output\collect\records\$part.jsonl.gz"
    $hash = (Get-FileHash -LiteralPath $record -Algorithm SHA256).Hash.ToLower()
    if (-not $receipt.completed -or $meta.state -ne 'completed' -or $hash -ne $meta.record_sha256) {
        throw "収集完了票またはSHA不一致: $part"
    }
    $file = [IO.File]::OpenRead($record)
    $gzip = New-Object IO.Compression.GZipStream($file, [IO.Compression.CompressionMode]::Decompress)
    $reader = New-Object IO.StreamReader($gzip)
    $last = ''
    try { while (-not $reader.EndOfStream) { $last = $reader.ReadLine() } } finally { $reader.Dispose(); $file.Dispose() }
    $terminal = $last | ConvertFrom-Json
    if ($terminal.kind -ne 'complete') { throw "終端complete行なし: $part" }
    $audit = @{source=$part; exit_code=$null; completed=$true; process_absent=$true;
        sha256=$hash; terminal=$terminal; reason='実行中にjob.shを更新したため終了通知のみ未保存。収集完了票・SHA・gzip終端を照合'}
    [IO.File]::WriteAllText("$output\jobs\collect_$part.recovered.json", ($audit | ConvertTo-Json -Depth 6), $encoding)
    [IO.File]::WriteAllText($done, "0`n", $encoding)
    Write-Output "完了通知復元: $part (元終了コードは未採取、原票完了確認済み)"
}
