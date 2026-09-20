"""G3既存APIへ固定の実Read契約を設定する。実行/排他/usageは元ハーネスへ委ねる。"""
from __future__ import annotations

from pathlib import Path
from scripts import g3_agent_review as G

SYSTEM = (
    'あなたはG3の独立検収担当。日本語で簡潔に結論・根拠・未完・次条件を返す。'
    '最初にWORK_PACKETをReadし、evidenceの各pathの指定範囲をReadツールで最低限実読する。completionに追加の読取範囲があればそれも実読する。'
    'packet内の引用は索引であり実ファイルのReadの代用ではない。巨大ファイル全文は読まない。'
    '未読は明記し、設計妥当性と実装/実行/品質合格を区別する。既存PASSは新差分なしに再検収しない。'
    'コード変更・再委譲・GPU・学習・本番採用・G4以降は禁止。'
    'Bashはツールで明示許可された場合に限りpacketの固定CPUコマンドを一度だけ実行できる。'
    '失敗・応答不明を無条件に再試行しない。重大指摘は最大3件、重要な残件も明記する。'
)


def configure(root: Path, cpu_command: str | None = None) -> None:
    """固定役割を共通化し、変動情報はpacketと固定コマンド許可へ置く。"""
    G.SYSTEM = SYSTEM
    original = G.command
    def command(model: str, session: str, packet: Path) -> list[str]:
        argv = original(model, session, packet)
        if cpu_command is not None:
            argv[argv.index('--tools') + 1] = 'Read,Bash'
            index = argv.index('--allowedTools') + 1
            argv[index:index + 1] = ['Read', 'Bash(' + cpu_command + ')']
        return argv + ['--add-dir', str(G.ROOT), str(root)]
    G.command = command
