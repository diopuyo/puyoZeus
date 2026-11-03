"""PowerShellのパイプを使わず、子のUTF-8出力を新しいログへ直接保存する。"""
import argparse
import json
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open('xb') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    print(json.dumps(dict(exit_code=result.returncode, log=str(args.log))))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
