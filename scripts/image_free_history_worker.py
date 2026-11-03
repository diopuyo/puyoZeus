"""固定本体を変更せず、履歴高速化の独立照合を実行する入口。"""
from pathlib import Path
import sys
from unittest.mock import patch


def main() -> None:
    source = Path(sys.argv[sys.argv.index('--source')+1])
    sys.path.insert(0, str(source))
    from scripts import image_free_parallel as parallel, image_free_stream as stream
    from image_free_history_proof import run
    with patch.object(stream, 'run', run):
        parallel.main()


if __name__ == '__main__':
    main()
