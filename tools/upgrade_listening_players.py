"""Update only player HTML; keep frozen WAVs, test IDs and blinding keys."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from musicmix.listening import enable_seek_controls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    for directory in parser.parse_args().directories:
        path = directory/'index.html'
        original = path.read_bytes()
        upgraded = enable_seek_controls(original.decode('utf-8')).encode('utf-8')
        if original == upgraded:
            print(f'{directory}: already upgraded')
            continue
        revision = {'created_utc': datetime.now(timezone.utc).isoformat(),
                    'scope': 'Player controls only; WAVs, trial identities and keys unchanged',
                    'previous_html_sha256': hashlib.sha256(original).hexdigest(),
                    'html_sha256': hashlib.sha256(upgraded).hexdigest(),
                    'player_version': 2}
        path.write_bytes(upgraded)
        (directory/'player_revision.json').write_text(json.dumps(revision, indent=2)+'\n', encoding='utf-8')
        print(f'{directory}: upgraded')
