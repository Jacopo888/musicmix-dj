"""Private book library: extract and search the books in `libri/` (branch main).

The books are the user's copies for private research. Their text stays outside the
repository (default $MUSICMIX_LIBRI, else ../libri next to the checkout); notes and short quotations go in conoscenza_mix/libri/.

    python tools/libri.py prepare            # extract text per page from origin/main:libri/
    python tools/libri.py search "pickup" -n 20 [--book Ohriner]
"""
import argparse, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from musicmix.paths import external_dir  # noqa: E402

DEST = Path(os.environ.get('MUSICMIX_LIBRI', str(external_dir() / 'libri')))


def prepare(ref):
    (DEST/'pages').mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', '-C', str(ROOT), 'fetch', '-q', '--depth=1', 'origin', 'main'], check=False)
    names = subprocess.run(['git', '-C', str(ROOT), 'ls-tree', '--name-only', ref, 'libri/'],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    for name in names:
        src = DEST/Path(name).name
        if not src.exists():
            src.write_bytes(subprocess.run(['git', '-C', str(ROOT), 'show', f'{ref}:{name}'],
                                           capture_output=True, check=True).stdout)
        stem = src.stem
        if src.suffix == '.pdf':
            text = subprocess.run(['pdftotext', '-layout', str(src), '-'], capture_output=True,
                                  text=True, errors='replace').stdout
            pages = text.split('\f')
        elif src.suffix == '.epub':
            text = subprocess.run(['pandoc', '-t', 'plain', str(src)], capture_output=True,
                                  text=True, errors='replace').stdout
            pages = [text[i:i + 3000] for i in range(0, len(text), 3000)]  # no page numbers in EPUB
        else:
            continue
        out = DEST/'pages'/f'{stem}.tsv'
        with out.open('w', encoding='utf-8') as f:
            for n, page in enumerate(pages, 1):
                f.write(f'{n}\t' + re.sub(r'\s+', ' ', page).strip() + '\n')
        print(f'{len(pages):5d} pagine  {stem}')


def search(query, n, book):
    rx = re.compile(query, re.I)
    hits = 0
    for tsv in sorted((DEST/'pages').glob('*.tsv')):
        if book and book.lower() not in tsv.stem.lower():
            continue
        for line in tsv.open(encoding='utf-8'):
            page, text = line.rstrip('\n').split('\t', 1)
            for m in rx.finditer(text):
                s = max(0, m.start() - 160)
                print(f'[{tsv.stem[:40]} | pdf p.{page}] …{text[s:m.end() + 160]}…')
                hits += 1
                if hits >= n:
                    return


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--ref', default='origin/main')
    q = sub.add_parser('search'); q.add_argument('query'); q.add_argument('-n', type=int, default=20)
    q.add_argument('--book')
    a = p.parse_args()
    prepare(a.ref) if a.cmd == 'prepare' else search(a.query, a.n, a.book)
