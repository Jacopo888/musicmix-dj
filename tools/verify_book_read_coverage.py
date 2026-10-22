"""Check the reading ledgers against original page/spine manifests.

This checks accounting, not comprehension: extraction and file existence are
never treated as evidence that a model read a page.
"""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK_ALIASES = {
    'butler_unlocking_the_groove_2006': 'Butler',
    'katz_groove_music_2012': 'Katz',
    'williams_rhymin_stealin_2013': 'Williams',
}

def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def verify(manifest_path, output, snapshot='6f03b33'):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    books = {book['slug']: book for book in manifest['books']}
    expected = {}
    for slug, book in books.items():
        if book['format'] == 'pdf':
            expected[slug] = {str(i) for i in range(1, book['page_count']+1)}
        else:
            expected[slug] = {row['chunk_id'] for row in records(Path(book['chunks_jsonl']))}
    seen = {slug: {} for slug in books}
    errors = []
    accepted = {'text_read', 'visual_read', 'blank_verified'}
    ledger_root = output/'readers'
    for ledger in sorted(ledger_root.glob('*.jsonl')):
        try:
            rows = records(ledger)
        except (OSError, ValueError) as error:
            errors.append({'file': ledger.name, 'error': str(error)})
            continue
        for row in rows:
            original_slug = row.get('book_id')
            slug = BOOK_ALIASES.get(original_slug, original_slug)
            if slug not in books:
                errors.append({'file': ledger.name, 'error': 'Unknown book', 'book': slug})
                continue
            value = row.get('physical_pdf_index', row.get('unit_id', row.get('page')))
            if books[slug]['format'] == 'epub':
                value = row.get('chunk_id', value)
            else:
                match = re.fullmatch(r'(?:physical-|pdf-)?(\d+)', str(value))
                if match:
                    value = int(match.group(1))
            key = str(value)
            if key not in expected[slug]:
                errors.append({'file': ledger.name, 'error': 'Unexpected unit', 'book': slug, 'unit': key})
                continue
            if key in seen[slug]:
                errors.append({'file': ledger.name, 'error': 'Duplicate unit', 'book': slug, 'unit': key})
            if not str(row.get('topic','')).strip():
                errors.append({'file': ledger.name, 'error': 'Missing reading note', 'book': slug, 'unit': key})
            seen[slug][key] = {'status': row.get('status'), 'ledger': ledger.name}
    result = {'snapshot': snapshot, 'coverage_is_reader_attestation_not_comprehension_proof': True,
              'books': {}, 'errors': errors}
    for slug, wanted in expected.items():
        covered = {k for k,v in seen[slug].items() if v['status'] in accepted}
        missing = wanted-covered
        statuses = Counter(v['status'] for v in seen[slug].values())
        result['books'][slug] = {
            'format': books[slug]['format'], 'expected_units':len(wanted), 'read_units':len(covered),
            'missing_units':sorted(missing, key=(lambda x: int(x)) if books[slug]['format']=='pdf' else str),
            'status_counts':dict(statuses), 'complete':not missing,
            'source_sha256':books[slug]['source_sha256'],
        }
    result['complete'] = not errors and all(b['complete'] for b in result['books'].values())
    result['read_pdf_pages'] = sum(b['read_units'] for b in result['books'].values() if b['format']=='pdf')
    result['expected_pdf_pages'] = sum(b['expected_units'] for b in result['books'].values() if b['format']=='pdf')
    result['read_epub_chunks'] = sum(b['read_units'] for b in result['books'].values() if b['format']=='epub')
    (output/'COVERAGE.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({**{k:result[k] for k in ['complete','read_pdf_pages','expected_pdf_pages','read_epub_chunks']}, 'error_count':len(errors)},ensure_ascii=False))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT.parent/'.book_audit_private_20261021'/'manifest.json')
    parser.add_argument('--output',type=Path,default=ROOT/'conoscenza_mix'/'libri'/'audit_2026-10-21')
    parser.add_argument('--snapshot',default='6f03b33',help='Production revision reviewed; not a source-book revision')
    args=parser.parse_args()
    result = verify(args.manifest,args.output,args.snapshot)
    raise SystemExit(0 if result['complete'] else 1)
