"""Controllo offline di integrità documentale e link bibliografici.

Non verifica HTTP, copyright o correttezza dei risultati dei paper.
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = root / 'ricerca'
rows = list(csv.DictReader((base / 'BIBLIOGRAFIA_COMPLETA.csv').open(encoding='utf-8-sig',newline='')))
assert len(rows) == 33
paper_txt = (base / 'PAPER_LINKS.txt').read_text(encoding='utf-8')
all_txt = (base / 'TUTTI_I_LINK.txt').read_text(encoding='utf-8')
for row in rows:
    if int(row['N']) <= 23:
        assert row['Paper full text'] in paper_txt, row['Titolo']
    assert row['Paper full text'] in all_txt, row['Titolo']
    if row['Altro link/codice']:
        assert row['Altro link/codice'] in all_txt, row['Titolo']

schema = json.loads((root / 'schemas/transition_record.schema.json').read_text(encoding='utf-8'))
example = json.loads((root / 'examples/candidate_example.json').read_text(encoding='utf-8'))
assert schema['properties']['schema_version']['const'] == example['schema_version']
assert example['transition']['cue_out_sample'] < example['track_a']['duration_samples']
assert example['transition']['cue_in_sample'] < example['track_b']['duration_samples']

for name in ('01_ROADMAP.md','02_DESIGN_DOC.md','03_CONTRATTI_DATI.md',
             '04_PROTOCOLLO_VALUTAZIONE.md','05_DEBUG_CAUSA_EFFETTO.md',
             '06_TRACCIABILITA_RICERCA.md','07_DECISIONI_ADR.md','08_BACKLOG.md'):
    doc = (root / 'docs' / name).read_text(encoding='utf-8')
    assert 'https://' in doc, name

print(f'OK: {len(rows)} riferimenti e JSON di esempio coerenti; docs presenti.')
try:
    import jsonschema
except ImportError:
    print('Verifica JSON Schema rigorosa: saltata (installare jsonschema).')
else:
    jsonschema.validate(example, schema)
    print('OK: validazione completa schema JSON.')
