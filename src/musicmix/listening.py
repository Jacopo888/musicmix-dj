"""Deterministic blind A/B packaging; human judgements remain external data."""
from __future__ import annotations

from html import escape
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

from .paths import localize


def _sha_file(path: Path) -> str:
    """Return a content identity for an audio file without decoding it.

    The renderer writes deterministic WAV files, so hashing the bytes is a
    sufficient waveform/PCM identity for the listening pack.  Keeping this
    helper byte based also makes the deduplication rule independent of the
    audio decoder available on the host.
    """
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate_pairs(pairs: list[dict]) -> tuple[list[tuple[dict, tuple[str, str]]], dict]:
    """Select a stable set of trials whose audio hashes are all distinct.

    A pair is rejected when its two sides are the same bytes, or when either
    side was already accepted in an earlier pair.  The first-wins policy is
    intentional and auditable: it preserves caller ordering and never silently
    replaces a candidate with a different baseline.  The skipped records make
    the tradeoff visible when a baseline is reused across many candidates.
    """
    accepted: list[tuple[dict, tuple[str, str]]] = []
    skipped: list[dict] = []
    seen: dict[str, dict] = {}
    reason_counts: dict[str, int] = {}

    for input_index, pair in enumerate(pairs):
        left_path = localize(pair['left_path'])
        right_path = localize(pair['right_path'])
        hashes = (_sha_file(left_path), _sha_file(right_path))
        pair_id = str(pair.get('pair_id', input_index))
        if hashes[0] == hashes[1]:
            reason = 'identical_audio_in_pair'
            skipped.append({'input_index': input_index, 'pair_id': pair_id,
                            'reason': reason, 'duplicate_sides': ['left', 'right'],
                            'sha256': [hashes[0], hashes[1]]})
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue

        duplicate_sides = [side for side, digest in zip(('left', 'right'), hashes)
                           if digest in seen]
        if duplicate_sides:
            conflicts = []
            for digest in hashes:
                prior = seen.get(digest)
                if prior is not None and prior not in conflicts:
                    conflicts.append(prior)
            reason = 'audio_reused_across_trials'
            skipped.append({'input_index': input_index, 'pair_id': pair_id,
                            'reason': reason, 'duplicate_sides': duplicate_sides,
                            'sha256': [hashes[0], hashes[1]],
                            'conflicts_with': conflicts})
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue

        accepted.append((pair, hashes))
        seen[hashes[0]] = {'input_index': input_index, 'pair_id': pair_id, 'side': 'left'}
        seen[hashes[1]] = {'input_index': input_index, 'pair_id': pair_id, 'side': 'right'}

    audit = {
        'enabled': True,
        'selection_strategy': 'stable_first_accepted_pair',
        'input_pair_count': len(pairs),
        'accepted_pair_count': len(accepted),
        'skipped_pair_count': len(skipped),
        'unique_audio_count': len(seen),
        'reason_counts': reason_counts,
        'skipped': skipped,
        # This is a policy reminder for downstream study design, not a
        # perceptual score: repeated baselines can make a global unique-WAV
        # pack admit only one of several otherwise valid candidates.
        'fairness_note': ('A global unique-audio pack admits one trial per reused '
                          'baseline hash; use separate baseline strata when every '
                          'new candidate needs an independent comparison.'),
    }
    return accepted, audit


def enable_seek_controls(html: str) -> str:
    """Upgrade player controls without changing audio, trial order or answers."""
    if 'data-player-version="2"' in html:
        return html
    counter = 0
    def player(match):
        nonlocal counter
        identity = f'musicmix_player_{counter}'
        counter += 1
        tag = match.group(0).replace('<audio ', f'<audio id="{identity}" ', 1)
        return (tag + f'<span data-player-version="2"><button type="button" '
                f'onclick="musicmixSeek(\'{identity}\',-5)">−5 s</button> '
                f'<button type="button" onclick="musicmixSeek(\'{identity}\',5)">+5 s</button> '
                f'<button type="button" onclick="musicmixJump(\'{identity}\',10)">Al raccordo · 10 s</button></span>')
    html = re.sub(r'<audio\b[^>]*></audio>', player, html)
    script = '''<script>
function musicmixJump(id,time){const audio=document.getElementById(id);
 if(!Number.isFinite(audio.duration)){
  audio.addEventListener('loadedmetadata',()=>musicmixJump(id,time),{once:true});audio.load();return;}
 audio.currentTime=Math.max(0,Math.min(time,audio.duration));}
function musicmixSeek(id,delta){const audio=document.getElementById(id);musicmixJump(id,audio.currentTime+delta);}
</script>'''
    return html.replace('</html>', script+'</html>')


def make_listening_test(output: Path, pairs: list[dict], seed: int = 271828,
                        *, run_id: str | None = None, title: str = 'MusicMix',
                        feedback_filename: str | None = None,
                        no_repeated_audio: bool = False,
                        collect_motion: bool = False) -> dict:
    """Copy matched-policy audio pairs; save a separate key, never in HTML.

    Each pair has left_path/right_path, left_system/right_system, pair_id.
    This is an informal paired listening pilot, not a powered user study.
    With ``no_repeated_audio=True``, file-byte SHA-256 identities enforce a
    strict unique-audio budget and the skipped-pair audit is persisted beside
    the pack.  The default remains permissive for compatibility with existing
    judged packs.  With ``collect_motion=True``, the V2 export also requests
    an external continuation/resolution annotation for each A/B segment.
    """
    # Keep the legacy/default path byte-for-byte compatible.  The optional
    # strict mode is deliberately opt-in because older judged packs may reuse
    # a baseline or even use one fixture on both sides.
    input_pairs = list(pairs)
    if no_repeated_audio:
        selected_records, dedup_audit = _deduplicate_pairs(input_pairs)
    else:
        selected_records = [(pair, None) for pair in input_pairs]
        dedup_audit = None

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    identity_payload = {'run_id': run_id, 'seed': seed,
                        'pairs': [{k: str(v) for k, v in pair.items()} for pair in input_pairs]}
    if no_repeated_audio:
        identity_payload.update({
            'no_repeated_audio': True,
            'accepted_pair_ids': [str(pair.get('pair_id', index))
                                  for index, (pair, _hashes) in enumerate(selected_records)],
            'accepted_audio_sha256': [list(hashes) for _pair, hashes in selected_records],
        })
    if collect_motion:
        # Keep this field absent when disabled so old pack identities remain
        # byte-for-byte compatible with their already judged counterparts.
        identity_payload['collect_motion'] = True
    identity = json.dumps(identity_payload, sort_keys=True)
    test_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', title).strip('_')[:90] or 'MusicMix'
    filename = feedback_filename or f'giudizi_{slug}_{test_id}.json'
    if Path(filename).name != filename or not re.fullmatch(r'[a-zA-Z0-9_.-]+\.json', filename):
        raise ValueError('feedback_filename must be a plain JSON filename')
    metadata = {'test_id': test_id, 'run_id': run_id, 'title': title, 'feedback_filename': filename}
    if no_repeated_audio:
        metadata.update({
            'no_repeated_audio': True,
            'audio_dedup': {key: value for key, value in dedup_audit.items()
                            if key not in {'skipped', 'fairness_note'}},
            # Context is an external listening annotation.  These labels are
            # never inferred from RMS, chroma, or rap/vocal presence.
            'context_labels': ['expectation', 'tension', 'continuation', 'release'],
            'context_label_policy': 'external_annotation_only',
        })
    if collect_motion:
        metadata.update({
            'collect_motion': True,
            'motion_labels': ['continues', 'resolves', 'interrupts', 'unclear'],
            'motion_label_policy': 'external_annotation_only',
        })
    # Balance system position within each test instead of allowing all new
    # proposals to appear on the same side by chance.
    flips = [bool(index % 2) for index in range(len(selected_records))]
    rng.shuffle(flips)
    trials = []
    key = []
    blocks = []
    for number, (pair, hashes) in enumerate(selected_records, 1):
        sources = [(localize(pair['left_path']), pair['left_system'], None if hashes is None else hashes[0]),
                   (localize(pair['right_path']), pair['right_system'], None if hashes is None else hashes[1])]
        if flips[number-1]:
            sources.reverse()
        audio = []
        for label, (source, system, audio_hash) in zip(['A', 'B'], sources):
            target = output/f'trial_{number:02d}_{label}.wav'
            shutil.copyfile(source, target)
            audio.append((label, target.name))
            row = {'trial': number, 'label': label, 'system': system,
                   'source': str(source.resolve()), 'pair_id': pair['pair_id']}
            if no_repeated_audio:
                row['audio_sha256'] = audio_hash
            key.append(row)
        trial = {'trial': number, 'pair_id': pair['pair_id'], 'answer': None,
                 'word_cut_a': None, 'word_cut_b': None, 'comment': ''}
        # Preserve an explicitly supplied semantic context annotation without
        # manufacturing a quality label or deriving one from the waveform.
        if 'context_label' in pair:
            trial['context_label'] = pair['context_label']
        if 'context_labels' in pair:
            labels = pair['context_labels']
            trial['context_labels'] = [labels] if isinstance(labels, str) else list(labels)
        if collect_motion:
            trial['motion_a'] = None
            trial['motion_b'] = None
        trials.append(trial)
        motion_options = (
            '<option value="">Non valutato</option>'
            '<option value="continues">continua</option>'
            '<option value="resolves">trova una risoluzione</option>'
            '<option value="interrupts">si interrompe</option>'
            '<option value="unclear">non chiaro</option>'
        )
        motion_controls = ''
        if collect_motion:
            motion_controls = (
                '<p>Dopo il cambio, l’andamento (relazione tra i segmenti, non volume): '
                f'<label>A <select id="motion_a_{number}">{motion_options}</select></label> '
                f'<label>B <select id="motion_b_{number}">{motion_options}</select></label></p>'
            )
        controls = ''.join(f'<p>{label}<br><audio controls preload="none" src="{escape(name)}"></audio></p>'
                           for label, name in audio)
        block = (f'<section><h2>Confronto {number}</h2>{controls}'
                 f'<label>Preferenza <select id="answer_{number}"><option value="">Scegli</option>'
                 '<option>A</option><option>B</option><option value="tie">Pari</option>'
                 '<option value="neither">Nessuna convincente</option></select></label>'
                 f'<p><label>Parola tagliata in A <input id="cut_a_{number}" type="checkbox"></label> '
                 f'<label>in B <input id="cut_b_{number}" type="checkbox"></label></p>'
                 + motion_controls
                 + f'<label>Nota <input id="comment_{number}" type="text"></label></section>')
        blocks.append(block)
    template = json.dumps(trials, ensure_ascii=True).replace('<', '\\u003c')
    public_metadata = json.dumps(metadata, ensure_ascii=True).replace('<', '\\u003c')
    pack_status = ('no_unique_comparisons' if no_repeated_audio and not trials
                   else 'awaiting_human_judgements')
    if no_repeated_audio:
        metadata['status'] = pack_status
        public_metadata = json.dumps(metadata, ensure_ascii=True).replace('<', '\\u003c')
    if no_repeated_audio and not trials:
        save_controls = '<p role="status">Nessun confronto nuovo disponibile.</p>'
    else:
        motion_save = ''
        if collect_motion:
            motion_save = ('motion_a:document.getElementById(`motion_a_${r.trial}`).value||null,'
                           'motion_b:document.getElementById(`motion_b_${r.trial}`).value||null,')
        save_controls = ('<button onclick="save()">Salva giudizi JSON</button><p id="save_status" role="status"></p><script>'
            + 'const template=' + template + ';'
            + 'const metadata=' + public_metadata + ';'
            + 'async function save(){const rows=template.map(r=>({...r,'
              'answer:document.getElementById(`answer_${r.trial}`).value||null,'
              'word_cut_a:document.getElementById(`answer_${r.trial}`).value?document.getElementById(`cut_a_${r.trial}`).checked:null,'
              'word_cut_b:document.getElementById(`answer_${r.trial}`).value?document.getElementById(`cut_b_${r.trial}`).checked:null,'
              + motion_save
              + 'comment:document.getElementById(`comment_${r.trial}`).value}));'
              'const payload={schema_version:"musicmix.listening.2",...metadata,judgements:rows};'
              'const status=document.getElementById("save_status");'
              'if(location.protocol==="http:"){try{const response=await fetch("/api/judgements",'
              '{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});'
              'const result=await response.json();if(!response.ok)throw new Error(result.error||"Salvataggio fallito");'
              'status.textContent="Salvato: "+result.saved_path;return;'
              '}catch(error){status.textContent="Errore: "+error.message;return;}}'
              'const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}));'
              'const a=document.createElement("a");a.href=url;a.download=metadata.feedback_filename;a.click();'
              'status.textContent="File scaricato: "+metadata.feedback_filename;'
              'setTimeout(()=>URL.revokeObjectURL(url),1000);}'
            + '</script>')
    html = ('<!doctype html><html lang="it"><meta charset="utf-8"><title>MusicMix — ascolto A/B</title>'
            '<style>body{font:17px system-ui;max-width:750px;margin:40px auto;padding:0 20px;line-height:1.5}'
            'section{border-top:1px solid #aaa;margin:32px 0;padding-top:12px}audio{width:100%}'
            'input[type=text]{width:75%}button,select{font:inherit;padding:8px}</style>'
            f'<h1>{escape(title)}</h1><p>Ascolta A e B allo stesso volume. Valuta continuità '
            'musicale, fraseggio e parole interrotte. I due estratti usano la stessa modalità di rendering; '
            'l’ordine dei sistemi è nascosto. È un pilot informale.</p>'
            + ''.join(blocks)
            + save_controls + '</html>')
    (output/'index.html').write_text(enable_seek_controls(html), encoding='utf-8')
    (output/'key.json').write_text(json.dumps({'seed': seed, **metadata, 'trials': key}, indent=2), encoding='utf-8')
    (output/'test.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'judgements_template.json').write_text(json.dumps(trials, indent=2), encoding='utf-8')
    result = {'path': str((output/'index.html').resolve()), 'trials': len(trials), 'seed': seed, **metadata,
              'key_path': str((output/'key.json').resolve()), 'status': pack_status}
    if no_repeated_audio:
        (output/'dedup.json').write_text(json.dumps(dedup_audit, ensure_ascii=False, indent=2), encoding='utf-8')
        result['dedup'] = dedup_audit
    return result
