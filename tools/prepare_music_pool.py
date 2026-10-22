"""Canonical pool inventory: exact sources, provider binding, explicit gaps."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from musicmix.audio import load_audio
from musicmix.ingest import decode_audio, sha256_file

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus', '.webm', '.aac'}


def prepare(config, output):
    settings = json.loads(config.read_text(encoding='utf-8'))
    inventory = []
    for spec in settings['tracks']:
        record = dict(spec, status='pending', missing=[])
        if spec.get('source_path'):
            sources = [ROOT/spec['source_path']]
        else:
            sources = sorted(p for p in (ROOT/spec['source_directory']).glob('*') if p.suffix.lower() in AUDIO_EXTENSIONS)
        if len(sources) != 1 or not sources[0].is_file():
            record['missing'].append('unique_source_audio')
            inventory.append(record); continue
        source = sources[0].resolve(); canonical = ROOT/spec['canonical_path']
        ingest_path = canonical.with_suffix('.ingest.json')
        if not canonical.exists():
            provenance = decode_audio(source, canonical)
            ingest_path.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        elif ingest_path.exists():
            provenance = json.loads(ingest_path.read_text(encoding='utf-8'))
            if provenance['source_sha256'] != sha256_file(source) or provenance['canonical_sha256'] != sha256_file(canonical):
                raise ValueError(f'Stale canonical source for {spec["slug"]}')
        track = load_audio(canonical)
        if track.sample_rate != 48000:
            raise ValueError('Pool requires canonical 48 kHz audio')
        record.update(source_path=str(source), source_sha256=sha256_file(source),
                      canonical_path=str(canonical.resolve()), track_id=track.track_id,
                      sample_rate=track.sample_rate, duration_samples=track.duration_samples,
                      duration_seconds=track.duration_samples/track.sample_rate)
        metadata = list(source.parent.glob('*.info.json'))
        if metadata:
            info = json.loads(metadata[0].read_text(encoding='utf-8'))
            record['download_metadata'] = {k: info.get(k) for k in ('id','title','channel','channel_id','webpage_url','duration','extractor','format_id','acodec','abr')}
            record['download_metadata_sha256'] = sha256_file(metadata[0])
        for evidence in ('structure', 'vocals'):
            path = ROOT/spec[evidence+'_path']; record[evidence+'_path'] = str(path.resolve())
            if not path.is_file():
                record['missing'].append(evidence); continue
            document = json.loads(path.read_text(encoding='utf-8'))
            if (document.get('track_id'),document.get('sample_rate'),document.get('duration_samples')) != (track.track_id,track.sample_rate,track.duration_samples):
                raise ValueError(f'{spec["slug"]}: {evidence} identity mismatch')
            if document.get('status') in {'unavailable','failed','timeout','error'} or document.get('analysis_status') == 'fallback':
                record['missing'].append(evidence+'_provider_unavailable')
            record[evidence+'_sha256'] = sha256_file(path)
        record['status'] = 'ready' if not record['missing'] else 'canonical_ready_evidence_pending'
        inventory.append(record)
    identities = [t['track_id'] for t in inventory if t.get('track_id')]
    if len(set(identities)) != len(identities): raise ValueError('Duplicate canonical audio in pool')
    ready = sum(t['status']=='ready' for t in inventory)
    result = {'schema_version':'musicmix.pool-inventory.1','requested_count':len(inventory),
              'ready_count':ready,'status':'ready' if ready == len(inventory) else 'incomplete',
              'config_path':str(config.resolve()),'config_sha256':sha256_file(config),'tracks':inventory,
              'directed_pair_count_when_complete':len(inventory)*(len(inventory)-1)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8'); temporary.replace(output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/music_pool_v8.json')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/pool_v8/catalog.json')
    args = parser.parse_args(); report = prepare(args.config, args.output)
    print(json.dumps({'status':report['status'],'ready':report['ready_count'],
                      'tracks':[{k:t.get(k) for k in ('slug','status','duration_seconds','missing')} for t in report['tracks']]},indent=2))
