"""Convert runs/dj_phrase_v1/render_manifest.json into a musicmix.dj-review.1 manifest.

The listening page (tools/build_listening_artifact.py) reads dj-review.1; markers come
from each clip's renderer metadata (overlap, effect sweep, phrase fader switch).
"""
import argparse, json
from pathlib import Path

LABELS = {'phrase_cut': 'taglio a fine verso (B dal pickup)',
          'phrase_filter_out': 'filter out a fine verso',
          'same_cue_downbeat_cut': 'stesso punto, taglio sul downbeat',
          'dj_select_default_cut': 'piano standard (senza fine verso)'}


def markers(meta):
    sr = meta['sample_rate']
    sw = meta['switch']
    switch = sw.get('fader_switch_output_sample', sw['output_sample'])
    starts = [meta['overlap']['output_start_sample'], switch]
    eff = meta.get('effect') or {}
    sweep = eff.get('sweep_output_samples')
    if isinstance(sweep, (list, tuple)) and sweep:
        starts.append(sweep[0])
    ends = [meta['overlap']['output_end_sample'], switch]
    return min(starts) / sr, switch / sr, max(ends) / sr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    src = json.loads(a.manifest.read_text(encoding='utf-8'))
    groups = []
    for g in src['groups']:
        items = []
        for it in g['items']:
            meta = json.loads(Path(it['json']).read_text(encoding='utf-8'))
            start, switch, end = markers(meta)
            kind = 'filter_out' if 'filter' in it['version'] else 'cut'
            items.append({'id': f"{g['id']}__{it['version']}", 'kind': kind,
                          'label': LABELS.get(it['version'], it['version']),
                          'audio_path': it['wav'], 'json_path': it['json'],
                          'duration_s': it['duration_s'], 'sample_rate': meta['sample_rate'],
                          'transition_start_s': start, 'switch_s': switch, 'transition_end_s': end,
                          'qa': {'verdict': it['qa_verdict'], 'warn_reasons': it.get('qa_warn') or []},
                          'sha256': it['sha256']})
        reasons = (g.get('exit', {}).get('reasons') or []) + (g.get('entry', {}).get('reasons') or [])
        plan = g.get('plan', {})
        groups.append({'id': g['id'], 'title': f"{g['a']} → {g['b']}",
                       'A_out_s': plan.get('a_out_s'), 'B_in_s': plan.get('b_in_s'),
                       'A_out_bar': plan.get('a_out_bar'), 'B_in_bar': plan.get('b_in_bar'),
                       'reasons': reasons, 'items': items})
    out = {'schema_version': 'musicmix.dj-review.1', 'run': 'dj_phrase_v1',
           'title': 'MusicMix — quando tagliare: fine verso e pickup contro taglio sul downbeat',
           'source': {'render_manifest': str(a.manifest)}, 'groups': groups}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(len(groups), 'gruppi', sum(len(g['items']) for g in groups), 'versioni')


if __name__ == '__main__':
    main()
