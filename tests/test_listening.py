import json
from pathlib import Path
import tempfile
import unittest

from musicmix.listening import enable_seek_controls, make_listening_test


class ListeningTests(unittest.TestCase):
    def test_player_upgrade_preserves_sources_and_is_idempotent(self):
        original = '<html><audio controls src="trial_01_A.wav"></audio><script>const template=[1];</script></html>'
        upgraded = enable_seek_controls(original)
        self.assertIn('src="trial_01_A.wav"', upgraded)
        self.assertIn('const template=[1]', upgraded)
        self.assertIn('musicmixSeek', upgraded)
        self.assertIn('Al raccordo', upgraded)
        self.assertEqual(enable_seek_controls(upgraded), upgraded)

    def test_key_separate_audio_preserved_and_seed_repeatable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'one.wav').write_bytes(b'one')
            (root/'two.wav').write_bytes(b'two')
            pair = {'left_path': root/'one.wav', 'right_path': root/'two.wav',
                    'left_system': 'original', 'right_system': 'new', 'pair_id': 'pair'}
            one = make_listening_test(root/'test1', [pair], seed=42)
            two = make_listening_test(root/'test2', [pair], seed=42)
            key = json.loads((root/'test1/key.json').read_text())
            self.assertEqual(key, json.loads((root/'test2/key.json').read_text()))
            html = (root/'test1/index.html').read_text(encoding='utf-8')
            self.assertNotIn('"system":', html)
            self.assertNotIn('original', html)
            for row in key['trials']:
                target = root/'test1'/f'trial_01_{row["label"]}.wav'
                self.assertEqual(target.read_bytes(), Path(row['source']).read_bytes())
            self.assertEqual(one['status'], 'awaiting_human_judgements')
            self.assertEqual(two['trials'], 1)

    def test_balanced_positions_and_export_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'one.wav'; source.write_bytes(b'audio')
            pairs = [{'left_path': source, 'right_path': source, 'left_system': 'old',
                      'right_system': 'new', 'pair_id': f'pair_{index}'} for index in range(4)]
            result = make_listening_test(root/'test', pairs, run_id='run123', title='Forward V4')
            key = json.loads((root/'test/key.json').read_text())
            self.assertEqual(sum(r['system'] == 'new' for r in key['trials'] if r['label'] == 'A'), 2)
            self.assertEqual(result['run_id'], 'run123')
            self.assertEqual(json.loads((root/'test/test.json').read_text())['test_id'], result['test_id'])
            html = (root/'test/index.html').read_text(encoding='utf-8')
            self.assertIn('/api/judgements', html)
            self.assertNotIn('"system":', html)

    def test_no_repeated_audio_deduplicates_by_content_with_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            baseline = root/'baseline.wav'; baseline.write_bytes(b'baseline')
            candidate_one = root/'candidate_one.wav'; candidate_one.write_bytes(b'candidate-one')
            candidate_two = root/'candidate_two.wav'; candidate_two.write_bytes(b'candidate-two')
            same_baseline_copy = root/'baseline_copy.wav'; same_baseline_copy.write_bytes(b'baseline')
            identical_copy = root/'identical_copy.wav'; identical_copy.write_bytes(b'candidate-two')
            pairs = [
                {'left_path': baseline, 'right_path': candidate_one,
                 'left_system': 'v3', 'right_system': 'v4', 'pair_id': 'first'},
                # The baseline bytes are reused.  Strict mode leaves this
                # candidate out rather than copying the same WAV again.
                {'left_path': same_baseline_copy, 'right_path': candidate_two,
                 'left_system': 'v3', 'right_system': 'v4', 'pair_id': 'second'},
                # Different paths with identical bytes on both sides must be
                # rejected as an uninformative A/B comparison.
                {'left_path': identical_copy, 'right_path': candidate_two,
                 'left_system': 'v4', 'right_system': 'v4', 'pair_id': 'identical'},
            ]
            result = make_listening_test(root/'test', pairs, no_repeated_audio=True)
            self.assertEqual(result['trials'], 1)
            audit = result['dedup']
            self.assertEqual(audit['input_pair_count'], 3)
            self.assertEqual(audit['accepted_pair_count'], 1)
            self.assertEqual(audit['skipped_pair_count'], 2)
            self.assertEqual(audit['reason_counts'], {
                'audio_reused_across_trials': 1,
                'identical_audio_in_pair': 1,
            })
            self.assertTrue((root/'test/dedup.json').exists())
            key = json.loads((root/'test/key.json').read_text())
            hashes = [row['audio_sha256'] for row in key['trials']]
            self.assertEqual(len(hashes), len(set(hashes)))
            self.assertEqual(len(hashes), 2)
            self.assertEqual(key['audio_dedup']['accepted_pair_count'], 1)

    def test_no_repeated_audio_keeps_context_annotation_external(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            left = root/'left.wav'; left.write_bytes(b'left')
            right = root/'right.wav'; right.write_bytes(b'right')
            result = make_listening_test(root/'test', [{
                'left_path': left, 'right_path': right,
                'left_system': 'old', 'right_system': 'new', 'pair_id': 'context',
                'context_label': 'tension',
            }], no_repeated_audio=True)
            template = json.loads((root/'test/judgements_template.json').read_text())
            self.assertEqual(template[0]['context_label'], 'tension')
            self.assertEqual(result['context_labels'],
                             ['expectation', 'tension', 'continuation', 'release'])
            self.assertEqual(result['context_label_policy'], 'external_annotation_only')

    def test_collect_motion_adds_relation_controls_and_v2_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            left = root/'left.wav'; left.write_bytes(b'left')
            right = root/'right.wav'; right.write_bytes(b'right')
            pair = {'left_path': left, 'right_path': right,
                    'left_system': 'old', 'right_system': 'new', 'pair_id': 'motion'}
            legacy = make_listening_test(root/'legacy', [pair])
            result = make_listening_test(root/'motion', [pair], collect_motion=True)
            self.assertTrue(result['collect_motion'])
            self.assertNotEqual(result['test_id'], legacy['test_id'])
            template = json.loads((root/'motion/judgements_template.json').read_text())
            self.assertEqual(template[0]['motion_a'], None)
            self.assertEqual(template[0]['motion_b'], None)
            html = (root/'motion/index.html').read_text(encoding='utf-8')
            self.assertIn('Dopo il cambio, l’andamento', html)
            self.assertIn('trova una risoluzione', html)
            self.assertIn('motion_a_1', html)
            self.assertNotIn('motion_a_1', (root/'legacy/index.html').read_text(encoding='utf-8'))

    def test_default_allows_legacy_repeated_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'same.wav'; source.write_bytes(b'same')
            pairs = [{'left_path': source, 'right_path': source,
                      'left_system': 'old', 'right_system': 'new', 'pair_id': str(i)}
                     for i in range(2)]
            result = make_listening_test(root/'test', pairs)
            self.assertEqual(result['trials'], 2)
            self.assertNotIn('dedup', result)
            self.assertFalse((root/'test/dedup.json').exists())

    def test_no_repeated_audio_empty_pack_is_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            same = root/'same.wav'; same.write_bytes(b'same')
            result = make_listening_test(root/'test', [{
                'left_path': same, 'right_path': same,
                'left_system': 'old', 'right_system': 'new', 'pair_id': 'same',
            }], no_repeated_audio=True)
            html = (root/'test/index.html').read_text(encoding='utf-8')
            self.assertEqual(result['trials'], 0)
            self.assertEqual(result['status'], 'no_unique_comparisons')
            self.assertIn('Nessun confronto nuovo disponibile', html)
            self.assertNotIn('Salva giudizi JSON', html)
            self.assertEqual(json.loads((root/'test/test.json').read_text())['status'],
                             'no_unique_comparisons')
            self.assertEqual(result['dedup']['accepted_pair_count'], 0)
