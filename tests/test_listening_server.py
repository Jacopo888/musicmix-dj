import http.client
from http.server import ThreadingHTTPServer
import json
import threading

import pytest

from musicmix.listening import make_listening_test
from musicmix.listening_server import make_handler, save_judgements


def fixture(tmp_path):
    source = tmp_path/'source.wav'
    source.write_bytes(b'audio')
    test_path = tmp_path/'test'
    make_listening_test(test_path, [{'left_path': source, 'right_path': source,
                                   'left_system': 'old', 'right_system': 'new', 'pair_id': 'pair'}],
                        run_id='run-1', title='Reminder verso Too Many Nights V4')
    metadata = json.loads((test_path/'test.json').read_text())
    tests = {metadata['test_id']: {'metadata': metadata, 'path': test_path}}
    payload = {'schema_version': 'musicmix.listening.2', **metadata,
               'judgements': json.loads((test_path/'judgements_template.json').read_text())}
    payload['judgements'][0].update(answer='A', word_cut_a=False, word_cut_b=True)
    return tests, payload


def test_saves_identified_feedback_to_fixed_folder_without_overwriting(tmp_path):
    tests, payload = fixture(tmp_path)
    first = save_judgements(payload, tests, tmp_path/'giudizi json')
    second = save_judgements(payload, tests, tmp_path/'giudizi json')
    assert first != second
    assert first.parent == second.parent == tmp_path/'giudizi json'
    assert 'Reminder_verso_Too_Many_Nights_V4' in first.name
    assert json.loads(first.read_text())['run_id'] == 'run-1'


def test_rejects_wrong_run_and_unknown_trials_before_writing(tmp_path):
    tests, payload = fixture(tmp_path)
    payload['run_id'] = 'other'
    with pytest.raises(ValueError, match='run_id'):
        save_judgements(payload, tests, tmp_path/'feedback')
    payload['run_id'] = 'run-1'
    payload['judgements'][0]['pair_id'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        save_judgements(payload, tests, tmp_path/'feedback')
    assert not (tmp_path/'feedback').exists()


def test_motion_annotations_are_saved_without_inference_and_invalid_labels_rejected(tmp_path):
    tests, payload = fixture(tmp_path)
    payload['judgements'][0].update(motion_a='resolves', motion_b='interrupts')
    path = save_judgements(payload, tests, tmp_path/'feedback')
    saved = json.loads(path.read_text())
    assert saved['judgements'][0]['motion_a'] == 'resolves'
    payload['judgements'][0]['motion_a'] = 'quiet_therefore_resolved'
    with pytest.raises(ValueError, match='motion'):
        save_judgements(payload, tests, tmp_path/'feedback')
    assert len(list((tmp_path/'feedback').glob('*.json'))) == 1


def test_local_http_roundtrip_and_private_key_is_not_served(tmp_path):
    tests, payload = fixture(tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(tests, tmp_path/'feedback'))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        client.request('GET', f'/test/{payload["test_id"]}/key.json')
        response = client.getresponse()
        assert response.status == 404
        response.read()
        client.request('POST', '/api/judgements', json.dumps(payload), {'Content-Type': 'application/json'})
        response = client.getresponse()
        assert response.status == 200
        target = json.loads(response.read())['saved_path']
        assert json.loads(open(target, encoding='utf-8').read())['test_id'] == payload['test_id']
        client.request('POST', '/api/judgements', json.dumps(payload), {'Origin': 'https://example.com'})
        response = client.getresponse()
        assert response.status == 403
        response.read()
        client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_audio_seek_supports_ranges_head_and_rejects_invalid_ranges(tmp_path):
    tests, payload = fixture(tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(tests, tmp_path/'feedback'))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'/test/{payload["test_id"]}/trial_01_A.wav'
    try:
        for requested, expected, content_range in [
            ('bytes=1-3', b'udi', 'bytes 1-3/5'),
            ('bytes=2-', b'dio', 'bytes 2-4/5'),
            ('bytes=-2', b'io', 'bytes 3-4/5'),
            ('bytes=3-999', b'io', 'bytes 3-4/5'),
        ]:
            client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
            client.request('GET', url, headers={'Range': requested})
            response = client.getresponse()
            assert response.status == 206
            assert response.getheader('Accept-Ranges') == 'bytes'
            assert response.getheader('Content-Range') == content_range
            assert int(response.getheader('Content-Length')) == len(expected)
            assert response.read() == expected
            client.close()
        for requested in ('bytes=9-', 'bytes=3-1', 'bytes=-0', 'bytes=-', 'bytes=0-1,3-4'):
            client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
            client.request('GET', url, headers={'Range': requested})
            response = client.getresponse()
            assert response.status == 416
            assert response.getheader('Content-Range') == 'bytes */5'
            assert response.read() == b''
            client.close()
        client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        client.request('HEAD', url, headers={'Range': 'bytes=1-3'})
        response = client.getresponse()
        assert response.status == 200
        assert response.getheader('Content-Length') == '5'
        assert response.getheader('Accept-Ranges') == 'bytes'
        assert response.read() == b''
        client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
