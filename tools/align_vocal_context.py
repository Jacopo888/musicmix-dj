"""Optional local Whisper word-timing experiment on short vocal contexts.

This is ASR on the mixed signal, not a validated singing/phoneme aligner.
Runtime and model downloads stay in .cache; the baseline has no ASR dependency.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('contexts', type=Path)
    parser.add_argument('--model', default='base.en')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime = root/'.cache/asr_runtime'
    if runtime.is_dir():
        sys.path.insert(0, str(runtime))
    from faster_whisper import WhisperModel
    started = time.perf_counter()
    print('Loading local ASR model; first execution downloads model weights.', flush=True)
    model = WhisperModel(args.model, device='cpu', compute_type='int8',
                         download_root=str(root/'.cache/models'), cpu_threads=4)
    output = {'model': args.model, 'backend': 'faster-whisper',
              'backend_version': importlib.metadata.version('faster-whisper'),
              'method': 'mixed-audio ASR word timestamps, not forced phoneme alignment',
              'warnings': ['Unverified on singing; probabilities are not calibrated alignment accuracy.'],
              'tracks': {}}
    for side, offset in [('a', 124.0), ('b', 3.0)]:
        print(f'Analysing context {side}', flush=True)
        segments, info = model.transcribe(str(args.contexts/f'context_{side}.wav'),
                                          language='en', beam_size=5, word_timestamps=True,
                                          vad_filter=False, condition_on_previous_text=False)
        words = []
        for segment in segments:
            for word in segment.words or []:
                words.append({'start_seconds': offset+word.start, 'end_seconds': offset+word.end,
                              'text': word.word, 'probability': word.probability})
        output['tracks'][side] = {'context_offset_seconds': offset, 'words': words}
        print(f'Context {side}: {len(words)} words', flush=True)
        (args.contexts/'asr_words.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    output['elapsed_seconds'] = time.perf_counter()-started
    (args.contexts/'asr_words.json').write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
