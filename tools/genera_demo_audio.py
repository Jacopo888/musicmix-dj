"""Create two original synthetic fixtures; these are not evidence of musical quality."""
from pathlib import Path
import argparse
import wave
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rate = 16000
    t = np.arange(20*rate)/rate
    envelope = 0.25 + 0.75*np.exp(-((t % 0.5)/0.035)**2)
    for name, frequencies in [('a', [220, 261.63, 329.63]), ('b', [220, 277.18, 329.63])]:
        notes = np.array(frequencies)[(t//4).astype(int) % len(frequencies)]
        phase = 2*np.pi*np.cumsum(notes)/rate
        signal = 0.5*envelope*(np.sin(phase) + 0.25*np.sin(2*phase))
        with wave.open(str(args.output/f'{name}.wav'), 'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate)
            stream.writeframes(np.round(signal*32767).astype('<i2').tobytes())
    print(args.output.resolve())


if __name__ == '__main__':
    main()
