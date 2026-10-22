"""Optional independent Audiobox clip diagnostics, never a seam-validity oracle."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from importlib import metadata
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.signal import resample_poly

from .audio import AudioTrack
from .ingest import sha256_file

VERSION = 'audiobox-seam-diagnostic-v1'
AXES = {'CE':'Content Enjoyment','CU':'Content Usefulness',
        'PC':'Production Complexity','PQ':'Production Quality'}


def seam_excerpt(track: AudioTrack, join_sample: int, window_seconds=10.0):
    if isinstance(join_sample,bool) or not isinstance(join_sample,int):
        raise ValueError('join_sample must be an integer')
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError('Positive finite window required')
    count=round(window_seconds*track.sample_rate); start=join_sample-count//2; end=start+count
    if start<0 or end>track.duration_samples:
        raise ValueError('Insufficient real context around seam; no padding')
    mono=np.mean(track.samples[start:end],axis=1,dtype=np.float64)
    if not np.isfinite(mono).all():raise ValueError('Non-finite input')
    divisor=math.gcd(track.sample_rate,16000)
    signal=resample_poly(mono,16000//divisor,track.sample_rate//divisor).astype(np.float32)
    return signal,{'source_start_sample':start,'source_end_sample':end,
                   'source_join_sample':join_sample,'source_sample_rate':track.sample_rate,
                   'critic_sample_rate':16000,'critic_duration_samples':len(signal),
                   'seam_seconds_in_excerpt':(join_sample-start)/track.sample_rate}


class AudioboxSeamCritic:
    def __init__(self,project_root:Path,threads=2):
        self.root=Path(project_root); self.threads=threads; self.predictor=None

    def _load(self):
        if self.predictor is not None:return
        for relative in ('.cache/structure_runtime/Lib/site-packages','.cache/asr_runtime','.cache/critic_runtime'):
            path=str(self.root/relative)
            if path not in sys.path:sys.path.insert(0,path)
        import torch
        from safetensors.torch import load_file
        from audiobox_aesthetics.model.aes import AesMultiOutput,Normalize
        from audiobox_aesthetics.infer import AesPredictor
        torch.set_num_threads(self.threads)
        model_root=self.root/'.cache/models/audiobox_aesthetics'
        weight=model_root/'model.safetensors'
        provenance=json.loads((model_root/'provenance.json').read_text(encoding='utf-8'))
        if sha256_file(weight)!=provenance['sha256']:raise ValueError('Critic weight hash mismatch')
        config=json.loads((model_root/'config.json').read_text(encoding='utf-8'))
        model=AesMultiOutput(**config)
        state=load_file(str(weight),device='cpu');model.load_state_dict(state,strict=True);del state
        model.to('cpu');model.eval()
        # Use the published inference implementation with an already loaded,
        # verified local safetensors model; no implicit network/pickle loading.
        predictor=object.__new__(AesPredictor)
        predictor.model=model;predictor.device=torch.device('cpu');predictor.sample_rate=16000
        predictor.data_col='path'
        predictor.target_transform={axis:Normalize(mean=model.target_transform[axis]['mean'],
                                                   std=model.target_transform[axis]['std']) for axis in AXES}
        self.predictor=predictor;self.torch=torch;self.provenance=provenance
        self.package_version=metadata.version('audiobox_aesthetics')

    def score(self,track:AudioTrack,join_sample:int)->dict:
        signal,excerpt=seam_excerpt(track,join_sample)
        self._load()
        waveform=self.torch.from_numpy(signal).unsqueeze(0)
        with self.torch.inference_mode():
            scores=self.predictor.forward([{'path':waveform,'sample_rate':16000}])[0]
        if set(scores)!=set(AXES) or not all(math.isfinite(float(v)) for v in scores.values()):
            raise ValueError('Invalid critic output')
        return {'schema_version':'musicmix.aesthetic-diagnostic.1','version':VERSION,
                'status':'learned_clip_diagnostic_only','track_id':track.track_id,
                'excerpt':excerpt,'axes':scores,'axis_descriptions':AXES,
                'model':{'repository':self.provenance['repo'],'revision':self.provenance['revision'],
                         'sha256':self.provenance['sha256'],'package_version':self.package_version,
                         'device':'cpu','threads':self.threads},
                'paper':'https://arxiv.org/abs/2502.05139','confidence':None,
                'musical_continuity':'not_validated_for_transitions',
                'selector_features_used':False,'used_for_reranking':False,
                'limitations':['Aesthetic axes measure clips, not whether B resolves an expectation from A.',
                               'Scores must be calibrated against held-out music-transition judgements.',
                               'PC is complexity, not a quality objective to maximize.']}
