"""Joint real-source evidence, with playback duration independent of analysis.

Combines word safety, measured vocal descriptors and four-context entry events.
The ranking is an explicitly uncalibrated engineering policy. It never converts
these observations into a perceptual rating or certified phrase completion.
"""
from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
import hashlib
import importlib.util
import os
import numpy as np

from .audio import load_audio
from .functional_event import assess_functional_event
from .ingest import sha256_file
from .paths import localize
from .phrase_evidence import analyze_phrase_evidence
from .pooled_trajectory import PooledTrajectory
from . import vocal_completion as vocal_module
from .vocal_completion import VocalFeatureTrack, extract_vocal_features, describe_vocal_cue


VERSION='joint-evidence-pipeline-v12.2-syntax'
SUPPORTED_EVENTS={'post_entry_attack_stabilized','sustained_motion','continuation_without_local_event'}


class EvidencePipeline:
    def __init__(self,catalog,*,pretrained_vad=True,syntax_factory=None,cache_dir=None):
        self.entries={e['track_id']:e for e in catalog['tracks']}
        self.tracks={};self.series={};self.words={};self.structure={};self.stems={}
        self.syntax={};self.syntax_factory=syntax_factory;self.pretrained_vad=pretrained_vad
        self.model_records={}
        self.cache_dir=Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir:self.cache_dir.mkdir(parents=True,exist_ok=True)

    def track(self,identity):
        if identity not in self.tracks:
            entry=self.entries[identity];track=load_audio(entry['canonical_path'])
            if (track.track_id,track.sample_rate,track.duration_samples)!=(identity,48000,entry['duration_samples']):
                raise ValueError('Canonical track changed')
            self.tracks[identity]=track
        return self.tracks[identity]

    def providers(self,identity):
        if identity in self.words:return
        entry=self.entries[identity]
        for kind,target in [('vocals',self.words),('structure',self.structure)]:
            path=localize(entry[kind+'_path'])
            if sha256_file(path)!=entry[kind+'_sha256']:raise ValueError(f'Changed {kind} provider')
            value=json.loads(path.read_text(encoding='utf8'))
            for key in ('track_id','sample_rate','duration_samples'):
                if value.get(key)!=entry[key]:raise ValueError(f'Provider {key} mismatch')
            target[identity]=value

    @lru_cache(maxsize=None)
    def profile(self,identity,cue,side):
        self.providers(identity)
        if identity not in self.series:
            self.series[identity]=PooledTrajectory(self.track(identity),[b['sample'] for b in self.structure[identity]['beats']])
        value=self.series[identity].profile(cue,side=side)
        return value if value['windows'] else None

    def stem_features(self,identity):
        self.providers(identity)
        if identity not in self.stems:
            doc=self.words[identity];separator=doc.get('separator',{})
            if separator.get('status')!='complete' or separator.get('input_sha256')!=identity:
                raise ValueError('A real canonical-bound separated vocal stem is required')
            path=localize(separator['output_path']);stem=load_audio(path)
            if (stem.sample_rate,stem.duration_samples)!=(48000,self.entries[identity]['duration_samples']):
                raise ValueError('Vocal stem clock mismatch')
            checkpoint='unavailable'
            spec=importlib.util.find_spec('silero_vad') if self.pretrained_vad else None
            if spec and spec.origin:
                weight=Path(spec.origin).parent/'data/silero_vad.jit'
                if weight.is_file():checkpoint=sha256_file(weight)
            key=hashlib.sha256(json.dumps([stem.track_id,sha256_file(Path(vocal_module.__file__)),self.pretrained_vad,checkpoint]).encode()).hexdigest()
            cache=self.cache_dir/f'{key}.npz' if self.cache_dir else None
            if cache and cache.exists():
                with np.load(cache,allow_pickle=False) as saved:
                    metadata=json.loads(saved['metadata'].item())
                    arrays={k:saved[k] for k in saved.files if k!='metadata'}
                features=VocalFeatureTrack(**metadata,**arrays)
            else:
                print('Measure separated vocals: '+self.entries[identity]['slug'],flush=True)
                features=extract_vocal_features(stem.samples,48000,use_pretrained_vad=self.pretrained_vad)
                if cache:
                    arrays={k:v for k,v in vars(features).items() if isinstance(v,np.ndarray)}
                    metadata={k:v for k,v in vars(features).items() if k not in arrays}
                    temp=cache.with_suffix(f'.{os.getpid()}.tmp')
                    with temp.open('xb') as raw:np.savez_compressed(raw,metadata=np.asarray(json.dumps(metadata)),**arrays)
                    temp.replace(cache)
            self.stems[identity]=features
            # Optional learned VAD availability is reported, never assumed.
            vad=getattr(features,'pretrained_vad',None)
            self.model_records[identity]={'stem_sha256':stem.track_id,'stem_path':str(path),
                'canonical_sha256':identity,'pretrained_vad':vad,'feature_cache_key':key,
                'feature_method':'documented DSP + optional pretrained Silero VAD'}
        return self.stems[identity]

    @lru_cache(maxsize=None)
    def voice(self,identity,cue):
        features=self.stem_features(identity);words=self.words[identity]
        value=describe_vocal_cue(features,cue,words=words.get('words'),segments=words.get('asr_segments'))
        return value

    @lru_cache(maxsize=None)
    def guard(self,identity,cue,side):
        self.providers(identity)
        return analyze_phrase_evidence(self.words[identity],cue,48000,self.entries[identity]['duration_samples'],side)

    def syntax_at(self,identity,cue):
        if self.syntax_factory is None:return {'available':False,'reason':'syntactic_provider_unavailable'}
        self.providers(identity)
        if identity not in self.syntax:
            self.syntax[identity]=self.syntax_factory(self.words[identity],self.entries[identity])
        return self.syntax[identity](cue)

    def assess(self,candidate,baseline_score=None,*,include_profiles=False):
        a,b=candidate.track_a,candidate.track_b;ca,cb=candidate.cue_out_sample,candidate.cue_in_sample
        pa,pb=self.guard(a,ca,'outgoing'),self.guard(b,cb,'incoming')
        profiles={'A_before':self.profile(a,ca,'outgoing'),'A_after':self.profile(a,ca,'incoming'),
                  'B_before':self.profile(b,cb,'outgoing'),'B_after':self.profile(b,cb,'incoming')}
        va,vb=self.voice(a,ca),self.voice(b,cb)
        syntax=self.syntax_at(a,ca)
        functional=assess_functional_event(**profiles,vocal_completion={
            'provider':'separate_word_and_measured_vocal_channels',
            'word_active_at_cue':bool(pa['active_utterance'] or pb['active_utterance']),
            # No learned semantic-completion label exists in these channels.
            'completion_supported':False,'pickup_supported':False})
        safe=all(g.get('timing_status',g['status'])=='supported' and not g['active_utterance'] and not g['pickup_uncertainty'] for g in (pa,pb))
        completion=va.get('completion_evidence',{}).get('score_uncalibrated')
        pickup=vb.get('incoming_pickup_evidence',{}).get('score_uncalibrated')
        physical_supported=not functional.abstained and functional.label in SUPPORTED_EVENTS
        available=[float(v) for v in (completion,pickup) if v is not None]
        bottleneck=min(available) if len(available)==2 else None
        # A conservative weakest-channel ordering, not fitted to listening
        # scores. Missing components are displayed and do not become zeros.
        syntax_open=syntax.get('syntax_status')=='open_core_dependency_arc'
        rank=(int(safe),int(physical_supported),int(not syntax_open),bottleneck if bottleneck is not None else -1.0,
              functional.score if functional.score is not None else -1.0,
              float(baseline_score) if baseline_score is not None else -1.0)
        result={'pipeline_version':VERSION,'candidate':asdict(candidate),'baseline_score':baseline_score,
                'word_guard':{'A':pa,'B':pb},'voice':{'A':va,'B':vb},'syntax_A':syntax,
                'functional_event':asdict(functional),'safe_word_timing':safe,
                'eligible':safe and physical_supported and not syntax_open,'rank_key':rank,'voice_bottleneck':bottleneck,
                'confidence':None,'musical_quality':None,'phrase_closure':'unknown',
                'ranking_semantics':'word timing gate, supported physical event, no observed open core dependency, weakest of A vocal ending/B onset descriptors, event adjustment, baseline tie-breaker; uncalibrated'}
        if include_profiles:result['profiles']=profiles
        return result
