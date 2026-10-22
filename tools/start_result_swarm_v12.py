"""Stage minimal task inputs and start one owned OpenCode server for V12."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
from opencode_platform import find_opencode, spawn_server  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/result_swarm_v12'
EXE = find_opencode()
COMMON = '''User explicitly requests DeepSeek4.1Flash-free through OpenCode. Be outcome-oriented: implement and RUN useful work, not just reports. Work only inside your own folder. Do not access other project/user folders or secrets. Source snapshot src/musicmix is provided; public research context in knowledge/. No raw user feedback or private logs are provided. Use normal OpenCode tools, webfetch/websearch/bash/edit with no approval questions. Save code and a concise REPORT.md with exact commands, observed outputs, source URLs and remaining blockers. Reuse official pretrained models/code rather than inventing replacements. CPU threads1. No global installs: pip --target runtime/ or local venv only, respect your download budget. Download only official/public/publisher/author resources. Archive paper HTML/PDF/text in research/originals and research/texts with sources.json (URL, DOI, SHA256, license, actual sections read, extraction vs reading). Keep research summaries Italian and distinguish measured evidence, theory, engineering hypothesis. Do not claim 10/10/SOTA or make up performance. A missing package is something to resolve within budget, not automatically a reason to stop. Do not fit to the user's prior ratings (not provided). Playback must stay20s (10s per side), analysis may use longer contexts. Musical tension is not RMS, phrase closure is not a word gap, chorus label is not resolution. Focus on one concrete result within25min. Use Python on Windows and PYTHONPATH=src for project imports. Finish a runnable artifact and concise report, not broad unbounded investigation.
'''
JOBS = {
 'official_tension': '''Own tools/run_official_tension_v12.py, runtime/, outputs/, research/. Goal actual OFFICIAL TenseMusic audio inference on staged real audio snippets. Official https://github.com/vivienbarchet/TenseMusic revision0e704c7bdb06fec2144477f7e35004931dd038ef; paper DOI10.1371/journal.pone.0296385. Read official source and dependency graph. Previous attempt stopped at missinglibrosa before audio; resolve dependencies in isolated runtime. Budget total downloaded500MB, CPU only. Prefer supported BasicPitch ONNX backend rather than TensorFlow if official code supports it; never replace six published features with homemade RMS/FFT and call it TenseMusic. Stage has inputs/real_01.wav and real_02.wav, each20sec, plus source clocks. Need actual output feature/tension curve from official routines, source/version/weights hashes and command. If full faithful path objectively exceeds budget, RUN at least a genuine supported official subcomponent and state exactly what is absent; archive complete environment recipe. No installing globalTorch/TF, no deleting caches. Inspect available Python packages before installing. No fitting/qualityclaim. Provide root JSON adapter contract with time seconds, six feature availability, tension available/unknown and methods.''',
 'vocal_completion': '''Own src/musicmix/vocal_completion.py, tests/test_vocal_completion.py, tools/run_vocal_completion_pilot.py, runtime/, outputs/, research/. Need useful vocal phrase/prosody feature extraction rather than another wordgapwrapper. Research primary rap phrase/prosodic boundary papers and existing small pretrained English boundary/prosody/pitch models with official weights. Budget120MB downloads; no Torch/TFgigabytes. Staged inputs/vocal_01.wav and vocal_02.wav are20sec separated vocals; inputs/source_clocks.json binds them to canonical source clock. Implement phrase event descriptors around candidate cue with voice onset/offset, syllabic/attack timing, pitch-contour availability, terminallengthening/breath/ASRsyntactic evidence as separate channels, explicit uncertainty. Reuse existing pretrained component if feasible; a documented established DSPalgorithm can provide measured descriptors, never label it learned/certified closure. ASR segment!=prosodic phrase. Incoming pickup and outgoingcompletion asymmetry. Actually run on staged vocals and save evidence; tests boundaryclock/gainrobustness/nonfinite/missing. Provide canonical-sample API root can run on full vocal stems and measured evidence for phrase-end ranking; no arbitrary threshold fitting or confident semantic closure. Save useful primary sources and per-section summary.''',
 'functional_event': '''Own src/musicmix/functional_event.py, tests/test_functional_event.py, tools/run_functional_event_pilot.py, outputs/, research/. Fix substantive weak continuation logic: existingexpectation accepts any2matchingdomains even when salientothersreverse. Research event segmentation, musical expectancy, phrase continuation vs arrival in B using primary papers. Implement event-centric comparator consuming A_before,A_after,B_before,B_after profiles, vocalcompletion evidence and functional section context. Distinguish pre-entry preparation/pickup, post-entry attack event, persistence/stabilization and sustainedmotion. LocalBbefore->Bafter event evidence must actually affect result (previous APIonlycountedBbeforecoverage). Preserve domain disagreement and abstention; no globalscalarfalse certainty, no chorus=resolution, no nativeAidentityreward. Model-free engineering comparator may select supported feature evidence but label transferunvalidated. Use existing source code and public research; no heavy installs, download20MB. Runnable synthetic perturbation controls + actual localstaged audio profilepilot (reusePooledTrajectory with observed beats only whenavailable, do not inventbeatgrid) to show sensitivity to realBentryevent rather than just matchingtwo slopes. New scorer contract must be usable by root to rerank actual pool. Save theory vs measurement explicitly.''',
 'external_benchmark': '''Own tools/build_external_transition_benchmark.py, outputs/, research/. Need an external, REAL, reproducible validation dataset for WHENmix, not just counts. Official M-DJCUE https://github.com/MarcoStringa/M-DJCUE (discover correctofficialrepo from papers ifdifferent), CUE-DETR EDM-CUE, Raveform and UnmixDB are leads. Download primary annotations <=40MB, parse exactsource/schema/license/revision. Implement benchmark reader, recording/artist/version grouped split and evaluator for cue prediction timing tolerance, IN/OUT roles, annotatoragreement, regioncoverage, abstention, baselines (position-only/downbeat only when available). Do NOT infer everyunannotatedpointnegative. Actually RUN meaningful baselineeval on completeannotations with deterministicgroupedtestsplit, report numbers plusdomainlimitations; datasetmetadata can supportpositionbaseline, not pretendouraudioalgorithmwasvalidated. Include transitionpair/professionalDJ evidence ifpublic and explicitly aligned. Aim root-ready benchmark command consuming prediction JSON. Save sources/originals/texts/summary. No copyrightedaudio downloads in this worker, nofakefixturesasrealdata, no fitting on test split.''',
 'critic_research': '''Own tools/critic_adoption_probe_v12.py, outputs/, research/. Deep targeted research for EXISTING pretrained automatic music transition evaluation and temporal expectancy models: MusicFlamingo/audioLLM, music structural/prosody representations, learned transition compatibility, selfsupervised boundarycritics, DJ transitiondatasets. Need PRIMARY papers+officialcode+weights+licenses+CPUfeasibility. SongEval whole-song coherence alreadyfailed as reliabletransitiongate: avoidsimplyreuseunvalidatedoverallqualityasreward. Budget25MB metadata/papers only, no weights>5MB. Produce scored-adoptiontable evidence (task/domain, audioinput, timestamp/localtask, actual weights, feasible runtime, validationsignal) and one RUNNING smallprobe using availablepackages orofficialfeaturecode ifpossible. Need explicit experimentaltest to demonstrate incremental improvement using heldout externalannotations/paired examples withoutnative-songidentityshortcut. Focus on finding concrete reusable technology and selectingonefeasiblepath, not archive50unreadpapers. Save 4-6 actuallyreadprimarysources, text and Italiansection summary.''',
 'role_planning': '''Own src/musicmix/role_opportunities.py, tests/test_role_opportunities.py, tools/run_role_planning_pilot.py, outputs/, research/. Need improveWHEN/WHERE candidate generation: existingprovidedcueadapter ignoresincoming/outgoingroles, ASRdecoder endpointsare notphrases and samefewregionsrepeat. Implement role-aware opportunity extraction/selection that consumes realbeatgrid/downbeats, CUE/LinkSegopportunities, ASRword/segment intervals and optionalvocalcompletion events. Preserveprovider timestamps/exactcanonicalclock; no fakebeat or >80ms shifts. Outgoingcompletion and incomingpickup vs actualarrival must be distinct, with lookaheadarrival delay explicit. Generate candidate eventplans includingwait/prepare/change and entrywindow, protectwords, diversifytracks/regions withoutinventingquality. Root needs usable API on existing6catalog tracks. Use sourcetests realAPIcontracts; rootwillprovideactualdataafterintegration, use syntheticcontracttests only labeledsynthetic. Research primaryDJphrasealignment/pickupmethods and eventplanning, download15MB. Prioritizecode that canproduce new betterfoundedopportunities ratherthangenericplan.'''
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def excerpt(source, destination, start_seconds=70, seconds=20):
    with wave.open(str(source),'rb') as inp:
        params=inp.getparams(); start=round(start_seconds*params.framerate)
        inp.setpos(start); data=inp.readframes(round(seconds*params.framerate))
    with wave.open(str(destination),'wb') as out:
        out.setparams(params);out.writeframes(data)
    return {'file':destination.name,'source_sha256':sha(source),'source_start_sample':start,
            'sample_rate':params.framerate,'duration_seconds':seconds,'sha256':sha(destination)}

def main():
    if RUN.exists(): raise FileExistsError('Preserve prior swarm output')
    try:
        with urllib.request.urlopen('http://127.0.0.1:4096/global/health',timeout=2):
            raise RuntimeError('Port4096 already owned by another service; do not replace it')
    except urllib.error.URLError: pass
    RUN.mkdir(parents=True)
    jobs=[]
    for name,task in JOBS.items():
        folder=RUN/'agents'/name;folder.mkdir(parents=True)
        shutil.copytree(ROOT/'src',folder/'src',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(ROOT/'pyproject.toml',folder/'pyproject.toml')
        (folder/'tests').mkdir();(folder/'tools').mkdir();(folder/'inputs').mkdir()
        knowledge=folder/'knowledge';knowledge.mkdir()
        for path in (ROOT/'conoscenza_mix').rglob('RIASSUNTO.md'):
            dest=knowledge/path.parent.name/'RIASSUNTO.md';dest.parent.mkdir(parents=True);shutil.copy2(path,dest)
        (folder/'TASK.md').write_text(COMMON+'\nTASK:\n'+task,encoding='utf8')
        jobs.append({'name':name,'directory':str(folder.resolve()),'status':'pending'})
    # Only two task-relevant 20s audio windows, not entire music/user folders.
    mixed=[ROOT/'runs/reminder_to_too_many_nights/audio/a.wav',ROOT/'runs/pool_v8/audio/creature.wav']
    vocal=[ROOT/'runs/vocals_v2/a_vocals.wav',ROOT/'runs/pool_v8/vocals/creature/a_vocals.wav']
    for name in ('official_tension','vocal_completion','functional_event'):
        inputs=RUN/'agents'/name/'inputs';clocks=[]
        for index,source in enumerate(mixed,1): clocks.append(excerpt(source,inputs/f'real_{index:02}.wav'))
        if name=='vocal_completion':
            for index,source in enumerate(vocal,1):clocks.append(excerpt(source,inputs/f'vocal_{index:02}.wav'))
        (inputs/'source_clocks.json').write_text(json.dumps(clocks,indent=2),encoding='utf8')
    server_dir=RUN/'server';server_dir.mkdir()
    config={'permission':{'edit':'allow','bash':'allow','webfetch':'allow','websearch':'allow','external_directory':'deny','question':'deny'},'provider':{'opencode':{'options':{'apiKey':'public'}}}}
    env=dict(os.environ,OPENCODE_CONFIG_CONTENT=json.dumps(config),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    threads={k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
    process=spawn_server(EXE,4096,server_dir,server_dir/'server.log',threads)
    (server_dir/'pid.txt').write_text(str(process.pid))
    manifest={'schema_version':'musicmix.result-swarm.v12','server_executable':str(EXE),'jobs':jobs,
              'model_requested':'opencode/deepseek-v4.1-flash-free','privacy':'Selected source code, public-source summaries and task-relevant20s audio only; no human feedback/private logs/APIconfig staged.',
              'playback_seconds':20,'goal':'Demonstrable musical selection improvement, real feature/model inference and external validation; do not stop at software tests.'}
    (RUN/'swarm.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    for _ in range(40):
        try:
            with urllib.request.urlopen('http://127.0.0.1:4096/global/health',timeout=2) as response: health=json.load(response)
            if health.get('healthy'):
                print(json.dumps({'status':'ready','jobs':len(jobs),'server_pid':process.pid,'health':health}));return
        except Exception:time.sleep(.5)
    raise RuntimeError('Ownedserverhealthfailed; inspectlocalserver.log')

if __name__=='__main__':main()
