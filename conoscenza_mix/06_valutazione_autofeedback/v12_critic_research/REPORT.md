# REPORT — critic/adoption probe v12

Data: 2026-10-20. Cartella: `runs/result_swarm_v12/agents/critic_research`.
Esito: **una via concreta selezionata** (SSM-Net-style local boundary novelty come *evidenza di candidato*, non come gate) + **una probe CPU eseguita**. Nessun peso preaddestrato è stato scaricato o eseguito. Nessuna affermazione 10/10, SOTA o di qualità musicale reale.

## 1. Vincoli rispettati

- Lavoro solo in questa cartella; nessun accesso ad altre cartelle/segreti.
- Dopo la correzione dell'orchestratore: **niente Torch/TF, niente nuovi pesi, niente nuove dipendenze**. La probe usa solo `numpy`, `scipy`, `librosa`, `soundfile` già presenti in `runtime/`.
- Nota di trasparenza: prima della correzione erano stati messi in `runtime/` `librosa==0.10.1`, `scikit-learn==1.3.1`, `pyyaml`; dopo la correzione non è stato installato né scaricato altro. I PDF/codici testuali sono gli unici download di ricerca (circa 6,2 MB totali, sotto il budget di 25 MB).
- Clip di ascolto: 20,000 s esatti = 10 s A + 10 s B (`outputs/examples/`).
- I giudizi utente non sono stati usati (non forniti). SongEval/qualità dell'intera canzone non sono usati come reward.

## 2. Comando eseguito

```powershell
$env:PYTHONPATH="$PWD\runtime;$PWD\src"
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:NUMBA_NUM_THREADS='1'
python tools\critic_adoption_probe_v12.py --pairs-per-seed 16 --seeds 0 1 2 3 4
```

Output osservato (stdout):

```json
{
  "baseline_auc": 0.34765625,
  "novelty_auc": 1.0,
  "combined_auc": 1.0,
  "delta_novelty": 0.65234375,
  "delta_combined": 0.65234375,
  "ci95": [0.4765625, 0.8076416015624999],
  "test_pairs": 32
}
```

Tempo misurato: **118,1 s** per 80 coppie (160 clip da 20 s) su 1 thread, inclusa sintesi e feature extraction. Non è il tempo del modello SSM-Net: è il tempo del front-end + novelty checkerboard senza pesi.

File prodotto: `outputs/PROBE_RESULTS.json` (64.953 byte, SHA256 `825ecbb12872b2fe27fe4bb5c36163fcbca4a16227159a862144ac9be3e275ab`).

## 3. Cosa fa la probe (e cosa non dimostra)

Coppie sintetiche paired, split per **seed del generatore**: 48 coppie train (seed 0-2), 32 coppie held-out (seed 3-4). Ogni clip è 20 s: A 10 s + B 10 s, stesso tempo (120 BPM), stessa tonalità/progressione, stessa fase di beat, RMS pareggiato con jitter casuale. Il positivo ha un **evento di arrivo esplicito sul seam** (crash + layer di lead); il negativo è lo stesso audio base senza quell'arrivo.

Feature:
- baseline: `-|RMS dB diff|` su finestra di 0,5 s attorno al seam;
- novelty: front-end pubblicato di SSM-Net (log-mel 80 bande, 92 ms / 23 ms, aggregazione 0,1 s, patch 4 s, hop 0,5 s) + SSM coseno + kernel checkerboard locale;
- onset: massimo di onset-strength entro ±100 ms dal seam (diagnostica).

Risultati held-out:
- baseline RMS: AUC **0,348**, paired accuracy **0,375** (il transient di arrivo alza l'RMS post-seam, quindi la baseline "premia" il negativo);
- novelty SSM-Net-spec: AUC **1,000**, paired accuracy **1,000**;
- delta AUC **0,652**, CI bootstrap 95% **[0,477, 0,808]** su 32 coppie;
- onset da solo: AUC 1,000; il contrasto sintetico è **saturo** e troppo facile.

**Limite esplicito.** Questo è un test di *feature plumbing* e di fallimento della proxy RMS, non una validazione musicale. Non dimostra che una transizione reale sia buona, che il rap/trap sia coperto, né che novelty alta = risoluzione. La validazione esterna con annotazioni umane e audio reale resta assente.

## 4. Tabella di adozione (sintesi)

Punteggio 0-100 con rubric esplicita in `outputs/ADOPTION.json` (6 criteri, max 14, penalità di integrazione). Non è una misura di qualità musicale.

| # | Tecnologia | Ruolo | Pesi / licenza | CPU | Punteggio | Decisione |
|---:|---|---|---|---|---:|---|
| 1 | **SSM-Net** (Peeters, ISMIR 2023) | boundary/novelty locale | 1,48 MB `nbatt1`; paper CC BY 4.0, code LGPL-3.0 | sì, ~500K param | **71** | **SELEZIONATA** come evidenza, non gate |
| 2 | **Beat This!** (Foscarin et al., ISMIR 2024) | beat/downbeat | small 8,1 MB; MIT | sì, ~20M param | 64 | companion timestamp, pesi oltre budget |
| 3 | **Medley2K** (Faber et al., 2020) | transizioni esterne | dataset MIDI; licenze da verificare | sì (MIDI) | 64 | protocollo held-out, non modello |
| 4 | **All-In-One** (Kim & Nam, WASPAA 2023) | beat/downbeat + label funzionali | 1,40 MB fold0; MIT | condizionata (NATTEN + demucs) | 57 | seconda scelta |
| 5 | **M-DJCUE** (Zehren et al.) | cue IN/OUT EDM | nessun peso; audio non distribuito; repo GPL-3.0 | sì (solo metadata) | 57 | prior di protocollo, non benchmark audio |
| 6 | **MusicFlamingo** (Ghosh et al., 2025) | audio-LLM whole-song | ~8B; noncommercial; code MIT | no | 29 | respinta come gate |

**Percorso scelto:** usare la novelty di confine SSM-Net come **una feature locale verificabile** del candidato A→B, insieme a beat/downbeat e ai guardrail voce/frase già nel progetto. Non collassare in un unico score. Calibrare su coppie held-out esterne, split per traccia/artista, con `UNKNOWN` quando manca evidenza.

**SongEval / whole-song quality:** esclusa come gate. La conoscenza fornita riporta che è già fallita come gate di transizione; la probe conferma solo che una proxy di loudness può essere ingannata da un evento di arrivo. Non è una rivalutazione di SongEval.

## 5. Fonti primarie lette (5)

Dettagli completi, sezioni lette, licenze e hash in `research/sources.json`; sintesi italiana in `research/SINTESI_IT.md`.

1. **Music Flamingo: Scaling Music Understanding in Audio Language Models** — Ghosh et al., 2025. arXiv:2511.10289, DOI 10.48550/arXiv.2511.10289. HTML: https://arxiv.org/html/2511.10289. Model card: https://huggingface.co/nvidia/music-flamingo-hf. Letto: abstract, introduzione/contributi, descrizione modello/dati, appendici A/B/D, model card. 8B, 20 min max, 30 s windows, NVIDIA OneWay Noncommercial.
2. **Beat This! Accurate Beat Tracking Without DBN Postprocessing** — Foscarin, Schlüter, Widmer, ISMIR 2024. arXiv:2407.21658, DOI 10.48550/arXiv.2407.21658. Code: https://github.com/CPJKU/beat_this. Letto: abstract, introduzione, §3.1-3.3, §4, conclusione. Modello small 8,1 MB, main 78 MB, MIT.
3. **Self-Similarity-Based and Novelty-based loss for music structure analysis** — Peeters, ISMIR 2023. arXiv:2309.02243, DOI 10.48550/arXiv.2309.02243. PDF: https://archives.ismir.net/ismir2023/paper/000089.pdf. Code: https://github.com/geoffroypeeters/ssmnet_ISMIR2023. Letto: abstract, §1, §2.1-2.3, §3.1-3.6, §4. 500K param; HR3F 0,713 RWC-Pop; 0,631/0,624 SALAMI Pop upper; state_dict 1.483.802 / 2.289.503 byte.
4. **All-In-One Metrical and Functional Structure Analysis with Neighborhood Attentions on Demixed Audio** — Kim & Nam, WASPAA 2023. arXiv:2307.16425, DOI 10.48550/arXiv.2307.16425. Code: https://github.com/mir-aidj/all-in-one. Weights: https://huggingface.co/taejunkim/allinone. Letto: abstract, introduzione, README ufficiale. 300K param; fold0 HEAD 1.400.571 byte; NATTEN + demucs richiesti; MIT.
5. **Medley2K: A Dataset of Medley Transitions** — Faber et al., 2020. arXiv:2008.11159, DOI 10.48550/arXiv.2008.11159. Letto: abstract, §3 dataset, validazione estrazione, esperimento. 2.000 medley / 7.712 transizioni; precision 90,70%, recall 57,07%; MIDI, non audio.

Hash degli originali: in `research/sources.json` e nel manifest sotto. Tutti i PDF sono stati estratti con `pypdf`; l'HTML di MusicFlamingo con `BeautifulSoup`. "Estratto" non significa "letto integralmente": le sezioni usate sono elencate.

## 6. Artefatti e hash

| File | Byte | SHA256 |
|---|---:|---|
| `tools/critic_adoption_probe_v12.py` | 21.623 | `04f38fa149939c22a0269f1f4fa923f3748361bfaab728b438f310fdfde06ad8` |
| `outputs/ADOPTION.json` | 13.225 | `c1a2959bcfa026474d2f5076f5d6cb1fe782ae797256b96444a299508ea8b2d7` |
| `outputs/PROBE_RESULTS.json` | 64.953 | `825ecbb12872b2fe27fe4bb5c36163fcbca4a16227159a862144ac9be3e275ab` |
| `research/SINTESI_IT.md` | 6.935 | `b04a6fae29b533a712518387ec8f31adc9641d73e4893793fc15caf4d80db515` |
| `research/sources.json` | 9.822 | `1f39c2c53049865d1cbee64cab2dbaec6e14992eb6f36387091bb63aecd0c520` |
| `outputs/examples/positive_20s.wav` | 882.044 | `7c2245245fc118d4de9691dbd09c817ec7a9bd6fccee37072a86e88c6620b4b6` |
| `outputs/examples/negative_20s.wav` | 882.044 | `62f2b2115024c0a5d5d11eb64bb7168e2172dfb472840bcd5be68ff427458795` |

Gli hash di `outputs/ADOPTION.json` e degli esempi possono cambiare se si rigenerano i file; il manifest sopra è quello osservato in questa run.

## 7. Blocker rimanenti

1. **Nessuna annotazione umana A→B con audio legale** in questa cartella: la probe è sintetica. Serve un set held-out con cue/giudizi esterni, split per traccia/artista.
2. **Pesi SSM-Net non eseguiti** in questo turno (vietato scaricarli): la fattibilità CPU è stimata dalla dimensione/architettura, non misurata sul modello.
3. **Licenza pesi SSM-Net**: il codice è LGPL-3.0; non è stata osservata una licenza separata per i `state_dict`. Va chiarita prima dell'integrazione.
4. **Nessun benchmark rap/trap**: le fonti sono EDM, DnB, Harmonix/SALAMI o MIDI. Il trasferimento è un'ipotesi.
5. **Boundary ≠ resolution**: nessun candidato trovato produce "B realizza l'aspettativa di A"; SSM-Net/All-In-One/Beat This! sono al massimo evidenza locale.
6. **Il contrasto sintetico è saturo** (AUC 1,000): non usare questo numero come prestazione musicale o come prova di superiorità. Serve una probe successiva su audio reale con cue esterni.
