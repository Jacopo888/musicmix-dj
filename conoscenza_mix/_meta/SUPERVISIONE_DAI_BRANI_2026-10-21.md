# Supervisione dai brani stessi: dove un brano si articola da solo (21/10/2026)

Strada 4 di [AUTONOMIA_VALUTAZIONE](AUTONOMIA_VALUTAZIONE_2026-10-21.md). Requisito dell'utente: niente voti continui e
nessun mix da fornire. Usiamo come supervisione **le canzoni stesse, non mixate**: i punti in cui un brano cambia da solo
(inizio di sezione sul downbeat, inizio di verso dopo una pausa con il pickup intero, rientro della voce) sono positivi;
le loro versioni spostate nello stesso brano sono negativi controllati. Ne esce un giudizio sul **dove** entrare o uscire
che non richiede ascolti.

Legenda: **[E]** risultato empirico di una fonte; **[S]** sistema o modello pubblicato; **[T]** teoria; **[P]** pratica DJ;
**[H]** scelta nostra; **[M]** misura nostra su questi dati. I voti ciechi dell'utente sono usati **solo** come controllo
esterno (§5), mai per addestrare né per scegliere soglie.

Codice: `src/musicmix/self_supervision.py`, `tools/build_insong_dataset.py` (fasi `points`, `dsp`, `mert`, `cocola`),
`tools/insong_eval.py` (`train`, `pairs`, `cocola-eval`, `external`, `cues`), test `tests/test_self_supervision.py`.
Run: `runs/insong_v1/` (`tracks.json`, `points.json`, `pairs.json`, `results_*.json`, `model/`, `cue_scores/`, cache
compatte in `cache/`).

## 1. Fonti

| Fonte | Tipo, letto | Cosa sostiene | Uso qui |
|---|---|---|---|
| Brewster & Broughton, *How to DJ Right* (2003), cap. 10 «How to match phrases» (EPUB pdf pp. 26–27) | [P], full (libro privato) | «Your mixes have to obey the same musical rules as your records»; frasi di 4 battute che compongono la struttura del brano | premessa del metodo: le articolazioni interne ai brani sono il modello del punto giusto; il negativo «metà sezione» (battuta 3 di un gruppo di 4) |
| Huron, *Sweet Anticipation* (2006), pp. 86 e 172 | [T]+[E citati], full (libro privato) | i confini di gruppo si formano dove le probabilità di transizione sono basse, appresi per sola esposizione («segmentation is primarily statistical», p. 172) | giustifica l'idea che i confini siano apprendibili dalla statistica interna della musica; non valida il nostro modello |
| Chen, Smith, Yang (2020), *Neural Loop Combiner*, ISMIR, arXiv:2008.02011 (già in [AUTO_CRITICO](AUTO_CRITICO_2026-10-21.md) §2.4) | [E], PDF | negativi sintetici dentro il brano (shift, reverse, rearrange) per la compatibilità di loop; la metrica sui negativi sintetici può divergere dal giudizio d'ascolto | precedente metodologico e avvertimento principale (§6) |
| Ciranni, Mariani, Mancusi, Postolache, Fabbro, Rodolà (2025), *COCOLA: Coherence-Oriented Contrastive Learning of Musical Audio Representations*, ICASSP 2025, arXiv:2404.16969 (PDF v3 sha256 `a53da5bf…`, fuori dal repo) | [E]+[S], full | contrastivo su stem: due sotto-mix della stessa finestra di 5 s (16 kHz) sono positivi, finestre di altri brani negativi; EfficientNet-B0 su HPSS mel 64, embedding 512, similarità bilineare. Accuratezza K=2 «HPS All» 93,9 % su MUSDB18-HQ tenuto fuori (Tab. I), cala con K (54 % a K=64, Tab. II); MOS di coerenza (41 partecipanti, chunk spostati in tonalità/tempo) vs punteggio: r = 0,54 | valutato come feature di compatibilità tra stem e del «dove» (§4) |
| Codice e checkpoint COCOLA: github.com/gladia-research-group/cocola @ `b3ac9480cd1594d990a1d2dfabbab7cf15af8dfa` (licenza MIT); checkpoint `COCOLA_HP_v1` (Google Drive del README), 59,6 MB, sha256 `951b70bc787fcac489208a6b665d0d4e1375d6c0511a91eedad2814986bfb0aa` | [S] | checkpoint senza licenza separata; addestrato su MoisesDB (licenza di ricerca non commerciale), Slakh2100 e CocoChorales (CC BY 4.0) | uso di ricerca, pesi in `/opt/musicmix/models/cocola/` (fuori dal repo); `efficientnet_pytorch` 0.7.1 in `/opt/musicmix/pydeps/cocola`; modello ricostruito dal codice ufficiale e caricato con `torch.load(weights_only=True)` (`tools/build_insong_dataset.py:Cocola`); RAM < 1 GB |
| MERT-v1-95M (Li et al., ICLR 2024) `m-a-p/MERT-v1-95M` @ `12af15fe` | [S] | già usato dal critico v2/v3 (AUC giunte 0,68) | embedding per brano, layer 6 e 12, media su 0,16 s |
| allin1 (Kim & Nam 2023), `dj_form`, `dj_phrase` ([QUANDO_TAGLIARE](QUANDO_TAGLIARE_2026-10-21.md), [INGRESSO_B](INGRESSO_B_2026-10-21.md)) | [S]+[H] | sezioni, griglia, parole raffinate, versi, regola R-B1 (pausa >= 1 battito) | **definiscono le etichette**: il modello impara ciò che questi moduli chiamano confine, con i loro errori |


## 2. Dati e metodo

**Brani** (`runs/insong_v1/tracks.json`): catalogo del ponte `runs/pool_bridge_v1/catalog_ready.json` (che contiene i 7 brani
V12), filtro `musicmix.track_filter` (en/it) ed esclusioni `configs/excluded_tracks.json`; niente pool CC. Restano **18 brani
in 8 gruppi d'artista** (Travis Scott 6, Drake 5, The Weeknd 2; 18K, 21 Savage, 6ix9ine, Metro Boomin, Pop Smoke 1 ciascuno).
Esclusi: `antidote` (griglia a metà tempo: i negativi ±battito sarebbero sbagliati, come nel set v3), `reminder` e
`too_many_nights` (stem vocale assente sulla macchina di analisi, niente modello `dj_phrase`), `life_is_good` (arrivato dopo il congelamento,
niente `dj_phrase`). Etichette e griglie sono quelle congelate in `runs/dj_set_v3_bridge` (`dj_plan/tracks`, `phrase/`),
lette senza modificarle; percorsi registrati via `musicmix.paths.localize`.

**Positivi** (281) [H sulle soglie, che sono quelle già in uso]: `section` 147 (confine allin1/dj_form che coincide col downbeat
della griglia entro 0,05 battute; i confini a ¼ di battuta, frequenti in Goosebumps e After Hours, sono ambigui e restano
fuori, ma sono protetti dai negativi); `line` 117 (verso di `dj_phrase` preceduto da una pausa della voce >= 1 battito, la
stessa regola R-B1 di [INGRESSO_B](INGRESSO_B_2026-10-21.md); punto = 30 ms prima dell'attacco, pickup intero);
`reentry` 17 (come `line` dopo >= 2 battute senza parole).

**Negativi controllati nello stesso brano** (3.110; nessuno entro 150 ms da un confine plausibile, compresi i confini
esclusi e gli inizi di verso senza pausa): `offbeat` ±½ battito (527), `beat` ±1 battito (523), `halfbar` ±½ battuta
(511), `bar` ±1 battuta (472), `midword` centro di una parola vicina (273), `midline` downbeat dentro un verso (243),
`mid_section` downbeat a metà sezione (battuta 3 di un gruppo di 4, >= 2 battute dai confini; è il nostro proxy di
"sezione sbagliata" per un punto: la scelta della sezione di B è una decisione di set e non si impara da un brano solo)
(280), `random` (281).

**Giunte** (`pairs.json`, 6.684): uscita da X a t1 + ingresso in Y a t2. Positivo = due punti naturali; negativo = lo stesso
con l'ingresso (tutti i tipi) o l'uscita (3 tipi a caso) spostati. `within` = stesso brano con salto >= 8 battute (424
positive); `cross` = brani di artisti diversi con tempo entro ±4 % (anche metà/doppio), 144 positive.

**Feature** (cache compatte per brano in `runs/insong_v1/cache/`, 80 MB in tutto, niente audio salvato): MERT-v1-95M layer 6
e 12 mediati su 0,16 s (float16) → distanze coseno e differenze post−pre su 2 s e 0,48 s (PCA 24 stimata sul solo training);
DSP a 50 Hz (energia in 4 bande, flusso spettrale, energia dello stem vocale Demucs) → livelli e salti ai due lati, picco
d'attacco; **voce** (`vox`) separata dal resto (`dsp_mix`) perché la pausa della voce prima del verso è parte della
definizione dei positivi; **griglia** (fase di battito e battuta, seno/coseno: nessuna informazione da sezioni o parole).
Classificatore: logistica L2 bilanciata (C = 0,3). **Split leave-one-artist-out** (8 fold): un brano e il suo artista non
sono mai in train e test insieme (`check_disjoint` in ogni fold, testato).

## 3. Risultati: dove il brano si articola (punti, brani tenuti fuori)

AUC fuori fold (positivi contro negativi del tipo; IC95 bootstrap per brano), `runs/insong_v1/results_points.json`:

| variante | tutti | ±½ battito | ±1 battito | ±½ battuta | ±1 battuta | metà parola | metà verso | metà sezione | casuale |
|---|---|---|---|---|---|---|---|---|---|
| griglia sola | 0,61 | 0,67 | 0,79 | 0,77 | 0,48 | 0,69 | 0,25 | 0,25 | 0,73 |
| DSP senza voce | 0,76 | 0,77 | 0,77 | 0,78 | 0,73 | 0,81 | 0,71 | 0,68 | 0,82 |
| DSP | 0,81 | 0,81 | 0,82 | 0,83 | 0,78 | 0,84 | 0,77 | 0,75 | 0,85 |
| MERT | 0,81 | 0,75 | 0,86 | 0,86 | 0,80 | 0,76 | 0,78 | 0,81 | 0,86 |
| audio senza voce (MERT+DSP mix) | 0,82 | 0,78 | 0,85 | 0,86 | 0,79 | 0,80 | 0,78 | 0,77 | 0,88 |
| **audio** (MERT+DSP+voce) | **0,85** | 0,81 [0,76–0,87] | 0,86 [0,83–0,91] | 0,88 [0,86–0,91] | **0,83** [0,80–0,87] | 0,84 [0,80–0,89] | **0,83** [0,77–0,87] | **0,82** [0,78–0,86] | 0,90 |
| **full** (audio + griglia) | **0,87** | 0,86 [0,81–0,91] | 0,91 [0,88–0,95] | 0,91 [0,88–0,94] | 0,83 [0,78–0,86] | 0,89 [0,84–0,93] | 0,78 [0,72–0,83] | 0,76 [0,71–0,81] | 0,93 |

Vittorie appaiate (il positivo batte il proprio negativo spostato, stesso brano) con `full`: 0,90 / 0,94 / 0,91 / 0,86 /
0,91 / 0,79 / 0,78 / 0,96. Per tipo di positivo (contro tutti i negativi): sezioni 0,86, versi 0,89, rientri 0,76.

Lettura [M]:
- I negativi **difficili** (stesso brano, ±1 battuta, downbeat a metà verso o a metà sezione: tutti sul downbeat, quindi la
  griglia da sola non li separa, AUC 0,25–0,48) sono riconosciuti dall'audio con AUC 0,82–0,83: il segnale non è "è un
  downbeat", è "qui il brano cambia".
- La voce aiuta (0,82 → 0,85) ma non è tutto: senza lo stem vocale l'audio resta a 0,82 (la fuga d'etichetta voluta dai
  positivi di verso è quindi limitata).
- La griglia migliora i tipi fuori battito (±½ e ±1 battito, metà parola) e peggiora metà verso/metà sezione (premia ogni
  downbeat): per il "quale downbeat" conviene `audio`, per il "sul battito o no" `full`.
- Il punteggio è un **ordinamento dentro il brano**, non una probabilità calibrata (es. downbeat del ritornello di
  Goosebumps 0,23 contro 0,02 a ±½ battito).

## 4. Stesso brano o brani diversi? (giunte)

`runs/insong_v1/results_pairs.json`, OOF per artista (una giunta è di test nei fold degli artisti che tocca):

| modello | AUC within | AUC cross | ingresso ±1 battito (cross) | ingresso ±1 battuta (cross) | ingresso metà verso (cross) | W>C tra positivi |
|---|---|---|---|---|---|---|
| full, tutte le giunte | 0,68 | 0,69 | 0,80 | 0,64 | 0,51 | 0,46 |
| full, addestrato **solo dentro i brani** | 0,68 | **0,69** | 0,80 | 0,64 | 0,51 | 0,45 |
| audio senza voce | 0,62 | 0,61 | 0,59 | 0,62 | 0,59 | 0,47 |
| solo similarità MERT tra i due lati | 0,55 | 0,51 | 0,52 | 0,48 | 0,49 | 0,43 |

[M] Il controllo "stesso brano = bene" è **negativo**: tra giunte naturali, quelle dentro lo stesso brano non ricevono
punteggi più alti di quelle tra brani diversi (AUC 0,45–0,47, la sola similarità MERT 0,43), e il modello addestrato solo
con giunte interne ai brani generalizza alle giunte tra brani (0,69, come quello addestrato con tutte). Le giunte valgono
meno dei punti (0,68 contro 0,87) perché la giunta mescola il contesto prima di X con quello dopo Y e perde il contrasto
interno al brano: per il "dove" va usato il punteggio dei punti di ciascun brano.

## 4b. COCOLA: coerenza voce/strumentale e "dove"

Esperimento (`tools/build_insong_dataset.py cocola`, `runs/insong_v1/cocola_stems.json`, `results_cocola.json`): 20 downbeat
con voce per brano (360 ancore); finestra di 5 s dello strumentale (mix − stem vocale Demucs) e della **voce della stessa
finestra** (positivo, la coppia su cui COCOLA è addestrato) contro la stessa voce spostata di ±½ battito, ±1 battito, ±½
battuta, ±1 battuta, +2 battute (negativi); poi tra brani (20 coppie con tempo entro ±3 %, voce di Y allineata al downbeat
di X). Modi `both`, `harmonic`, `percussive` del checkpoint HP. Tempo: ~1 h di CPU condivisa, RAM < 1 GB.

| confronto | AUC (`both`) | vittorie appaiate |
|---|---|---|
| voce allineata vs ±½ battito / ±1 battito / ±½ battuta / ±1 battuta / +2 battute (stesso brano, n = 360) | 0,50 / 0,50 / 0,50 / 0,51 / 0,51 | 0,52 / 0,51 / 0,48 / 0,51 / 0,53 |
| idem, modi armonico e percussivo | 0,50–0,50 | 0,47–0,53 |
| idem tra brani (n = 20) | 0,49–0,51 | 0,43–0,55 |
| voce dello **stesso brano** vs voce di un altro brano, entrambe allineate | **0,69** | — |
| punteggio armonico vs distanza Camelot tra brani (Spearman, n = 20) | −0,06 | — |

[M] COCOLA riconosce che voce e base vengono **dallo stesso brano** (0,69: è il suo compito di addestramento, con
negativi presi da altri brani), ma è **cieco allo spostamento temporale** della voce sulla base, perfino di mezzo battito.
Il motivo è architetturale: le due finestre sono codificate separatamente (EfficientNet con pooling globale) e confrontate
con una forma bilineare, quindi l'allineamento relativo non entra nel punteggio. Non distingue nemmeno le tonalità lontane
tra brani (n piccolo). **Decisione: COCOLA non è usato** né per il "dove" né come feature di compatibilità; resta registrato
(licenza, revisione, hash) per un eventuale uso come rilevatore di "stesso brano", che è proprio il segnale da non premiare.



## 5. Tensione → risoluzione dentro i brani (aggiunta dell'orchestratore)

Motivo: nei tornei dei cue (`giudizi_cloud/b_entry_fix_v1/`) l'utente non ha accettato nessuna variante in 6 tornei su 11
(7 su 11 secondo l'orchestratore, contando anche quelli con riserve) e il criterio dominante nelle note è «A deve caricare, B
deve risolvere», poi la base continua, la tonalità, «A tagliata al minutaggio sbagliato». Compito aggiunto: dentro i brani,
la fine di ciò che precede l'arrivo di un ritornello è un esempio di **carica**, l'arrivo del ritornello un esempio di
**arrivo**.

- Positivi (43 per ruolo): downbeat d'inizio di un ritornello allin1 preceduto da una sezione diversa. **[H] Proxy**:
  ritornello = arrivo. AGENTS.md vieta di equiparare ritornello e risoluzione: qui è solo l'etichetta disponibile senza
  ascolti, ed è proprio ciò che il controllo esterno smentisce (sotto).
- `tension_at_exit(track, t)`: solo il contesto **prima** di t (8 battute): energia per banda per battuta e pendenze, flusso
  (densità d'attacchi), voce, strati di `dj_form.3` (pendenza, etichetta, dropout della cassa: gesto anacrusico di Butler
  p. 189), croma rispetto al profilo del brano, MERT (4 s). Negativi: 2–4 battute dopo l'inizio di una strofa (128), metà
  strofa lontano dal ritornello (22), intro (19), 2 battute dopo l'arrivo (44).
- `resolution_at_entry(track, t)`: solo il contesto **dopo** t (4 battute) più il contrasto con le 2 battute precedenti
  dello stesso brano. Negativi: ripresa a metà strofa (22), inizio strofa (77, diagnostico), intro (21), 2 battute dopo
  l'arrivo (44).

AUC leave-one-artist-out (`results_tension.json`; IC95 bootstrap per brano):

| modello | tutti | inizio strofa / ripresa | metà strofa | intro | dopo l'arrivo |
|---|---|---|---|---|---|
| carica, DSP | 0,72 | 0,75 [0,63–0,85] | 0,69 [0,58–0,80] | 0,87 | 0,62 [0,47–0,75] |
| carica, DSP+forma+croma | 0,69 | 0,69 | 0,66 | 0,90 | 0,63 |
| arrivo, DSP | 0,55 | 0,55 | **0,41** [0,29–0,54] | 0,70 | 0,55 |
| arrivo, DSP+forma+croma+contrasto+MERT | 0,58 | 0,55 | 0,53 | 0,70 | 0,59 |

[M] La carica prima di un ritornello è riconoscibile in modo moderato (0,69–0,75 contro strofa e intro); l'**arrivo** del
ritornello, guardando solo ciò che viene dopo, **non** si distingue da una ripresa a metà strofa (0,41–0,53): nel rap le
strofe sono piene quanto i ritornelli. Con 43 positivi gli intervalli sono larghi.



## 6. Controlli esterni sui voti dell'utente (mai usati per addestrare né per scegliere soglie)

Tutti i punteggi sono calcolati con il modello del fold che **non ha visto l'artista** del brano (A per l'uscita, B per
l'ingresso). Numerosità piccola: nessun risultato è conclusivo.

**Galleria cieca del set v3** (`giudizi_cloud/dj_set_v3_bridge`, 12 gruppi; 19 clip con voto in 10 gruppi; le due versioni
di un gruppo hanno gli stessi cue) — `results_external.json`:

| punteggio | Spearman per clip (n=19) | per gruppo (n=10), p permutazione | AUC clip senza / con note «B nel secondo sbagliato, parola tagliata in B» (13/10) |
|---|---|---|---|
| naturalezza dell'ingresso di B (`full`) | 0,54 | 0,61, p = 0,06 | 0,51 |
| naturalezza dell'uscita di A (`full`) | 0,62 | 0,61, p = 0,07 | 0,50 |
| uscita di A (`audio`) | 0,64 | 0,64, p = 0,05 | 0,52 |
| critico v3 (riferimento) | 0,01 | 0,06, p = 0,88 | 0,43 |

[M] I punteggi dei punti vanno nella direzione dei voti più del critico v3, ma **non** riconoscono le clip con le note
«B nel secondo sbagliato / parola tagliata» (AUC ≈ 0,5). Le note sono state codificate con un'espressione regolare [H]
(«stessa cosa di prima» non è contata).

**Torneo dei cue** (`giudizi_cloud/b_entry_fix_v1`, chiave `runs/b_entry_fix_v1/clips_key.json`; 11 tornei, 60 varianti
votate, 21 accettate) — `results_tournament.json`:

| punteggio | AUC accettate vs rifiutate (21/39) | IC95 bootstrap per torneo | tornei con almeno un'accettata vs nessuna (5/6) | rango della migliore (t05, t07, t10, t12) |
|---|---|---|---|---|
| carica all'uscita di A (DSP+forma) | **0,76** | [0,53–0,93] | 0,63 | 1, 5, 3, 2 |
| arrivo all'ingresso di B (DSP+forma) | 0,35 | [0,10–0,71] | 0,47 | 1, 2, 1, 2 |
| combinato (media geometrica) | 0,60 | — | 0,47 | 1, 4, 3, 2 |
| naturalezza dell'uscita di A (`full`) | 0,61 | [0,37–0,85] | **0,80** | 1, 1, 1, 6 |
| naturalezza dell'ingresso di B (`full`) | 0,35 | — | 0,23 | 4, 1, 1, 4 |

[M] La carica all'uscita di A ordina accettate e rifiutate nella direzione dell'utente (0,76), ma quasi tutto il segnale è
**tra tornei** (quale coppia/regione di A), non dentro un torneo: nei tre tornei con accettate e rifiutate l'AUC interna è
0,3 / 0,5 / 0,8, e dove l'utente scrive «manca la carica di A» (t12_ed1141, A a 188,9 s) il modello dà la stessa carica
(0,01) della variante «A carica bene» (t12_db517f, 201,3 s); in t06 la variante «A non alza la tensione» ha la carica più alta
del suo torneo. Il proxy d'arrivo è **contrario** all'utente (0,35): le entrate accettate (5% TINT→Creature, Butterfly
Effect→Knife Talk, A Lot→Nonstop) arrivano su strofe («un 'ouuuu' e poi parte la rappata hard»), non su ritornelli. Questo
conferma, con i dati, che «ritornello = risoluzione» è sbagliato per il rap dell'utente: la risoluzione di B va cercata
altrove (attacco vocale netto dopo una pausa, cambio di densità, timbro/tonalità compatibili con la fine di A, testi
affini) e non deve essere addestrata sui voti.

## 7. API e cache per il sistema

- `self_supervision.cue_naturalness(track, t, variant="full"|"audio")` → `{p, status, nearest_t, what, oof}`; `status` =
  `cached` (candidato entro 60 ms: ogni battito e mezzo battito della griglia e ogni inizio verso), `computed` (dalle cache
  MERT/DSP del brano col modello registrato), `unavailable` (p = None: il chiamante si astiene).
- `self_supervision.tension_at_exit(track, t)` e `resolution_at_entry(track, t)` (variant `dsp_form`, nessun modello
  preaddestrato al momento dell'uso).
- Cache per brano `runs/insong_v1/cue_scores/<slug>.json` (18 brani, 1,7 MB): per ogni candidato `p_full`, `p_audio`,
  `tension_exit`, `resolution_entry`, con i file di modello usati; per i brani di addestramento i punteggi vengono dal fold
  che **non** ha visto l'artista (`oof: true`). Brani nuovi: `build_insong_dataset.py dsp|chroma|mert` e `insong_eval cues`
  dopo aver aggiunto il brano a `tracks.json` (serve il modello `dj_phrase` del brano).
- Modelli: `runs/insong_v1/model/point_{full,audio}[__wo_<artista>].npz`, `tension_{exit,entry}_<variante>[__wo_…].npz`,
  `folds.json`. `dj_set`, `dj_select`, `dj_phrase`, `dj_qa`, `dj_render`, `track_cards` **non** sono stati modificati: l'uso
  nelle schede brano e nel pianificatore è lasciato ai rispettivi agenti.
- Riproduzione: `python tools/build_insong_dataset.py points|dsp|chroma` ([venv]), `mert|cocola` ([venv-np1]),
  `train|tension|pairs|cocola-eval|external|tournament|cues` ([venv]). Fasi riprendibili (salta i brani già in cache,
  scritture atomiche, COCOLA con checkpoint per brano), attesa di >= 6 GB di RAM per MERT, arresto sotto 2 GB di disco.

## 8. Limiti

- **Cosa impara**: dove i brani si articolano da soli (continuità interna), non le scelte di un DJ: quale brano, quale
  sezione di B, ordine, durata della sovrapposizione, tecnica. Il controllo esterno lo conferma: il criterio dominante
  dell'utente («A carica, B risolve», base continua, tonalità) è di coppia e non è catturato.
- Le etichette vengono da modelli (allin1, Whisper + `dj_phrase`, `dj_form`) con i loro errori; i confini a ¼ di battuta
  dalla griglia sono esclusi, non corretti. Metà sezione è un proxy di "sezione sbagliata".
- 18 brani, 8 artisti, quasi tutti rap/trap americani (un brano italiano): gli AUC valgono per questo dominio.
- I positivi di verso dipendono dalla pausa della voce: la variante senza stem vocale misura quanto pesa (0,85 → 0,82).
- I punteggi sono ordinamenti, non probabilità calibrate.
- Tensione → risoluzione: proxy «ritornello = arrivo» debole (0,55–0,58) e smentito dal torneo; la carica (0,69–0,75) ordina
  i tornei solo tra coppie, non dentro la coppia. Va trattata come diagnostica, non come criterio di scelta.
- Controlli esterni: un solo ascoltatore, 10 gruppi e 11 tornei; p ≈ 0,05–0,07 al massimo. Nessuna soglia è stata scelta
  sui voti.
- COCOLA: un solo checkpoint (HP_v1), voce da Demucs (residui di strumentale), finestre di 5 s; il risultato negativo vale per
  questo uso, non per la valutazione di accompagnamenti per cui è stato pubblicato.
