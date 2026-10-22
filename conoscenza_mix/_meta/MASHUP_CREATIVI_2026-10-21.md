# Mashup creativi: note dei due brani, voce in loop, botta e risposta (21/10/2026)

Richiesta dell'utente: «giocare di più combinando le note delle due canzoni, o modificando la voce mettendola in loop o
altro (temperatura alta), tutti i metodi possibili». Contesto: nel set v3 la policy ha scelto quasi solo echo/filter/build
perché le coppie erano incompatibili di tonalità; i voti ciechi indicano come difetto principale l'ingresso di B nel punto
sbagliato o su una parola tagliata.

Codice: `src/musicmix/dj_mashup.py` (nuovo, `musicmix.dj_mashup.1`), `src/musicmix/_vendor/tivlib/` (TIV.lib vendorizzato,
MIT), test `tests/test_dj_mashup.py`, driver `tools/run_dj_mashup_v1.py`, run `runs/dj_mashup_v1/`. Ritocco di
`tools/build_listening_artifact.py`: voce `role: "riferimento"` ascoltabile ma non votabile. Non modificati: `dj_creative`,
`dj_render`, `dj_effects`, `dj_qa`, `dj_phrase`, `dj_events`.

Etichette: **[user]** indicazione dell'utente (voti ciechi); **[E]** risultato sperimentale; **[S]** sistema pubblicato valutato nel suo insieme; **[P]** pratica DJ
(manuali, scuole, documentazione); **[T]** teoria/analisi (anche i libri, con pagina del PDF privato); **[H]** ipotesi o
soglia nostra. **Nulla è stato ascoltato**: QA e critico misurano l'integrità tecnica, non la qualità musicale.

### Criteri [user] (voti ciechi del torneo `giudizi_cloud/b_entry_fix_v1/`, riferiti dall'orchestratore il 21/10/2026)

1. A "carica" la tensione e B "risolve" (es. 5% TINT → Creature: "B risolve benissimo con un 'ouuuu' all'inizio e poi
   parte la rappata hard"): A esce in un punto di carica (dopo un ritornello/bridge/strumentale o alla fine del
   pre-ritornello, non a inizio brano), B entra su un arrivo forte (ritornello/hook).
2. **Base continua** tra A e B ("la base si interrompe al taglio", "ci stava se la base di A sotto fosse continuata").
3. **Niente tonalità discordante**, neanche senza sovrapposizione (autotune).
4. **Testo affine** tra la fine di A e l'inizio di B (A finisce con "a-a-a-a" → B attacca con un suono affine;
   Butterfly Effect → Knife Talk: legame di suono tra 'k' e 'k').

Come entrano nel codice (pesi e soglie [H], non stimati dai voti; i voti restano diagnostica separata, AGENTS.md):
- scelta delle coppie (`run_dj_mashup_v1.user_terms`): al punteggio di mashability si sommano carica di A (+0,15:
  sezione prima del cue `chorus`/`bridge`/`inst` o sezione dopo il cue `chorus`, cue oltre il 35 % del brano), arrivo di
  B sul ritornello (+0,15), tonalità compatibile oggi (+0,12) e dopo la trasposizione (+0,08), legame di suono tra le
  ultime parole di A e le prime di B (+0,10 × punteggio del legame);
- trasposizione con **guardia di tonalità** (`MashupConfig.key_guard`, `key_compatible_shifts`): fra −3…+3 solo le
  trasposizioni che rendono le tonalità stimate stessa/adiacente/relativa, se ne esiste una; la TIV sceglie fra quelle;
- **legame di suono** (`sound_affinity`, `find_sound_links`): oltre a parola uguale, lemma, rima e Soundex di
  `dj_creative.word_match`, vocalizzo ("ooh"/"ah", 0,4) e allitterazione sulla prima consonante pronunciata (c/k/q
  uniti, kn→n, 0,35); `lyric_link` di `dj_creative` provato su tutti gli inizi frase di A vicini al cue;
- scelta delle varianti in galleria: prima le tecniche a base continua e con legame di suono (verso di A sulla base di
  B, beat swap, botta e risposta, lyric link), poi hook/loop/chop, per ultimo il freeze (la base si ferma).

---

## 1. Fonti (lette in questa sessione)

| Fonte | Cosa serve qui | Tipo | Letto |
|---|---|---|---|
| Davies, Hamel, Yoshii, Goto (2014), *AutoMashUpper*, IEEE/ACM TASLP 22(12):1726–1737, doi:10.1109/TASLP.2014.2347135 (PDF autore, sha256 `9908fbef…`) | mashability per battito: cromagramma sincronizzato ai battiti (mediana per battito), similarità coseno su tutte le trasposizioni, pitch shift con Rubber Band; peso maggiore all'armonia, poi ritmo e bilanciamento spettrale; bonus di tempo. Con voci sovrapposte la correlazione mashability–gradimento crolla (già in biblioteca) | [S]+[E] | §II-A e §III (testo; numeri dei pesi persi nell'estrazione del PDF) |
| Huang, Wang, Smith, Song, Wang (2021), AAAI 35(1):187–195 | finestra di pipeline ±3 semitoni per gli stem armonici, tempo 0,8–1,2; stretch globale sfasa i battiti | [S]+[E] | già in biblioteca (full, 20/10) |
| Bernardes, Cocharro, Caetano, Guedes, Davies (2016), *A multi-level tonal interval space…*, JNMR 45(4):281–294, doi:10.1080/09298215.2016.1182192 | spazio TIV (DFT del cromagramma pesata) e "small-scale compatibility" (dissonanza del misto × distanza) | [T]+[E] | tramite TIV.lib (formula nel codice e nel paper DAFx), articolo non aperto |
| Ramires, Bernardes, Davies, Serra (2020), *TIV.lib*, DAFx-20, arXiv:2008.11529 (sha256 `752862c9…`) | libreria ufficiale (Python, **MIT**, `github.com/aframires/TIVlib`, `TIVlib/tiv.py` sha256 `c2e5d3c6…`, versione 0.1dev, commit non recuperabile dalla macchina di analisi): pesi audio {3, 8, 11.5, 15, 14.5, 7.5}, `combine`, `dissonance`, `small_scale_compatibility`, `get_max_compatibility` sulle 12 trasposizioni | [S] | full (paper) + codice |
| Bernardes, Davies, Guedes (2018), *A hierarchical harmonic mixing method*, CMMR 2017 / Springer LNCS | mixing armonico gerarchico con TIV (citato da TIV.lib) | [S] | solo citazione |
| Gebhardt, Davies, Seeber (2016), *Psychoacoustic Approaches for Harmonic Music Mixing*, Applied Sciences 6(5):123, doi:10.3390/app6050123 (e DAFx 2015) | trasposizione ±6 semitoni scelta per **consonanza misurata** (rugosità, comunanza di altezza su parziali per sedicesimo) invece che per tonalità stimata; test d'ascolto: la rugosità da pochi parziali segue la consonanza percepita dai musicisti | [S]+[E] | abstract/risultati della ricerca (articolo non aperto) |
| Crossfader (scuola DJ), *12 high energy DJ transitions* (wearecrossfader.co.uk) | "Wordplay": due brani si scambiano le righe come botta e risposta (loop di 4 battiti sulla voce, stems); "Loop cutting": loop di 1 battito su una parola, dimezzato fino allo stutter; "Acapella pitch play"; "Pitch play with looping": loop della voce che si dimezza/raddoppia sul build | [P] | pagina (sintesi del fetch) |
| Brewster & Broughton, *How to DJ Right* (2003), copia privata, cap. 15 "How to avoid key clashes" (pdf p. 39) e "How to mix harmonically" (pdf p. 86) | con brani che stonano: «Cut them, don't blend them», «Mix them when one (or both) is just percussion», «Pitch-shift one of them»; mixare armonicamente: «Mix a melody or vocal from one track over the bassline of another», «Play an acappella of one track over a dub of another» | [P] | pagine lette |
| Butler, *Unlocking the Groove* (2006), pdf pp. 97–99 e 249 | 3+3+2 = distribuzione massimamente uniforme di 3 attacchi su 8 impulsi; allineare brani su multipli pure-duple (4, 8, 16 battute) | [T] | pagine lette |
| Williams, *Rhymin' and Stealin'* (2013), pdf p. 161 | voce sminuzzata fino a una sillaba e ripetuta a ritmo (scratch sulla voce di Biggie) come gesto di citazione | [T] | pagina letta |
| Katz, *Groove Music* (2012), pdf p. 97 | botta e risposta (MC/folla) dentro la routine di Grandmaster Flash | [T] | pagina letta |
| Rubber Band Library 3 (CLI `rubberband --help`) | `-3` motore R3, `-t` durata, `-p` semitoni, `-F` preserva l'inviluppo formantico nel pitch shift | documentazione | help del binario |

Ricerche senza esito: nessun codice ufficiale di AutoMashUpper (solo descrizione); nessuno studio sull'effetto percettivo
di loop/chop della voce o del botta e risposta tra due brani; nessuna misura del gradimento di uno swell inverso.

## 2. Cosa è stato riusato e cosa scritto

- **TIV.lib** (MIT) vendorizzato *senza modifiche* in `src/musicmix/_vendor/tivlib/` con `LICENSE` e `PROVENANCE.json`.
  `dj_mashup.tiv`/`tiv_compatibility` sono la copia vettoriale di `TIV.from_pcp` + `small_scale_compatibility`
  (la libreria crea un oggetto per frame: troppo lento su 300 coppie × 276 piani × 7 trasposizioni); un test confronta i
  valori con l'originale frame per frame e con `TIVCollection` (uguali entro 1e-9).
- **Rubber Band CLI R3** (già qualificato da `dj_render.stretch_backend_info`, registrato nei metadati) per stretch +
  trasposizione con formanti in una sola chiamata.
- **htdemucs** (pesi ufficiali `955717e8-8726e21a.th`, sha256 nei metadati) solo per lo stem di batteria di B nel beat
  swap, sull'estratto di B, con lo stesso percorso d'inferenza ufficiale di `vocal_boundaries.separate_vocals_demucs`
  (normalizzazione media/std, `apply_model` split, overlap 0,25, shifts 0); ricampionamento in memoria con torchaudio
  (non via ffmpeg: piccola differenza dichiarata). Stem in cache float16 cancellati a fine run.
- Da `dj_creative`: `_prepare`/`_finish` (stesso schema di metadati e clip), `trim_overlay` per gli effetti,
  `reverb_ir`, `synth_riser`, `spectral_freeze` (variante con glissando scritta qui), `lyric_link` e `vocal_swap`
  (con B trasposto della quantità della mashability) come varianti della galleria.
- Parole: ASR del catalogo; per `18k_no_kizzy` l'ASR rifatta in italiano (`runs/asr_language_v1/vocals/18k_no_kizzy/a.json`,
  stesso `track_id`), perché quella del catalogo congelato era inglese allucinato.

## 3. Tecniche (`dj_mashup`) e regole comuni

Regola di progetto [H]: **B suona com'è** (suo tempo, sua tonalità, entra pulito sul suo downbeat); si stira e si traspone
solo il materiale preso in prestito (frammenti della voce di A, o l'hook di B anticipato sopra A). Diverso da
`dj_creative.key_shift_b`, dove B resta trasposto.

Parole intere (risposta al difetto principale dei voti) [H]:
- A smette di cantare solo dopo l'ultima parola di un verso: pausa ASR ≥ 0,25 s dopo la parola, stem vocale quasi muto
  (≤ 15 % dei frame attivi) nei 100 ms dopo, 30 ms di margine (`a_exit_point`); altrimenti la tecnica non si fa.
- Un verso preso in prestito parte da un inizio verso (pausa ≥ 0,25 s prima) con 40 ms d'anticipo, pickup compreso, e
  finisce a un fine verso (`a_line_from`, `a_line_end_after`); nessuna parola di A attraversa il cue.
- La voce di B si apre solo all'inizio di una sua frase (pausa ≥ 0,25 s prima della prima parola), 40 ms prima, pickup
  compreso, anche prima del downbeat (`b_vocal_gate`); fino ad allora B è strumentale (mix − stem). Nella galleria le
  parole di B sono quelle raffinate da `dj_phrase` sullo stem (le stesse del controllo `b_entry` di `dj_qa.6`); il punto
  in cui la voce di B diventa udibile è dichiarato in `meta['b_entry']` (come `dj_render.add_b_lead_in`, che non si può
  usare qui perché accetta solo i tipi `B_LEAD_IN_KINDS`), così il QA verifica in modo indipendente (con la sua soglia di
  respiro) che nessuna parola sia tagliata e che la frase di B non sia già cominciata. Nella prima prova con le parole
  ASR il QA ha trovato 3 aperture a metà frase (`renders_run7`, sicko_mode → rich_flex: "can" dopo "21," a 0,06 s). Se la voce di B
  torna entro 2 battute dalla fine prevista, la transizione finisce lì (così il QA misura B intero dopo).
- Una sola voce alla volta: eco e code della voce di A si zittiscono dove si apre B (Davies 2014 [E]; Brungart 2001 [E]).

Mashability e trasposizione: cromagramma CQT (`librosa`, 22,05 kHz, `norm=None`) con mediana per battito della griglia
regolarizzata (AutoMashUpper [S]); compatibilità TIV per battito (Bernardes 2016) del materiale trasposto di s ∈ −3…+3
(Huang 2021 [S]) sopra ciò che lo ospita; si traspone solo se la compatibilità migliora di ≥ 0,01 su s = 0, a parità
la |s| minore (penalità 0,002/semitono) [H]. Con una voce rap parlata il cromagramma è piatto e le trasposizioni si
equivalgono: lo shift resta 0. La similarità coseno di AutoMashUpper è registrata accanto. Tempo: piegatura d'ottava
(1/2, 1, 2 battute per battuta) e stretch del frammento ≤ 12 % [H] (dj_render tiene 8 % per i mix interi).

| Tecnica | Cosa succede | Parametri (default) | Fonte |
|---|---|---|---|
| `acapella_mashup` | verso di A (stem) sopra lo strumentale di B per 2–8 battute, stirato e trasposto; la base di A si ferma sul downbeat d'ingresso di B; 2 ripetizioni d'eco dell'ultima parola a un battito di B; medi di B −3 dB sotto la voce | bars 4, shift auto, echo 2 | [P] B&B pdf p. 86; [S] AutoMashUpper, Huang 2021 |
| `beat_swap` | lo stesso verso sopra la **sola batteria** di B (Demucs) per 2 o 4 battute, poi basso e armonia di B sul downbeat; nessuna trasposizione (solo batteria sotto) | drum_bars 2/4 | [P] B&B pdf p. 39 |
| `hook_teaser` | prima frase del ritornello di B (allin1 `chorus`), stirata al tempo di A e trasposta in A, sopra lo strumentale di A nelle ultime 2 battute; B entra poi dal suo cue nella sua tonalità | teaser_bars 2, shift auto | [P] B&B pdf p. 86; [H] |
| `vocal_loop` | ultima parola del verso con cui A chiude, ripetuta sulla griglia di B dal suo downbeat: `bar`, `half`, `beat` o `halving` (2,2,1,1,½,½,¼×4 battiti), passa-alto 150→1200 Hz, guadagno 0→−6 dB, coda d'eco (½ battito) o riverbero | halving, 2 battute | [P] Crossfader "Loop cutting", loop roll |
| `vocal_chop` | ultime 3 parole tagliate parola per parola e suonate a turno sul 3+3+2 (ottavi) della griglia di B | 332, 2 battute | [T] Butler pdf pp. 97–99; Williams pdf p. 161 |
| `call_response` | spazi di 1–2 battute di B alternati: verso di A (stirato/trasposto) e frase dell'hook di B (al suo tempo), 2 giri; ogni frase conserva il suo posto nella battuta (pickup); versi < 1 s uniti al successivo | rounds 2, slot 2 o 1 | [P] Crossfader "Wordplay"; [T] Katz pdf p. 97 |
| `pivot_freeze` | freeze spettrale dell'ultimo battito armonico di A (HPSS) che entra mentre A sfuma, tiene 4 battiti e scivola in 4 battiti fino alla tonica di B (o al relativo se i modi differiscono), poi B pulito; freeze −12 dB sotto B | hold 4, glide 4 | [T] modulazione per accordo perno; [H] |
| `reverse_swell` | prima parola della prima frase di B: riverbero inverso (parola × IR rovesciata, RT60 2 s) che cresce per 4 battiti e finisce sull'attacco della parola vera; A con passa-alto | 4 battiti | [P] reverse reverb; [H] |

Metadati (compatibili con `dj_qa`, come `dj_creative`): `overlap`, `switch`, `clip` (10 s + transizione + 10 s),
`a_map`/`b_map` (anche i frammenti con `rate` = rapporto di stretch e la mappa sorgente → clip), `creative.technique`,
`params`, `keys` (tonalità, shift, tonalità dopo lo shift, relazione Camelot prima/dopo, tabella della mashability),
`regions` con `intended`, `a_exit`, `b_vocal_gate`, `a_overlay`/`b_overlay` con i parametri Rubber Band. La voce presa in
prestito è sommata al deck del brano da cui viene (come il loop di `loop_extend`) e alla sua "voce udibile", senza la
riduzione di `trim_overlay`: nella prima prova quella riduzione abbassava di 12 dB la voce di A sopra un ritornello forte
di B (misura in `runs/dj_mashup_v1/render.log`, prima esecuzione).

Test sintetici (`tests/test_dj_mashup.py`, 22 test): TIV identico a TIV.lib; trasposizione trovata (do maggiore su re
maggiore → +2; cromagramma piatto → 0); piegatura e ricampionamento dei battiti; cromagramma per battito; stretch +
pitch su un tono (440 → 493,9 Hz ± 2 Hz, lunghezza esatta); unità e versi; per ogni tecnica lunghezze e regola della
clip, mappe, nessun clipping (true peak ≤ −1 dBTP), tonalità dopo lo shift misurata su un tono (523 → 587 Hz con +2; hook
600 → 534,5 Hz con −2), ripetizioni del loop sulla griglia di B (attacco entro 0–20 ms del punto), chop sul 3+3+2, botta e
risposta alternata senza sovrapposizioni, batteria senza basso fino al downbeat dell'armonia, glissando del freeze (440 →
493,9 Hz), swell che finisce sulla parola di B, una voce alla volta, QA `dj_qa` su un render (durata, true peak, voci,
verso tagliato misurato = pass), voce di B nascosta al QA finché il gate è chiuso e `b_entry` misurato al gate = pass,
errori se A non chiude un verso, legami di suono (allitterazione, vocalizzo, rima), trasposizione su un frammento più corto
di 2 battiti con la guardia di tonalità; pagina d'ascolto con il riferimento non votabile.

## 4. Galleria creativa v2 (`runs/dj_mashup_v1/`)

Pool: catalogo congelato del ponte, 18 brani (filtro condiviso `musicmix.track_filter`: inglese/italiano, esclusioni
dell'utente; fuori `reminder` e `too_many_nights` senza stem, `antidote` tempo ambiguo; pool CC assente). 306 coppie
ordinate × tutti i piani di `dj_select.plan_pair` valutati (mashability + criteri [user]); 10 coppie scelte, al più 2
apparizioni per brano: 5 compatibili di tonalità oggi (sicko_mode→rich_flex, after_hours→mafia, 5_percent_tint→creature,
stargazing→a_lot, superhero→mafia) e 5 oggi in conflitto ma rese compatibili dalla trasposizione del materiale preso in
prestito (nonstop→sicko_mode −3, nonstop→rich_flex −3, butterfly_effect→creature −2, a_lot→knife_talk +3,
heartless→jimmy_cooks −3). Tutte con A che esce in un punto di carica e B che entra sul ritornello (allin1).

Render (`runs/dj_mashup_v1/renders/<A>__<B>/`, una variante per tecnica, sul cue del piano o sul più vicino inizio
frase di A, fino a 10, che la rende possibile; se QA o veto bocciano, il cue successivo): ammessi (QA non fail, nessun
veto del critico) acapella_mashup 7/10, vocal_loop 10/10, vocal_chop 10/10, pivot_freeze 10/10, reverse_swell 7/10,
hook_teaser 5/10, beat_swap 5/10, lyric_link 9/10 (legami: parola uguale, allitterazione, vocalizzo), **call_response
0/10** (18 non realizzabili: il verso non sta nello spazio o nessun verso parte al cue; 2 bocciati per salto di loudness
del critico 4–8 dB), vocal_swap di `dj_creative` con B trasposto 0/10 (6 QA fail). Nessuna clip ha la voce di A troncata
o la voce di B aperta a metà frase secondo il QA (`verse_cut`, `b_entry`); i warn sono quasi tutti "ritmo non misurabile /
griglia di B ±25–130 ms dalla griglia di A allo switch" (B entra pulito al suo tempo: senza sovrapposizione ritmica la
misura non ha cosa appaiare), più qualche salto di loudness 2–3 dB e buchi di 0,5–1,5 s dove A resta solo strumentale.

Galleria cieca (`runs/dj_mashup_v1/gallery/review_manifest.json`, chiave in `blind_key.json` da non pubblicare; pagina
`build_listening_artifact.py --blind`, mp3 192 kbps, 25,6 MB, doc `dj_mashup_v1__<clip>`): 10 gruppi, **16 clip votabili,
8 tecniche × 2** (acapella_mashup, beat_swap, hook_teaser, lyric_link, pivot_freeze, reverse_swell, vocal_chop,
vocal_loop) + 10 riferimenti non votabili (la scelta del set v3 sulla stessa coppia: 5 clip del set, 5 decisioni del set
v3 sui candidati della coppia in ordine di costo; tecniche build verso il ritornello ×5, halving build ×2, echo out ×2,
filter out ×1). Durate 22–36 s, tutte conformi alla regola (10 s + transizione + 10 s). Priorità di scelta [user]: base
continua e legami di suono prima, freeze per ultimo; al più 2 per tecnica e 2 per coppia; QA pass prima di warn.

| Gruppo | Coppia (tonalità) | Varianti (shift del materiale preso in prestito; QA) |
|---|---|---|
| m01 | sicko_mode → rich_flex (D#m → F#M) | hook_teaser (0; warn) |
| m02 | after_hours → mafia (Cm → Cm) | lyric_link "better"→"back" (allitterazione; warn), vocal_chop (0; warn) |
| m03 | 5_percent_tint → creature (Bm → Bm) | beat_swap (0, solo batteria; pass) |
| m04 | nonstop → sicko_mode (F#m → D#m) | acapella_mashup (−3 → stessa tonalità; warn), lyric_link (parola uguale; warn) |
| m05 | nonstop → rich_flex (F#m → F#M) | pivot_freeze (glissando −3; warn) |
| m06 | stargazing → a_lot (Gm → Cm) | acapella_mashup (0, adiacente; warn) |
| m07 | butterfly_effect → creature (C#m → Bm) | hook_teaser (+2 → stessa; warn), beat_swap (0; warn) |
| m08 | a_lot → knife_talk (Cm → Bbm) | vocal_loop (−2 → stessa; warn), vocal_chop (−2 → stessa; warn) |
| m09 | superhero → mafia (Fm → Cm) | pivot_freeze (glissando −5; warn), reverse_swell (warn) |
| m10 | heartless → jimmy_cooks (D#m → Fm) | vocal_loop (+2 → stessa; warn), reverse_swell (warn) |

Correzioni fatte durante la run (registrate perché cambiano i numeri): (1) la voce presa in prestito passava per
`trim_overlay` e veniva abbassata di 12 dB sopra un ritornello forte di B → ora è sommata al deck del brano d'origine;
(2) la transizione finisce quando torna la voce di B (se entro 2 battute), altrimenti il QA misurava B solo strumentale
(salti di −3 dB); (3) `b_map` dichiara la voce di B solo dal gate e `meta['b_entry']` il punto in cui si apre; il gate usa
le parole raffinate di `dj_phrase` (con quelle ASR il QA trovava aperture a metà frase); (4) fine verso "morbido" per voci
melodiche (pausa ASR ma stem non muto: dissolvenza di 120 ms) [H], senza cui 5% TINT e After Hours non avevano varianti;
(5) cromagramma di frammenti più corti di 2 battiti (prima vuoto → shift 0 di ripiego), 3 `vocal_loop` rifatti
(i vecchi in `*_superseded.*`). Run intermedie conservate: `renders_run6/7/8`, `references_run6`, log `render*.log`.

## 5. Limiti e ignoto

- Nessuna clip è stata ascoltata. "QA pass/warn" e "nessun veto" sono integrità tecnica (durata, true peak, salto di
  loudness, voci sovrapposte, parola tagliata, ritmo misurabile), non qualità musicale.
- La mashability su TIV ordina i candidati; non è validata sul rap né sul nostro ascoltatore. Su voci parlate la
  trasposizione scelta può essere rumore del cromagramma: la soglia di guadagno (0,01) e lo stretch del 12 % sono [H].
  La tonalità stimata (Camelot) e la compatibilità TIV del materiale reale possono non concordare: sono registrate
  entrambe.
- Pitch shift con formanti e stretch: artefatti possibili sulla voce (±3 semitoni sono il massimo della pipeline di
  Huang 2021, non una soglia percettiva). Separazione Demucs: artefatti nello strumentale (mix − stem) e nella batteria.
- Fine e inizio dei versi dalle pause ASR (0,25 s [H]) e dallo stem, non dal modello di versi di `dj_phrase`; l'ASR
  faster-whisper ha confidenze non calibrate.
- Il riferimento del set v3 per le coppie che non sono nel set è la prima decisione del set v3 ammessa da QA e veto sui
  candidati della coppia in ordine di costo (senza filtro dei generi lontani e senza Viterbi del set): un'approssimazione
  dichiarata.
- Le tecniche non sono ancora giuntabili nel set (`dj_set.clip_geometry` con frammenti stirati non è stato verificato).
- `beat_swap` non traspone (sotto c'è solo batteria), ma quando entra l'armonia di B la tonalità cambia: con coppie in
  conflitto (m07) resta il cambio di tonalità che l'utente segnala come difetto [user]. `pivot_freeze` ferma la base (il
  contrario della "base continua" [user]) e i glissandi di 5 semitoni (m09) sono lunghi.
- `call_response` non ha prodotto clip ammesse: sui rap densi i versi non stanno in 1–2 battute o non partono al cue.
- I criteri [user] sono tradotti in pesi [H] di ordinamento; i voti da cui vengono restano diagnostica (AGENTS.md), non
  validano queste scelte.
- `dj_render.add_b_lead_in` non è applicabile ai tipi di `dj_mashup` (accetta solo `B_LEAD_IN_KINDS`, modulo non
  modificabile qui): la stessa regola (voce di B dall'inizio di frase, pickup compreso) è nel gate e dichiarata in
  `meta['b_entry']` per il controllo `b_entry` di `dj_qa.6`.
