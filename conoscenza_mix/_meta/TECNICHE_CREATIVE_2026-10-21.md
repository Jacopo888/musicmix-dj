# Tecniche creative di transizione e mashup: catalogo, fonti, regole d'uso, implementazione

Data: 2026-10-21 (notte 20→21/10). Obiettivo 3 dell'utente: "rendere più fantasiosi mashup e transizioni (allungare uno strumento, associare una nota, associare le parole cantate) … il metodo giusto al momento giusto, con la durata giusta".
Codice: `src/musicmix/dj_creative.py` (render), `src/musicmix/dj_policy.py` (scelta), test `tests/test_dj_creative.py`, `tests/test_dj_policy.py`, driver `tools/run_dj_creative_v1.py`, run `runs/dj_creative_v1/`.
Non modificati: `dj_render.py`, `dj_effects.py`, `dj_select.py`, `dj_set.py`, `dj_qa.py`, `INDEX.md`, `CHANGELOG.md`.

Etichette (come nelle revisioni precedenti):
- **[E]** risultato empirico misurato (N dichiarato);
- **[S]** regola di un sistema pubblicato, valutata solo nel suo insieme;
- **[P]** pratica DJ dichiarata da manuali, documentazione ufficiale di software/mixer, didattica (Digital DJ Tips, scuole): **pratica, non evidenza**;
- **[T]** teoria, rassegna, storia;
- **[U]** giudizi dell'utente (galleria v1: filter out 9, echo out 9, blend 8 battute 8, echo out 7, drop secco sul ritornello 3; un solo ascoltatore, usati come priorità dichiarate e non come validazione);
- **[H]** ipotesi di progetto, da verificare con l'ascolto.

"Letto" indica cosa è stato letto davvero in questa sessione: **full**, **parziale**, **abstract/metadati**, **pagina pratica** (documentazione o articolo didattico letto nella pagina web), **solo risultato di ricerca** (titolo/estratto del motore, non aperto).

---

## 0. Sintesi

1. **Non esiste uno studio empirico sui tipi di transizione nei mix reali** (lacuna già confermata in `REVISIONE_SET_TRANSIZIONI_2026-10-20.md` §1.2): né per l'EDM né per l'hip-hop. Le tecniche "creative" (loop, acapella, vocal swap, word play, freeze, spinback, brake, build, riser) sono documentate solo come **pratica [P]** (manuali, documentazione di Serato/rekordbox/Traktor/Pioneer DJM/Mixxx, didattica) e, per i mashup, da **sistemi pubblicati [S]** con piccoli test d'ascolto (AutoMashUpper, Lee 2015, Huang 2021).
2. **L'unico vincolo forte e misurato per i mashup è la voce**: una sola voce alla volta (Davies 2014 [E]: con voci sovrapposte la correlazione compatibilità–gradimento crolla; Brungart 2001 [E] mascheramento informativo; MosaikBox [S]). Tutte le tecniche implementate qui mantengono "una voce alla volta" tramite gli stem Demucs (mute della voce uscente, eco della voce zittita dove B canta).
3. **Le tecniche si dividono in due famiglie operative** con requisiti diversi:
   - *con sovrapposizione beatmatchata* (loop_extend, acapella_over, vocal_swap): richiedono tempo entro ±8% (MosaikBox [S]; Ishizaki 2009 [E]) e tonalità compatibile o resa compatibile con ±1 semitono (Kim 2020 [E]: il 94,3% delle trasposizioni reali è di 1 semitono; Vande Veire 2018 [S]);
   - *senza sovrapposizione* (build_into_chorus, note_bridge, echo_freeze, reverb_wash, backspin, tape_stop, lyric_link): B entra pulito sul suo downbeat al suo tempo; funzionano anche con tempi lontani, come echo/filter out già votati 9/10 [U].
4. **Durate**: la pratica le esprime in battiti e battute sulla griglia — effetti di uscita 1–4 battiti (rekordbox Release FX: 1/2/4/8 battiti [P]), build 2–4 battute prima del drop [P], loop roll che dimezza la lunghezza fino a 1/16 [P], acapella/blend su frasi di 4–8 battute [P][E: Kim 2020, multipli di 8 battute nei blend reali]. Le durate scelte qui sono tutte in battute/battiti della griglia regolarizzata.
5. **La scelta "metodo giusto al momento giusto"** è resa esplicita in `dj_policy.POLICY_TABLE` (§4): legame di parole → lyric link; ritornello in ingresso → build (al posto del drop secco votato 3/10 [U]); fine verso di A + B che canta → acapella/vocal swap; outro di A troppo corto o voci in entrambi → loop strumentale di A; tempo lontano → freeze/pad/spinback/brake. Ogni proposta porta durata, motivazione e fonte; il `prior` ordina soltanto e **non è una stima di qualità**. La scelta finale spetta all'ascolto o al critico automatico (`dj_critic`, in scrittura da un altro agente).
6. **Generazione con modelli**: nessun modello pubblico genera transizioni DJ da due brani con qualità dimostrata superiore alle regole. DJtransGAN 2022 [E] non batte in modo significativo il crossfade/regole; Levy et al. 2023 (diffusione con guida) elenca le transizioni fra due brani tra i casi d'uso ma è un preprint senza valutazione nell'abstract; DJ-AI 2025 (MusicGen) ha evidenza debole. Per questo qui si usano DSP espliciti + stem separati, non generazione neurale.

---

## 1. Fonti consultate in questa sessione

### 1.1 Pratica DJ (fonti autorevoli di pratica, non evidenza)

| Fonte | Cosa dice (rilevante) | Letto |
|---|---|---|
| Serato, *Software Overview / Stems* (support.serato.com, art. 14187272465295; Serato DJ Pro 3.0/4.0) | Pad Stems: Vocal, Melody, Bass, Drums on/off; toggle **Acapella** e **Instrumental** dal deck; **Roll**: loop roll attivo finché il pad è premuto, al rilascio la traccia riprende; loop manuale In/Out/Reloop | solo risultato di ricerca (estratti della pagina ufficiale) |
| rekordbox (Pioneer DJ / AlphaTheta), manuale 6.x/7.x, *Release FX* | Release FX = **Echo, Vinyl Brake, Backspin** con durata d'azione **1/2/4/8 battiti**, legati al BPM; spegne gli effetti attivi e fa la transizione | solo risultato di ricerca (manuale JP e pagine formative); manuale EN non aperto |
| Pioneer DJ DJM-750MK2 (pagina prodotto e manuale) | Beat FX: Delay, Echo, Ping Pong, Spiral, Reverb, Trans, Flanger, Pitch, **Roll**, **Vinyl Brake**, Helix; Sound Color FX: Sweep, **Filter**, Crush, Dub Echo, Noise, **Space** | solo risultato di ricerca |
| Native Instruments, *Traktor Pro manual – effect reference*; blog NI | Delay e Delay T3 sincronizzati al tempo con pulsante **Freeze**; effetti a buffer (BeatMasher, Gater, Reverse Grain, Transpose Stretch, Beat Slicer); Delay T3 consigliato "alla fine di una frase per 4 o 8 battute" (blog NI) | solo risultato di ricerca (pagina del manuale non raggiungibile, 404) |
| Mixxx User Manual 2.4 (manual.mixxx.org), cap. *Effects* e *DJing with Mixxx* | Echo e Reverb continuano dopo che l'ingresso è tagliato dal fader (code); effetti con tempo sincronizzati al deck; sync con raddoppio/dimezzamento (140↔70); key lock; "There are no rules on which method should be used" per le transizioni | pagina pratica (letta) |
| Mixed In Key, *Change energy with the Camelot wheel* e guida *Advanced harmonic mixing* | mosse sicure: stesso codice, ±1, A↔B; **Energy Boost**: +2 sul Camelot (es. 8A→10A) = un tono sopra; in una guida più vecchia "uno o due semitoni sopra" | solo risultato di ricerca |
| Digital DJ Tips, *How To DJ With Acapellas*; *4 Easy Steps To Making Your First Mashup* | acapella su strumentale nella **stessa tonalità o compatibile**, BPM simile (**circa ±5 BPM**), ingresso su un punto strutturale chiaro (breakdown, drop), sync e conteggio in otto | solo risultato di ricerca |
| Digital DJ Tips, *How To Dropmix Like A Pro Hip Hop DJ* | **dropmix**: passaggio brusco ma a tempo sull'"uno" della battuta, cue a inizio frase (intro, verso, ritornello) | solo risultato di ricerca |
| Digital DJ Tips, *Transitions Toolbox* (corso) | elenca **Wordplay**, toneplay, mashup, mini-mix tra le routine; aggiornamento con stems, key sync, loop | solo risultato di ricerca (pagina commerciale) |
| Digital DJ Tips, *Three Ways To Use Loops*; Violet Recording, *How to Use Loops*; Engine DJ, *Understanding Roll Mode* | **loop roll classico**: loop di una battuta dimezzato ripetutamente (8→4→2→1 battiti …) fino al "buzz", rilascio sul downbeat; rischio: abuso, loop di voci note | solo risultato di ricerca |
| Digital DJ Tips, *Are You Stuck In The Tempo Trap?*; Serato blog *Using Sync To Change Tempo*; DJ.Studio *Double and half tempo* | **tempo ride**: spostare il tempo a piccoli passi (es. 130→125 condiviso); **half/double time** per 70↔140, "estremo"; DJ.Studio limita lo stretch in una transizione a ~25–30% (prodotto, non evidenza) | solo risultato di ricerca |
| London Sound Academy, *10 Hip-Hop DJ Transitions* | 10 ricette su BPM uguale, B in fase "sull'1": bass swap a metà intro; mid swap nelle prime 3 battute; filter; sweep; space; **intro loop**; **vinyl backspin sull'ultima battuta della frase**; **Beat FX echo 1, 1/2, 1/4** con BPM dell'effetto corretto; combo Crush+Filter, Filter+Space | pagina pratica (letta) |
| zZounds blog, *Improve your mix with a well-placed backspin* | backspin come transizione (cambio di genere o BPM), con fade o filtro mentre il piatto torna indietro | solo estratto (pagina 403) |
| Broughton & Brewster, *How to DJ Right* (Grove 2003; UK *How to DJ (Properly)*, Bantam 2002/2006) | copre scratching, hot-mixing, **beat-juggling**, frasi e struttura dei brani; Vande Veire 2018 lo cita per la pratica; Raveform 2026 per il vocabolario delle sezioni | metadati (cataloghi); indice non accessibile |
| Katz, *Groove Music* (OUP 2012) | storia: Kool Herc estende i break alternando due copie dello stesso disco ("merry-go-round" non confermato nelle fonti trovate); Flash e Theodore sviluppano tecniche per ripetere/estendere i break (backspin, scratch) | recensioni (Symposium, Muse) |
| Katz, *Capturing Sound* (UC Press 2004, ed. riv. 2010) | cap. 6 "The Turntable as Weapon: Understanding the Hip-Hop DJ Battle"; l'ed. 2010 aggiunge i **mashup** e Auto-Tune | indice (UC Press) |

Libri: nessuna edizione open access trovata per Broughton & Brewster, Katz, *DJ Culture in the Mix* (Bloomsbury 2013, a pagamento). Non è stato scaricato nulla di non legale. Il manuale Mixxx è libero (progetto open source) e consultabile online.

### 1.2 Ricerca (sistemi e studi)

| Fonte | Rilevanza | Letto |
|---|---|---|
| Davies, Hamel, Yoshii, Goto (2014). *AutoMashUpper*. IEEE/ACM TASLP 22(12):1726–1737. DOI 10.1109/TASLP.2014.2347135 | mashability su frasi (armonia con trasposizione, ritmo, spettro); 15 ascoltatori: medie 6,7/4,7/4,3; **voce sovrapposta = principale causa di insuccesso** [S]+[E] | full (sessione 20/10, `REVISIONE_SET` §4) |
| Lee, Lin, Yao, Lee, Wu (2015). ISMIR, pp. 399–405 | mashability verticale (lead su fondo) e orizzontale (raccordo tra segmenti) [S]+[E] | parziale (20/10) |
| Huang, Wang, Smith, Song, Wang (2021). AAAI 35(1):187–195 | compatibilità di stem; pipeline ±3 semitoni e tempo 0,8–1,2; **stretch globale sfasa i beat** → stretch a tratti [S]+[E] | full (20/10) |
| Chen, Smith, Yang (2020). *Neural Loop Combiner*. ISMIR | compatibilità di loop, 116 partecipanti [E] | parziale (20/10) |
| Sankaranarayanan et al. (2023). *Mixboard*. NIME 2023, pp. 500–505 | app di mashup per principianti: separazione in stem, tempo e tonalità comuni, segmenti scelti dall'utente; studio utenti qualitativo | abstract |
| Delabaere et al. (2025). *AutoMashup*. arXiv:2508.06516 (GRETSI'25) | separazione + compatibilità di stem con COCOLA; CLAP/MERT zero-shot non riproducono la compatibilità percepita; **compatibilità asimmetrica** (chi fa voce e chi accompagnamento) | abstract |
| Wenner, Bazin, Sorkine-Hornung, Kim, Gross (2013). *Scalable Music: Automatic Music Retargeting and Synthesis*. Computer Graphics Forum 32(2):345–354 (Eurographics). DOI 10.1111/cgf.12054 | allungare/accorciare un brano con salti globalmente ottimi (seam carving, video textures; segmentazione per clustering spettrale; cammino a lunghezza vincolata con programmazione dinamica) [S] | abstract/metadati |
| Lamere (2012). *The Infinite Jukebox* (musicmachinery.com) | grafo di battiti simili (timbro, pitch, loudness, durata, posizione nella battuta) con salti casuali tra battiti "gemelli" → estensione infinita [S, demo, nessuna valutazione] | pagina (estratti) |
| Kim, Yang, Nam (2021) NIME; (2022) DAFx | stima di **fader ed EQ** dai mix reali (5.040 mix MixesDB); **nessuna stima di echo/reverb/loop**: la "stima degli effetti DJ" oltre fader+EQ non è stata trovata (lacuna) [E] metodo | già in biblioteca (REVISIONE_DJ §2) |
| Chen, Hsu, Liao, Martínez Ramírez, Mitsufuji, Yang (2022). *DJtransGAN*. ICASSP 2022. arXiv:2110.06525 | GAN su EQ e fader differenziabili; ascolto: competitivo ma non significativamente migliore delle regole [E] | abstract (+ full nella revisione 20/10) |
| Levy, Di Giorgi, Weers, Katharopoulos, Nickson (2023). *Controllable Music Production with Diffusion Models and Guidance Gradients*. arXiv:2311.00613 | diffusione con guida a tempo di campionamento: continuazione, inpainting, rigenerazione e **transizioni lisce tra due brani**; preprint, l'abstract non riporta valutazione | abstract |
| DJ-AI (Audio Mostly 2025, Sabancı) | transizioni generate con MusicGen; evidenza debole (20 partecipanti) | full (20/10) |
| *MixWeaver* (sottomissione ISMIR 2026, nota solo da un modulo di revisione pubblico) | ricerca di traiettorie di guadagno per stem nella sovrapposizione, confronto con MosaikBox | solo modulo di revisione: **non verificabile**, non citare come risultato |
| Tesi Princeton (s.d.), *A Supervised Learning Framework for Generating DJ Transitions* | CNN che predice curve di volume/EQ da 1001Tracklists | metadati (letteratura grigia) |
| Solberg & Dibben (2019), Solberg & Jensenius (2016/17) | drop dopo build/breakdown: risposta fisiologica e movimento maggiori (EDM) [E] | full/abstract (20/10) |

Ricerche senza esito (lacune): nessun paper 2024–2026 sulla generazione di transizioni DJ con diffusione oltre Levy 2023; nessun articolo su "word play" o legami testuali tra brani nel DJing (solo pratica); PopMash (Xing et al. 2020, MTAP) citato da Huang 2021 ma non trovato in questa sessione; nessuna statistica sull'uso di backspin/brake/echo nei set reali.

---

## 2. Catalogo delle tecniche

Colonne: quando si usa (contesto), durata tipica, parametri, rischi, fonte, stato nel codice. Le durate in battute assumono 4/4.

| Tecnica | Quando | Durata tipica | Parametri principali | Rischi | Fonte | Stato |
|---|---|---|---|---|---|---|
| **Cut / dropmix sull'uno** | cambio netto a inizio frase, tempi anche diversi | istantaneo (micro-fade 5–20 ms) | cue su downbeat di inizio frase | parola troncata, salto di energia | [P] DDJT dropmix; [E] Zehren 2020 | `dj_render` (esistente) |
| **Bass swap / EQ blend** | tempi entro ±8%, intro/outro strumentali | 4–16 battute, swap a metà | bassi di una sola traccia per volta | doppio basso, voci sovrapposte | [S] Vande Veire 2018; [P] LSA 1–2; [E] Kim 2020/2022 | `dj_render` (esistente) |
| **Filter out / filter swap** | uscita di A senza beatmatch lungo, o scambio filtri (A passa-alto, B passa-basso che si apre) | 1–4 battute | sweep geometrico HPF 30 Hz→1,2 kHz | filtro troppo lento = "fango" | [P] LSA 3; [U] 9/10 | `dj_effects` (esistente; filter_in = filter swap) |
| **Echo out** | fine frase, tempi o tonalità lontani | 1/4–1 battito di delay, 1–2 battute | delay a tempo, feedback ~0,5, HPF nel loop | eco fuori tempo se il BPM dell'effetto è sbagliato (LSA 8) | [P] rekordbox, DJM, LSA 8; [U] 9 e 7 | `dj_effects` (esistente) |
| **Echo freeze** | come echo out ma con coda sostenuta (sospensione prima di B) | freeze 1 battuta + coda 1 battuta | spettro dell'ultimo battito congelato + eco a 1 battito (fb 0,6), HPF 150→1500 Hz, −12 dB sotto B | statico, "nebbia" se troppo lungo | [P] Traktor Delay Freeze, rekordbox Echo; [U] famiglia eco | **`dj_creative.echo_freeze`** |
| **Reverb wash** | chiusura morbida, B più calmo, fine set | wash 1–2 battute + coda 2 battute | RT60 ~3,5 s, mandata crescente, coda HPF 250 Hz −12 dB sotto B | impasto con B, perdita di attacco | [P] DJM Reverb/Space, Mixxx (code), LSA 5 | **`reverb_wash`** |
| **Backspin / spinback** | cambio di genere o di BPM, energia alta, A che canta sull'uscita | 1–2 battiti (≈ ultima battuta della frase) | fermo 40 ms, lancio all'indietro fino a 3×, decadimento | effetto marcato: abuso, "trucco" | [P] rekordbox Release FX, LSA 7, zZounds; [T] Katz 2012 (Flash) | **`backspin`** |
| **Brake / tape stop** | fine brano, calo di energia, cambio drastico | 1/2/4 battiti (rekordbox 1/2/4/8) | velocità (1−u)^1,5 | come sopra; voce che "scende" | [P] rekordbox Vinyl Brake, DJM Vinyl Brake | **`tape_stop`** |
| **Loop / estensione strumentale** | outro di A troppo corto, voce di A che riprende, B ha intro lungo | loop 1–2 battute ripetuto per 4–8 battute | inizio/fine su downbeat, crossfade di giunzione 10 ms, sorgente mix o strumentale (stem) | ripetizione statica, giunture udibili | [P] Serato/Mixxx loop, LSA 6, Violet Recording; [T] Katz (estensione dei break); [S] Wenner 2013, Infinite Jukebox | **`loop_extend`** |
| **Loop roll / chop / stutter** | ultima battuta prima di un drop | 1 battuta, lunghezze 1/2→1/16 di battito | dimezzamento a ogni battito, rilascio sul downbeat | abuso; su voce nota = cliché | [P] DDJT loops, Engine DJ Roll, Serato Roll, DJM Roll | **`build_into_chorus(stutter_bars=1)`** |
| **Build-up con snare roll e sweep** | B entra su un ritornello/drop (sostituisce il drop secco) | 2–4 battute | HPF crescente su A, rullo 1/4→1/8→1/16→1/32, crescendo −14→−6 dB | può suonare EDM nel rap | [P] pratica EDM/open format; [E] Solberg & Dibben 2019 (EDM); [U] drop secco 3/10 | **`build_into_chorus`** |
| **Riser in tonalità** | dentro il build | = build | glissando che arriva sulla tonica di B, LPF 400→6000 Hz, rumore HPF 300→6000 Hz | tonica sbagliata se la tonalità è stimata male | [P] produzione EDM; [H] glissando sulla tonica | **dentro `build_into_chorus`** |
| **Nota / pad ponte** | tempi lontani con tonalità compatibili, "respiro" tra due brani | pad 1 battuta prima + 0–1 battuta sola + 1–2 battute di rilascio | accordo con note comuni alle due scale (triade di B se possibile), voicing 220–660 Hz | estraneo al timbro dei brani; tonalità stimata male | [P] pad/ambient bed (open format); [H] scelta dell'accordo | **`note_bridge`** (sintetico o freeze di A) |
| **Acapella su strumentale** | A chiude un verso, B inizia a cantare; tempo ±5 BPM circa; tonalità uguale/compatibile | 4–8 battute (frase), poi swap | voce di B (stem) su strumentale di A, mid cut −3 dB, anacrusi di B, swap su downbeat | artefatti di separazione, due timbri di batteria | [P] Serato Stems, DDJT acapellas; [E] Davies 2014; [S] MosaikBox | **`acapella_over`** |
| **Vocal swap** | come sopra, con passaggio graduale degli strumentali | 4 + 2 battute | voce a B sul downbeat, strumentale di B senza bassi che sale, bass swap, A esce | due strumentali sovrapposti (tonalità!) | [P] stems; [S] Vande Veire bass swap | **`vocal_swap`** |
| **Word play / lyric link** | ultima parola di A e prima di B uguali, con lo stesso lemma o in rima | dalla parola di A a quella di B (1–3 battute) | eco della parola isolata (stem) con periodo ≈ 1/2, 1, 2 battiti scelto perché la K-esima ripetizione cada sulla parola di B | ASR sbagliata; parola poco intelligibile; lingue diverse | [P] DDJT Transitions Toolbox ("Wordplay"), pratica di DJ Premier/Pete Rock (fonte secondaria) | **`lyric_link`** + `find_lyric_links` |
| **Key change / pitch shift di un semitono** | tonalità a un semitono dalla compatibilità | per tutta la transizione e oltre | Rubber Band R3, formanti preservate sulla voce | timbro alterato; B resta trasposto | [E] Kim 2020 §6.2 (2,5% trasposte, 94,3% di 1 semitono); [S] Vande Veire | **`key_shift_b`** in loop/acapella/vocal swap |
| **Energy boost (+2 Camelot)** | alzare l'energia del set | — (scelta della coppia) | un tono sopra | — | [P] Mixed In Key | non implementato (selezione) |
| **Tempo ride** | BPM vicini ma non uguali, set che sale/scende | su 1 ritornello o più | passi piccoli, stretch condiviso tra A e B | key lock "ruvido" (DDJT) | [P] DDJT, Serato sync; [E] Ishizaki 2009 (dividere lo stretch riduce il disagio) | non implementato (`dj_set` ha il limite noto del salto di tempo dopo il blend) |
| **Half-time / double-time** | 70↔140 (trap) | come un blend | piegatura d'ottava della griglia | cambio di "feel" che può svuotare la pista (DDJT) | [P] Mixxx sync 140/70, DDJT, DJ.Studio | già nella piegatura di `dj_render`/`dj_select` |
| **Mid cut per la voce** | durante acapella o blend con voce | come la sovrapposizione | −3 dB sulla banda 200–2500 Hz dello strumentale | voce che resta "sopra" troppo | [P] LSA 2 (mid swap) | **dentro `acapella_over`** |
| **Double drop** | due drop allineati (EDM, DnB) | fade-in 16, fade-out 32 misure | drop di A e B sullo stesso downbeat | troppa densità, nel rap due voci | [S] Vande Veire 2018 §4.2.3 | non implementato: nel rap = due ritornelli con voce (scontro) [H] |
| **Beat juggling** | routine dal vivo con due copie | battute singole | alternanza di break tra due deck | è performance, non transizione | [P] Broughton & Brewster; [T] Katz 2012 | non implementato (fuori scopo) |
| **Scratch** | routine, accenti | battiti | gesti del piatto | richiede un modello di gesto | [E] Hansen et al. 2011 (3 DJ) | non implementato |

---

## 3. Cosa dice la letteratura sulle singole scelte (e cosa no)

- **Loop ed estensione.** Wenner et al. 2013 [S] e l'Infinite Jukebox [S] mostrano che un brano si può allungare saltando tra punti simili; la pratica DJ usa invece loop corti allineati alla griglia [P]. Qui il loop resta sulla griglia (inizio/fine su downbeat, lunghezza 1–2 battute intere), con giuntura che porta la coda del loop nel materiale che precede il loop: il downbeat ripetuto arriva intero. Nessuno studio misura quanto a lungo un loop regge prima di stancare [H: 4–8 battute].
- **Acapella e vocal swap.** Davies 2014 [E] e Huang 2021 [E] confermano che la compatibilità verticale (voce su accompagnamento) va valutata separatamente; AutoMashup 2025 aggiunge che è **asimmetrica**. Il nostro `acapella_over` usa la voce di B su A (e non viceversa) perché il passaggio va da A a B; l'altra direzione (voce di A su strumentale di B) è un'estensione semplice.
- **Build e riser.** Solberg & Dibben 2019 [E] e Solberg & Jensenius [E] riguardano l'EDM. Nel rap il build è un'importazione di pratica open format [P]; il voto 3/10 al drop secco [U] dice solo che il drop senza preparazione non è piaciuto in quel caso.
- **Word play.** È documentato solo come pratica (corsi, aneddoti su DJ Premier e Pete Rock). L'implementazione cerca legami lessicali, di lemma (spaCy, solo inglese), di rima ortografica e Soundex. **Semantica non disponibile**: `en_core_web_sm` non ha vettori di parola e nessun modello di frase è installato; non è stata simulata.
- **Spinback, brake, freeze, reverb.** Sono funzioni standard di software e mixer [P] con durate in battiti (rekordbox 1/2/4/8). Nessuno studio ne misura il gradimento.
- **Modelli generativi.** DJtransGAN [E] non supera le regole; Levy 2023 è un preprint; MixWeaver non è verificabile. Per ora la generazione neurale non dà un vantaggio documentato su DSP esplicito + stem [H].

---

## 4. Il metodo giusto al momento giusto: tabella contesto → tecnica (`dj_policy.POLICY_TABLE`)

Ingressi misurati (da `dj_set.transition_context` + `dj_policy.creative_context`): stretch ripiegato d'ottava, relazione Camelot e trasposizione ±1 possibile, quota vocale per battuta di A prima e dopo il cue e di B dopo il cue (stem Demucs), voce di A nell'ultimo battito, sezione `allin1` di B all'ingresso, energia relativa delle sezioni, legami di parole (ASR faster-whisper), tecniche usate di recente.

| Regola | Contesto | Tecniche | Durata | Fonte |
|---|---|---|---|---|
| C1_lyric | parola finale di A e parola iniziale di B legate (uguale, lemma, rima, Soundex) con ASR affidabile | lyric_link (eco della parola di A, K-esima ripetizione sulla parola di B) | dalla parola di A alla parola di B (1–3 battute) | [P] word play; [H] allineamento dell'eco |
| C2_chorus | B entra su un ritornello/hook | build_into_chorus (HPF + rullo + riser in tonalità di B, stutter se A canta sull'uscita) | 2 battute (4 se B è molto più energico) | [U] drop secco 3/10; [E] Solberg & Dibben 2019; [P] build-up |
| C3_verse_handoff | A finisce un verso sul cue, B canta sul cue, tempo ±8%, tonalità compatibile o a ±1, stem | acapella_over (4; 8 se stretch ≤ 5% e B canta 8 battute), vocal_swap (4 + 2) | 4–8 battute | [P] stems; [E] Brungart 2001, Davies 2014; [E] Kim 2020 |
| C4_extend | tempo ±8%, tonalità ok, A con poche battute strumentali dopo il cue o voci in entrambi | loop_extend (loop strumentale di 1–2 battute) | 8 battute se stretch ≤ 5%, altrimenti 4 | [P] loop; [S] Vande Veire bass swap; [E] Kim 2020 |
| C5_far_tempo | tempo oltre ±8% | echo_freeze; note_bridge con 1 battuta di solo pad (tonalità compatibile); backspin (A canta o spinge); tape_stop (B più calmo) | 1 battuta + coda | [U] eco 9/10; [P] spinback/brake; [H] pad |
| C6_energy_drop | energia di B < A − 6 dB | reverb_wash, tape_stop | 1 + 2 battute / 2 battiti | [P]; [H] |
| C7_default | resto | echo_freeze, reverb_wash (e note_bridge senza pausa se la tonalità è compatibile) | 1–3 battute | [U]; [H] |

Ordinamento: `prior` a priori per tecnica (`dj_policy.PRIOR`, con fonte), +0,1 per le regole specifiche di contesto (C1–C4), −0,15 se la tecnica è stata usata di recente (Williams 2025 [T], evidenza debole). Si restituiscono 2–4 proposte con parametri pronti per `dj_creative.render_creative`; `considered` registra anche le tecniche scartate con il motivo. Le soglie sono quelle già usate in `dj_select`/`dj_set` o [H]; nessuna è stata stimata dai giudizi.

---

## 5. Implementazione (`src/musicmix/dj_creative.py`)

Regole comuni: griglie regolarizzate di `dj_select`; cue già sul downbeat; normalizzazione −14 LUFS integrata per brano (stem con il guadagno del mix); B beatmatchato con `dj_render.render_transition(kind="blend", cue_convention="start")` (time map per battuta di Rubber Band R3), mix e stem vocale con le stesse griglie, strumentale = mix − stem; section gain match di `dj_render` attivo; overlay (rullo, riser, pad, freeze, riverbero, eco) tarati sul livello di A e ridotti se il massimo momentary nella regione supera A prima di +1 dB; limitatore true-peak −1 dBTP; clip = 10 s + transizione + 10 s. Metadati come dj_render + `creative`, `exit_effect` (solo se A esce davvero da un effetto), mappe `a_map`/`b_map` e deck `vocals_a_heard`/`vocals_b_heard` per il QA (una voce alla volta misurata sulla voce effettivamente udibile).

| Funzione | Cosa fa | Parametri (default) | Base |
|---|---|---|---|
| `loop_extend` | loop di 1–2 battute di A (mix se la voce copre ≤ 10% del loop, altrimenti strumentale) sotto l'ingresso di B beatmatchato; curve del blend di dj_render (bass swap a metà) | loop_bars 1, bars 8, key_shift_b 0 | [P][S][H] |
| `acapella_over` | voce di B (stem, anacrusi inclusa) sullo strumentale di A con mid cut −3 dB per N battute, poi swap completo sul downbeat (bassi di A via con lo swap, medi/alti in 1 battito) | bars 4, mid_cut_db −3, swap_release_beats 1 | [P][E][S] |
| `vocal_swap` | voce a B sul cue, base di A per N battute mentre lo strumentale di B entra senza bassi, bass swap, A esce in 2 battute | bars 4, out_bars 2 | [P][S] |
| `build_into_chorus` | HPF 30→1200 Hz su A, rullo (sintetico o estratto da A con HPSS) che dimezza a ogni battuta, riser con glissando sulla tonica di B, stutter facoltativo, pausa facoltativa; B pulito sul downbeat del ritornello | build_bars 2, snare auto, riser on | [P][E][U][H] |
| `note_bridge` | pad nella tonalità comune (`bridge_chord`) o freeze spettrale dell'armonia di A; A sfuma in 2 battiti; B pulito dopo 0–1 battute di pad; pad rilasciato in 2 battute | gap_bars 0, pad_level −8 dB | [P][H] |
| `echo_freeze` | freeze spettrale dell'ultimo battito + eco a 1 battito, HPF crescente, B pulito, coda −12 dB | hold 1, tail 1 | [P][U] |
| `reverb_wash` | IR sintetica RT60 3,5 s, mandata crescente nell'ultima battuta, coda −12 dB sotto B | wash 1, tail 2 | [P] |
| `backspin` | lettura a velocità variabile (fermo, lancio −3×, decadimento) negli ultimi battiti, anti-aliasing, fader abbassato nell'ultimo battito | spin_beats 2, fader_out_beats 1 | [P][T] |
| `tape_stop` | velocità (1−u)^1,5 negli ultimi battiti, fader abbassato nell'ultimo battito | stop_beats 2, fader_out_beats 1 | [P] |
| `lyric_link` | eco della parola isolata di A, voce di A zittita dopo la parola, filter out della base; periodo dell'eco scelto (±6% da 1/2, 1, 2 battiti) perché la K-esima ripetizione (K ≥ 2) cada sull'attacco della parola di B; se le parole sono troppo vicine B entra 1–2 battute dopo l'uscita di A (come `entry_bars` di echo_out); eco zittita dove B canta | feedback 0,6, filter_bars 1, entry_delay_bars auto | [P][U][H] |
| `find_lyric_links`, `word_match`, `rhyme_key`, `soundex` | ricerca dei legami (ultime 3 parole di A entro 2 battute prima del cue, prime 4 di B da 1/4 di battuta prima a 2 battute dopo; esatto 1,0, lemma 0,85, rima 0,7, Soundex 0,5 × √confidenza ASR) | min_confidence 0,3 | [H] |
| `auto_key_shift`, `bridge_chord`, `pitch_shift` | trasposizione ±1 che rende la coppia stessa/adiacente/relativa; accordo con note comuni; Rubber Band `-3 -p` (formanti sulla voce) | — | [E] Kim 2020; [S] Vande Veire |

**Nudge di fase (correzione dopo la run preliminare).** Le griglie regolarizzate di `dj_select` allineano i downbeat delle griglie, ma nella prima esecuzione i render con sovrapposizione sono stati bocciati dal controllo ritmico con errori costanti di 18–36 ms tra gli attacchi (Beat This! sui deck): es. loop_extend Kamenolom → Jonay 21 ms su 32 battiti, IQR 9 ms. Una stima della fase sulle sorgenti (Beat This! sugli estratti prima/dopo i cue) non prevedeva l'errore sui deck (nudge stimato 0 ms contro 21 ms misurati). Il driver ora fa come un DJ che corregge a orecchio: prima passata, misura dell'errore B − A sui deck con la stessa misura di `dj_qa`, seconda passata con `phase_nudge_s` (B anticipato o ritardato, ±40 ms al massimo, solo se l'errore è ≥ 5 ms e coerente, IQR ≤ 25 ms); si tiene la passata con l'esito migliore e si registrano entrambe. **Conseguenza dichiarata**: per i render corretti il controllo ritmico del QA non è più indipendente (stesso tracker); resta indipendente il controllo per correlazione degli attacchi, spesso non affidabile quando i contenuti ritmici differiscono. Nella prova mirata: loop_extend da 20,9 ms (fail) a 0,1 ms (pass); acapella_over Steves → Slim Marion da fail a warn (deriva non significativa su 8 battiti).

Test: `tests/test_dj_creative.py` (25 test sintetici: normalizzazione in place identica a quella con copia; fader che rende graduale l'uscita dello spin; nudge che sposta B del numero di campioni richiesto; lyric link che ritarda B quando le parole sono troppo vicine; ripetizioni identiche del loop e voce rimossa; acapella senza strumentale di B prima dello swap e voce di A zittita; trasposizione di +1 semitono misurata sullo spettro; bass swap del vocal swap; numero ed eventi del rullo; riser che arriva sulla tonica; pad nella battuta sola; code ridotte sotto B; profili di velocità di backspin e tape stop; K-esima eco sulla parola di B; mappe delle parole; `dj_qa` eseguito su un render) e `tests/test_dj_policy.py` (6 test sulle regole).

---

## 6bis. Prova `runs/dj_creative_v1` (21/10/2026)

Driver `tools/run_dj_creative_v1.py`: 45 brani disponibili (pool V12 + CC completi), 237 coppie ordinate con tempo entro ±8% e tonalità compatibile o a ±1 semitono; per ogni coppia si sceglie il piano di `dj_select.plan_pair` che abilita più regole specifiche della policy (C1–C4), si scartano le uscite in dissolvenza, si renderizzano fino a 3 proposte (si passa alla successiva se il QA fallisce) più il riferimento standard dell'albero `dj_set` sullo stesso cue. Ogni render gira in un sottoprocesso (la prima esecuzione è stata uccisa più volte dall'OOM del cgroup condiviso con gli altri agenti); l'audio è passato come estratto (A −45/+60 s, B −30/+90 s) con la LUFS dell'intero brano, quindi la normalizzazione delle tecniche creative è quella di dj_render; il riferimento standard (dj_render/dj_effects) normalizza invece l'estratto (dichiarato nel manifest).

Esito QA (`dj_qa.evaluate_render`, integrità tecnica, non qualità): 9 gruppi, 37 render: **20 pass, 12 warn, 5 fail** (fail esclusi dalla pagina, in `rejected/`).

| Gruppo | Coppia (stretch, tonalità) | Regole | Render (esito) |
|---|---|---|---|
| c01 | Kamenolom → Jonay (−2,6%, adiacente) | C1, C2, C4 | lyric link «Japan!»→«man» pass; build 2 + stutter pass; loop 2+8 pass (nudge 20,9 ms); rif. echo out pass |
| c02 | 5% TINT → Creature (−1,4%, stessa) | C1, C2, C4 | lyric link «know»→«know» warn (ritmo non misurabile); build pass; loop pass (nudge −7,3 ms); rif. echo out pass |
| c03 | Steves J Bryan → Slim Marion (−2,1%, relativa) | C2, C3, C4 | build 4 warn; acapella 4 warn (deriva n.s.); loop pass; vocal swap **fail** (36 ms, deriva); rif. filter verso il ritornello warn |
| c04 | Jonay → Doctor Frost (0%, stessa) | C2, C3 | build pass; acapella 8 pass; vocal swap pass; rif. filter verso il ritornello pass |
| c05 | DK Music → Porrazzo (0%, stessa) | C4 | loop **fail** (96 ms: griglia di B instabile); echo freeze pass; nota ponte warn; reverb wash warn; rif. blend 4 warn |
| c06 | Afrocream → TSF (1,1%, adiacente) | C4 | loop pass; echo freeze pass; nota ponte pass; rif. blend 8 pass |
| c07 | 2+2 → Mit Rich (−6,7%, adiacente) | C3 | acapella warn; vocal swap warn; echo freeze warn (ritmo non misurabile); rif. blend 4 **fail** |
| c08 | Porrazzo → Kamenolom (−4,8%, stessa) | C3 | acapella **fail** (deriva 29 ms); vocal swap warn (picco +2,0 dB nella sovrapposizione; nudge −13,8 ms); echo freeze pass; nota ponte warn; rif. blend 8 **fail** |
| c09 | Afrocream → TSF, stesso cue di c06 (dimostrazione) | — | backspin pass; tape stop pass (aggiunti a mano: la policy non li propone con tempi compatibili) |

Osservazioni (diagnostica, non validazione):
- **Tutti i fail sono ritmici** e riguardano solo le tecniche con sovrapposizione beatmatchata (loop, acapella, vocal swap) e due riferimenti standard (blend): stesso problema di fase/deriva delle griglie già visto in `dj_set_v1`. Le tecniche senza sovrapposizione non hanno fail.
- Il **nudge** in seconda passata ha recuperato 3 render (loop c01 e c02, vocal swap c08); ha peggiorato o non cambiato i casi con deriva (griglia non costante), dove serve una griglia migliore, non uno spostamento costante.
- I warn "mix-only: B beat grid ±19–420 ms from A's grid at the switch" dei render senza sovrapposizione misurano lo sfasamento tra la griglia di A estrapolata e B che entra al suo tempo: per costruzione non c'è beatmatch; non è un difetto di per sé.
- Prima della correzione del fader, backspin e tape stop fallivano il controllo "verso tagliato" (uscita giudicata brusca perché il riferimento di livello cade dentro lo spin); con il fader abbassato nell'ultimo battito (pratica dichiarata) passano.
- Pagina d'ascolto: `tools/build_listening_artifact.py runs/dj_creative_v1/review_manifest.json --out …` costruisce 9 gruppi, 32 clip, 0 escluse, tutte conformi alla regola delle clip. Non pubblicata.

---

## 6. Limiti e ignoto

- Nessuna di queste tecniche è stata ascoltata dall'utente: un `pass` del QA è integrità tecnica (ritmo, doppio basso, salto di loudness, true peak, voci, parola tagliata, durata), non qualità musicale.
- La qualità di acapella, vocal swap, loop strumentale e lyric link dipende dalla separazione Demucs (artefatti nello strumentale = mix − stem) e dall'ASR (faster-whisper small, confidenze non calibrate; parole in lingue diverse dal modello). La rima è ortografica, non fonetica.
- Il riser e il rullo sono di matrice EDM; il pad è sintetico: possono suonare estranei al rap [H]. L'estrazione del rullante da A (HPSS sui backbeat) è un'euristica con controllo spettrale e ripiego sul sintetico.
- Le durate e i livelli sono a priori ([P]/[H]); vanno tarati con confronti A/B forzati sulla stessa coppia (Mantiuk 2012), tenendo i sei voti v1 come diagnostica separata.
- La prova copre 8 coppie compatibili + 1 gruppo dimostrativo; backspin e tape stop sono stati ascoltabili solo nel gruppo dimostrativo; reverb wash in un solo gruppo.
- Per i render con nudge il controllo ritmico del QA non è indipendente (stesso tracker usato per correggere).
- Non implementati: tempo ride graduale, energy boost +2, double drop, beat juggling, scratch, voce di A sullo strumentale di B (direzione inversa dell'acapella).

---

## 7. Righe da aggiungere (non applicate: INDEX.md e CHANGELOG.md non sono stati modificati)

Per `conoscenza_mix/INDEX.md`, nuova sezione dopo "Implementazione del set DJ v1":

```
## Tecniche creative di transizione e mashup (2026-10-21)

Catalogo di 23 tecniche (loop, loop roll, acapella, vocal swap, word play, echo freeze, reverb, backspin, brake, build con rullo e riser in tonalità, nota/pad ponte, trasposizione di un semitono, tempo ride, half-time, filter swap, mid cut, double drop, beat juggling, scratch) con contesto, durata, parametri, rischi e fonte; regole contesto → tecnica e implementazione: [`_meta/TECNICHE_CREATIVE_2026-10-21.md`](_meta/TECNICHE_CREATIVE_2026-10-21.md) (`src/musicmix/dj_creative.py`, `src/musicmix/dj_policy.py`, run `runs/dj_creative_v1/`). Le tecniche sono pratica DJ [P] o sistemi [S]; nessuno studio empirico ne misura l'uso o il gradimento nei mix reali, tantomeno nel rap.
```

Per `conoscenza_mix/CHANGELOG.md`:

```
## 2026-10-21 — tecniche creative v1
- Nuovo `_meta/TECNICHE_CREATIVE_2026-10-21.md`: catalogo delle tecniche di transizione/mashup con fonti di pratica (Serato, rekordbox, Pioneer DJM, Traktor, Mixxx, Mixed In Key, Digital DJ Tips, London Sound Academy) e di ricerca (AutoMashUpper, Lee 2015, Huang 2021, Wenner 2013, Infinite Jukebox, Mixboard 2023, AutoMashup 2025, DJtransGAN, Levy 2023).
- Nuovi `src/musicmix/dj_creative.py` (10 tecniche + ricerca dei legami di parole) e `src/musicmix/dj_policy.py` (tabella contesto → tecnica, 2–4 proposte motivate); test `tests/test_dj_creative.py`, `tests/test_dj_policy.py`; driver `tools/run_dj_creative_v1.py`; run `runs/dj_creative_v1/` con QA.
- Lacune confermate: nessuno studio su tipi/gradimento delle tecniche creative nei mix reali; nessuna stima automatica di echo/reverb/loop dai mix (solo fader+EQ, Kim 2021/2022); nessun modello generativo di transizioni con vantaggio documentato.
```
