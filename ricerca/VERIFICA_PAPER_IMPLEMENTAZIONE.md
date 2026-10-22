# Verifica paper → implementazione

**Data audit:** 18 ottobre 2026 · **Perimetro:** testi integrali prioritari e
confronto con la baseline presente nel repository.

Questo audit separa ciò che è stato letto nel corpo del lavoro da ciò che è
solo una proposta del progetto. Le prestazioni riportate dai paper appartengono
ai loro dataset e task: non sono risultati trasferibili automaticamente al
matching A→B di questo repository.

## Stato dell'evidenza

| Fonte | Lettura verificata | Parti utilizzate | Nota di accesso |
|---|---|---|---|
| Bittner et al., *Automatic Playlist Sequencing and Transitions* (2017) | **Testo integrale** | Sec. 3, 3.1–3.2, Sec. 4, 4.1–4.3, Algoritmo 1, Sec. 5 | [PDF ISMIR, 7 pp.](https://archives.ismir.net/ismir2017/paper/000086.pdf), pp. 442–447 |
| Lee et al., *Automatic Mashup Creation by Considering Both Vertical and Horizontal Mashabilities* (2015) | **Testo integrale** | Sec. 3–8, Eq. (1)–(3), Fig. 5–7, esperimenti | [PDF ISMIR, 7 pp.](https://www.ismir2015.uma.es/articles/302_Paper.pdf), pp. 399–405 |
| Davies et al., *AutoMashUpper* (2014) | **Testo integrale** | Sec. II–III, Eq. (1)–(12), Sec. V–VII | [Pagina con HTML/PDF completo](https://www.researchgate.net/publication/265130656_AutoMashUpper_Automatic_Creation_of_Multi-Song_Music_Mashups), pp. 1727–1737. Il download DSpace indicato in `PAPER_LINKS.txt` non è stato estratto dal browser. |
| Kim & Nam, *All-In-One Metrical and Functional Structure Analysis* (2023) | **Testo integrale HTML** | Sec. 2–5, Tab. 1, ablation | [arXiv HTML](https://arxiv.org/html/2307.16425), sezioni HTML esplicite |
| Kim et al., *CREPE* (2018) | **Testo integrale PDF** | Sec. 2–4, Eq. (1)–(4), Tab. 1–2 | [arXiv PDF](https://arxiv.org/pdf/1802.06182), pp. 1–4 |
| Schulze-Forster et al., *Phoneme Level Lyrics Alignment…* (2021) | **Testo integrale preprint** | Sec. III–VII, Eq. (1)–(7), Tab. II–VI | Il link locale in `PAPER_LINKS.txt` restituisce 404; testo completo verificato nel [preprint dell'autore](https://perso.telecom-paristech.fr/grichard/Publications/2021_05_TASLP_preprint.pdf), 14 pp. |

Le altre voci della bibliografia possono essere utili, ma **non sono state
riesaminate integralmente in questo audit**: Modeling Stem Compatibility,
COCOLA, AutoMashup 2025, Neural Loop Combiner, Diversity, Cue Points, Raw
Audio to Seamless Mix, DJ Mix Alignment, DJGAN, Spectral Clustering, Basic
Pitch, Demucs/Hybrid Transformers, BS-RoFormer, CrossSong e il paper
psychoacoustic. Le schede in `ANALISI_PAPER_2026-10-18.md` restano note di
ricerca precedenti; questa verifica non le promuove a evidenza indipendente.

## Risultati implementabili dai testi letti

### 1. Spotify 2017: matrice di costo su regioni beat-sincrone

Il lavoro separa due problemi. Il sequenziamento di una playlist è uno
shortest Hamiltonian path su un grafo di brani (Sec. 3.1.2, p. 443); la scelta
del punto di transizione è invece una ricerca pairwise su due tracce (Sec. 4.1,
pp. 444–446). Il primo problema non giustifica un ranker di cue locali.

Per ogni brano il paper costruisce, sulla griglia dei beat, matrice timbrica
`T`, chroma `C`, loudness `ℓ` e probabilità di vocalità `v` (Sec. 4.1.1,
p. 445). I punti iniziali sono downbeat `M`; le sezioni `S` e i drop `D` danno
priorità strutturale. Per una transizione di `n` beat, la regione `[i:i+n]` di
A viene confrontata con `[j:j+n]` di B. L'Algoritmo 1 (p. 445) calcola in
sostanza:

```text
ΛT[i,j] = norm(TA[i:i+n] - TB[j:j+n])
ΛC[i,j] = norm(CA[i:i+n] - CB[j:j+n])
Λℓ[i,j] = mean(2 - (ℓA + ℓB))
Λv[i,j] = mean(vA) + mean(vB)
ΛD[i,j] = 1[endA ∉ D_A] + 1[endB ∉ D_B]
ΛS[i,j] = 1[endA ∉ S_A] + 1[endB ∉ S_B]
standardizza ogni Λx in [0,1]
Λ = weightedAvg(ΛT, ΛC, Λℓ, Λv, ΛD, ΛS)
(i*,j*) = argmin Λ
```

Il paper considera solo metrica quaternaria (4/4) e, dopo aver fissato la
durata, pruna l'uscita nell'ultimo 25% di A e l'ingresso nel primo 20% di B
(Sec. 4.1.3, p. 445). Questo pruning è una scelta del loro prodotto DJ, non un
requisito del progetto: per ritornello↔strofa e parola↔parola va rimosso o reso
policy esplicita. Il rendering allinea progressivamente i beat: se il beat `n`
di A e B ha durate `d1`, `d2`, la durata interpolata è

`d_out = ((N-n)/N) d1 + (n/N) d2`,

con fattori di time-stretch `d1/d_out` e `d2/d_out` (Sec. 4.2, p. 446).

Il test comprende 48 coppie, quattro curatori professionali e un vincolo di
tempo differenziale massimo di 5 BPM. Le valutazioni sono 64,13% Good, 28,26%
OK e 7,61% Bad; gli errori segnalati sono 15,22% beat non allineati, 6,52%
transizioni a metà voce, 2,17% non-downbeat, 2,17% punti goffi e 0% key clash
(Tab. 2–3, p. 446). La conclusione attribuisce il collo di bottiglia
principalmente a beat/downbeat tracking (Sec. 5, p. 447). Sono numeri
diagnostici per progettare i test, non un benchmark della baseline locale.

**Scarto rispetto al repository:** `analysis.py` emette `kind=onset_grid` da
spettro FFT e flux; `search.py` confronta descriptor puntuali (cosine chroma,
RMS/onset ratio, cosine timbre, tempo ratio) e non regioni beat-sincrone. Non
sono ancora disponibili downbeat, sezioni, drop, vocalness, matrici
standardizzate o time-stretch. `vertical=unknown_no_overlap` e
`creative=unknown_not_measured` sono quindi etichette corrette. Il baseline
attuale va chiamato **Spotify-inspired cue similarity**, non implementazione
dell'Algoritmo 1.

**Prossimo adattamento:** introdurre un provider M02 per beat/downbeat/section,
generare finestre di `n` beat su tutto il brano, conservare una matrice per
feature e confrontare A→B con B→A. Prima del provider completo si può mantenere
la baseline attuale come B0 tecnico, ma non usare le sue metriche come evidenza
del paper.

### 2. Lee 2015: successione orizzontale e sovrapposizione verticale

Il sistema assume multitrack con lead/background, unità allineate a battute e
prevalenza di 4/4; usa 104 brani pop con circa cinque tracce per brano (Sec.
3–4 e Sec. 7.1, pp. 399–403). La distinzione è concreta:

- `Sv` misura la sovrapponibilità lead/background;
- `Sh` misura la continuità del gruppo di background con il gruppo successivo;
- la composizione deve anche conservare la continuità della lead tra due
  gruppi.

La parte verticale non è una cosine generica. Il paper calcola `Sc` da chroma,
la percentuale di beat stabili e la sua trasformazione sigmoide in un tasso di
cambiamento armonico `ξ`; la complementarità è:

`wt = 1 - |ξp - (1 - ξq)|`  (Eq. 1, p. 401).

La ponderazione di volume `wv` penalizza unità lead quasi inudibili (soglia
indicata nel testo a circa −40 dB), e la combinazione è:

`Sv = Sc · wt · wv + wτ`  (Eq. 3, p. 401),

dove `wτ` premia tempo vicino. La fase orizzontale costruisce due liste
ordinate `Rh` e `Rv`, applica soglie ai punteggi e gestisce quattro casi (Sec.
5.2, pp. 401–402): entrambi presenti; solo `Rv`; solo `Rh`; nessuno. Quando
entrambe le liste sono non vuote ma non intersecano, il paper preferisce `Rv`
perché il test umano preferisce il ponte tramite lead continua al cambio di
lead sopra un background affine.

Nel primo test 27 ascoltatori (21 uomini, 6 donne) hanno usato una scala
Likert 1–7; il caso 1 è secondo solo alla traccia senza transizione, il caso 2
è in genere preferito al caso 3, mentre il caso 4 è il peggiore (Sec. 7.2,
pp. 402–403). Nel confronto verticale hanno partecipato 24 ascoltatori; il
metodo con `wt` è generalmente sopra AutoMashUpper, ma due gruppi non sono
significativi (Sec. 7.3, p. 403). Questo supporta un'**ablation separata**,
non il trasferimento diretto dei pesi.

**Scarto rispetto al repository:** il renderer `neutral_linear_v1` concatena
finestre del mix e il retrieval non sovrappone lead/background; perciò la
verticalità di Lee non è ancora calcolabile. Il primo ranker deve mantenere
`vertical=None` fino a quando non esistono stem o finestre sovrapposte. Per la
fase M3/M8 servono ruoli di stem, continuità A→B e B→A e un test separato
`vertical`/`horizontal`; non basta rinominare la cosine attuale.

### 3. AutoMashUpper 2014: ricerca locale nel dominio di trasformazione

Il testo assume tempo quasi costante e 4/4. La segmentazione usa semitone
spectrogram, gruppi non sovrapposti di quattro beat da ogni downbeat, matrice di
autosimilarità e kernel checkerboard gaussiano di 16 downbeat; una regolarità
successiva privilegia durate di 2, 4, 8 e 16 downbeat (Sec. II.B, pp. 1727–1728).

La mashability è locale, non una key globale: per ogni frase cerca posizione e
trasposizione che massimizzano la compatibilità. L'armonia usa una convoluzione
2-D su patch chroma e tutte le rotazioni di chiave (Sec. III.A, pp. 1728–1730);
il ritmo campiona kick e snare in 12 posizioni per beat e forma 24 dimensioni
per beat (Sec. III.B, p. 1730); il bilanciamento spettrale somma bande bassa
`≤220 Hz`, media `220–1760 Hz` e alta `>1760 Hz` (Sec. III.C, pp. 1730–1731).
I tre contributi e un bonus di tempo sono combinati con pesi regolabili
dall'utente (Eq. 8–9, Sec. III.D, pp. 1731–1732); il paper dice che nei test
informali l'armonia riceve il peso maggiore, quindi non va inventata una
terna numerica come se fosse un risultato validato.

La valutazione di segmentazione usa RWC Popular Music (100 brani); il test
soggettivo usa 12 estratti di 32 beat da 90 brani e 15 partecipanti. Le medie
di gradimento sono 6,7, 4,7 e 4,3 per alto, medio e basso ranking; il primo è
generalmente migliore del medio, ma il paper osserva anche casi con vocal
overlap piacevole e casi ad alto score non graditi (Sec. V, pp. 1734–1735).
Le limitazioni esplicite sono tempo quasi costante, 4/4, dipendenza dalla
tuning e difficoltà con voce trasposta/overlap (Sec. VI–VII, pp. 1735–1737).

**Scarto rispetto al repository:** `analysis.py` mira a una finestra FFT locale
di circa 46 ms con hop 1/4 finestra; a 48 kHz, la scelta concreta di 2304
campioni viene arrotondata alla potenza di due successiva (4096, circa 85,3 ms)
dal backend attuale. La durata è quindi source/configuration-dependent e non va
presentata come costante del paper. L'analisi non produce chroma beat-sincrona,
kick/snare sub-beat o bande spettrali di AutoMashUpper. Il confronto locale può
restare una baseline economica, ma va documentato come **AutoMashUpper-inspired**
solo dopo aver separato le sue componenti e validato su cue uguali.

### 4. All-In-One 2023: provider strutturale da integrare

Il paper usa spettrogrammi di sorgenti separate, tre layer convoluzionali e
blocchi di Neighborhood Attention: DiNA 1-D per dipendenze temporali lunghe e
NA 2-D per dipendenze tra strumenti (Sec. 2.1–2.3). Il modello principale ha
circa 300k parametri; le dilatazioni raggiungono receptive field di circa 41 e
82 secondi. Beat/downbeat passano da DBN, mentre confini e label usano
normalizzazione su finestra scorrevole di 24 s e peak picking (Sec. 2.3).

Su Harmonix Set il test è 8-fold cross-validation, senza data augmentation o
dataset aggiuntivi. Tabella 1 riporta per All-In-One: beat F1 `0,958`, downbeat
F1 `0,915`, structure boundary HR.5F `0,660`, labeling PWF `0,738` e Sf
`0,769` (Sec. 3–4). L'ablation mostra che beat/downbeat e segmentazione si
aiutano; il risultato non implica che ogni boundary sia un cue di transizione
percepito come buono.

**Azione:** usare il repository [mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one)
come adapter M02 versionato, con output `beat`, `downbeat`, `boundary_prob`,
`label` e confidence. Il baseline onset-grid va conservato come fallback, non
spacciato per All-In-One. L'effetto di Demucs sulla predizione va misurato nel
nostro corpus, poiché i risultati Harmonix non sono una validazione dei nostri
audio.

### 5. CREPE 2018: F0 utile solo su voce monofonica e con maschera

CREPE prende crop di 1024 campioni a 16 kHz, sei convoluzioni e produce 360
attivazioni su bin da 20 cent, da C1 a B7. La stima è la media pesata su **tutti
i 360 bin** (il testo non prescrive un filtro vicino all'argmax):

`ĉ = Σ_i ŷ_i c_i / Σ_i ŷ_i`,  `f̂ = 10 · 2^(ĉ/1200)`  (Eq. 1–2, Sec. 2,
pp. 1–2).

Il modello è monofonico e stima ogni frame indipendentemente; non impone
continuità temporale come pYIN (Sec. 4, p. 4). Su MDB-stem-synth ottiene RPA
`0,967 ± 0,091` a 50 cent e `0,909 ± 0,126` a 10 cent; il dataset contiene
stems sintetizzati con F0 perfetto, non voci commerciali miste (Tab. 1–2,
pp. 2–3). In presenza di rumore la robustezza varia per spettro e strumento.

**Azione:** nel futuro M3 calcolare F0 su voce separata o su un segmento vocale
con `voicing_confidence`, mascherare consonanti non voiced e usare semitoni
relativi/DTW. Non interpretare la robustezza del paper come garanzia dopo
Demucs, riverbero, cori, autotune o sovrapposizione di due voci. La baseline
attuale non ha F0: `no_F0` non è un'ablation, è lo stato corrente.

### 6. Lyrics alignment 2021: DTW-attention e confidenza esplicita

Il paper non è un semplice forced aligner. Un text encoder BLSTM e un audio
encoder BLSTM producono sequenze `H` e `G`; il punteggio frame/fonema è
`s(m,n) = g_nᵀ W h_m`. Il cammino monotono parte da `(1,1)`, termina in
`(M,N)` e ammette passi `(0,1)` o `(1,1)`. La ricorrenza è:

`d(m,n) = s(m,n) + max(d(m,n−1), d(m−1,n−1))`  (Eq. 3–5, Sec. III.B,
pp. 4–5).

In training usa softmax colonnare degli score accumulati; al test esegue
backtracking DTW e definisce l'onset del fonema come il primo frame assegnato
(Eq. 6–7, Sec. III.B.1–2). La presenza di un VAD/spazio riduce errori su lunghi
strumentali (p. 6). Sul corpus NUS-48E, JOINT3 ha errore medio fonema 57 ms e
PCAS 85,94% in solo singing; su mix a SNR 0 dB ha 77 ms e 82,17%, a −5 dB 143
ms e 76,21% (Tab. II, p. 9). Sul word alignment, JOINT3-VAD è entro 0,3 s per
85% Hansen e 81% Jamendo, con AE medio 0,79/0,88 s (Tab. III, p. 10).

Il paper dichiara i failure mode: testo incompleto causa onset anticipati;
strumenti simili alla voce possono battere il token spazio; più cantanti e
fonemi simultanei rendono la separazione meno affidabile (Sec. V–VI,
pp. 9–12). Quindi lyrics esatte e confidence sono prerequisiti del nostro
word-mode, non dettagli cosmetici.

**Azione:** M3 deve salvare parola, fonemi, onset/offset e confidence; M6 deve
emettere `word` solo sopra una soglia preregistrata e mantenere un fallback
senza lyrics. Il test deve misurare errore onset/offset su un gold subset prima
di usare una parola come evidenza per il ranking. La baseline attuale non ha
lyrics né DTW.

## Audit della baseline presente

| Componente attuale | Evidenza osservabile | Interpretazione corretta | Gap paper |
|---|---|---|---|
| `audio.py` + `ingest.py` | WAV PCM decodificato; ingest FFmpeg 48 kHz stereo; hash e offset policy | M00 tecnico parziale | non verifica ancora offset encoder o round-trip su fixture reali |
| `analysis.py` | FFT, RMS, chroma snapshot, timbre, flux; cue `onset_grid`; confidence euristica | M02/M05 fallback deterministico | nessun beat/downbeat/section/voice detector; non All-In-One |
| `search.py` | griglia di coppie cue, top-K, cosine/ratio descriptors, cap e spacing | M06–M07 baseline economica | non regioni n-beat, non standardizzazione `Λ`, non direzione/contesto |
| `ranking.py` | ranker interpretabile con `horizontal`, `vertical`, `continuity`, `creative` | contratto di score; vertical/creative mancanti | non è ancora Spotify/Lee completo; pesi ipotesi |
| `preview.py` | `neutral_linear_v1`, crossfade lineare 0,5 s, gain 0,9, nessun pitch/tempo | M09 renderer neutro disponibile | non misura Gate B e non applica beat alignment |
| `pipeline.py` + CLI | manifest, eventi JSONL, candidate records, report e preview | demo end-to-end riproducibile | nessun ascolto umano, nessuna prova Gate A |

Il fatto che i test automatici passino prova contratti e determinismo del
software; non prova precisione dei cue, qualità percepita o adeguatezza del
ranker. Nelle prossime run il report deve mantenere espliciti `vertical` e
`creative` come unknown quando non ci sono dati, invece di convertirli in zero.

## Cosa è documentato e cosa resta ipotesi

**Documentato dai paper:** ricerca su griglia beat-sincrona e costo
standardizzato (Spotify); separazione horizontal/vertical e complementarità di
cambiamento armonico (Lee); match locale con trasposizione e ritmo sub-beat
(AutoMashUpper); struttura multi-task da audio demixato (All-In-One); F0
monofonico a bin in cent (CREPE); allineamento fonema monotono tramite DTW e
limiti su testo mancante (Lyrics 2021).

**Ipotesi del progetto, ancora da testare:** 48 kHz come timeline sufficiente;
contesti di 2 o 10 s; peso `horizontal=0,40`; soglia confidence 0,50; shift
massimo di 2 semitoni; tempo relativo massimo 8%; che una stessa parola con F0
simile aumenti musicalità; che il top-K diversificato migliori l'interesse;
che un cue onset sia una buona proxy di downbeat; che i risultati sui dataset
dei paper generalizzino a pop/rap/lingue e audio leciti del progetto.

## Vincoli di valutazione per il prossimo sprint

1. Congelare la baseline tecnica attuale come B0, con la dicitura **onset-grid
   heuristic** e con `vertical=unknown`.
2. Implementare un provider All-In-One o un adapter stub che fallisca in modo
   esplicito; non cambiare `kind=onset_grid` in `downbeat` senza output del
   provider.
3. Aggiungere cue strutturali a più finestre e matrici di costo per regioni,
   conservando breakdown per armonia, ritmo, loudness, timbro, boundary e
   vocalità quando disponibili.
4. Per ogni coppia creare A→B e B→A, più una randomità condizionata a cue
   validi; usare lo stesso renderer neutro per tutti.
5. Solo dopo recall/oracle e controlli di offset, eseguire ascolto umano cieco
   B0 vs baseline musicale vs proposta. Questo è il **Gate A umano**: è ancora
   non verificato.

