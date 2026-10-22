# Imparare dai set DJ di riferimento: allineamento mix → brani e prior (21/10/2026)

Obiettivo (utente): dare al sistema una direzione musicale senza voti continui, imparando
**distribuzioni di scelte** da mix reali (dove si esce da A, dove si entra in B, quanto dura la
sovrapposizione, che tecnica, se la base resta continua). Priorità dell'orchestratore: dataset
pubblici con audio legale; i mix indicati dall'utente sono un'aggiunta facoltativa.
Etichette: **[M]** misura nostra; **[P]** numero pubblicato; **[S]** sistema/codice altrui;
**[H]** nostra ipotesi o soglia.

## 1. Fonti e riuso (dettagli e hash in `03_struttura_transizioni_dj/SOURCES.json`)

| Fonte | Revisione / file | Licenza | Uso |
|---|---|---|---|
| Kim et al. 2020, ISMIR (`texts/kim2020_djmix.txt`) | — | articolo | metodo |
| `mir-aidj/djmix-dataset` (pacchetto `djmix`) | e32d179 (2022-09-07) | **nessuna dichiarata** | metodo letto e **reimplementato**, codice non copiato |
| `mir-aidj/djmix-analysis` | a2ae903 | nessuna | figure già digitalizzate (`DJMIX_STATS_HIPHOP_2026-10-20.md`) |
| UnmixDB (Zenodo 1422385) | set222, 85 MB | CC BY-NC-ND | **validazione** (solo ricerca; scaricato, feature estratte, file cancellati) |
| `Ircam-RnD/unmixdb-creation` | abc606a | MIT | semantica delle etichette di verità |
| Raveform (HF `taejunkim/raveform`) | raveform.zip 479 MB, sha256 10c97fa9… | CC BY 4.0 | **prior generici** (solo metadati: niente audio) |
| Mixotic (Sonnleitner et al. 2016, JKU) | — | CC | **non accessibile**: i link Google Drive chiedono il login (21/10/2026); 9,4 GB di brani comunque fuori budget |
| Kim, Yang, Nam DAFx 2022 | solo metadati | — | modello somma di spettri con fader+EQ (letto nel codice) |

Ricerca di un dataset hip-hop/rap con audio legale e verità di transizione: **nessuno trovato**
(due ricerche web, Zenodo, HF; Raveform contiene 24 mix con tag MixesDB "Hip Hop", solo metadati,
spesso misti con elettronica: 8 puntate Clubnight 2004–2006, Diplo, Boiler Room, KISS Grime…).

## 2. Metodo (`src/musicmix/mix_alignment.py`, schema `musicmix.reference-mix.1`)

1. **Battiti**: mix con Beat This! (stesso checkpoint del QA, `dj_qa.BeatThisTracker`); brani di
   libreria con la griglia della loro scheda (`tempo.phase_s + k·period_s`), così battute e
   allineamento usano la stessa griglia e non serve rifare il beat tracking dei brani.
2. **Allineamento** [S, Kim 2020 reimplementato]: croma + MFCC 1–19 medi per mezzo battito,
   standardizzati con media/deviazione del brano, coseno, DTW di sottosequenza, tratti diagonali
   con 2 punti perdonati, linea principale = tratto più lungo. Aggiunte [H], tarate sul v3
   (sviluppo): MFCC 0 escluso (fader); multipli di tempo 1/2/½ affinando sempre la griglia più
   grossa; penalità 0,25 sui passi non diagonali (senza, la diagonale si interrompeva in anticipo
   dove il brano si ripete); taglio del tratto diagonale al pezzo più lungo con costo ≤ 20°
   percentile + 0,2; brano "trovato" solo se costo medio ≤ 0,45 (veri 0,01–0,18, falsi 0,75–0,85
   sul v3); trasposizioni del croma facoltative (±1 nelle prove degradate, Kim: 12).
3. **Mappa temporale**: ancore sui punti diagonali + estensione sulla stessa linea fino a 32
   battiti (come `extend_wp`).
4. **Curve di guadagno** [S, Kim 2022 semplificato]: mix ≈ Σ g_k · spettro mel del brano k riportato
   sul tempo del mix; NNLS per frame (46 ms, ±1 frame) e per banda (< 200 Hz, 200–2500, ≥ 2500);
   dB rispetto alla mediana nel tratto diagonale. Niente vincoli di monotonia (cvxpy assente).
5. **Transizioni** [H]: presenza = max(banda intera, medi, alti); udibilità con isteresi
   (−12 / −22 dB, tratti ≥ 1 battito); ingresso di B, fine di A, switch (prima dominanza di B),
   sovrapposizione in battiti; **zona d'effetto** = da fine diagonale di A all'ingresso di B (sul v3
   cade sull'inizio di eco/filtro/build/loop). Tecnica: `blend` (B udibile ≥ 1 battito prima che A
   smetta di suonare pulito e sovrapposizione ≥ 2 battiti), `echo` (zona che cala ≤ −12 dB e decade
   ≥ 3 dB/battito), `filter` (bassi o alti di A giù ≥ 6 dB relativi: **filtro o build, non
   distinti**), `loop` (materiale simile ad A fuori diagonale), `cut`. Brano mancante: zona fra
   le diagonali > 64 battiti con residuo > 0,8 → coppia `adjacent: false`, tecnica `unknown`.
6. **Posizioni nel brano** (`track_position`): battuta, downbeat più vicino, battito dal downbeat,
   frase (battuta nella sezione mod 8/16), sezione, sezione precedente, battute alla fine della
   sezione, **battute al prossimo ritornello**, battute dall'ultimo ritornello, frazione del brano.

## 3. Validazione senza dati esterni (`tools/run_reference_mix_validation.py`; risultati in `runs/reference_mix_v1/VALIDATION_SUMMARY.json`)

Verità: segmenti grezzi e tempi delle transizioni dei nostri `set.json` (campioni esatti), stretch
di B nella clip, punto di giunzione; UnmixDB `.labels.txt`. Il v3 è il set di **sviluppo** (le
soglie [H] sono state scelte guardandolo); v1b e v2 erano previsti come set tenuti da parte, ma
**6–7 brani su 10 di ciascuno non hanno più l'audio sulla macchina di analisi** (CC in lingua straniera cancellati
dall'utente il 21/10): sono diventati la prova "brani mancanti". Distrattori: tutti i brani del
catalogo non suonati (v3: 4; UnmixDB: 2 estratti per mix).

| Prova | Ordine / trovati / falsi positivi | Mappa nei tratti a brano solo | Switch (battiti, mediana / p90) | Ingresso di B nel brano (battiti) | Battuta d'ingresso esatta | Tempo di B (errore %) | Blend sì/no |
|---|---|---|---|---|---|---|---|
| v3 (sviluppo, 14 brani, 13 transizioni) | esatto, 14/14, 0 | mediana 15,7 ms; 94 % ≤ 50 ms | 0,18 / 3,4 | mediana 0,22 | 9/13 | mediana 0,07 | 12/13 |
| v3 mp3 128 kbps | esatto, 14/14, 0 | 15,4 ms; 91 % | 0,18 / 3,4 | 1,74 | 7/13 | 0,06 | 12/13 |
| v3 +3 % velocità (tempo e +0,5 st), 128 kbps | esatto, 14/14, 0 | 13,6 ms; 90 % | 1,15 / 8,5 | 0,45 | 6/11 | 0,12 | 11/13 |
| v3 +1 semitono (rubberband), 128 kbps | esatto, 14/14, 0 | 10,8 ms; 90 % | 1,84 / 6,0 | 0,24 | 6/11 | 0,07 | 11/13 |
| v1b + v2 (audio presente per 4/10 e 3/10) | i 7 brani presenti trovati, 0 falsi; tutte le 4 coppie separate da brani mancanti segnate `unknown` | 16,8 / 16,2 ms | 1 coppia adiacente: 0,33 | 0,36 | 1/1 | 0,02 | 1/1 |
| UnmixDB set222 (24 mix sintetici CC, 6 varianti) | 18/24 esatti, 1 falso positivo, 40/48 coppie | 20–37 ms per mix | 4,2 / 7,9 | (s) 1,9 s | — | mediana 0,05 | 12/40 |

Altre misure sul v3: fine di A come continuazione mediana 1,9 battiti di errore; inizio della
zona d'effetto 1,1; mappa di B allo switch vero 0,02 battiti (p90 1,85); tecnica esatta 7/13
(eco 5/5; filtro/build 2/6, gli altri chiamati cut o eco; il loop del "halving build" chiamato
eco; il blend di 4 battute con B stirato −7 % chiamato cut). Sovrapposizione: errore mediano 3,1
battiti; **sottostimata** nei blend lunghi (v3: 3,4 battiti stimati su 16; UnmixDB: 5,8 su ~15,
B udibile in media 2,3 battiti tardi e A 5,9 battiti prima): le dissolvenze lineari nella parte
bassa (sotto −12 dB) e l'NNLS sugli spettri sommati perdono le code. **Base continua** (B
udibile prima che A sparisca, sovrapposizione ≥ −0,5 battiti): riconosciuta in 34/40 blend di
UnmixDB (85 %).

Lettura: ordine, identità, mappa temporale, tempo e battuta d'ingresso di B sono affidabili e
robusti a mp3/velocità/trasposizione; la **durata** della sovrapposizione e la **tecnica** sono
diagnostiche deboli (validate su 13 transizioni di sviluppo e 40 sintetiche, non su DJ reali).

## 4. Statistiche e prior (`tools/reference_mix_stats.py`, `src/musicmix/reference_priors.py`)

Formato `musicmix.reference-priors.1`: domini con peso e distribuzioni (tecnica, sovrapposizione
complessiva e per tecnica, battuta nella frase all'ingresso/uscita, distanza dal downbeat,
sezione d'ingresso/uscita, sezione precedente all'uscita, battute a fine sezione, battute al
prossimo ritornello di B, base continua, frazione del brano, cambio di tempo, battiti suonati, e
per Raveform il drop di B rispetto all'uscita di A). Costo additivo
`-Σ_d w_d·log(p_d(x)/u_d)` con Laplace (α = 1); valori fuori vocabolario del dominio non costano
(le sezioni EDM non vengono equiparate a strofa/ritornello). `load_reference_priors()` legge
`configs/reference_priors.json`: **il file non è stato creato**, quindi il default resta
invariato. Candidato pronto: `runs/reference_mix_v1/reference_priors_raveform_only.json`
(domini elettronico w = 0,15 e "hip hop taggato" w = 0,3); `dj_set`/`dj_select` non toccati.

Numeri Raveform [M] (mix point = estremi del tratto allineato, non inizio udibile):

- **Elettronica** (4.887 mix; 29.883 coppie con tratti sovrapposti, 24.892 senza escluse):
  sovrapposizione mediana 67 battiti (p25 32, p75 120) — coerente con i ~64 di Kim 2020 [P];
  cambio di tempo mediano +0,18 % (p10 −2,3, p90 +3,3); brano suonato mediano 511 battiti.
  Su 2.629 ingressi in brani annotati: intro 52 %, buildup 25 %, drop 10 %, breakdown 8 %;
  battuta 0 della frase da 16 nel 15 % (battuta 8 nel 9 %, picchi a 7/15: frasi da 8 ±1).
  Uscite: outro 34 %, drop 29 %, cooldown 23 %, breakdown 8 %; l'uscita cade a 8 battute
  mediane dalla fine della sezione. **Il primo drop di B cade in mediana 42 battiti dopo
  l'uscita di A** (solo il 5 % entro ±4 battiti; nel 28 % prima che A esca). La distanza dal
  downbeat è trimodale (−1/0/+1 battito): sfasamento di indice fra battiti stimati e annotati,
  da non usare.
- **Mix con tag "Hip Hop"** (24 mix, 122 coppie con sovrapposizione, 239 senza): sovrapposizione
  mediana 22,5 battiti (p25 7, p75 64), brano suonato mediano 248 battiti (metà dell'elettronica),
  cambio di tempo mediano +1,2 % (p10 −3,4, p90 +6,7). Le 239 coppie senza sovrapposizione dei
  tratti allineati non dicono che la base si interrompe: con effetti/EQ il tratto "pulito" finisce
  prima, quindi la frequenza della base continua **non è misurabile** dai mix point di Raveform.

Per le domande dei voti ciechi dell'utente ("A carica la tensione, B risolve"; "la base non deve
interrompersi") i descrittori sono pronti (`exit_prev_section`, `exit_bars_to_section_end`,
`entry_bars_to_chorus`, `base_continuous`, `b_drop_vs_a_exit_beats`), ma nel rap non ci sono
ancora dati: l'unico riscontro è elettronico e va letto come baseline di un altro dominio.

## 5. Cosa vale per il rap e cosa no

- Vale (misurato sui nostri set rap/trap renderizzati): allineamento, ordine, identità, tempo,
  cue d'ingresso di B in battute, riconoscimento dei brani mancanti, base continua sì/no.
- Non vale per il rap: tutte le distribuzioni Raveform (durate, sezioni EDM, posizione del drop);
  la tecnica stimata (classificatore [H] tarato su 13 transizioni nostre); le soglie [H].
- Ignoto: come escono/entrano i DJ rap reali rispetto a strofe, ritornelli e build; servono mix
  rap veri (facoltativi, §6).

## 6. Cosa serve dall'utente (facoltativo)

Testo pronto: «Se vuoi, indicami 3–5 mix rap/trap di DJ che ti piacciono, da 20 a 60 minuti, con
la tracklist (artista – titolo in ordine; i minutaggi aiutano ma non servono). Formato: un unico
file continuo WAV/FLAC o MP3 ≥ 192 kbps (128 kbps regge nelle prove). Mettili in una cartella locale
e importali con `musicmix ingest <cartella>`.
Servono in libreria anche i brani della tracklist, nella stessa versione suonata (album, edit,
clean/explicit): importali dalla tua libreria con `musicmix ingest`. Bastano
il 70–80 % dei brani: ogni brano mancante toglie le sue due transizioni (la coppia attorno viene
segnata come non adiacente, non inventata); con N brani e k mancanti non consecutivi restano
circa N−1−2k transizioni utili. Niente voti: il mix viene analizzato in automatico.»

Il progetto non scarica nulla da YouTube/SoundCloud/siti terzi: solo file locali dell'utente.

## 7. Uso

```
python tools/analyze_reference_mix.py MIX.mp3 --tracklist tracklist.json --out runs/reference_mix_v1/<nome>.json
python tools/reference_mix_stats.py --mix rap_user:1.0:runs/reference_mix_v1/<nome>.json \
    --raveform raveform.zip --out runs/reference_mix_v1/reference_priors.json
# attivazione esplicita nel pianificatore: copiare in configs/reference_priors.json
```

Costi: Beat This! sul mix ≈ 3 min per 17 min di audio a macchina libera (13 min con 4 core
contesi); feature in cache (`runs/reference_mix_v1/cache`, ignorata da git, rigenerabile).
