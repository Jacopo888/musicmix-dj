# Statistiche empiriche sulle transizioni nei mix DJ reali: hip-hop/rap contro l'insieme

Data: 2026-10-20. Lavoro in sola lettura sul codice: nessun file in `src/` modificato, nessun commit.
Script, risultati intermedi e tabelle: `runs/djmix_stats_v1/` (riproducibile con `bash runs/djmix_stats_v1/run_all.sh`, solo libreria standard di Python 3.13).
Etichette: **[M]** misura riprodotta dagli script su dati pubblici; **[P]** numero pubblicato dagli autori e ricontrollato sulle loro figure; **[H]** ipotesi di questa nota, non verificata.

---

## 1. Esito in breve

1. **Per il rap non esistono dati pubblici sulle transizioni.** Le tre risorse di Taejun Kim et al. non pubblicano allineamenti mix-traccia, durate di transizione, stretch, trasposizioni, posizioni dei cue o curve EQ/fader per mix singolo. Pubblicano solo metadati dei mix (`djmix-dataset`) e figure aggregate su tutti i generi (`djmix-analysis`). Ottenere quei numeri richiede di scaricare l'audio e rieseguire l'allineamento (`mix.align()`, `mix.transitions()`, `mix.cvxopt()` del pacchetto `djmix`): non fatto, perché la consegna escludeva l'audio.
2. **I mix hip-hop nel dataset sono pochissimi** [M]: 27 mix su 5.040 (0,54%) portano il tag MixesDB `Hip Hop` o `R&B`; 1.019 tracce su 102.932 (0,99%); 500 transizioni "disponibili" (con entrambe le tracce collegate a un video) su 64.764 (0,77%). Nelle figure del paper ISMIR 2020 (1001Tracklists) `Hip Hop` compare in 2 mix su 1.557 (0,13%) e in 72 tracce su 24.202 (0,30%) [P].
3. **Per genere si può misurare una sola cosa, ed è un proxy**: l'intervallo tra gli ingressi di due tracce consecutive nella tracklist (timestamp scritti dai compilatori di MixesDB, risoluzione di 1 minuto). Nei mix hip-hop tagged la mediana è **2,0 min** (IQR 2–3; 13 mix, 447 intervalli); nei mix elettronici di riferimento **4,0 min** (IQR 3–5; 1.684 mix, 25.516 intervalli). Non è la durata della transizione (§4).
4. Le statistiche di transizione che esistono (durata in battute, stretch, trasposizione) valgono per l'insieme dei mix, prevalentemente House/Trance, e vanno trattate come baseline non adatta al rap (§3).

---

## 2. Fonti, versioni e licenze

Verifica dei nomi e degli URL: i tre repository esistono con i nomi indicati, tutti pubblici, organizzazione `mir-aidj`. Clonati con `git clone --depth 1` (lettura anonima), `GIT_LFS_SKIP_SMUDGE=1`; nessun file audio scaricato. Hash dei file usati in `runs/djmix_stats_v1/sources.json`.

| Repository | Commit (data) | Cosa contiene di utile | Licenza |
|---|---|---|---|
| https://github.com/mir-aidj/djmix-dataset (pacchetto `djmix` 0.2.1) | `e32d179b8529156202566aabaa14bbd344a62d5b` (2022-09-07) | `dataset/djmix-dataset.json`, 11.595.801 byte, SHA-256 `5f585a1a…c6aec6b44`: 5.040 mix MixesDB con titolo, URL, fonte audio, tag di categoria (generi inclusi), tracklist con timestamp, conteggi. Codice per scaricare l'audio e calcolare allineamento, transizioni, EQ/fader. | **Nessuna dichiarata** (nessun file LICENSE, nessuna voce nella pagina GitHub, nessun cenno nel README). |
| https://github.com/mir-aidj/djmix-analysis | `a2ae903f038f90b9789e60e2c101b5017456f4d4` (2020-10-21) | Codice del paper ISMIR 2020, la tracklist di un solo mix (Palms Trax, house), e le figure SVG con i dati aggregati su 1.557 mix di 1001Tracklists. Nessun CSV/JSON di risultati; il dataset 1001Tracklists non è pubblico (ottenuto dagli autori per comunicazione personale). | **Nessuna dichiarata.** |
| https://github.com/mir-aidj/djmixer-estimation | `84a6aa6cc4cfc483e9260a481cfee6d7798f11f2` (2022-09-06) | Solo un README: "deprecated", rimanda al notebook `notes/tutorial.ipynb` di `djmix-dataset`. Nessun dato. | **Nessuna dichiarata.** |
| https://github.com/mir-aidj/transition-analysis (NIME 2021; ispezionato, clone poi cancellato perché contiene mp3 di demo) | `bc2ae4f1d345ab2edc0332b23afe67ad68aa07c1` (2021-06-15) | 10 transizioni di un solo mix house, curve JSON in `demo/dist/curve`. Nessuna statistica per genere. | Non verificata. |

Note su licenze e uso:
- Senza licenza esplicita i diritti restano degli autori [H: lettura prudente]. Qui si usano per analisi interna e si pubblicano solo numeri derivati: il JSON non è stato copiato nella repo di progetto.
- I metadati derivano da MixesDB e 1001Tracklists. La licenza dei contenuti di queste due fonti non è stata verificata.
- Il paper di riferimento per il dataset è Kim, Yang, Nam, DAFx 2022 (5.040 mix, 50.742 transizioni). Il paper ISMIR 2020 è Kim, Choi, Sacks, Yang, Nam (arXiv 2008.10267).
- Ricerca web per release di allineamenti (Zenodo o altro): nessun record trovato. Ho cercato una sola volta con una sola query: è un'assenza non provata in modo esaustivo.

---

## 3. Numeri sull'insieme dei mix (figure pubblicate, 1001Tracklists, House/Trance)

Le figure SVG di `djmix-analysis/img` contengono i dati degli istogrammi come rettangoli vettoriali. `digitize_svg.py` e `analyze_figures.py` leggono i rettangoli e le etichette degli assi. **Controllo di validità** [M]: la somma delle barre del tempo è 24.202 (= tracce giocate nel paper); la trasposizione somma 24.202 (23.602 + 566 + 25 + 3 + 3 + 3), con 600 tracce trasposte (2,48%) di cui 566 di un semitono (94,3%); le quote di stretch ricalcolate sono 86,5% / 94,4% / 98,6% entro 5/10/20% contro 86,1% / 94,5% / 98,6% riportati dal paper (differenza dovuta alla larghezza dei bin di 0,33%). I valori sono quindi affidabili per le conclusioni sotto, con l'errore di bin indicato.

Tutte le righe sono **[P]/[M] su tutti i generi insieme**; non c'è scomposizione per genere nelle figure.

| Grandezza | Valore | Fonte figura |
|---|---|---|
| Durata della transizione (battute, definizione degli autori: Fig. 6 del paper), n = 18.551 transizioni nel range 1–249 battute (su 20.765 totali) | mediana ≈ 64,5; quartili ≈ 33,6 e 104,9; 10° percentile ≈ 15; 90° ≈ 157,6 (centro di bin, bin da 3,1 battute) | `fig_cue_translength.svg` |
| Quota di transizioni brevi nello stesso istogramma | < 8 battute: 4,7%; ≤ 16: 10,8%; ≤ 32: 22,9%; ≤ 64: 49,2%; ≥ 128: 16,9% | idem |
| Picchi ai multipli di 32 battute (conteggio nel bin del multiplo rispetto alla media dei 4 bin vicini) | 32: 555 vs 441; 64: 762 vs 424; 96: 398 vs 247; 128: 282 vs 147; 160: 116 vs 101 | idem |
| Stesso, in secondi al tempo medio del dataset (127 BPM, dal paper) [calcolo] | 32 battute ≈ 15 s; mediana ≈ 30 s; tagli con meno di 8 battute ≈ meno di 4 s | derivato |
| Tempo adjustment (|variazione| rispetto all'originale), n = 24.202 | ≤ 1%: 47,3%; ≤ 2%: 65,2%; ≤ 5%: 86,5%; ≤ 8%: 92,3%; ≤ 10%: 94,4%; ≤ 20%: 98,6%. Segno: 41,1% negativo, 48,6% positivo (escluso lo 0 ±0,17%) | `fig_diff_bpm.svg` |
| Trasposizione | 23.602 non trasposte; 566 di 1 semitono; 25 di 2; 9 di 3 o più | `fig_diff_key.svg` |
| Distanza tra cue dello stesso brano scelti da DJ diversi (n = 48.289 coppie entro 64 battute) | zero battute: 27,4% del plottato (23,6% sul totale secondo il paper); ≤ 4 battute: 46,9% del plottato | `fig_cue_diff.svg` |
| Generi dei mix (multi-etichetta; 1.557 mix) | Electro House 862, Progressive House 787, House 698, Trance 189, Deep House 114, **Trap 90**, Tech House 82, Dubstep 51, Techno 20, … **Hip Hop 2**, Drum & Bass 2 | `fig_genre_mix.svg` |
| Generi delle tracce (24.202) | House 2.945, Electro House 1.874, Trance 1.634, … **Trap 137**, Pop 123, **Hip Hop 72**, **R&B / Funk 3**, **Reggae / Dancehall 3** | `fig_genre_track.svg` |

Cosa non c'è, nemmeno per l'insieme:
- **Posizione dei cue rispetto all'inizio e alla fine della traccia**: le figure pubblicate riportano solo l'accordo tra DJ diversi sullo stesso brano, non dove cadono i cue nel brano. Non calcolabile senza i cue estratti.
- **Frequenza di tagli netti contro blend lunghi** come classificazione: esiste solo l'istogramma delle durate (sopra). Un taglio netto in battute non è distinguibile da un blend breve con questi dati.
- **Stime EQ/fader**: il paper DAFx 2022 le calcola ma non rilascia le curve per mix.

Sul `Trap` della figura dei generi [H]: è l'etichetta di 1001Tracklists, che presumibilmente indica la trap da festival/EDM e non il rap. Non l'ho trattata come hip-hop.

---

## 4. Metadati MixesDB per genere (`djmix-dataset.json`)

Script: `analyze_metadata.py`. Gruppi definiti sui tag `Category:*` di MixesDB:
- **HH_tag**: tag `Hip Hop` o `R&B` → 27 mix. Esclusi per lettura manuale `Trap Mix` (è il nome di una serie di dBridge, drum & bass), `Hip House` (house) e `DJ Rap` (nome d'artista).
- **HH_tag_pure**: HH_tag senza alcun tag elettronico (House, Techno, Trance, Drum & Bass, Grime, Bassline, …) → 15 mix.
- **HH_tag_no_clubnight**: HH_tag senza il tag `Clubnight` → 19 mix.
- **ELEC_ref**: almeno un tag elettronico e nessun tag hip-hop → 4.829 mix.

Composizione di HH_tag [M]: 8 puntate della trasmissione radio *Clubnight* (DJ Kitsune, DJ Mem-Brain, Julian Smith, 2004–2006); 1 mix grime (KISS Grime); 18 altri mix con tag `Hip Hop`/`R&B` accanto a generi elettronici, jazz o funk (Boiler Room, Diplo, Peyo, Rinse FM, Resident Advisor, serie drum & bass e così via). Elenco in `hh_tagged_mixes.csv`. Il rap "puro" è quindi poco rappresentato anche dentro il gruppo.

### 4.1 Intervallo tra ingressi di tracce consecutive ("slot")

Calcolato sui timestamp tra parentesi quadre nei titoli delle tracce (`[mm]`, `[mm:ss]`, `[hh:mm]`); il formato di ogni mix è dedotto da una mediana plausibile degli intervalli (0,3–12 min); mix con più del 10% di intervalli negativi scartati. Risoluzione: 1 minuto per `[mm]` e `[hh:mm]`.

| Gruppo | Mix con timestamp usabili | Intervalli | Mediana (min) | IQR | IC 95% mediana (bootstrap per mix) | ≤ 1 min | ≤ 2 min | ≥ 6 min | Tracce/ora (mediana tra mix) |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| HH_tag | 13 | 447 | 2,0 | 2–3 | 2,0–3,0 | 15,7% | 50,1% | 2,2% | 22,9 |
| HH_tag_pure | 9 | 380 | 2,0 | 2–3 | 2,0–3,0 | 16,3% | 52,1% | 1,3% | 23,8 |
| HH_tag_no_clubnight | 5 | 71 | 2,97 | 2–3,7 | 2,0–4,0 | 12,7% | 40,8% | 7,0% | 21,1 |
| ELEC_ref | 1.684 | 25.516 | 4,0 | 3–5 | 4,0–4,0 | 2,1% | 12,7% | 16,1% | 14,5 |
| tutti i mix non hip-hop | 1.757 | 26.702 | 4,0 | 3–5 | 4,0–4,0 | 2,2% | 12,9% | 16,0% | 14,5 |

Altri conteggi [M]: tracce per mix 37,7 (HH_tag) contro 20,2 (ELEC_ref); quota di tracce identificate (con link) 69,8% contro 80,7%; transizioni disponibili per mix 18,5 contro 12,8.

Test di permutazione a livello di mix (5.000 permutazioni, seme 20261020; la p minima raggiungibile è 0,0002) sulla differenza delle mediane degli intervalli:
- HH_tag contro ELEC_ref: −2,0 min, p = 0,0002;
- HH_tag_pure contro ELEC_ref: −2,0 min, p = 0,0002;
- HH_tag_no_clubnight contro ELEC_ref: −1,03 min, p = 0,017.

Controllo con tracklist [H, euristica]: una lista di 79 artisti rap/R&B noti, cercata con confini di parola nel campo artista, trova **0** mix non taggati con almeno il 25% di tracce corrispondenti (8 mix oltre il 10%, 2 oltre il 15%; il massimo è 20%, un mix Hardwell). Nel dataset non ci sono quindi mix rap nascosti sotto altri tag, con la riserva che la lista è parziale.

### 4.2 Cosa significa e cosa no

**Misura** [M]: nei mix taggati hip-hop/R&B le tracce restano in tracklist circa metà del tempo rispetto ai mix elettronici (2 contro 4 minuti di mediana) e si cambia traccia circa 23 volte all'ora contro 14,5. La differenza è grande rispetto all'incertezza, ma poggia su 13 mix, di cui 8 sono puntate di una sola trasmissione radio (3 DJ, 2004–2006); senza di esse restano 5 mix e 71 intervalli.

**Ipotesi** [H], da non scambiare con misure:
- Gli slot più corti riflettono soprattutto la durata dei brani (3–4 minuti nel rap contro 6–8 nelle versioni club estese) e non la durata della transizione. Il dato non separa le due cose.
- Una mediana di 2 minuti con 16% di intervalli ≤ 1 minuto è compatibile con un uso frequente di tagli e blend brevi, ma anche con medley di spezzoni; non è stato misurato nessun taglio.
- La risoluzione di 1 minuto, il formato dei timestamp scritto a mano e la scelta di cosa un compilatore chiama "ingresso" della traccia aggiungono rumore non quantificato.
- Nulla qui permette di dire che un modello appreso su House/Trance valga o non valga per il rap: manca la misura, non c'è una prova contraria.

---

## 5. Limiti

1. **Nessun numero per genere su durata in battute/secondi, stretch, trasposizione, cue di inizio/fine traccia, tagli contro blend, EQ/fader**: i dati per questi non sono pubblicati. I mix hip-hop (27 in MixesDB; 2 nel dataset ISMIR 2020) restano pochi anche dopo l'allineamento audio: le transizioni utili sarebbero al massimo 500 (quelle "disponibili", con entrambe le tracce collegate), prima di scartare gli allineamenti falliti, con intervalli di confidenza larghi.
2. Le figure digitalizzate hanno bin di 0,33% (tempo) e 3,1 battute (durata). Le quote sono approssimate di circa ±0,5 punti percentuali e la definizione esatta di "durata di transizione" è quella degli autori (non riverificata sul loro codice qui).
3. Il tag di genere MixesDB è per mix, multi-etichetta, assegnato da volontari; un mix con `Hip Hop` può essere in gran parte altro (es. Boiler Room, drum & bass). HH_tag_pure riduce ma non elimina il problema.
4. Il gruppo di riferimento ELEC_ref e il gruppo hip-hop differiscono per epoca (hip-hop prevalentemente 2004–2006 e 2013–2020), fonte audio (8 dei 27 mix hip-hop vengono da `html_audio`, 30%, contro 4,5% in ELEC_ref) e tipo di evento (radio contro club/podcast). Nessun aggiustamento per questi fattori.
5. Il dataset 1001Tracklists del paper ISMIR 2020 non è pubblico: le figure sono l'unica traccia dei suoi risultati.

---

## 6. Cosa fare per avere numeri sul rap (proposte [H])

1. Per ottenere durata, stretch e cue servono audio e allineamento, quindi una decisione dell'utente su scaricare i 27 mix hip-hop con link (16 SoundCloud, 8 html_audio, 1 YouTube, 2 Mixcloud) ed eseguire `djmix` (pacchetto Python 3.9–3.10, madmom, cvxpy; incompatibile con il Python 3.13 della macchina di analisi senza un ambiente separato). Anche così N resta piccolo.
2. Fonte con più rap: raccogliere mix hip-hop/open-format da altre fonti (tracklist con timestamp da 1001Tracklists/Mixcloud) e applicare lo stesso allineamento, ma richiede audio e verifica dei termini d'uso.
3. Fino ad allora, le soglie per il rap nel renderer restano ipotesi di progetto, non numeri empirici. I valori di §3 sono una baseline dichiaratamente da House/Trance.

## 7. File prodotti

- `runs/djmix_stats_v1/fetch_sources.sh`, `run_all.sh`: recupero a commit fissati e riproduzione.
- `runs/djmix_stats_v1/digitize_svg.py`, `analyze_figures.py` → `figure_stats.json` (istogrammi digitalizzati e quote).
- `runs/djmix_stats_v1/analyze_metadata.py` → `metadata_stats.json`, `mix_table.csv` (mix con timestamp usabili), `hh_tagged_mixes.csv` (i 27 mix hip-hop/R&B).
- `runs/djmix_stats_v1/sources.json`: commit, date, hash dei file usati.
