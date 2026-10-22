# Set DJ continuo v1: cosa è implementato e da quale fonte

Data: 2026-10-20. Codice: `src/musicmix/dj_set.py` (puro, senza I/O), `tests/test_dj_set.py` (17 test sintetici), driver `tools/run_dj_set_v1.py`, run `runs/dj_set_v1/`. Fonti: `REVISIONE_SET_TRANSIZIONI_2026-10-20.md` (§1, §2, §4, §7, §8) e `REVISIONE_DJ_2026-10-20.md`. I moduli `dj_select`, `dj_render`, `dj_effects`, `dj_qa` sono usati solo tramite le loro funzioni pubbliche e non sono stati modificati.

Etichette: [E] empirico, [S] sistema pubblicato, [P] pratica, [T] teoria/rassegna, [H] ipotesi di progetto, **[U] dai giudizi utente v1** (`giudizi_cloud/dj_gallery_v1`: filter out 9, echo out 9, blend 8 battute 8, echo out 7, 7, drop diretto sul ritornello 3; un solo ascoltatore). I voti [U] entrano come priorità dichiarate, non come validazione: nessuna soglia è stata stimata da quei sei voti (AGENTS.md; Cawley & Talbot 2010).

## 1. Albero di decisione del tipo di transizione (livello transizione, Williams 2025)

Modello: albero esplicito e leggibile, come suggerito da Thalmann et al. 2018 [S, solo abstract] e Vande Veire 2018 [S]. La prima regola che si applica decide; ogni decisione salva la traccia delle condizioni valutate e le alternative ordinate da provare se il QA fallisce. Ingressi (da `dj_select`): stretch dopo la piegatura d'ottava, relazione Camelot e confidenza, voce di A nell'ultimo battito e nelle battute di sovrapposizione, voce di B nelle prime battute, sezione di B all'ingresso (allin1), confine di sezione all'uscita di A, energia relativa delle sezioni, confidenza delle griglie.

| Regola | Condizione | Uscita | Fonte |
|---|---|---|---|
| R1 | griglia di A o B a confidenza bassa | echo out | [H] |
| R2 | stretch > 8% | cut sul downbeat se A esce senza voce a confine di sezione e B parte strumentale a inizio sezione (non ritornello); altrimenti echo out | [S] MosaikBox ±8%; [E] Ishizaki 2009; [T] Williams 2025; [U] echo 9/7/7 |
| R3 | B entra su un ritornello | **mai drop secco**: filter su A di 2 battute e B sul downbeat del ritornello; echo out se A canta sull'ultimo battito | [U] drop 3/10; [E] Solberg & Dibben 2019 (EDM); [E] Brungart 2001; [S] Bittner 2017 |
| R4 | tonalità non stessa/±1/relativa, o non confidente | niente sovrapposizione: filter out (echo out se A canta sull'uscita) | [S] Vande Veire 2018 §4.2.1 (vincolo duro per i blend) |
| R5 | energia di B < energia di A − 6 dB (relativi) | echo out | [H] |
| R6 | voci sovrapposte già su 4 battute (clash > 1) | una voce alla volta: echo out / filter out a fine frase, mai blend | [E] Brungart 2001; [S] Vande Veire; [S] MosaikBox |
| R7 | A e B strumentali nella sovrapposizione (quota vocale ≤ 0,25) | blend 16 se stretch ≤ 5% e 16 battute strumentali, altrimenti blend 8 | [E] Kim 2020 §6.1, §6.3; [U] blend 8 = 8/10 |
| R8 | al massimo una voce alla volta (clash ≤ 1) | blend 8 se stretch ≤ 5%, altrimenti blend 4 | [S] Vande Veire (1 battuta tollerata); [H] |
| R9 | battute insufficienti | filter out | [U] filter 9/10 |

Parametri in `dj_set.TREE_PARAMS` (valore + fonte), costo a priori dei tipi in `dj_set.TYPE_COST` (filter 0,10, echo 0,20, blend 8 0,20 [U]; blend 4/16, filter verso ritornello 0,30, cut 0,50 [H, non ancora ascoltati]). Nel driver si aggiungono due vincoli già usati nelle gallerie: uscita di A in dissolvenza = cue scartato (galleria v1) e guardia vocale dei cut (stem + parole ASR, galleria v2: un cut con voce sull'uscita diventa l'alternativa con effetto).

## 2. Ordine del set (livello traccia)

- Cammino minimo **esatto** di Held–Karp su sottoinsiemi (`held_karp_path`): stato = insieme visitato + ultimo brano, la posizione nel set è la cardinalità dell'insieme, quindi anche l'arco di energia per posizione resta esatto. Fino a 15 candidati; oltre si preselezionano i 15 con gli archi migliori [H]. Lunghezza scelta = la più vicina all'obiettivo (10) tra 8 e 15; se nessuna da 8 in su è ammissibile, il set più lungo possibile (annotato). Bittner 2017 [S] usa il greedy (HAM-1/HAM-2) che lascia accoppiamenti cattivi alle estremità; qui il problema è piccolo e si risolve esattamente.
- Costo d'arco: tempo in scala log2 con equivalenza d'ottava (70 ↔ 140, Bittner 2017 [S]), (s/8%)² più un gradino oltre ±10% (Ishizaki 2009 [E]), vietato oltre 20% [H; Kim 2020: 98,6% entro 20%]; tonalità Camelot pesata 1 se il miglior tipo è un blend, 0,4 altrimenti (Kell & Tzanetakis 2013 [E]: la tonalità non è criterio primario nell'ordine; Bittner 2017 [E]: key clash 0%); timbro = distanza tra MFCC (media e deviazione, 40 dimensioni) standardizzati sul pool (Kell & Tzanetakis 2013 [E]: ordine reale più coerente del rimescolato; Bittner 2017 [S]); costo del miglior cue (`cue_score` di dj_select) e del tipo deciso dall'albero; salto di energia (peso piccolo: alternanza locale ammessa, Neto 2024).
- Costo di nodo: scarto quadratico dall'arco di energia richiesto (0,35 → picco 0,95 al 70% del set → 0,5), parametrizzabile (Cliff 2000 [S] come interfaccia "traiettoria richiesta"; la forma è [H]). Energia = media degli z-score di densità di onset, flusso spettrale e centroide sull'audio a −14 LUFS, poi rango percentile nel pool [H: non è una misura validata di "energia"].
- Non implementato: "passi regolari" di Kell & Tzanetakis (richiede uno stato del secondo ordine), primo/ultimo brano fissati sono opzioni (`--start/--end`).

## 3. Cue lungo il set e render (livello mix)

- Per ogni coppia consecutiva tutti i piani di `dj_select.plan_pair` (cue su downbeat di inizio frase), ciascuno con la decisione dell'albero. Viterbi esatto (`choose_cues`) con: durata di ogni brano nel set 60–120 s (dall'ingresso all'uscita successiva; primo brano dall'inizio se rientra, ultimo fino alla fine se rientra, altrimenti su un downbeat di frase con fade di 2 battute [H]); spazio minimo di 8 s tra la fine di una transizione e l'inizio della successiva [H]; penalità 0,15 per lo stesso tipo due volte di fila (Williams 2025 [T], che cita Hawkins 2013, tesi non letta: evidenza debole).
- Render: blend con `dj_render.render_transition` (convenzione "start", stretch Rubber Band, EQ e bass swap, ducking), cut con `dj_render`, echo/filter con `dj_effects`; section gain match di `dj_render` attivo con rampa di 8 battute [H] che il driver prolunga sull'audio intero dopo la clip. QA con `dj_qa.evaluate_render` (deck, stem, parole ASR, `grid_support`); se `fail`: alternative dell'albero sullo stesso cue, poi altri cue compatibili con i vicini (al massimo 10 tentativi), tutti registrati in `set.json` e `transitions/*.json`.
- Giunte: fra due clip il brano suona dall'audio intero normalizzato con lo stesso guadagno LUFS usato dal renderer; crossfade di 10 ms tra campioni equivalenti; il controllo `junctions` di `set.json` misura il salto di loudness 3 s prima/dopo ogni giunta.
- **Limite noto [H]**: dopo un blend B torna al tempo originale sul primo downbeat dopo la transizione (salto pari allo stretch, ≤ 8%, riportato in `tempo_reset_pct`); un DJ terrebbe il pitch o lo riporterebbe lentamente.

## 4. Prima esecuzione (`runs/dj_set_v1`)

Brani disponibili al momento dell'esecuzione: 17 = 7 del pool V12 (`runs/dj_plan_v1`) + 3 CC già pianificati (`runs/dj_plan_cc_v1`) + 7 CC con struttura, voce e provider completi, analizzati con `dj_plan_pool.analyse_track` in `.cache/dj_set_v1/tracks` (l'analisi CC in corso non è stata toccata). Coppie ordinate ammissibili (salto di tempo ≤ 20% e almeno un cue non in dissolvenza) 176 su 272; solo 8 hanno almeno un cue con blend secondo l'albero: le coppie "compatibili" per un blend sono poche (tonalità e voci), ma echo/filter/cut permettono comunque un set di 10.

- **Ordine** (10 brani, costo 14,83; costi per L = 8…15 in `set.json`): Heartless → Coehn&Foehrb "Alles Coehn und gut" → Mit-Rich "Assurance" → TIMES REAL "Scelta" → Too Many Nights → K.Os Rhymes "Der Lauf der Dinge" → KAMENOLOM → Vaina Dura "Emperador del Rap" → Afrocream → TSF "Making my own way". Tempi tra 82,5 e 98 BPM, salti consecutivi ≤ 4,5%.
- **Durata**: 15:12 (912 s), 71–112 s per brano; true peak −1,18 dBTP; giunte verificate campione per campione (identiche alla sorgente normalizzata; guadagno continuo attraverso la rampa del section gain).
- **Transizioni finali**: 1 pass, 6 warn, 2 fail tenuti (tutti i 10 tentativi falliti; tenuto il meno grave). Tipi: 3 echo out, 3 filter out, 2 filter verso il ritornello (R3), 1 blend 4. Tentativi in totale 49: 1 pass, 6 warn, 42 fail.
- **Causa dominante dei fail**: il controllo `loudness_jump` di `dj_qa` confronta i 3 s *prima dello switch*, che per echo/filter sono dentro l'effetto (A filtrato o coda d'eco, più bassi per costruzione), con i 3 s di B dopo lo switch: salti di +3…+7 dB. Il confronto A asciutto → B (`transition_edges`) resta entro ±2 dB in tutte le transizioni tenute. Per echo e filter out, che sono i tipi meglio votati in v1, il controllo misura quindi l'effetto stesso più che un salto di livello tra i brani. Va deciso se escludere la regione dell'effetto dalla finestra "prima"; la decisione spetta a chi mantiene `dj_qa`, qui non è stata cambiata. Gli altri fail sono di ritmo (fase di Beat This! 15–36 ms) nei blend 4/8 e nei filtri con griglie a confidenza media.

## 5. Cosa resta ignoto

- Il set non è stato ascoltato: un `pass` del QA è integrità tecnica, non qualità musicale.
- L'albero non è stato confrontato con alternative su clip tenute da parte; i pesi del costo di cammino sono [H].
- Nessuna statistica su tipi di transizione o curve di energia in set rap reali (lacuna confermata dalla revisione): l'arco e i costi dei tipi andrebbero tarati con confronti A/B forzati (Mantiuk 2012) su set interi.
