# Renderer, loudness, tempo e stem

## Audit integrale dei nove libri — 21 ottobre 2026

Probe locale Windows: fallback FFmpeg af_rubberband con time_ratio1.05 trasforma un seno82Hz in circa78Hz; durata4→4,2s corretta. Difetto riprodotto direttamente, senza planner. Il fail del bass test dipende dalla reference82Hz dopo lo spostamento; downbeat test presenta onset mancanti/riusati e non prova drift uniforme. Qualificare il backend per pitch/durata/transienti; non adattare le soglie per far passare il difetto. Separare trasformazioni artistiche intenzionali descritte da Williams e contratto tecnico di pitch invariato. Questi risultati non descrivono automaticamente la macchina Linux Linux/RubberBandR3.

Dettagli, fonti/pagine e criteri di accettazione: [revisione completa](../libri/audit_2026-10-21/REVISIONE_PIPELINE.md).

## Evidenza misurata

La ricerca su **Daito Manabe (2026)** propone diagnostiche interpretabili per deck A, deck B e master: loudness/headroom, continuità spettrale, smoothness del gain e stabilità stereo. Il principio applicabile è separare difetti tecnici: un singolo numero non identifica la qualità musicale.

**MosaikBox (Sowula e Knees, 2024)** studia modifica di stem e allineamento stretto al beat, con valutazioni soggettive. È una dimostrazione che un renderer può modificare componenti e metrica, ma non offre etichette di closure rap né una regola universale di crossfade.

**UnmixDB/CMMR** aggiunge un controllo costruito: se le tracce note vengono rimosse dal mix, il calo RMS e l'errore di allineamento possono rivelare un renderer rotto. Questo non deve essere confuso con preferenza, continuità o suspense.

## Ipotesi ingegneristica

Per ogni render archiviare almeno: picco e loudness per deck/master, headroom e clipping, differenza RMS al seam, continuità della curva di gain, salto spettrale, stereo e presenza di silenzio anomalo, oltre a tempo/beat alignment. Ogni flag deve indicare la componente e non una decisione estetica aggregata. Applicare controlli pre-ascolto a tutti i candidati e scartare solo difetti tecnici espliciti; la qualità musicale resta un asse separato.

## Limiti di dominio

Stem e beat-grid possono migliorare l'esecuzione di una transizione, ma non creano una frase o un evento atteso. Il tempo non corretto o una sorgente non separata può rendere non applicabile la metrica di rimozione. La musica rap può usare microtiming e non-allineamento intenzionale: correggerli automaticamente rischia di eliminare informazione stilistica.


## Implementazione collegata: `src/musicmix/dj_render.py` (20/10/2026)

Renderer DJ standard (P0 di `_meta/REVISIONE_DJ_2026-10-20.md`), non usa i renderer precedenti.

- **[S] MosaikBox**: decks normalizzati a −14 LUFS integrati (pyloudnorm, BS.1770-4), stretch massimo ±8%, attenuazione della voce uscente (stem Demucs) se le voci si sovrappongono per più di 2 s.
- **[S] Vande Veire 2018 §4.3**: B entra senza bassi, swap dei bassi sul downbeat centrale, mai due bassi pieni.
- **[E]/[S] Kim 2020 §4.1**: convenzione dei cue (cue-out = inizio transizione in A, cue-in = fine transizione in B); Kim DAFx 2022: fader + EQ a 3 bande monotoni. Le forme delle curve (seno/coseno, alti da −12 dB) sono **[H]**, non stimate da mix reali.
- Beatmatching solo su B con Rubber Band R3 (CLI `rubberband-cli`, time map per battuta) e intonazione invariata; fallback ffmpeg `af_rubberband` a rapporto costante (R2 real-time, meno preciso).
- Limite misurato: l'errore "≤10 ms" è rispetto alle griglie Beat This! (frame da 20 ms). Su MAFIA → After Hours la griglia di B nell'intro senza batteria devia fino a 38 ms da una retta, quindi l'aggancio alla griglia non garantisce allineamento udibile; la misura inter-deck senza griglie (cross-correlazione di flusso spettrale) non ha trovato contenuto ritmico comune (xcorr < 0,5). Prova: `runs/dj_render_smoke/`.

## Implementazione collegata: `src/musicmix/dj_effects.py` (20/10/2026)

Transizioni tipiche del DJ hip-hop/open-format, costruite sopra le funzioni pubbliche di `dj_render` (normalizzazione −14 LUFS, limitatore true-peak −1 dBTP, `split_bands`, `choose_tempo_fold`, `render_transition(kind="cut")`) e con metadati dello stesso tipo più un blocco `effect`. Test sintetici: `tests/test_dj_effects.py`.

**Statuto dell'evidenza: [H], ipotesi di pratica DJ, non risultati empirici.** Nessuno studio empirico su durata, tipo o posizione delle transizioni nei mix hip-hop è stato trovato (`_meta/REVISIONE_DJ_2026-10-20.md` §3, ultima riga, e §6). Tempi di delay, feedback, frequenze e curve sono scelte di progetto plausibili, non stimate da mix reali né validate da ascolto.

- **`echo_out`** [H]: sul downbeat di uscita il segnale di A entra in un delay di 1/2 o 1 battito della griglia di A (un periodo catturato, poi il dry di A è tagliato); feedback 0,5 (−6 dB a ripetizione), passa-alto nel loop da 150 a 2500 Hz crescente (zero-phase, periodo esatto); B entra pulito sul suo downbeat dopo 0, 1 o 2 battute di A, senza stretch; la coda sotto B è attenuata di 12 dB. Idea pratica: l'eco "chiude" A senza sovrapporre due groove, quindi funziona anche con tempi incompatibili per un blend.
- **`filter_out`** [H]: passa-alto Butterworth del 4° ordine in sweep geometrico 30→1200 Hz per 1–4 battute fino al downbeat di uscita, poi taglio sul downbeat con B; opzione `filter_in` (passa-basso su B che si apre da 250 Hz a 18 kHz in 1–4 battute di B). Filtri causali a blocchi, come quelli di un mixer.
- **`drop_in`** [H]: B entra sul primo downbeat del ritornello/hook al posto di A (cut di `dj_render`); che `cue_in_b` sia inizio sezione è garantito dal chiamante e solo registrato, non verificato.
- Clip: 10 s prima dell'inizio dell'effetto (downbeat di uscita, o inizio dello sweep) + effetto + 10 s dopo l'ingresso di B; per il drop 10 + 10 s.

Verifiche sintetiche (click track, non giudizi di qualità): periodo dell'eco = battito di A al campione; −6 dB per ripetizione; passa-alto che riduce l'energia < 200 Hz di A di oltre 20 dB a fine sweep; B che inizia esattamente al campione atteso; LUFS dei deck a −14 e true peak ≤ −1 dBTP; durate delle clip.

Prova reale `runs/dj_effects_smoke/` (MAFIA → After Hours, cue scelti a mano solo per la prova): render tecnicamente corretti, ma il salto di loudness misurato è +3…+11 dB perché il punto di uscita scelto in MAFIA (downbeat 144) è nella coda in dissolvenza dell'outro (−28 LUFS nei 4 s prima dell'effetto). È un limite della scelta del cue, non del renderer: un selettore deve evitare di uscire su un fade dell'originale o compensarlo. Il cue del drop (After Hours, downbeat 66) è un confine di segmento euristico, **non** un hook verificato.

## Implementazione collegata: section gain match, guardia vocale dei cut, fase del tracker (20/10/2026, galleria v2)

Motivo misurato (galleria `runs/dj_gallery_v1`, congelata): 12 fail su 18 render, di cui 9 per salto di loudness > 3 dB, 4 per parola di A troncata nei cut, e falsi positivi probabili del ritmo su MAFIA (Beat This! sui controtempi: +268 ms ≈ mezzo battito a 110 BPM, mentre allin1 e madmom coincidono con la griglia del render su tutti i beat).

- **Section gain match** (`dj_render.SectionGainConfig`, parametro `section_gain` comune a `DJRenderConfig` e `DJEffectsConfig`, spento per default) **[H]**: loudness BS.1770 a blocchi short-term di 3 s, 75% di sovrapposizione, gate −70 LUFS / −10 LU **[S per la misura]**, di A nelle ultime 2 battute prima dell'inizio della transizione (A ancora dry) e di B nelle prime 4 battute dopo lo switch (per il blend sulla sorgente B già stretchata, in coordinate dell'uscita); guadagno statico su B = L_A − L_B limitato a ±6 dB. **Scelta documentata**: dopo la finestra di B, e mai prima della fine della transizione, B torna al proprio livello normalizzato (−14 LUFS sull'intero brano) con una rampa a coseno rialzato in dB di 16 battute (≤ 0,4 dB/battuta con 6 dB): preferita al mantenimento perché un trim lasciato su B sposterebbe il resto del brano; nella clip da 10 s la rampa è appena iniziata (metadati: `section_gain.ramp`, `gain_db_at_clip_end`). Il limitatore true-peak −1 dBTP resta l'ultimo stadio. Ampiezze 2/4/16 battute fissate a priori, non stimate da mix reali.
- **Guardia vocale dei cut** **[H]**: `dj_select.plan_transition(..., cut_vocal_guard=True)` (copertura vocale ≤ 0,1 dell'ultimo battito, altrimenti `echo_out`), `dj_select.relocate_cut` (downbeat di inizio frase valido più vicino entro 16 battute, prima il precedente) e, nel driver v2, una prova sullo stem Demucs (RMS 20 ms sopra la soglia dello stem, come dj_qa) nell'ultimo battito e nei 150 ms dopo il cut, più nessuna parola ASR entro ±150 ms. Senza cue libero si ripiega su echo_out o filter_out (esenti).
- **Ambiguità di fase del tracker** (`dj_qa`, `grid_support`) **[H]**: se l'errore è a mezzo battito entro la soglia del ritmo (15 ms) e per entrambi i deck allin1 o madmom (indipendenti dalla griglia usata) coincidono con la griglia del render su ≥ 80% dei beat, il fail diventa warn "possibile ambiguità di fase del tracker". Non dimostra che il beatmatch sia giusto: rende esplicito che la misura non lo smentisce.

Esito su `runs/dj_gallery_v2` (stessi 6 piani, `tools/run_dj_gallery_v2.py`): 3 pass / 8 warn / 6 fail su 17 (v1: 4 / 2 / 12 su 18). I salti tra A-solo prima e B-solo dopo la transizione (la grandezza che il gain allinea) scendono da max 4,8 dB a max 2,4 dB. Restano 5 fail di loudness tutti sulla finestra attorno allo switch, che il metodo richiesto non misura: nei blend A cala già dentro la sovrapposizione (g01 Creature b.111: −19,6 LUFS prima dello switch contro −13,6 prima della transizione; g03 After Hours b.120: −17,5 contro −11,6), negli effetti la finestra prima dello switch contiene A filtrato o la coda d'eco. Il gain su B (fino a +4,3 dB) fa lavorare il limitatore fino a 4,7 dB di riduzione (g03 blend 4) e porta g06 blend 4 a un nuovo fail di scontro vocale (1,22 s > 1 s: il QA ora conta il gain di B nell'udibilità). La guardia vocale elimina i 4 fail per parola troncata, ma con lo stem a soglia p95 − 24 dB quasi nessun downbeat di inizio frase è libero nel rap denso: un solo cue spostato (drop su After Hours, A da b.95 a b.99, pass), quattro ripieghi su filter_out, un cut tolto (g04, echo e filter già presenti). Nessun ascolto di verifica: integrità tecnica, non qualità musicale.
