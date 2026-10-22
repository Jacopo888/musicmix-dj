# Correzioni strutturate, approvazione del set, torneo dei cue, passaggi di riferimento e accordo (21/10/2026)

Origine: la revisione esterna condivisa dall'utente ([`REVISIONE_ANDAMENTO_ASTRA_2026-10-21.md`](REVISIONE_ANDAMENTO_ASTRA_2026-10-21.md)).
Il sistema sa eseguire passaggi fluidi ma sbaglia ancora **quale** passaggio scegliere; un voto 1–10 su clip che
condividono lo stesso cue non può correggere un cue sbagliato. Serve una piccola verità di riferimento: sapere che una
clip diventa buona facendo entrare B una battuta prima vale più di un 4. Traguardo: un piccolo set dei brani
dell'utente approvato dall'inizio alla fine, con ogni correzione registrata, e la misura di quante decisioni il
sistema prende giuste da solo.

Livelli: [E] risultato sperimentale di una fonte, [T] teoria/metodo, [P] pratica, [H] scelta ingegneristica nostra,
[U] osservazione dell'utente. Nessuna regola qui sotto è validata da ascolti: sono strumenti per raccoglierli.

## 1. Protocollo: cue fisso o effetto fisso, mai entrambi liberi

- **Effetto variabile, cue fisso** (galleria a versioni, `musicmix.dj-review.*`): dice quale tecnica funziona *su quel
  cue*. Non dice nulla sul cue.
- **Cue variabile, effetto fisso** (torneo dei cue, `musicmix.cue-tournament.1`): 4–6 varianti della stessa coppia con
  la **stessa tecnica** e uscita di A e/o ingresso di B diversi. Il manifest d'ascolto elenca i confronti espliciti
  (`solo_cue_B`, `solo_cue_A`, `cue_A_e_B`, `solo_pickup_B`, con Δ in battute), così un giudizio si legge come
  effetto di un solo fattore quando il confronto lo isola.
- Ordine casuale riproducibile (seme registrato = sha256 di run, torneo, seme) e ascolto cieco: i cue e la "scelta del
  sistema" compaiono solo dopo aver salvato la scelta (Margulis 2014 p. 15 [E], già usato per la galleria cieca;
  randomizzazione e riferimento nascosto come in MUSHRA, ITU-R BS.1534-3 [T], letto da fonti secondarie, vedi
  `REVISIONE_SET_TRANSIZIONI_2026-10-20.md` §39–44). Il numero di ascolti prima del salvataggio è registrato
  (`plays_before_save`; Margulis p. 101, Huron pp. 133–134 [E]: ascolti ripetuti non sono indipendenti).
- Scelta della migliore + insieme delle accettabili, senza scala: è un dato di preferenza (scelta forzata tra
  alternative, Thurstone 1927 / Bradley–Terry 1952 [T]) e non un voto assoluto. Con pochi tornei non si stima alcun
  modello: si usano come casi verificati (§4).
- Prima i cue, poi gli effetti sul cue migliore (ciclo proposto dalla revisione esterna, decisione dell'orchestratore).

## 2. Correzioni strutturate nel modulo di voto (`musicmix.dj-review.2`)

Ogni modulo (galleria a versioni, approvazione del set, varianti del torneo) ha, oltre a voto/note/tag, un blocco
"Correzioni (facoltative)" (`tools/listening_corrections.py`). Convenzioni, uguali ovunque:

| campo | valori | significato |
|---|---|---|
| `a_exit` | −2, −1, ok, +1, +2, `fine_frase` | uscita di A spostata sulla griglia di A; il contenuto di B allo switch non cambia (B arriva prima/dopo con l'uscita di A) |
| `b_entry` | −2, −1, ok, +1, +2 | punto d'ingresso nel brano B spostato sulla griglia di B; l'uscita di A non cambia (−1 = allo switch si sente una battuta di B in più) |
| `b_pickup` | sì/no | B deve entrare con il pickup (sillabe prima del downbeat) |
| `b_section` | ritornello, strofa, intro, breakdown/bridge, altra | B deve entrare in un'altra sezione |
| `wrong_b` | sì/no | brano B sbagliato per questo momento |
| `stay_longer_a` | sì/no | restare su A più a lungo |
| `technique` | `type_key` del runner (taglio, blend 4/8/16, echo/filter out, build, halving, loop...) o "altra" | tecnica sbagliata → quella indicata |
| `grid` | doppio tempo, metà tempo | la griglia di un brano è sbagliata di un fattore 2 |
| `grid_offbeat` | dal tag già esistente `battute_sfasate` | battuta sfasata |

Un campo vuoto non è un'approvazione esplicita (`ok` sì). I documenti `.1` restano leggibili (correzioni vuote). La
pagina salva anche il cue del gruppo (`cue`: battute e secondi di uscita/ingresso) perché le correzioni sono relative
al cue della clip giudicata, non a quello del set. Seek, "vai alla transizione", modalità cieca e `plays_before_save`
restano come prima.

## 3. Approvazione del set intero (`musicmix.set-approval.1`)

`tools/build_listening_artifact.py SET.json --kind set-approval [--blind]` (`tools/listening_pages.py`): tutto il set in
un player (bande dei brani, regioni delle transizioni, switch, tracklist cliccabile); per ogni transizione "ascolta la
zona" (da 10 s prima dell'inizio a 10 s dopo la fine, loop facoltativo), verdetto **approvata / da correggere / da
rifare con altro brano**, scelte rapide, correzioni, nota → doc `<run>__set_tNN`; giudizio complessivo (verdetto del
set, flusso, energia, varietà, ripetitività 1–5, nota) → doc `<run>__set`. Ogni doc porta il piano del sistema
(battute, secondi, tecnica): l'elaborazione non dipende dalla pagina. Audio: il WAV del set (hash verificato)
codificato in MP3 CBR al bitrate più alto ≤ 128 kbps sotto il limite dei file degli Artifact (set v3: 96 kbps,
12,6 MB). Con `--blind` tecnica, regola e QA compaiono dopo il verdetto.

## 4. Dalle correzioni al piano e ai passaggi di riferimento

`tools/apply_corrections.py plan --set SET.json --judgments DIR... --out corrections_plan.json [--references DIR]`
legge i giudizi esportati (file singoli o "Copia JSON") e scrive `musicmix.corrections-plan.1`. Traduzione [H]:

- uscita ±k → `a_out_bar + k`; `fine_frase` → primo downbeat dopo la fine del verso di A che attraversa l'uscita
  (modello `dj_phrase` della run, ASR affidabile, entro 8 battute), senza modello la prima battuta successiva con
  `phrase_level ≥ 4`; "resta su A" → prima battuta ≥ +8 con `phrase_level ≥ 8`;
- ingresso ±k → `b_in_bar + k`; sezione → inizio di sezione di B con quell'etichetta più vicino; pickup →
  `dj_phrase.b_lead_in` (o `entry_point`) sul nuovo downbeat; B a metà frase per il modello → `needs_manual`; senza
  modello un battito;
- tecnica → `type_key` (le tecniche dell'albero si costruiscono, quelle creative solo se la policy le propone su quel
  cue); brano B sbagliato → coppia vietata, ordine da rifare; griglia doppia/metà o sfasata → `manuale` (le battute
  non sono affidabili finché l'analisi non è rifatta).

Priorità tra fonti per la stessa transizione: torneo (confronto esplicito) > approvazione del set > galleria; a parità
l'ultimo salvato; gli altri documenti sono elencati come `superseded_docs`.

La sezione `runner` del piano (cue fissati per tutte le transizioni, coppie vietate, ordine congelato bloccato se
nessuna coppia è vietata, transizioni da rifare, render congelati riutilizzabili) è letta dall'hook minimo di
`tools/run_dj_set_v2_books.py` (`SetHooks.corrections`): `pin_pair` lascia ammissibile solo il cue fissato
(costruendolo con `dj_select.plan_transition` se non era tra le candidate), `prefer_technique` mette per prima la
tecnica fissata, `reuse_render` riusa audio e QA della transizione congelata quando coppia, cue e tecnica coincidono
(niente render, QA né veto ripetuti), il `set.json` registra la provenienza in `corrections`.
`apply_corrections.py rerender --plan ... --out ...` lancia il set v3 con il piano e scrive le transizioni accettate
nella run nuova (il runner v3 le scriverebbe altrimenti nella run congelata, `OUT` globale).

**Passaggi di riferimento** (`giudizi_cloud/reference_transitions/`, `musicmix.reference-transition.1`): solo
transizioni **verificate all'ascolto** — verdetto "approvata" senza correzioni nella pagina di approvazione (se il set
è un ri-render corretto, la correzione diventa così approvata e il riferimento cita il pin e il piano) e variante
migliore di un torneo senza ulteriori correzioni. Contenuto: brani con sha256 dell'audio e dell'analisi, griglia
(BPM, periodo, provider, confidenza, battuta, downbeat), uscita di A e ingresso di B in secondi, campioni e battute,
lead-in, sezione, parola e verso vicini (solo un estratto di 60 caratteri: i testi sono protetti, bastano le coordinate
in campioni), tecnica, `verified_by`, documento di giudizio con file e sha256, piano di provenienza. I file non si
sovrascrivono mai (contenuto diverso → nuovo nome con hash).

## 5. Separazione tra dati di taratura e dati di verifica

- I passaggi di riferimento hanno `use: "verifica"`: **non** si usano per scegliere soglie, pesi o regole. Servono a
  misurare se una regola nuova li riproduce (come i giudizi V8/V10 tenuti a parte, `00_metodo`).
- La taratura usa altri giudizi (gallerie cieche, `tools/calibrate_dj_critic.py` con la divisione deterministica
  selezione/verifica per gruppo). Se una regola viene adattata guardando un set approvato, la verifica va fatta sul
  set successivo (verifica nel tempo), non sullo stesso.
- Una correzione proposta ma non riascoltata resta nel piano (`status: corretta`), non è un riferimento.
- Integrità tecnica (QA, critico) non è qualità musicale; un riferimento è un'osservazione dell'utente [U], non una
  legge del dominio: vale per quella coppia, quei cue e quell'ascoltatore.

## 6. Metrica di accordo (`musicmix.decision-agreement.1`)

`tools/decision_agreement.py --case SET.json DIR [--case ...]`: per ogni transizione giudicata nella pagina di
approvazione, quattro decisioni del sistema: **ordine** (il brano B: approvato se non "da rifare con altro brano" né
"B sbagliato"), **uscita** (cue di A uguale dopo la traduzione delle correzioni), **ingresso** (cue di B e lead-in
uguali), **tecnica**. Riporta tasso di approvazione per decisione e per transizione intera, approvazioni esplicite
(verdetto "approvata" o campo `ok`) separate da quelle implicite (campo vuoto in "da correggere"), spostamenti in
battute delle decisioni corrette (|Δ| medio, mediana, prima/dopo) e IC bootstrap percentile al 95 % (ricampionando le
transizioni, a grappoli per set con più set; seme fisso). Le transizioni "da rifare" contano solo per l'ordine;
"da correggere" senza campi (`manuale`) solo per l'ordine e come transizione non approvata per intero. Sotto 10
transizioni il report avverte che gli intervalli sono larghi. Nessuna soglia è stimata.

## 7. Torneo dei cue: formato del manifest delle varianti (`musicmix.cue-variants.1`)

Descritto nel docstring di `tools/listening_pages.py`. Per torneo: `id`, `transition_index` (facoltativo), `a`/`b`
(slug, titolo, artista, sha audio), `technique` (`type_key`, uguale per tutte le varianti: le altre sono escluse con
`tecnica_diversa`), `system` (cue del sistema, facoltativo); per variante: `id`, `audio_path` (.wav codificato o .mp3
copiato), `sha256`, `a_out_bar`, `b_in_bar`, `a_out_s`, `b_in_s`, `b_lead_in_samples`, `transition_start_s`,
`switch_s`, `transition_end_s`, `duration_s` (obbligatorio per gli mp3 senza ffprobe), `label`, `qa`. Clip secondo
AGENTS.md (un avviso se fuori regola). Meno di due varianti valide → torneo escluso; fuori da 4–6 → avviso.

## 8. Implementazione e verifiche

- `tools/listening_corrections.py` (campi, HTML/JS condivisi, lettura dei giudizi), `tools/listening_pages.py`
  (approvazione del set, torneo), `tools/build_listening_artifact.py` (`--kind`, `--seed`, `--set-bitrate`, dj-review.2),
  `tools/apply_corrections.py`, `tools/decision_agreement.py`, hook in `tools/run_dj_set_v2_books.py`.
- Test: `tests/test_listening_corrections.py` (moduli, pagine, piano, riferimenti, hook, metrica). JavaScript provato
  con jsdom (salvataggio nel database simulato e in locale, rivelazione dopo il voto, zona −10 s), non su un telefono.
- Prove sul set congelato `runs/dj_set_v3_bridge` con giudizi **finti** (solo in scratchpad, nessun file in
  `giudizi_cloud/`); la run congelata è rimasta identica (hash di `set.json` e `transitions/` prima e dopo):
  - solo piano (`--plan-only`), 4 correzioni e nessuna coppia vietata: ordine congelato riprodotto (14 brani), cue
    corretti fissati; con una coppia vietata ordine e cue cambiano quasi tutti: vietare un arco è una
    ripianificazione, non un ritocco;
  - ri-render con **una** correzione (T03, A alla fine della frase: battuta 82 → 86): 12 transizioni riusate dal set
    congelato (render e QA), 1 renderizzata (warn), set di 13 transizioni chiuso (17:37), 290 s;
  - ri-render con 4 correzioni finte: le tre che spostano il cue di B o chiedono il blend sul ritornello falliscono
    il QA (B entra a metà frase: `b_entry`; voci sovrapposte), il runner toglie l'arco e con l'ordine bloccato il set
    si accorcia. Uno spostamento in battute non garantisce un ingresso a inizio frase: il torneo dei cue serve proprio
    a scegliere tra cue già renderizzati e controllati. Il pickup chiesto su `18k_no_kizzy` non si applica: il modello
    `dj_phrase` del set congelato (ASR inglese, poi rifatto in italiano altrove) vede B a metà frase; un lead-in di un
    battito imposto troncava una parola, quindi ora la richiesta va in `needs_manual`.

## Limiti

- Nessun ascolto ancora: le pagine raccolgono la verità di riferimento, non la contengono.
- Traduzione delle correzioni con regole [H] (fine frase, resta su A, sezione, pickup di un battito senza modello `dj_phrase`).
- Una tecnica creativa richiesta che la policy non propone su quel cue non si forza (resta la nota).
- Con l'ordine bloccato, una correzione che fallisce il QA fa togliere l'arco dal ripianificatore del runner e il set
  si accorcia: la correzione va rivista (o provata nel torneo), non forzata.
- Il riuso dei render congelati vale solo se coppia, cue e tecnica coincidono; le transizioni riusate non ripassano
  il QA/critico attuali (restano quelle ascoltate).
- "Battuta sfasata" e doppio/metà tempo richiedono di rifare l'analisi del tempo: segnalati, non corretti.
- Con un solo ascoltatore e pochi set la metrica di accordo descrive questo utente su questi brani.
