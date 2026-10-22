# Libreria scientifica MusicMix

Archivio mantenuto, con copie locali e sintesi in italiano. Data di raccolta: **2026-10-20**. Il manifest completo è [`_meta/SOURCES_MANIFEST.json`](_meta/SOURCES_MANIFEST.json); ogni sezione ha anche `SOURCES.json` e cartelle `originals/`, `texts/` e, per i fetch rotti, `failed_fetches/`.

| Sezione | Domanda | Letture chiave |
|---|---|---|
| [00 metodo](00_metodo/RIASSUNTO.md) | Come distinguere evidenza, teoria e ipotesi | protocollo, hash, read-depth, held-out |
| [01 psicologia](01_psicologia_aspettativa/RIASSUNTO.md) | Cosa significa tensione, aspettativa e completamento | Farbood 2012; Lehne/Koelsch 2015; Hansen 2021; Turrell 2025 |
| [02 rap/voce](02_fraseggio_rap_voce/RIASSUNTO.md) | Come termina o continua una frase vocale | MCFlow; Adams 2020; Duinker 2020 |
| [03 DJ](03_struttura_transizioni_dj/RIASSUNTO.md) | Come proporre e verificare una transizione | VandeVeire 2018; Kim 2020; CMMR/UnmixDB |
| [04 modelli](04_modelli_preaddestrati/RIASSUNTO.md) | Cosa fanno davvero cue/structure models | CUE-DETR; Zehren; Raveform; LinkSeg |
| [05 renderer](05_renderer_loudness_tempo/RIASSUNTO.md) | Come separare difetto tecnico e giudizio musicale | Daito Manabe; MosaikBox; UnmixDB |
| [06 valutazione](06_valutazione_autofeedback/RIASSUNTO.md) | Come validare senza scalare tutto in un punteggio | TenseMusic; SongEval; protocollo a componenti |
| [07 benchmark](07_dati_benchmark/RIASSUNTO.md) | Quali dati sono annotati e quali leakage hanno | M-DJCUE parse reale; Raveform; EDM-CUE |

## Risultato della prima raccolta

Sono archiviate **23 voci di provenienza**, di cui 19 copie valide di paper/dataset/artifact, 2 fetch dichiarati invalidi, 1 archivio parziale conservato per audit e 1 metadata-only (LinkSeg, paper non presente localmente). Più di otto fonti primarie hanno copia o testo locale: Farbood, Lehne/Koelsch, Hansen, Turrell, MCFlow, Adams, Duinker, VandeVeire, Kim, CMMR, CUE-DETR, Zehren, Raveform, Daito, TenseMusic, SongEval e M-DJCUE.

La conclusione operativa è circoscritta: la letteratura sostiene candidati metrici/strutturali, protezione del fraseggio vocale, diagnostiche tecniche e trend di tensione separati. Nessuna fonte trovata valida un modello che decida da sola “continuazione o arrivo in B” per una coppia rap arbitraria. Il [mapping evidenza → implementazione V11](IMPLEMENTAZIONE_V11.md) distingue le componenti eseguite dai limiti ancora aperti.

## Fonti aggiunte il 2026-10-20: tipo di transizione, set, preferenze, mashup

Schede complete (citazione, DOI/URL, tipo [E]/[S]/[P]/[T], cosa è stato letto, numeri, implicazioni e priorità) in [`_meta/REVISIONE_SET_TRANSIZIONI_2026-10-20.md`](_meta/REVISIONE_SET_TRANSIZIONI_2026-10-20.md). Copie locali non archiviate in `originals/` (vedi §9 della revisione).

- **03 DJ, tipo di transizione e pratica**: Bittner et al. 2017 (ISMIR, full); Williams et al. 2025 (TIME/LIPIcs, full); Thalmann et al. 2018 (WAC, abstract); Gates et al. 2006 (DIS, abstract); Katz 2012 (OUP, recensioni); Broughton & Brewster 2003/2000 (metadati); Smith 2013 (Ashgate, recensione); Hansen et al. 2011 (Acta Acustica, abstract); Mouraviev 2024/25 (Performance Research, abstract); Hirai et al. 2015; DJ-AI 2025 (Audio Mostly, full, evidenza debole).
- **03 DJ, sequenziamento di set**: Cliff 2000 (HPL-2000-104, abstract); Kell & Tzanetakis 2013 (ISMIR, full); Ishizaki et al. 2009 (ISMIR, full); Liebman et al. 2015 (AAMAS, full); Bonnin & Jannach 2014 (ACM CSUR, metadati); Jannach et al. 2015 (RecSys, abstract); Flexer et al. 2008 (ISMIR, abstract); Cunningham et al. 2006 (ISMIR, abstract); Neto et al. 2024 e Herrmann et al. 2024 (arXiv, album).
- **01 psicologia, groove ed energia**: Janata et al. 2012; Witek et al. 2014; Matthews et al. 2019; Madison 2006; Madison et al. 2011; Stupacher et al. 2016; Senn et al. 2018; Wesolowski & Hofmann 2016; Streich & Herrera 2005 (danceability Essentia).
- **01 psicologia, aspettativa (solo diagnostica)**: Huron 2006 (descrizione e recensioni); Solberg & Jensenius 2016/2017 (EMR, abstract); Solberg & Dibben 2019 (Music Perception, full); Solberg 2014 (Dancecult, solo voce bibliografica); Yadati et al. 2014; Agres et al. 2017; Butler 2006.
- **02 voce e 05 renderer, mashup e voci sovrapposte**: Davies et al. 2014 AutoMashUpper (TASLP, full); Lee et al. 2015 (ISMIR, parziale); Huang et al. 2021 (AAAI, full); Chen et al. 2020 Neural Loop Combiner; Xing et al. 2020 (solo citazione); Spleeter 2020 (JOSS); HT-Demucs 2023 (ICASSP); Bregman 1990; Brungart 2001 (JASA 109(3)) e Brungart et al. 2001 (JASA 110(5)); Bürgel et al. 2021; Condit-Schultz & Huron 2015.
- **06 valutazione, preferenze e protocolli d'ascolto**: Thurstone 1927; Bradley & Terry 1952; Chu & Ghahramani 2005; Houlsby et al. 2011 (BALD); González et al. 2017 (PBO); Chen et al. 2013 (Crowd-BT); Mantiuk et al. 2012; Pérez-Ortiz et al. 2020 (scala JOD); ITU-R BS.1534-3 (2015); Schoeffler et al. 2018 (webMUSHRA); Cawley & Talbot 2010.

Lacune confermate dalla ricerca: nessuno studio classifica i tipi di transizione (cut, blend, echo, filter, spinback) in mix reali; nessuno misura curve di energia su molti set DJ; nessuno collega groove e transizioni; nessuno riguarda il rap.

## Implementazione del set DJ v1 (2026-10-20)

Albero del tipo di transizione, ordine dei brani (Held–Karp) e cue lungo il set: [`_meta/IMPLEMENTAZIONE_SET_DJ_V1_2026-10-20.md`](_meta/IMPLEMENTAZIONE_SET_DJ_V1_2026-10-20.md) (`src/musicmix/dj_set.py`, run `runs/dj_set_v1/`).

## Tecniche creative di transizione e mashup (2026-10-21)

Catalogo di 23 tecniche (loop, loop roll, acapella, vocal swap, word play, echo freeze, reverb, backspin, brake, build con rullo e riser in tonalità, nota/pad ponte, trasposizione di un semitono, tempo ride, half-time, filter swap, mid cut, double drop, beat juggling, scratch) con contesto, durata, parametri, rischi e fonte; regole contesto → tecnica e implementazione: [`_meta/TECNICHE_CREATIVE_2026-10-21.md`](_meta/TECNICHE_CREATIVE_2026-10-21.md) (`src/musicmix/dj_creative.py`, `src/musicmix/dj_policy.py`, run `runs/dj_creative_v1/`). Le tecniche sono pratica DJ [P] o sistemi [S]; nessuno studio empirico ne misura l'uso o il gradimento nei mix reali, tantomeno nel rap.

## Quando tagliare: fine verso e pickup (2026-10-21)

Fonti (MCFlow come corpus, Adams 2009, Ohriner 2019, Pearce 2010, GTTM, Narmour, Wightman 1992, Knösche 2005, Neuhaus 2006, Silva 2014, Zacks 2007, Sridharan 2007) e risultati in [`_meta/QUANDO_TAGLIARE_2026-10-21.md`](_meta/QUANDO_TAGLIARE_2026-10-21.md); modulo `src/musicmix/dj_phrase.py`, tabelle `configs/dj_phrase_mcflow_v1.json`, run `runs/dj_phrase_v1/`.

## Giudice automatico dei mix (2026-10-21)

v1–v3 in [`_meta/AUTO_CRITICO_2026-10-21.md`](_meta/AUTO_CRITICO_2026-10-21.md) (`src/musicmix/dj_critic.py`, run `runs/dj_critic_v1/`): affidabile come veto tecnico, non come gusto. v4 in [`_meta/AUTO_CRITICO_V4_2026-10-21.md`](_meta/AUTO_CRITICO_V4_2026-10-21.md) (`src/musicmix/dj_critic_books.py`, `score_transition(..., version="v4")`, run `runs/dj_critic_v4/`): blocchi `technical` (veto) e `musical` (diagnostica) con canali dai libri (Huron, Butler, Webber); il flag di fase misurato con Beat This! sulla clip dà falsi positivi sui blend; calibrazione solo su voti ciechi con `tools/calibrate_dj_critic.py`.

## Libri (copie private, testo fuori dal repo) — 21/10/2026

- `libri/` — cinque libri sul mestiere DJ (Katz 2012, Brewster & Broughton 2003, Webber 2008, Butler 2006, Williams 2013): schede per capitolo, note di sintesi, `libri/MAPPA_PIPELINE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md` (processo per processo, con pagina), `libri/PROPOSTE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md` (patch per render/effects/creative/select/critic). Implementato: profilo "books" di dj_set e dj_policy, modulo dj_form (categorie di Butler sull'audio).
- `libri/` — quattro libri su flow rap, aspettativa e ripetizione (Ohriner 2019, Krims 2000, Huron 2006, Margulis 2014): schede per capitolo, `libri/MAPPA_PIPELINE_OHRINER_KRIMS_HURON_MARGULIS.md`, `libri/PROPOSTE_OHRINER_KRIMS_HURON_MARGULIS.md`. Ricerca nel testo: `python tools/libri.py search`.
- `libri/INTEGRAZIONE_PROPOSTE_2026-10-21.md` — stato di ogni proposta dei nove libri (integrata nel profilo books / opzionale spenta / rinviata / al critico), con file, test e pagina.

## Set DJ v3 "ponte" e galleria A/B cieca (2026-10-21)

Catalogo del ponte congelato (21 brani, 18 usabili, 14 nel set), profilo libri + `dj_phrase` + tecniche di `dj_policy`
(build verso il ritornello, halving build), veto tecnico del critico (fase solo se non dichiarata dal render, dopo il giudice v4)
e galleria A/B cieca default vs libri: [`_meta/SET_V3_BRIDGE_2026-10-21.md`](_meta/SET_V3_BRIDGE_2026-10-21.md)
(`tools/run_dj_set_v3_bridge.py`, run `runs/dj_set_v3_bridge/`). Non ascoltato: QA e critico = integrità tecnica.

## Audit integrale dei nove libri — 21 ottobre 2026

[Revisione critica aggiornata](libri/audit_2026-10-21/REVISIONE_PIPELINE.md):25 blocchi Luna5.6 xhigh,2.527/2.527 pagine PDF e75/75 unità EPUB; ledger verificati, fonti SHA e riletture del coordinatore. [Indice audit](libri/audit_2026-10-21/README.md), [backlog](libri/audit_2026-10-21/BACKLOG.json). Test e controesempi separati dai giudizi percettivi; produzione6f03b33, ultimo delta bec3420 ricontrollato. Le raccomandazioni correnti correggono e sostituiscono le proposte storiche in contrasto, senza dichiarare già implementati i nuovi interventi.

## P0 misure e regole dopo l'audit dei libri (2026-10-21)

Tre controesempi dell'audit corretti con test (forma senza core → `unknown`; onset dello stem mai oltre la parola, sillabe
incerte; percentile timbrico degenere → astensione), livelli di evidenza [E]/[T]/[A]/[P]/[H] con pagine rilette per ogni
regola di `BOOK_RULES` e dei profili libri (`dj_phrase`, `dj_set`, `dj_policy`), pagine di Butler corrette, finestre dei cue
per tecnica come opzione (default invariato) con misura sul catalogo del ponte:
[`_meta/P0_MISURE_REGOLE_2026-10-21.md`](_meta/P0_MISURE_REGOLE_2026-10-21.md). Specifica del piano per eventi (non
implementata): [`_meta/PIANO_EVENTI_SPEC_2026-10-21.md`](_meta/PIANO_EVENTI_SPEC_2026-10-21.md). Stato per item:
[`libri/audit_2026-10-21/STATO_BACKLOG_CLOUD.md`](libri/audit_2026-10-21/STATO_BACKLOG_CLOUD.md).

## Piano per eventi (P0.5) e cassa 808 (2026-10-21)

`musicmix.event_plan.1` implementato (`src/musicmix/dj_events.py`): per ogni transizione evento di A (sezione, completato o
saltato, verso da `dj_phrase`), lead-in e evento obiettivo di B con il ritardo, alternative di cue/durata/tecnica con motivo,
trace dei moduli. Run di sola pianificazione sul catalogo congelato del ponte (`runs/dj_set_v3_events_plan/`, set v3 riprodotto
13/13), opzione sperimentale [H] spenta (B&B cap. 13 'Even tighter'; Butler pp. 246–247), `dj_form.3` con la cassa 808:
[`_meta/PIANO_EVENTI_IMPL_2026-10-21.md`](_meta/PIANO_EVENTI_IMPL_2026-10-21.md). Descrizione, non giudizio: niente ascolti.

## P0 render / QA / critico — 21 ottobre 2026

[Nota](_meta/P0_QA_RENDER_2026-10-21.md): backend di time-stretch qualificato a ogni processo (82/440 Hz, durata, clic;
il vecchio fallback ffmpeg R2 `crisp` porta 82 Hz a 78,1 Hz anche su Linux ed è rifiutato; nuovo fallback R3 via
ffmpeg), QA nel tempo renderizzato (time-map a tratti di B, stato di copertura per controllo, parola tagliata con
uscita a effetto), metadati di B stirato in `dj_creative`, blocco `technical` del critico v4 allineato al veto del set
v3. Ricontrollo del set v3: `runs/dj_set_v3_bridge/qa_v5_recheck.json`. Stato degli item:
[`libri/audit_2026-10-21/STATO_BACKLOG_CLOUD.md`](libri/audit_2026-10-21/STATO_BACKLOG_CLOUD.md).

## Correzioni strutturate, approvazione del set, torneo dei cue e passaggi di riferimento (2026-10-21)

Protocollo cue fisso / effetto fisso, campi di correzione nel voto (`musicmix.dj-review.2`: uscita di A e ingresso di B
±battute, fine frase, pickup, sezione, brano B sbagliato, resta su A, tecnica, griglia), pagina di approvazione del set
intero (`musicmix.set-approval.1`), torneo dei cue alla cieca (`musicmix.cue-tournament.1`, manifest
`musicmix.cue-variants.1`), traduzione delle correzioni nel piano del set (`musicmix.corrections-plan.1`, hook
`SetHooks.corrections` del runner), passaggi di riferimento verificati all'ascolto
(`giudizi_cloud/reference_transitions/`, `musicmix.reference-transition.1`, solo verifica, mai taratura) e metrica di
accordo con IC bootstrap: [`_meta/RIFERIMENTI_E_APPROVAZIONE_2026-10-21.md`](_meta/RIFERIMENTI_E_APPROVAZIONE_2026-10-21.md)
(`tools/listening_corrections.py`, `tools/listening_pages.py`, `tools/apply_corrections.py`, `tools/decision_agreement.py`).
Strumenti per raccogliere la verità di riferimento: nessun ascolto ancora.

## Mashup creativi: note dei due brani, voce in loop, botta e risposta (2026-10-21)

`src/musicmix/dj_mashup.py`: mashability con trasposizione −3…+3 su cromagramma sincronizzato ai battiti nello spazio TIV
(AutoMashUpper, Huang 2021, Bernardes 2016; TIV.lib ufficiale vendorizzato, MIT), e otto tecniche in cui B suona com'è e
si stira/traspone solo il materiale preso in prestito: verso di A sulla base di B, verso di A sulla sola batteria di B
(Demucs), hook di B anticipato sopra A, loop e chop (3+3+2) della voce di A sulla griglia di B, botta e risposta, freeze
dell'accordo di A che scivola nella tonalità di B, riverbero inverso della prima parola di B. Parole intere: A tace solo a
fine verso, la voce di B si apre solo a inizio frase (pickup compreso). Galleria cieca v2 con il riferimento del set v3
non votabile: [`_meta/MASHUP_CREATIVI_2026-10-21.md`](_meta/MASHUP_CREATIVI_2026-10-21.md) (`tools/run_dj_mashup_v1.py`,
run `runs/dj_mashup_v1/`). Non ascoltato: QA e critico = integrità tecnica.

## Schede brano persistenti e affidabilità del pianificatore (2026-10-21)

Schede `musicmix.track-card.1` (`track_cards/<slug>.json`, legate allo SHA-256 dell'audio e agli hash degli ingressi,
costruite una volta e rifatte solo se un ingresso cambia; il ponte le produce da solo dopo l'analisi): griglia e accordo
dei provider, mezzo/doppio tempo sospetto, tonalità, sezioni e `dj_form`, lingua e ASR, versi `dj_phrase`, ingressi puliti e
uscite a fine verso per ogni cue, regioni strumentali, stem con provenienza, limiti di stretching, casi incerti (ascolto /
dati / aggirabili) e correzioni manuali riusate in ogni set. Affidabilità [H] per brano e per candidato, opzione `prefer` del
runner (spenta di default), copertura della libreria e varietà tra set:
[`_meta/SCHEDE_BRANO_AFFIDABILITA_2026-10-21.md`](_meta/SCHEDE_BRANO_AFFIDABILITA_2026-10-21.md)
(`src/musicmix/track_cards.py`, `src/musicmix/reliability.py`, `tools/build_track_cards.py`, `tools/plan_reliability_report.py`,
run `runs/plan_reliability_v1/`). Descrizione delle analisi, non giudizio: niente ascolti.

## Ingresso di B: diagnosi della galleria v3 e correzione (2026-10-21)

Il difetto dominante dei voti ciechi del set v3 è l'ingresso di B: in tutte le 23 clip B diventava udibile esattamente sul
suo downbeat, e in 6 delle 7 transizioni con note su B c'è una parola di B troncata o una frase di B iniziata prima.
Regola della prima frase di B intera (lead-in fino a una battuta dal confine di frase: pausa ≥ 1 battito o fine verso di
`dj_phrase` con pausa notata), cue di B spostati o non ammissibili senza ripiego (`dj_select.relocate_entry`,
`dj_set.entry_guard`), lead-in nel render (`dj_render.add_b_lead_in`, voce di B sola fino al downbeat), controllo `b_entry`
in `dj_qa.6`, ripianificazione invece del «fail meno grave» in `run_dj_set_v2_books.build_set`, verifica a tecnica fissa
con cue variabili (torneo dei cue, non pubblicato): [`_meta/INGRESSO_B_2026-10-21.md`](_meta/INGRESSO_B_2026-10-21.md)
(`tools/run_b_entry_fix_v1.py`, run `runs/b_entry_diagnosis/`, `runs/b_entry_fix_v1/`). Integrità tecnica, nessun ascolto.

## Set DJ di riferimento: allineamento mix → brani e prior (2026-10-21)

Reimplementazione del metodo di Kim et al. 2020 (codice `mir-aidj/djmix-dataset` senza licenza: riscritto, non copiato)
con curve di guadagno per banda (Kim 2022, NNLS al posto di cvxpy), zona d'effetto, tecnica [H], brani mancanti e
contesto di frase/sezione/ritornello: [`_meta/SET_DI_RIFERIMENTO_2026-10-21.md`](_meta/SET_DI_RIFERIMENTO_2026-10-21.md)
(`src/musicmix/mix_alignment.py`, `src/musicmix/reference_priors.py`, `tools/analyze_reference_mix.py`,
`tools/validate_reference_mix.py`, `tools/run_reference_mix_validation.py`, `tools/reference_mix_stats.py`, run
`runs/reference_mix_v1/`). Validato sui nostri set (v3: ordine esatto, mappa 16 ms, switch 0,18 battiti, tempo 0,07 %),
su mp3/velocità/trasposizione e su UnmixDB (CC BY-NC-ND); durata della sovrapposizione e tecnica restano deboli.
Prior generici dai metadati Raveform (CC BY 4.0, elettronica: dominio diverso dal rap, peso basso), file
`configs/reference_priors.json` non creato (default invariato). Mixotic non accessibile; nessun dataset rap con audio legale.

## Supervisione dai brani stessi: dove, carica e arrivo (2026-10-21)

Strada 4 dell'autonomia: punti in cui i brani si articolano da soli (sezione sul downbeat, verso dopo una pausa, rientro della
voce) contro le loro versioni spostate nello stesso brano (±½/1 battito, ±½/1 battuta, metà parola, metà verso, metà
sezione); giunte tra brani per il controllo "stesso brano"; COCOLA (ICASSP 2025, MIT) sulla coerenza voce/strumentale;
compito tensione → risoluzione; controlli esterni sui voti ciechi (mai addestramento):
[`_meta/SUPERVISIONE_DAI_BRANI_2026-10-21.md`](_meta/SUPERVISIONE_DAI_BRANI_2026-10-21.md) (`src/musicmix/self_supervision.py`:
`cue_naturalness`, `tension_at_exit`, `resolution_at_entry`; `tools/build_insong_dataset.py`, `tools/insong_eval.py`; run
`runs/insong_v1/`, cache `runs/insong_v1/cue_scores/<slug>.json`). Impara la continuità interna ai brani, non le scelte di un DJ.

## Carica di A, risoluzione di B, base continua, tonalità e testo (2026-10-22)

I criteri dei voti ciechi dell'utente («A carica, B risolve», base continua, tonalità discordante anche senza sovrapposizione,
testi affini) come punteggi spiegabili con fonti (Huron pdf pp. 26, 39–40, 337; Butler pp. 238, 246; Krims p. 64; Ohriner
p. 91; B&B pp. 33–34, 38–39) e pesi [H] pre-registrati prima dei voti; controllo esterno su torneo dei cue e galleria v3
(carica 0,77, affinità di testo 0,69; risoluzione di B e continuità pianificata non concordano); integrazione nel
pianificatore (`--tension`) e set v4: [`_meta/TENSIONE_RISOLUZIONE_2026-10-22.md`](_meta/TENSIONE_RISOLUZIONE_2026-10-22.md)
(`src/musicmix/dj_tension.py`, `tools/run_dj_tension_v1.py`, `tools/run_dj_set_v4.py`, run `runs/dj_tension_v1/`, `runs/dj_set_v4/`).
Un solo ascoltatore e pochi voti: diagnostica, non validazione; il set v4 non è stato ascoltato.

## Palestra automatica degli errori e sforzo umano (2026-10-21)

Gate obbligatorio per ogni aggiornamento della pipeline (AGENTS.md, "Autonomia"): 31 riferimenti (voto utente >= 6,
set v3 con QA pass, set v3 con lead-in di B, continuazioni naturali dentro i brani) e 469 varianti con difetti
controllati (parola mozzata in A/B, pickup tolto, sfasamento ½/¼ battito/30 ms, B ±1 battuta, doppio basso, +3/+6 dB,
buco, clic, tonalità ±1 nel blend, eco fuori tempo, combinazioni), rilevamento appaiato di QA e critico, soglie [H] in
`configs/defect_gym_gate.json`, misura dell'intervento umano per 100 brani e 10 set:
[`_meta/PALESTRA_DIFETTI_2026-10-21.md`](_meta/PALESTRA_DIFETTI_2026-10-21.md) (`src/musicmix/defect_gym.py`,
`tools/defect_gym.py`, `tools/human_effort_report.py`, run `runs/defect_gym_v1/`, `runs/human_effort_v1/`). Gate oggi
FAIL: sfasamento negli ingressi puliti, fuori frase, buco, clic ed eco non sono rifiutati; falsi allarmi su
continuazioni e su clip votate. Riconoscere un peggioramento non è riconoscere una bella transizione.

## Palestra dei difetti v2: `dj_qa.7` (2026-10-22)

Fase di B a un ingresso pulito misurata solo se affidabile (due rivelatori concordi; `fail` qualunque fase dichiari il
render, altrimenti `unavailable`), soglie calibrate sul rumore di 196 giunte naturali su brani disgiunti dai riferimenti
(`tools/calibrate_qa_noise.py`), controllo `phrase` (B fuori frase), `dropout` `fail` e acceso di default, `click`,
`echo_tempo` (warn), deck B passato anche nelle tecniche creative, metà di sviluppo/verifica dei riferimenti dichiarate
prima delle correzioni: [`_meta/PALESTRA_DIFETTI_2026-10-21.md`](_meta/PALESTRA_DIFETTI_2026-10-21.md) §6 (run
`runs/defect_gym_v2/`, ricontrollo del set v3 `runs/dj_set_v3_bridge/qa_v7_recheck.json`).

