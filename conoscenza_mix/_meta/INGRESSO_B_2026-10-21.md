# Ingresso di B: diagnosi della galleria v3, regola e correzione (21/10/2026)

Problema n. 1 secondo l'utente dopo la galleria cieca del set v3 ([SET_V3_BRIDGE](SET_V3_BRIDGE_2026-10-21.md)): l'**ingresso di
B** («B tagliato nel secondo sbagliato», «parola tagliata in B», «A e B tagliati al momento sbagliato», «B non risolve bene
A», tag `punto_frase_sbagliato`), indipendente dalla tecnica. Il piano per eventi ([PIANO_EVENTI_IMPL](PIANO_EVENTI_IMPL_2026-10-21.md))
aveva trovato pickup di B non suonati in T3, T9, T11, T12. Livelli: **[E]** misura (nostra o di un corpus), **[T]** teoria,
**[A]** analisi di un autore, **[P]** pratica DJ, **[H]** ipotesi/soglia nostra. I voti sono diagnostica tenuta separata
(AGENTS.md): nessuna soglia è stata adattata a una clip; integrità tecnica ≠ qualità; nessuna clip nuova è stata ascoltata.

Tool: `tools/run_b_entry_fix_v1.py` (`run`, `summary`). Uscite: `runs/b_entry_diagnosis/diagnosis.json` (diagnosi),
`runs/b_entry_fix_v1/results.json` (verifica), `runs/b_entry_fix_v1/clips/*.mp3` + `clips_manifest.json` / `clips_key.json`
(galleria futura, **non pubblicata**). Test: `tests/test_b_entry_fix.py`.

## 1. Diagnosi (clip votate, render rifatti identici al bit)

Per ognuna delle 23 clip votate delle 12 transizioni il render è stato rifatto con la stessa decisione e lo stesso cue
(**23/23 clip identiche al bit** al WAV votato, sha256) e nel punto in cui B diventa udibile si sono proiettate le parole di
B (parole raffinate di `dj_phrase` e ASR grezzo, `dj_qa.words_to_output(deck="b")`), lo stato del verso
(`dj_phrase.entry_point` e la regola nuova `b_lead_in`, §2), la sezione allin1 e il QA `dj_qa.6`. In tutte le 23 clip B
diventa udibile **esattamente sul suo downbeat di cue** (±10 ms di micro-fade; il blend di T4 dall'inizio dell'intro):
nessuna tecnica suonava ciò che precede il downbeat. Cause per transizione [E, misura nostra]:

| causa (regola) | transizioni | con nota su B dell'utente |
|---|---|---|
| parola di B troncata nel punto udibile (`b_word_cut`, > 50 ms per lato) | T3 «.30», T8 «time!», T11 «said», T12 «magic» | T3, T11, T12 |
| frase di B iniziata prima, nessun respiro (< 3/16 di battito) prima della prima parola udibile | T1, T2, T6, T9 | T1, T2, T9 |
| pickup/lead-in di B disponibile (≤ 1 battuta) ma non suonato | T3, T5, T9, T11, T12 | T3, T9, T11, T12 |
| nessun ingresso pulito a quel cue (`mid_line`: frase di B iniziata > 1 battuta prima o parola troncata) | T1, T2, T6, T8 | T1, T2 |
| nessun difetto dell'ingresso di B misurato | T4 (blend dall'intro, strofa 8 battute dopo), T7, T10 | T4 |

- Delle **7 transizioni con note su B** (g01, g02, g03, g04, g09, g11, g12) **6** hanno un difetto misurato all'ingresso di B
  (parola troncata o frase iniziata prima); la settima (T4, blend: «A e B tagliati al momento sbagliato, ma è fluido») non ha
  parole tagliate: B entra dall'intro e la sua strofa arriva 8 battute dopo l'ingresso, 4 dopo la fine del blend
  (`b_event_late`; piano per eventi): diagnosi di sezione/attesa, non di parola. `verifica_t01` (halving build verso un brano tedesco, «B secondo sbagliato»): ingresso
  strumentale, prima voce 8,3 s dopo: anche qui il difetto non è una parola troncata (sezione/attesa; trascrizione in tedesco).
- L'unico commento positivo (g07, «l'entrata di B è giusta come motion»): T7 è una delle 3 transizioni senza difetto misurato
  (frase di B che parte sul downbeat dopo un respiro). Lo stesso vale per T10, senza note.
- Difetti di B senza nota: T5 (pickup di 30 ms non suonato: trascurabile), T6, T8 (voto 4, nessuna nota).
- Lato A (non oggetto di questa correzione): parola di A tagliata (`verse_cut` di `dj_qa.5`) in T5, T8, T10, T12 e nelle
  versioni default di T2 — già segnalati dal ricontrollo P0 ([P0_QA_RENDER](P0_QA_RENDER_2026-10-21.md) §2).
- QA `dj_qa.6` sulle clip votate: `b_entry` **fail 15 / pass 7 / n.a. 1** su 23. Le note «B tagliato» stanno tutte tra i fail
  tranne T4. Con 23 clip, un ascoltatore e voti 2–7 è un accordo descrittivo, non una validazione della soglia.
- Il modello di frase di `dj_phrase` (confini p ≥ 0,5) da solo non bastava: in T3 il pickup secondo `entry_point` era «:30»,
  ma «7» lo precede di 0,36 s; con la sola pausa ≥ 1 battito, invece, nel rap quasi nessun confine esiste (0,5–1 % dei
  passaggi tra parole sui 10 brani B: il flow è continuo). La regola del §2 combina le due cose.
- `18k_no_kizzy` (B di T2) ha l'ASR inglese congelato su un brano italiano: tutte le sue parole sono allucinate e ogni cue
  risulta `mid_line`; la diagnosi di T2 vale quanto quell'ASR.

## 2. Regola

**R-B1 [H] (prima frase di B intera).** Dal primo vocabolo di B non finito al downbeat d'ingresso D si risale fino a un
confine di frase: una pausa della voce di B ≥ **1 battito** (`b_rest_beats`) oppure un fine verso del modello di
`dj_phrase` (p ≥ 0,5, tabelle MCFlow) seguito da una **pausa notata** (≥ ¾ di sedicesimo, `rest_min_16th`); la frase di B
comincia alla prima parola dopo il confine. Sui 10 brani B del set la regola dà confini nel 4–17 % dei passaggi tra parole,
come la frequenza delle frasi di MCFlow (~1 ogni 7 parole) — calibrazione sulla distribuzione, non sui voti. Se comincia prima di D, B deve essere udibile da lì (**lead-in**, arrotondato
all'ottavo dentro la pausa) se il lead-in è al più **una battuta** (`b_lead_in_max_beats` = 4), altrimenti B entrerebbe a
metà frase (`mid_line`) e il cue va cambiato. Stati: `instrumental`, `on_downbeat`, `lead_in` (ok), `mid_line`, `unknown`.
Funzioni: `dj_phrase.b_lead_in`, `dj_phrase.entry_clear_test` (versione `musicmix.dj_phrase.b_entry.1`).

Perché non solo `dj_phrase.entry_point`: quello accetta un pickup di al massimo 2 battiti e qualunque confine p ≥ 0,5,
anche senza pausa (nel rap il 41 % delle fini di frase non ha pausa, MCFlow, [QUANDO_TAGLIARE](QUANDO_TAGLIARE_2026-10-21.md)
§0): una parola udibile attaccata alla precedente suona troncata. Perché non solo la pausa: con la sola pausa ≥ 1 battito i
confini sono lo 0,5–1 % dei passaggi (flow continuo) e quasi nessun cue sarebbe ammesso (prima versione provata e scartata
su questa distribuzione, prima di guardare i risultati delle clip).
Con l'ASR inaffidabile la stessa regola usa l'attività dello stem vocale (pause < 1 battito chiuse).

**R-B2 [H] (cue di B).** Se l'ingresso non è pulito: lo stesso cue 1–2 battute prima (l'evento di B — inizio di strofa o
ritornello — arriva ancora, intero, 1–2 battute dopo che B è udibile), poi gli altri cue di frase di B per distanza, con la
stessa uscita di A (`dj_select.relocate_entry`, simmetrico di `relocate_cut`). Nel set: `dj_set.entry_guard`
(`musicmix.dj_set.entry_guard.1`): candidati spostati o **non ammissibili**; se nessun candidato della coppia resta
ammissibile la coppia ha costo d'arco infinito (nessun ripiego; `relax_if_empty=True` è solo un'opzione esplicita). Il
blend è esente (B sale col fader: una parola all'inizio della dissolvenza non è troncata).

**R-B3 [H] (render del lead-in).** `dj_render.add_b_lead_in` (`musicmix.dj_render.b_lead_in.1`) per echo/filter out
(`dj_effects`) e build/halving/break (`dj_creative`), B senza stretch: ricostruisce il mix dai deck (verifica: il limitatore
rifatto dà lo stesso PCM) e aggiunge da lì al downbeat la **sola voce di B** (stem Demucs, stesso guadagno di B; modo `stem`,
come il cut di `dj_phrase`), che sul downbeat passa a B intero (guadagni complementari): il beat di B arriva ancora sull'uno.
A «si chiude» (`a_close='auto'`) solo se la sua voce è attiva in più del 10 % del lead-in mentre A suona ancora, altrimenti
resta come base (niente buco); la coda d'eco scende di 6 dB fino al downbeat di B. Il punto udibile è dichiarato in
`meta['b_entry']`; `switch.output_sample` resta sul downbeat (regola delle clip invariata).

**R-B4 [H] (QA).** `dj_qa.6`, controllo `b_entry` (richiesto, severità `fail`, simmetrico di `verse_cut`): nel punto in cui B
diventa udibile (dichiarato, misurato sul deck B o dai metadati) **nessuna parola di B** inizia prima e finisce dopo di oltre
**50 ms** (confini ASR) e la prima parola udibile non è attaccata alla precedente di B («frase iniziata
prima»; soglia = la stessa pausa notata, 3/16 di battito). Stati di copertura: `measured`, `unavailable` (niente parole né stem continuo di B → un pass diventa warn),
`not_applicable` (B a fader). Il runner passa le parole **raffinate** di `dj_phrase` (Whisper incolla le parole: con le
grezze la regola della frase scatterebbe quasi sempre). `b_entry_check="warn"|"off"` per altre scelte.

Fonti (pagine rilette con `tools/libri.py` il 21/10/2026):

- [P] Brewster & Broughton 2003, cap. 10 «How to match phrases» (EPUB, unità 28): «If you cut one off halfway or start halfway
  through a phrase, it will throw your dancers off»; «Cue up your incoming record so its first beat is the start of a phrase».
  Parla di frasi di 4/8 battute della musica, non del verso rap: lo trasferiamo alla frase vocale di B [H].
- [A] Krims 2000, pdf pp. 112–113: lo «upbeat layer» = combinazioni di una battuta «that directly precede points of formal
  articulation (e.g., beginnings of verses, beginnings of refrains)»; BC 4 = «pickup». Fonda il limite di una battuta di
  lead-in e lo spostamento di 1–2 battute prima della sezione [H sul numero].
- [A] Ohriner 2019, pdf p. 83 (nel suo corpus i pickup stanno nella battuta −1) e pdf pp. 217–218 (anacrusi via via più lunghe
  aumentano l'aspettativa della rima): il pickup appartiene al verso che segue.
- [A] Butler 2006 p. 189 (le variazioni «almost always occur just before a metrical or hypermetrical downbeat»), già usato per
  l'808 in `dj_form.3`.
- [E] MCFlow (Condit-Schultz 2017, misura nostra): 42,7 % delle frasi inizia entro 1 battito prima di un downbeat, 13,3 % tra 1 e
  2; solo il 15,2 % sul downbeat ([QUANDO_TAGLIARE](QUANDO_TAGLIARE_2026-10-21.md) §0).
- Nessuna fonte dà «1 battito», «3/16 di battito» né «50 ms»: sono [H], da stimare con i confronti a coppie (torneo dei cue).

## 3. Ripieghi tolti (orchestratore, revisione esterna del 21/10/2026)

- `dj_set.entry_guard`: niente rilassamento di default (sopra).
- `tools/run_dj_set_v2_books.build_set` (quindi anche il set v3): quando tutti i tentativi di un arco falliscono QA/veto non si
  tiene più il «meno grave». Default `SetHooks.on_all_fail = "replan"`: seconda passata con A prolungata alla frase successiva e
  altri ingressi di B compatibili con i vicini (`extended_attempts`), poi l'arco è tolto (costo infinito) e ordine e cue si
  rifanno (`max_replans` = 6); se il set non si chiude, `set.json` dichiara `closed: false` con gli archi tolti e nessun audio.
  `"keep_least_bad"` = comportamento precedente, solo come opzione esplicita. Il runner v1 (`run_dj_set_v1.py`, congelato) non
  è cambiato.

## 4. Verifica a tecnica fissa (`runs/b_entry_fix_v1`)

`runs/b_entry_fix_v1/results.json`, 68 render (67 clip, 1 errore dichiarato). Per rispondere alla domanda «quale cue è
giusto a parità di effetto» (orchestratore, revisione esterna) la **tecnica è fissa** per transizione (quella del set v3) e
variano: **ingresso di B** con l'uscita di A del set (E0 vecchio senza lead-in, E1 nuovo, E2 nuovo ±1 battuta, E3 frase
successiva di B) e, separatamente, **uscita di A** con l'ingresso nuovo di B (X0 = E1, X1 frase precedente di A, X2 frase
successiva = A prolungata). Ogni clip ha QA `dj_qa.6` e critico v4. Clip mp3 192 kbps secondo AGENTS.md (effetti/blend: 10 s
+ transizione + 10 s; tutte `rule_ok`) in `runs/b_entry_fix_v1/clips/`; confronti etichettati in `clips_manifest.json`
(gruppi `tNN_entry` / `tNN_exit`, ordine casuale, chiave in `clips_key.json`) e torneo dei cue in `cue_variants.json`
(`musicmix.cue-variants.1`, con `comparisons`). Pagina generata e **non pubblicata** in
`<scratch>/torneo_cue_v1/` (12 tornei, 67 varianti,
33,8 MB: per stare sotto i 40 MB la copia della pagina è ricodificata a 160 kbps; le clip della run restano a 192).

| T | A → B | tecnica | vecchio: cue di B, punto udibile, prima parola | nuovo: cue, punto udibile, lead-in, prima parola | pickup suonato | QA (`dj_qa.6`/b_entry) vecchio → nuovo | critico v4 |
|---|---|---|---|---|---|---|---|
| 1 | After Hours → MAFIA | echo out | b. 54, 118,28 s, «feet» (frase iniziata, `mid_line`) | spostato b. 52 (cue di frase), udibile 113,58 s (b. 51,84), 0,35 s, «Cleanest» | sì | fail/fail → warn/pass | ok |
| 2 | MAFIA → No Kizzy | build | b. 42, 80,92 s, «in» (`mid_line`) | nessun ingresso pulito entro 16 battute (ASR inglese su brano italiano): invariato | no | fail/fail → fail/fail | ok |
| 3 | No Kizzy → Goosebumps | filter out | b. 10, 18,93 s, taglia «:30» | stesso cue, udibile 18,25 s, 0,69 s, «:30 in the night» («7» resta fuori: E2 a b. 9 lo include) | sì | fail/fail → warn/pass | ok |
| 4 | Goosebumps → 5% TINT | blend 4 | b. 0, intro (fader) | esente (blend) | no | warn/n.a. → warn/n.a. | ok |
| 5 | 5% TINT → Creature | build | b. 48, 81,59 s, «Man» | stesso cue, lead-in 0,03 s (trascurabile) | (sì, 30 ms) | fail/pass → fail/pass (A: «hear») | ok |
| 6 | Creature → BUTTERFLY EFFECT | halving build | b. 20, 81,70 s, «up» (frase iniziata) | spostato b. 19 (battuta prima), udibile 75,71 s (b. 18,24), 2,59 s, «Feel the stuff…» | sì | fail/fail → **pass/pass** | ok |
| 7 | BUTTERFLY EFFECT → Knife Talk | build | b. 24, «type» (sul downbeat) | invariato (pulito) | no | warn/pass → warn/pass | ok |
| 8 | Knife Talk → STARGAZING | echo out | b. 18, 63,98 s, taglia «time!» | spostato b. 14 (cue di frase), «In» sul downbeat | no | fail/fail → fail/pass (A: «shit,»; buco 0,86 s) | warn |
| 9 | STARGAZING → A Lot | filter out | b. 48, 79,96 s, «a» dopo «Pin» | stesso cue, udibile 79,56 s, 0,42 s, «Pin a…» | sì | fail/fail → fail/pass (ritmo: 1 downbeat appaiato) | ok |
| 10 | A Lot → Nonstop | echo out | b. 13, «Catch» (sul downbeat) | invariato (pulito) | no | fail/pass → fail/pass (A: «end») | ok |
| 11 | Nonstop → Sicko Mode | echo out | b. 31, 99,07 s, taglia «said» | stesso cue, udibile 98,22 s, 0,86 s, «So I said…» | sì | fail/fail → **pass/pass** | ok |
| 12 | Sicko Mode → Rich Flex | build | b. 25, 85,16 s, taglia «magic» | stesso cue, udibile 84,79 s, 0,39 s, «and magic…» | sì | fail/fail → fail/pass (A: «yeah») | ok |

Riassunto [E]: `b_entry` vecchio fail 8 / pass 3 / n.a. 1 → nuovo **pass 10 / fail 1 (T2) / n.a. 1**; cue di B: 7 puliti
(di cui 6 con lead-in), 3 spostati (T1, T6, T8), 1 senza alternativa (T2), 1 esente (T4); pickup/lead-in suonato in 7. Verdetto
QA complessivo vecchio fail 10 / warn 2 → nuovo pass 2 / warn 4 / fail 6: i fail rimasti sono **lato A** (parola di A tagliata
all'uscita a effetto: T5, T8, T10, T12, come nel ricontrollo P0), T2 (ingresso di B) e T9 (ritmo misurato sul solo mix con la
voce di B nel lead-in: 1 downbeat appaiato). Critico v4 blocco technical: 11 ok, 1 warn (T8, buco). Su tutte le 67 varianti:
`b_entry` pass 41 / fail 21 / n.a. 5 — le varianti fail sono quelle che la regola sconsiglia (E0 vecchie, cue `mid_line`),
incluse apposta nel confronto. Un errore dichiarato: T10 E3 (lead-in di 2,71 s più lungo della regione dell'eco, 1,64 s:
`add_b_lead_in` rifiuta; nel set quel candidato fallirebbe e si ripianifica).

## 5. Default, versioni, riproduzione

- Acceso di default nel runner del set v3 (`B_ENTRY = {"on": True, "mode": "stem", "a_close": "auto"}`): guardia dei cue di B
  in `filter_pair_v3`, lead-in nel render (`render_v3`), parole raffinate di B nel QA. `--no-b-entry-fix` riproduce il set v3;
  `tools/recheck_set_v3_qa.py` lo spegne per riprodurre i render congelati.
- Versioni: `dj_qa.6`, `musicmix.dj_render.b_lead_in.1` (DJ_RENDER_VERSION invariata: nuova funzione, render di default
  invariati), `musicmix.dj_phrase.b_entry.1`, `musicmix.dj_set.entry_guard.1`; `dj_select.relocate_entry`/`bar_cue` nuove.

## Test

`tests/test_b_entry_fix.py` (11: regola del lead-in, confine con fine verso e pausa notata, astensione, `relocate_entry`,
`entry_guard` senza ripiego e con l'opzione, lead-in sul render di `dj_effects`, `check_b_entry`, `build_set` che ripianifica
invece di tenere un fail e dichiara un set che non si chiude). Mirati (qa, select, set, phrase, v3 bridge, effects, creative,
render): 185 passati. Suite completa: **776 passati, 1 saltato** (19 min 50 s).

## 6. Limiti

- Nessuna clip nuova ascoltata; i verdetti sono integrità tecnica. Le soglie (1 battito, 3/16 di battito, 1 battuta, 50 ms, 10 %, −6 dB) sono [H].
- La regola lascia fuori parole che l'ascoltatore può sentire unite alla frase (T3: «7» prima di «:30»): il confronto E1/E2 lo
  chiede esplicitamente; non si è cambiata la regola per far rientrare quella parola (AGENTS.md).
- Il lead-in non può superare la regione dell'effetto (eco di una battuta, build): il render lo rifiuta e il set ripianifica.
- Le parole sono quelle di `runs/dj_set_v3_bridge/phrase` (Whisper `small.en`); per `18k_no_kizzy` (italiano) l'ASR inglese
  congelato è inaffidabile anche se il filtro lo dava per buono (l'ASR italiano rifatto da un altro agente non è nei modelli
  congelati usati qui).
- Il lead-in in modo `stem` dipende dalla separazione Demucs (residui di strumentale nella voce di B); il modo `full` (B intero
  dal lead-in) è un'opzione non verificata.
- I cut restano gestiti da `dj_phrase.phrase_switch` (pickup ≤ 2 battiti, confini p ≥ 0,5): la regola R-B1 è più severa e si
  applica nel set solo tramite `entry_guard`; unificare le due regole è aperto.
- La diagnosi «evento di B tardi» usa i bersagli del piano per eventi (solo struttura allin1/dj_form).
