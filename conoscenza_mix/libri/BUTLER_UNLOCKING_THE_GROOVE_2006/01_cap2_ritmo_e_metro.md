# Butler 2006 — Cap. 2 "Conceptualizing Rhythm and Meter in EDM" (pp. 76–115)

## Tesi
Nella EDM il metro "as a structure … appears unproblematic; as a phenomenon, it is rich" (p. 115). I generi si dividono in **four-on-the-floor** e **breakbeat-driven**; i ritmi in **even**, **diatonic**, **syncopated**; il metro si costruisce per strati che entrano e escono.

## Categorie (con pagina) [A]
- **Four-on-the-floor vs breakbeat** (pp. 77–79): four-on-the-floor = cassa in quarti continui (techno, house, trance). Breakbeat = pattern di batteria completi campionati da break funk (Funky Drummer, Amen), accelerati (es. da 92 a 160 BPM nel jungle; +20–30 BPM nel big beat), con microtiming espressivo, accenti sui tempi deboli e irregolarità; il produttore cambia break in certi punti per non far "accomodare" l'ascoltatore (p. 79).
- **Durate "pure-duple"** (p. 80): i cicli sono di 2, 4, 8, 16, 32, 64 unità dal sedicesimo fino a **ipermisure di 16 battute** (con Fikentscher: frasi di 4, 8, 16, 24, 32 battute).
- **Ritmi even** (p. 81): cassa 1/5/9/13 (quasi universale nel four-on-the-floor), backbeat 5/13 (rullante o battimani, house/garage), controtempo 3/7/11/15 (hi-hat aperto; stab nella trance), hi-hat in sedicesimi.
- **Ritmi diatonic** (pp. 81–84): divisioni asimmetriche "massimamente uniformi": 3+3+2, 3+3+3+3+4, ecc.; si allineano ai quarti solo sull'uno e sul levare → hanno carattere **anacrusico** (p. 92).
- **Ritmi syncopated** (pp. 84–88): tipici dei breakbeat; i colpi "danzano" attorno al battito senza cadervi e così lo richiamano; nel complesso il composito esprime chiaramente il metro.
- **Loop** (pp. 89–90): unità fondamentale = pattern ripetuto associato a uno strumento; da un sedicesimo fino a 16 battute.

## La cassa, i breakdown, il "withholding the beat" (pp. 90–92) [A]+[osservazione]
- "The most common phenomenon involving the beat is the removal of the bass drum — followed, of course, by its eventual return" — presente "at some point in nearly every track" (p. 90).
- **Breakdown** = sezione **senza cassa**, "typically lasting for at least four measures", a volte oltre un minuto; nella trance spesso lunga e "ambient" (synth tenuti, archi), spesso con manipolazioni timbriche come delay crescente (p. 91). (Butler distingue *break* = sezione di sola percussione da *breakdown* = assenza di cassa, p. 90 n. 15.)
- **Withholding the beat** = il DJ toglie la cassa dal vivo: la pista esita, poi al ritorno "they dance with greater energy than in any other portion of the track" (p. 91).
- **Posizione**: i withholding avvengono di solito alla **fine di un'unità metrica**: ultimo battito di una battuta, **ultima battuta di un gruppo di 4**, **ultime 4 battute di un gruppo di 16** (p. 91). Orientamento anacrusico generale della EDM, in rapporto con "the One" del funk (p. 92).

## Tessitura (pp. 92–96) [A]
- Tessitura eterogenea, strati quasi paritari e ben distinti per timbro e registro; ogni strato ha un ritmo proprio (p. 92–93).
- Nel mix i due dischi si fondono in "the third record"; il DJ "rides the mix"; con le EQ manipola gli strati del singolo disco (p. 93).
- "Jerical" (Jeff Mills): gli strati entrano uno alla volta (cassa → hi-hat → secondo hi-hat → rullante → battimani → riff); il **completamento della tessitura coincide con il completamento del metro** (4/4 chiaro solo all'arrivo dei riff) (pp. 94–96). Un taglio di tutte le parti per due battiti, alla 16ª ripetizione dei battimani, segna la prima articolazione formale (p. 94).

## Processi metrici (pp. 110–112)
- Inizi **sottodeterminati**: spesso solo cassa (Panikattack) o cassa + hi-hat: si capisce la suddivisione ma non il raggruppamento in battute (pp. 110–111). Poi compare un raggruppamento di mezza battuta, infine la battuta intera.
- DJ Shiva conta in battiti (64 battiti, non 16 battute) (p. 114).

## Regole operative ricavate
| # | Regola | Pagina | Etichetta |
|---|---|---|---|
| R1 | Unità strutturali in potenze di 2 fino a 16 battute: cue e uscite su queste griglie | p. 80 | [A] |
| R2 | Breakdown = assenza di cassa ≥ 4 battute | p. 91 | [A] (definizione) |
| R3 | Il rientro della cassa dopo una sua assenza è un picco; gli svuotamenti cadono alla fine di unità di 4 o 16 battute | p. 91 | [A]+osservazione |
| R4 | Four-on-the-floor vs breakbeat è la principale divisione percepita dai fan | pp. 77, 115 | [A] |
| R5 | Inizio dei brani: strati che entrano progressivamente, metro inizialmente sottodeterminato | pp. 94–96, 110–111 | [A] |

## Feature computabili (implementate in `src/musicmix/dj_form.py`)
- `kick_presence` per battuta (energia della banda della cassa sullo stem drums o, in mancanza, sul mix) → breakdown = run ≥ 4 battute senza cassa (R2).
- `drop` = prima battuta con cassa dopo un breakdown/withholding (R3).
- `layers_active` per battuta (stem drums, bass, other, vocals sopra soglia) → curva di accumulo all'inizio del brano (R5) e "build".
- `pulse_type`: four-on-the-floor vs breakbeat stimato dalla regolarità degli onset di cassa sui quarti (R4) — [H], soglie non validate.

## Limiti
Analisi su pochi brani techno con trascrizioni dell'autore; "fine di unità metrica" e "picco al ritorno" sono osservazioni etnografiche, non misure. L'hip-hop non è analizzato.
