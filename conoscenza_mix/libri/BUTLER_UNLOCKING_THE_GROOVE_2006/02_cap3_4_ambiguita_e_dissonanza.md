# Butler 2006 — Capp. 3 "Ambiguity" (pp. 121–136) e 4 "Metrical Dissonance" (pp. 138–174)

## Tesi
Ambiguità e "dissonanza metrica" (strati che non si allineano) sono risorse estetiche della EDM, non difetti: chiedono all'ascoltatore e al ballerino di costruire attivamente il metro (p. 136). L'ambiguità è **processuale**: esiste nel momento, anche se poi si risolve (p. 122).

## Cap. 3 — Ambiguità [A]
- **Ambiguità d'inizio** (pp. 123–128): il tipo di metro è chiaro ma non dove cade l'uno (Carl Craig "Televised Green Smoke", James Ruskin "Connected"). Ascoltatori "conservatori" e "radicali" (p. 125). L'**entrata della cassa** spesso fissa un'interpretazione metrica decisa (p. 127) — talvolta una terza, diversa dalle precedenti.
- Un DJ può far partire un disco da qualunque punto: provare a "drop the needle" in punti diversi mostra che la stessa sezione si sente con l'uno in posti diversi (p. 127–128 e n. 10).
- **Ambiguità di tipo metrico** (pp. 128–136): sottodeterminazione (un solo strato), diatonic vs even, reinterpretazione all'ingresso di un nuovo strato, due metri equivalenti contemporanei (Kenny Larkin "Track", 9/8 contro 4/4).
- **Passaggio chiave per il mix** (p. 129): i passaggi sottodeterminati "often occur at the beginning or end of a track, where they enable smoother transitions between records. Stripping records down to their beats alone makes it much easier to match beats, and having a relatively small number of textural elements decreases the likelihood of a clash created by […]"

## Cap. 4 — Dissonanza metrica [A]
- Modello di Krebs: strati di moto; dissonanza di **spostamento** (Dn+k) e di **raggruppamento** (Ga/b) (pp. 138–139, 105–107).
- **Turning the beat around (TBA)** (pp. 140–151): un nuovo strato (di solito una cassa/percussione grave, più forte) entra spostato e **sposta l'uno percepito** (Underworld "Cups": la cassa entra un ottavo prima; dopo 8 battute un secondo synth conferma il nuovo uno, p. 142). "the tendency of low drumbeats to occur on the beats in EDM (and of hi-hats to occur on the offbeats)" (p. 142 n. 4). Il TBA è associato soprattutto all'**aggiunta** di strati (p. 151).
- Juan Atkins, mixando "Reborn" come ottavo brano di un set, crea un downbeat due battiti prima eliminando mezza battuta (p. 149) [osservazione di un mix].
- **Double copies** (pp. 151–154): due copie dello stesso disco sfasate, usate al **climax** (DJ Shiva: "find the most absolutely floor-wrecking track … then do double copies", p. 151); la seconda copia entra a metà del ciclo (sull'ottavo in levare se lo strato dei quarti è forte) (p. 153). Differenza da Reich: nella EDM si raggiunge il punto di spostamento voluto e ci si resta; **lo sfasamento graduale suggerisce un beatmatch fallito** ed è difficile da ballare (p. 154). Il "phasing" dei DJ (copie quasi in sincrono) dura poco e resta un ritardo, non una separazione (p. 154 n. 16). Usare due copie per allungare l'intro è un'altra tecnica (p. 152 n. 13).
- Gruppi 3 contro 4 (terzine, 3+3+2) e pattern sovrapposti di lunghezza diversa (pp. 154–171); le dissonanze nella EDM tendono a diventare **permanenti** nel brano, non ornamentali (p. 169).
- Valore culturale: disorientamento, effetto "psichedelico", rapporto con la tecnologia (pp. 172–174).

## Regole operative ricavate
| # | Regola | Pagina | Etichetta |
|---|---|---|---|
| D1 | Mixare dove la tessitura è ridotta (pochi strati, solo ritmo): meno scontri di ritmo e altezza | p. 129 | [A]+[P] riportata |
| D2 | L'ingresso della cassa fissa (o sposta) il downbeat percepito: verificare il downbeat dopo l'ingresso della cassa, non solo nell'intro | pp. 127, 142, 149 | [A] |
| D3 | Lo sfasamento progressivo tra due tracce è percepito come errore di beatmatch | p. 154 | [A]/osservazione |
| D4 | Effetti di raddoppio/sovrapposizione si riservano al climax | p. 151 | [P] riportata |

## Implicazioni per MusicMix
- D1 → feature `layer_count` per battuta in `dj_form` e costo in `dj_set` per uscite/ingressi dove la tessitura è densa (`w_dense_overlap`, profilo `books`).
- D2 → **rischio misurabile**: negli intro sottodeterminati Beat This! può scegliere un downbeat che la cassa poi smentisce. Proposta per `dj_critic`/QA: confrontare la fase del downbeat stimata nell'intro con quella dopo l'ingresso della cassa.
- D3 → conferma l'importanza di uno stretch costante e di griglie corrette nel blend (deriva = difetto grave).

## Limiti
Analisi di brani techno anni '90 con l'ascolto dell'autore; "conservatori/radicali" è una tipologia informale. Nessuna verifica su hip-hop/pop.
