# Benchmark esterno per i cue di WHENmix

## Esito concreto

È stato costruito ed eseguito un benchmark riproducibile sui **134 file JAMS completi di M-DJCUE**. Il repository ufficiale è `MZehren/M-DJCUE` alla revisione `7fd692c793a4a4744ab61e409854302269b0974c` (2019-11-05). Il link `MarcoStringa/M-DJCUE` indicato nel task restituisce HTTP 404; il repository scoperto dai paper è `MZehren/M-DJCUE`. Non è stato scaricato audio.

Il benchmark consuma un JSON di predizioni e valuta:

- tempo del cue con tolleranza configurabile (default 0,5 s, la stessa del paper);
- ruolo `IN` / `OUT`;
- accordo tra annotatori;
- copertura delle regioni;
- astensione esplicita o per predizione mancante;
- baseline di sola posizione, addestrate solo sullo split di training;
- split deterministico raggruppato per registrazione/artista/versione.

## Dati e schema

- 134 tracce, 1.849 eventi `cue_point` grezzi: 1.037 `IN`, 812 `OUT`.
- 402 schede annotatore: 3 annotatori per traccia, 5 annotatori nel dataset.
- Dopo merge a 0,5 s: 1.082 cluster (565 `IN`, 517 `OUT`).
- Supporto dei cluster: 547 con 1 annotatore, 305 con 2, 230 con 3.
- Tutti i file hanno un solo namespace, `cue_point`; non ci sono beat/downbeat.
- Le annotazioni `IN` sono vincolate all'intro fino al primo beat del core; le `OUT` non prima del core. Il dataset non annota tutti i punti possibili.

Licenza del repository: GPL-3.0 per annotazioni e metadata. L'audio è copyrighted e non è distribuito. Il paper ISMIR 2019 late-breaking è CC BY 4.0; il preprint arXiv 2007.08411 ha licenza arXiv non esclusiva.

Quattro tracce in vinile hanno eventi oltre la durata dichiarata nel metadata. Il benchmark usa `duration_s = max(durata dichiarata, ultimo evento annotato)` e registra ogni caso in `reading_issues`; non li nasconde e non scarica audio per verificarli.

## Split

Split deterministico con seed `20261020`, frazione test 0,3:

- 119 componenti di raggruppamento (unione per stesso artista e stessa famiglia di versione del titolo);
- 94 tracce train, 40 tracce test, 36 gruppi test;
- 83 artisti train, 36 artisti test, **zero sovrapposizioni** di artista o famiglia di versione.

Il test non è usato per addestrare le baseline. Le baseline usano solo la mediana delle posizioni dei cluster di training e la durata/metadata della traccia test.

## Evidenza misurata in questo run

Baseline di sola posizione, tolleranza 0,5 s, split test congelato:

| Baseline | Predizioni in regione | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mediana della frazione di durata | 73 | 3 | 70 | 343 | 0,0411 | 0,0087 | 0,0143 |
| mediana del tempo assoluto | 70 | 2 | 68 | 344 | 0,0286 | 0,0058 | 0,0096 |

La baseline a frazione di durata è meno debole di quella assoluta, ma entrambe sono vicine al caso: la posizione mediana di un cue non basta a colpire uno qualunque degli 8-9 cue annotati per traccia entro 0,5 s. Questo è un risultato sulla difficoltà del compito e sulla plausibilità della pipeline di valutazione, **non** una validazione di un algoritmo audio di WHENmix.

Accordo tra annotatori su tutte le 134 tracce (stesso brano/label, matching greedy entro 0,5 s):

- pairwise F1 medio `IN` 0,621 (402 coppie);
- pairwise F1 medio `OUT` 0,424 (400 coppie);
- pairwise F1 medio complessivo 0,523 (802 coppie);
- accordo sulla presenza della label: `IN` 1,00, `OUT` 0,96.

Copertura empirica sul test:

- regione `IN`: 25,9% della durata totale, 40/40 tracce;
- regione `OUT`: 56,9% della durata totale, 40/40 tracce;
- gap non valutato tra fine regione `IN` e inizio regione `OUT`: media 97,8 s, mediana 75,3 s.

Le regioni sono proxy: `IN = [0, max(IN)+0,5 s]`, `OUT = [min(OUT)-0,5 s, durata]`. Non sono la verità sulla copertura umana e non autorizzano a trattare ogni punto non annotato come negativo.

## Risultati riportati dagli autori, non riprodotti

Il paper 2019 dichiara le linee guida e il vincolo intro/core; il paper 2020 riporta 86% precision e 49% recall con tutte le regole, su questo dataset EDM. Sono risultati degli autori, non riprodotti qui. La regola “downbeat” del paper non è verificabile in questo rilascio perché manca la griglia di beat/downbeat.

## Teoria e ipotesi ingegneristica

Ipotesi, non ancora validata: usare M-DJCUE come test di *timing* dei cue e come controllo di astensione; usare le regioni empiriche per non penalizzare candidati in zone non annotate; tenere separate le metriche di cue da quelle di qualità del blend, di frase e di voce. Un cue corretto entro 0,5 s non dimostra che il raccordo A→B sia musicalmente buono.

## Limiti e blocker

- Solo EDM 1987-2016, non rap/trap; nessuna evidenza di trasferimento al dominio WHENmix.
- Nessun audio: il benchmark non può verificare allineamento campione, loudness, tagli di parola o artefatti di render.
- Nessuna coppia A→B e nessuna transizione di DJ professionista allineata: le annotazioni sono per traccia e indipendenti dalla traccia successiva. Non è stato inventato alcun abbinamento.
- Niente downbeat/beat: baseline downbeat non disponibile.
- Le regioni sono proxy empiriche; il gap tra intro e core/outro non è valutato.
- La baseline di posizione non è un algoritmo audio e non va presentata come tale.
- Il benchmark non produce clip: la regola di playback 20 s (10 s per lato) resta un requisito del prodotto, non una metrica di questo dataset.
