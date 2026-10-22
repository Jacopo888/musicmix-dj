# Eventi funzionali, aspettativa e continuazione A -> B

Aggiornato al 2026-10-20. Le copie, gli hash SHA-256, le licenze, i testi estratti e le sezioni effettivamente lette sono in [`sources.json`](sources.json). `originals/` conserva i byte scaricati; `texts/` conserva l'estrazione, che non sostituisce la lettura. Scaricato in questo turno: **858.603 byte** (6 fonti, 4 full text, 2 solo abstract).

## Evidenza misurata

**Hansen, Kragness, Vuust, Trainor e Pearce (2021)**, *Psychological Science* 32(9), DOI [10.1177/0956797621997349](https://doi.org/10.1177/0956797621997349). Letto il PDF completo ospitato dall'autore: abstract, introduzione, due esperimenti, risultati, discussione. Nel compito implicito di ascolto auto-regolato (N = 38) i partecipanti indugiano più a lungo su toni con entropia/incertezza alta; nel compito esplicito (N = 31) sequenze che terminano su toni ad alta entropia sono giudicate più complete. Gli autori riportano che gli effetti sono indipendenti da sorpresa retrospettiva e posizione nella frase. È un risultato su sequenze di toni controllate, non su mix rap/trap né su una soglia audio.

**Sridharan, Levitin, Chafe, Berger e Menon (2007)**, *Neuron* 55(3), DOI [10.1016/j.neuron.2007.07.003](https://doi.org/10.1016/j.neuron.2007.07.003). In questo archivio è disponibile solo l'abstract (editore paywall). Abstract: fMRI durante sinfonie naturali, finestre di 10 s attorno ai passaggi tra movimenti; una rete ventrale risponde alla salienza del cambiamento e una rete dorsale aggiorna attenzione/memoria di lavoro. È evidenza misurata di segmentazione a un confine strutturale, non una prova che il confine sia una risoluzione musicale.

**Farbood (2012)**, *Music Perception* 29(4), DOI [10.1525/mp.2012.29.4.387](https://doi.org/10.1525/mp.2012.29.4.387). Solo abstract/metadati in questo archivio. L'abstract descrive due esperimenti e un modello temporale della tensione con finestra percettiva mobile e *trend salience*; la tensione è multidimensionale e temporale. Non autorizza a usare una singola pendenza come tensione.

## Teoria

**Kurby e Zacks (2008)**, *Trends in Cognitive Sciences* 12(2), DOI [10.1016/j.tics.2007.11.004](https://doi.org/10.1016/j.tics.2007.11.004). Letto il full text PMC: abstract, *Making sense by segmenting*, *Events and their boundaries*, *Segmentation tasks*, automaticità, gerarchia, memoria. La Event Segmentation Theory propone che i confini emergono quando l'errore di predizione aumenta transitoriamente e il modello dell'evento viene aggiornato; la segmentazione è multi-scala e gerarchica. Per il nostro comparatore: un evento di ingresso in B è un cambiamento di feature predette, non un calo di RMS o un'etichetta di sezione. La teoria non fornisce soglie audio.

**Lehne e Koelsch (2015)**, *Frontiers in Psychology* 6:79, DOI [10.3389/fpsyg.2015.00079](https://doi.org/10.3389/fpsyg.2015.00079). Letto il full text Europe PMC: abstract, definizione, componenti, previsione, misure e discussione. La tensione/suspense è definita come stato affettivo legato a conflitto, dissonanza, instabilità o incertezza, orientato al futuro e con urgenza di risoluzione. È un modello domain-independent, non una regola di mixaggio.

**Adams (2020)**, *Music Theory Online* 26.2, DOI [10.30535/mto.26.2.1](https://doi.org/10.30535/mto.26.2.1). Letto il full text HTML: abstract, definizioni, rima, metro, parametri armonici, sintattici e motivici, interazioni, conclusione. La frase hip-hop è definita da direzione e chiusura; la norma dei couplet di due battute è così comune da definire solo debolmente una frase. La sintassi può prevalere sulla rima; un respiro o una rima finale non bastano da soli. Questo sostiene la distinzione fra pickup/preparazione e arrivo, e vieta di trattare un gap di parola come chiusura di frase.

## Ipotesi ingegneristica implementata

Il comparatore in `src/musicmix/functional_event.py` usa quattro profili con battiti osservati: `A_before`, `A_after`, `B_before`, `B_after`. Per ogni dominio (`energy`, `events`, `bass`, `timbre`, `tonal_proxy`) calcola direzione pre, direzione stabilizzata post, contrasto locale `B_before -> B_after` e ritenzione post-ingresso. La continuazione richiede una maggioranza pesata di domini concordi e rifiuta inversioni salienti; non basta che due domini coincidano. L'evento locale richiede un contrasto positivo in almeno due domini e una componente fra energia/eventi/basso, poi verifica persistenza/stabilizzazione. La preparazione/pickup è distinta dall'attacco post-ingresso; se la preparazione non viene portata attraverso l'ingresso, il risultato è `preparation_not_realized` con punteggio negativo. Il moto sostenuto è l'alternativa senza salto locale.

Le soglie sono ipotesi ingegneristiche versionate, non probabilità calibrate. `confidence` resta `None`; in caso di profili mancanti, conflitto o assenza di evidenza il comparatore si astiene e non produce un punteggio. Le etichette di sezione sono usate solo come contesto di cambiamento/continuità: nessuna etichetta, incluso `chorus`, viene mappata a risoluzione. Non vengono usati identità della traccia, similarità di forma d'onda o un premio per A nativo. Il contratto `FunctionalEventScorer.score/rerank/rerank_pool` è utilizzabile dal root per riordinare un pool reale.

## Misura fatta in questo turno

Pilot su PCM locale `inputs/real_01.wav` (A) e `inputs/real_02.wav` (B), cue a 10 s, clip di ascolto da 20 s (10 s per lato). I battiti sono **onset osservati** dal rilevatore spettrale esistente, non una beat grid validata; nessun battito è stato inventato. Dettagli e hash in `outputs/pilot/results.json`.

- B reale: `pickup_or_preparation`, adjustment `+0.38`; il contrasto locale è supportato solo dal basso.
- B con i primi 0,42 s dopo il cue attenuati di 12 dB: `preparation_not_realized`, adjustment `-0.30`; `energy` ed `events` diventano rilasci negativi. B_before è invariato.
- B con il primo evento sostituito da un onset reale donatore di B: `preparation_not_realized`, adjustment `-0.18`.
- Controllo sintetico con la stessa pendenza lineare di B_after ma senza bump locale: `post_entry_attack_transient`, `-0.41`; con bump locale: `post_entry_attack_stabilized`, `+0.70`. Questo isola l'evidenza dell'evento `B_before -> B_after` dal semplice confronto di due pendenze.
- Controllo con due domini concordi e due inversioni salienti: `conflict_entry_vs_continuation`, nessun punteggio.

## Limiti e cose non misurate

Non ci sono giudizi umani, rating dell'utente o ascolto validato. Non ci sono beat/downbeat provider validati, vocali separati o completion evidence per queste clip; pertanto la parte voce/frase resta `unknown`. I due brani sono ritagli locali di 20 s, non un pool. Sridharan 2007 e Farbood 2012 sono solo abstract in questo archivio. Le soglie e le etichette non sono state validate su una popolazione o su un pool: il trasferimento resta **unvalidated**. Non si afferma 10/10, SOTA, risoluzione musicale o equivalenza chorus = closure.
