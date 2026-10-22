# Psicologia dell'aspettativa, tensione e completamento

## Audit integrale dei nove libri — 21 ottobre 2026

Huron: tre accordi in tonalità vicine (stamp210–213) non validano3secondi tra generi né percentile90 timbrico. Aspettativa, sorpresa, reazione e appraisal vanno distinti; descrivere una struttura non ne prova la funzione (stamp372). Margulis:1/4/8 a stamp8 sono secondi, non copie; categorie di Lidov a51 non sono un massimo empirico di loop. Psicologia diagnostica come daAGENTS; nessun nuovo veto di selezione ricavato automaticamente dai libri.

Dettagli, fonti/pagine e criteri di accettazione: [revisione completa](../libri/audit_2026-10-21/REVISIONE_PIPELINE.md).

## Evidenza misurata

**Farbood (2012)** costruisce un modello temporale e parametrico della tensione a partire da due esperimenti: giudizi su parametri musicali e risposte continue su estratti più lunghi. Il punto utile per MusicMix è la memoria temporale: una tensione è una traiettoria con contesto, non un'etichetta di un fotogramma. Il modello combina contributi di armonia, altezza, aspettativa melodica, dinamica, onset, tempo, metro, regolarità e sincopi. È una fonte primaria forte per separare trend e livello istantaneo; non è una validazione su rap/trap.

**Lehne e Koelsch (2015)**, in XML JATS completo da PMC, propongono un quadro domain-independent: conflitto, instabilità, dissonanza o incertezza attivano processi predittivi verso eventi futuri rilevanti. È una teoria che spiega perché un finale aperto può essere percepito come attesa, ma non fornisce una soglia audio né una regola di cambio tra due brani.

**Hansen et al. (2021)** controllano la dinamica informativa di sequenze di toni: N=38 nel compito di ascolto e N=31 per i giudizi di completezza. L'entropia/uncertainty predittiva ha effetto sulla segmentazione e sulla completezza, distinto dalla sorpresa retrospettiva e dalla posizione nella frase. Il trasferimento a una voce registrata è un'ipotesi, non una metrica già dimostrata.

**Turrell et al. (2025)** raccolgono rating continui da 34 partecipanti su break routine EDM. Nel loro compito il build-up aumenta la tensione e il drop atteso è seguito da un calo; la relazione con emozione è mediata dalla variazione di tensione. Questo è il caso più vicino al pattern “cresce → arrivo”, ma è una routine EDM con drop fortemente atteso: un calo di loudness, un confine LinkSeg o un cutoff temporale non sono automaticamente un drop percepito.

**Grosz et al. (2025)** è incluso con DOI e record eScholarship/SAGE, ma il PDF locale non è stato validato: il download ha restituito una pagina CloudFront HTML di 919 byte. L'abstract disponibile propone una grammatica narrativa EDM `Intro/Breakdown → Buildup → Core → Outro/Cut`, analoga a una sequenza di tensione e rilascio. È quindi una fonte teorica/abstract-only per l'organizzazione EDM, non una lettura full text né una misura di confine.

## Teoria da trasferire con cautela

Le fonti convergono su contesto, memoria e incertezza prospettica. Non convergono su un'unica feature acustica. La domanda utile non è “la curva scende?”, ma “quale aspettativa era attiva, quale evento la realizza e con quale grado di incertezza?”. Per MusicMix: mantenere separati trend di A, evento di ingresso B, frase vocale e qualità tecnica.

## Ipotesi ingegneristica verificabile

Per ogni candidato A→B produrre un record con: pendenza e memoria della traiettoria in A; tipo di evento in B; distanza da beat/downbeat; stato di parola/voce; livello/headroom; `UNKNOWN` quando un elemento non è misurabile. La fonte non autorizza una somma in un “resolution score”. Un filtro prudente può astenersi se A ha tensione crescente ma B non presenta un arrivo compatibile o se il join cade dentro una parola.

## Limiti di dominio

La psicologia musicale e i lavori EDM/classici parlano di ascolto e stimoli, non del mixaggio simultaneo di due stems rap. Trap, hip-hop e pop possono realizzare continuità con un pickup, un enjambment o una dissonanza non risolta. Le soglie di build-up/drop non vanno importate nel pool senza annotazioni nuove.

Le copie valide e i fallimenti di fetch sono elencati in [`SOURCES.json`](SOURCES.json); le ancore verificabili sono nel manifest globale. Il PDF Grosz e il PDF Lehne da 519 byte restano in `failed_fetches/` e non sono conteggiati come full text.


## Implementazione V12 (2026-10-20)

Le route separate continuazione/arrivo (`functional-event-v1.2`) sono integrate in `src/musicmix/functional_event.py`. La nota [`v12_functional_event/INTEGRAZIONE.md`](v12_functional_event/INTEGRAZIONE.md) distingue la teoria (Lehne & Koelsch, Kurby & Zacks, Adams) dalle soglie ingegneristiche e riporta la validazione su 8400 candidati del pool. Il gate "A diretto" su 8 beat è vicino al livello di rumore.
