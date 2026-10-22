# Modelli preaddestrati e segmentazione

## Audit integrale dei nove libri — 21 ottobre 2026

Riusare BeatThis/allin1/stem e port ufficiale Ohriner già presenti; nessuna evidenza dai libri impone un nuovo modello end-to-end da ricostruire. Il port simbolico non valida il front-end vocale audio. Prima di aggiungere un ulteriore modello, misurare errore, copertura e contributo fuori campione dei componenti esistenti; annotazioni di eventi e proprietà del renderer sono colli di bottiglia prioritari. Nessuna nuova inferenza pesante in questo audit.

Dettagli, fonti/pagine e criteri di accettazione: [revisione completa](../libri/audit_2026-10-21/REVISIONE_PIPELINE.md).

## Evidenza misurata

**CUE-DETR (Argüello et al., 2024)** tratta cue-point estimation come object detection su spettrogrammi. EDM-CUE contiene 4.710 tracce e circa 21.461 cue manuali; il best F1 pubblicato è 0,46 nel compito single-track EDM e il pattern più rappresentato è 16/8 battute. È un generatore di candidati, non un decisore A→B.

**Zehren et al. (2020)** usano novelty di energia, timbro, onset di batteria e armonia in regole per switch point su M-DJCUE. La precisione intorno all'86% con tutte le regole è una proprietà del loro set EDM e si accompagna a recall limitato. La fonte suggerisce feature interpretabili e astensione, non un criterio di gusto.

**Raveform (Kim et al., 2026)** separa struttura metrica e funzionale su EDM dentro mix: 4.902 mix, 56.873 tracce uniche, 1.423 annotate, 53.780 transizioni disponibili secondo gli ancoraggi del paper. Mostra anche il degrado cross-dataset. Il vocabolario intro/buildup/drop/breakdown/cooldown/outro è utile per EDM, ma non è una tassonomia rap.

**LinkSeg** è stato realmente eseguito nel run V10 con commit ufficiale fissato: 86 sezioni e 80 confini interni sui sei brani. I due render raw archiviati hanno mostrato salti RMS circa 11,89 e 11,58 dB, uno con bordo quasi silenzioso di almeno 50 ms. Questo è un controllo tecnico concreto: il provider non è una prova di transizione buona. Il paper non è presente nella copia locale; il manifest conserva repository, commit e limite di lettura.

## Modelli, dati e trasferimento

I benchmark riportati misurano cue single-track o confini di struttura. Non validano “resta in A/cambia in B”, non validano la risoluzione della tensione e non contengono stems rap simultanei. La differenza EDM→rap è quindi una incertezza di dominio, non un dettaglio da risolvere con una soglia.

## Ipotesi ingegneristica

Usare CUE-DETR/Zehren/LinkSeg solo per aumentare la copertura dei candidati, con `provider`, hash, offset e incertezza espliciti. Intersecare i candidati con `word_guard`, beat/downbeat, struttura e diagnostica renderer. Un confine senza protezione di livello o parola viene marcato come rischio; nessun embedding o label funzionale diventa un oracolo di compatibilità.
