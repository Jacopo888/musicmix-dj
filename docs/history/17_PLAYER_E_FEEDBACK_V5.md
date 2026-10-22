# Player, ripetizioni e feedback V5

## Player aggiornato

Il server precedente rispondeva soltanto con l'intero WAV e non supportava HTTP Range o HEAD. Questo impediva la navigazione affidabile con il player del browser. Ora restituisce byte range con stato 206, `Accept-Ranges`, `Content-Range` e lunghezze corrette, supporta HEAD e rifiuta range fuori file con 416. La risposta viene trasmessa a blocchi senza caricare tutto il file per ogni seek.

Tutte le quattro interfacce V4 sono state aggiornate con pulsanti −5 s, +5 s e salto al raccordo (10 s), oltre alla barra nativa. Gli URL restano gli stessi, così come WAV, ordine A/B, chiavi, identificativi e giudizi salvati. `player_revision.json` registra solo la modifica HTML; le vecchie run restano documenti storici con gli hash del codice originale.

Verifica sui server riavviati: 36 WAV, HEAD e 108 richieste parziali (inizio/centro/fine) corrispondenti ai byte dei file locali. Risultato in `runs/player_transport_audit_v5.json`. Le prove verificano il trasporto e l'HTML; non costituiscono ascolto percettivo.

## Sorgenti e ripetizioni

L'audit `runs/audio_lineage_audit_v5.json` ricostruisce originali, canonici, manifest, chiavi e PCM dei render. Sessanta WAV delle prove storiche sono compatibili con i due canonici e i cue dichiarati; non emerge un incrocio di direzione o l'introduzione di un altro file sorgente. L'audit non riconosce percettivamente un brano dall'ascolto.

La ripetizione ha due livelli differenti:

- V4 diversifica i cue selezionati su ciascun brano, ma ciò non prova che i segmenti abbiano un diverso carattere musicale. Lo spacing di 8 s non è una segmentazione in strofa/ritornello, ruolo vocale o arco.
- La confezione A/B confrontava la stessa proposta prima con V3 e poi con il controllo casuale: lo stesso WAV nuovo veniva quindi ascoltato due volte. Inoltre due coppie V3 sono sopravvissute fra le proposte V4. Il feedback “stesso di prima” è coerente con questi riusi, non viene scartato perché i cue del top-K sono distinti.

Per il prossimo ascolto occorre contare esposizioni per hash e famiglia di segmento, distinguere chiaramente prove nuove e riferimenti, e non riempire la sessione con lo stesso candidato contro più controlli. Le prove già giudicate non vengono rimescolate o riscritte.

I dodici giudizi V4 sono importati in `giudizi json/feedback_forward_v4.json` e `feedback_reverse_v4.json`. Un caso negativo concreto è forward `rank_1/B`: Reminder 213,090 s → Too Many Nights 182,200 s, descritto con crescita seguita da un ingresso piatto e con parola tagliata. Il report automatico non rilevava overlap con parole ASR note: il feedback evidenzia quindi anche il limite reale del guard vocale. Caso conservato in `examples/feedback/v4_forward_rank1_motion_and_wordcut.json`, senza attribuire arbitrariamente il taglio alla voce uscente o entrante.

## Come proseguire

1. Importare e collegare le annotazioni V4 alle chiavi conservando i commenti su andamento, ruoli vocali, ripetizione e sovrapposizione. Nessun aggiustamento automatico sul solo caso sufficiente.
2. Valutare l'arco su 8/16/32 beat prima/dopo: energia, densità/attacchi, movimento melodico e armonico, stabilità/cadenza, ruoli voce/strumentale. Prima registrare le misure e gli errori osservati, poi confrontare un reranking separato. Un calo di energia non prova una risoluzione; un aumento non prova suspense. Il primo profilo osservativo è in `runs/motion_audit_v5.json` e `docs/history/16_MOTION_E_PROSSIMI_PASSI.md`.
3. Introdurre diversità fra famiglie di segmenti e contesti musicali, oltre alla distanza temporale. L'obiettivo è trovare alternative plausibili, non escludere a priori tutte le transizioni fine/inizio o rappato/strumentale.
4. Su cue già giudicati adatti, confrontare separatamente handoff attuale e sovrapposizione breve/beat-sincrona. Il renderer attuale usa due rampe sequenziali da 12 ms: può sembrare un taglio anche con cue corretti. Questo intervento va misurato a cue fissi, senza confonderlo con il miglioramento del ranking.

Il motion non viene rimandato a un ritocco estetico: riguarda la scelta dei punti. Restano necessari confini vocali più affidabili sul cantato, ruoli/sezioni musicali e una valutazione su un insieme più ampio dopo aver stabilizzato questi criteri.
