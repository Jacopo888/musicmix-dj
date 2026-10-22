# Esperimento V7: scegliere l’ingresso mantenendo fissa l’uscita

Questo pilot prova un criterio aggiuntivo per la continuità del movimento musicale. Non costituisce una misura validata di suspense o risoluzione. Il ranking stabile V4 rimane disponibile come riferimento.

## Disegno del confronto

Ogni confronto contiene lo stesso estratto uscente di 10 secondi. Cambia il punto di ingresso del secondo brano: scelta del punteggio locale V4 contro scelta del punteggio locale più evidenza sperimentale. Il raccordo è a 10 secondi. I due ingressi distano almeno 8 secondi nella sorgente. Tra confronti diversi, le uscite e gli ingressi distano almeno 12 secondi. Massimo tre confronti per direzione, senza allentare automaticamente questi vincoli.

Il rendering resta quello precedente: 10 secondi per lato, raccordo sequenziale con rampe di 12 ms, guadagno 0,9. L’eventuale sovrapposizione dei due brani richiede un esperimento successivo a cue fissi, per distinguere il contributo della selezione da quello del mixer.

Le posizioni A/B sono bilanciate e nascoste. La ripetizione dei primi 10 secondi all’interno del confronto è intenzionale. Nessun WAV completo viene riutilizzato tra prove. I giudizi chiedono anche se ciascun passaggio continua, risolve o interrompe l’aspettativa; restano annotazioni umane.

## Criterio e limiti

`expectation.py` confronta traiettorie su 8/16/32 beat: energia, eventi, bassi, timbro e proxy cromatico. Una continuazione richiede almeno due domini con direzione coerente oltre la soglia dichiarata. Questi descrittori restano correlati e non equivalgono a intenzione musicale.

Il peso aggiuntivo è fissato a 0,20 come ipotesi sperimentale. Unknown resta neutrale; un’interruzione supportata deve avere segno negativo. La somiglianza di chroma, il solo calo RMS e l’allineamento al beat non producono una risoluzione. La closure richiede un provider esplicito di arrivo di frase/struttura, oggi assente: il pilot attuale può quindi cercare continuità senza dichiarare di aver riconosciuto una vera risoluzione.

La modalità è `offline_reference`: la direzione del seguito naturale di A contribuisce all’evidenza di continuazione. Non si tratta di un modello causale per un flusso live; non è richiesto che B copi quel seguito.

Il caso negativo V4 213,090 → 182,200 e il primo riferimento sufficiente 192,100 → 16,420 non vengono usati per ottimizzare soglie o pesi. La storia degli ascolti serve esclusivamente a escludere regioni già giudicate.

## Diversità e tracciabilità

Il registro ricostruisce 23 geometrie dagli audio effettivamente presenti nei quattro JSON normalizzati e confermati. Gli SHA collegano audio e timestamp; zero sorgenti irrisolte. Per questo pilot si escludono candidati quando una delle due regioni è già presente nel registro entro 8 secondi, anche nell’altra direzione. Si intende quindi “non presente negli ascolti giudicati”, non una garanzia che l’utente non l’abbia mai sentita altrove.

Il pool su tutti i beat conserva 4512 record ammessi, corrispondenti a 2256 coppie di cue. Le varianti di finestra vengono confrontate senza raddoppiare la stessa geometria nell’ascolto. Dopo il filtro di novità restano 27 coppie forward e 47 reverse. Il baseline sceglie nell’identico pool eleggibile della policy.

La prima esportazione della cache completa è fallita per disco pieno. Sono stati compressi con NTFS, senza modificarne contenuto o percorso, sette JSON storici. La nuova cache usa gzip e checkpoint dei profili per limitare spazio e lavoro ripetuto. I modelli, i brani e i giudizi sono conservati.

Comandi riproducibili dalla cartella del progetto:

```powershell
python tools/update_audition_history.py
python tools/run_expectation_experiment.py
python tools/audit_expectation_experiment.py runs/expectation_v7
python -m pytest tests -q
```

Ogni esecuzione richiede una cartella nuova e assegna un identificativo distinto. Cue e punteggi sono deterministici sugli stessi input; il run ID identifica invece la sessione di raccolta. Il report conserva hash di input, codice, sorgenti e WAV, snapshot del registro, parametri, motivi di esclusione e chiave privata A/B.

## Ricerca e lavoro ancora aperto

La revisione critica è in `19_VERIFICA_POLICY_MOTION_V7.md`; i 15 riferimenti e i dettagli di lettura sono in `../ricerca/MOTION_SUSPENSE_RICERCA.md`. Restano da verificare percettivamente le nuove scelte, l’accordo tra scale temporali e il riconoscimento di frasi/arrivi musicali. I guard vocali precedenti non garantiscono ancora confini corretti sul cantato; il forced alignment specifico resta aperto.

## Risultato effettivo

La cache `runs/candidate_trajectories_v7_allbeat.json.gz` è completa: 4512 record, 2256 coppie, 162 profili; 39 riutilizzati e 123 calcolati, zero errori di validazione. Le classificazioni descrittive sono 1172 continuation e 3340 unknown; nessuna closure o interruzione supportata con il provider attuale.

In forward tutti i nove gruppi eleggibili mantengono lo stesso ingresso, oppure differiscono meno di 8 secondi: nessun confronto informativo. In reverse cambiano due gruppi, di cui uno escluso per vicinanza fra prove. Rimane **un confronto**, senza allentare filtri né pesi per aumentarne il numero.

Nel confronto reverse l’uscita Too Many Nights è a 132,320 s. Il baseline sceglie Reminder a 150,170 s (score 0,90309, evidenza unknown); la policy sceglie 123,860 s (baseline 0,82469, aggiustato 0,93365, evidenza continuation). Questi numeri descrivono la decisione, non la qualità musicale. Le corrispondenze ai player A/B restano nel file privato `key.json`.

Ascolto: http://127.0.0.1:8768/test/1aad9cb0df4e346e/index.html. [Timestamp senza nomi dei sistemi](../runs/expectation_v7/TIMESTAMPS.md). Salvataggio previsto: `giudizi json/giudizi_Too_Many_Nights_verso_Reminder_V7_motion_1aad9cb0df4e346e.json`, con revisioni successive.

Verifiche: 129 test; due WAV da 20 s, stereo 48 kHz, zero clipping, identità PCM con le due sorgenti negli intervalli senza rampe, primi 10 s identici tra varianti, copie del listening pack coerenti. Sei risposte Range e due HEAD confrontate coi byte locali; controlli seek presenti. Report: `runs/expectation_v7/experiment.json`, `scores.json`, `audio_qa.json`, `transport_qa.json`.
