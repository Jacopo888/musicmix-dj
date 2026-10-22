# Stato del progetto — 19 ottobre 2026

## V10 — modelli preaddestrati eseguiti

CUE-DETR: 33 cue sui sei brani. LinkSeg: 86 sezioni, 80 confini interni, rete e decoder ufficiali. SongEval + MuQ: cinque inferenze diagnostiche, nessun filtro di accettazione validato. Pool6 pronto con Beat This!, Demucs e Whisper; 30 direzioni. Due selezioni CUE-DETR e due LinkSeg esportate, più quattro controlli condivisi. La galleria mostra otto audio distinti, seek ±5s e osservazioni tecniche: sui due LinkSeg salti di livello di circa12dB, su uno anche bordo quasi silenzioso. Lo storico V10 esclude 41 geometrie, senza nuovi voti umani. **219 test e 3 subtest passati**, otto HEAD/Range/416 verificati. All-In-One Linux/WSL: decoder corretto, forward CPU interrotto; nessuna inferenza strutturale completata. LinkSeg è il provider funzionale eseguito sul pool. [Ascolto](http://127.0.0.1:8771/test/review/index.html) · [Report completo](23_MODELLI_PREADDESTRATI_V10.md).

## Stato corrente V9: diagnosi e ricerca sul momento del cambio

Importati e verificati i quattro giudizi V8: zero vittorie nei due confronti expectation, quattro interruzioni del movimento, due parole tagliate. Lo storico ha 31 geometrie giudicate e 33 già presentate. Gli otto render superavano i controlli tecnici: questi non certificano la qualità musicale.

Otto filoni DeepSeek sul server condiviso: quattro completati, quattro fermati dalla quota e completati separatamente dall'orchestratore. Dossier integrati con fonti, lettura dichiarata, correzioni e provenienza. Revisione Luna 5.6 xhigh. Il piano aggiunge decisione stay/cambio, frase/arrivo, tempi condivisi e critica indipendente; non è ancora implementato né validato. Nessun nuovo modello/audio V9 eseguito.

Launcher corretto per lock temporaneo Windows e ripresa delle sessioni senza duplicazione; **198 test e 3 subtest passati**. Sessioni OpenCode archiviate e cancellate, server posseduto arrestato. [Sintesi e priorità](22_QUANDO_CAMBIARE_E_AUTOFEEDBACK_V9.md) · [Dossier](../ricerca/v9/INDEX.md). Le sezioni seguenti sono storiche; il precedente ascolto V8 non è un nuovo risultato.

## Stato corrente V8: pool e auto-feedback

Otto task DeepSeek V4.1 Flash Free, variant high, eseguiti tramite un unico server OpenCode; risposte archiviate e sessioni cancellate. Pool di quattro brani pronto, incluse 5% TINT e Creature con Beat This!, Demucs e ASR completati. Ricerca su 12 direzioni, griglia completa di 1.611.642 coppie; prefiltro vocale esatto e cache riducono il lavoro senza cambiare i risultati verificati.

Pronti quattro confronti A/B, otto WAV distinti da 20 s con contesto uscente identico per coppia. Due confronti di reranking e due di esplorazione di un altro brano. Ascolto: http://127.0.0.1:8769/test/3bdeea23988e22af/index.html. Navigazione Range/HEAD, pulsanti seek e didascalie sorgente verificati. Giudizi nella consueta cartella; registro di 33 geometrie per evitare di riproporre anche audio mostrati ma non ancora giudicati.

Auto-feedback tecnico sui render più Audiobox Aesthetics realmente eseguito in locale, con assi separati e un controllo saturato. Nessun premio neurale entra nella selezione. QA su otto WAV superata; 24 Range e otto HEAD superati. **196 test e 3 subtest passati**. La configurazione pytest limita la raccolta alla suite `tests/`, evitando le copie archiviate nelle cartelle degli agenti.

Restano da validare risoluzione musicale, precisione dei confini del cantato e preferenze V8: sei tagli su otto sono sensibili a un'espansione ipotetica ASR di 50 ms. Nessun risultato SOTA dichiarato. [Report completo](21_POOL_AUTO_FEEDBACK_V8.md) e [ricerca](../ricerca/v8/INDEX.md). Le sezioni seguenti conservano lo storico.

## Stato corrente V7: confronto controllato sul movimento

Agenti Luna 5.6 xhigh per feature e ranking, revisione di ricerca con effort max. Cache completa su tutti i beat: 4512 record / 2256 coppie / 162 profili; 39 profili riutilizzati, zero errori di validazione dei profili. Il criterio sperimentale osserva continuazione multifeature; senza provider di frase/struttura non dichiara una risoluzione. Soglie e peso 0,20 espliciti, modalità offline con riferimento direzionale al seguito naturale di A.

Registro di 23 geometrie dai giudizi confermati; il filtro di novità lascia 27 coppie forward e 47 reverse. Risultato: nessun cambio d’ingresso utile forward, due gruppi reverse con vincitore cambiato, un solo confronto dopo la distanza minima fra prove. Stessa uscita Too Many Nights a 132,320 s; ingressi Reminder a 123,860 e 150,170 s. Due WAV distinti di 20 s; contesto uscente identico e sorgenti PCM verificate. Link attivo: http://127.0.0.1:8768/test/1aad9cb0df4e346e/index.html. Giudizi salvati nella cartella abituale con direzione e V7 nel nome.

**129 test passati**, sei Range e due HEAD verificati sulle copie servite. La prima esportazione della cache era fallita per disco pieno: recuperata con compressione lossless dei JSON, gzip streaming e checkpoint. Dettagli: `20_ESPERIMENTO_MOTION_V7.md`. Restano aperti qualità percettiva, vere frasi/arrivi e precisione dei confini vocali; il pilot non prova un miglioramento musicale.

## Stato corrente V6: ricerca dedicata e traiettorie

Completata la ricerca di 15 riferimenti da un agente gpt-5.6-luna con effort **max**: `ricerca/MOTION_SUSPENSE_RICERCA.md`. Il problema è aspettativa/tensione non continuata né risolta, non una proprietà negativa del rap. Il solo calo di energia/proxy non viene più chiamato risoluzione nel prototipo.

Misurate traiettorie reali su 18 selezioni più un riferimento separato: A-prima, A-seguito-originale e B-dopo, con serie per beat, energia/timbro/attacchi/bassi/chroma e finestre 8/16/32. Dati in `runs/trajectory_audit_v6.json`; nessun modello semantico di suspense o nuovo reranking è dichiarato. Nuove opzioni della pipeline per annotare continua/risolve/interrompe e per evitare riusi audio nei futuri ascolti. **117 test passati**. Dettagli e decisione implementativa in `18_MOTION_IMPL_V6.md`.

## Stato corrente V5: player e diagnosi del motion

Corretto il server audio con Range/HEAD, verificato su 36 WAV e 108 richieste parziali; aggiunti −5/+5 s e salto al raccordo sugli stessi URL. Chiavi e audio delle prove giudicate sono invariati. Audit PCM su 60 render storici coerenti con le due sorgenti e i cue dichiarati.

Importati anche i 12 nuovi giudizi V4. Confermati riusi byte-identici nella confezione rank/random e due coppie già presenti nella V3. Il feedback segnala mismatch di mood/andamento e un word-cut anche dove il guard ASR non vedeva overlap. Questi casi restano evidenza per la validazione, senza ritoccare pesi sul singolo candidato.

Implementato un primo modulo osservativo motion e generati profili reali su 18 selezioni: finestre 8/16 beat disponibili in tutti i casi, 32 beat in 13/18. Nessun provider di tensione, nessun reranking o nuovo mix applicato: il riconoscimento di continuazione/risoluzione resta aperto. **104 test passati**. Vedi `17_PLAYER_E_FEEDBACK_V5.md`, `16_MOTION_E_PROSSIMI_PASSI.md` e `14_VALUTAZIONE_UMANA.md`.

## Stato corrente V4: feedback importati e diversità

I due JSON di ascolto V3 sono stati associati alle direzioni dopo conferma dell'utente: cinque preferenze per le proposte strutturate e un “nessuna convincente”, su sei confronti. Il caso sufficiente è conservato senza ritoccare pesi/soglie. Questo pilot non supera Gate A.

Corretti il ramo word-gap ASR e il riuso degli stessi cue nel top-K. Le run V4 restituiscono quattro proposte per direzione con quattro cue distinti su ciascuna timeline, inclusi punti interni. Tutte le coppie di downbeat sono state valutate. **94 test passati**, otto WAV da 20 s con controlli automatici superati. Precisione del cantato e qualità delle nuove proposte richiedono ascolto.

La nuova UI locale bilancia A/B, confronta V4 con V3 e con un controllo casuale valido, include identificativi run/test e salva direttamente nella cartella `giudizi json` accanto al progetto, con nomi comprensibili e revisioni. Vedi `15_DIVERSITA_E_ASCOLTI_V4.md` e `14_VALUTAZIONE_UMANA.md`. Le sezioni successive conservano lo storico.

## Stato corrente: sprint struttura e voce

Le sezioni successive conservano lo storico delle iterazioni. Stato aggiornato: Beat This! small0 eseguito su entrambi i brani (291/294 beat, 75/76 downbeat); separazione Demucs htdemucs e trascrizione faster-whisper small.en complete (487/397 parole). Evidenze ricalcolate dai soli stem e timestamp salvati con soglia acustica conservativa, senza ripetere l'inferenza.

Implementati ranking su contesti prima/dopo da quattro/otto beat, rifiuto delle coperture vocali sconosciute, limiti sugli spostamenti dai beat e confronto A/B a renderer identico. Il vecchio taglio della proposta 3 a 132,096 s risulta non sicuro anche nell'analisi vocale completa. **75 test superati**; non sono una misura di preferenza musicale. Dettagli e risultati delle run in `docs/history/13_STRUTTURA_E_FRASI.md`.

Gli agenti audio_features e candidate_search hanno completato i moduli assegnati; l'agente paper_audit ha raggiunto il limite d'uso durante la fase vocale. L'orchestratore ha ripreso integrazione e verifiche dai file già prodotti. La separazione in processi locali si è conclusa per entrambe le tracce.

Restano aperti: verifica annotata dei timestamp sul cantato, riconoscimento di vere frasi/sezioni musicali, baseline forte e controllo casuale con ascolti, correzione tempo/pitch e planner N-brani. Gate A e Gate B restano non verificati.

## Risultato dello sprint orchestrato

Pacchetto originale estratto in `music_mixing_progetto`. Tre agenti **gpt-5.6-luna**, ragionamento **xhigh**, hanno lavorato in parallelo; l'orchestratore ha definito i contratti, integrato CLI e renderer, revisionato i risultati ed eseguito le verifiche finali.

| Area | Responsabile | Risultato |
|---|---|---|
| Ingest e descrittori locali | Agente audio_features | Loader WAV, FFT, chroma, RMS, timbro, attacchi, stima tempo opzionale e diagnostica |
| Generazione e ranking | Agente candidate_search | Griglia limitata su cue interni, punteggi spiegabili, top-K diversificato e regressioni |
| Verifica fonti e task | Agente paper_audit | Lettura del testo completo dei paper accessibili, deviazioni dalla letteratura, task board |
| Integrazione e revisione | Orchestratore | FFmpeg 48 kHz stereo, CLI, preview neutre, manifest/hash/log, replay JSON e demo |

Non sono stati attivati job ricorrenti: gli agenti lavorano nell'esecuzione della chat. La task board conserva il lavoro successivo e le dipendenze, senza presupporre esecuzione in background dopo la conclusione del turno.

## Cosa funziona

`python -m musicmix search A B --output CARTELLA_NUOVA` accetta due file audio decodificabili da FFmpeg, produce cue interni, confronta le coppie candidate e salva anteprime stereo. I punti possono trovarsi in qualunque parte utile del brano. Il limite della griglia e i filtri sono registrati: non viene promessa una ricerca esaustiva di ogni campione.

Ogni run salva audio canonico, hash degli originali e del canonico, versioni Python/dipendenze/decoder, parametri, descrittori, candidati, punteggi, motivi di esclusione e SHA-256 dei render. I record dei candidati si possono ricaricare nella CLI storica. L'output esistente viene rifiutato per evitare sovrascritture.

## Verifica concreta

- **23 test automatici superati** su loader PCM, silenzio, attacchi, confini fra batch FFT, punteggi, direzione A→B, diversità, armonia senza trasposizione implicita, preview, ingest fallito e pipeline FFmpeg.
- Demo originale sintetica: `examples/demo_audio/a.wav` e `b.wav`.
- Run verificata: `runs/demo_verificata/REPORT.md`, **49 candidati**, **5 anteprime WAV**, tutti i 49 record conformi allo schema JSON originale.
- CLI storica su `examples/candidate_example.json` e controllo del pacchetto superati.

La demo verifica l'esecuzione, non dimostra capacità musicale su canzoni reali. I punteggi elevati su segnali sintetici non costituiscono una metrica di successo scientifico.

## Limiti attuali e prossime priorità

**M0 e M1 sono parziali; Gate A e Gate B sono aperti.** Nessun risultato di ascolto è stato raccolto. Nessuna separazione AI, trascrizione di parole, allineamento fonetico o ricerca di riff è stata implementata. Il programma completo per N canzoni è ancora da costruire.

La baseline usa attacchi euristici e finestre FFT brevi, non downbeat/sezioni riconosciuti né sequenze cromatiche sincronizzate ai beat come nei paper. `confidence` è una misura euristica del rilevatore, non una probabilità calibrata di musicalità. Un confronto sul mix originale non riconosce la presenza vocale; due voci potrebbero sovrapporsi o essere tagliate. I descrittori presso il cue sono simmetrici, mentre un modello direzionale dovrà confrontare esplicitamente contesto prima dell'uscita A e dopo l'entrata B.

La preview applica soltanto fade lineare fisso, guadagno fisso e raccordo degli estremi del clip. Non sincronizza beat, non cambia tonalità, non corregge loudness. L'audio canonico PCM16 è una scelta della baseline documentata nel manifest; il target PCM24 del design originario resta futuro. I file completi vengono caricati in RAM; non è un motore streaming. Il log iniziale contiene eventi di fase e durata cumulativa: la granularità e lineage per ogni modulo previsti da T002 restano parziali.

Ordine successivo:

1. Eseguire un pilot su due brani reali e annotare cue buoni/cattivi; mantenere casi di fallimento riproducibili.
2. Integrare un provider beat/downbeat/sezioni sostituibile e verificare i timestamp su gold annotati (T003); migliorare le finestre direzionali (T004/T006).
3. Aggiungere random su cue validi, una baseline forte riprodotta dal paper e uno studio cieco con identica render policy (T009).
4. Integrare separazione, parole allineate e pitch con confidenze verificate, poi DTW e hard negatives (T010–T014).
5. Superare Gate A prima di perfezionare renderer e planner N-brani, salvo esperimenti isolati.

## Aggiornamento: prima coppia reale richiesta dall'utente

Brani di prova dalla libreria locale dell'utente: **Reminder — The Weeknd**, versione Starboy (MP3), e **Too Many Nights — Metro Boomin / Future / Don Toliver**, versione HEROES & VILLAINS (FLAC). Titolo, artista e album verificati dai metadati; hash in `local_audio/DOWNLOAD_PROVENANCE.json`.

La ricerca iniziale Weeknd+Reminder era vuota; Reminder+Starboy ha trovato la versione corretta. Un peer FLAC è rimasto in coda fino al timeout, poi la copia MP3 è stata scaricata da un altro peer. Impostata codifica UTF-8 per evitare errori della console su nomi Unicode.

- `runs/reminder_to_too_many_nights`: 8.827 candidati, 10 preview brevi, 3 preview aggiuntive da 19,5 s in `listening/`.
- `runs/too_many_nights_to_reminder`: esperimento nella direzione inversa, 10 preview.
- Entrambi i render usano la policy neutra; nessun modello voce/lyrics o correzione tempo/pitch.
- Le periodicità stimate sono circa 40,18 e 43,95 BPM. La confidence numerica 1,0 non implica correttezza musicale: occorre misurare errori di ottava e fase rispetto a beat annotati.
- Nessuna valutazione umana raccolta; Gate A rimane aperto. Diagnostica e collegamenti audio nelle cartelle delle run.

Il confronto nelle due direzioni restituisce gli stessi primi abbinamenti invertiti: le distanze attuali sono simmetriche. Gli ID e il renderer distinguono A→B da B→A, ma il ranking non modella ancora la direzione del fraseggio. Questo limite è ora osservato sui brani reali e motiva le finestre prima/uscita e dopo/entrata previste da T006.

I due brani ora sono disponibili per migliorare gli adapter. Il prossimo esperimento deve confrontare la selezione con cue annotati e random validi usando identica render policy.


## Aggiornamento: feedback sulla parola tagliata nella proposta 3

L'utente giudica promettente il collegamento della proposta 3 in direzione Reminder → Too Many Nights, ma segnala una parola spezzata. Caso registrato in `examples/feedback/proposal3_word_cut.json`.

Eseguito un esperimento ASR locale opzionale sui due contesti con faster-whisper base.en e small.en (CPU int8; runtime e checkpoint isolati in `.cache/`). I due modelli collocano una parola nel cue originale a 132,096 s, con timestamp diversi. Nessuno dei due è un aligner del cantato validato; le probabilità di token non sono precisione calibrata dei confini.

Variante in `runs/proposal3_word_boundary_v1/proposal3_word_boundary.wav`: uscita A 133,350 s (+1,254), ingresso B invariato a 10,944 s, fade 80 ms. Il fade uscente è interamente in una pausa prevista da small.en con margine 50 ms. Peak 0,8988, zero campioni clippati. Il confine vocale entrante resta non verificato. La qualità va confermata dall'ascoltatore.

Aggiunto `src/musicmix/word_guard.py`: protegge l'intero fade dai word span, gestisce sovrapposizioni e confidenze mancanti, limita gli spostamenti, può operare solo sull'uscita e rifiuta i confini non supportati. Il punteggio originale è marcato stale dopo ogni spostamento. Il modulo è isolato, non ancora attivato automaticamente da `search`; il caso non rende il sistema universalmente capace di evitare tagli di parole.

Suite completa: **33 test superati**. L'esperimento ASR avanza T011/T014 come prototipo locale, ma separazione, forced alignment e verifica gold del cantato restano aperti. Gate A resta non verificato.
