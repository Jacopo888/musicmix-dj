# Sprint: struttura, contesto direzionale e confini vocali

La baseline iniziale confrontava brevi descrittori presso due attacchi. Il feedback sulla proposta 3 ha mostrato che un abbinamento promettente può interrompere una parola. Questa iterazione separa tre evidenze: posizione ritmica, contesto musicale direzionale e presenza vocale presso il taglio.

## Componenti

- `structure.py` definisce eventi beat/downbeat e sezioni versionate. `tools/analyze_structure.py` esegue il modello opzionale **Beat This! small0**. I valori sigmoid conservati non sono probabilità calibrate di correttezza dei timestamp. Le sezioni aggiuntive da cambiamento spettrale restano euristiche: non sono strofe, ritornelli o frasi riconosciute.
- `directional.py` confronta quattro/otto beat prima dell'uscita A e dopo l'ingresso B. Aggrega più finestre spettrali per beat, conserva sequenze cromatiche, timbro, livello e andamento dell'energia, stima il tempo su più beat e registra l'ambiguità metà/doppio. Non applica trasposizioni implicite.
- La sicurezza vocale riguarda l'intero intervallo del fade. Le coperture mancanti e i confini poco affidabili sono sconosciuti, non silenzi. La modalità ordinaria rifiuta i casi non supportati; la modalità esplorativa è esplicita e non viene usata per le preview presentate come protette.
- `musical_preview.py` produce un handoff con due brevi rampe sequenziali: il cue B comincia quando si raggiunge il cue A, senza l'anticipo introdotto dal vecchio overlap. Il renderer non corregge differenze di BPM, tonalità o loudness. Gli eventuali spostamenti dai downbeat sono dichiarati.
- `structured_pipeline.py` collega le evidenze, verifica hash/durata/sample rate e salva una copia delle analisi usate. `listening.py` prepara un confronto A/B a renderer fisso, con chiave separata e modulo locale per salvare i giudizi.

## Esecuzione del ranking

I due WAV devono essere esattamente quelli ai quali appartengono gli hash delle analisi. L'inferenza dei provider resta separata, con dipendenze e checkpoint opzionali. Il comando non scarica modelli e non ripiega silenziosamente sulla griglia onset.

```powershell
python -m pip install -e ".[audio,test]"
python -m musicmix search-structured `
  runs/reminder_to_too_many_nights/audio/a.wav `
  runs/reminder_to_too_many_nights/audio/b.wav `
  --structure-a runs/structure_v2/a.json --structure-b runs/structure_v2/b.json `
  --vocals-a runs/vocals_v2/a.json --vocals-b runs/vocals_v2/b.json `
  --output runs/nuovo_esperimento `
  --baseline-run runs/reminder_to_too_many_nights --top-k 5 --max-candidates 20000
python -m pytest tests -q
```

La cartella di output deve essere nuova. `REPORT.md` collega le proposte; `listening_ab/index.html` confronta le prime tre con la vecchia selezione usando la stessa modalità di rendering. La chiave dei sistemi è salvata in un file separato. Preferenza e parole interrotte vanno valutate dall'ascoltatore: la creazione del test non costituisce un risultato sperimentale.

## Cosa resta da dimostrare

Il modello ritmico è preaddestrato, ma i due brani non sono un gold set annotato. Anche parole e pause possono essere imprecise sul cantato e sugli stem separati. Le regole di ranking e le soglie sono ipotesi dichiarate. La fase musicale, l'evoluzione delle sezioni e la preferenza umana richiedono ancora verifiche; questa versione non equivale a un DJ capace di riconoscere autonomamente ogni frase musicale.

Riferimenti implementativi: [Beat This! — repository ufficiale](https://github.com/CPJKU/beat_this), [Spotify 2017 — §4](https://archives.ismir.net/ismir2017/paper/000086.pdf), [Lee 2015](https://www.ismir2015.uma.es/articles/302_Paper.pdf), e la lettura del paper Lyrics 2021 in `ricerca/VERIFICA_PAPER_IMPLEMENTAZIONE.md`.

## Correzioni emerse nella verifica

La prima run strutturata (`structured_*_v2`) selezionava quasi soltanto intro/outro. L'audit ha individuato una confusione: un segmento ASR contiene anche pause fra parole, quindi non può contraddire da solo il basso livello dello stem vocale. Il conflitto viene ora attribuito soltanto alle parole effettivamente sovrapposte. Due regressioni verificano sia l'accettazione della pausa misurata interna al segmento, sia il rifiuto del taglio dentro una parola. I casi acustici sono etichettati `acoustic_boundary_supported`, senza attribuire loro un riconoscimento di frase musicale.

La cache vocale per singola ricerca evita di ricalcolare la stessa evidenza per ogni coppia. Le run successive usano un limite di 20.000 coppie per coprire interamente la griglia di downbeat dei due brani. Questo non rende esaustiva la ricerca di tutti i possibili cue audio.

Suite completa: **75 test superati**. Inferenza effettiva: Beat This! small0, Demucs htdemucs, faster-whisper small.en su CPU. Gli inventari opzionali e gli hash dei checkpoint/evidenze sono conservati localmente. Le probabilità ASR non sono accuratezza calibrata dei confini sul cantato.

## Risultati verificati della run V3

| Direzione | Coppie valutate | Proposte ammesse | Estratti | Durata ricerca/render |
|---|---:|---:|---:|---:|
| Reminder → Too Many Nights | 9.660 / 9.660 | 14 | 5 | 16,287 s |
| Too Many Nights → Reminder | 10.224 / 10.224 | 56 | 5 | 16,784 s |

Le proposte ammesse includono due lunghezze di contesto per alcune coppie; non sono tutte coppie di cue distinte. Nessuna griglia è stata troncata. I tempi escludono l'inferenza dei provider, eseguita e salvata prima.

- [Report avanti](../runs/structured_forward_v3/REPORT.md) e [confronto A/B avanti](../runs/structured_forward_v3/listening_ab/index.html).
- [Report inverso](../runs/structured_reverse_v3/REPORT.md) e [confronto A/B inverso](../runs/structured_reverse_v3/listening_ab/index.html).
- `tools/verify_structured_run.py` verifica hash delle sorgenti/evidenze/audio, limiti sugli spostamenti, assenza di parole note/incerte nei fade e PCM stereo 48 kHz. Dieci estratti da 20 s, nessun campione clippato; dettagli nei rispettivi `QA.json`.

**Limite osservato, ancora aperto:** tutte le cinque uscite in avanti sono a 192,1 s di Reminder. Nell'inversa quattro ingressi sono nello stesso punto. Tutti i selezionati hanno evidenza di bassa energia vocale, non di frase musicale riconosciuta. La correzione della copertura è valida ma da sola non risolve la scarsità di alternative interne. Un punteggio diverso nelle due direzioni e l'assenza di overlap con parole ASR non dimostrano continuità musicale. I confronti A/B contengono tre coppie per direzione; i giudizi restano da raccogliere.

Verifica aggiuntiva su tutti i beat, mantenendo gli stessi vincoli vocali e almeno 10 s da entrambi gli estremi: Reminder offre punti supportati soltanto intorno a 10,6–11,3 s e 191,3–192,9 s; Too Many Nights anche intorno a 131,6–132,3 s e 173,9 s. Dati in `runs/structure_v2/BEAT_CUE_AUDIT.json`. Ampliare soltanto la griglia ritmica non risolve dunque il collo di bottiglia di Reminder.

Prossima priorità motivata dai paper: allineamento fonema/parola sul cantato con verifica annotata, poi provider di confini di frase musicale. Conservare downbeat come ancore primarie e valutare beat secondari separatamente. L'ASR attuale e il solo livello RMS non danno evidenza sufficiente per promuovere automaticamente i tagli interni mancanti. Riferimenti di audit: Schulze-Forster et al., sezioni III–VII; Spotify 2017, §4.1; Lee 2015, equazioni 1–3 e §7. Non viene attribuito a questi lavori un risultato di validazione sui due brani locali.
