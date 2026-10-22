# Fraseggio rap, voce, respiro e confini

## Audit integrale dei nove libri — 21 ottobre 2026

Ohriner: replica del port simbolico distinta da accuratezza degli onset audio;5ms a stamp187 misurano ripetibilità manuale di un annotatore su un brano. Il p-center è controverso. Il nuovo detector RMS da stem è un proxy da validare. Riprodotta assegnazione di onset1040ms a parola terminata1000ms: snapped_share100% non prova correttezza. Krims descrive sviluppo dentro lo stesso stile di flow; niente classe unica come detector di motion. Distinguere confini prosodici, grammaticali e respiro.

Dettagli, fonti/pagine e criteri di accettazione: [revisione completa](../libri/audit_2026-10-21/REVISIONE_PIPELINE.md).

## Evidenza misurata e analitica

**MCFlow (Condit-Schultz, 2016)** è il corpus più diretto per il flow: la dissertazione descrive 124 brani, 47 artisti, 6.107 misure e 54.248 parole nella versione locale; le analisi riportano frasi prosodiche guidate da contorno di altezza, declinazione, ritmo e pause/allungamenti. La frase prototipica è spesso vicina alla misura, ma il corpus include sottounità di due beat, pickup prima del beat 1 e phrasing incrociato in 3/4 e 5/4. Le annotazioni sono simboliche e non sono una trascrizione audio verificata percettivamente.

**Adams (2020)** formalizza tre parametri di frase hip-hop: armonia del beat, sintassi lirica e motivo ritmico/melodico. Il default di due battute è troppo regolare per comunicare da solo direzione o chiusura; un evento sintattico o motivico può prevalere sulla rima. È una teoria musicologica basata su esempi analizzati, non un classificatore audio.

**Duinker (2020)** analizza 472 versi da 160 brani e propone cinque regole di segmentazione (rima, sintassi/narrativa, respiro, somiglianza ritmica e relativi conflitti) e quattro relazioni flow/beat (allineata, annidata, raggruppata non allineata, dislocata). Una rima anticipata o una dissonanza metrica non implicano che il sistema debba “correggere” il beat: la non-risoluzione può essere intenzionale.

## Conseguenze per i confini

Un downbeat, un silenzio o un loudness drop non è una closure vocale. La rima finale è evidenza debole se la sintassi continua; il respiro da solo è insufficiente; un pickup prima del beat 1 è un inizio normale, non un errore. Un taglio sicuro richiede almeno l'assenza di parola attiva e un contesto di frase compatibile, con le altre cue registrate separatamente.

## Ipotesi ingegneristica

Il boundary detector dovrebbe produrre componenti indipendenti: `word_guard` (parola attiva e coda), contorno/energia vocale, respiro, rima o fine sintattica quando disponibile, beat/downbeat, struttura strumentale. La decisione A e la decisione B sono asimmetriche: A deve poter concludere o proiettare continuazione; B può iniziare con pickup e risolversi dopo. Un candidato può quindi essere ammesso come “metrico ma vocalmente aperto” senza chiamarlo closure.

## Trasferimento

MCFlow/Adams/Duinker sono la base più forte per rap/trap, ma non dicono dove un DJ debba fare switch né come interagiscano due voci. I conteggi di frase nei paper non sono prestazioni sul pool MusicMix. Conservare V8/V10 come held-out e non adattare soglie a un singolo clip.
