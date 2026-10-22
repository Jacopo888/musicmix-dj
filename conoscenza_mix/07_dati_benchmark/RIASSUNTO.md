# Dati e benchmark: conteggi, licenze e leakage

## Audit integrale dei nove libri — 21 ottobre 2026

Corpus dei libri:8PDF/2.527 pagine+1EPUB/75chunk su67spine; tutte le unità attestate, fonti SHA256 riconfermate. Test su6f03b33:242pass/8skip/6fail; tre per librerie mancanti, tre sullo stesso render. Il successivo bec3420 è stato ricontrollato staticamente, senza attribuirgli questi test. Nessun giudizio di qualità musicale da test sintetici, accordo tra rappresentazioni o conteggio QA. Conservare split e giudizi storici congelati.

Dettagli, fonti/pagine e criteri di accettazione: [revisione completa](../libri/audit_2026-10-21/REVISIONE_PIPELINE.md).

## M-DJCUE: controllo reale dell'archivio

È stato recuperato il tree ufficiale GitHub alla revisione `7fd692c793a4a4744ab61e409854302269b0974c` come `mdjcue_repo_complete.tar.gz` (966.957 byte, sha256 `b0581389d6b680f21443ae870152e3bfe7d53ad2ff43d7727c88baf2c21d8add`) e sono stati estratti e parsati tutti i record JAMS, senza scaricare audio. Il precedente `mdjcue_repo.zip` (634.121 byte) è conservato in `failed_fetches` come tentativo parziale: manca l'EOCD e `tar` recupera solo 133 file; non è usato nei conteggi accettati.

| Oggetto | Conteggio verificato | Nota |
|---|---:|---|
| file `.jams` | 134 | tutti i file nell'albero ufficiale sono presenti |
| schede annotatore | 402 | 3 per file; Marco/Paolo 134, Mickael 104, Jakob/Michael 15 |
| eventi `cue_point` grezzi | 1.849 | 1.037 `IN`, 812 `OUT` |
| eventi con durata positiva | 590 | duration usata per segmenti/fade OUT |
| cluster dopo merge ±0,5 s | 1.082 | 565 `IN`, 517 `OUT` |
| cluster per accordo annotatori | 547 size 1; 305 size 2; 230 size 3 | 535 cluster hanno almeno 2 annotatori |

I valori sono ricavati da ogni JSON JAMS completo, non dal numero riportato nel paper. Le linee guida dicono: cue sul principio di battuta, precisione 0,1 s; IN limitati all'intro fino al primo beat core; OUT non prima del core; duration opzionale per fade. Questo vincolo crea **leakage di protocollo**: un modello che entra a metà brano può sembrare “sbagliato” perché il dataset non annota quell'uso, non perché l'ingresso sia musicalmente cattivo. Non valutare fuori dalle regioni coperte senza dichiararlo.

Il metadata include titoli, anni, durate, flag vinyl e URL mirror YouTube. L'audio non è distribuito e resta coperto da diritti dei brani; la licenza GPL-3.0 del repository non concede i diritti sui file audio. Non copiare o scaricare audio per costruire un benchmark senza congelare licenza e split.

## Raveform e dataset reali di DJ

Raveform conserva struttura funzionale e metrica EDM dentro mix reali, ma solo 1.423 tracce sono annotate rispetto a decine di migliaia di tracce suonate. MixesDB e allineamenti possono introdurre duplicati, versioni diverse e punti derivati: raggruppare per traccia/artista/versione prima di dividere train/test. La card locale conferma CC BY 4.0 nel front matter, ma va verificata la licenza dei singoli file.

EDM-CUE/CUE-DETR contiene metadata/cue senza audio redistribuito; M-DJCUE ha audio assente; UnmixDB è utile per controlli sintetici con tracce CC, non per gusto professionale. JamendoLyrics/Signalign separano forced alignment, ASR e copertura parola: una MAE calcolata sulle parole recuperate nasconde i dropout.

## Regole di split e held-out

Congelare hash, revisioni, artisti e versioni prima di scegliere modello o soglie. Dividere per traccia e registrazione originale, non per singola transizione. Non trasformare ogni punto non scelto in negativo: annotatori e DJ possono avere alternative valide. I giudizi V8/V10 restano held-out e non entrano nel tuning. Un benchmark di closure A→B per rap non è presente in queste risorse e deve essere costruito con annotazioni nuove.
