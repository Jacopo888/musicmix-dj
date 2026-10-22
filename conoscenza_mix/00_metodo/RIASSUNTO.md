# Metodo della libreria scientifica

## Audit integrale dei nove libri — 21 ottobre 2026

25 blocchi Luna5.6 xhigh coprono tutte le2.527 pagine PDF e75 unità EPUB. Ledger per pagina, hash verificati, rilettura del coordinatore delle selezioni e delle figure essenziali. La contabilità delle attestazioni non prova comprensione. Le16 letture precedenti sono recuperate con manifest di provenienza; il corpus integrale resta fuoriGit. Fonte, teoria, analisi, pratica e ipotesi di trasferimento rimangono distinte.

Dettagli, fonti/pagine e criteri di accettazione: [revisione completa](../libri/audit_2026-10-21/REVISIONE_PIPELINE.md).

Aggiornata al 2026-10-20 (Europe/Rome). Questa cartella è il registro mantenuto della conoscenza utile a MusicMix. Le copie binarie in `originals/`, gli estratti in `texts/` e i manifest JSON sono prodotti da [`tools/collect_science_library.py`](../../tools/collect_science_library.py). Ogni fonte ha DOI o URL, data di raccolta, hash SHA-256, licenza/accesso, profondità reale di lettura e ancore di pagina/sezione.

## Regola di evidenza

Una fonte primaria può sostenere una frase solo entro il compito misurato nel lavoro. I numeri citati qui appartengono ai dataset dei rispettivi lavori, non al pool MusicMix. Una fonte può essere:

- **misurata**: risultati di esperimento, corpus o benchmark con conteggi e metriche locali;
- **teorica**: modello o analisi musicologica che propone un meccanismo, senza dimostrare la prestazione audio del progetto;
- **ipotesi ingegneristica**: uso prudente nel codice, da validare con diagnostica separata e senza trasformare la fonte in ground truth.

La chiusura musicale non è sinonimo di cutoff temporale (incluso il cutoff a 10 s), calo RMS/loudness o confine di un segmentatore. Un confine strutturale può essere un candidato; una transizione ben livellata può essere tecnicamente pulita ma musicalmente aperta. L'aspettativa A→B va quindi registrata come vettore di verifiche: continuità metrica, frase/voce non tagliata, traiettoria di tensione, evento di ingresso in B e diagnostica del renderer. `UNKNOWN` resta un risultato valido e impedisce una confidenza inventata.

## Cosa è conservato

Le fonti scaricate dai tentativi precedenti sono state ricopiate senza alterare i byte. I PDF validi sono anche estratti in UTF-8 tramite `pypdf`; XML JATS è trasformato in testo lineare. Il testo di Adams 2020 conserva artefatti di font, perciò il PDF originale e le ancore HTML restano la lettura di riferimento. I file Lehne da 519 byte e Grosz da 919 byte sono risposte HTML fallite, conservate in `01_psicologia_aspettativa/failed_fetches/` e marcate come non leggibili. Non vengono conteggiati come paper.

Il parser M-DJCUE ha letto tutti i 134 `.jams` dal tree ufficiale completo e ne ha riportato i conteggi in `07_dati_benchmark/RIASSUNTO.md`. Il vecchio ZIP troncato (assenza di EOCD; `tar` segnala input troncato) è conservato separatamente per audit e non viene usato come benchmark accettato.

## Protocollo per i turni futuri

1. Aggiungere una fonte solo da URL/DOI ufficiale o repository autore; conservare il file originale e il termine di licenza.
2. Validare header e parser prima di riassumere. Un HTML salvato con estensione `.pdf` diventa una voce `failed_fetch`, mai una prova.
3. Riportare le pagine realmente lette e distinguere abstract, testo completo e metadati.
4. Aggiornare un `RIASSUNTO.md`, il manifest di sezione e `CHANGELOG.md` nello stesso turno.
5. Lasciare i giudizi V8/V10 come diagnostica storica/held-out: non usarli per scegliere soglie che facciano apparire buono un solo clip.
