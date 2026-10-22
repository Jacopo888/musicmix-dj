# 07 — Registro decisioni architetturali (ADR)

Questo file conserva il **perché** delle scelte. Cambiare un'ADR richiede motivazione, alternativa ed esperimento: è essenziale per non confondere modifiche e risultati.

## ADR-001 — Python prima, C++ dopo

**Stato:** accettata. **Scelta:** orchestrazione e modelli ML in Python; C++ solo quando profiling mostra che il DSP è il collo di bottiglia. **Alternative:** C/C++ completo (più lavoro d'integrazione), Python-only sempre (meno controllo real-time). **Conseguenza:** protocolli backend sostituibili.  
**Riferimenti:** [AutoMashup 2025](https://arxiv.org/abs/2508.06516), [Hybrid Transformers 2022](https://arxiv.org/abs/2211.08553).

## ADR-002 — Ricerca delle transizioni disaccoppiata dal rendering

**Stato:** accettata. **Scelta:** `candidate_id` descrive solo brani, punti, modalità e proposta di allineamento; rendering è un passo separato. **Motivo:** consentire listening test a cue fissi e attribuzione della qualità.  
**Riferimenti:** [Spotify 2017](https://archives.ismir.net/ismir2017/paper/000086.pdf), [DJGAN 2022](https://arxiv.org/abs/2110.06525).

## ADR-003 — Punti arbitrari, non solo intro/outro

**Stato:** accettata. **Scelta:** retrieval su tutti i downbeat + ancore voce/riff; filtrare per contesto e qualità, mai con regola globale primo/ultimo x% del brano. **Motivo:** requisito centrale del prodotto. **Costo:** ricerca più ampia; usare index e top-K.  
**Riferimenti:** [Spotify 2017](https://archives.ismir.net/ismir2017/paper/000086.pdf), [CrossSong 2015](https://staff.aist.go.jp/m.goto/PAPER/SMC2015smith.pdf).

## ADR-004 — Componenti con confidence, non feature binarie

**Stato:** accettata. **Scelta:** lyrics, stems, F0 e struttura sono opzionali e versionati; output non disponibile ≠ score zero. **Motivo:** canto spesso invalida il riconoscimento parole. **Rischio:** calibrazione confidence fra modelli eterogenei.  
**Riferimenti:** [Phoneme Lyrics 2021](https://schufo.github.io/files/2021_TASLP3091817.pdf), [All-In-One 2023](https://arxiv.org/abs/2307.16425).

## ADR-005 — Ranker interpretabile prima di ML

**Stato:** accettata. **Scelta:** somma di contributi per horizontal, vertical, continuity, creative, transform, artifact; successivo reranker supervisionato solo dopo annotazioni. **Motivo:** ablation, causal debugging, possibilità di contestare score.  
**Riferimenti:** [Lee 2015](https://www.ismir2015.uma.es/articles/302_Paper.pdf), [COCOLA 2025](https://arxiv.org/abs/2404.16969).

## ADR-006 — Timeline canonica a 48 kHz

**Stato:** proposta da convalidare. **Scelta:** un PCM master per ogni traccia, adapter resample indipendenti, offsets espliciti. **Motivo:** evitare drift nella conversione MP3 e nei modelli multipli. **Costo:** spazio disco e I/O.  
**Riferimento:** [All-In-One software](https://github.com/mir-aidj/all-in-one).

## ADR-007 — Nessun training end-to-end all'inizio

**Stato:** accettata. **Scelta:** modelli pretrained come estrattori, baseline interpretabile e studio d'ascolto; apprendimento del ranker successivamente se giustificato. **Motivo:** dati di preferenza costosi e rischio di proxy sbagliate.  
**Riferimenti:** [Modeling Stem Compatibility 2021](https://arxiv.org/abs/2103.14208), [AutoMashup 2025](https://arxiv.org/abs/2508.06516).

## ADR-008 — Ricerca sul grafo dei segmenti per N brani

**Stato:** proposta da testare dopo Gate A. **Scelta:** stato con song e ingress/egress/durata e set brani usati, non un grafo completo di sola compatibilità brano→brano. **Conseguenza:** beam search e politiche di copertura necessarie.  
**Riferimenti:** [CrossSong 2015](https://staff.aist.go.jp/m.goto/PAPER/SMC2015smith.pdf), [Spotify 2017](https://archives.ismir.net/ismir2017/paper/000086.pdf).
