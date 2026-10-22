# 10 — Registro dei modelli/estrattori: template

Compilare una riga per ogni provider effettivamente installato e per ogni modifica rilevante dei suoi pesi. NON assumere che il nome di un algoritmo corrisponda a un checkpoint ufficiale.

| Modulo | Provider candidato | Output | Versione / checkpoint hash | Licenza codice | Licenza pesi | Precisione su dataset interno | Costo | Stato |
|---|---|---|---|---|---|---|---|---|
| M01 | [Demucs / Hybrid Transformers](https://github.com/facebookresearch/demucs) | stem | Da verificare | Da verificare | Da verificare | Non misurata | Da misurare | Candidato |
| M01 | [BS-RoFormer](https://arxiv.org/abs/2309.02612) | stem | Da verificare | Da verificare | Da verificare | Non misurata | Da misurare | Candidato |
| M02 | [All-In-One](https://github.com/mir-aidj/all-in-one) | beat/downbeat/section | Da verificare | Da verificare | Da verificare | Non misurata | Da misurare | Candidato |
| M03 | [Lyrics aligner](https://github.com/schufo/lyrics-aligner) | parola/fonema | Da verificare | Da verificare | Da verificare | Non misurata | Da misurare | Candidato |
| M03 | [CREPE](https://github.com/marl/crepe) | F0 voiced | Da verificare | Da verificare | Da verificare | Non misurata | Da misurare | Candidato |
| M04 | [Basic Pitch](https://github.com/spotify/basic-pitch) | note/onsets | Da verificare | Da verificare | Da verificare | Non misurata | Da misurare | Candidato |
| M08 | [COCOLA](https://github.com/gladia-research-group/cocola) | stem coherence | Da verificare | Da verificare | Da verificare | Non misurata | Da misurare | Esperimento futuro |

Per ogni modello aggiungere in un file `models/<id>.json`: `name`, `version`, `model_sha256`, `dependency_lock_hash`, `license_text_url`, `weights_license_url`, `test_dataset_id`, `metric` e `failure_examples`. Separare esplicitamente *funziona nel suo benchmark* da *migliora il nostro ascolto di transizioni*.
