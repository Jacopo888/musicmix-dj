# 09 — Risk register: licenze, dataset, modelli e validità

| ID | Rischio | Impatto | Mitigazione | Modulo |
|---|---|---|---|---|
| R01 | Audio/testi non autorizzati a essere redistribuiti | Legale/etico | Dataset lecito, autorizzazioni, manifest licenze, niente embedding audio nel repository | M00/M10 |
| R02 | Confondere licenza del codice con licenza dei pesi | Blocco distribuzione | Model registry con licenza esatta dei checkpoint e limiti di uso | M01–M04 |
| R03 | MP3/decoder con offset variabile | Cue sbagliati | WAV canonico con test offset su sample, conversioni annotate | M00 |
| R04 | ASR su canto/lingue miste inaffidabile | Match parola falso | Gold lyrics per subset, confidence, fallback e word alignment specialistico | M03 |
| R05 | Separazione stems non trasparente | Artefatti/cambiamenti timbrici | Quality report, confronto con original mix, fallback mode | M01/M12 |
| R06 | Scoring favorisce matching testuale ma stona | Misure fuorvianti | Hard gates armonici/ritmici, hard negatives, valutazione umana | M07/M08 |
| R07 | Modello COCOLA come «verità» | Overfitting al proxy | Usare score come feature e confrontarlo con preferenze reali | M08/M10 |
| R08 | Leakage canzoni in train/test | Risultati ottimisti | Split per song e preferibilmente artista, versioni duplicate unite | M10 |
| R09 | Differenze audio tra sistemi nei listening test | Confondimento | Stessa politica di preview nella fase I, stessi cue in fase II | M09/M10/M12 |
| R10 | Costi GPU su N canzoni | Scalabilità insufficiente | Preprocessing offline, cache, ANN, ranking a due stadi | M01–M08 |
| R11 | Ambiguità di «musicalità» umana | Etichette rumorose | Scomposizione criteri e studio cieco con intervalli di confidenza | M10 |
| R12 | Pianificatore crea sequenze incoerenti globalmente | Mix che non ha senso | Stati con tempo ingresso/uscita, contesto e vincoli di copertura | M11 |

## Dataset / strumenti candidati (verificare le condizioni prima dell'uso)

- [MUSDB18](https://sigsep.github.io/datasets/musdb.html): sorgenti per misurare separazione, ma verificare licenze e idoneità alla condivisione di audio.
- [MedleyDB](https://medleydb.weebly.com/): annotazioni e multitraccia; controllare licenza e accessibilità.
- [AutoMashup dataset Zenodo](https://zenodo.org/records/16921593): esempi e materiale associato al paper, verificare metadata e diritti.
- [M-DJCUE](https://github.com/MZehren/M-DJCUE): cue annotati, potenzialmente non rappresentativi di tutti i generi.
- [Demucs](https://github.com/facebookresearch/demucs), [All-In-One](https://github.com/mir-aidj/all-in-one), [COCOLA](https://github.com/gladia-research-group/cocola), [Basic Pitch](https://github.com/spotify/basic-pitch), [Lyrics aligner](https://github.com/schufo/lyrics-aligner): per ciascuno salvare versione, licenze e dipendenze.

**Nota:** Le licenze non sono state sottoposte qui ad audit legale. Le regole possono variare per dataset, modello, checkpoint e output trasformati.
