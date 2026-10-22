# Diversità dei tagli e ascolti V4

## Problema e cause osservate

L'utente ha segnalato la ripetizione di fine/principio dei brani e ha chiesto di non ottimizzare tutto sulla prima proposta sufficiente. V3 sceglieva cinque volte la stessa uscita di Reminder. Questo dipendeva da due fattori correggibili:

1. Il ramo word-gap cercava una singola riga di copertura ASR; il documento finale separava copertura ASR e presenza energetica dello stem. Di conseguenza quasi tutti i piccoli intervalli interni tra parole venivano esclusi anche con timestamp disponibili. Ora si richiede che gli estremi confidenti siano osservati nello stesso segmento ASR, con margine e senza sovrapposizioni con intervalli incerti. Questa è evidenza ASR temporale: non viene dichiarata silenzio acustico o forced alignment.
2. La selezione eliminava soltanto coppie identiche. Ora ogni ancora può comparire una sola volta e i cue selezionati devono distare almeno 8 s su entrambe le timeline. Se i vincoli consentono meno di K risultati, il sistema restituisce meno risultati e registra il motivo. Gli 8 s sono una scelta euristica esplicita, non una frase musicale riconosciuta.

I giudizi importati sono conservati separatamente e non cambiano pesi o soglie. Il caso sufficiente forward V3/rank 3 resta una singola osservazione utile: Reminder 192,100 s → Too Many Nights 16,420 s. Il ranking non contiene una regola dedicata a quella coppia.

## Risultati automatici

| Direzione | Griglia valutata | Proposte ammesse | Coppie cue uniche ammesse | Selezionate |
|---|---:|---:|---:|---:|
| Reminder → Too Many Nights | 9.660 / 9.660 | 110 | 55 | 4 |
| Too Many Nights → Reminder | 10.224 / 10.224 | 174 | 87 | 4 |

Le due lunghezze di contesto possono produrre record distinti per la stessa coppia. Il controllo casuale deduplica le coppie e campiona uniformemente dal pool ammesso prima della diversificazione, con seed 314159. Un'eventuale estrazione identica alla proposta selezionata è registrata e omessa dall'ascolto, senza sostituire il sorteggio.

Quattro uscite e quattro ingressi distinti in ogni direzione. Nessuna griglia troncata. Otto anteprime stereo 48 kHz da 20 s, zero clipping e nessuna sovrapposizione dei fade con parole note o intervalli incerti. Verifiche salvate nei rispettivi `QA.json`. Suite completa: **94 test passati**.

La presenza di energia nello stem resta alta (A 89,94%, B 85,20% del tempo). Non distingue backing vocal da bleed del separatore. Le soglie acustiche non sono state allentate per aumentare il numero di proposte. L'allineamento specifico del cantato resta da implementare/validare; i nuovi tagli ASR possono ancora interrompere fonemi o code non rappresentati correttamente nei timestamp.

## Timestamp V4

I timestamp sono nei brani completi; il raccordo nei file da 20 s è al secondo 10.

| Direzione | # | Uscita | Entrata |
|---|---:|---:|---:|
| Reminder → Too Many Nights | 1 | 03:33.090 | 03:02.200 |
| Reminder → Too Many Nights | 2 | 03:12.100 | 00:02.740 |
| Reminder → Too Many Nights | 3 | 01:18.100 | 00:19.140 |
| Reminder → Too Many Nights | 4 | 02:18.100 | 02:00.100 |
| Too Many Nights → Reminder | 1 | 02:57.440 | 03:12.100 |
| Too Many Nights → Reminder | 2 | 00:13.680 | 00:00.040 |
| Too Many Nights → Reminder | 3 | 00:21.860 | 00:09.080 |
| Too Many Nights → Reminder | 4 | 01:57.320 | 02:18.100 |

## Confronti e salvataggio dei giudizi

Ogni direzione contiene tre confronti V4/V3 e tre confronti V4/casuale valido, con renderer, gain e durata identici. Le posizioni A/B sono bilanciate entro il test; nella V3 erano casualmente tutte dalla stessa parte. I test V3 restano immutati.

- [Avanti, interfaccia locale](http://127.0.0.1:8766/test/0eff5afcebe403d2/index.html)
- [Inversa, interfaccia locale](http://127.0.0.1:8766/test/30391fd8f08c3189/index.html)
- [Report avanti](../runs/structured_forward_v4/REPORT.md)
- [Report inverso](../runs/structured_reverse_v4/REPORT.md)

Il pulsante **Salva giudizi JSON** nell'interfaccia HTTP scrive direttamente in `<workspace>/../giudizi json`. Il nome include direzione, V4 e identificativo test; ogni nuova versione viene salvata con suffisso di revisione e non sovrascrive la precedente. Il JSON include `test_id` e `run_id`, così la prossima associazione alla direzione non richiede una domanda. La chiave dei sistemi non viene servita dal browser locale. Aprendo direttamente l'HTML senza server si usa invece il download del browser, che non può garantire la cartella di destinazione.

Per riavviare l'interfaccia locale quando necessario:

```powershell
$env:PYTHONPATH = 'src'
python -m musicmix.listening_server --test runs/structured_forward_v4/listening_ab --test runs/structured_reverse_v4/listening_ab --port 8766
```

Il default del server è la cartella `giudizi json` accanto al progetto. Le run salvano snapshot, hash codice/parametri e motivi delle esclusioni. Il miglioramento di diversità è verificato automaticamente; la qualità musicale della V4 resta da valutare con i nuovi ascolti. Le sei osservazioni V3 documentano un pilot, non il superamento di Gate A.

## Esperimento separato: tutti i beat

Il nuovo parametro `--cue-policy beats` esplora tutti i beat rilevati, mantenendo la loro provenienza e gli stessi vincoli vocali/di diversità. Il default resta `downbeats`. L'esperimento non presume che interrompere una battuta sia musicalmente preferibile.

| Direzione | Coppie valutate | Proposte ammesse | Selezionate |
|---|---:|---:|---:|
| Reminder → Too Many Nights | 154.008 / 154.008 | 1.884 | 5 |
| Too Many Nights → Reminder | 156.240 / 156.240 | 2.628 | 5 |

Dieci ulteriori WAV da 20 s, con hash, integrità PCM, fade senza overlap con parole note/intervalli incerti e diversità verificati (`QA.json`). Le due run confrontano le prime tre scelte beat con le prime tre scelte downbeat usando lo stesso renderer. Non è un confronto a cue fissi né una misura isolata della precisione dei beat: valuta due politiche di selezione. Due proposte su cinque per direzione includono almeno un lato `word_gap_only`; l'evidenza ASR resta meno forte di un confine annotato e non sostituisce l'ascolto.

- [Beat contro downbeat, avanti](http://127.0.0.1:8767/test/79ef88775655c46c/index.html)
- [Beat contro downbeat, inversa](http://127.0.0.1:8767/test/52f9a7dc791b229b/index.html)
- [Tutti i timestamp e WAV, avanti](../runs/structured_forward_beats_v4/REPORT.md)
- [Tutti i timestamp e WAV, inversa](../runs/structured_reverse_beats_v4/REPORT.md)

Anche questi test salvano i giudizi direttamente in `giudizi json`, con `beat_contro_downbeat` nel nome. Per riavviarli: `python -m musicmix.listening_server --test runs/structured_forward_beats_v4/listening_ab --test runs/structured_reverse_beats_v4/listening_ab --port 8767` (con `PYTHONPATH=src`).

Riferimenti: la griglia beat-sincrona e i contesti prima/dopo sono motivati dalla lettura di Spotify 2017, §4.1, e AutoMashUpper, §II.B (vedi `ricerca/VERIFICA_PAPER_IMPLEMENTAZIONE.md`). Il bisogno di un allineamento specifico del cantato è documentato da [Schulze-Forster et al. 2021](https://perso.telecom-paristech.fr/grichard/Publications/2021_05_TASLP_preprint.pdf); [Park et al. 2024](https://arxiv.org/abs/2401.09200) studia un contesto classico diverso dai due brani locali. Questi lavori orientano i moduli e le verifiche, senza costituire una validazione dei risultati locali.
