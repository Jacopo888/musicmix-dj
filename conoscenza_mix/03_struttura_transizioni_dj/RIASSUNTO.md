# Struttura e transizioni DJ

## Audit integrale dei nove libri — 21 ottobre 2026

Brewster/Broughton capp11/13, Webber stamp225, Butler stamp232–247 e Katz stamp228/232 sostengono attenzione a arrangiamento, pickup, confini e ruoli. Sono pratica/analisi; gli allineamenti di Butler6.6–6.9 sono ipotetici. Held–Karp, Viterbi, profilo books, durata a sezioni e dj_form esistono: priorità a candidati per ruolo/tecnica e integrazione phrase/critic nel runner. Il caso senza core che diventa tutto outro è riprodotto e richiede semantica esplicita.

Dettagli, fonti/pagine e criteri di accettazione: [revisione completa](../libri/audit_2026-10-21/REVISIONE_PIPELINE.md).

## Evidenza misurata

**VandeVeire e De Bie (2018)** costruiscono un sistema automatico completo per Drum and Bass: beat/downbeat, struttura, cue, crossfade e selezione playlist. Nel corpus dichiarano 91% di canzoni con annotazioni MIR complete corrette (160 brani, più hold-out di 220) e un test soggettivo con 18 fan. Il risultato mostra che un sistema deve coordinare più componenti, ma resta DnB-specifico.

**Kim et al. (2020)** lavorano su mix pubblicati e usano 1.557 mix, 13.728 tracce e 20.765 transizioni. Le finestre di hit a 15, 30 e 60 secondi misurano l'allineamento con eventi ricavati dal mix; non sono rating di “risoluzione” né confini vocali. Sono tolleranze di retrieval da mantenere separate da quelle di parola o frase.

**Schwaab et al. (2019)** definiscono un reverse engineering in cinque passi (DTW, allineamento campione, rimozione traccia, curva volume tempo-frequenza, cue). UnmixDB consente ground truth costruito e controlli come errore di allineamento e suppression ratio; il paper riporta errore mediano circa 25 ms e casi degradati attorno a 100 ms. Questo misura clock/renderer, non gusto del DJ.

## Cosa non è dimostrato

La durata della transizione non è una costante universale. 8/16/32 battute, 15/30/60 secondi e cue di intro sono protocolli di lavori diversi. Un cutoff a 10 s è una finestra di ricerca o una scelta di prodotto, non una teoria della chiusura musicale. Neppure un boundary strutturale dice se A doveva restare o se B completa l'aspettativa.

## Ipotesi ingegneristica

Trattare il piano come superficie `cue_A × cue_B × durata × fade`, con candidati allineati al beat ma vincolati da voce/parola. Valutare prima clock, sample alignment, headroom e suppression ratio su controlli sintetici; solo dopo aggiungere indizi di struttura e ascolto. Se le assunzioni UnmixDB (tracce note, livelli separabili, tempo stabile) non valgono, emettere “non applicabile”.

## Trasferimento

I lavori DJ sono soprattutto EDM/DnB/4-4. Il principio multi-componente trasferisce; le durate e le priorità no. Rap/trap richiedono la voce e la sintassi prima della regola intro/drop.
