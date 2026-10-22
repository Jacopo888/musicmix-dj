# Ohriner 2019 — cap. 6 «Flow, Metric Complexity, and Text in Eminem» (pp. 135–159)

Letto: testo completo.

## Tesi

- Eminem usa la **complessità metrica** dei groove come narrazione ritmica: la complessità cresce quando cresce la competenza/il potere del protagonista e cala quando si arrende (pp. 141–156) [A].
- La complessità vale solo se percepita come intenzionale: lo stesso spostamento dal beat può suonare come incompetenza (cap. 6.3.5, «8 Mile») [T/A].
- Cautela sul valore della complessità (Kautny 2015; discorsi razziali sul ritmo, Agawu 1995, Perchard 2015): la complessità non è un valore universale nella cultura hip-hop; dipende dal contesto d'ascolto (introspettivo vs sociale) (pp. 156–159) [T].

## Dati e numeri

- Eminem: il duplo copre 1/5 delle battute (M = 0,20, DS = 0,22) contro quasi metà nel genere (0,45, DS 0,34), t(81) = −4,4 (p. 137 n. 1) [E].
- 2/5 delle strofe di Eminem sono interamente non-duple, contro l'11 % del genere (n. 2) [E].
- <323_2222> 3 volte più frequente in Eminem; ruotato su 2 in > 1/4 dei casi contro il 5 % nel genere (pp. 138–139) [E].
- Tabelle di complessità: «Go to Sleep» 0 → 0,94 → 2,41 → 1,66 → 2,6 → 1,66 (Tab. 6.1 p. 145); «The Re-Up» 2,41 → 1,66 → 0,94 (IAI medio 2,34, più alto di tutte tranne 8 strofe su 75, p. 145); «8 Mile» v.1 1,66 → 3,66 → 4,00 (Tab. 6.2 p. 154).
- «Soldier»: accenti 1mod2 nelle prime 4 battute 2 → 5 → 9 nelle tre strofe, con lo stesso numero di accenti (28); grooviness 60 % → 75 % (pp. 147–148).
- Nessuna differenza di complessità metrica o IAI medio tra brani in classifica Billboard e no nel corpus (p. 157 n. 16).

## Esempi (minutaggi)

«Go to Sleep» (2003; pitch mediano per verso in salita, Es. 6.6; sparo sul battere di m. 17 con ingresso di rullante e hi-hat = nuovo groove), «The Re-Up» (2006, mm. 19–24, 1:11–1:28), «Soldier» (2002), «Till I Collapse» (1:24–1:42; <323_2222>@2 che suggerisce un contro-metro di 5+5+6), Phonte «The Good Fight» (2011, 0:43–0:52; @4), «8 Mile» (strofe di 16, 28, 45 battute; m. 9 «b'nuh nuh nuh»: la parola cede alla sola altezza del piano).

## Regole operative

1. Complessità metrica di un groove = Σ complessità dei tipi d'accento nei 4 beat (0–4) (p. 137) [A]; implementata in `dj_phrase.groove_complexity`.
2. Una **traiettoria** di complessità crescente + altezza vocale crescente costruisce energia verso il confronto; il ritorno a groove semplici segna risoluzione o resa (pp. 142–147) [A, analisi di un artista].
3. Cambi di groove coincidono con cambi di argomento e con eventi dello strumentale (sparo, ingresso di batteria) (pp. 143–144) [A].

## Limiti

- Analisi di un solo artista; legame testo–ritmo interpretativo («some readers will find this analysis to be a stretch», p. 151).

## Implicazioni per MusicMix

- **Arco del set** [H]: la complessità metrica media e la saturazione delle ultime battute di A/prime di B sono un indice di «energia ritmica vocale» indipendente dal volume (AGENTS.md: tensione ≠ volume). Proposta: in `dj_set` usarla come diagnostica dell'arco (salita graduale, picco, discesa), mai come unico criterio.
- Un ingresso di B con groove molto più complesso di A è un «flip» marcato; può funzionare come climax se preparato (Huron, delay) — da ascoltare.
