# Huron 2006 — cap. 13 «Creating Predictability» (pp. 239–268)

Capitolo chiave per il mestiere DJ (fill, ipermetro, ripetizione).

## Tesi

- Quattro modi di rendere prevedibile la musica: **schematico, dinamico, veridico, cosciente** (p. 240) [T].
- **Veridico**: >99 % degli ascolti riguarda passaggi già sentiti; chi ascolta 3 h/giorno di radio sente >330 brani/settimana di cui 8–10 nuovi; 5 album ≈ 90 % dell'ascolto autoprogrammato; i DJ radiofonici sanno che i brani noti attirano più pubblico, «la novità è difficile da vendere» (pp. 241–242) [stime citate].
- **Schematico — tempo**: 544 canti tedeschi in 4/4 (Tab. 13.1, p. 243): gerarchia metrica + **prossimità metrica** (un attacco rende probabile quello vicino successivo); successione più frequente beat 3 → beat 4, poi **beat 4 → downbeat** (Fig. 13.1, p. 244); posizioni 2, 6, 8, 12, 16 sono sempre seguite dal sedicesimo successivo (**«tendency moments»**, Fig. 13.2 p. 245) [E].
- Il ritmo che definisce il metro: nota lunga seguita da 1–2 brevi (Longuet-Higgins & Lee) (p. 245) [E/algoritmo].
- **Anticipazione** (nota anticipata prima della risoluzione): riduce poco l'incertezza sul *cosa* e molto quella sul *quando* → più forte prediction effect (Fig. 13.3–13.4 pp. 245–247) [T].
- **Anticipazione ipermetrica = fill di batteria** (pp. 247–250) [A]: nelle canzoni pop un fill precede il **downbeat ipermetrico** (inizio di frase di 4/8 battute; strofa di 12 battute in «At the Hop», 1958, con un fill diverso a ogni strofa); un singolo colpo sul 4° beat basta («Pennsylvania 6-5000»); il fill «elimina ogni confusione sull'ipermetro» anche se il downbeat successivo non ha un colpo di rullante. Variante **«stop-and-go»**: una battuta intera di silenzio del rullante prima del downbeat ipermetrico (Janis Joplin, «Buried Alive in the Blues»), che funziona come una sospensione e aumenta il desiderio del ritorno (Fig. 13.8 p. 250). Backbeat e fill sono i cliché ritmici principali del pop; il fill è il punto dove si varia.
- Armonia: barocco V→I 79 %; il pop usa meno accordi per brano ma **cicli** di 4–8 accordi ripetuti → meno prevedibilità schematica compensata da prevedibilità dinamica (pp. 250–253) [E].
- **Dinamico**: il 94 % dei passaggi è ripetuto; la ripetizione è **raggruppata** (AAABBB o AABBBA più di ABABAB) — dopo un segmento nuovo i musicisti lo ripetono subito: «razionare» il materiale nuovo massimizza il prediction effect (pp. 254–255) [E, Huron & Ollen, 50 brani].
- Figure/ostinati brevi (<2 s), sequenze, temi e motivi: un **tratto distintivo** è frequente nel brano e raro altrove (alto rapporto entropia esterna/interna) (pp. 255–262) [T + analisi statistica di Brahms].
- Opera vs genere: distinzione legata al tipo di memoria (veridica vs schematica) (pp. 262–266) [T]. Per un'identità forte i «marcatori» devono essere **frequenti, distintivi e presentati presto** (p. 265).
- Partch: scala di 43 note e strumenti inventati restano piacevoli perché **ritmo e motivi sono convenzionali** (pp. 267–268) [A].

## Regole operative

1. **Un fill (anche un solo colpo sul 4° beat) o una battuta di stop prima del downbeat ipermetrico rende certo il «quando»** (pp. 247–250) [A su repertorio pop; coerente con Krims p. 98 sugli «upbeat»].
2. Dopo materiale nuovo, ripeterlo subito (clumping) (p. 255) [E].
3. Se un parametro è nuovo (timbro, scala), mantenere convenzionale il ritmo (pp. 267–268) [A].
4. L'elemento distintivo di un brano va mostrato presto e ripetuto (p. 265) [T].

## Limiti

- Nessun esperimento diretto sui fill o sui «tendency moments» (detto dall'autore, p. 245); repertori occidentali; dati sulle canzoni pop: 70 brani.

## Implicazioni per MusicMix

- **Raccordo (fill/stop)** [H fondata su A]: preparare l'ingresso di B con un'anticipazione ipermetrica nell'ultima battuta della frase di A: fill (anche una sola nota/colpo sul 4° beat), roll, oppure **una battuta di stop** (stop-and-go). È il modello psicologico del «drop» dopo un break. Proposta per `dj_creative`/`dj_policy` in `PROPOSTE_*`: rilevare se A ha già un fill naturale nell'ultima battuta della frase (aumento di attacchi percussivi nel 4° beat) e preferire quel punto di uscita.
- **Tipo di transizione tra generi** [H]: quando B cambia timbro/genere, mantenere continuo il beat (beatmatching e downbeat comune) — principio di Partch: un parametro nuovo alla volta.
- **Selezione** [E→H]: brani noti al pubblico hanno un vantaggio veridico (pp. 241–242); nel set bilanciare noti e nuovi; introdurre un brano nuovo con un ingresso che ne mostri subito il tratto distintivo (hook) e lo ripeta.
- **Loop/estensione** [E→H]: ripetere subito una frase di A (clumping) è coerente con la struttura della musica reale; i limiti vengono da Margulis.
- **Valutazione** [H]: il modello dei «tendency moments» (attacchi su posizioni deboli che implicano il sedicesimo successivo) suggerisce che **uscire su una posizione debole con attacco** (es. sillaba sul 16° sedicesimo) lascia un'implicazione pendente: l'uscita va fatta dopo la risoluzione sul downbeat successivo, oppure con B che la risolve (pickup di B coerente).
