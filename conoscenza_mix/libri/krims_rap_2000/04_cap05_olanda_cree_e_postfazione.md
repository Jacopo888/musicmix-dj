# Krims 2000 — cap. 5 «Two cases of localized (and globalized) musical poetics» (pp. 152–197) e Postfazione (pp. 198–204)

Letto: testo completo (OCR; Fig. 5.2–5.3, griglie di attacchi, illeggibili nell'OCR: si riportano le conclusioni dell'autore). Fonti di campo: interviste dell'autore (S. Slegers/DJAX 1997, Spookrijders 1997, Bannock), Wermuth 1996.

## Tesi

- Il rap globale va pensato insieme al locale: la circolazione globale (in realtà americana) fornisce credibilità, il locale fornisce il «represent» (pp. 152–155, 171) [T].
- **Olanda**: dicotomia Nederhop (in olandese, bianco, sonorità hard-rock, flow sung) vs rap «internazionale»/americano (pp. 157–173) [osservazione etnografica]. Gli Spookrijders (*De Echte Shit*) si posizionano come «vero hip-hop olandese» con flow effusive e campionamento (pp. 165–172).
- **Edmonton/Hobbema (Cree)**: l'MC Bannock usa uno stile West Coast sung (Too $hort, 2Pac, Mack 10) per articolare un'identità Cree; l'arco ritmico del brano mette in rilievo le parole identitarie (pp. 180–197) [A].
- Postfazione: la poetica musicale è un livello di mediazione sociale; quanto e come analizzarla dipende dal contesto, senza «mistificare» né rifiutare l'analisi (pp. 198–204) [T].

## Dati ed esempi

- Il passaggio al rap «hardcore» (Public Enemy, gangsta) in Olanda allontana parte del pubblico anche perché **più difficile da ballare** (Wermuth 1996, p. 158; n. 6: i cambi di produzione introdotti da Public Enemy erano «inimical to much club use») [osservazione citata].
- **Osdorp Posse** (Nederhop): chitarre elettriche, power chord, scratch usati come percussione (upbeat veloci), campioni sparsi, **tempi più veloci del rap, vicini all'hardcore rock**; voce gridata in distici, rime finali, gruppi pari di versi (pp. 161–162) [A].
- **Flow Nederhop** (p. 170): rime finali costanti su **BC 2 o 3**, spesso in distici; ritmi regolari e ripetuti; altri membri del gruppo raddoppiano la voce a fine verso → «ultra-sung», percepito in Olanda come «bianco».
- Spookrijders «Pappa's Kleine Meid»: alternanza Am9–F7, poco layering (solo ispessimento nel ritornello) = convenzioni R&B/canzone d'amore (p. 168).
- Nessun suono specificamente «canadese»; predominano stili identificati con New York (speech-effusive, layering complesso) (p. 180).
- **Bannock «Step Up»** (live, World Music Concert, Edmonton, marzo 1998, con DJ Roach su due Technics 1200) — 40 versi, ogni verso = una battuta (pp. 191–192):
  - Versi 1–6 (Fig. 5.2): **cesura costante su BC 3** nei primi 4 versi («l'aspetto più tipico» dello stile sung) e cesura su BC 1 (o un sedicesimo prima) nei primi 5; le parole in rima coincidono con le cesure; anticipazioni del battere («pad» sostenuto oltre il downbeat = sincope); il verso 6 è un «**rhythmic crescendo**» (attacchi su ogni sedicesimo dei primi due beat) e chiude l'unità di 6 versi (pp. 192–193).
  - Il brano procede per unità regolari chiuse da eventi più fitti e irregolari che coincidono con fini di unità sintattiche/semantiche (p. 193).
  - Ultimi 8 versi (Fig. 5.3): la cesura passa da BC 3 a **BC 2**, quindi **BC 3 diventa disponibile come upbeat forte** verso il verso successivo; i versi confluiscono nei successivi con upbeat elaborati che accentuano i downbeat (p. 194). Due upbeat identici (metà di BC 2, BC 3, metà di BC 3 = l'ultimo beat e mezzo diviso in valori uguali) **rallentano l'attività ritmica e insieme aumentano lo slancio del downbeat seguente**; su quei downbeat cadono «Neechee» e «Chief» (pp. 194–195).
  - «But I'm a CHIEF» con crescendo improvviso e grido: il momento più ricordato dal pubblico (osservazione informale); in studio l'autore, come produttore, aveva **diradato il layering in quel punto per mettere in primo piano la voce** (p. 196).
- Generalizzazione: molte canzoni rap procedono da strutture regolari ricorrenti verso eventi più rapidamente mutevoli e localmente unici; ciò è possibile anche restando nello stile sung (pp. 196–197) [T/A].

## Regole operative

1. **Cesura su BC 3 separa i versi; cesura su BC 2 lascia BC 3 per un upbeat che spinge nel verso successivo** (p. 194) [A, un brano] → la posizione della pausa interna alla battuta distingue fine verso «chiusa» da fine verso con **pickup**. Implementabile: ultima sillaba prima di BC 3 + silenzio su BC 3 = chiusura; attacchi su BC 3 dopo una pausa su BC 2 = pickup verso il downbeat seguente.
2. **Upbeat a valori lunghi e uguali nell'ultimo beat e mezzo = downbeat enfatizzato** (pp. 194–195) [A] → un arrivo forte di B sul downbeat può essere preparato da un upbeat rallentato di A (o da un pickup di B).
3. **Diradare gli strati nel momento vocale culminante** (p. 196) [A, pratica di produzione dell'autore] → nella transizione, abbassare/togliere strati di A quando entra una frase vocale chiave di B.
4. **Rhythmic crescendo** (aumento di densità di attacchi) chiude un'unità di versi (p. 193) [A].
5. **Ballabilità e club**: produzione hardcore densa vs party più ballabile (p. 158) [osservazione citata]; Nederhop più veloce (p. 162).

## Limiti

- Interviste e due casi locali; analisi ritmiche dell'autore su un brano live; osservazioni del pubblico raccolte informalmente (nessun N). Le regole 1–4 sono descrizioni di singoli brani, non leggi.

## Implicazioni per MusicMix

- **Pickup e fine verso** [H, da validare]: in `dj_phrase` la feature `pickup_bc3` (attacchi nella finestra di BC 3 dopo una pausa su BC 2) può distinguere un'uscita «chiusa» da una che spinge verso il downbeat; su MCFlow si può misurare se una pausa su BC 3 precede più spesso una fine verso annotata (vedi `PROPOSTE_*`).
- **Raccordo** [H]: quando la frase d'ingresso di B è un «momento chiave» (parola tonica sul downbeat dopo un upbeat lungo), lasciare spazio togliendo strati di A, coerente con il bass swap e con il principio Krims p. 196.
- **Set** [H]: brani da club/party vs hardcore densi (p. 158) come asse di ballabilità; resta diagnostica.
- Postfazione: nessuna regola tecnica; legittima l'uso della poetica musicale come descrittore ma non come giudizio di valore (pp. 201–204).
