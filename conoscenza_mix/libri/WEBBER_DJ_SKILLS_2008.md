# Nota di sintesi — Webber, *DJ Skills: The Essential Guide to Mixing and Scratching* (Focal Press 2008)

Schede per capitolo: [`WEBBER_DJ_SKILLS_2008/`](WEBBER_DJ_SKILLS_2008/README.md). Natura: manuale didattico [P] (Berklee) con storia e interviste (BT, Oakenfold, QBert, Kid Koala, Logic, Radar, Shadow…).

## Tesi
Il giradischi (e i suoi equivalenti digitali) è uno strumento; mixare bene richiede fraseggio, conoscenza dei brani e misura negli effetti.

## Le regole più utili (con pagina)
1. **Fraseggio**: frasi di 2/4/8/16 battute; "Syncing the beats of the tracks without syncing their phrases will result in musically unstable mixes"; non interrompere una frase (p. 225) [P].
2. Prima del beatmatch: BPM, quale accelerare, come suona accelerato/rallentato, **incontrarsi a metà?**, armonia, fraseggio (p. 238) [P].
3. Rallentare "bigger" ma può sgonfiare; accelerare dà "edge" ma può diventare "frantic" (p. 238) [P].
4. **Armonia**: stessa tonalità, relativa, quarta o quinta (Camelot ±1); key lock con limite di artefatti (pp. 238–239) [P].
5. Senza consonanza né intro di batteria: far partire B **dopo** la fine di A (p. 221) [P].
6. Intro/outro senza armonia per sovrapposizioni lunghe (p. 224) [P].
7. **Hip-hop 89–105 BPM** e tabella dei tempi per genere (p. 231) [P].
8. Set: deve "andare da qualche parte"; **chunking** in blocchi di 2–3 brani; "low to high then back down" (pp. 241, 243) [P].
9. Effetti "are not the point. They can easily be overused" (p. 184); gain a unity, EQ kill nelle transizioni (pp. 180–181) [P].
10. Interviste: **bass swap** di Oakenfold (p. 42), shelving <150 Hz di BT (p. 37), dischi-ponte per cambiare genere/tonalità (p. 43), discesa graduale dopo un DJ più veloce (BT, p. 34), backspin di 2 battute ≈ 3 giri (p. 251), "accuracy and flow" (p. 252).

## Implicazioni per MusicMix (stato)
- Implementato: `key_clash_cut` (anche Webber p. 238), costo frasi da 4, Camelot già nell'albero, `w_tempo_down`.
- Proposto: scelta del tempo d'incontro "a metà" in `dj_render`; check del backbeat in `dj_critic`; chunking in `dj_set` (premiare coppie già giudicate buone, tenendole fuori dalla validazione).

## Limiti
Manuale per principianti; nessuna soglia quantitativa sul pitch "ridicolo" o sugli artefatti.
