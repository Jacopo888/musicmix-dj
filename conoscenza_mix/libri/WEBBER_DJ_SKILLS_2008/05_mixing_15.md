# Webber 2008 — Cap. 15 "Mixing Skills" (pp. 212–239)

## Tesi
Il mix è il mestiere di base; fatto male "can be excruciating for the audience" (p. 212). Le abilità: cue (radio e slip), beatmatching, sincronizzazione, fraseggio, armonia, remix dal vivo (p. 213).

## Tecniche passo-passo
- **Cue in cuffia, "splitting your brain"** (pp. 215–218): un orecchio sulla sala, uno sulla cuffia; esercizio: ascoltare solo il basso dagli altoparlanti e solo il rullante in cuffia (p. 217). Nei locali grandi il basso arriva in cabina **in ritardo** rispetto alla pista (p. 217).
- **Radio cue** (pp. 218–220): Start/Stop, lasciare circa un pollice prima del primo suono perché il piatto vada a regime.
- **Transizioni di base** (pp. 220–221): "The key and tempo of each song, along with the exact nature of the intro and outro must be taken into account" (p. 221). Procedura (per un DJ mobile con lista imposta): se le tonalità sono consonanti o c'è un intro di batteria, crossfade o beatmatch; altrimenti **far partire B subito dopo la fine di A** (p. 221).
- **Slip cue** (pp. 221–224): rilascio sul battito; serve per beatmatch, cut e per "flying in elements from one record (like horns, vocals, pads, sound effects, etc.) over another" (p. 221).
- **Beatmatching** (p. 224): molti dischi dance hanno intro e outro estesi "with little or no harmonic information" proprio per sovrapporsi senza preoccuparsi della tonalità (p. 224).
- **Fraseggio** (p. 225): frasi di **2, 4, 8 o 16 battute**; "Syncing the beats of the tracks without syncing their phrases will result in musically unstable mixes". Una strofa o un ritornello = 2–4 frasi; non interrompere una frase a metà "unless you're after a particular dramatic effect". Nei brani lineari ascoltare le frasi lunghe (16 battute). Oakenfold: estensione di 16/32 battute con due copie.
- **Sincronizzare due copie** (pp. 225–230): partire sul downbeat contando 1-2-3-4; confrontare le casse e poi verificare con i rullanti; nudging avanti/indietro con pitch, etichetta, perno, bordo del piatto; poi blend e scambio dei ruoli. Due copie quasi in sync danno phasing/flanging (p. 230).
- **Due dischi diversi** (p. 231): scegliere coppie la cui differenza di BPM si colma con il pitch "without making the track sound ridiculous". **Tempi per genere**: reggae 70–82, trip hop 80–92, **hip-hop 89–105**, house 119–128, break/electro 130–132, progressive 130–135, trance 135–140, techno 140+, hardcore 150–160, drum'n'bass 160–170 (p. 231). BPM = battiti in 15 s × 4.
- Beat detection automatica: funziona meglio su 4/4 regolare e "four on the floor" (p. 232).
- **Componenti del ritmo** (pp. 235–237): cassa in quarti (four on the floor), rullante su 2 e 4, hi-hat che suddivide; rock: cassa su 1 e 3; **hip-hop: hi-hat swingato** (sensazione di terzina); 6/8; drum'n'bass sull'Amen. Nel beatmatch si allineano cassa e rullante "so they work together, rather than in conflict" (p. 237).
- **Prima del beatmatch** (p. 238), check-list: BPM di entrambi; quale è più veloce; quanto accelerare il lento; come suona accelerato; come suona il veloce rallentato; **incontrarsi a metà?**; funzionano armonicamente?; poi ascoltare il fraseggio.
- **Rallentare/accelerare** (p. 238): rallentare può far sembrare un brano "bigger, but if done too much it may deflate the energy"; accelerare dà "more edge" ma può diventare "frantic". Mancuso non usava il pitch.
- **Mix armonico** (pp. 238–239): due tonalità diverse "have the potential to clash horribly"; è più gradevole andare alla **relativa**, alla **quarta o alla quinta**; ruota Camelot: ±1 ora o la stessa ora. Key lock: "there is a limit to how far from the original pitch you can take a track before digital artifacts begin to appear" (p. 239).
- **Mix tape** (p. 239): registrare le prove per riascoltare le transizioni a mente libera.

## Regole operative (con pagina)
| # | Regola | Pagina | Etichetta |
|---|---|---|---|
| X1 | Allineare le frasi (4/8/16), non solo i battiti; non tagliare una frase a metà salvo effetto voluto | p. 225 | [P] |
| X2 | Senza consonanza tonale né intro di batteria: niente sovrapposizione, B parte quando A è finito | p. 221 | [P] |
| X3 | Sovrapposizioni lunghe sugli intro/outro senza armonia | p. 224 | [P] |
| X4 | Scelta del tempo comune: considerare l'incontro a metà, ascoltare l'effetto su energia ("bigger"/"deflate", "edge"/"frantic") | p. 238 | [P] |
| X5 | Tonalità: stessa, relativa, quarta o quinta (Camelot ±1) | pp. 238–239 | [P] |
| X6 | Pitch/key-shift oltre un limite produce artefatti | p. 239 | [P] |
| X7 | Hip-hop 89–105 BPM; genere ≈ intervallo di tempo | p. 231 | [P] |
| X8 | Verifica dell'allineamento anche sul rullante (2 e 4) | pp. 226, 237 | [P] |

## Implicazioni per MusicMix
- X1 è già il criterio di `dj_select` (phrase_level) e del nuovo `w_phrase4` di `dj_set` (profilo `books`): punti d'uscita/ingresso su frontiere di 4 battute.
- X2 coincide con la regola B&B "cut them, don't blend them" (cap. 15): implementata come `key_clash_cut` (R4b) nel profilo `books`.
- X4: il punto d'incontro del tempo è oggi deciso in `dj_render`; **proposta**: per salti grandi preferire l'incontro a metà se entrambi gli stretch restano sotto soglia, e penalizzare il rallentamento forte del brano energico (in `dj_set` è già presente `w_tempo_down`).
- X7: l'intervallo 89–105 (hip-hop) è compatibile con i BPM del pool; brani a 70–80 BPM vanno trattati come mezzo tempo (già fatto con il ripiegamento d'ottava).
- X8: proposta di controllo QA sul backbeat per `dj_critic`.

## Limiti
Manuale per principianti; nessun dato sperimentale. Le soglie (pitch "ridicolo", limite del key lock) non sono quantificate.
