# Mappa: cinque libri sul mestiere DJ → processi della pipeline MusicMix

Libri (schede per capitolo nelle cartelle omonime, note di sintesi `*.md` in questa cartella):
**Katz** 2012 *Groove Music* (OUP) · **B&B** = Brewster & Broughton 2003 *How to DJ Right* (Grove; ebook senza pagine: "cap. N", pagina d'inizio se nota) · **Webber** 2008 *DJ Skills* (Focal) · **Butler** 2006 *Unlocking the Groove* (Indiana UP) · **Williams** 2013 *Rhymin' and Stealin'* (Michigan UP).

Etichette: **[P]** pratica dichiarata da DJ/manuali, **[A]** analisi accademica/etnografica, **[H]** ipotesi o valore scelto da noi, **[U]** giudizi dell'utente (galleria v1). Nessun libro misura l'effetto di una regola sulla qualità percepita: sono pratica e analisi, non evidenza sperimentale. Colonna "Stato": **impl.** = implementato nel profilo `books` (default invariati), **prop.** = proposta in `PROPOSTE_KATZ_BB_WEBBER_BUTLER_WILLIAMS.md`, **già** = la pipeline lo faceva già, **no** = non applicabile/ignoto.

Moduli: `dj_select` (cue, compatibilità), `dj_set` (albero del tipo, ordine Held–Karp, Viterbi dei cue), `dj_policy` (tecniche creative), `dj_render` (blend/cut, EQ, stretch, loudness), `dj_effects` (echo/filter), `dj_creative` (backspin, tape stop, build, loop, acapella…), `dj_qa` (controlli tecnici), `dj_critic` (giudice automatico), **`dj_form` (nuovo: categorie di Butler)**.

---

## 1. Selezione e ordine dei brani
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Scelta e ordine contano più della tecnica: "Where you mix is more important than how you mix" | B&B cap. 13 (p. 76) | già (Held–Karp con costi di arco) |
| Selezione e ordine determinano l'umore della sala | Webber pp. 96–97 (Kid Koala) | già |
| Equilibrio tra brani noti e nuovi; "human jukebox" da evitare | Webber pp. 240, 242; Katz p. 161 | no (manca la popolarità) |
| **Chunking**: blocchi di 2–3 brani già provati insieme | Webber p. 243; Butler p. 240 (Shiva: "blocks of things ... lock in like nothing else") | prop. (`dj_set.order_set`) |
| Dischi-ponte per cambiare genere o tonalità | Webber p. 43 (Oakenfold) | prop. (`dj_set`: costo di genere + nodo ponte) |
| Un brano famoso al punto culminante; classici verso la fine | Butler pp. 49, 252; Katz p. 250 | no (manca la popolarità) |
| Brano-àncora ricorrente e cambio su riff/basso simile | Katz pp. 80–82 | prop. (`dj_select`) |
| Raggruppare per "colore tonale" simile | Webber p. 130 (Radar) | già (`w_timbre`, `w_key_soft`) |

## 2. Compatibilità di tempo
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Pitch ±8 % come limite storico | Katz p. 169; Butler p. 54 | già (`blend_max_stretch` 8 %) |
| Oltre +4 % i brani suonano male; −3 % passa inosservato | B&B capp. 21, 25 | già (costo quadratico /8 %) |
| Hip-hop 89–105 BPM; genere ≈ intervallo di tempo | Webber p. 231 | già (ripiegamento d'ottava 70↔140) |
| Rallentare = "bigger" ma sgonfia; accelerare = "edge" ma "frantic"; valutare l'**incontro a metà** | Webber p. 238 | prop. (`dj_render.choose_tempo_fold`) |
| Lo stretch forte degrada l'audio | Webber pp. 166, 239 | già (limiti di stretch) |
| Nel set il tempo sale poco ("just a few beats per minute really makes a difference") | Butler pp. 250–252; B&B cap. 21 | impl. (`w_tempo_down`) |
| Dopo un DJ più veloce: stesso passo e discesa graduale su ~5 brani | Webber p. 34 (BT) | no (set singolo) |

## 3. Compatibilità di tonalità
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Stessa, relativa, quarta/quinta (Camelot ±1) | Webber pp. 238–239; B&B cap. 31 | già (vincolo dei blend) |
| Tonalità che cozzano: **cut**, non blend, o mix su parti solo percussive | B&B cap. 15, cap. 31; Webber p. 221 | impl. (`key_clash_cut`, R4b) |
| Key lock con limite di artefatti; trasposizioni di 1 semitono | Webber p. 239; Williams pp. 149–155 | già (`auto_key_shift` ±1 in `dj_policy`) |
| Due linee di synth/bassi dissonanti: toglierne una con l'EQ, o non mixarle | Butler pp. 53, 245 | prop. (`dj_render`: taglio di banda della linea in conflitto) |
| Stonatura tra strati = "cucitura" udibile | Williams p. 10 | già (vincolo dei blend) |

## 4. Compatibilità di energia
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Spinback/power-off verso un ingresso forte ("punchy") di B | B&B cap. 12 | impl. (`backspin_far_tempo`; `tape_stop_punchy` in `dj_policy`) |
| Dopo un culmine si può ripartire da un tempo/energia diversi ("fresh start") | B&B cap. 24 | già (R5, C6) |
| L'intensità ("harder, wilder, more driving", più strati, più effetti) tiene insieme il set | Butler pp. 252–253 | impl. parziale (`dj_form` conta gli strati; l'energia d'ordine resta onset/flusso/centroide) |

## 5. Compatibilità di genere e di ritmo
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Ritmi compatibili se rullanti e hi-hat cadono negli stessi punti o si completano; sincopi opposte = "fight in a drum factory" | B&B cap. 14 (p. 84) | prop. (`dj_select`: correlazione dei pattern di attacco per battito) |
| Four-on-the-floor vs breakbeat è la divisione percepita dai fan | Butler pp. 77, 115 | prop. (`dj_form.pulse_type`) |
| Codici di genere (timbro, strumentazione) più della sintassi; implicano spazi d'ascolto | Williams pp. 55, 71 | già in parte (`w_timbre`) |
| Mesh: un suono sui battiti dell'uno, l'altro sui levare | Butler pp. 243–244 | prop. (`dj_select`) |

## 6. Uscita di A (dove)
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Mai a metà frase; azioni su confini di 4/8 battute; "always let a singer finish their verse or chorus" | B&B capp. 10, 13; Webber p. 225; Katz p. 52 | già (cue su frase) + impl. (`w_exit_midsection`) |
| Hip-hop: uscire **subito dopo un ritornello** | B&B cap. 13 "Even tighter" | impl. (`w_exit_after_verse`) |
| Uscire dal break a 2/3–3/4 del brano; "mix out" nel primo breakdown | B&B cap. 13; Butler pp. 221–222 | impl. (`w_exit_breakdown_bonus`, da `dj_form`) |
| Non uscire mentre A "costruisce" verso il suo core ("the builds won't be the same") | Butler pp. 222–224, 242 | impl. (`w_exit_in_build`) |
| Non togliere il disco "when the shit was about to explode" | B&B cap. 13 (Herc) | impl. (stessa regola) |
| L'outro è spesso saltato; Frankie Bones taglia a 24 battute di outro | Butler p. 223, p. 225 n. 22 | già (uscita prima dell'outro, `dj_select` ≥ 35 %) |

## 7. Ingresso di B (dove)
| Cosa dicono | Pagina | Stato |
|---|---|---|
| B parte all'inizio di una frase ("the 1") | B&B cap. 10; Katz p. 232; Webber p. 129 | già + impl. (`w_entry_midsection`) |
| Ingresso sul ritorno della cassa ("dropping the beat") | Butler p. 3, pp. 245–247 | impl. (`w_entry_drop_bonus`) |
| Saltare intro lunghi; meglio usare la **prima cassa** come guida | Butler p. 225 n. 22, p. 233 | già (cue d'ingresso ≤ 50 %) |
| Lasciare suonare l'ingresso riconoscibile di B prima di effetti | Katz pp. 161, 168 | già (effetti su A, B pulito) |
| La voce di B non deve entrare troppo presto | Webber p. 45 (Oakenfold) | già (R6, guardia voce) |

## 8. Tipo di transizione
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Hip-hop: più cut che blend; cut sull'uno tra parti percussive | B&B capp. 9, 11 (p. 68); Katz pp. 56–57, 138 | impl. (`hiphop_cut` R6b, costo cut 0,20) |
| Spinback come "escape hatch" a fine frase | B&B cap. 12; Webber pp. 250–251 | impl. (R2b, `dj_policy` C5) |
| Blend lunghi solo con intro/outro senza armonia | Webber p. 224; B&B cap. 13 | già (R7 strumentale) |
| Tessiture sottili per mixare; niente blend "busy" | Butler pp. 129, 231, 244 *(corretto 21/10/2026: pp. 129, 232, 245)* | impl. (R7c_dense, `w_dense_overlap`) |
| Mai due voci predominanti insieme | Butler p. 243; B&B cap. 31 | già (R6, guardia voce) |
| Acapella su strumentale | B&B cap. 31; Williams p. 122 | già (`acapella_over` in `dj_policy` C3) |

## 9. Durata della transizione
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Quick blend nell'hip-hop | B&B cap. 9 | impl. (blend strumentale ≤ 8) |
| Ogni 4/8 battute un passo del mix (B dentro, bassi, A fuori) | B&B cap. 10 | già (blend 4/8/16 su frase) |
| "Third record": due dischi insieme a lungo (EDM) | Butler pp. 93, 242 | no per hip-hop (B&B: hip-hop corto) |
| Battle routine ~6 min, quick mix: più brani all'ora | Webber p. 75; Katz p. 228 | già (durate per brano) |

## 10. EQ e fader
| Cosa dicono | Pagina | Stato |
|---|---|---|
| **Bass swap**: un solo basso alla volta | B&B capp. 9, 28; Webber p. 42 (Oakenfold), p. 37 (BT <150 Hz) | già (`dj_render` low 200 Hz, swap 20 ms; QA basso) |
| "Slam in the bass" dopo qualche battuta senza bassi | B&B cap. 28 | prop. (`dj_render`) |
| Kill di banda: togliere tutto il basso o lasciare solo cassa e basso | Webber p. 181 | già |
| Gain staging: livelli medi pari; margine (headroom) | Webber pp. 180–181; B&B cap. 27 | già (LUFS per sezione, limiter) |
| Curva del crossfader: dolce per dissolvenze, ripida per tagli | Katz p. 133; Webber p. 182 | già (cut 10 ms, blend equal-power) |
| Togliere la cassa di un disco per allinearlo al breakdown dell'altro | Butler pp. 242–243 | prop. (`dj_render`) |

## 11. Effetti (echo, filter, backspin, scratch, loop, juggling)
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Effetti con moderazione: "not the point", "secret weapon, not constant companion" | Webber p. 184; B&B cap. 29; QBert in Webber p. 80 | impl. (`w_fx_run`, `fx_budget_penalty`) |
| Delay/eco a tempo (suddivisioni del battito) | Webber pp. 149, 184 | già (`echo_out` ½ o 1 battito) |
| Eco manuale a picchi decrescenti | Webber pp. 265–266 | già (feedback 0,5) |
| Backspin di 2 battute (≈3 giri); quick mix contando i backspin | Webber pp. 250–251; Katz p. 51 | già (`dj_creative.backspin` 1/2/4 battiti) |
| Withholding/drop: effetti che salgono, poi cassa "clean" | Butler pp. 91, 245–247 | impl. (`drop_entry_build` in `dj_policy`) |
| Beat-mute sulla punchline | Katz p. 106 | prop. (`dj_creative`) |
| Repeat trick (ripetere l'ultima battuta della strofa) | B&B cap. 32 | prop. (`dj_creative`) |
| Loop/estensione: rientro esatto sul downbeat; "accuracy and flow" | Webber p. 252; Katz pp. 56–57 | già (`loop_extend`) |
| Drum tip come fill sul downbeat d'ingresso | Webber p. 260 | prop. (`dj_creative`) |
| Double copies al climax; phasing breve | Butler pp. 151–154; Katz pp. 117–118 | no (rischio alto, non richiesto) |
| Scratch/juggling: tecniche di battle, devono restare ballabili | Webber cap. 19; B&B capp. 34–35; Katz cap. 4 | no |

## 12. Durata del brano nel set
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Club hip-hop: "two verses and chorus, or even just a chorus" | Katz p. 228 | impl. (`dur_mode="sections"`, 1–3 sezioni core, obiettivo 2) |
| Core = "the song", "the meat" | Butler p. 222 | impl. (core ≈ strofa + ritornello [H]) |
| Lasciar finire strofa/ritornello | B&B cap. 13; Webber p. 225 | impl. (uscita a fine sezione) |

## 13. Arco di energia
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Più picchi e valli, l'ultimo il più alto | Butler p. 250; B&B cap. 24 | impl. (`arc_shape="waves"`) |
| "goes up from low to high then it goes back down" | Webber p. 243 | già (arco `single`) |
| Warm-up lento; non suonare subito il brano più caldo; picco quando la pista è piena | B&B cap. 21; Webber p. 247 | impl. (nodo iniziale dell'arco a 0,3) |
| Il set deve andare da qualche parte senza esaurire il pubblico | Webber p. 241; Butler p. 250 (Jimmi Journey) | già (vincolo dell'arco) |
| Arco di una routine: familiarizza → sali → head-nod → scendi | Katz p. 168 | già (forma dell'arco) |

## 14. Apertura e chiusura
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Aperture: intro ambient senza cassa; intro "di cornice"; "bang it" | Butler pp. 247–249 | già (inizio dal primo cue o da 0 con fade-in 0,5 s) |
| Chiusure: stop secco; abbassare; outro simile all'intro; power-off | Butler p. 249; B&B cap. 12; Katz p. 196 | prop. (`dj_creative`: coda tape stop finale) |
| Primo brano noto ma non il più caldo | Webber p. 247 | no (manca la popolarità) |

## 15. Mashup e campionamento (Williams)
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Beat e flow separabili; acapella su break di un altro disco | Williams p. 122; B&B cap. 31 | già (`acapella_over`, `vocal_swap`) |
| Frasi cicliche di 4/8 battute; frammenti non quadrati o stonati si sentono | Williams pp. 10, 155 | già (quadratura) + prop. (controllo ciclicità) |
| Voce spostata su altra base: tempo −8 %, modo cambiato, eco sulla voce citata | Williams pp. 134–136 | prop. (`dj_creative` "acapella bridge" [H]) |
| Break del canone riconoscibili anche rallentati | Williams p. 34 | no |

## 16. Giudizio di qualità
| Cosa dicono | Pagina | Stato |
|---|---|---|
| Train wreck = errore grave ("shoes in a dryer"); deriva = beatmatch fallito | Butler pp. 55, 154; Webber p. 244 | già (`dj_qa` deriva/downbeat) |
| "Busy", "muddled": due superfici attive | Butler p. 244 | impl. (`dj_form.overlap_profile`) + prop. (`dj_critic`) |
| Due voci predominanti | Butler p. 243 | già (`dj_qa` clash) |
| Backbeat (2 e 4) allineato, non solo la cassa | Webber pp. 129, 226, 237 | prop. (`dj_critic`/`dj_qa`) |
| Downbeat percepito dopo l'ingresso della cassa (turning the beat around) | Butler pp. 127, 142, 233 | prop. (`dj_qa`) |
| Il momento musicale conta più della pulizia tecnica | B&B cap. 50; Katz p. 177 | no (resta ascolto) |
| Il pubblico valuta transizioni e scelta, i DJ notano ogni errore | Webber p. 244 | no |
