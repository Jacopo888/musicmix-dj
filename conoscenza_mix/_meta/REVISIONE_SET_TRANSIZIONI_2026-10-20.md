# Revisione bibliografica: tipo di transizione, set continui, preferenze, mashup

Data: 2026-10-20. Ricerca bibliografica in sola lettura: nessun file in `src/`, `tests/` o `runs/` modificato, nessun commit.
Completa e non ripete `REVISIONE_DJ_2026-10-20.md` (Kim 2020/2021/2022, Raveform, All-In-One, Vande Veire 2018, DJtransGAN, Zehren/M-DJCUE, CUE-DETR, MosaikBox, UnmixDB, Gebhardt 2016) e `DJMIX_STATS_HIPHOP_2026-10-20.md` (statistiche MixesDB/1001Tracklists). Quelle fonti qui sono solo richiamate.

Legenda (estende quella della revisione DJ):
- **[E]** risultato empirico misurato dagli autori (N dichiarato);
- **[S]** regola di progetto di un sistema pubblicato, valutata solo nel suo insieme;
- **[P]** pratica, storia o etnografia del DJing (manuali, interviste, recensioni), non una misura;
- **[T]** teoria o quadro concettuale;
- **[H]** ipotesi di questa revisione, da verificare con ascolti o dati propri.

"Letto" indica cosa è stato effettivamente letto in questa sessione: **full** (testo completo estratto da PDF ad accesso aperto), **parziale** (sezioni), **abstract/metadati** (abstract, record editoriale o Crossref), **recensioni** (solo recensioni accademiche o pratiche). Tutti i DOI elencati sono stati controllati su Crossref (api.crossref.org) salvo dove indicato.

Priorità: **alta** = utile subito per i passi (a)–(d) senza nuovi dati; **media** = utile quando ci saranno più ascolti o un set intero; **bassa** = contesto.

---

## 0. Sintesi

1. **Nessuno studio classifica i tipi di transizione (cut, blend, echo, filter, spinback, scratch) in mix reali**, né per l'EDM né per l'hip-hop. Esistono solo: durate (Kim 2020), curve fader/EQ (Kim 2021/2022), elenchi di tecniche implementate dai sistemi (Williams et al. 2025, che citano slam/cut, echo out, power down, loop, double drop) e una macchina a stati per la DnB (Vande Veire 2018). La scelta del tipo per il rap resta una decisione da costruire con i nostri ascolti.
2. **I costi che i sistemi pubblicati usano per scegliere dove e come transitare sono concreti e riusabili**: downbeat obbligatorio, peso forte sui confini di sezione, costo alto se entrambe le tracce hanno voce, costo alto se la regione è a basso volume (Bittner et al. 2017). Il primo difetto percepito dai curatori Spotify è il beat non allineato (15,2% delle transizioni), non la tonalità (0%).
3. **Ordinare un set è un problema di cammino minimo** su uno spazio di tempo (con equivalenza di ottava 70↔140), tonalità (circolo delle quinte e relativa) e timbro (Bittner 2017). Nei set reali di EDM l'ordine conta (timbro più vicino che in un ordine casuale, p < 0,0001) e i passi regolari imitano i DJ meglio della "vicinanza al primo brano" (Kell & Tzanetakis 2013).
4. **Il tempo ha una soglia percettiva misurata**: con 96 ascoltatori il disagio cresce oltre ±10% di stretch, con sensibilità maggiore al rallentamento; dividere lo stretch tra A e B riduce il disagio (Ishizaki et al. 2009).
5. **La sovrapposizione di due voci è il rischio più documentato**: nel parlato competitivo domina il mascheramento informativo, peggiore quando le voci sono simili (Brungart 2001); la voce principale cattura l'attenzione nel mix (Bürgel et al. 2021); nei mashup la voce sovrapposta indebolisce il legame tra compatibilità stimata e gradimento (Davies et al. 2014).
6. **Groove ed energia** hanno descrittori audio misurati (varianza RMS, flusso spettrale in bassa frequenza, densità di eventi, sincope a U rovesciata), ma il gusto dell'ascoltatore pesa più di tutti (Senn et al. 2018). Nessuno studio collega queste misure alla qualità di una transizione.
7. **Con 6 giudizi non si stima un modello di preferenza.** La letteratura indica confronti a coppie forzati (varianza minima, Mantiuk et al. 2012), Bradley–Terry/Thurstone o GP di preferenza, selezione attiva delle coppie (BALD, Crowd-BT) e un hold-out vero (Cawley & Talbot 2010).
8. **Il "drop" funziona dopo una preparazione** (breakdown, build-up): nei dati EDM la risposta elettrodermica è più alta nel drop che nelle altre sezioni (Solberg & Dibben 2019) e il movimento cala nel breakdown e risale al drop (Solberg & Jensenius 2016/2017). È coerente con il 3/10 del drop diretto sul ritornello di B, ma resta diagnostica.

---

## 1. Tipo di transizione e pratica DJ hip-hop

### 1.1 Fonti

**1. Bittner, Gu, Hernandez, Humphrey, Jehan, McCurry, Montecchio (2017).** *Automatic Playlist Sequencing and Transitions.* Proc. 18th ISMIR, Suzhou, pp. 442–448.
- URL: https://archives.ismir.net/ismir2017/paper/000086.pdf (CC BY 4.0). Letto: **full**.
- Tipo: [S] + [E] (pilot con curatori professionisti, A/B su utenti Spotify).
- Transizioni: punti solo sui downbeat, "heavily weighted" sui confini di sezione e sui drop (stimati dai dati di scrubbing del playhead). Potatura: uscita di A nell'ultimo 25% del brano, ingresso di B nel primo 20%. Costo per ogni coppia di candidati = somma pesata e standardizzata di sei matrici: distanza timbrica, distanza di chroma, loudness inversa (regioni basse "sound awkward and abrupt"), "vocalness" sommata delle due tracce (rischio di voci sovrapposte o taglio a metà frase), fine non su drop, fine non su confine di sezione (penalità 2 se nessuna delle due, 1 se una). Tempo: rampa battuta per battuta da A a B.
- Valutazione: 48 coppie di brani dance con differenza ≤ 5 BPM, 4 curatori, ascolto integrale. **Good 64,13%, OK 28,26%, Bad 7,61%, media 2,56/3.** Cause segnalate: beat non allineati 15,22%, transizione a metà voce 6,52%, brani in contrasto 4,34%, non sul downbeat 2,17%, punti goffi 2,17%, **key clash 0%**. Gli autori attribuiscono i primi due a errori di beat tracking e di rilevazione vocale.
- A/B su 10% degli utenti: transizioni curate da DJ in alcune playlist → +1,4 punti percentuali di ritorno giornaliero alla playlist. Circa il 95% degli utenti non usa il crossfade fisso.
- Limite dichiarato: "The constraints we imposed may not be necessary or sufficient for other genres of music, for example rap."
- **Implica per MusicMix (alta)**: la matrice di costo di Bittner è un baseline interpretabile per il passo (a) e per la scelta dei punti: vocalness congiunta, loudness bassa e fine su confine sono già quasi tutte nel nostro gate. Il dato "key clash 0%" su coppie entro 5 BPM [E, EDM/dance] non autorizza a togliere il gate tonale per i blend lunghi, ma suggerisce di non trattarlo come criterio dominante rispetto all'allineamento dei beat.

**2. Williams, Meehan, Lattner, Pauwels, Barthet (2025).** *Temporal Considerations in DJ Mix Information Retrieval and Generation (Short Paper).* TIME 2025, LIPIcs 355, art. 20, pp. 20:1–20:8.
- DOI: https://doi.org/10.4230/LIPIcs.TIME.2025.20 (CC BY 4.0; DOI non in Crossref, verificato sulla pagina Dagstuhl). Letto: **full** (6 pagine).
- Tipo: [T] rassegna con tassonomia.
- Gerarchia temporale: (1) livello transizione, (2) livello traccia (quale brano dopo), (3) livello mix (flusso e contenuto del set). "To the best of our knowledge, no existing works attempt" a operare su tutti e tre i livelli. Elenca le tecniche implementate dai sistemi pubblicati: crossfade beatmatched a durata fissa, EQ a 3 bande, evitare clash vocali, double drop/rolling/relaxed, loop out, loop come rullata, campioni o voce da MC, **slam/cut/switch, echo out, power down**. Nessun sistema di generazione automatica di transizioni ha eguagliato i DJ umani. "No such corpus [di mix con audio ed etichette] is currently publicly available."
- Cita una tesi triennale (Hawkins 2013, Ohio State, https://kb.osu.edu/handle/1811/54438) per l'affermazione che usare sempre le stesse poche tecniche rende le transizioni riconoscibili come automatiche e peggio giudicate all'ascolto ripetuto. Evidenza debole: tesi non peer-reviewed, non letta.
- **Implica (alta)**: architettura a tre livelli per i passi (a) e (b). La scelta del tipo (livello 1) deve ricevere vincoli dal livello 3 (energia del set) e non essere ottimizzata da sola. La varietà dei tipi su un set di 10–15 brani è un requisito plausibile [H], da misurare con ascolto di set interi.

**3. Thalmann, Thompson, Sandler (2018).** *A User-Adaptive Automated DJ Web App with Object-Based Audio and Crowd-Sourced Decision Trees.* Web Audio Conference, Berlino.
- Record: https://core.ac.uk/works/8850420 (testo non accessibile, 403). Letto: **abstract/metadati** (via ricerca e bibliografia di Williams 2025).
- Tipo: [S]. Parte da un albero di decisione predefinito "modellato sulla pratica DJ comune" e lo adatta al gusto dell'utente con scelte interattive semplici; usato tra le tecniche citate: cut, echo out, power down.
- **Implica (media)**: per il passo (a) un albero di decisione esplicito (voce sì/no, sezione, compatibilità di tempo, energia) aggiornato dai giudizi è più adatto ai pochi dati di un classificatore appreso.

**4. Gates, Subramanian, Gutwin (2006).** *DJs' perspectives on interaction and awareness in nightclubs.* Proc. 6th Conference on Designing Interactive Systems (**DIS '06**, non CHI), pp. 70–79.
- DOI: https://doi.org/10.1145/1142405.1142418. Letto: **abstract/metadati**.
- Tipo: [E] qualitativo (interviste a 11 DJ di più città).
- I DJ raccolgono molte informazioni sul pubblico e le usano per pianificare e modellare la serata; sono scettici verso tecnologie che danno al pubblico il controllo della musica.
- **Implica (bassa)**: in un set offline non c'è pubblico; la "curva di energia" è un sostituto della lettura della pista e va dichiarata come scelta di progetto [H], non come imitazione validata.

**5. Katz, Mark (2012).** *Groove Music: The Art and Culture of the Hip-Hop DJ.* New York: Oxford University Press. ISBN 978-0-19-533112-7.
- Letto: **recensioni** (Scratched Vinyl, 30/05/2012; *Symposium* vol. 53 e *Notes* 70, 2013, pp. 109–111, solo come metadati; *J. Society for American Music* 2015, paywall). Nessun estratto del libro letto.
- Tipo: [P] storia ed etnografia, basata su interviste con DJ e sull'esperienza dell'autore come DJ.
- Contenuti attestati dalle recensioni: il giradischi come strumento; cap. 2 "Mix and Scratch" sulle innovazioni di Grandmaster Flash e GrandWizzard Theodore; cultura delle battle e turntablism; diffusione, razza e genere; effetti delle tecnologie digitali. La pratica di Kool Herc (break ripetuti passando da un disco all'altro) e la "Quick Mix Theory" di Flash (backspin e copie doppie per allungare i break) sono descritte da fonti secondarie (Wikipedia; MIT OCW 21M.380, lez. 16), non verificate nel libro.
- **Implica (media)**: conferma come pratica [P] che il vocabolario hip-hop è fatto di break, cut e sezioni corte, coerente con cut "sull'uno" e blend brevi. Non dà numeri: non può validare durate o soglie.

**6. Broughton, Frank & Brewster, Bill.**
- *How to DJ Right: The Art and Science of Playing Records.* New York: Grove Press, 2003 (ISBN 978-0-8021-3995-5; ed. UK *How to DJ (Properly)*, Bantam, 2002).
- *Last Night a DJ Saved My Life: The History of the Disc Jockey.* New York: Grove Press, 2000 (©1999); edizione aggiornata 2006; edizione rivista Grove 2014. L'edizione UK Headline 1999 non è stata confermata.
- Letto: **metadati** (cataloghi, Publishers Weekly). Tipo: [P] manuale e storia.
- **Implica (bassa)**: fonte di pratica per il glossario dei tipi di transizione; da leggere (indice/estratti) prima di citarne regole specifiche.

**7. Smith, Sophy (2013).** *Hip-Hop Turntablism, Creativity and Collaboration.* Farnham/Burlington: Ashgate. ISBN 978-1-4094-4337-7.
- Letto: **metadati e recensione** (*Organised Sound*, Cambridge). Tipo: [P] etnografia di team di turntablism (Scratch Perverts, The Mixologists…), con un sistema di notazione delle routine.
- **Implica (bassa)**: riguarda la performance di scratch in team, non il mix continuo di brani.

**8. Hansen, Fabiani, Bresin (2011).** *Analysis of the Acoustics and Playing Strategies of Turntable Scratching.* Acta Acustica united with Acustica 97(2):303–314.
- DOI: https://doi.org/10.3813/AAA.918410. Letto: **abstract** (PDF dell'autore scaricato, non analizzato in dettaglio).
- Tipo: [E] 3 DJ professionisti, improvvisazioni con emozioni diverse, audio e gesti registrati; prime stime degli intervalli funzionali; il pitch conta meno che in altri strumenti.
- **Implica (bassa)**: lo scratch non è tra i tipi da implementare ora; se servisse, è un riferimento per parametri di gesto.

**9. Mouraviev, Ivan (2024/2025).** *DJ Performance Analysis: Issues, concepts, methods.* Performance Research 29(6):130–137.
- DOI: https://doi.org/10.1080/13528165.2024.2537579 (date discordanti: 2024 nel record Huddersfield, 2025 altrove). Letto: **abstract**.
- Tipo: [T]/[P] metodi per analizzare set registrati (track listing, tempo mapping, spettrogrammi), applicati a un set dub di Bristol.
- **Implica (bassa)**: metodo per annotare a mano mix rap di riferimento (P2 della revisione DJ).

**10. Altre fonti minori.**
- Hirai, Doi, Morishima (2015). *MusicMixer: Computer-aided DJ system based on an automatic song mixing.* ACE 2015, pp. 1–5. DOI: https://doi.org/10.1145/2832932.2832942. Metadati. [S].
- "DJ-AI" (autori non estratti dal PDF; Sabancı University, repository https://research.sabanciuniv.edu/52373/). *DJ AI: Optimizing Playlist Alignment and Generating Transitions with Generative and Embedding Models.* Audio Mostly 2025 (DOI segnaposto "XXXXXXX" nel PDF). Letto: **full**. [S] gate Camelot identico/adiacente/complementare, ±2 BPM, energia crescente con variazione ≤ 0,2 "RMS units", cammino più lungo con DFS; transizioni generate con MusicGen. Ascolto: 20 partecipanti × 10 coppie contro crossfade; metriche oggettive di chroma molto vicine (0,3735 contro 0,3888). Evidenza debole, priorità **bassa**.

### 1.2 Lacuna esplicita

Nessuna fonte trovata misura la frequenza dei tipi di transizione nei mix reali, né il loro gradimento per contesto (voce, sezione, energia, tempo). Ricerche fatte: "DJ mix transition type classification", "DJ technique detection DJ mixes", "transition recognition spinback/echo/scratch", più la bibliografia di Williams 2025 e di Kim 2020. Il dataset Raveform annota struttura, non tipi di transizione. Questa assenza va conservata come "ignoto": ogni regola di scelta del tipo per il rap è [H].

---

## 2. Sequenziamento di set e playlist

**11. Cliff, Dave (2000).** *Hang the DJ: Automatic Sequencing and Seamless Mixing of Dance-Music Tracks.* HP Laboratories Technical Report HPL-2000-104.
- URL: http://www.hpl.hp.com/techreports/2000/HPL-2000-104.html (server HP non raggiungibile in questa sessione; copia ResearchGate non scaricata). Letto: **abstract** (via indice di ricerca) e citazioni in Bittner 2017 e Williams 2025.
- Tipo: [S]. L'utente fornisce un insieme di brani e una descrizione approssimativa dello stile del mix; il sistema produce un unico file continuo. Bittner 2017 lo cita per le "tempo trajectories" come modello del modo in cui i DJ strutturano la salita e discesa di energia. Il seguito hpDJ (Cliff 2006, "with floorshow feedback") è stato provato su pubblico in un club di Londra (fonte secondaria).
- **Implica (media)**: per il passo (b), la traiettoria di tempo/energia come input dell'utente (es. "sali da 75 a 95 BPM e scendi") è un'interfaccia più onesta di una curva "ottimale" inventata.

**12. Bittner et al. (2017)**, parte sequenziamento (vedi fonte 1).
- [S] Spazio euclideo: vettori acustici (CNN che predice vettori di collaborative filtering, ridotti a 8 dimensioni, "mostly timbral"), tonalità in R³ con quinte e relative equidistanti, tempo in scala log₂ o come vettore unitario su un cerchio di ottava (44, 88, 176 BPM sullo stesso angolo). Pesi per applicazione (in una playlist da ballo il tempo pesa di più).
- Ordine = cammino hamiltoniano minimo (NP-completo): HAM-1 greedy dal seme, HAM-2 greedy da testa o coda; in alternativa ciclo con 2-approssimazione. Il greedy lascia accoppiamenti cattivi alle estremità.
- [E] pilot: curatori professionisti, 6 playlist di circa 30 brani ordinate (HAM-2) contro casuali. In 3 playlist su 6 preferenza unanime per l'ordinata, in 2 nessuna preferenza, in 1 mista. Coppie "brusche" su 29: **media 4,2 (casuale) contro 2,4 (ordinata)**. Nella playlist "Underground Hip-Hop, Funk": 3,8 contro 1,2; nella playlist "Abstract Hip-Hop, Indietronica": 2,7 contro 3,3, cioè l'ordinata è peggiore.
- **Implica (alta)**: il passo (b) può partire da questo schema con le nostre feature (BPM con piegatura d'ottava, Camelot, energia, timbro da embedding), con ottimizzazione esatta (10–15 brani: programmazione dinamica di Held–Karp, fattibile) invece del greedy. Il risultato misto sull'hip-hop astratto avverte che il solo timbro non basta.

**13. Kell, Thor & Tzanetakis, George (2013).** *Empirical Analysis of Track Selection and Ordering in Electronic Dance Music using Audio Feature Extraction.* Proc. 14th ISMIR, pp. 505–510.
- URL: https://archives.ismir.net/ismir2013/paper/000210.pdf. Letto: **full**.
- Tipo: [E]. Dati: 114 set dell'Essential Mix BBC (2009–2011), confrontati con 189 album, 100 set di brani EDM casuali e versioni rimescolate degli stessi set. Estratti di 2 minuti ogni 4 minuti.
- Risultati: i set dei DJ sono timbricamente più coerenti dei brani EDM casuali (meno degli album) e **l'ordine originale ha transizioni timbriche minori dell'ordine rimescolato (p < 0,0001)**; tempo e loudness sono controllati dal DJ (variazioni minime); per la tonalità (Echo Nest) **nessuna differenza significativa**: "DJs do not use key as a primary concern when selecting and ordering". Euristiche di ordinamento: le inversioni rispetto all'ordine del DJ sono minori per "Median" (171,85) ed "Equal-Step" (172,75) che per l'ordinamento per similarità decrescente dal primo brano (186,27; casuale circa 200, feature Echo Nest).
- **Implica (alta)**: preferire passi di cambiamento regolari lungo il set ("equal step") a "sempre il più simile". La tonalità nell'ordine dei set EDM non è un criterio primario: per noi serve soprattutto come vincolo locale della sovrapposizione nei blend.

**14. Ishizaki, Hoashi, Takishima (2009).** *Full-Automatic DJ Mixing System with Optimal Tempo Adjustment based on Measurement Function of User Discomfort.* Proc. 10th ISMIR, Kobe, pp. 135–140.
- URL: https://doi.org/10.5281/zenodo.1418231. Letto: **full**.
- Tipo: [E] + [S]. 96 ascoltatori indicano quando lo stretch (SOLA o SoundTouch) diventa fastidioso. Picchi dell'istogramma del disagio a **fattore 1,10 (accelerazione) e 0,90 (rallentamento)**; ascoltatori **più sensibili al rallentamento**; dentro 0,90–1,10 provano disagio il 15,42% (SOLA) e l'11,31% (SoundTouch). Funzione di disagio usata per scegliere il tempo intermedio e dividere lo stretch tra A e B (es. A +5% e B −5% invece di A +10%). Valutazione: 27 soggetti, 165 giudizi per condizione, metodo proposto migliore su tutti gli item (t appaiato, p < 0,001); alcune coppie restano inaccettabili "regardless of" stretch.
- **Implica (alta)**: per il passo (b) i salti di tempo oltre circa ±10% tra brani consecutivi vanno pagati con un costo forte o con un cut; il nostro limite ±8% per i blend è coerente. Dividere lo stretch tra A e B (rampa) è un'opzione del renderer da provare [H].

**15. Liebman, Saar-Tsechansky, Stone (2015).** *DJ-MC: A Reinforcement-Learning Agent for Music Playlist Recommendation.* Proc. AAMAS 2015, Istanbul.
- URL: https://arxiv.org/abs/1401.1880 ; https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2015/aamas/p1983.pdf. Letto: **full** (preprint arXiv).
- Tipo: [S] + [E]. MDP con ricompensa separata per brano e per **transizione** (3.400 feature binarie: coppie di decili di 34 descrittori). Esperimento con 47 studenti: 24 con baseline greedy (solo preferenze di brano), 23 con DJ-MC; 25 brani di esplorazione casuale e 25 di sfruttamento; ogni brano 60 s con crossfade; like/dislike per brano e per transizione. DJ-MC ha ricompensa di transizione maggiore nella fase di sfruttamento ("small but significant boost"), mentre la ricompensa di brano è comparabile.
- Cautela metodologica: la significatività (p ≪ 0,01) deriva da 250.000 ricampionamenti di sottoinsiemi di 8 partecipanti seguiti da un t-test, una procedura che gonfia la significatività con N piccolo.
- **Implica (media)**: per il passo (c) tenere separati "mi piace B" e "mi piace la transizione A→B", come già fanno i giudizi per clip. Non imitare la statistica: con pochi ascoltatori servono intervalli onesti.

**16. Rassegne e playlist.**
- Bonnin, Geoffray & Jannach, Dietmar (2014). *Automated Generation of Music Playlists: Survey and Experiments.* ACM Computing Surveys 47(2), art. 26, pp. 1–35. DOI: https://doi.org/10.1145/2652481. Letto: **metadati**. [T] rassegna: generazione di playlist e valutazione offline/utente. Priorità **bassa**.
- Jannach, Lerche, Kamehkhosh (2015). *Beyond "Hitting the Hits": Generating Coherent Music Playlist Continuations with the Right Tracks.* RecSys 2015, pp. 187–194. DOI: https://doi.org/10.1145/2792838.2800182. Letto: **abstract**. [S]+[E] riordino che rende la continuazione coerente con i brani appena suonati. **Bassa**.
- Flexer, Schnitzer, Gasser, Widmer (2008). *Playlist Generation Using Start and End Songs.* ISMIR 2008, pp. 173–178. URL: https://archives.ismir.net/ismir2008/paper/000143.pdf. Letto: **abstract e introduzione**. [S]+[E] transizione graduale tra un brano iniziale e uno finale scelti dall'utente, solo similarità audio, valutata con generi e test d'ascolto. **Media**: un set con primo e ultimo brano fissati è un vincolo naturale per il passo (b).
- Cunningham, Bainbridge, Falconer (2006). *"More of an art than a science": Supporting the creation of playlists and mixes.* ISMIR 2006. URL: https://archives.ismir.net/ismir2006/paper/000085.pdf. Letto: **abstract**. [E] qualitativo (interviste e post, grounded theory). **Bassa**.

**17. Sequenze di album (non DJ).**
- Neto, Hartmann, Luck, Toiviainen (2024). *The algorithmic nature of song-sequencing: statistical regularities in music albums.* arXiv:2408.04383 (preprint). Letto: **parziale**. [E] dati Spotify API (475.342 album di 26.248 artisti prima dei filtri): brani con valence, energia e loudness alte più probabili in apertura; **tra brani consecutivi energia e valence tendono ad alternare salite e discese**.
- Herrmann, Ashley, Schmidhuber (2024). *Automatic Album Sequencing.* arXiv:2411.07772. Letto: **abstract**. [S] trasformatore inferiore all'approccio "narrative essence", superiore al caso.
- **Implica (media)**: la curva di energia non deve essere monotona; un'alternanza locale con una forma globale è più vicina ai dati sugli album [E per gli album, H per i set DJ].

**18. Lacuna.** Nessuno studio trovato misura la curva di energia su molti set DJ reali (warm-up, picco, chiusura). Le descrizioni di "energy arc" sono di guide pratiche, non valutate. Dancecult ha aperto una call per un numero speciale "The DJ Set" (scadenza 1/6/2026); da ricontrollare in futuro.

---

## 3. Groove, sincope ed energia

**19. Janata, Tomic, Haberman (2012).** *Sensorimotor coupling in music and the psychology of the groove.* J. Exp. Psychol.: General 141(1):54–75.
- DOI: https://doi.org/10.1037/a0024208 ; copia Zenodo https://zenodo.org/record/997607. Letto: **parziale** (abstract, Studio 1).
- Tipo: [E]. Studio 1: 19 studenti, 148 estratti di 30 s (folk, jazz, rock, soul/R&B). Rating di groove molto coerenti tra ascoltatori (α standardizzato 0,81). Groove maggiore per tempo più veloce (F(1,114) = 25,12, p < 0,0001) e per soul/R&B; massimo 108,7/127 per *Superstition*. **Correlazione groove–gradimento r = 0,82**, groove–familiarità r = 0,57. Studi successivi: il groove è inversamente legato alla difficoltà percepita di sincronizzarsi; brani ad alto groove inducono movimento spontaneo.
- **Implica (media)**: "groove" e "piacere" quasi coincidono a livello di brano; non va trattato come una dimensione indipendente di qualità. Il tempo più alto aumenta il groove percepito: una salita di tempo nel set ha un effetto atteso su energia e gradimento [E su estratti, H sui set].

**20. Witek, Clarke, Wallentin, Kringelbach, Vuust (2014).** *Syncopation, Body-Movement and Pleasure in Groove Music.* PLoS ONE 9(4):e94446.
- DOI: https://doi.org/10.1371/journal.pone.0094446 (CC BY). Letto: **parziale** (abstract, metodo).
- Tipo: [E] sondaggio web, 66 partecipanti (57 analizzati), 50 drum break funk con sincope diversa (indice di sincope tipo Longuet-Higgins & Lee). **Relazione a U rovesciata**: sincope media → più voglia di muoversi e più piacere, soprattutto per chi ama ballare; l'entropia audio è un cattivo predittore.

**21. Matthews, Witek, Heggli, Penhune, Vuust (2019).** *The sensation of groove is affected by the interaction of rhythmic and harmonic complexity.* PLoS ONE 14(1):e0204539.
- DOI: https://doi.org/10.1371/journal.pone.0204539. Letto: **parziale**. [E] 201 partecipanti online: U rovesciata per la complessità ritmica; accordi molto complessi attenuano l'effetto del ritmo sul piacere; il training musicale accentua la U.

**22. Madison, Guy.**
- (2006) *Experiencing Groove Induced by Music: Consistency and Phenomenology.* Music Perception 24(2):201–208. DOI: https://doi.org/10.1525/mp.2006.24.2.201. Letto: **abstract**. [E] 64 esempi commerciali; forte accordo tra ascoltatori; quattro fattori (regolare–irregolare, groove, swing, flowing).
- Madison, Gouyon, Ullén, Hörnström (2011). *Modeling the tendency for music to induce movement in humans: First correlations with low-level audio descriptors across music genres.* J. Exp. Psychol.: HPP 37(5):1578–1594. DOI: https://doi.org/10.1037/a0024323. Letto: **abstract**. [E] 100 esempi da 5 generi tradizionali; correlati più forti: densità di eventi tra i beat e salienza del beat; alcuni effetti dipendono dal genere.

**23. Stupacher, Hove, Janata (2016).** *Audio Features Underlying Perceived Groove and Sensorimotor Synchronization in Music.* Music Perception 33(5):571–589.
- DOI: https://doi.org/10.1525/mp.2016.33.5.571. Letto: **abstract**.
- Tipo: [E]. Varianza della curva RMS e flusso spettrale in bassa frequenza predicono il groove; una misura della variabilità tra beat batte altre misure di densità; il groove non è solo volume (manipolazione di loudness); **bassi più gravi → groove più alto, tap più forti e meno variabili**.
- **Implica (alta, come descrittore)**: per la curva di energia del passo (b) usare descrittori misurati sul nostro audio (LUFS a breve termine, varianza RMS, flusso spettrale sotto ~200 Hz, densità di onset dallo stem di batteria) invece di un'unica "energia". Restano descrittori, non una misura validata di groove sul rap [H].

**24. Senn, Kilchenmann, Bechtold, Hoesl (2018).** *Groove in drum patterns as a function of both rhythmic properties and listeners' attitudes.* PLoS ONE 13(6):e0199604.
- DOI: https://doi.org/10.1371/journal.pone.0199604. Letto: **abstract** (PDF scaricato).
- Tipo: [E] 248 pattern di batteria ricostruiti (pop, rock, funk, metal, **hip hop**, soul, R&B), 665 partecipanti, 8.329 rating. Sincope e densità di eventi associate positivamente al groove (più nei professionisti); salienza del beat, microtiming residuo e variabilità senza effetto. **L'effetto singolo maggiore è il "style bias" dell'ascoltatore (R² = 0,123)**: stile gradito e familiarità predicono più delle proprietà ritmiche.
- **Implica (alta per (c))**: il gusto pesa più delle proprietà del segnale; un modello di preferenza personale (uno solo, l'utente) è più giustificato di una formula universale di groove.

**25. Wesolowski & Hofmann (2016).** *There's More to Groove than Bass in Electronic Dance Music: Why Some People Won't Dance to Techno.* PLoS ONE 11(10):e0163938. DOI: https://doi.org/10.1371/journal.pone.0163938. Letto: **abstract**. [E] 99 studenti, 198 estratti da 11 sottogeneri EDM; cluster con basso non isocrono e variazione ritmica medio-alta più graditi; differenze tra cluster piccole (circa 0,15 logit). **Bassa**.

**26. Streich, Sebastian & Herrera, Perfecto (2005).** *Detrended Fluctuation Analysis of Music Signals: Danceability Estimation and further Semantic Characterization.* AES 118th Convention.
- URL: https://mtg.upf.edu/node/397 ; implementazione: https://essentia.upf.edu/reference/std_Danceability.html. Letto: **metadati e documentazione Essentia**.
- Tipo: [S]. Danceability da DFA, output Essentia circa 0–3. Non validata sul rap.
- **Implica (bassa)**: descrittore disponibile in Essentia (già usata nel progetto per KeyExtractor nel venv `venv-allin1`, secondo il CHANGELOG); utile solo come feature in più, non come criterio.

**27. Lacuna.** Nessuno studio trovato collega groove/danceability alla qualità di una transizione o al tipo di transizione scelto. Il legame "energia del set ↔ tipo di transizione" resta [H].

---

## 4. Mashability e sovrapposizione di voci

**28. Davies, Hamel, Yoshii, Goto (2014).** *AutoMashUpper: Automatic Creation of Multi-Song Music Mashups.* IEEE/ACM TASLP 22(12):1726–1737.
- DOI: https://doi.org/10.1109/TASLP.2014.2347135 ; copia autore https://staff.aist.go.jp/m.goto/PAPER/IEEETASLP201412davies.pdf. Versione breve ISMIR 2013. Letto: **full** (r e p delle figure non estraibili dal PDF).
- Tipo: [S] + [E]. Mashability tra sezioni di frase calcolata su similarità armonica (con trasposizione), ritmica e bilanciamento spettrale, consentendo cambio di tonalità e tempo. Ascolto: 15 partecipanti, 12 estratti di 32 beat da 90 brani, per ognuno il mashup di rango massimo, medio e minimo (36 stimoli), voto 0–10 rispetto all'originale. **Medie 6,7 / 4,7 / 4,3**; differenza significativa tra primo e medio, non tra medio e ultimo; correlazione mashability–gradimento significativa. Un mashup ben classificato prende 3,6 per voci sovrapposte e cambi d'accordo fuori dal downbeat. Separando i casi, **la correlazione è molto più forte senza sovrapposizione vocale**; alcuni ascoltatori tollerano le voci sovrapposte, altri no.
- **Implica (alta per (d))**: la mashability è utile come ordinamento dei candidati, non come garanzia; la voce sovrapposta è il principale fattore di insuccesso e di disaccordo tra ascoltatori. Per l'utente serve un test esplicito "voce su voce" contro "voce su strumentale".

**29. Lee, Lin, Yao, Lee, Wu (2015).** *Automatic Mashup Creation by Considering both Vertical and Horizontal Mashabilities.* ISMIR 2015, pp. 399–405.
- URL: https://www.ismir2015.uma.es/articles/302_Paper.pdf. Letto: **parziale**.
- Tipo: [S] + [E]. Più tracce di fondo (strumentali) e più tracce lead (voci); mashability verticale (lead su fondo, con "harmonic change balance" e peso del volume) e orizzontale (raccordo tra segmenti consecutivi); quattro livelli di mashability confermati da test soggettivi. Senza rhythmic matching per le voci (nessun kick/snare).
- **Implica (media)**: per il passo (d) separare la compatibilità verticale (voce di A su strumentale di B) da quella orizzontale (come si entra e si esce dal mashup).

**30. Huang, Wang, Smith, Song, Wang (2021).** *Modeling the Compatibility of Stem Tracks to Generate Music Mashups.* AAAI 2021, 35(1):187–195.
- DOI: https://doi.org/10.1609/aaai.v35i1.16092 ; https://arxiv.org/abs/2103.14208. Letto: **full**.
- Tipo: [S] + [E]. Stem separati (voce, armonia, percussioni). Pipeline: stem armonici entro **±3 semitoni** e percussivi entro **rapporto di tempo 0,8–1,2** dalla voce seme, poi Rubber Band. Addestramento auto/semi-supervisionato su 33.192 clip interne (positivi = stem dello stesso brano). Ascolto: 40 soggetti (28 per generazione, 12 per retrieval; Cronbach α 0,79 e 0,73), scala 1–5 su ritmo, armonia, timbro, innovazione, globale. I modelli appresi battono AutoMashUpper su globale e armonia. **Lo stretch globale causa sfasamento dei beat nelle battute successive** quando il tempo varia nel clip.
- **Implica (alta per (d))**: (i) le finestre ±3 semitoni e 0,8–1,2 sono limiti di pipeline [S], più larghi dei nostri per i blend; (ii) lo stretch va fatto a tratti sulla griglia dei beat (già così nel nostro renderer) e verificato su tutta la durata del mashup; (iii) un modello appreso richiede decine di migliaia di esempi: per noi solo regole + ascolto.

**31. Chen, Smith, Yang (2020).** *Neural Loop Combiner: Neural Network Models for Assessing the Compatibility of Loops.* ISMIR 2020, pp. 424–431. https://arxiv.org/abs/2008.02011. Letto: **parziale**. [E] 116 partecipanti; CNN > Siamese > AutoMashUpper. **Bassa**.

**32. Xing et al. (2020).** *PopMash: an automatic musical-mashup system using computation of musical and lyrical agreement for transitions.* Multimedia Tools and Applications 79:21841–21871. Solo citazione da Huang 2021, non letto. Rilevante perché considera anche il testo; **bassa** finché non letto.

**33. Separazione delle sorgenti usata nel DJing.**
- Hennequin, Khlif, Voituret, Moussallam (2020). *Spleeter: a fast and efficient music source separation tool with pre-trained models.* JOSS 5(50):2154. DOI: https://doi.org/10.21105/joss.02154 (MIT). Letto: **metadati**. 4 stem circa 100× più veloce del tempo reale su GPU.
- Rouard, Massa, Défossez (2023). *Hybrid Transformers for Music Source Separation.* ICASSP 2023. DOI: https://doi.org/10.1109/ICASSP49357.2023.10096956 ; https://arxiv.org/abs/2211.08553. Letto: **abstract**. +0,45 dB SDR rispetto a Hybrid Demucs con 800 brani in più di addestramento. È la famiglia Demucs già usata nel progetto per gli stem vocali (e da MosaikBox).
- Pratica commerciale [P]: Serato Stems in Serato DJ Pro/Lite 3.0 (ottobre 2022; MusicTech 05/10/2022); VirtualDJ e djay hanno introdotto la separazione nel 2020 (stessa fonte). rekordbox non verificato.
- **Implica (media)**: lo scambio di stem (acapella di A sopra strumentale di B, o mute della voce di A) è pratica DJ corrente, non un artificio di laboratorio. MosaikBox (già in biblioteca) è l'unico studio con test d'ascolto sull'effetto della rimozione di stem in un mix automatico.

**34. Percezione di due voci simultanee.**
- Bregman, Albert S. (1990). *Auditory Scene Analysis: The Perceptual Organization of Sound.* MIT Press (paperback 1994). Letto: **metadati** (parte del libro online sul sito del laboratorio McGill). [T] segregazione in flussi per differenze di frequenza, F0, timbro, posizione.
- Brungart, Douglas S. (2001). *Informational and energetic masking effects in the perception of two simultaneous talkers.* JASA 109(3):1101–1109. DOI: https://doi.org/10.1121/1.1345696. Letto: **abstract**. [E] Con due parlatori **domina il mascheramento informativo**; intelligibilità migliore con voci di sesso diverso, peggiore con lo stesso parlatore; l'intelligibilità si appiattisce sotto 0 dB di SNR e una differenza di livello può persino aiutare.
- Brungart, Simpson, Ericson, Scott (2001). *Informational and energetic masking effects in the perception of multiple simultaneous talkers.* JASA 110(5):2527–2538. DOI: https://doi.org/10.1121/1.1408946. Letto: **abstract**. Estende a più parlatori.
- Bürgel, Picinali, Siedenburg (2021). *Listening in the Mix: Lead Vocals Robustly Attract Auditory Attention in Popular Music.* Frontiers in Psychology 12:769663. DOI: https://doi.org/10.3389/fpsyg.2021.769663. Letto: **abstract**. [E] tre esperimenti online: la voce principale è il target rilevato meglio nei mix pop, anche dopo filtraggi e cambi di livello.
- Condit-Schultz, Nathaniel & Huron, David (2015). *Catching the Lyrics: Intelligibility in Twelve Song Genres.* Music Perception 32(5):470–483. DOI: https://doi.org/10.1525/mp.2015.32.5.470. Letto: **abstract**. [E] 30 partecipanti, 120 estratti, 12 generi incluso il rap; intelligibilità dal 48% (classica) al 96% (jazz), media 72%. Il valore del rap non è stato verificato.
- **Implica (alta)**: due voci rap (spesso maschili, registro simile) sovrapposte sono il caso peggiore previsto dal mascheramento informativo [E sul parlato, H sul rap]. Abbassare di qualche dB la voce uscente può non bastare: il criterio sicuro per i blend resta "una sola voce alla volta" (mute dello stem, o cambio a fine frase), coerente con Vande Veire e MosaikBox. Per i mashup (d), la voce di A sopra lo strumentale di B è il formato a minor rischio.

---

## 5. Modelli di preferenza da giudizi a coppie e piccoli dati

**35. Modelli classici.**
- Thurstone, L. L. (1927). *A law of comparative judgment.* Psychological Review 34(4):273–286. DOI: https://doi.org/10.1037/h0070288. [T] modello a utilità latente con rumore gaussiano.
- Bradley, R. A. & Terry, M. E. (1952). *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons.* Biometrika 39(3–4):324–345. DOI: https://doi.org/10.1093/biomet/39.3-4.324. [T] P(i > j) = πᵢ/(πᵢ + πⱼ), equivalente logistico di Thurstone.
- Letto: **metadati** (Crossref). Sono la base standard; nessun numero da riportare.

**36. Chu, Wei & Ghahramani, Zoubin (2005).** *Preference Learning with Gaussian Processes.* ICML 2005, pp. 137–144. DOI: https://doi.org/10.1145/1102351.1102369 ; https://mlg.eng.cam.ac.uk/zoubin/papers/icml05chuwei-pl.pdf. Letto: **abstract**. [T]/[S] GP su utilità latente con verosimiglianza per coppie: generalizza a item mai confrontati tramite le feature.
- **Implica (media)**: con feature delle transizioni (tipo, durata, voce, sezione, Δtempo, ΔCamelot, ΔLUFS) un GP di preferenza permette di estendere pochi confronti a transizioni simili, con incertezza esplicita.

**37. Houlsby, Huszár, Ghahramani, Lengyel (2011).** *Bayesian Active Learning for Classification and Preference Learning.* arXiv:1112.5745. Letto: **abstract**. [S] criterio BALD (massimo disaccordo dei parametri), esteso alla preference learning con GP.

**38. González, Dai, Damianou, Lawrence (2017).** *Preferential Bayesian Optimization.* ICML 2017, PMLR 70:1282–1291. https://arxiv.org/abs/1704.03651. Letto: **abstract**. [S] ottimizzazione di una funzione latente interrogabile solo tramite "duelli" a coppie; modellare le correlazioni tra alternative riduce molto i confronti necessari.
- **Implica (media)**: adatta a tarare pochi parametri continui di un tipo di transizione (lunghezza dell'echo, frequenza finale del filtro, battute di blend) con duelli A/B sulla stessa coppia di brani.

**39. Chen, Bennett, Collins-Thompson, Horvitz (2013).** *Pairwise Ranking Aggregation in a Crowdsourced Setting.* WSDM 2013, pp. 193–202. DOI: https://doi.org/10.1145/2433396.2433420. Letto: **parziale**. [S]+[E] Crowd-BT: Bradley–Terry con affidabilità per annotatore e scelta attiva di coppia e annotatore; sui loro dati reali "90% of the best gold-standard accuracy with only 3% of the total labeling cost". Numero specifico del loro dataset, non trasferibile.

**40. Mantiuk, Tomaszewska, Mantiuk (2012).** *Comparison of Four Subjective Methods for Image Quality Assessment.* Computer Graphics Forum 31(8):2478–2491. DOI: https://doi.org/10.1111/j.1467-8659.2012.03188.x. Letto: **abstract e metodo**. [E] confronto tra single stimulus, double stimulus, confronto a coppie a scelta forzata e giudizi di similarità: **il confronto a coppie forzato ha la varianza minore** ed è il più efficiente in tempo con un numero moderato di condizioni. Dominio immagini.

**41. Pérez-Ortiz, Mikhailiuk, Zerman, Hulusic, Valenzise, Mantiuk (2020).** *From Pairwise Comparisons and Rating to a Unified Quality Scale.* IEEE TIP 29:1139–1151. DOI: https://doi.org/10.1109/TIP.2019.2936103. Letto: **abstract e pagina progetto**. [S]+[E] unifica rating e confronti su una scala JOD: **1 JOD = 75% degli osservatori preferisce una condizione**. Codice pubblico (pwcmp).
- **Implica (alta)**: permette di usare insieme i voti 1–10 già raccolti e futuri confronti A/B senza buttare i primi, tenendo i 6 giudizi attuali come insieme diagnostico separato.

**42. Protocolli d'ascolto.**
- ITU-R BS.1534-3 (10/2015). *Method for the subjective assessment of intermediate quality level of audio systems* (MUSHRA). Pagina ITU attesa https://www.itu.int/rec/R-REC-BS.1534 (non aperta). Letto: **fonti secondarie** (Wikipedia, AudioLabs Erlangen, FORCE Technology; testo ITU non scaricato). In vigore la revisione -3 approvata a ottobre 2015. [T] più versioni sulla stessa schermata, scala 0–100, riferimento nascosto e ancore degradate (passa-basso) per calibrare e scartare ascoltatori inaffidabili; per difetti piccoli si usa BS.1116-3.
- Schoeffler, Bartoschek, Stöter, Roess, Westphal, Edler, Herre (2018). *webMUSHRA — A Comprehensive Framework for Web-based Listening Tests.* J. Open Research Software 6(1):8. DOI: https://doi.org/10.5334/jors.187 (CC BY). Letto: **abstract**. Supporta MUSHRA, BS.1116 e scelta forzata nel browser.
- **Implica (alta)**: MUSHRA presuppone un riferimento "corretto", che per una transizione non esiste. Si può usare una variante "MUSHRA-like" senza riferimento nascosto ma con un'ancora (es. cut fuori battuta, già noto come difettoso) e con la transizione fatta a mano da un DJ come termine di confronto. È una deviazione dallo standard da dichiarare [H]. Con un solo ascoltatore il confronto a coppie forzato è più efficiente (Mantiuk 2012).

**43. Cawley, Gavin C. & Talbot, Nicola L. C. (2010).** *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.* JMLR 11:2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html. Letto: **abstract**. [T]+[E] un criterio di selezione con alta varianza (come la cross-validation su pochi dati) produce overfitting in fase di selezione; la perdita è dell'ordine delle differenze tra algoritmi.
- **Implica (alta)**: con 6 clip ogni scelta di soglia o di peso guidata dai voti è selezione su dati di test. Serve un insieme di ascolto tenuto da parte prima di guardare i risultati, come già chiede `AGENTS.md`.

---

## 6. Psicologia dell'aspettativa (solo diagnostica)

**44. Huron, David (2006).** *Sweet Anticipation: Music and the Psychology of Expectation.* Cambridge, MA: MIT Press. ISBN 978-0-262-08345-4.
- Letto: **descrizione editoriale e recensioni** (Aversa, *Music Theory Online* 15(3–4), 2009; Zilberg, *Leonardo Reviews* 2007; Stewart, *Trends in Cognitive Sciences* 2007, solo citazione). Indice dei capitoli non verificato.
- Tipo: [T]. Teoria ITPRA: cinque risposte (imagination, tension, prediction, reaction, appraisal); l'incertezza prima di un evento produce tensione; dispositivi come sincope, cadenza, metro, tonalità e climax sfruttano queste risposte. Le recensioni mettono in dubbio l'universalità della teoria (Zilberg).
- **Implica (bassa, diagnostica)**: offre un lessico per descrivere perché un ingresso preparato può funzionare meglio di uno improvviso; non fornisce soglie né criteri di selezione.

**45. Solberg, Ragnhild Torvanger & Jensenius, Alexander Refsum (2016/2017).** *Pleasurable and Intersubjectively Embodied Experiences of Electronic Dance Music.* Empirical Musicology Review 11(3–4):301–318. DOI: https://doi.org/10.18061/emr.v11i3-4.5023 (Crossref: 2017; numero datato 2016).
- Letto: **abstract e commento di Greasley**. Tipo: [E] motion capture di 16 ballerini su un mix di 15 minuti con 4 brani house, 2 con break routine e 2 di controllo. La quantità di moto cala nel breakdown e cresce in modo significativo al drop; nei brani di controllo varia poco; il brano con il breakdown (*Icarus*) è il più gradito.

**46. Solberg, R. T. & Dibben, Nicola (2019).** *Peak Experiences with Electronic Dance Music: Subjective Experiences, Physiological Responses, and Musical Characteristics of the Break Routine.* Music Perception 36(4):371–389.
- DOI: https://doi.org/10.1525/mp.2019.36.4.371 ; copia White Rose https://eprints.whiterose.ac.uk/145911/. Letto: **full**.
- Tipo: [E] 24 partecipanti, 4 estratti EDM con break routine e 1 senza; conduttanza cutanea, affetto, sensazioni corporee. **Effetto principale della sezione sull'ampiezza SCR (p < 0,01): drop > breakdown (p < 0,05) e drop > build-up (p < 0,01)**, nessuna differenza tra breakdown e build-up; nessuna differenza nel numero di risposte per sezione; nessuna correlazione con la familiarità. I partecipanti attribuiscono le esperienze di picco soprattutto al drop.
- Solberg, R. T. (2014). *"Waiting for the bass to drop": Correlations between intense emotional experiences and production techniques in build-up and drop sections of electronic dance music.* Dancecult 6(1):61–82. Verificato solo come voce bibliografica in Solberg & Dibben 2019; non letto.

**47. Yadati, Larson, Liem, Hanjalic (2014).** *Detecting Drops in Electronic Dance Music: Content based approaches to a socially significant music event.* ISMIR 2014, pp. 143–148. https://archives.ismir.net/ismir2014/paper/000297.pdf. Letto: **abstract**. [S] rilevazione del drop in due fasi, commenti a tempo di SoundCloud come etichette deboli. **Bassa**.

**48. Agres, Herremans, Bigo, Conklin (2017).** *Harmonic Structure Predicts the Enjoyment of Uplifting Trance Music.* Frontiers in Psychology 7:1999. DOI: https://doi.org/10.3389/fpsyg.2016.01999. Letto: **abstract e partecipanti**. [E] 20 partecipanti, brani generati da un modello su 100 anthem; relazione a U rovesciata (curva di Wundt) tra complessità/ripetizione armonica e gradimento. **Bassa**.

**49. Butler, Mark J. (2006).** *Unlocking the Groove: Rhythm, Meter, and Musical Design in Electronic Dance Music.* Bloomington: Indiana University Press. Letto: **indice da catalogo**. [T] il capitolo finale tratta "form from the record to the set". **Bassa**, da leggere se si lavora sulla forma del set.

**Uso consentito.** Queste fonti sono su EDM e trance, non sul rap. Come stabilito in `AGENTS.md`, non guidano la selezione. Servono solo a formulare ipotesi da verificare con ascolti (§7).

---

## 7. I giudizi dell'utente letti con queste fonti (diagnostica, non validazione)

Giudizi disponibili (6 clip, un ascoltatore): filter out ed echo out su inizio frase 9/10; blend di 8 battute 8/10; echo out 7/10; drop diretto sul ritornello di B 3/10. Sei voti di un solo ascoltatore non permettono di stimare alcun effetto. Le righe sotto sono ipotesi [H] coerenti con la letteratura, da mettere alla prova su clip nuove e tenute da parte.

- **Drop diretto sul ritornello di B (3/10).** Due spiegazioni compatibili, non distinguibili con questi dati:
  - (i) manca la preparazione: nei dati EDM il drop è efficace dopo breakdown e build-up (Solberg & Dibben 2019; Solberg & Jensenius 2016/2017);
  - (ii) il ritornello di B porta voce piena e densa che entra mentre A ha ancora contenuto: costo vocale e loudness alto (Bittner 2017), mascheramento informativo (Brungart 2001), attrazione dell'attenzione della voce (Bürgel 2021).

  Test discriminante proposto: lo stesso ingresso sul ritornello preceduto da un filter/echo out di A di 1–2 battute, contro l'ingresso diretto; e lo stesso punto di B con lo stem vocale di B ritardato di una battuta.
- **Filter out ed echo out su inizio frase (9/10).** Togliere bassi e alti ad A, o lasciare una coda d'eco, svuota la texture prima dell'ingresso di B. È analogo a un mini-breakdown che prepara il downbeat di B [H]; inoltre elimina la sovrapposizione vocale e non richiede beatmatch lungo, quindi regge differenze di tempo maggiori. Sono questi i tipi da usare come default quando tempo, tonalità o voci impediscono il blend.
- **Blend di 8 battute (8/10).** Coerente con la pratica e con i blocchi da 32 beat (Kim 2020), purché senza voci sovrapposte.
- **Echo out 7/10 contro 9/10.** La differenza tra i due echo out (7 contro 9) suggerisce che il contesto (punto di frase, contenuto vocale, densità di B) pesa quanto il tipo, coerente con Davies 2014 e Senn 2018 (variabilità forte tra casi e ascoltatori).

---

## 8. Cosa implica per i passi (a)–(d)

### (a) Scelta del tipo di transizione — priorità alta
1. Albero di decisione esplicito e leggibile (sul modello di Vande Veire e Thalmann), con ingressi misurati: rapporto di tempo dopo la piegatura d'ottava, distanza Camelot, voce attiva in A e in B nelle battute di sovrapposizione (stem Demucs), etichetta di sezione `allin1` all'ingresso di B, Δ energia (descrittori del §3). Regole iniziali [H]:
   - tempo oltre ±8% o voce di B attiva all'ingresso → filter out o echo out di A su fine frase, B sul downbeat a inizio frase;
   - tempo e tonalità compatibili e almeno una delle due tracce senza voce per 8 battute → blend di 8 battute con bass swap;
   - ingresso di B su ritornello o drop → solo dopo una preparazione su A (filter/echo di 1–2 battute), mai diretto.
2. Matrice di costo alla Bittner (timbro, chroma, loudness bassa, vocalness congiunta, fine su confine) per scegliere i punti dentro ogni tipo.
3. Varietà dei tipi lungo un set come vincolo morbido [H], da verificare con ascolto di set interi.

### (b) Set continui di 10–15 brani — priorità alta
1. Spazio delle feature alla Bittner: log₂ BPM con equivalenza d'ottava (essenziale per il trap a 70/140), Camelot su cerchio, embedding timbrico, energia.
2. Ottimizzazione esatta del cammino (Held–Karp per n ≤ 15) con:
   - costo forte per salti di tempo oltre circa ±10% (Ishizaki 2009) o forzatura di un tipo cut/echo;
   - costo di transizione dipendente dal tipo scelto in (a);
   - primo e ultimo brano fissabili (Flexer 2008), curva di energia richiesta dall'utente (Cliff 2000);
   - passi regolari invece del "più simile al precedente" (Kell & Tzanetakis 2013) e alternanza locale dell'energia consentita (Neto et al. 2024).
3. Valutare il set intero con ascolto, non solo le singole transizioni: è il livello "mix" che, secondo Williams 2025, nessun sistema copre.

### (c) Modello di preferenza dai giudizi — priorità media (dati insufficienti oggi)
1. Non stimare nulla dai 6 voti: tenerli come diagnostica (AGENTS.md; Cawley & Talbot 2010).
2. Raccogliere confronti A/B forzati tra tipi diversi sulla stessa coppia e sullo stesso cue (Mantiuk 2012), salvati nel database dell'Artifact; stimare Bradley–Terry per contesto, poi un GP di preferenza sulle feature della transizione (Chu & Ghahramani 2005).
3. Scegliere le coppie da ascoltare con un criterio attivo (BALD, Houlsby 2011) e tarare i parametri continui con duelli (PBO, González 2017).
4. Unire voti 1–10 e confronti su una scala JOD (Pérez-Ortiz 2020); dichiarare un insieme di clip di test bloccato prima dell'analisi.
5. Separare il gradimento della transizione da quello dei brani (Liebman 2015).

### (d) Mashup e sovrapposizioni voce-strumentale — priorità media
1. Formato a basso rischio: voce di A (stem) sopra lo strumentale di B, una sola voce alla volta (Brungart 2001; Davies 2014; MosaikBox).
2. Compatibilità verticale (armonia della voce rispetto all'accompagnamento, ritmo, bilanciamento spettrale) separata da quella orizzontale (ingresso e uscita) (Lee 2015).
3. Stretch a tratti sulla griglia dei beat e controllo dello sfasamento su tutta la durata (Huang 2021).
4. Le finestre ±3 semitoni e 0,8–1,2 di Huang 2021 sono limiti di pipeline, non soglie percettive: partire dai nostri limiti più stretti (±8%, Camelot ±1) e allargarli solo dopo ascolto.

---

## 9. Limiti di questa revisione

- **Lette per intero**: Bittner 2017, Kell & Tzanetakis 2013, Ishizaki 2009, Liebman 2015, Davies 2014, Huang 2021, Williams 2025, Solberg & Dibben 2019, DJ-AI 2025. Le altre solo come abstract, metadati, recensioni o parti del testo, come indicato voce per voce.
- **Non lette**: Katz 2012 e Broughton & Brewster (solo recensioni o cataloghi); Cliff 2000 (server HP non raggiungibile); Solberg 2014; Huron 2006 (solo descrizione e recensioni); il testo ITU-R BS.1534-3; Xing 2020; Hawkins 2013.
- **Correzioni rispetto alla richiesta**: Gates et al. 2006 è in DIS '06, non in CHI. Brungart 2001 "two simultaneous talkers" è JASA 109(3), distinto da Brungart et al. 2001 "multiple simultaneous talkers", JASA 110(5). Solberg & Jensenius è datato 2016 nel numero e 2017 in Crossref. Kell 2013 ha come coautore George Tzanetakis.
- **Lacune confermate**: nessuno studio classifica i tipi di transizione in mix reali; nessuno misura le curve di energia su molti set DJ; nessuno collega groove e transizioni; nessuno è sul rap.
- **Copie locali**: i PDF ad accesso aperto sono stati scaricati solo nella scratchpad della sessione, per la lettura, e non sono stati copiati in `originals/` né aggiunti a `SOURCES_MANIFEST.json`. Alcuni hanno diritti riservati (copie degli autori o dei repository). Prefissi SHA-256 (16 caratteri) dei file letti, per ritrovarli: Bittner 2017 `debebcb9165701c9`; Kell 2013 `8ce42479e7f1be9a`; Ishizaki 2009 `ace76fc91dfef8c6`; Liebman (arXiv 1401.1880) `a44a093d4e96d510`; Davies 2014 `9908fbef89205cb1`; Huang 2021 (arXiv v1) `56c010da13e98c3e`; Williams 2025 `65060f5c7bb1dc7b`; Solberg & Dibben 2019 `b3ca5a7b9aff8e34`; Janata 2012 (Zenodo) `3e3afdc3f7bcb530`; Witek 2014 `768a6b475c39751d`. Archiviarli nella biblioteca è una decisione da prendere caso per caso secondo la licenza.
