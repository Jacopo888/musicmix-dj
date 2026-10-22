"""Collect and validate the maintained MusicMix science library.

The collector is deliberately conservative: it copies sources that already
exist in this workspace, records their provenance and hashes, and never
downloads or silently replaces a failed fetch.  It is safe to rerun after new
source files are added to the source map below.

Usage (from the project root):
    python tools/collect_science_library.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_ROOT = PROJECT_ROOT / "conoscenza_mix"
COLLECTION_DATE = "2026-10-20"


def _source(
    source_id: str,
    section: str,
    title: str,
    authors: str,
    year: int,
    url: str,
    source_path: str | None,
    *,
    doi: str | None = None,
    license: str = "da verificare nella versione distribuita",
    access: str = "open access / copia locale già presente",
    read_depth: str,
    anchors: str,
    evidence: str,
    transfer: str,
    note: str = "",
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "section": section,
        "title": title,
        "authors": authors,
        "year": year,
        "date_collected": COLLECTION_DATE,
        "url": url,
        "doi": doi,
        "source_path": source_path,
        "license": license,
        "access": access,
        "read_depth": read_depth,
        "page_or_section_anchors": anchors,
        "strongest_evidence": evidence,
        "transfer_limit": transfer,
        "note": note,
    }


# Every path below is relative to PROJECT_ROOT.  The map intentionally
# includes both papers and official dataset/repository artifacts.
SOURCES: list[dict[str, Any]] = [
    _source(
        "farbood2012",
        "01_psicologia_aspettativa",
        "A Parametric, Temporal Model of Musical Tension",
        "Morwaread M. Farbood",
        2012,
        "https://doi.org/10.1525/mp.2012.29.4.387",
        "runs/swarm_v9_research/agents/psychology_when/sources/farbood2012_tension.pdf",
        doi="10.1525/mp.2012.29.4.387",
        license="publisher article; local research copy; redistribution rights not audited",
        access="local PDF; DOI landing page",
        read_depth="full text archived/extracted; sections read: abstract/model framing and temporal model/limitations (PDF pp. 1-3, 33-35)",
        anchors="PDF pp. 1-3 (abstract/model framing), printed pp. 420-421 / PDF pp. 33-35 (temporal model and limitations)",
        evidence="La tensione è un fenomeno temporale e multidimensionale: il modello combina parametri musicali con trend e memoria, e la lettura fuori contesto può perdere la dinamica osservata.",
        transfer="Stimoli soprattutto tonali/classici; non trasferire pesi o soglie direttamente a rap/trap.",
    ),
    _source(
        "lehne_koelsch2015",
        "01_psicologia_aspettativa",
        "Toward a general psychological model of tension and suspense",
        "Moritz Lehne; Stefan Koelsch",
        2015,
        "https://doi.org/10.3389/fpsyg.2015.00079",
        "runs/science_v11/agents/psychology_library/research/01_psicologia_aspettativa/originals/lehne_koelsch_2015.xml",
        doi="10.3389/fpsyg.2015.00079",
        license="Frontiers open access; PMC XML archive",
        access="local JATS XML from PMC; companion 519-byte PDF fetch is retained as failed",
        read_depth="full XML archived/extracted; sections read: abstract, predictive processes, conflict/uncertainty, domain-independent model",
        anchors="front/abstract; sections on predictive processes, conflict/uncertainty, neural and domain-independent model",
        evidence="Propone che tensione e suspense nascano da conflitto, instabilità, dissonanza o incertezza che attivano processi predittivi orientati a eventi futuri.",
        transfer="È un quadro teorico generale, non una validazione di un confine vocale o di un cambio A→B.",
        note="La copia PDF collegata al tentativo v11 era HTML troncato; la fonte valida conservata è l'XML JATS.",
    ),
    _source(
        "lehne2015_pdf_failed",
        "01_psicologia_aspettativa",
        "Toward a general psychological model of tension and suspense (failed PDF fetch)",
        "Moritz Lehne; Stefan Koelsch",
        2015,
        "https://doi.org/10.3389/fpsyg.2015.00079",
        "runs/science_v11/agents/psychology_library/research/01_psicologia_aspettativa/originals/lehne_koelsch_2015.pdf",
        license="unknown; fetch failed",
        access="failed fetch retained for audit",
        read_depth="not readable",
        anchors="none",
        evidence="Nessuna evidenza estratta: il file è una risposta HTML di 519 byte, non un PDF.",
        transfer="Usare l'XML JATS valido con lo stesso DOI; non usare questo binario.",
        note="Header ricevuto: <!DOCTYPE ht; il file è copiato solo nella cartella failed_fetches.",
    ),
    _source(
        "hansen2021",
        "01_psicologia_aspettativa",
        "Predictive Uncertainty Underlies Auditory Boundary Perception",
        "Niels Chr. Hansen; Haley E. Kragness; Peter Vuust; Laurel Trainor; Marcus T. Pearce",
        2021,
        "https://doi.org/10.1177/0956797621997349",
        "runs/swarm_v9_research/agents/psychology_when/sources/hansen2021_boundary.pdf",
        doi="10.1177/0956797621997349",
        license="publisher article; local research copy; redistribution rights not audited",
        access="local PDF; DOI landing page",
        read_depth="full text archived/extracted; sections read: PDF pp. 1-2 and 7 (design, discussion)",
        anchors="PDF pp. 1-2 (abstract/design), p. 7 (discussion); N=38 dwell task, N=31 completeness ratings",
        evidence="L'incertezza predittiva (entropia) influenza la percezione di confine/completamento indipendentemente dalla sorpresa retrospettiva, in sequenze melodiche controllate.",
        transfer="Le sequenze sono melodiche e pitch-based; non dimostra che un gap vocale o un downbeat risolva un mix.",
    ),
    _source(
        "turrell2025",
        "01_psicologia_aspettativa",
        "Building the anticipation: How variation in tension mediates emotions in music",
        "Amelia S. Turrell; Andrea R. Halpern; Katy Bannister; Dawn Chai-Wi-Ting; Amir-Homayoun Javadi",
        2025,
        "https://doi.org/10.1525/mp.2024.aa004",
        "runs/science_v11/agents/psychology_library/research/01_psicologia_aspettativa/originals/turrell_2025_anticipation.pdf",
        doi="10.1525/mp.2024.aa004",
        license="author accepted manuscript; repository access, redistribution terms to verify",
        access="Kent Academic Repository local PDF (31 pages)",
        read_depth="full text archived/extracted; sections read: PDF pp. 1-2 and break-routine methods/results",
        anchors="PDF pp. 1-2 (abstract), break-routine continuous ratings; 34 participants; build-up/drop comparison",
        evidence="In break routine EDM, la tensione cresce durante il build-up e cala dopo il drop atteso; il calo è coerente con evento atteso nel compito specifico.",
        transfer="EDM e drop altamente attesi; non prova che ogni calo RMS o ogni cambio traccia sia chiusura musicale.",
    ),
    _source(
        "grosz2025_failed",
        "01_psicologia_aspettativa",
        "An outline of the narrative grammar of electronic dance music",
        "Patrick Georg Grosz; Ragnhild Torvanger Solberg; Jonah Katz; Mai Ha Vu; Alexander Refsum Jensenius; Pritty Patel-Grosz",
        2025,
        "https://doi.org/10.1177/10298649251321709",
        "runs/science_v11/agents/psychology_library/research/01_psicologia_aspettativa/originals/grosz_2025_edm_grammar.pdf",
        doi="10.1177/10298649251321709",
        license="CC BY 4.0 per la versione SAGE; la copia eScholarship indica CC BY-NC; verificare la versione prima di redistribuire",
        access="DOI/SAGE e record eScholarship accessibili; PDF locale non recuperato (403 CloudFront, 919 byte)",
        read_depth="abstract and metadata only; PDF non leggibile",
        anchors="SAGE abstract/metadata; eScholarship record qt2138v3n3; online-first 2025, Musicae Scientiae 29(4):556-575",
        evidence="L'abstract propone una grammatica narrativa EDM: Intro/Breakdown→Buildup→Core→Outro/Cut come analogo Establisher–Initial–Peak–Release, con meccanismi cognitivi di tensione/rilascio.",
        transfer="È un'analisi teorica di EDM; non è un annotatore audio e non dimostra closure rap o qualità cross-song.",
        note="Il binario ricevuto resta copiato solo in failed_fetches: la validazione ha riscontrato <!DOCTYPE HTML e 919 byte. Il contenuto dell'abstract è metadata/abstract, non full text.",
    ),
    _source(
        "mcflow2016",
        "02_fraseggio_rap_voce",
        "MCFlow: A Digital Corpus of Rap Flow",
        "Nathaniel Condit-Schultz",
        2016,
        "https://rave.ohiolink.edu/etdc/view?acc_num=osu1461250949",
        "runs/swarm_v9_research/agents/hiphop_structure/sources/condit_schultz_2016_dissertation.pdf",
        license="OhioLINK ETD / academic repository; dataset access and redistribution must be checked separately",
        access="local dissertation PDF (197 pages)",
        read_depth="full text archived/extracted; sections read: Ch. 3.2.2, 3.2.4, 3.3 and 5.1.4",
        anchors="Ch. 3.2.2 pp. 59-63; Ch. 3.2.4 pp. 83-87; Ch. 3.3 pp. 87-89; Ch. 5.1.4 pp. 152-155; abstract",
        evidence="Corpus simbolico di flow: la frase è prosodica e ritmica, spesso attorno a una misura ma con pickup, sottounità di due beat e phrasing in 3/4 o 5/4; il corpus della dissertazione dichiara 124 brani, 47 artisti, 6.107 misure e 54.248 parole.",
        transfer="È rap, quindi utile per la forma vocale; non contiene audio allineato né giudizi di transizione DJ.",
    ),
    _source(
        "adams2020",
        "02_fraseggio_rap_voce",
        "Harmonic, Syntactic, and Motivic Parameters of Phrase in Hip-Hop",
        "Kyle Adams",
        2020,
        "https://mtosmt.org/issues/mto.20.26.2/mto.20.26.2.adams.html",
        "runs/swarm_v9_research/agents/psychology_when/sources/adams2020_phrase.pdf",
        license="Society for Music Theory / MTO open access",
        access="local PDF plus open HTML landing page",
        read_depth="full text archived/extracted; sections read: §§0.1-1.16, 2.1-2.24, 3.1-3.4, 4-6; PDF font artifacts remain",
        anchors="§§0.1-1.16 (norms/rhyme), §§2.1-2.24 (harmony), §§3.1-3.4 (syntax), §§4-5 (motive/interactions), §6",
        evidence="La frase hip-hop non è garantita dal default di due battute: armonia del beat, sintassi e motivo ritmico/melodico possono concordare o competere; la sintassi può prevalere sulla rima.",
        transfer="Analisi musicologica di esempi, non esperimento percettivo né corpus audio; usarlo per progettare cue multipli, non per fissare soglie.",
    ),
    _source(
        "duinker2020",
        "02_fraseggio_rap_voce",
        "Diversification and Post-Regionalism in North American Hip-Hop Flow",
        "Benjamin Duinker",
        2020,
        "https://mcgill.scholaris.ca/items/2d39727c-4429-4dfc-b273-c9294ab471e1",
        "runs/swarm_v9_research/agents/hiphop_structure/sources/duinker_dissertation.pdf",
        license="McGill institutional repository; open access thesis",
        access="local dissertation PDF (306 pages)",
        read_depth="full text archived/extracted; sections read: Ch. 6.2-6.4 (SR1-SR5, PR1-PR2, alignment/non-alignment)",
        anchors="Ch. 6.2 pp. 169-175 (SR1-SR5); Ch. 6.3 pp. 176-179 (PR1-PR2); Ch. 6.4 pp. 194-201 (alignment/non-alignment)",
        evidence="Propone cinque regole di segmentazione che possono confliggere (rima, sintassi, respiro, ritmo), distingue frasi allineate/nested/non-aligned e avverte che la dissonanza metrica non richiede sempre risoluzione.",
        transfer="Regole analitiche non percettivamente validate; trasferimento a trap e mix richiede audio annotato.",
    ),
    _source(
        "vandeveire2018",
        "03_struttura_transizioni_dj",
        "From raw audio to a seamless mix: creating an automated DJ system for Drum and Bass",
        "Len VandeVeire; Tijl De Bie",
        2018,
        "https://doi.org/10.1186/s13636-018-0134-8",
        "runs/swarm_v9_research/agents/cue_point_models/sources/vandeveire_2018.pdf",
        doi="10.1186/s13636-018-0134-8",
        license="CC BY 4.0 (publisher record)",
        access="local open-access PDF (21 pages)",
        read_depth="full text archived/extracted; sections read: abstract, methods, evaluation on main/hold-out sets",
        anchors="Abstract; methods for beat/downbeat/structure/cue/fade; evaluation on 160 + hold-out 220 DnB songs; subjective test with 18 fans",
        evidence="Un sistema end-to-end integra beat/downbeat, segmentazione, cue, crossfade e playlist; 91% delle canzoni nel corpus principale ha annotazioni complete corrette secondo il protocollo DnB.",
        transfer="DnB e proprietà EDM; durata/cue non è una legge per rap o voce in primo piano.",
    ),
    _source(
        "kim2020_djmix",
        "03_struttura_transizioni_dj",
        "Automatic DJ Mix Analysis from Online Mixes",
        "Jongpil Kim et al.",
        2020,
        "https://arxiv.org/abs/2008.10267",
        "runs/swarm_v9_research/agents/cue_point_models/sources/kim2020_djmix_alignment.pdf",
        license="ISMIR/arXiv paper; local research copy",
        access="local PDF (7 pages)",
        read_depth="full text archived/extracted; sections read: dataset and transition evaluation",
        anchors="Dataset section and transition evaluation; 1.557 mix, 13.728 tracks, 20.765 transitions; hit windows 15/30/60 s",
        evidence="L'analisi di mix reali offre target e tolleranze temporali per la ricerca di transizioni, ma descrive mix pubblicati e non un giudizio di risoluzione A→B.",
        transfer="Dataset dance/online e identificazione da fingerprint; non è annotazione di phrase closure rap.",
    ),
    _source(
        "cmmr2019",
        "03_struttura_transizioni_dj",
        "Methods and Datasets for DJ-Mix Reverse Engineering",
        "Alexandre Schwaab et al.",
        2019,
        "https://hal.science/hal-02172427",
        "runs/swarm_v9_research/agents/cue_point_models/sources/cmmr2019_djmix.pdf",
        license="paper access via HAL; UnmixDB CC tracks / Zenodo record",
        access="local conference PDF (13 pages)",
        read_depth="full text archived/extracted; sections read: five-step method, UnmixDB dataset, alignment/distortion results",
        anchors="Five-step pipeline DTW→sample alignment→track removal→volume curve→cue; UnmixDB tables; median alignment error ~25 ms, distortion case ~100 ms",
        evidence="UnmixDB permette controlli tecnici con ground truth costruito e suppression ratio, separando allineamento/clock da gusto o qualità estetica.",
        transfer="Mix sintetici e dance; il ground truth tecnico non è una preferenza professionale.",
    ),
    _source(
        "cuedetr2024",
        "04_modelli_preaddestrati",
        "Cue Point Estimation using Object Detection",
        "Giulia Argüello; Luca A. Lanzendörfer; Roger Wattenhofer",
        2024,
        "https://arxiv.org/abs/2407.06823",
        "runs/swarm_v9_research/agents/cue_point_models/sources/cue_detr_2024.pdf",
        license="CC BY 4.0 for paper; official code/checkpoint MIT per repository records",
        access="local ISMIR PDF (8 pages) and official repository",
        read_depth="full text archived/extracted; sections read: EDM-CUE dataset, method and results tables, limitations",
        anchors="Abstract; §3 EDM-CUE (4.710 tracks, 21.461 cue); §§4.3-4.5 tables/results; §5 limitations",
        evidence="DETR su spettrogramma tratta cue estimation come object detection; su EDM-CUE il best F1 riportato è 0,46 e il dataset è 4/4 EDM con cue phrase-aligned.",
        transfer="È locator single-track, non planner cross-song e non misura risoluzione; rap/hip-hop e stems simultanei fuori dominio.",
    ),
    _source(
        "zehren2020",
        "04_modelli_preaddestrati",
        "Automatic Detection of Cue Points for the Emulation of DJ Mixing",
        "Mickaël Zehren; Marco Alunno; Paolo Bientinesi",
        2020,
        "https://arxiv.org/abs/2007.08411",
        "runs/swarm_v9_research/agents/cue_point_models/sources/zehren_2020_cue.pdf",
        license="arXiv open access; repository license separate",
        access="local PDF (7 pages)",
        read_depth="full text archived/extracted; sections read: M-DJCUE, novelty features, rule algorithm, precision/recall results",
        anchors="M-DJCUE description; CQT/PCP novelty features; rule algorithm; precision/recall vs MSAF/Mixed In Key",
        evidence="Regole di novelty energia/timbro/drum/harmonia possono generare candidati economici; il paper riporta precisione circa 86% con recall limitato sul proprio EDM set.",
        transfer="EDM/4-4 e cue switch su singolo brano; non usare precisione come qualità percettiva sul pool rap.",
    ),
    _source(
        "raveform2026",
        "04_modelli_preaddestrati",
        "Raveform: A Dataset of Metrical and Functional Structure Annotations for EDM Tracks in DJ Mixes",
        "Taejun Kim; Jongsoo Kim; Hyungyu Kim; Juhan Nam",
        2026,
        "https://doi.org/10.5334/tismir.288",
        "runs/swarm_v9_research/agents/cue_point_models/sources/raveform_tismir2026.pdf",
        doi="10.5334/tismir.288",
        license="Open access TISMIR; dataset card terms must be checked per file",
        access="local TISMIR PDF (13 pages)",
        read_depth="full text archived/extracted; sections read: vocabulary/dataset, benchmark tables, cross-dataset notes",
        anchors="§§3-4 vocabulary/dataset; Table 3 (4.902 mixes, 56.873 unique tracks, 1.423 annotated); §5 metrics/cross-dataset; §6 notes",
        evidence="Raveform separa struttura metrica e funzionale in EDM dentro mix DJ e mostra degrado cross-dataset; la versione annotata è 1.423 tracce dentro 4.902 mix.",
        transfer="Vocabolario intro/buildup/drop/breakdown/cooldown/outro è EDM-centric; punti di mix allineati/derivati non sono tutti giudizi manuali.",
    ),
    _source(
        "linkseg_official",
        "04_modelli_preaddestrati",
        "LinkSeg official music structure segmentation model",
        "Morgan et al. / official repository authorship as published",
        2023,
        "https://github.com/morgan76/LinkSeg",
        None,
        license="repository/checkpoint terms must be verified at pinned commit",
        access="official repository and local V10 execution manifest; paper PDF not present in current archive",
        read_depth="official README/repository metadata plus executed V10 manifest, not paper full text",
        anchors="Pinned commit 38dc0dabed14daa5fbef7a61ea3cb046517e74a0; V10 six-track run: 86 sections / 80 internal boundaries",
        evidence="È stato eseguito con encoder/decoder ufficiali e ha prodotto confini strutturali; due render raw hanno evidenziato salti RMS circa 11,9/11,6 dB e uno seam quasi silenzioso.",
        transfer="Boundary provider ≠ transizione riuscita; il paper non è archiviato qui e i risultati V10 sono diagnostici, non ground truth.",
        note="Incluse le informazioni ufficiali disponibili; recuperare il paper primario in un turno successivo senza sostituire la provenienza del run V10.",
    ),
    _source(
        "daito_manabe2026",
        "05_renderer_loudness_tempo",
        "Interpretable diagnostics for DJ-mix quality (Daito Manabe)",
        "Daito Manabe et al. (as recorded in local research report)",
        2026,
        "local research copy; canonical URL in ricerca/v9/cue_point_models/sources.json",
        "runs/swarm_v9_research/agents/cue_point_models/sources/daito_manabe_2026_djmix_quality.pdf",
        license="preprint/practitioner copy; redistribution rights not audited",
        access="local PDF (16 pages; pypdf-validated)",
        read_depth="full text archived/extracted; sections read: deck A/deck B/master component diagnostics",
        anchors="Deck A/deck B/master component diagnostics; local report § cue-point models",
        evidence="Propone di separare loudness/headroom, continuità spettrale, smoothness del gain e stereo; non ridurre la qualità a uno scalare unico.",
        transfer="Framework diagnostico, non validazione percettiva del pool; componenti tecniche non equivalgono a chiusura semantica.",
    ),
    _source(
        "mosaikbox2024",
        "05_renderer_loudness_tempo",
        "MosaikBox: Music mixing with stem modification",
        "Sowula; Knees",
        2024,
        "local research copy; canonical URL in ricerca/v9/cue_point_models/sources.json",
        "runs/swarm_v9_research/root_sources/mosaikbox.pdf",
        license="conference paper; local research copy",
        access="local PDF (source copy)",
        read_depth="full text archived/extracted; sections read: stem modification, beat-grid/time alignment, subjective evaluation",
        anchors="stem modification, beat-grid/time alignment, subjective quality correlation reported in source report",
        evidence="La modifica degli stem e l'allineamento del beat possono aiutare il mixing, ma la qualità resta soggettiva e dipendente dalla coppia.",
        transfer="Non è un benchmark per frase rap o suspense risolta; non sostituisce controlli separati del renderer.",
    ),
    _source(
        "tensemusic2024",
        "06_valutazione_autofeedback",
        "TenseMusic: An automatic prediction model for musical tension",
        "Barchet; Rimmele; Pelofi",
        2024,
        "https://doi.org/10.1371/journal.pone.0296385",
        "runs/swarm_v9_research/agents/benchmark_datasets/sources/tenseMusic_plosone2024.pdf",
        doi="10.1371/journal.pone.0296385",
        license="PLOS open access (publisher record)",
        access="local PLOS PDF (18 pages; pypdf-validated)",
        read_depth="full text archived/extracted; sections read: pp. 1-5 model/features and pp. 12-16 results, limitations, conclusion",
        anchors="Automatic feature extraction; slope estimation; 38 classical excerpts; public Python/OSF artifacts",
        evidence="Un modello audio-input usa sei feature, trend e memoria per stimare tensione su estratti classici; una curva in calo non certifica una risoluzione di mix.",
        transfer="Classica e singolo estratto; nessun cross-song, nessuna voce rap, nessun giudizio DJ.",
    ),
    _source(
        "songeval2025",
        "06_valutazione_autofeedback",
        "SongEval: A Benchmark for Evaluating Generated Music",
        "authors as published in arXiv record",
        2025,
        "https://arxiv.org/abs/2505.10793",
        "runs/swarm_v9_research/agents/automatic_critic_models/sources/songeval.pdf",
        license="arXiv open access; dataset/model terms separate",
        access="local PDF (13 pages; pypdf-validated) and extracted root text",
        read_depth="full text archived/extracted; sections read: dataset, five expert-rated dimensions, limitations",
        anchors="Dataset and five expert-rated dimensions; local V10 diagnostic run; limitations on whole-song vs transition quality",
        evidence="Benchmark di canzoni generate con dimensioni estetiche separate, utile come rubrica di coerenza/fraseggio ma non come filtro automatico di un singolo raccordo.",
        transfer="Whole-song ratings ≠ A→B transition labels; evitare leakage e non addestrare sulle valutazioni V8/V10.",
    ),
    _source(
        "mdjcue_archive",
        "07_dati_benchmark",
        "M-DJCUE annotated cue-point repository",
        "Zehren; Alunno; Bientinesi",
        2019,
        "https://github.com/MZehren/M-DJCUE",
        "runs/science_v11/agents/benchmark_data/research/07_dati_benchmark/originals/mdjcue_repo_complete.tar.gz",
        license="repository GPL-3.0 for code/artifacts; audio recordings are copyrighted and absent",
        access="official GitHub tree archive plus locally parsed JAMS annotations",
        read_depth="README/LICENSE full; all 134 JAMS parsed",
        anchors="README annotation guidelines 1-6; JAMS namespace cue_point; local counts in 07_dati_benchmark/RIASSUNTO.md",
        evidence="Complete official tree archive: 134 `.jams` files, 402 annotator records, 1.849 raw events (1.037 IN/812 OUT), 590 positive durations. Merging ±0,5 s gives 1.082 clusters with agreement sizes 1:547, 2:305, 3:230.",
        transfer="EDM-only; IN restricted to intro and OUT not before core by annotation design; agreement/timing labels are useful, not semantic closure rap.",
        note="Revision 7fd692c has all 134 annotation files; the prior 634121-byte ZIP remains archived as mdjcue_partial_archive and is explicitly incomplete.",
    ),
    _source(
        "mdjcue_partial_archive",
        "07_dati_benchmark",
        "M-DJCUE earlier partial archive (preserved failed attempt)",
        "Zehren; Alunno; Bientinesi",
        2019,
        "https://github.com/MZehren/M-DJCUE",
        "runs/science_v11/agents/benchmark_data/research/07_dati_benchmark/originals/mdjcue_repo.zip",
        license="repository GPL-3.0 for code/artifacts; audio recordings are copyrighted and absent",
        access="local partial archive retained for provenance audit",
        read_depth="README/LICENSE and 133 recoverable JAMS; archive integrity failure recorded",
        anchors="old ZIP header and tar warning; superseded by official complete tree archive",
        evidence="The old 634121-byte ZIP has no EOCD and recovers only 133 JAMS; it must not be used as the accepted benchmark record.",
        transfer="No benchmark claims from this partial file; use complete revision 7fd692c instead.",
        note="Preserved to show the failed retrieval, with sha256 80d28e14460128c0ae8388a737572661cad1dc06f474ebde305fef5b1fab0546.",
    ),
    _source(
        "raveform_card",
        "07_dati_benchmark",
        "Raveform official dataset card / license note",
        "Raveform maintainers",
        2026,
        "https://mir-aidj.github.io/raveform/",
        "runs/science_v11/agents/benchmark_data/research/07_dati_benchmark/originals/raveform_hf_README.md",
        license="CC BY 4.0 stated in local card front matter; verify individual data files",
        access="local 30-byte card fragment; full dataset not downloaded",
        read_depth="metadata only",
        anchors="local README front matter; full paper in 04_modelli_preaddestrati",
        evidence="La card locale conferma il campo di licenza, mentre conteggi e limiti sono presi dal paper Raveform archiviato.",
        transfer="Non usare la card da sola per conteggi o label.",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_pdf(path: Path) -> tuple[str, int | None, str | None]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages: list[str] = []
        for idx, page in enumerate(reader.pages, start=1):
            try:
                pages.append(f"\n\n--- PDF page {idx} ---\n{page.extract_text() or ''}")
            except Exception as exc:  # keep the source even if one page is damaged
                pages.append(f"\n\n--- PDF page {idx}; extraction error: {exc!r} ---\n")
        return "".join(pages), len(reader.pages), None
    except Exception as exc:
        return "", None, repr(exc)


def extract_xml(path: Path) -> tuple[str, int | None, str | None]:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "xml")
        return "\n".join(soup.stripped_strings), None, None
    except Exception as exc:
        return "", None, repr(exc)


def extract_text(path: Path) -> tuple[str, int | None, str | None]:
    if path.suffix.lower() == ".pdf":
        return extract_pdf(path)
    if path.suffix.lower() == ".xml":
        return extract_xml(path)
    # Archives are preserved byte-for-byte and audited with their own parser;
    # decoding compressed bytes as a giant text artifact would create a false
    # "paper text" and make provenance harder to inspect.
    if path.suffix.lower() in {".zip", ".gz", ".tgz", ".tar"}:
        if path.name == "mdjcue_repo_complete.tar.gz":
            try:
                with tarfile.open(path, "r:gz") as archive:
                    chunks: list[str] = []
                    for member_name in ("M-DJCUE-master/README.md", "M-DJCUE-master/LICENSE"):
                        member = archive.extractfile(member_name)
                        if member is not None:
                            chunks.append(f"--- {member_name} ---\n{member.read().decode('utf-8', errors='replace')}")
                    return "\n\n".join(chunks), None, None
            except Exception as exc:
                return "", None, repr(exc)
        return "", None, None
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None, None
    except Exception as exc:
        return "", None, repr(exc)


def validate_source(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    first = raw[:16]
    result: dict[str, Any] = {
        "size_bytes": len(raw),
        "sha256": sha256(path),
        "header_hex": first.hex(),
        "header_ascii": first.decode("ascii", errors="replace"),
        "valid_pdf_header": path.suffix.lower() != ".pdf" or raw.startswith(b"%PDF-"),
    }
    text, pages, parse_error = extract_text(path)
    result.update({"pages": pages, "text_chars": len(text), "parse_error": parse_error})
    if path.suffix.lower() == ".pdf" and not result["valid_pdf_header"]:
        result["status"] = "failed_fetch_invalid_header"
    elif parse_error:
        result["status"] = "parse_error"
    else:
        result["status"] = "valid"
    return result, text


def main() -> int:
    (LIBRARY_ROOT / "_meta").mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    copied = 0
    failed = 0
    for item in SOURCES:
        source_path = PROJECT_ROOT / item["source_path"] if item["source_path"] else None
        record = dict(item)
        if source_path is None:
            record.update({"archived_path": None, "validation": {"status": "metadata_only"}})
            manifest.append(record)
            continue
        if not source_path.exists():
            record.update({"archived_path": None, "validation": {"status": "missing_source"}})
            manifest.append(record)
            failed += 1
            continue

        validation, text = validate_source(source_path)
        is_partial_archive = item["source_id"] == "mdjcue_partial_archive"
        if is_partial_archive:
            validation["status"] = "partial_archive"
        is_failed = validation["status"] == "failed_fetch_invalid_header" or is_partial_archive
        folder = LIBRARY_ROOT / item["section"] / ("failed_fetches" if is_failed else "originals")
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / source_path.name
        shutil.copy2(source_path, destination)
        record["archived_path"] = destination.relative_to(PROJECT_ROOT).as_posix()
        record["validation"] = validation
        if text:
            text_folder = LIBRARY_ROOT / item["section"] / "texts"
            text_folder.mkdir(parents=True, exist_ok=True)
            text_destination = text_folder / f"{item['source_id']}.txt"
            text_destination.write_text(text, encoding="utf-8")
            record["extracted_text_path"] = text_destination.relative_to(PROJECT_ROOT).as_posix()
        else:
            record["extracted_text_path"] = None
        manifest.append(record)
        copied += 1
        failed += int(is_failed or bool(validation.get("parse_error")))

    output = {
        "schema": "musicmix.science_library.manifest/v1",
        "collection_date": COLLECTION_DATE,
        "generated_by": "tools/collect_science_library.py",
        "source_count": len(manifest),
        "copied_count": copied,
        "failed_or_parse_warning_count": failed,
        "sources": manifest,
    }
    manifest_path = LIBRARY_ROOT / "_meta" / "SOURCES_MANIFEST.json"
    manifest_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for section in sorted({record["section"] for record in manifest}):
        section_records = [record for record in manifest if record["section"] == section]
        section_path = LIBRARY_ROOT / section / "SOURCES.json"
        section_path.parent.mkdir(parents=True, exist_ok=True)
        section_path.write_text(
            json.dumps(
                {
                    "schema": "musicmix.science_library.section_sources/v1",
                    "section": section,
                    "collection_date": COLLECTION_DATE,
                    "sources": section_records,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({k: output[k] for k in output if k != "sources"}, ensure_ascii=False))
    for record in manifest:
        validation = record["validation"]
        print(f"{record['source_id']}: {validation.get('status')} {validation.get('size_bytes', '')} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
