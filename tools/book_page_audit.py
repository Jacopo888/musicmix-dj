"""Create a page-addressable private corpus for the nine supplied music books.

The input books live beside the worktree in ``libri/``.  Extracted text and
render/OCR scratch files default to the sibling directory
``.book_audit_private_20261021`` and are deliberately outside git.  PDFs are
kept page-addressable: ``books/<slug>/pages/0001.txt`` is physical PDF page 1
and each line of ``pages.jsonl`` records its text, images, scan/OCR status and
printed-page label.  EPUBs are kept in spine order; their records use stable
spine/chapter IDs and character spans, never fabricated page numbers.

Examples (PowerShell):

    python tools/book_page_audit.py prepare
    python tools/book_page_audit.py search "downbeat|phrase" --book Ohriner
    python tools/book_page_audit.py manifest

The extractor uses PyMuPDF when available (the launcher may install it into
the private runtime directory).  It falls back to pypdf for text/page counts.
Image-only pages are never considered read merely because extraction returned
an empty string.  If Tesseract is available, OCR is attempted automatically;
otherwise the manifest reports the unavailable OCR status explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET


WORKTREE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKTREE.parent
SOURCE_DIR = Path(os.environ.get("BOOK_AUDIT_SOURCE", str(PROJECT_ROOT / "libri")))
PRIVATE_ROOT = Path(
    os.environ.get("BOOK_AUDIT_PRIVATE", str(PROJECT_ROOT / ".book_audit_private_20261021"))
)
# ``runtime_full`` is preferred because PyMuPDF 2.x may be installed as a
# namespace-only binary bundle when pip is interrupted; the launcher keeps a
# complete 1.x fallback beside it.
PRIVATE_RUNTIME = PRIVATE_ROOT / "runtime_full"
OCR_SCRATCH = PRIVATE_ROOT / "ocr_scratch"
DEFAULT_CHUNK_SIZE = 12000

BOOKS = [
    # Author/stable-slug mapping supplied by the user: Brewster & Broughton
    # wrote the EPUB; Webber is the DJ Skills PDF.
    ("Brewster", "How to DJ right the art and science of playing records.epub"),
    ("Ohriner", "Flow The Rhythmic Voice in Rap Music.pdf"),
    ("Katz", "Groove Music The Art and Culture of the Hip-Hop DJ.pdf"),
    ("Webber", "DJ Skills The Essential Guide to Mixing and Scratching.pdf"),
    ("Margulis", "On Repeat How Music Plays the Mind.pdf"),
    ("Krims", "Rap music and the poetics of identity.pdf"),
    ("Williams", "Rhymin and stealin musical borrowing in hip-hop.pdf"),
    ("Huron", "Sweet Anticipation Music and the Psychology of Expectation.pdf"),
    ("Butler", "Unlocking the Groove Rhythm, Meter, and Musical Design in Electronic Dance Music.pdf"),
]
BOOK_BY_FILE = {name: slug for slug, name in BOOKS}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compact_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value.replace("\x00", ""))
    return re.sub(r"[ \t\r\f\v]+", " ", value).strip()


def safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower()
    return value or "book"


def load_fitz():
    """Load fitz from the private runtime before consulting global packages."""
    if PRIVATE_RUNTIME.exists():
        sys.path.insert(0, str(PRIVATE_RUNTIME))
    try:
        import fitz  # type: ignore
        if not callable(getattr(fitz, "open", None)):
            raise ImportError("fitz namespace has no compatible open API")
        return fitz
    except Exception:
        # PyMuPDF 2.x stopped shipping the deprecated ``fitz`` alias.  Keep
        # the local variable name for the compatible open/load_page API.
        try:
            import pymupdf  # type: ignore
            if not callable(getattr(pymupdf, "open", None)):
                raise ImportError("pymupdf namespace has no compatible open API")
            return pymupdf
        except Exception:
            return None


def load_pypdf():
    try:
        from pypdf import PdfReader  # type: ignore

        return PdfReader
    except Exception:
        return None


def command_path(name: str) -> str | None:
    explicit = os.environ.get(name.upper().replace("-", "_"))
    if explicit and Path(explicit).exists():
        return explicit
    return shutil.which(name)


def tesseract_path() -> str | None:
    value = os.environ.get("TESSERACT_CMD")
    candidates = [
        value,
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return None


def page_labels_with_pypdf(path: Path, page_count: int) -> list[str | None]:
    """Return one label per physical page, retaining null when unavailable."""
    Reader = load_pypdf()
    if Reader is None:
        return [None] * page_count
    try:
        reader = Reader(str(path), strict=False)
        labels = list(getattr(reader, "page_labels", []))
        if len(labels) == page_count:
            return [str(label) if label not in (None, "") else None for label in labels]
    except Exception:
        pass
    return [None] * page_count


def detect_footer_label(text: str) -> str | None:
    """Conservative fallback for printed page labels in extracted text.

    Only accept a solitary short token at the first/last line.  This avoids
    treating an arbitrary number in prose as the printed page number.
    """
    lines = [compact_text(line) for line in text.splitlines() if compact_text(line)]
    for line in (lines[:2] + lines[-2:]):
        token = line.strip("[](){}<>.,;:")
        if re.fullmatch(r"(?:[ivxlcdm]{1,8}|\d{1,5})", token, re.I):
            return token
    return None


def classify_pdf_page(text: str, physical_index: int, page_count: int, image_only: bool) -> str:
    """Best-effort navigation class; it is a locator aid, not semantic proof."""
    stripped = compact_text(text)
    low = stripped.lower()
    if not stripped:
        return "image_only" if image_only else "blank"
    if re.search(r"\bindex\b", low) and (physical_index > page_count * 0.7 or len(stripped) > 500):
        return "index"
    if re.search(r"\b(bibliography|references|works cited|discography)\b", low) and physical_index > page_count * 0.6:
        return "bibliography"
    if physical_index <= max(12, int(page_count * 0.04)):
        return "frontmatter"
    if re.search(r"\b(appendix|endnotes|notes)\b", low) and physical_index > page_count * 0.7:
        return "backmatter"
    return "body"


def classify_epub_spine(idref: str, headings: list[str], text: str, spine_index: int, spine_count: int) -> str:
    """Stable EPUB navigation class based on publisher IDs and headings."""
    key = f"{idref} {' '.join(headings)}".lower()
    low = text.lower()
    if re.search(r"cover|blank", key) or not text.strip():
        return "cover_or_blank"
    if re.search(r"title|halftitle|copyright|dedication|contents|toc|front", key):
        return "frontmatter"
    if re.search(r"index|glossary|bibliograph|references|discograph|notes|appendix", key):
        return "backmatter"
    if spine_index <= 10:
        return "frontmatter"
    if re.search(r"\bindex\b", low) and spine_index > spine_count * 0.7:
        return "backmatter"
    if re.search(r"part\d+", idref.lower()) and len(text.strip()) < 300:
        return "part_divider"
    return "chapter_or_body"


def pdf_page_images(page: Any, fitz_module: Any | None) -> int:
    if fitz_module is not None:
        try:
            return len(page.get_images(full=True))
        except Exception:
            pass
    try:
        images = getattr(page, "images", None)
        if images is not None:
            return len(images)
    except Exception:
        pass
    try:
        resources = page.get("/Resources", {})
        xobject = resources.get("/XObject", {})
        return sum(1 for obj in xobject.values() if obj.get("/Subtype") == "/Image")
    except Exception:
        return 0


def run_ocr(fitz_page: Any, fitz_module: Any, book_slug: str, page_no: int, tess: str | None) -> tuple[str, str]:
    if not tess:
        return "", "unavailable"
    page_dir = OCR_SCRATCH / safe_slug(book_slug)
    page_dir.mkdir(parents=True, exist_ok=True)
    image_path = page_dir / f"{page_no:04d}.png"
    try:
        pix = fitz_page.get_pixmap(matrix=fitz_module.Matrix(1.6, 1.6), alpha=False)
        pix.save(str(image_path))
        completed = subprocess.run(
            [tess, str(image_path), "stdout", "--psm", "3"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
            check=False,
        )
        text = completed.stdout.strip()
        return text, "used" if text else "attempted_empty"
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return "", "failed"
    finally:
        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            pass


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def pdf_text_from_page(page: Any, fitz_module: Any | None) -> str:
    if fitz_module is not None:
        try:
            return page.get_text("text") or ""
        except Exception:
            return ""
    try:
        return page.extract_text() or ""
    except Exception:
        return ""


def extract_pdf(path: Path, book_dir: Path, slug: str) -> dict[str, Any]:
    fitz = load_fitz()
    Reader = load_pypdf()
    doc = None
    reader = None
    if fitz is not None:
        doc = fitz.open(str(path))
        page_count = int(doc.page_count)
    elif Reader is not None:
        reader = Reader(str(path), strict=False)
        page_count = len(reader.pages)
    else:
        raise RuntimeError("Neither PyMuPDF nor pypdf is available for PDF extraction")

    labels = page_labels_with_pypdf(path, page_count)
    pages_dir = book_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    pages_jsonl = book_dir / "pages.jsonl"
    tess = tesseract_path()
    records: list[dict[str, Any]] = []
    stats = {
        "text_pages": 0,
        "empty_pages": 0,
        "image_only_pages": 0,
        "image_pages": 0,
        "ocr_used_pages": 0,
        "ocr_attempted_empty_pages": 0,
        "ocr_failed_pages": 0,
        "ocr_unavailable_pages": 0,
        "total_text_chars": 0,
        "total_image_count": 0,
    }
    for index in range(page_count):
        if doc is not None:
            page = doc.load_page(index)
        else:
            page = reader.pages[index]  # type: ignore[union-attr]
        extracted = pdf_text_from_page(page, fitz)
        chars_before = len(extracted.strip())
        image_count = pdf_page_images(page, fitz)
        image_only = image_count > 0 and chars_before == 0
        sparse_image = image_count > 0 and chars_before < 80
        ocr_status = "not_needed"
        ocr_text = ""
        if image_only:
            stats["image_only_pages"] += 1
            if fitz is not None:
                ocr_text, ocr_status = run_ocr(page, fitz, slug, index + 1, tess)
                if ocr_status == "used":
                    extracted = ocr_text
                    stats["ocr_used_pages"] += 1
                elif ocr_status == "attempted_empty":
                    stats["ocr_attempted_empty_pages"] += 1
                elif ocr_status == "failed":
                    stats["ocr_failed_pages"] += 1
                elif ocr_status == "unavailable":
                    stats["ocr_unavailable_pages"] += 1
            else:
                ocr_status = "unavailable_no_renderer"
                stats["ocr_unavailable_pages"] += 1
        text = extracted.replace("\r\n", "\n").replace("\r", "\n")
        text_path = pages_dir / f"{index + 1:04d}.txt"
        text_path.write_text(text, encoding="utf-8", newline="\n")
        text_chars = len(text)
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        printed = labels[index] if index < len(labels) else None
        printed_source = "pdf_page_labels" if printed else None
        if not printed:
            fallback = detect_footer_label(text)
            if fallback:
                printed, printed_source = fallback, "text_heuristic"
        if text.strip():
            stats["text_pages"] += 1
        else:
            stats["empty_pages"] += 1
        if image_count:
            stats["image_pages"] += 1
        stats["total_text_chars"] += text_chars
        stats["total_image_count"] += image_count
        scan_flag = "image_only" if image_only else ("sparse_text_with_image" if sparse_image else "text")
        record = {
            "book_slug": slug,
            "format": "pdf",
            "physical_pdf_index": index + 1,
            "printed_page_label": printed,
            "printed_page_label_source": printed_source,
            "text_path": str(text_path.relative_to(book_dir)).replace("\\", "/"),
            "text_chars": text_chars,
            "text_sha256": text_sha,
            "image_count": image_count,
            "scan_flag": scan_flag,
            "page_kind": classify_pdf_page(text, index + 1, page_count, image_only),
            "image_only": image_only,
            "ocr_status": ocr_status,
            "text_source": "ocr" if ocr_status == "used" and chars_before == 0 else "pdf_text",
            "readable_text": bool(text.strip()),
            "preview": compact_text(text)[:240],
        }
        records.append(record)
    if doc is not None:
        doc.close()
    write_jsonl(pages_jsonl, records)
    stats.update(
        {
            "page_count": page_count,
            "ocr_available": bool(tess and fitz is not None),
            "ocr_executable": tess,
            "pages_jsonl": str(pages_jsonl),
            "pages_dir": str(pages_dir),
        }
    )
    return stats


def epub_rootfile(zf: zipfile.ZipFile) -> str:
    root = ET.fromstring(zf.read("META-INF/container.xml"))
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "rootfile":
            return element.attrib["full-path"]
    raise ValueError("EPUB has no rootfile in META-INF/container.xml")


def epub_text_and_metadata(raw: bytes, item_path: str) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Return readable XHTML text, headings and publisher pagebreak offsets."""
    from bs4 import BeautifulSoup  # type: ignore

    soup = BeautifulSoup(raw, "xml")
    headings: list[str] = []
    pieces: list[str] = []
    pagebreaks: list[dict[str, Any]] = []
    for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "br", "div", "section", "span"]):
        tag = node.name.lower()
        if tag == "br":
            pieces.append("\n")
            continue
        text = compact_text(node.get_text(" ", strip=True))
        if not text:
            continue
        if tag.startswith("h"):
            headings.append(text)
            pieces.append(f"\n{text}\n")
        else:
            pieces.append(text)
            pieces.append("\n")
        attrs = {str(k).lower(): str(v).lower() for k, v in node.attrs.items()}
        epub_type = attrs.get("epub:type", "")
        role = attrs.get("role", "")
        classes = attrs.get("class", "")
        if "pagebreak" in epub_type or "doc-pagebreak" in role or "pagebreak" in classes:
            label = node.get("title") or node.get("aria-label") or node.get_text(" ", strip=True)
            pagebreaks.append({"label": compact_text(str(label)) or None, "marker_text": text})
    text = re.sub(r"\n{3,}", "\n\n", "".join(pieces)).strip() + "\n"
    # The DOM walk can visit nested elements twice.  Prefer a simple body text
    # fallback if no content was found; headings/pagebreaks remain metadata.
    if not text.strip():
        text = soup.get_text("\n", strip=True).strip() + "\n"
    return text, headings, pagebreaks


def split_spine_text(text: str, max_chars: int) -> list[tuple[int, int, str]]:
    """Split long chapters at paragraph boundaries with stable character spans."""
    if len(text) <= max_chars:
        return [(0, len(text), text)]
    spans: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        target = min(len(text), start + max_chars)
        if target < len(text):
            boundary = text.rfind("\n\n", start + max_chars // 2, target)
            if boundary > start:
                target = boundary + 2
        chunk = text[start:target]
        spans.append((start, target, chunk))
        start = target
    return spans


def extract_epub(path: Path, book_dir: Path, slug: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> dict[str, Any]:
    chapters_dir = book_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    chapter_records: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as zf:
        opf_path = epub_rootfile(zf)
        opf_dir = Path(opf_path).parent
        root = ET.fromstring(zf.read(opf_path))
        metadata: dict[str, str] = {}
        for element in root.iter():
            local = element.tag.rsplit("}", 1)[-1]
            if local in {"title", "creator", "language", "identifier"} and element.text:
                metadata.setdefault(local, compact_text(element.text))
        manifest: dict[str, dict[str, str]] = {}
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] == "item":
                manifest[element.attrib["id"]] = dict(element.attrib)
        spine_items: list[str] = []
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] == "itemref" and element.attrib.get("idref"):
                spine_items.append(element.attrib["idref"])
        all_chunks = []
        for spine_index, idref in enumerate(spine_items, 1):
            item = manifest.get(idref, {})
            href = item.get("href")
            if not href:
                continue
            href_path = (opf_dir / href).as_posix()
            href_path = str(Path(href_path)).replace("\\", "/")
            try:
                raw = zf.read(href_path)
            except KeyError:
                # EPUB hrefs may include URL fragments; strip them for the ZIP lookup.
                href_path = href_path.split("#", 1)[0]
                raw = zf.read(href_path)
            text, headings, pagebreaks = epub_text_and_metadata(raw, href_path)
            stable_id = f"spine-{spine_index:03d}-{safe_slug(idref)}"
            chapter_path = chapters_dir / f"{stable_id}.txt"
            chapter_path.write_text(text, encoding="utf-8", newline="\n")
            for span_index, (start, end, chunk) in enumerate(split_spine_text(text, chunk_size), 1):
                chunk_id = f"{stable_id}-span-{span_index:04d}"
                chunk_path = chapters_dir / f"{chunk_id}.txt"
                chunk_path.write_text(chunk, encoding="utf-8", newline="\n")
                chunk_breaks = [
                    marker
                    for marker in pagebreaks
                    # marker offsets are not exposed by all XHTML structures;
                    # preserve markers at chapter scope and include them on each chunk.
                ]
                all_chunks.append(
                    {
                        "book_slug": slug,
                        "format": "epub",
                        "record_type": "spine_chunk",
                        "spine_index": spine_index,
                        "spine_idref": idref,
                        "chapter_id": stable_id,
                        "chunk_id": chunk_id,
                        "char_start": start,
                        "char_end": end,
                        "text_path": str(chunk_path.relative_to(book_dir)).replace("\\", "/"),
                        "text_chars": len(chunk),
                        "text_sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                        "section_headings": headings,
                        "chapter_role": classify_epub_spine(idref, headings, text, spine_index, len(spine_items)),
                        "publisher_pagebreak_markers": chunk_breaks,
                        "physical_pdf_index": None,
                        "printed_page_label": None,
                        "readable_text": bool(chunk.strip()),
                        "preview": compact_text(chunk)[:240],
                    }
                )
            chapter_records.append(
                {
                    "book_slug": slug,
                    "format": "epub",
                    "record_type": "spine_chapter",
                    "spine_index": spine_index,
                    "spine_idref": idref,
                    "chapter_id": stable_id,
                    "source_href": href_path,
                    "text_path": str(chapter_path.relative_to(book_dir)).replace("\\", "/"),
                    "text_chars": len(text),
                    "section_headings": headings,
                    "chapter_role": classify_epub_spine(idref, headings, text, spine_index, len(spine_items)),
                    "publisher_pagebreak_markers": pagebreaks,
                    "chunk_count": len(split_spine_text(text, chunk_size)),
                }
            )
        epub_manifest = {
            "book_slug": slug,
            "format": "epub",
            "metadata": metadata,
            "spine_count": len(spine_items),
            "chapters": chapter_records,
            "text_policy": "spine_order; stable chapter IDs; no fabricated pages; long chapters use char spans",
        }
        (book_dir / "epub_manifest.json").write_text(
            json.dumps(epub_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    chunks_jsonl = book_dir / "chunks.jsonl"
    write_jsonl(chunks_jsonl, all_chunks)
    return {
        "spine_count": len(spine_items),
        "chunk_count": len(all_chunks),
        "chapter_count": len(chapter_records),
        "total_text_chars": sum(item["text_chars"] for item in chapter_records),
        "chapters_dir": str(chapters_dir),
        "chunks_jsonl": str(chunks_jsonl),
        "ocr_available": False,
        "page_count": None,
    }


def source_entries() -> list[tuple[str, Path]]:
    entries = []
    for slug, filename in BOOKS:
        path = SOURCE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing supplied book: {path}")
        entries.append((slug, path))
    return entries


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    books_root = PRIVATE_ROOT / "books"
    books_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "generated_utc": utc_now(),
        "source_dir": str(SOURCE_DIR),
        "private_output_dir": str(PRIVATE_ROOT),
        "text_outside_git": True,
        "ocr": {
            "tesseract": tesseract_path(),
            "renderer": "PyMuPDF" if load_fitz() is not None else "pypdf_fallback",
            "note": "image-only pages with unavailable/failed OCR remain unreadable; empty extraction is never proof of read coverage",
        },
        "books": [],
    }
    for slug, path in source_entries():
        print(f"[{slug}] {path.name}", flush=True)
        book_dir = books_root / slug
        book_dir.mkdir(parents=True, exist_ok=True)
        fmt = path.suffix.lower().lstrip(".")
        info: dict[str, Any] = {
            "slug": slug,
            "title": path.stem,
            "source_filename": path.name,
            "source_path": str(path),
            "format": fmt,
            "source_bytes": path.stat().st_size,
            "source_sha256": sha256(path),
            "book_output_dir": str(book_dir),
        }
        if fmt == "pdf":
            info.update(extract_pdf(path, book_dir, slug))
        elif fmt == "epub":
            info.update(extract_epub(path, book_dir, slug, args.chunk_size))
        else:
            raise ValueError(f"Unsupported book format: {path}")
        manifest["books"].append(info)
        (PRIVATE_ROOT / "manifest.partial.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"slug": slug, "format": fmt, "page_count": info.get("page_count"), "spine_count": info.get("spine_count"), "total_text_chars": info.get("total_text_chars")}, ensure_ascii=False), flush=True)
    manifest_path = PRIVATE_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        (PRIVATE_ROOT / "manifest.partial.json").unlink()
    except FileNotFoundError:
        pass
    print(f"MANIFEST {manifest_path}")
    return manifest


def load_manifest() -> dict[str, Any]:
    path = PRIVATE_ROOT / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Run prepare first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def search(args: argparse.Namespace) -> None:
    rx = re.compile(args.query, re.I)
    manifest = load_manifest()
    matched = 0
    book_filter = args.book.lower() if args.book else None
    for book in manifest.get("books", []):
        if book_filter and book_filter not in book["slug"].lower() and book_filter not in book["title"].lower():
            continue
        book_dir = Path(book["book_output_dir"])
        jsonl_name = "pages.jsonl" if book["format"] == "pdf" else "chunks.jsonl"
        jsonl_path = book_dir / jsonl_name
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            text_path = book_dir / record["text_path"]
            text = text_path.read_text(encoding="utf-8", errors="replace")
            for hit in rx.finditer(text):
                left = max(0, hit.start() - args.context)
                right = min(len(text), hit.end() + args.context)
                locator = (
                    f"pdf p.{record['physical_pdf_index']}"
                    if book["format"] == "pdf"
                    else f"spine {record.get('spine_index')} {record.get('chunk_id')}"
                )
                print(f"[{book['slug']} | {locator}] …{compact_text(text[left:right])}…")
                matched += 1
                if matched >= args.limit:
                    return


def print_manifest(args: argparse.Namespace) -> None:
    manifest = load_manifest()
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    # PowerShell may expose a legacy cp1252 stdout; extracted books contain
    # typographic punctuation and zero-width characters, so keep CLI search
    # and diagnostics printable instead of crashing on encoding.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare", help="extract all supplied books into the private corpus")
    prep.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="maximum EPUB chapter chunk size; never a page label")
    q = subs.add_parser("search", help="search extracted page/chunk files")
    q.add_argument("query")
    q.add_argument("--book")
    q.add_argument("--limit", type=int, default=20)
    q.add_argument("--context", type=int, default=180)
    subs.add_parser("manifest", help="print the private manifest")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "search":
        search(args)
    else:
        print_manifest(args)


if __name__ == "__main__":
    main()
