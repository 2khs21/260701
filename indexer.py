"""Context Management RAG starter — indexer.

Walks documents/, chunks each file, embeds chunks, persists the index to disk
so the chat backend can load it without re-indexing.

TODO: implement chunk_text(). The embedding and storage code is provided so
you can focus on the structure.
"""
import math
import pickle
import re
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Multilingual (50+ languages), 384-dim — same model as the /embedding project.
# Lets the corpus and the queries be in different languages and still match.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = Path(__file__).parent / "index.pkl"
DOCS_DIR = Path(__file__).parent / "documents"
MIN_CHUNK_CHARS = 80
SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf")
HYBRID_POOL_K = 40
RRF_K = 60
BM25_RRF_K = 40
VECTOR_GUARANTEE = 3
BM25_GUARANTEE = 6
BM25_K1 = 1.5
BM25_B = 0.75

_SECTION_START_RE = re.compile(r"^§\s*\d+[\d.]*", re.MULTILINE)
_CFR_PAGE_BREAK_RE = re.compile(r"^--\s*\d+\s+of\s+\d+\s*--\s*$", re.MULTILINE | re.IGNORECASE)
_CFR_RUNNING_HEADER_RE = re.compile(
    r"^Federal Aviation Administration, DOT\s+§\s*\d+[\d.]*\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_CFR_EDITION_LINE_RE = re.compile(
    r"^14 CFR Ch\.[^\n]*§\s*\d+[\d.]*\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_CFR_LONE_PAGE_NUM_RE = re.compile(r"^\d{3,4}\s*$", re.MULTILINE)
_SECTION_ID_RE = re.compile(r"§\s*(\d+(?:\.\d+)*)")
_SECTION_TITLE_RE = re.compile(
    r"^§\s*(\d+(?:\.\d+)*)\s+([\s\S]+?)\.\s*\([a-z]\)",
    re.MULTILINE | re.IGNORECASE,
)
_PART_IN_SOURCE_RE = re.compile(r"part(\d+)", re.IGNORECASE)


# ════════════════════════════════════════════════════════════════
# TODO — implement chunk_text
#
# Split `text` into overlapping chunks. A reasonable default:
#   - ~1000 characters per chunk
#   - ~100 characters of overlap
#   - try to break on paragraph boundaries (\n\n) when possible
#
# Return a list of non-empty strings.
# See the lecture slide on chunking for one working implementation.
# ════════════════════════════════════════════════════════════════

def _split_into_sections(text: str) -> list[str]:
    """Split on §-style section boundaries; keep each section intact when possible."""
    matches = list(_SECTION_START_RE.finditer(text))
    if not matches:
        return [text.strip()] if text.strip() else []

    sections: list[str] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(preamble)

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if block:
            sections.append(block)
    return sections


def _split_long_block(block: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Split an oversized section on paragraph boundaries, then by character window."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", block) if p.strip()]
    if not paragraphs:
        paragraphs = [block]

    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= target_chars:
            pieces.append(para)
        else:
            step = max(target_chars - overlap_chars, target_chars // 2)
            for start in range(0, len(para), step):
                pieces.append(para[start : start + target_chars])

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > target_chars:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = (tail + "\n\n" + piece) if tail else piece
        else:
            current = (current + "\n\n" + piece) if current else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, target_chars: int = 1000, overlap_chars: int = 100) -> list[str]:
    """Pack §-aligned sections into overlapping chunks (~target_chars each)."""
    sections = _split_into_sections(text)
    if not sections:
        return _split_long_block(text, target_chars, overlap_chars)

    chunks: list[str] = []
    current = ""
    for section in sections:
        if len(section) > target_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_block(section, target_chars, overlap_chars))
            continue
        if current and len(current) + 2 + len(section) > target_chars:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = (tail + "\n\n" + section) if tail else section
        else:
            current = (current + "\n\n" + section) if current else section

    if current:
        chunks.append(current)

    # Markdown: merge lone # titles into the following chunk.
    merged: list[str] = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        is_short_title = chunk.lstrip().startswith("#") and len(chunk) < MIN_CHUNK_CHARS
        if is_short_title and i + 1 < len(chunks):
            merged.append(chunk + "\n\n" + chunks[i + 1])
            i += 2
        else:
            merged.append(chunk)
            i += 1

    return [c for c in merged if c.strip() and not is_noise_text(c)]


def _has_noise_pattern(text: str) -> bool:
    """Dot-leader / TOC patterns without applying minimum-length rules."""
    stripped = text.strip()
    if not stripped:
        return True
    if re.search(r"\.{8,}", stripped):
        return True
    dot_ratio = stripped.count(".") / len(stripped)
    if dot_ratio > 0.12:
        return True
    lines = [ln.strip() for ln in stripped.split("\n") if ln.strip()]
    if lines:
        leader_lines = sum(
            1 for ln in lines
            if re.search(r"\.{4,}\s*\d*\s*$", ln) or re.fullmatch(r"[\d.\s]+", ln)
        )
        if leader_lines / len(lines) > 0.6:
            return True
    return False


def is_noise_text(text: str) -> bool:
    """Low-signal PDF artifacts: dot leaders, TOC lines, very short blocks."""
    stripped = text.strip()
    if len(stripped) < MIN_CHUNK_CHARS:
        return True
    return _has_noise_pattern(stripped)


def part_from_source(source: str) -> str | None:
    """Extract CFR part number from filenames like ...-part91.pdf."""
    match = _PART_IN_SOURCE_RE.search(source)
    return match.group(1) if match else None


def is_toc_chunk_text(text: str) -> bool:
    """Section index listings (many cross-refs, no operative paragraphs)."""
    if re.search(r"\([a-z]\)", text, re.IGNORECASE):
        return False
    section_refs = len(re.findall(r"\b\d+\.\d+\b", text))
    if section_refs >= 4:
        return True
    return text.count("§") >= 4


def extract_chunk_metadata(text: str, source: str) -> dict:
    """Derive CFR-oriented metadata for a chunk (part, section, title, toc flag)."""
    part = part_from_source(source)
    sections = list(dict.fromkeys(_SECTION_ID_RE.findall(text)))
    primary = sections[0] if sections else None

    title: str | None = None
    title_match = _SECTION_TITLE_RE.search(text)
    if title_match:
        title = re.sub(r"\s+", " ", title_match.group(2)).strip().rstrip(".")

    is_toc = is_toc_chunk_text(text)
    if sections and not is_toc:
        chunk_type = "regulation"
    elif is_toc:
        chunk_type = "toc"
    else:
        chunk_type = "general"

    return {
        "part": part,
        "section": primary,
        "sections": sections[:8],
        "section_title": title,
        "is_toc": is_toc,
        "chunk_type": chunk_type,
    }


def metadata_search_tokens(meta: dict) -> list[str]:
    """Extra BM25 terms from structured metadata."""
    extra: list[str] = []
    part = meta.get("part")
    if part:
        extra.extend([f"part{part}", f"part {part}"])
    for section in meta.get("sections") or []:
        extra.append(section)
        extra.append(section.replace(".", ""))
    title = meta.get("section_title")
    if title:
        extra.extend(tokenize(title)[:12])
    return extra


def enrich_record_metadata(record: dict) -> None:
    """Ensure structured metadata and BM25 token fields are present."""
    record.update(extract_chunk_metadata(record["text"], record["source"]))


def build_record_tokens(text: str, meta: dict) -> list[str]:
    return tokenize(text) + metadata_search_tokens(meta)


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25 — splits hyphens/underscores for compound terms."""
    normalized = re.sub(r"[-_/]", " ", text.lower())
    raw = re.findall(r"[\w]+(?:\.[\w]+)*", normalized)
    return [t for t in raw if len(t) >= 2]


def _build_bm25_stats(records: list[dict]) -> dict:
    """Precompute IDF and doc lengths from record token lists."""
    n = len(records)
    if n == 0:
        return {"idf": {}, "avgdl": 0.0, "doc_lens": []}

    doc_lens: list[int] = []
    df: dict[str, int] = {}
    for record in records:
        tokens = record.get("tokens")
        if tokens is None:
            tokens = tokenize(record["text"])
            record["tokens"] = tokens
        doc_lens.append(len(tokens))
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    avgdl = sum(doc_lens) / n
    idf = {
        term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }
    return {"idf": idf, "avgdl": avgdl, "doc_lens": doc_lens}


_bm25_cache: tuple[int, dict] | None = None


def _get_bm25_stats(records: list[dict]) -> dict:
    global _bm25_cache
    cache_key = id(records)
    if _bm25_cache and _bm25_cache[0] == cache_key:
        return _bm25_cache[1]
    stats = _build_bm25_stats(records)
    _bm25_cache = (cache_key, stats)
    return stats


def bm25_search(
    query: str,
    records: list[dict],
    k: int = HYBRID_POOL_K,
) -> list[tuple[int, float]]:
    """Return top-k (chunk_id, bm25_score) pairs, highest score first."""
    stats = _get_bm25_stats(records)
    idf = stats["idf"]
    avgdl = stats["avgdl"]
    doc_lens = stats["doc_lens"]
    if not records or avgdl == 0:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return []

    scored: list[tuple[int, float]] = []
    for i, record in enumerate(records):
        tokens = record.get("tokens") or []
        if not tokens:
            continue
        tf: dict[str, int] = {}
        for term in tokens:
            tf[term] = tf.get(term, 0) + 1
        dl = doc_lens[i]
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            term_idf = idf.get(term, 0.0)
            freq = tf[term]
            denom = freq + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
            score += term_idf * (freq * (BM25_K1 + 1)) / denom
        if score > 0:
            scored.append((record["chunk_id"], score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def rrf_merge(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over ranked chunk_id lists."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rrf_merge_dual(
    vector_ranking: list[int],
    bm25_ranking: list[int],
) -> list[tuple[int, float]]:
    """RRF with a lower k for BM25 so keyword matches rank higher."""
    scores: dict[int, float] = {}
    for rank, chunk_id in enumerate(vector_ranking, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    for rank, chunk_id in enumerate(bm25_ranking, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (BM25_RRF_K + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def read_pdf(path: Path) -> str:
    """Extract text from a PDF, one block per page."""
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def clean_cfr_noise(text: str) -> str:
    """Remove CFR PDF page breaks, running headers, and line-break hyphenation."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CFR_PAGE_BREAK_RE.sub("", text)
    text = _CFR_RUNNING_HEADER_RE.sub("", text)
    text = _CFR_EDITION_LINE_RE.sub("", text)
    text = _CFR_LONE_PAGE_NUM_RE.sub("", text)
    # Rejoin words split across lines: "ey-\n\ne" or "comm\nand" → "eye" / "command"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"(?<=[a-z,;])\s*\n\s*(?=[a-z])", " ", text)
    return text


def _strip_dot_leader_lines(text: str) -> str:
    """Remove CFR TOC / weather-table dot-leader lines from a block."""
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"\.{4,}\s*\d*\s*$", stripped):
            continue
        if re.fullmatch(r"[\d.\s]+", stripped):
            continue
        if re.search(r"\.{8,}", stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept).strip()


def _recover_paragraphs(paragraphs: list[str]) -> list[str]:
    """Salvage regulatory text from noisy PDF blocks and split at § boundaries."""
    recovered: list[str] = []
    for para in paragraphs:
        candidates = [para]
        if _has_noise_pattern(para):
            stripped = _strip_dot_leader_lines(para)
            if stripped:
                candidates = [stripped]

        for candidate in candidates:
            if len(candidate) > 1500:
                sections = _split_into_sections(candidate)
                if len(sections) > 1:
                    recovered.extend(_recover_paragraphs(sections))
                    continue
            if candidate and not _has_noise_pattern(candidate):
                recovered.append(candidate)
    return recovered


def normalize_extracted_text(text: str) -> str:
    """Collapse PDF extraction noise while keeping paragraph breaks."""
    text = clean_cfr_noise(text)
    text = re.sub(r"[ \t]+", " ", text)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in text.split("\n") if line.strip()]
    paragraphs = _recover_paragraphs(paragraphs)
    return "\n\n".join(paragraphs)


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return normalize_extracted_text(path.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".pdf":
        return normalize_extracted_text(read_pdf(path))
    raise ValueError(f"Unsupported file type: {path.name}")


# ════════════════════════════════════════════════════════════════
# Provided: embedding (sentence-transformers, no API key required)
# ════════════════════════════════════════════════════════════════

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model ({MODEL_NAME})... (one-time download ~470MB)")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns unit-normalized 384-dim vectors."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


# ════════════════════════════════════════════════════════════════
# Provided: build / save / load / search
# ════════════════════════════════════════════════════════════════

def build_index() -> list[dict]:
    """Walk DOCS_DIR, chunk each file, embed, return list of records."""
    records: list[dict] = []
    chunk_id = 0
    for path in sorted(DOCS_DIR.glob("*")):
        if path.is_dir() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            text = read_document(path)
        except Exception as exc:
            print(f"  {path.name}: skipped ({exc})")
            continue
        if not text.strip():
            print(f"  {path.name}: skipped (no extractable text)")
            continue
        chunks = chunk_text(text)
        if not chunks:
            print(f"  {path.name}: skipped (no chunks after filtering)")
            continue
        vectors = embed(chunks)
        kept = 0
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            meta = extract_chunk_metadata(chunk, path.name)
            records.append({
                "chunk_id": chunk_id,
                "source": path.name,
                "chunk_index": i,
                "text": chunk,
                "embedding": vec,
                "tokens": build_record_tokens(chunk, meta),
                **meta,
            })
            chunk_id += 1
            kept += 1
        print(f"  {path.name}: {kept} chunks")
    return records


def save_index(records: list[dict]) -> None:
    payload = {"version": 3, "records": records}
    with INDEX_PATH.open("wb") as f:
        pickle.dump(payload, f)


def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {INDEX_PATH}. Run `python indexer.py` from the project root first."
        )
    with INDEX_PATH.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        records = data["records"]
    else:
        records = data
    for record in records:
        enrich_record_metadata(record)
        record["tokens"] = build_record_tokens(record["text"], record)
    global _bm25_cache
    _bm25_cache = None
    return records


def cosine_distance(a: list[float], b: list[float]) -> float:
    # Both vectors are unit-normalized, so cosine distance == 1 - dot product.
    return 1.0 - sum(x * y for x, y in zip(a, b))


def vector_search(
    query: str,
    records: list[dict],
    k: int = 5,
    max_distance: float | None = None,
) -> list[dict]:
    """Embed the query, return top-k records by cosine distance (lower = closer)."""
    [query_vec] = embed([query])
    scored = [(cosine_distance(r["embedding"], query_vec), r) for r in records]
    scored.sort(key=lambda x: x[0])
    hits: list[dict] = []
    for dist, record in scored:
        if is_noise_text(record["text"]):
            continue
        if max_distance is not None and dist > max_distance:
            continue
        hits.append({**record, "distance": dist, "vector_distance": dist})
        if len(hits) >= k:
            break
    return hits


def search(
    query: str,
    records: list[dict],
    k: int = 5,
    max_distance: float | None = None,
    pool_k: int | None = None,
) -> list[dict]:
    """Hybrid vector + BM25 retrieval fused with RRF."""
    return hybrid_search(
        query,
        records,
        k=k,
        max_distance=max_distance,
        pool_k=pool_k or max(k * 4, HYBRID_POOL_K),
    )


def hybrid_search(
    query: str,
    records: list[dict],
    k: int = 5,
    max_distance: float | None = None,
    pool_k: int = HYBRID_POOL_K,
) -> list[dict]:
    """Merge vector and BM25 rankings with Reciprocal Rank Fusion."""
    if not records:
        return []

    vector_hits = vector_search(query, records, k=pool_k, max_distance=None)
    bm25_hits = bm25_search(query, records, k=pool_k)

    by_id = {r["chunk_id"]: r for r in records}
    vector_ranking = [h["chunk_id"] for h in vector_hits]
    bm25_ranking = [cid for cid, _ in bm25_hits]
    bm25_ids = set(bm25_ranking)

    if not vector_ranking and not bm25_ranking:
        return []

    fused = rrf_merge_dual(vector_ranking, bm25_ranking)
    fused_scores = dict(fused)
    vector_dist = {h["chunk_id"]: h["distance"] for h in vector_hits}

    def _make_hit(chunk_id: int, rrf_score: float) -> dict | None:
        record = by_id.get(chunk_id)
        if record is None or is_noise_text(record["text"]):
            return None
        dist = vector_dist.get(chunk_id, 1.0)
        if max_distance is not None and dist > max_distance and chunk_id not in bm25_ids:
            return None
        return {
            **record,
            "distance": dist,
            "vector_distance": dist,
            "rrf_score": rrf_score,
        }

    must_include: list[int] = []
    for cid in vector_ranking[:VECTOR_GUARANTEE]:
        if cid not in must_include:
            must_include.append(cid)
    for cid in bm25_ranking[:BM25_GUARANTEE]:
        if cid not in must_include:
            must_include.append(cid)

    ordered_ids: list[int] = []
    seen: set[int] = set()
    for cid in sorted(must_include, key=lambda c: -fused_scores.get(c, 0)):
        if cid not in seen:
            ordered_ids.append(cid)
            seen.add(cid)
    for chunk_id, _ in fused:
        if chunk_id in seen:
            continue
        ordered_ids.append(chunk_id)
        seen.add(chunk_id)
        if len(ordered_ids) >= k:
            break

    hits: list[dict] = []
    for chunk_id in ordered_ids[:k]:
        hit = _make_hit(chunk_id, fused_scores.get(chunk_id, 0.0))
        if hit is not None:
            hits.append(hit)

    hits.sort(key=lambda h: (-h.get("rrf_score", 0), h.get("distance", 1.0)))
    return hits


def main() -> None:
    print(f"Indexing documents from {DOCS_DIR}/")
    records = build_index()
    save_index(records)
    print(f"\n✓ Indexed {len(records)} chunks → {INDEX_PATH.name}")


if __name__ == "__main__":
    main()
