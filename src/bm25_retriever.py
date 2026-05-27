"""
src/bm25_retriever.py
Config C: BM25 pure keyword search — tanpa embedding, tanpa ChromaDB.
Mengindeks chunk-chunk (bukan full dokumen) dengan granularitas Config B
untuk perbandingan metodologi yang valid dengan Config A & B.

CLI: python src/bm25_retriever.py [--rebuild]
"""

import argparse
import logging
import pickle
import time
from pathlib import Path

from src.logger_manager import log_ingestion_event, log_system_event


import frontmatter
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from rank_bm25 import BM25Okapi

from src.config import (
    CORPUS_DIR,
    BM25_INDEX_DIR,
    BM25_INDEX_PATH,
    BM25_K,
    BM25_MIN_TOKEN_LEN,
    CHUNK_SIZE_B,
    CHUNK_OVERLAP_B,
    CHUNK_SEPARATORS,
    MIN_CHUNK_LENGTH,
    REQUIRED_YAML_FIELDS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Hierarki heading — konsisten dengan ingestion.py
_HEADERS_TO_SPLIT = [
    ("#",    "header_1"),
    ("##",   "bab"),
    ("###",  "bagian"),
    ("####", "pasal"),
]


def _tokenize(text: str) -> list[str]:
    """
    Tokenisasi sederhana: lowercase, split whitespace, filter token pendek.
    Tidak menggunakan NLTK atau stopword eksternal (menjaga dependensi minimal).
    """
    tokens = text.lower().split()
    return [t for t in tokens if len(t) >= BM25_MIN_TOKEN_LEN]


def _split_to_chunks(content: str, base_metadata: dict) -> list[dict]:
    """
    Split konten dokumen menjadi chunks menggunakan granularitas Config B.
    BM25 mengindeks chunk (bukan full dokumen) untuk perbandingan fair.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    header_chunks = header_splitter.split_text(content)

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_B,
        chunk_overlap=CHUNK_OVERLAP_B,
        separators=CHUNK_SEPARATORS,
    )
    final_chunks = size_splitter.split_documents(header_chunks)

    result = []
    for chunk in final_chunks:
        if len(chunk.page_content.strip()) < MIN_CHUNK_LENGTH:
            continue
        result.append({
            "content": chunk.page_content,
            "metadata": {
                "doc_id":   base_metadata.get("doc_id", ""),
                "title":    base_metadata.get("title", ""),
                "category": base_metadata.get("category", ""),
                "bab":      chunk.metadata.get("bab", ""),
                "bagian":   chunk.metadata.get("bagian", ""),
                "preview":  chunk.page_content[:150],
            },
        })
    return result


def build_index() -> None:
    """
    Bangun indeks BM25 dari chunk-chunk semua file .md di CORPUS_DIR.
    Simpan ke BM25_INDEX_PATH sebagai pickle {index, texts, metadata}.
    """
    md_files = sorted(CORPUS_DIR.glob("*.md"))
    if not md_files:
        logger.warning(f"Tidak ada file .md di {CORPUS_DIR}")
        return

    all_texts: list[str]  = []
    all_metadata: list[dict] = []

    for md_file in md_files:
        t_start = time.time()
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)
        except Exception as e:
            logger.warning(f"SKIP {md_file.name}: {e}")
            log_system_event("warning", "BM25_PARSE", f"Gagal parse {md_file.name}: {e}")
            continue

        meta = dict(post.metadata)

        # Validasi field wajib
        missing = [f for f in REQUIRED_YAML_FIELDS if not meta.get(f)]
        if missing:
            logger.warning(f"SKIP {md_file.name}: field wajib kosong: {missing}")
            continue

        # Split ke chunks (granularitas Config B)
        chunks = _split_to_chunks(post.content, meta)

        # Tambahkan juga summary chunk dari retrieval_summary
        summary_text = meta.get("retrieval_summary", "")
        if len(summary_text.strip()) >= MIN_CHUNK_LENGTH:
            chunks.insert(0, {
                "content": summary_text,
                "metadata": {
                    "doc_id":   meta.get("doc_id", ""),
                    "title":    meta.get("title", ""),
                    "category": meta.get("category", ""),
                    "bab":      "Ringkasan Dokumen",
                    "bagian":   "Ringkasan Dokumen",
                    "preview":  summary_text[:150],
                },
            })

        for chunk in chunks:
            all_texts.append(chunk["content"])
            all_metadata.append(chunk["metadata"])

        duration = time.time() - t_start
        # Catat event per berkas ke ingestion_report
        log_ingestion_event(
            config="c",
            file_processed=md_file.name,
            chunks_generated=len(chunks),
            chunks_inserted=len(chunks),
            chunks_duplicate_skipped=0,
            execution_time_seconds=duration
        )

        logger.info(f"  ✓ {md_file.name}: {len(chunks)} chunk diindeks")

    if not all_texts:
        logger.error("Tidak ada chunk yang berhasil diindeks.")
        return

    tokenized_corpus = [_tokenize(text) for text in all_texts]
    bm25_index = BM25Okapi(tokenized_corpus)

    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(
            {
                "index":    bm25_index,
                "texts":    all_texts,
                "metadata": all_metadata,
            },
            f,
        )
    logger.info(
        f"Indeks BM25 disimpan ke: {BM25_INDEX_PATH} "
        f"({len(all_texts)} chunk dari {len(md_files)} file)"
    )


def _load_index() -> dict:
    """Muat indeks BM25 dari disk. Raises RuntimeError jika belum dibangun."""
    if not BM25_INDEX_PATH.exists():
        raise RuntimeError(
            f"Indeks BM25 belum ada di {BM25_INDEX_PATH}!\n"
            "Jalankan: python src/bm25_retriever.py --rebuild"
        )
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


# Cache indeks di memori agar tidak reload setiap query
_index_cache: dict | None = None


def retrieve_chunks_bm25(query: str) -> list[dict]:
    """
    Ambil BM25_K chunk paling relevan untuk query.
    Mengembalikan list[dict] dengan format kompatibel get_response() di chain.py:
    [{"content": str, "metadata": dict, "score": float}]
    """
    global _index_cache
    if _index_cache is None:
        _index_cache = _load_index()

    bm25: BM25Okapi  = _index_cache["index"]
    texts: list[str] = _index_cache["texts"]
    metas: list[dict]= _index_cache["metadata"]

    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Ambil top BM25_K indeks, urutkan descending
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:BM25_K]

    results = []
    for idx in top_indices:
        results.append({
            "content":  texts[idx],
            "metadata": metas[idx],
            "score":    float(scores[idx]),
        })
    return results


def main():
    """Entry point CLI untuk membangun indeks BM25."""
    parser = argparse.ArgumentParser(
        description="Build BM25 index untuk Config C UNSRAT RAG"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Bangun ulang indeks dari nol (overwrite jika sudah ada)",
    )
    args = parser.parse_args()

    if not args.rebuild and BM25_INDEX_PATH.exists():
        logger.info(
            f"Indeks sudah ada di {BM25_INDEX_PATH}. "
            "Gunakan --rebuild untuk membangun ulang."
        )
        return

    build_index()


if __name__ == "__main__":
    main()
