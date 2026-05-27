"""
src/retriever.py
Unified retrieval interface untuk semua konfigurasi (A, B, C).
Semua config mengembalikan format identik: list[dict] dengan
keys: content, metadata, score.

chain.py tidak perlu tahu config mana yang aktif — cukup panggil retrieve().
"""

import logging
from pathlib import Path

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    CHROMA_DIR_A,
    CHROMA_DIR_B,
    CHROMA_COLLECTION_A,
    CHROMA_COLLECTION_B,
    RETRIEVAL_K,
    SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL_NAME,
    GOOGLE_API_KEY,
)
from src.bm25_retriever import retrieve_chunks_bm25

logger = logging.getLogger(__name__)

# Cache ChromaDB clients agar tidak reinisialisasi setiap query
_chroma_clients: dict[str, chromadb.Collection] = {}

# Cache embedding model
_embedding_model: GoogleGenerativeAIEmbeddings | None = None


def _get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Dapatkan atau buat instance embedding model (lazy init, task_type=query)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            task_type="retrieval_query",
            google_api_key=GOOGLE_API_KEY,
        )
    return _embedding_model


def _get_chroma_collection(config: str) -> chromadb.Collection:
    """Dapatkan atau buat ChromaDB collection untuk config tertentu."""
    if config not in _chroma_clients:
        if config == "a":
            chroma_dir = CHROMA_DIR_A
            collection_name = CHROMA_COLLECTION_A
        elif config == "b":
            chroma_dir = CHROMA_DIR_B
            collection_name = CHROMA_COLLECTION_B
        else:
            raise ValueError(f"Config tidak valid untuk ChromaDB: '{config}'")

        if not Path(chroma_dir).exists():
            raise RuntimeError(
                f"Database ChromaDB tidak ditemukan di {chroma_dir}!\n"
                f"Jalankan: python src/ingestion.py --config {config} --rebuild"
            )

        client = chromadb.PersistentClient(path=str(chroma_dir))
        _chroma_clients[config] = client.get_collection(collection_name)

    return _chroma_clients[config]


def retrieve(query: str, config: str) -> list[dict]:
    """
    Ambil chunk paling relevan untuk query sesuai config yang dipilih.

    Args:
        query:  Pertanyaan dari pengguna.
        config: "a" (ChromaDB 500 char), "b" (ChromaDB 2000 char),
                atau "c" (BM25 chunk-based).

    Returns:
        list[dict] dengan keys: content (str), metadata (dict), score (float).
        List kosong jika tidak ada chunk yang lolos threshold (Config A/B)
        atau jika indeks kosong (Config C).

    Raises:
        RuntimeError: jika database/indeks belum dibangun.
        ValueError: jika config tidak valid.
    """
    if config == "c":
        return retrieve_chunks_bm25(query)

    if config not in ("a", "b"):
        raise ValueError(
            f"Config tidak valid: '{config}'. Pilihan: 'a', 'b', atau 'c'."
        )

    collection = _get_chroma_collection(config)
    embeddings = _get_embedding_model()

    query_vector = embeddings.embed_query(query)

    raw_results = collection.query(
        query_embeddings=[query_vector],
        n_results=RETRIEVAL_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        raw_results["documents"][0],
        raw_results["metadatas"][0],
        raw_results["distances"][0],
    ):
        if dist > SIMILARITY_THRESHOLD:
            continue  # buang chunk yang terlalu jauh secara semantik
        chunks.append({
            "content":  doc,
            "metadata": meta,
            "score":    float(dist),
        })

    if not chunks:
        logger.info(
            f"Tidak ada chunk lolos threshold {SIMILARITY_THRESHOLD} "
            f"untuk query: '{query[:50]}...'"
        )

    return chunks
