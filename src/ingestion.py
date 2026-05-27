"""
src/ingestion.py
Pipeline ingestion: baca .md → chunk → embed → simpan ke ChromaDB.
Mendukung Config A (chunk 500) dan Config B (chunk 2000).
Jalankan via CLI: python src/ingestion.py --config [a|b] [--rebuild]
"""

import argparse
import hashlib
import logging
import time
from pathlib import Path

from src.logger_manager import log_ingestion_event, log_system_event


import frontmatter
import chromadb
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    CORPUS_DIR,
    CHROMA_DIR_A,
    CHROMA_DIR_B,
    CHROMA_COLLECTION_A,
    CHROMA_COLLECTION_B,
    CHROMA_DISTANCE_FN,
    CHUNK_SIZE_A,
    CHUNK_OVERLAP_A,
    CHUNK_SIZE_B,
    CHUNK_OVERLAP_B,
    CHUNK_SEPARATORS,
    MIN_CHUNK_LENGTH,
    EMBEDDING_MODEL_NAME,
    GOOGLE_API_KEY,
    REQUIRED_YAML_FIELDS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Hierarchy heading untuk MarkdownHeaderTextSplitter
HEADERS_TO_SPLIT = [
    ("#",    "header_1"),
    ("##",   "bab"),
    ("###",  "bagian"),
    ("####", "pasal"),
]


def generate_chunk_id(doc_id: str, content: str) -> str:
    """
    Buat ID unik untuk chunk berdasarkan MD5 hash doc_id + konten.
    Digunakan untuk cek duplikat (idempotency).
    """
    raw = f"{doc_id}::{content}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def parse_markdown_file(file_path: Path) -> tuple[dict, str]:
    """
    Baca file .md dan kembalikan (metadata_dict, content_string).
    Raises ValueError jika frontmatter tidak ada atau field wajib kosong.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)
    except Exception as e:
        raise ValueError(f"Gagal membaca {file_path.name}: {e}")

    metadata = dict(post.metadata)
    content = post.content

    for field in REQUIRED_YAML_FIELDS:
        if not metadata.get(field):
            raise ValueError(
                f"Field wajib '{field}' kosong atau tidak ada "
                f"di frontmatter {file_path.name}"
            )

    return metadata, content


def create_summary_chunk(metadata: dict) -> dict:
    """
    Buat satu chunk mandiri dari field retrieval_summary.
    Chunk ini diberi metadata default untuk field bab/bagian/pasal.
    """
    return {
        "content": metadata["retrieval_summary"],
        "metadata": {
            "doc_id":       metadata["doc_id"],
            "title":        metadata["title"],
            "category":     metadata["category"],
            "content_type": metadata["content_type"],
            "chunk_type":   "summary",
            "bab":          "Ringkasan Dokumen",
            "bagian":       "Ringkasan Dokumen",
            "pasal":        "Ringkasan Dokumen",
            "priority":     metadata.get("priority", 3),
            "status":       metadata.get("status", "active"),
        },
    }


def split_content(
    content: str,
    base_metadata: dict,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    Split konten dokumen menjadi daftar chunk dengan metadata.
    Tahap 1: MarkdownHeaderTextSplitter (struktural).
    Tahap 2: RecursiveCharacterTextSplitter (normalisasi ukuran).

    Catatan: chunk_strategy dan chunk_notes di YAML frontmatter tidak dibaca
    oleh fungsi ini — pipeline selalu menggunakan two-stage splitter yang sama.
    Ini disengaja (lihat Design Spec D-A1).
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    header_chunks = header_splitter.split_text(content)

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHUNK_SEPARATORS,
    )
    final_chunks = size_splitter.split_documents(header_chunks)

    result = []
    for chunk in final_chunks:
        chunk_metadata = {
            "doc_id":       base_metadata["doc_id"],
            "title":        base_metadata["title"],
            "category":     base_metadata["category"],
            "content_type": base_metadata["content_type"],
            "chunk_type":   "content",
            "bab":          chunk.metadata.get("bab", ""),
            "bagian": chunk.metadata.get("bagian", ""),
            "pasal":        chunk.metadata.get("pasal", ""),
            "priority":     base_metadata.get("priority", 3),
            "status":       base_metadata.get("status", "active"),
        }
        result.append({
            "content":  chunk.page_content,
            "metadata": chunk_metadata,
        })

    return result


def get_chroma_collection(
    chroma_dir: Path,
    collection_name: str,
    rebuild: bool,
) -> chromadb.Collection:
    """Inisialisasi atau buka ChromaDB collection. Hapus jika rebuild=True."""
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if rebuild:
        try:
            client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' dihapus (--rebuild).")
        except Exception:
            pass  # Collection belum ada — tidak masalah

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": CHROMA_DISTANCE_FN},
    )
    return collection


def run_ingestion(
    chunk_size: int,
    chunk_overlap: int,
    chroma_dir: Path,
    collection_name: str,
    rebuild: bool,
) -> None:
    """Jalankan pipeline ingestion lengkap untuk satu konfigurasi chunking."""
    logger.info("=== Memulai Ingestion ===")
    logger.info(f"Collection : {collection_name}")
    logger.info(f"Chunk size : {chunk_size} | Overlap : {chunk_overlap}")
    logger.info(f"Rebuild    : {rebuild}")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        task_type="retrieval_document",
        google_api_key=GOOGLE_API_KEY,
    )

    chroma_dir.mkdir(parents=True, exist_ok=True)
    collection = get_chroma_collection(chroma_dir, collection_name, rebuild)

    md_files = sorted(CORPUS_DIR.glob("*.md"))
    if not md_files:
        logger.warning(f"Tidak ada file .md di {CORPUS_DIR}")
        return

    total_inserted  = 0
    total_duplicate = 0
    total_short     = 0

    for md_file in md_files:
        logger.info(f"\nMemproses: {md_file.name}")
        t_start = time.time()

        try:
            metadata, content = parse_markdown_file(md_file)
        except ValueError as e:
            logger.warning(f"SKIP {md_file.name}: {e}")
            log_system_event("warning", "INGEST_PARSE", f"Gagal parse {md_file.name}: {e}")
            continue

        all_chunks = [create_summary_chunk(metadata)]
        all_chunks.extend(
            split_content(content, metadata, chunk_size, chunk_overlap)
        )

        before = len(all_chunks)
        all_chunks = [
            c for c in all_chunks
            if len(c["content"].strip()) >= MIN_CHUNK_LENGTH
        ]
        skipped_short = before - len(all_chunks)
        total_short += skipped_short

        inserted = 0
        skipped_dup = 0
        for chunk in all_chunks:
            chunk_id = generate_chunk_id(metadata["doc_id"], chunk["content"])

            existing = collection.get(ids=[chunk_id])
            if existing["ids"]:
                skipped_dup += 1
                continue

            # Tambahkan jeda mikro 0.2 detik per chunk untuk meredam lonjakan RPM secara alami
            time.sleep(0.2)
            
            # Panggil embedding dengan retry policy tangguh terhadap rate limit gratis (429)
            embedding_vector = None
            for attempt in range(5):
                try:
                    embedding_vector = embeddings.embed_query(chunk["content"])
                    break
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower() or "limit" in str(e).lower():
                        sleep_time = 10 * (attempt + 1)
                        logger.warning(
                            f"  [API Limit 429] Terkena rate limit embedding pada chunk. "
                            f"Menunda {sleep_time} detik (perc. {attempt + 1}/5)..."
                        )
                        log_system_event(
                            "warning", "INGEST_API",
                            f"API Limit 429 terdeteksi saat embedding. Sleep {sleep_time}s."
                        )
                        time.sleep(sleep_time)
                    else:
                        raise e
            
            if embedding_vector is None:
                raise RuntimeError("Gagal mendapatkan embedding setelah 5 kali percobaan akibat batas kuota.")

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding_vector],
                documents=[chunk["content"]],
                metadatas=[chunk["metadata"]],
            )
            inserted += 1

        total_inserted  += inserted
        total_duplicate += skipped_dup
        duration = time.time() - t_start
        
        # Catat event per berkas
        log_ingestion_event(
            config="a" if chunk_size == 500 else "b",
            file_processed=md_file.name,
            chunks_generated=len(all_chunks) + skipped_short,
            chunks_inserted=inserted,
            chunks_duplicate_skipped=skipped_dup,
            execution_time_seconds=duration
        )
        
        logger.info(
            f"  ✓ {inserted} di-insert | "
            f"{skipped_dup} duplikat | {skipped_short} terlalu pendek"
        )

    logger.info("\n=== Ingestion Selesai ===")
    logger.info(f"Total di-insert     : {total_inserted}")
    logger.info(f"Total duplikat      : {total_duplicate}")
    logger.info(f"Total terlalu pendek: {total_short}")
    logger.info(f"Total di database   : {collection.count()}")


def main():
    """Entry point CLI untuk pipeline ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingestion pipeline untuk UNSRAT RAG"
    )
    parser.add_argument(
        "--config",
        choices=["a", "b"],
        required=True,
        help="Pilih konfigurasi: 'a' (500 char) atau 'b' (2000 char)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Hapus database lama dan bangun ulang dari nol",
    )
    args = parser.parse_args()

    if args.config == "a":
        run_ingestion(
            chunk_size=CHUNK_SIZE_A,
            chunk_overlap=CHUNK_OVERLAP_A,
            chroma_dir=CHROMA_DIR_A,
            collection_name=CHROMA_COLLECTION_A,
            rebuild=args.rebuild,
        )
    else:
        run_ingestion(
            chunk_size=CHUNK_SIZE_B,
            chunk_overlap=CHUNK_OVERLAP_B,
            chroma_dir=CHROMA_DIR_B,
            collection_name=CHROMA_COLLECTION_B,
            rebuild=args.rebuild,
        )


if __name__ == "__main__":
    main()
