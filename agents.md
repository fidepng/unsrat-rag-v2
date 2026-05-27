# AGENTS.md — Panduan Implementasi untuk AI Coding Agent

## Sistem Chatbot Informasi Akademik UNSRAT (RAG)

> **Untuk siapa file ini:** AI coding agent (Claude Code, Copilot, dsb.)
> dan developer pembantu yang akan mengimplementasikan proyek ini.
>
> **Sumber kebenaran:** `prd+srs.md` — baca itu dulu sebelum file ini.
> File ini adalah panduan implementasi step-by-step. `prd+srs.md` adalah
> spesifikasinya.

---

## ATURAN EMAS (BACA SEBELUM APAPUN)

```
1. SATU FILE = SATU TANGGUNG JAWAB. Jangan campurkan logika RAG ke app.py.
2. SEMUA ANGKA dari config.py. Tidak ada magic number di file lain.
3. SEMUA API KEY dari .env. Tidak pernah hardcode.
4. SETIAP FUNGSI punya docstring.
5. BAHASA ERROR = Bahasa Indonesia (untuk yang ditampilkan ke pengguna).
6. JANGAN tambah fitur yang tidak ada di prd+srs.md.
7. TANYA sebelum mengasumsikan sesuatu yang tidak jelas.
8. MINTA DOKUMENTASI sebelum mengimplementasikan library yang tidak
   100% familiar. Jangan asumsikan nama class/fungsi/import — terutama
   untuk library yang sering berubah API-nya (Ragas, LangChain, dsb.).
   Katakan kepada user: "Saya butuh link dokumentasi resmi [LIBRARY]
   versi [X.X] sebelum lanjut." Ini mencegah halusinasi dan bad practice.
```

---

## PETA DEPENDENSI FILE

Urutan ini penting. File di bawah bergantung pada file di atas:

```
.env
  └── src/config.py          ← Semua file bergantung padanya
        ├── src/ingestion.py  ← Tidak bergantung pada file src/ lain
        ├── src/retriever.py  ← Bergantung pada config.py
        └── src/chain.py      ← Bergantung pada config.py + retriever.py
              ├── app.py       ← Bergantung pada chain.py (BUKAN retriever.py langsung)
              └── evaluation.py ← Bergantung pada chain.py
```

**Implikasi:** Implementasikan dalam urutan ini:

1. `environment.yml` dan `.gitignore`
2. `src/config.py`
3. `src/ingestion.py`
4. `src/retriever.py`
5. `src/chain.py`
6. `app.py`
7. `evaluation.py`

---

## FASE 0 — SETUP PROJECT

### File: `environment.yml`

```yaml
name: unsrat-rag
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
      - langchain>=0.3.0
      - langchain-google-genai>=2.0.0
      - langchain-chroma>=0.1.0
      - langchain-community>=0.3.0
      - langchain-core>=0.3.0
      - chromadb>=0.5.0
      - ragas>=stable
      - streamlit>=1.40.0
      - python-dotenv>=1.0.0
      - python-frontmatter>=1.0.0
      - pandas>=2.0.0
      - pyyaml>=6.0
      - matplotlib>=3.8.0
      - google-generativeai>=0.8.0
      - rank-bm25>=0.2.2 # Config C: BM25 pure keyword search baseline
      - scipy>=1.11.0 # Uji signifikansi Wilcoxon signed-rank
```

### File: `.gitignore`

```
.env
chroma_db/
__pycache__/
*.pyc
.conda/
*.egg-info/
.DS_Store
```

### File: `src/__init__.py`

```python
# File kosong — wajib ada agar Python mengenali src/ sebagai package
```

---

## FASE 1 — `src/config.py`

Ini file pertama yang dikerjakan. Semua konstanta ada di sini.

**Skeleton lengkap:**

```python
"""
src/config.py
Pusat konfigurasi sistem. Semua parameter dan konstanta ada di sini.
Tidak ada magic number di file lain — selalu impor dari sini.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── PATH ────────────────────────────────────────────────────
ROOT_DIR          = Path(__file__).parent.parent
CORPUS_DIR        = ROOT_DIR / "data" / "corpus"
CHROMA_BASE_DIR   = ROOT_DIR / "chroma_db"
CHROMA_DIR_A      = CHROMA_BASE_DIR / "config_a"
CHROMA_DIR_B      = CHROMA_BASE_DIR / "config_b"
BM25_INDEX_DIR    = ROOT_DIR / "bm25_index"          # Config C
BM25_INDEX_PATH   = BM25_INDEX_DIR / "bm25_index.pkl"
EVAL_DATASET_PATH = ROOT_DIR / "eval" / "dataset" / "ground_truth.csv"
EVAL_RESULTS_DIR  = ROOT_DIR / "eval" / "results"

# ── CHROMADB COLLECTIONS ────────────────────────────────────
CHROMA_COLLECTION_A = "unsrat_rag_config_a"
CHROMA_COLLECTION_B = "unsrat_rag_config_b"
CHROMA_DISTANCE_FN  = "cosine"

# ── MODEL ───────────────────────────────────────────────────
# Isi nama model sesuai yang tersedia di akun GCP Anda.
# Generator dan evaluator HARUS BERBEDA (mitigasi self-eval bias — D-16).
LLM_MODEL_NAME       = "gemini-3.5-flash"
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EVALUATOR_MODEL_NAME = "gemini-3.1-pro-preview"

# Daftar model yang bisa dipilih di UI (sidebar).
# Isi dengan nama model yang tersedia di akun GCP Anda.
AVAILABLE_MODELS: list[str] = [
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]

# ── CHUNKING — CONFIG A (Eksperimen) ─────────────────────────
CHUNK_SIZE_A    = 500
CHUNK_OVERLAP_A = 100

# ── CHUNKING — CONFIG B (Best Practice / Aplikasi Utama) ─────
CHUNK_SIZE_B    = 2000
CHUNK_OVERLAP_B = 200

# ── SEPARATORS (Sama untuk Config A & B) ─────────────────────
CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]

# ── RETRIEVAL ────────────────────────────────────────────────
RETRIEVAL_K          = 4
SIMILARITY_THRESHOLD = 0.65
MIN_CHUNK_LENGTH     = 50

# ── BM25 — CONFIG C ──────────────────────────────────────────
BM25_K               = 4      # Jumlah dokumen BM25 yang dikembalikan per query
BM25_MIN_TOKEN_LEN   = 2      # Panjang minimum token (filter stopword kasar)

# ── LLM GENERATION ──────────────────────────────────────────
LLM_TEMPERATURE       = 0.1
LLM_MAX_OUTPUT_TOKENS = 800
LLM_TOP_P             = 0.95

# ── MEMORI ──────────────────────────────────────────────────
MEMORY_K = 5

# ── RETRY POLICY ─────────────────────────────────────────────
MAX_RETRIES  = 3
RETRY_DELAYS = [2, 5]

# ── EVALUASI ─────────────────────────────────────────────────
# Kolom metrik Ragas yang diukur — urutan ini dipakai di chart & CSV.
METRICS_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    # "context_entity_recall",   # Opsional — aktifkan jika diperlukan
]
# Jumlah sampel terburuk untuk analisis kegagalan kualitatif
ERROR_ANALYSIS_N = 10

# ── CATEGORY DETECTION KEYWORDS ──────────────────────────────
CATEGORY_KEYWORDS: dict = {
    "academic": [
        "SKS", "KRS", "IPK", "IPS", "UTS", "UAS", "DO", "drop out",
        "pasal", "peraturan", "yudisium", "cuti", "mata kuliah",
        "nilai", "ujian", "wisuda", "ijazah", "skripsi", "sidang",
        "kurikulum", "beban belajar", "semester"
    ],
    "calendar": [
        "jadwal", "tanggal", "kapan", "batas", "deadline", "registrasi",
        "pengisian", "periode", "libur", "minggu ke", "kalender",
        "perkuliahan dimulai", "uts dimulai", "uas dimulai"
    ],
    "institution_profile": [
        "sejarah", "visi", "misi", "lambang", "bendera", "mars",
        "hymne", "akreditasi", "rektor", "berdiri", "didirikan",
        "profil universitas"
    ],
    "faq": [
        "FAQ", "pertanyaan umum", "sering ditanya"
    ],
}

# ── SYSTEM PROMPT (TERKUNCI) ─────────────────────────────────
SYSTEM_PROMPT = """Anda adalah agen asisten informasi akademik resmi \
Universitas Sam Ratulangi.
Tugas Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan dokumen \
konteks yang disediakan.

Jika jawaban tidak ada di dalam dokumen konteks, katakan secara jujur bahwa \
Anda tidak menemukan informasinya dan arahkan mereka untuk menghubungi \
bagian administrasi kampus terkait.

JANGAN PERNAH mengarang informasi, tanggal, atau angka SKS.
Jawab dalam Bahasa Indonesia yang ramah dan mudah dipahami."""

# ── FALLBACK RESPONSE ────────────────────────────────────────
FALLBACK_RESPONSE = (
    "Maaf, saya tidak menemukan informasi yang relevan mengenai "
    "pertanyaan Anda dalam dokumen yang tersedia. "
    "Untuk informasi lebih lanjut, silakan hubungi:\n\n"
    "• Bagian Akademik UNSRAT\n"
    "• Portal INSPIRE: inspire.unsrat.ac.id\n"
    "• Atau datang langsung ke Gedung Rektorat UNSRAT"
)

# ── API KEY ──────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY tidak ditemukan!\n"
        "Buat file .env di root proyek dan isi: GOOGLE_API_KEY=your_key_here"
    )
```

**Checkpoint:** Jalankan `python src/config.py` — harus tidak ada error
(kecuali jika `.env` belum dibuat, itu memang akan error — itu benar).

---

## FASE 2 — `src/ingestion.py`

Pipeline untuk membangun database ChromaDB dari file .md.

**Tugas file ini:**

1. Baca semua `.md` dari `CORPUS_DIR`
2. Parse YAML frontmatter
3. Buat summary chunk
4. Split konten menjadi chunk
5. Filter chunk terlalu pendek
6. Cek duplikat via hash
7. Embed dan simpan ke ChromaDB

**Skeleton:**

```python
"""
src/ingestion.py
Pipeline ingestion: baca .md → chunk → embed → simpan ke ChromaDB.
Jalankan via CLI: python src/ingestion.py --config [a|b] [--rebuild]
"""

import argparse
import hashlib
import logging
from pathlib import Path

import frontmatter
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    CORPUS_DIR, CHROMA_DIR_A, CHROMA_DIR_B,
    CHROMA_COLLECTION_A, CHROMA_COLLECTION_B, CHROMA_DISTANCE_FN,
    CHUNK_SIZE_A, CHUNK_OVERLAP_A, CHUNK_SIZE_B, CHUNK_OVERLAP_B,
    CHUNK_SEPARATORS, MIN_CHUNK_LENGTH,
    EMBEDDING_MODEL_NAME, GOOGLE_API_KEY,
)

# Setup logging — cetak progress ke terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Hierarchy heading untuk MarkdownHeaderTextSplitter
HEADERS_TO_SPLIT = [
    ("#",    "header_1"),
    ("##",   "bab"),
    ("###",  "bagian"),
    ("####", "pasal"),
]

# Field YAML yang wajib ada di setiap dokumen
REQUIRED_YAML_FIELDS = [
    "doc_id", "title", "category", "content_type",
    "valid_from", "status", "retrieval_summary", "chunk_strategy",
    "last_updated",
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

    # Validasi field wajib
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
    Tahap 1: MarkdownHeaderTextSplitter (struktural)
    Tahap 2: RecursiveCharacterTextSplitter (normalisasi ukuran)
    """
    # Tahap 1 — Split berdasarkan heading Markdown
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    header_chunks = header_splitter.split_text(content)

    # Tahap 2 — Normalisasi ukuran
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHUNK_SEPARATORS,
    )
    final_chunks = size_splitter.split_documents(header_chunks)

    # Konversi ke format dict standar dengan metadata lengkap
    result = []
    for chunk in final_chunks:
        chunk_metadata = {
            "doc_id":       base_metadata["doc_id"],
            "title":        base_metadata["title"],
            "category":     base_metadata["category"],
            "content_type": base_metadata["content_type"],
            "chunk_type":   "content",
            "bab":          chunk.metadata.get("bab", ""),
            "bagian":       chunk.metadata.get("bagian", ""),
            "pasal":        chunk.metadata.get("pasal", ""),
            "priority":     base_metadata.get("priority", 3),
            "status":       base_metadata.get("status", "active"),
        }
        result.append({
            "content": chunk.page_content,
            "metadata": chunk_metadata,
        })

    return result


def get_chroma_collection(
    chroma_dir: Path,
    collection_name: str,
    rebuild: bool,
) -> chromadb.Collection:
    """
    Inisialisasi atau buka ChromaDB collection.
    Jika rebuild=True, hapus collection lama dan buat baru.
    """
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if rebuild:
        try:
            client.delete_collection(collection_name)
            logger.info(f"Collection '{collection_name}' dihapus (--rebuild).")
        except Exception:
            pass   # Collection belum ada — tidak masalah

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
    """
    Jalankan pipeline ingestion lengkap untuk satu konfigurasi chunking.
    """
    logger.info(f"=== Memulai Ingestion ===")
    logger.info(f"Collection : {collection_name}")
    logger.info(f"Chunk size : {chunk_size} | Overlap : {chunk_overlap}")
    logger.info(f"Rebuild    : {rebuild}")

    # Siapkan embedding model (task_type WAJIB "retrieval_document")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        task_type="retrieval_document",
        google_api_key=GOOGLE_API_KEY,
    )

    # Siapkan ChromaDB collection
    chroma_dir.mkdir(parents=True, exist_ok=True)
    collection = get_chroma_collection(chroma_dir, collection_name, rebuild)

    # Ambil semua file .md
    md_files = sorted(CORPUS_DIR.glob("*.md"))
    if not md_files:
        logger.warning(f"Tidak ada file .md di {CORPUS_DIR}")
        return

    total_inserted = 0
    total_skipped_duplicate = 0
    total_skipped_short = 0

    for md_file in md_files:
        logger.info(f"\nMemproses: {md_file.name}")

        # Parse frontmatter — skip jika error
        try:
            metadata, content = parse_markdown_file(md_file)
        except ValueError as e:
            logger.warning(f"SKIP {md_file.name}: {e}")
            continue

        # Kumpulkan semua chunk (summary + content)
        all_chunks = [create_summary_chunk(metadata)]
        all_chunks.extend(
            split_content(content, metadata, chunk_size, chunk_overlap)
        )

        # Filter chunk terlalu pendek
        before_filter = len(all_chunks)
        all_chunks = [
            c for c in all_chunks
            if len(c["content"].strip()) >= MIN_CHUNK_LENGTH
        ]
        skipped_short = before_filter - len(all_chunks)
        total_skipped_short += skipped_short

        # Insert ke ChromaDB (cek duplikat via hash)
        inserted = 0
        skipped_dup = 0
        for chunk in all_chunks:
            chunk_id = generate_chunk_id(
                metadata["doc_id"], chunk["content"]
            )

            # Cek apakah chunk sudah ada
            existing = collection.get(ids=[chunk_id])
            if existing["ids"]:
                skipped_dup += 1
                continue

            # Embed dan insert
            embedding_vector = embeddings.embed_query(chunk["content"])
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding_vector],
                documents=[chunk["content"]],
                metadatas=[chunk["metadata"]],
            )
            inserted += 1

        total_inserted += inserted
        total_skipped_duplicate += skipped_dup
        logger.info(
            f"  ✓ {inserted} chunk di-insert | "
            f"{skipped_dup} duplikat | {skipped_short} terlalu pendek"
        )

    logger.info(f"\n=== Ingestion Selesai ===")
    logger.info(f"Total di-insert    : {total_inserted}")
    logger.info(f"Total duplikat     : {total_skipped_duplicate}")
    logger.info(f"Total terlalu pendek: {total_skipped_short}")
    logger.info(f"Total di database  : {collection.count()}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingestion pipeline untuk UNSRAT RAG"
    )
    parser.add_argument(
        "--config",
        choices=["a", "b"],
        required=True,
        help="Pilih konfigurasi chunking: 'a' (500 char) atau 'b' (2000 char)",
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
```

**Checkpoint:** `python src/ingestion.py --config b --rebuild`
Harus mencetak log per file dan total chunk tanpa error.

---

## FASE 2.5 — `src/bm25_retriever.py`

BM25 Config C: keyword search murni sebagai baseline komparasi.
File ini mandiri — tidak bergantung pada ChromaDB atau embedding.

> ⚠️ **Sebelum coding:** Jalankan `pip show rank-bm25` dan baca README
> resminya di https://github.com/dorianbrown/rank_bm25 jika perlu.
> Jika link tidak tersedia, minta user untuk memberikannya.

```python
"""
src/bm25_retriever.py
Config C: BM25 pure keyword search — tanpa embedding, tanpa ChromaDB.
Membangun indeks dari corpus .md yang sama dengan Config A & B.
Digunakan sebagai baseline komparasi dalam evaluasi.

CLI: python src/bm25_retriever.py --rebuild
"""

import argparse
import logging
import pickle
from pathlib import Path

import frontmatter
from rank_bm25 import BM25Okapi

from src.config import (
    CORPUS_DIR, BM25_INDEX_DIR, BM25_INDEX_PATH,
    BM25_K, BM25_MIN_TOKEN_LEN,
    REQUIRED_YAML_FIELDS,  # Impor dari ingestion atau definisikan ulang di sini
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Field YAML wajib (harus konsisten dengan ingestion.py)
_REQUIRED_YAML_FIELDS = [
    "doc_id", "title", "category", "content_type",
    "valid_from", "status", "retrieval_summary",
    "chunk_strategy", "last_updated",
]


def _tokenize(text: str) -> list[str]:
    """
    Tokenisasi sederhana: lowercase, split whitespace, filter token pendek.
    Tidak menggunakan NLTK atau stopword eksternal untuk menjaga dependensi minimal.
    """
    tokens = text.lower().split()
    return [t for t in tokens if len(t) >= BM25_MIN_TOKEN_LEN]


def build_index() -> None:
    """
    Bangun indeks BM25 dari semua file .md di CORPUS_DIR.
    Simpan indeks + metadata corpus ke BM25_INDEX_PATH (pickle).
    """
    md_files = sorted(CORPUS_DIR.glob("*.md"))
    if not md_files:
        logger.warning(f"Tidak ada file .md di {CORPUS_DIR}")
        return

    corpus_texts: list[str] = []
    corpus_metadata: list[dict] = []

    for md_file in md_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)
        except Exception as e:
            logger.warning(f"SKIP {md_file.name}: {e}")
            continue

        meta = dict(post.metadata)
        content = post.content

        # Validasi field wajib
        for field in _REQUIRED_YAML_FIELDS:
            if not meta.get(field):
                logger.warning(f"SKIP {md_file.name}: field '{field}' kosong")
                break
        else:
            # Gunakan seluruh konten sebagai satu "dokumen" BM25
            # (BM25 tidak memotong chunk — ini sengaja, untuk melihat dampaknya)
            full_text = f"{meta.get('retrieval_summary', '')} {content}"
            corpus_texts.append(full_text)
            corpus_metadata.append({
                "doc_id":   meta.get("doc_id", ""),
                "title":    meta.get("title", ""),
                "category": meta.get("category", ""),
                "preview":  content[:150],
            })
            logger.info(f"  ✓ {md_file.name} diindeks ({len(full_text)} karakter)")

    if not corpus_texts:
        logger.error("Tidak ada dokumen yang berhasil diindeks.")
        return

    tokenized_corpus = [_tokenize(text) for text in corpus_texts]
    bm25_index = BM25Okapi(tokenized_corpus)

    BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(
            {"index": bm25_index, "texts": corpus_texts, "metadata": corpus_metadata},
            f,
        )
    logger.info(f"Indeks BM25 disimpan ke: {BM25_INDEX_PATH} ({len(corpus_texts)} dokumen)")


def _load_index() -> dict:
    """
    Muat indeks BM25 dari disk. Raises RuntimeError jika belum dibangun.
    """
    if not BM25_INDEX_PATH.exists():
        raise RuntimeError(
            f"Indeks BM25 belum ada di {BM25_INDEX_PATH}!\n"
            "Jalankan: python src/bm25_retriever.py --rebuild"
        )
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


# Cache indeks agar tidak dibaca ulang setiap query
_cache: dict | None = None


def retrieve_chunks_bm25(query: str) -> list[dict]:
    """
    Cari BM25_K dokumen paling relevan untuk query.

    Returns:
        List of dict dengan key: content, metadata, distance (skor BM25 ternormalisasi).
        Format kompatibel dengan output retrieve_chunks() di retriever.py.
    """
    global _cache
    if _cache is None:
        _cache = _load_index()

    bm25: BM25Okapi = _cache["index"]
    texts: list[str] = _cache["texts"]
    metadata: list[dict] = _cache["metadata"]

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)

    # Ambil top-K indeks (skor tertinggi)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_K]

    results = []
    max_score = max(scores[i] for i in top_indices) if top_indices else 1.0
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        # Normalisasi: konversi ke "distance" (0=relevan, 1=tidak relevan)
        normalized_distance = 1.0 - (scores[idx] / max_score) if max_score > 0 else 1.0
        results.append({
            "content":  texts[idx],
            "metadata": metadata[idx],
            "distance": round(normalized_distance, 4),
        })

    return results


def main():
    """Bangun ulang indeks BM25 via CLI."""
    parser = argparse.ArgumentParser(description="Bangun indeks BM25 untuk Config C")
    parser.add_argument("--rebuild", action="store_true", required=True,
                        help="Hapus dan bangun ulang indeks BM25")
    args = parser.parse_args()

    if args.rebuild:
        if BM25_INDEX_PATH.exists():
            BM25_INDEX_PATH.unlink()
            logger.info("Indeks lama dihapus.")
        build_index()


if __name__ == "__main__":
    main()
```

**Checkpoint:**

```bash
python src/bm25_retriever.py --rebuild
# Harus mencetak log per file dan path output tanpa error
```

---

## FASE 3 — `src/retriever.py`

Logika pencarian chunk relevan dari ChromaDB.

```python
"""
src/retriever.py
Terima query → detect kategori → cari di ChromaDB → kembalikan chunks.
Dipanggil oleh src/chain.py, bukan langsung dari app.py.
"""

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    CHROMA_DIR_A, CHROMA_DIR_B,
    CHROMA_COLLECTION_A, CHROMA_COLLECTION_B, CHROMA_DISTANCE_FN,
    RETRIEVAL_K, SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL_NAME, GOOGLE_API_KEY,
    CATEGORY_KEYWORDS,
)

# ── Embedding model untuk query ──────────────────────────────────────────
# PENTING: task_type="retrieval_query" untuk query, BUKAN "retrieval_document"
_embedding_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    task_type="retrieval_query",
    google_api_key=GOOGLE_API_KEY,
)

# Cache ChromaDB client agar tidak buka koneksi baru setiap query
_chroma_clients: dict[str, chromadb.PersistentClient] = {}


def _get_collection(config: str) -> chromadb.Collection:
    """
    Ambil ChromaDB collection sesuai config yang dipilih.
    Raises RuntimeError jika database belum dibangun.
    """
    if config == "a":
        chroma_dir = CHROMA_DIR_A
        collection_name = CHROMA_COLLECTION_A
    else:
        chroma_dir = CHROMA_DIR_B
        collection_name = CHROMA_COLLECTION_B

    if not chroma_dir.exists():
        raise RuntimeError(
            f"Database ChromaDB untuk Config {config.upper()} belum ada!\n"
            f"Jalankan dulu: python src/ingestion.py --config {config} --rebuild"
        )

    if config not in _chroma_clients:
        _chroma_clients[config] = chromadb.PersistentClient(path=str(chroma_dir))

    return _chroma_clients[config].get_collection(collection_name)


def detect_category(query: str) -> str | None:
    """
    Deteksi kategori dari query menggunakan keyword matching.
    Kembalikan nama kategori dengan skor tertinggi, atau None.
    """
    query_lower = query.lower()
    scores: dict[str, int] = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in query_lower)
        if score > 0:
            scores[category] = score

    if not scores:
        return None
    return max(scores, key=scores.get)


def retrieve_chunks(query: str, config: str = "b") -> list[dict]:
    """
    Ambil chunk relevan dari ChromaDB untuk query yang diberikan.

    Args:
        query:  Pertanyaan dari pengguna.
        config: "a" atau "b" — pilih collection ChromaDB.

    Returns:
        List of dict, setiap dict berisi 'content' dan 'metadata'.
        List kosong jika tidak ada chunk yang lolos similarity threshold.
    """
    collection = _get_collection(config)

    # Embed query
    query_embedding = _embedding_model.embed_query(query)

    # Deteksi kategori untuk metadata pre-filter
    detected_category = detect_category(query)
    where_filter = None
    if detected_category:
        where_filter = {"category": detected_category}

    # Cari di ChromaDB
    # Ambil lebih banyak dari RETRIEVAL_K karena ada yang akan difilter threshold
    n_results = min(RETRIEVAL_K * 2, collection.count())
    if n_results == 0:
        return []

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_params["where"] = where_filter

    results = collection.query(**query_params)

    # Proses hasil — filter berdasarkan similarity threshold
    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Buang chunk yang terlalu jauh (tidak relevan)
        if distance > SIMILARITY_THRESHOLD:
            continue
        chunks.append({
            "content":  doc,
            "metadata": meta,
            "distance": distance,
        })

    # Trim ke RETRIEVAL_K (ambil yang terdekat/paling relevan)
    chunks = chunks[:RETRIEVAL_K]

    return chunks
```

---

## FASE 4 — `src/chain.py`

Inti dari sistem RAG. Gabungkan retriever + memory + LLM.

```python
"""
src/chain.py
Logika RAG: ambil chunk → gabung konteks → panggil LLM → kembalikan jawaban.
Mendukung dua mode: streaming (untuk UI) dan batch (untuk evaluasi).
"""

import time
from typing import Generator
from src.bm25_retriever import retrieve_chunks_bm25

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import (
    LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_OUTPUT_TOKENS, LLM_TOP_P,
    SYSTEM_PROMPT, FALLBACK_RESPONSE, MEMORY_K,
    MAX_RETRIES, RETRY_DELAYS, GOOGLE_API_KEY,
)
from src.retriever import retrieve_chunks

# ── Inisialisasi LLM ─────────────────────────────────────────────────────
_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    temperature=LLM_TEMPERATURE,
    max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
    top_p=LLM_TOP_P,
    google_api_key=GOOGLE_API_KEY,
)

def reinitialize_llm(model_name: str) -> None:
    """
    Reinisialisasi LLM dengan model berbeda tanpa restart server.
    Dipanggil oleh app.py saat pengguna mengganti model di sidebar.
    """
    global _llm
    _llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
        top_p=LLM_TOP_P,
        google_api_key=GOOGLE_API_KEY,
    )

# ── Template prompt ───────────────────────────────────────────────────────
_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Konteks dokumen yang relevan:\n{context}\n\nPertanyaan: {question}"),
])


def trim_history(history: list, k: int) -> list:
    """
    Trim chat history ke k giliran terakhir (= k*2 pesan).
    Satu giliran = 1 pesan user + 1 pesan AI.
    """
    return history[-(k * 2):]


def format_sources(chunks: list[dict]) -> list[dict]:
    """
    Konversi list chunk mentah ke format sumber yang siap ditampilkan di UI.
    """
    sources = []
    for chunk in chunks:
        meta = chunk["metadata"]
        sources.append({
            "doc_id":   meta.get("doc_id", ""),
            "title":    meta.get("title", ""),
            "bab":      meta.get("bab", "Ringkasan Dokumen"),
            "bagian":   meta.get("bagian", ""),
            "pasal":    meta.get("pasal", ""),
            "preview":  chunk["content"][:150],
            "category": meta.get("category", ""),
        })
    return sources


def _call_llm_with_retry(prompt_value) -> str:
    """
    Panggil LLM dengan exponential backoff retry.
    Raises RuntimeError setelah semua percobaan habis.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = _llm.invoke(prompt_value)
            return response.content
        except Exception as e:
            error_str = str(e)
            is_last = (attempt == MAX_RETRIES - 1)

            if is_last:
                raise RuntimeError(
                    f"Gagal menghubungi AI setelah {MAX_RETRIES} percobaan. "
                    f"Error: {error_str}"
                )

            delay = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else RETRY_DELAYS[-1]
            time.sleep(delay)

    raise RuntimeError("Tidak terduga: semua retry habis.")


def get_response(
    question: str,
    chat_history: list,
    config: str = "b",
    streaming: bool = False,
) -> dict | Generator:
    """
    Dapatkan respons RAG untuk pertanyaan yang diberikan.

    Args:
        question:     Pertanyaan dari pengguna.
        chat_history: List HumanMessage/AIMessage dari percakapan sebelumnya.
                      Kirimkan [] (list kosong) untuk evaluasi.
        config:       "a", "b" (ChromaDB/vector), atau "c" (BM25 keyword search).
        streaming:    True untuk UI (kembalikan Generator).
                      False untuk evaluasi (kembalikan dict).

    Returns:
        Jika streaming=False:
            {
                "answer":  str,          # Teks jawaban
                "sources": List[dict],   # List sumber chunk
                "found":   bool,         # False jika tidak ada chunk relevan
            }
        Jika streaming=True:
            Generator yang yield token satu per satu.
            CATATAN: sources dan found tidak tersedia dalam mode streaming.
            Panggil get_response(..., streaming=False) jika butuh sources.
    """
    # 1. Ambil chunk relevan sesuai config
    if config == "c":
        chunks = retrieve_chunks_bm25(question)
    else:
        chunks = retrieve_chunks(question, config=config)

    # 2. Jika tidak ada chunk relevan → kembalikan fallback TANPA panggil LLM
    if not chunks:
        if streaming:
            def _fallback_gen():
                yield FALLBACK_RESPONSE
            return _fallback_gen()
        return {
            "answer":  FALLBACK_RESPONSE,
            "sources": [],
            "found":   False,
        }

    # 3. Siapkan konteks dari chunk
    context = "\n\n---\n\n".join(
        [f"[{c['metadata'].get('title','')} — {c['metadata'].get('bab','')}]\n{c['content']}"
         for c in chunks]
    )

    # 4. Trim history ke MEMORY_K giliran
    trimmed_history = trim_history(chat_history, MEMORY_K)

    # 5. Format prompt
    prompt_value = _prompt.format_messages(
        chat_history=trimmed_history,
        context=context,
        question=question,
    )

    # 6. Panggil LLM
    if streaming:
        def _stream_gen():
            for chunk_token in _llm.stream(prompt_value):
                yield chunk_token.content
        return _stream_gen()
    else:
        answer = _call_llm_with_retry(prompt_value)
        return {
            "answer":  answer,
            "sources": format_sources(chunks),
            "found":   True,
        }
```

**Checkpoint:** Buat file `test_chain.py` sementara di root:

```python
from langchain_core.messages import HumanMessage, AIMessage
from src.chain import get_response

result = get_response(
    question="Apa itu SKS?",
    chat_history=[],
    config="b",
    streaming=False,
)
print("Answer:", result["answer"][:200])
print("Found:", result["found"])
print("Sources:", len(result["sources"]))
```

Hapus file ini setelah chain berjalan dengan benar.

---

## FASE 5 — `app.py`

UI Streamlit. Hanya memanggil fungsi dari `src/chain.py`.

```python
"""
app.py
Antarmuka pengguna berbasis Streamlit.
ATURAN: File ini HANYA berisi kode UI.
Semua logika RAG ada di src/chain.py dan src/retriever.py.
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.config import MEMORY_K, AVAILABLE_MODELS

from src.chain import get_response
from src.config import MEMORY_K

# ── Konstanta key session state ───────────────────────────────────────────
SESSION_MESSAGES      = "messages"       # List[dict] untuk tampilan gelembung chat
SESSION_CHAT_HISTORY  = "chat_history"   # List[HumanMessage/AIMessage] untuk LangChain
SESSION_ACTIVE_CONFIG = "active_config"  # "a" atau "b"

# ── Inisialisasi session state ────────────────────────────────────────────
if SESSION_MESSAGES not in st.session_state:
    st.session_state[SESSION_MESSAGES] = []
if SESSION_CHAT_HISTORY not in st.session_state:
    st.session_state[SESSION_CHAT_HISTORY] = []
if SESSION_ACTIVE_CONFIG not in st.session_state:
    st.session_state[SESSION_ACTIVE_CONFIG] = "b"   # Default: Config B

# ── Layout utama ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UNSRAT RAG Chatbot",
    page_icon="🎓",
    layout="wide",
)

tab_chat, tab_eval = st.tabs(["💬 Chatbot", "📊 Evaluasi Ragas"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — CHATBOT
# ══════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.title("🎓 Asisten Informasi Akademik UNSRAT")
    st.caption("Didukung oleh arsitektur RAG + Google Gemini")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pengaturan")

        selected_config = st.radio(
            "Konfigurasi Retrieval:",
            options=["b", "a", "c"],
            format_func=lambda x: {
                "b": "Config B (RAG 2000 char) — Default",
                "a": "Config A (RAG 500 char) — Eksperimen",
                "c": "Config C (BM25 Keyword) — Baseline",
            }[x],
            index=0,
        )
        st.session_state[SESSION_ACTIVE_CONFIG] = selected_config

        st.divider()

        # Model switcher — isi AVAILABLE_MODELS di config.py sebelum pakai ini
        selected_model = st.selectbox(
            "Model LLM:",
            options=AVAILABLE_MODELS,
            index=0,
            key="selected_model",
        )
        # Reinisialisasi LLM jika model berubah
        if st.session_state.get("_last_model") != selected_model:
            from src.chain import reinitialize_llm
            reinitialize_llm(selected_model)
            st.session_state["_last_model"] = selected_model

        st.divider()

        if st.button("🔄 Reset Percakapan", use_container_width=True):
            st.session_state[SESSION_MESSAGES] = []
            st.session_state[SESSION_CHAT_HISTORY] = []
            st.rerun()

        st.caption(
            f"Config: **{selected_config.upper()}** | "
            f"Model: **{selected_model.split('-')[0]}...**"
        )

    # Tampilkan riwayat chat
    for message in st.session_state[SESSION_MESSAGES]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Tampilkan sumber jika ada
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander(f"📚 Lihat Sumber ({len(message['sources'])} referensi)"):
                    for i, src in enumerate(message["sources"]):
                        st.markdown(f"**[{i+1}] {src['title']} — {src['bab']}**")
                        st.text(src["preview"])

    # Input pengguna
    if user_input := st.chat_input("Tanyakan sesuatu tentang akademik UNSRAT..."):
        # Tampilkan pesan pengguna
        st.session_state[SESSION_MESSAGES].append({
            "role": "user",
            "content": user_input,
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        # Dapatkan respons (streaming untuk UX, batch untuk sumber)
        with st.chat_message("assistant"):
            try:
                # Streaming untuk tampilan efek mengetik
                stream = get_response(
                    question=user_input,
                    chat_history=st.session_state[SESSION_CHAT_HISTORY],
                    config=st.session_state[SESSION_ACTIVE_CONFIG],
                    streaming=True,
                )
                response_text = st.write_stream(stream)

                # Dapatkan sumber (panggil non-streaming untuk metadata)
                batch_result = get_response(
                    question=user_input,
                    chat_history=st.session_state[SESSION_CHAT_HISTORY],
                    config=st.session_state[SESSION_ACTIVE_CONFIG],
                    streaming=False,
                )
                sources = batch_result.get("sources", [])

                if sources:
                    with st.expander(f"📚 Lihat Sumber ({len(sources)} referensi)"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**[{i+1}] {src['title']} — {src['bab']}**")
                            st.text(src["preview"])

            except RuntimeError as e:
                response_text = str(e)
                sources = []
                st.error(response_text)

        # Simpan ke history display
        st.session_state[SESSION_MESSAGES].append({
            "role": "assistant",
            "content": response_text,
            "sources": sources,
        })

        # Update LangChain chat history (untuk context window berikutnya)
        st.session_state[SESSION_CHAT_HISTORY].append(
            HumanMessage(content=user_input)
        )
        st.session_state[SESSION_CHAT_HISTORY].append(
            AIMessage(content=response_text)
        )

    # Disclaimer wajib
    st.caption(
        "⚠️ Chatbot ini berbasis AI dan mungkin membuat kesalahan. "
        "Verifikasi informasi penting ke Bagian Akademik Kampus."
    )

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUASI RAGAS
# ══════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.header("📊 Hasil Evaluasi Ragas")
    st.caption("Jalankan `python evaluation.py --config [a|b|c]` untuk menghasilkan data ini.")

    import pandas as pd
    from src.config import EVAL_RESULTS_DIR, METRICS_COLS

    configs = {
        "a": ("Config A (RAG 500 char)", EVAL_RESULTS_DIR / "hasil_config_a.csv"),
        "b": ("Config B (RAG 2000 char)", EVAL_RESULTS_DIR / "hasil_config_b.csv"),
        "c": ("Config C (BM25 Keyword)", EVAL_RESULTS_DIR / "hasil_config_c.csv"),
    }

    cols = st.columns(3)
    for col, (key, (label, path)) in zip(cols, configs.items()):
        with col:
            st.subheader(label)
            if path.exists():
                df = pd.read_csv(path)
                display_cols = [m for m in METRICS_COLS if m in df.columns]
                if "response_time_seconds" in df.columns:
                    display_cols.append("response_time_seconds")
                if display_cols:
                    st.dataframe(df[display_cols].describe().round(3))
            else:
                st.info(f"Belum ada hasil.\n`python evaluation.py --config {key}`")

    # Tabel uji Wilcoxon
    stats_path = EVAL_RESULTS_DIR / "statistical_test.csv"
    if stats_path.exists():
        st.subheader("🔬 Uji Signifikansi Wilcoxon (Config A vs B)")
        st.dataframe(pd.read_csv(stats_path).round(4))
    else:
        st.info("Belum ada hasil statistik. `python evaluation.py --stats`")

    # Chart perbandingan
    path_chart = EVAL_RESULTS_DIR / "perbandingan_visual.png"
    if path_chart.exists():
        st.subheader("Perbandingan Visual")
        st.image(str(path_chart))
    else:
        st.info("Belum ada chart. `python evaluation.py --visualize`")
```

---

## FASE 6 — `evaluation.py`

Pipeline evaluasi Ragas via CLI.

```python
"""
evaluation.py
Pipeline evaluasi Ragas + latensi + uji statistik + analisis kegagalan.

CLI:
  python evaluation.py --config [a|b|c]
  python evaluation.py --stats
  python evaluation.py --visualize
"""

import argparse
import logging
import time
import pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.chain import get_response
from src.config import (
    EVAL_DATASET_PATH, EVAL_RESULTS_DIR,
    EVALUATOR_MODEL_NAME, GOOGLE_API_KEY,
    METRICS_COLS, ERROR_ANALYSIS_N,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── CATATAN UNTUK IMPLEMENTOR ─────────────────────────────────────────────
# Import metrik Ragas di bawah WAJIB diverifikasi dengan dokumentasi
# resmi versi yang terinstall (`pip show ragas`).
# Jika nama import berbeda (misal: class-based vs instance), minta user
# untuk memberikan link docs: https://docs.ragas.io/en/v{VERSI}/
# Jangan hardcode import tanpa verifikasi!
# ─────────────────────────────────────────────────────────────────────────


def run_evaluation(config: str) -> None:
    """
    Jalankan evaluasi lengkap untuk satu konfigurasi:
    Ragas metrics + response_time + error analysis export.
    """
    from ragas import evaluate
    # ⚠️ Verifikasi nama import berikut dengan docs Ragas versi terinstall:
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    # from ragas.metrics import context_recall   # Tambahkan setelah verifikasi nama import
    from ragas.llms import LangchainLLMWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI
    from datasets import Dataset

    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"File ground truth tidak ditemukan: {EVAL_DATASET_PATH}\n"
            "Buat file eval/dataset/ground_truth.csv terlebih dahulu."
        )

    df = pd.read_csv(EVAL_DATASET_PATH, encoding="utf-8")
    logger.info(f"Memuat {len(df)} pertanyaan dari ground truth.")

    eval_data = {
        "user_input": [],
        "reference": [],
        "response": [],
        "retrieved_contexts": [],
        "response_time_seconds": [],   # ← BARU: dimensi latensi
    }

    for idx, row in df.iterrows():
        logger.info(f"[{idx+1}/{len(df)}] {row['user_input'][:60]}...")

        # KRITIS: Reset memory + ukur latensi
        t_start = time.time()
        result = get_response(
            question=row["user_input"],
            chat_history=[],
            config=config,
            streaming=False,
        )
        response_time = time.time() - t_start

        eval_data["user_input"].append(row["user_input"])
        eval_data["reference"].append(row["reference"])
        eval_data["response"].append(result["answer"])
        eval_data["retrieved_contexts"].append(
            [s["preview"] for s in result["sources"]] if result["sources"] else [""]
        )
        eval_data["response_time_seconds"].append(round(response_time, 3))

    # Siapkan LLM evaluator (BERBEDA dari generator — D-16)
    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=EVALUATOR_MODEL_NAME,   # ← Beda model dari LLM_MODEL_NAME
            google_api_key=GOOGLE_API_KEY,
        )
    )

    # Jalankan Ragas
    # ⚠️ Sesuaikan daftar metrik dengan yang berhasil diimport di atas
    logger.info("Menjalankan evaluasi Ragas...")
    ragas_cols = [m for m in METRICS_COLS if m != "response_time_seconds"]
    dataset_for_ragas = {k: v for k, v in eval_data.items()
                         if k != "response_time_seconds"}
    dataset = Dataset.from_dict(dataset_for_ragas)
    result_ragas = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],  # ← update sesuai import
        llm=evaluator_llm,
    )

    # Gabungkan hasil Ragas + response_time
    result_df = result_ragas.to_pandas()
    result_df["response_time_seconds"] = eval_data["response_time_seconds"]

    # Simpan ke CSV
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_RESULTS_DIR / f"hasil_config_{config}.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Hasil disimpan ke: {output_path}")

    # Cetak ringkasan
    print(f"\n{'='*55}")
    print(f"RINGKASAN EVALUASI — CONFIG {config.upper()}")
    print(f"{'='*55}")
    for col in METRICS_COLS:
        if col in result_df.columns:
            m = result_df[col]
            print(f"{col:30s}: {m.mean():.3f} ± {m.std():.3f} "
                  f"(min: {m.min():.3f}, max: {m.max():.3f})")
    if "response_time_seconds" in result_df.columns:
        rt = result_df["response_time_seconds"]
        print(f"{'response_time_seconds':30s}: {rt.mean():.2f} ± {rt.std():.2f} detik")
    print(f"{'='*55}\n")

    # Generate error analysis (hanya untuk Config A & B, bukan C)
    if config in ("a", "b"):
        _generate_error_analysis(result_df, config)


def _generate_error_analysis(result_df: pd.DataFrame, config: str) -> None:
    """
    Ekspor ERROR_ANALYSIS_N sampel dengan skor rata-rata Ragas terendah.
    Kolom failure_type diisi manual oleh peneliti setelah file dihasilkan.
    """
    available_metrics = [m for m in METRICS_COLS if m in result_df.columns]
    if not available_metrics:
        logger.warning("Tidak ada kolom metrik tersedia untuk error analysis.")
        return

    result_df = result_df.copy()
    result_df["avg_metric_score"] = result_df[available_metrics].mean(axis=1)
    worst = result_df.nsmallest(ERROR_ANALYSIS_N, "avg_metric_score").reset_index(drop=True)
    worst.insert(0, "rank", range(1, len(worst) + 1))
    worst["failure_type"] = ""    # Diisi manual oleh peneliti
    worst["failure_notes"] = ""   # Diisi manual oleh peneliti

    output_cols = ["rank", "user_input", "reference", "response",
                   "avg_metric_score", "failure_type", "failure_notes"]
    available_output_cols = [c for c in output_cols if c in worst.columns]

    output_path = EVAL_RESULTS_DIR / f"error_analysis_config_{config}.csv"
    worst[available_output_cols].to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Error analysis disimpan ke: {output_path}")
    logger.info(f"→ Isi kolom 'failure_type' secara manual di file tersebut.")


def run_statistical_tests() -> None:
    """
    Jalankan Wilcoxon Signed-Rank Test antara Config A vs B per metrik.
    Simpan hasil ke statistical_test.csv.
    """
    from scipy import stats

    path_a = EVAL_RESULTS_DIR / "hasil_config_a.csv"
    path_b = EVAL_RESULTS_DIR / "hasil_config_b.csv"

    for p in (path_a, path_b):
        if not p.exists():
            raise FileNotFoundError(
                f"File tidak ditemukan: {p}\n"
                "Jalankan evaluasi Config A dan B terlebih dahulu."
            )

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    if len(df_a) != len(df_b):
        raise ValueError(
            f"Jumlah baris tidak sama: Config A ({len(df_a)}) vs Config B ({len(df_b)}). "
            "Pastikan ground_truth.csv yang sama digunakan."
        )

    test_results = []
    for metric in METRICS_COLS:
        if metric not in df_a.columns or metric not in df_b.columns:
            logger.warning(f"Metrik '{metric}' tidak ditemukan di salah satu CSV, dilewati.")
            continue

        scores_a = df_a[metric].dropna()
        scores_b = df_b[metric].dropna()

        try:
            stat, p_val = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
            significant = p_val < 0.05
            mean_a = scores_a.mean()
            mean_b = scores_b.mean()
            winner = "Config B" if mean_b > mean_a else "Config A" if mean_a > mean_b else "Tie"

            test_results.append({
                "metric": metric,
                "mean_config_a": round(mean_a, 4),
                "mean_config_b": round(mean_b, 4),
                "wilcoxon_statistic": round(stat, 4),
                "p_value": round(p_val, 4),
                "significant_at_0.05": significant,
                "winner": winner if significant else "n.s.",
            })
            sig_label = "✓ SIGNIFIKAN" if significant else "✗ tidak signifikan"
            print(f"{metric:30s}: p={p_val:.4f} ({sig_label}) | Winner: {winner}")
        except Exception as e:
            logger.warning(f"Wilcoxon gagal untuk '{metric}': {e}")

    output_path = EVAL_RESULTS_DIR / "statistical_test.csv"
    pd.DataFrame(test_results).to_csv(output_path, index=False)
    logger.info(f"Hasil uji statistik disimpan ke: {output_path}")


def generate_visualization() -> None:
    """
    Generate bar chart 3 config (A/B/C) + annotasi p-value.
    """
    paths = {
        "a": EVAL_RESULTS_DIR / "hasil_config_a.csv",
        "b": EVAL_RESULTS_DIR / "hasil_config_b.csv",
        "c": EVAL_RESULTS_DIR / "hasil_config_c.csv",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"File CSV belum ada untuk config: {', '.join(missing)}\n"
            "Jalankan evaluasi semua config terlebih dahulu."
        )

    dfs = {k: pd.read_csv(p) for k, p in paths.items()}
    ragas_metrics = [m for m in METRICS_COLS if m in dfs["a"].columns]

    means = {k: [dfs[k][m].mean() if m in dfs[k].columns else 0 for m in ragas_metrics]
             for k in ("a", "b", "c")}

    # Load p-values jika tersedia
    pvals = {}
    stats_path = EVAL_RESULTS_DIR / "statistical_test.csv"
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path).set_index("metric")
        for m in ragas_metrics:
            if m in stats_df.index:
                p = stats_df.loc[m, "p_value"]
                sig = stats_df.loc[m, "significant_at_0.05"]
                pvals[m] = f"p={p:.3f}{'*' if sig else ''}"

    x = range(len(ragas_metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"a": "#4C72B0", "b": "#DD8452", "c": "#55A868"}
    labels = {"a": "Config A (RAG 500)", "b": "Config B (RAG 2000)", "c": "Config C (BM25)"}
    offsets = {"a": -width, "b": 0, "c": width}

    for k in ("a", "b", "c"):
        bars = ax.bar([i + offsets[k] for i in x], means[k], width,
                      label=labels[k], color=colors[k])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    # Anotasi p-value di atas pasangan A-B
    for i, m in enumerate(ragas_metrics):
        if m in pvals:
            y_max = max(means["a"][i], means["b"][i]) + 0.07
            ax.text(i, y_max, pvals[m], ha="center", fontsize=8, color="gray",
                    style="italic")

    ax.set_ylabel("Skor Ragas")
    ax.set_title("Perbandingan Evaluasi RAG: Config A vs B vs C (BM25 Baseline)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.replace("_", "\n") for m in ragas_metrics])
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output = EVAL_RESULTS_DIR / "perbandingan_visual.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Visualisasi disimpan ke: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline evaluasi UNSRAT RAG"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", choices=["a", "b", "c"],
                       help="Evaluasi satu konfigurasi")
    group.add_argument("--stats", action="store_true",
                       help="Wilcoxon A vs B → statistical_test.csv")
    group.add_argument("--visualize", action="store_true",
                       help="Chart perbandingan 3 config (butuh semua CSV)")
    args = parser.parse_args()

    if args.config:
        run_evaluation(args.config)
    elif args.stats:
        run_statistical_tests()
    else:
        generate_visualization()


if __name__ == "__main__":
    main()
```

---

## CHECKLIST IMPLEMENTASI

### Fase 0 — Setup

- [ ] `environment.yml` dibuat sesuai spec
- [ ] `conda env create -f environment.yml` sukses
- [ ] `.gitignore` berisi `.env` dan `chroma_db/`
- [ ] `.env` dibuat dengan `GOOGLE_API_KEY`
- [ ] `src/__init__.py` dibuat (kosong)
- [ ] Semua folder dibuat: `data/corpus/`, `eval/dataset/`, `eval/results/`

### Fase 1 — Config

- [ ] `python src/config.py` tidak error (kecuali `.env` belum ada)
- [ ] Semua path menggunakan `Path()` bukan string hardcoded
- [ ] `GOOGLE_API_KEY` berhasil dibaca dari `.env`

### Fase 2 — Ingestion

- [ ] `python src/ingestion.py --config b --rebuild` sukses
- [ ] Log menampilkan nama file, jumlah chunk, jumlah skip
- [ ] `python src/ingestion.py --config a --rebuild` sukses
- [ ] Folder `chroma_db/config_a/` dan `chroma_db/config_b/` ada
- [ ] Count chunk > 0 di kedua collection
- [ ] Menjalankan ingestion kedua kali (tanpa --rebuild) tidak menambah duplikat

### Fase 2.5 — BM25 (Config C)

- [ ] `python src/bm25_retriever.py --rebuild` sukses
- [ ] `bm25_index/bm25_index.pkl` ada
- [ ] `retrieve_chunks_bm25("syarat yudisium")` mengembalikan list non-kosong
- [ ] `get_response(..., config="c", streaming=False)` mengembalikan dict valid

### Fase 3 & 4 — Retriever & Chain

- [ ] `retrieve_chunks("syarat yudisium", config="b")` mengembalikan list non-kosong
- [ ] `detect_category("kapan batas KRS?")` mengembalikan `"calendar"` atau `"academic"`
- [ ] `get_response(..., streaming=False)` mengembalikan dict dengan key `answer`, `sources`, `found`
- [ ] `get_response(..., streaming=True)` mengembalikan Generator
- [ ] Query tidak relevan (`found=False`) tidak memanggil LLM

### Fase 5 — App

- [ ] `streamlit run app.py` berjalan tanpa error
- [ ] Tab Chatbot merespons pertanyaan
- [ ] Sitasi sumber muncul di expander
- [ ] Tombol Reset membersihkan chat
- [ ] Disclaimer muncul di bawah

### Fase 6 — Evaluasi

- [ ] `python evaluation.py --config b` berjalan dan cetak ringkasan
- [ ] File `eval/results/hasil_config_b.csv` dibuat
- [ ] `python evaluation.py --config a` berjalan
- [ ] `python evaluation.py --visualize` membuat `perbandingan_visual.png`
- [ ] Tab Evaluasi di Streamlit menampilkan chart dan tabel
- [ ] `python evaluation.py --config c` berjalan → `hasil_config_c.csv` ada
- [ ] `python evaluation.py --stats` berjalan → `statistical_test.csv` ada
- [ ] `error_analysis_config_a.csv` dan `error_analysis_config_b.csv` ada
- [ ] Kolom `failure_type` di error analysis diisi manual oleh peneliti
- [ ] `python evaluation.py --visualize` menghasilkan chart 3 config + p-value
- [ ] `AVAILABLE_MODELS` di `config.py` berisi nama model nyata (bukan placeholder)
- [ ] `LLM_MODEL_NAME` ≠ `EVALUATOR_MODEL_NAME`

---

## JEBAKAN UMUM (COMMON PITFALLS)

```
❌ Lupa task_type embedding:
   - Dokumen: task_type="retrieval_document"
   - Query:   task_type="retrieval_query"
   Gejala: retrieval bekerja tapi kualitas rendah secara senyap.

❌ Threshold salah arah:
   - ChromaDB cosine DISTANCE: 0.0 = identik, 2.0 = berlawanan
   - Buang chunk dengan distance > THRESHOLD (bukan <)
   Gejala: semua chunk dibuang, selalu fallback response.

❌ Memory tidak direset di evaluation.py:
   - chat_history=[] WAJIB untuk setiap pertanyaan evaluasi
   Gejala: skor evaluasi tinggi palsu karena "contekan" dari pertanyaan sebelumnya.

❌ Logika RAG di app.py:
   - app.py hanya panggil get_response() dari chain.py
   Gejala: UI tidak bisa diganti framework lain.

❌ Dua API call per pesan di chat (streaming + batch):
   - Ini memang disengaja untuk mendapat sources dari batch call
   - Alternatif: simpan sources dari streaming call (lebih kompleks)
   - Untuk skala prototipe ini, dua call per pesan masih acceptable

❌ ChromaDB collection tidak ditemukan:
   - Pastikan --config sesuai saat ingestion dan saat chat
   - Cek folder chroma_db/config_a/ dan config_b/ ada

❌ Mengisi placeholder [ISI_NAMA_MODEL_*] dengan nama yang tidak ada:
   - Verifikasi nama model di GCP Console / AI Studio sebelum mengisi config.py
   Gejala: error 404 atau "model not found" saat pertama kali memanggil API.

❌ Self-evaluation bias: Generator = Evaluator model:
   - LLM_MODEL_NAME dan EVALUATOR_MODEL_NAME HARUS berbeda (D-16)
   Gejala: skor Ragas meninggi artifisial karena model "mengenali" gayanya sendiri.

❌ Import metrik Ragas salah karena berubah antar versi:
   - Selalu verifikasi nama import dengan `pip show ragas` + docs resmi
   - Minta user untuk memberikan link dokumentasi jika tidak yakin
   Gejala: ImportError atau skor selalu NaN.

❌ Wilcoxon dengan ukuran sampel berbeda (Config A vs B):
   - Pastikan jumlah baris di kedua CSV sama (ground_truth.csv yang identik)
   Gejala: ValueError dari scipy.stats.wilcoxon.
```
