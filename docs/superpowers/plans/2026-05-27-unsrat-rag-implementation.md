# UNSRAT RAG Chatbot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membangun sistem chatbot RAG akademik UNSRAT dengan tiga konfigurasi retrieval (A/B/C) yang dapat dibandingkan melalui evaluasi Ragas 0.4.x.

**Architecture:** Pipeline sequential: ingestion (offline CLI) → index (ChromaDB/BM25) → runtime chain (LangChain + Gemini) → Streamlit UI (dua tab: Chat + Evaluasi). Semua konstanta di `config.py`, semua logika bisnis di `src/`, `app.py` hanya UI.

**Tech Stack:** Python 3.11, LangChain ≥0.3, ChromaDB ≥0.5, Google Gemini API, Ragas 0.4.x, Streamlit ≥1.40, rank-bm25, scipy, python-frontmatter.

**Spec:** `docs/superpowers/specs/2026-05-27-unsrat-rag-design.md` v1.2

**Urutan wajib:** Task 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7. Jangan melompati fase.

---

## File Structure

```
unsrat-rag/
├── environment.yml          ← Task 0: buat/update
├── .gitignore               ← Task 0: buat/update
├── .env                     ← Task 0: buat manual (tidak di-commit)
├── src/
│   ├── __init__.py          ← Task 0: file kosong
│   ├── config.py            ← Task 1: SEMUA konstanta
│   ├── ingestion.py         ← Task 2: pipeline ChromaDB A & B
│   ├── bm25_retriever.py    ← Task 3: BM25 chunk-based indexing
│   ├── retriever.py         ← Task 4: unified retrieval interface
│   └── chain.py             ← Task 5: RAG chain + streaming + retry
├── app.py                   ← Task 6: Streamlit UI (2 tab)
├── evaluation.py            ← Task 7: Ragas 0.4.x + Wilcoxon + chart
└── docs/superpowers/
    ├── specs/2026-05-27-unsrat-rag-design.md
    └── plans/2026-05-27-unsrat-rag-implementation.md  ← file ini
```

---

## Task 0: Setup Project Foundation

**Files:**
- Create/Modify: `environment.yml`
- Create/Modify: `.gitignore`
- Create: `src/__init__.py`
- Create (manual): `.env`

- [ ] **Step 0.1: Verifikasi atau buat `environment.yml`**

Pastikan file ini ada di root proyek dengan konten berikut (perhatikan `ragas>=0.4.0`):

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
      - ragas>=0.4.0
      - streamlit>=1.40.0
      - python-dotenv>=1.0.0
      - python-frontmatter>=1.0.0
      - pandas>=2.0.0
      - pyyaml>=6.0
      - matplotlib>=3.8.0
      - google-generativeai>=0.8.0
      - rank-bm25>=0.2.2
      - scipy>=1.11.0
```

- [ ] **Step 0.2: Verifikasi atau buat `.gitignore`**

```
.env
chroma_db/
__pycache__/
*.pyc
.conda/
*.egg-info/
.DS_Store
bm25_index/
```

- [ ] **Step 0.3: Buat `src/__init__.py`** (file kosong, wajib untuk package)

```python
# File kosong — wajib ada agar Python mengenali src/ sebagai package
```

- [ ] **Step 0.4: Buat file `.env`** (manual via text editor, JANGAN commit)

```bash
GOOGLE_API_KEY=isi_dengan_api_key_anda_di_sini
```

- [ ] **Step 0.5: Buat atau aktifkan conda environment**

```bash
conda env create -f environment.yml
conda activate unsrat-rag
```

Jika environment sudah ada dan ingin update:
```bash
conda env update -f environment.yml --prune
conda activate unsrat-rag
```

- [ ] **Step 0.6: Verifikasi instalasi library**

```bash
python --version
```
Expected: `Python 3.11.x`

```bash
python -c "import langchain; print('langchain:', langchain.__version__)"
python -c "import chromadb; print('chromadb:', chromadb.__version__)"
python -c "import ragas; print('ragas:', ragas.__version__)"
python -c "import streamlit; print('streamlit:', streamlit.__version__)"
python -c "import frontmatter; print('python-frontmatter OK')"
python -c "from rank_bm25 import BM25Okapi; print('rank-bm25 OK')"
python -c "from scipy.stats import wilcoxon; print('scipy OK')"
```

Expected: semua output `OK` atau versi, tidak ada `ModuleNotFoundError`.

- [ ] **Step 0.7: Commit setup awal**

```bash
git add environment.yml .gitignore src/__init__.py
git commit -m "chore: setup project foundation — environment, gitignore, src package"
```

---

## Task 1: `src/config.py`

**Files:**
- Create/Modify: `src/config.py`

- [ ] **Step 1.1: Tulis `src/config.py` lengkap**

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

# ── PATH ────────────────────────────────────────────────────────────────────
ROOT_DIR          = Path(__file__).parent.parent
CORPUS_DIR        = ROOT_DIR / "data" / "corpus"
CHROMA_BASE_DIR   = ROOT_DIR / "chroma_db"
CHROMA_DIR_A      = CHROMA_BASE_DIR / "config_a"
CHROMA_DIR_B      = CHROMA_BASE_DIR / "config_b"
BM25_INDEX_DIR    = ROOT_DIR / "bm25_index"
BM25_INDEX_PATH   = BM25_INDEX_DIR / "bm25_index.pkl"
EVAL_DATASET_PATH = ROOT_DIR / "eval" / "dataset" / "ground_truth.csv"
EVAL_RESULTS_DIR  = ROOT_DIR / "eval" / "results"

# ── CHROMADB COLLECTIONS ────────────────────────────────────────────────────
CHROMA_COLLECTION_A = "unsrat_rag_config_a"
CHROMA_COLLECTION_B = "unsrat_rag_config_b"
CHROMA_DISTANCE_FN  = "cosine"

# ── MODEL ───────────────────────────────────────────────────────────────────
# Generator dan evaluator HARUS BERBEDA (mitigasi self-eval bias — D-16).
# Ganti dengan nama model yang tersedia di akun GCP Anda.
LLM_MODEL_NAME       = "gemini-3.5-flash"
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EVALUATOR_MODEL_NAME = "gemini-3.1-pro-preview"

# Daftar model yang bisa dipilih di UI sidebar.
AVAILABLE_MODELS: list[str] = [
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]

# ── CHUNKING — CONFIG A (Eksperimen, chunk kecil) ───────────────────────────
CHUNK_SIZE_A    = 500
CHUNK_OVERLAP_A = 100

# ── CHUNKING — CONFIG B (Best Practice, chunk besar) ───────────────────────
CHUNK_SIZE_B    = 2000
CHUNK_OVERLAP_B = 200

# ── SEPARATORS (sama untuk Config A, B, dan BM25) ──────────────────────────
CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]

# ── RETRIEVAL ───────────────────────────────────────────────────────────────
RETRIEVAL_K          = 4      # Jumlah chunk yang diambil per query
SIMILARITY_THRESHOLD = 0.65   # Cosine distance threshold (0=identik, 1=orthogonal)
                               # Chunk dengan distance > 0.65 dibuang
MIN_CHUNK_LENGTH     = 50     # Minimum karakter setelah strip() untuk chunk valid

# ── BM25 — CONFIG C ─────────────────────────────────────────────────────────
BM25_K             = 4   # Jumlah chunk BM25 yang dikembalikan per query
BM25_MIN_TOKEN_LEN = 2   # Panjang minimum token (filter kasar stopword)

# ── LLM GENERATION ──────────────────────────────────────────────────────────
LLM_TEMPERATURE       = 0.1   # Rendah untuk konsistensi dan reprodusibilitas
LLM_MAX_OUTPUT_TOKENS = 800
LLM_TOP_P             = 0.95

# ── MEMORI PERCAKAPAN ───────────────────────────────────────────────────────
MEMORY_K = 5   # Simpan 5 pasang Q&A terakhir (= 10 pesan)

# ── RETRY POLICY ─────────────────────────────────────────────────────────────
MAX_RETRIES  = 3
RETRY_DELAYS = [2, 5]   # detik: attempt2 tunggu 2 detik, attempt3 tunggu 5 detik

# ── EVALUASI ─────────────────────────────────────────────────────────────────
# evaluation.py WAJIB mengimport METRICS_COLS — DILARANG hardcode list metrik (NFR-03)
METRICS_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    # "context_entity_recall",  # Opsional — uncomment di sini untuk mengaktifkan
]
# evaluation.py WAJIB mengimport ERROR_ANALYSIS_N — DILARANG hardcode angka (NFR-03)
ERROR_ANALYSIS_N = 10

# ── YAML VALIDATION ──────────────────────────────────────────────────────────
# Field YAML wajib yang divalidasi saat ingestion.
# Definisi ada HANYA di sini — ingestion.py dan bm25_retriever.py mengimport dari sini.
REQUIRED_YAML_FIELDS = [
    "doc_id",
    "title",
    "category",
    "content_type",
    "valid_from",
    "status",
    "retrieval_summary",
    "chunk_strategy",
    "last_updated",
]

# Subset field yang benar-benar dibaca oleh retriever/chain saat runtime.
# Field lainnya adalah corpus metadata untuk dokumentasi manusia.
PIPELINE_CONSUMED_FIELDS = [
    "doc_id",
    "title",
    "category",
    "content_type",
    "status",
]

# ── CATEGORY DETECTION KEYWORDS ──────────────────────────────────────────────
# Digunakan untuk label UI saja — TIDAK mempengaruhi retrieval ChromaDB.
# Menambah kategori baru: cukup tambah key baru di dict ini.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "academic": [
        "SKS", "KRS", "IPK", "IPS", "UTS", "UAS", "DO", "drop out",
        "pasal", "peraturan", "yudisium", "cuti", "mata kuliah",
        "nilai", "ujian", "wisuda", "ijazah", "skripsi", "sidang",
        "kurikulum", "beban belajar", "semester",
    ],
    "calendar": [
        "jadwal", "tanggal", "kapan", "batas", "deadline", "registrasi",
        "pengisian", "periode", "libur", "minggu ke", "kalender",
        "perkuliahan dimulai", "uts dimulai", "uas dimulai",
    ],
    "institution_profile": [
        "sejarah", "visi", "misi", "lambang", "bendera", "mars",
        "hymne", "akreditasi", "rektor", "berdiri", "didirikan",
        "profil universitas",
    ],
    "faq": [
        "FAQ", "pertanyaan umum", "sering ditanya",
    ],
}

# ── SESSION STATE KEYS ───────────────────────────────────────────────────────
# Konstanta untuk st.session_state keys — gunakan ini di app.py, jangan hardcode string.
SESSION_MESSAGES      = "messages"        # list[dict] riwayat chat
SESSION_ACTIVE_CONFIG = "active_config"   # "a", "b", atau "c"
# SESSION_CHAT_HISTORY tidak digunakan — kita pakai manual list (deviasi D-A4).

# ── SYSTEM PROMPT (TERKUNCI) ─────────────────────────────────────────────────
SYSTEM_PROMPT = """Anda adalah agen asisten informasi akademik resmi \
Universitas Sam Ratulangi.
Tugas Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan dokumen \
konteks yang disediakan.

Jika jawaban tidak ada di dalam dokumen konteks, katakan secara jujur bahwa \
Anda tidak menemukan informasinya dan arahkan mereka untuk menghubungi \
bagian administrasi kampus terkait.

JANGAN PERNAH mengarang informasi, tanggal, atau angka SKS.
Jawab dalam Bahasa Indonesia yang ramah dan mudah dipahami."""

# ── FALLBACK RESPONSE ─────────────────────────────────────────────────────────
FALLBACK_RESPONSE = (
    "Maaf, saya tidak menemukan informasi yang relevan mengenai "
    "pertanyaan Anda dalam dokumen yang tersedia. "
    "Untuk informasi lebih lanjut, silakan hubungi:\n\n"
    "• Bagian Akademik UNSRAT\n"
    "• Portal INSPIRE: inspire.unsrat.ac.id\n"
    "• Atau datang langsung ke Gedung Rektorat UNSRAT"
)

# ── API KEY ───────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY tidak ditemukan!\n"
        "Buat file .env di root proyek dan isi: GOOGLE_API_KEY=your_key_here"
    )
```

- [ ] **Step 1.2: Verifikasi config berjalan tanpa error**

```bash
python src/config.py
```

Expected: tidak ada output, tidak ada error. (Jika `.env` valid, tidak ada Exception.)

- [ ] **Step 1.3: Commit**

```bash
git add src/config.py
git commit -m "feat: tambah src/config.py — pusat konfigurasi sistem"
```

---

## Task 2: `src/ingestion.py`

**Files:**
- Create: `src/ingestion.py`

- [ ] **Step 2.1: Tulis `src/ingestion.py` lengkap**

```python
"""
src/ingestion.py
Pipeline ingestion: baca .md → chunk → embed → simpan ke ChromaDB.
Mendukung Config A (chunk 500) dan Config B (chunk 2000).
Jalankan via CLI: python src/ingestion.py --config [a|b] [--rebuild]
"""

import argparse
import hashlib
import logging
from pathlib import Path

import frontmatter
import chromadb
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    CORPUS_DIR,
    CHROMA_DIR_A, CHROMA_DIR_B,
    CHROMA_COLLECTION_A, CHROMA_COLLECTION_B,
    CHROMA_DISTANCE_FN,
    CHUNK_SIZE_A, CHUNK_OVERLAP_A,
    CHUNK_SIZE_B, CHUNK_OVERLAP_B,
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
    """Buat ID unik untuk chunk berdasarkan MD5 hash doc_id + konten."""
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
            "bagian":       chunk.metadata.get("bagian", ""),
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

        try:
            metadata, content = parse_markdown_file(md_file)
        except ValueError as e:
            logger.warning(f"SKIP {md_file.name}: {e}")
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

            embedding_vector = embeddings.embed_query(chunk["content"])
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding_vector],
                documents=[chunk["content"]],
                metadatas=[chunk["metadata"]],
            )
            inserted += 1

        total_inserted  += inserted
        total_duplicate += skipped_dup
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
```

- [ ] **Step 2.2: Jalankan ingestion Config B**

```bash
python src/ingestion.py --config b --rebuild
```

Expected output (jumlah chunk akan bervariasi):
```
INFO: === Memulai Ingestion ===
INFO: Collection : unsrat_rag_config_b
INFO: Chunk size : 2000 | Overlap : 200
...
INFO: === Ingestion Selesai ===
INFO: Total di-insert     : [angka > 0]
INFO: Total di database   : [angka > 0]
```

- [ ] **Step 2.3: Jalankan ingestion Config A**

```bash
python src/ingestion.py --config a --rebuild
```

Expected: log serupa dengan Config B.

- [ ] **Step 2.4: Verifikasi jumlah chunk di database**

```bash
python -c "
import chromadb
from src.config import CHROMA_DIR_A, CHROMA_DIR_B, CHROMA_COLLECTION_A, CHROMA_COLLECTION_B
ca = chromadb.PersistentClient(path=str(CHROMA_DIR_A)).get_collection(CHROMA_COLLECTION_A)
cb = chromadb.PersistentClient(path=str(CHROMA_DIR_B)).get_collection(CHROMA_COLLECTION_B)
print(f'Config A: {ca.count()} chunks')
print(f'Config B: {cb.count()} chunks')
"
```

Expected: Config B biasanya lebih sedikit chunk dari Config A (chunk lebih besar = lebih sedikit potongan).

- [ ] **Step 2.5: Commit**

```bash
git add src/ingestion.py
git commit -m "feat: tambah src/ingestion.py — pipeline ChromaDB Config A & B"
```

---

## Task 3: `src/bm25_retriever.py`

**Files:**
- Create: `src/bm25_retriever.py`

> **PENTING — Refaktor dari agents.md:** BM25 mengindeks CHUNK (bukan full dokumen).
> Granularitas yang digunakan adalah Config B (chunk_size=2000) untuk perbandingan fair.
> Lihat Design Spec D-A3.

- [ ] **Step 3.1: Tulis `src/bm25_retriever.py` lengkap**

```python
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
from pathlib import Path

import frontmatter
from langchain.text_splitter import (
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
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)
        except Exception as e:
            logger.warning(f"SKIP {md_file.name}: {e}")
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
```

- [ ] **Step 3.2: Bangun indeks BM25**

```bash
python src/bm25_retriever.py --rebuild
```

Expected output:
```
INFO:   ✓ Peraturan_Akademik_UNSRAT_2025_RAG_REVISED.md: [N] chunk diindeks
INFO:   ✓ Kalender_Akademik_UNSRAT_Genap_2025-2026.md: [N] chunk diindeks
...
INFO: Indeks BM25 disimpan ke: .../bm25_index/bm25_index.pkl ([M] chunk dari 9 file)
```

M harus lebih besar dari 9 (karena per-chunk, bukan per-file).

- [ ] **Step 3.3: Verifikasi pickle terbuat**

```bash
python -c "
import pickle
from src.config import BM25_INDEX_PATH
with open(BM25_INDEX_PATH, 'rb') as f:
    data = pickle.load(f)
print(f'Jumlah chunk terindeks: {len(data[\"texts\"])}')
print(f'Contoh metadata[0]: {data[\"metadata\"][0]}')
"
```

Expected: jumlah chunk > 9, metadata berisi doc_id/title/category.

- [ ] **Step 3.4: Commit**

```bash
git add src/bm25_retriever.py
git commit -m "feat: tambah src/bm25_retriever.py — BM25 chunk-based Config C"
```

---

## Task 4: `src/retriever.py`

**Files:**
- Create: `src/retriever.py`

- [ ] **Step 4.1: Tulis `src/retriever.py` lengkap**

```python
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
    CHROMA_DIR_A, CHROMA_DIR_B,
    CHROMA_COLLECTION_A, CHROMA_COLLECTION_B,
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
```

- [ ] **Step 4.2: Verifikasi retrieval Config B**

```bash
python -c "
from src.retriever import retrieve
results = retrieve('Berapa SKS maksimal yang bisa diambil per semester?', config='b')
print(f'Config B: {len(results)} chunk ditemukan')
if results:
    print(f'Score terbaik: {results[0][\"score\"]:.4f}')
    print(f'Preview: {results[0][\"content\"][:100]}...')
"
```

Expected: 1-4 chunk dengan score < 0.65.

- [ ] **Step 4.3: Verifikasi retrieval query irrelevant (harus return kosong)**

```bash
python -c "
from src.retriever import retrieve
results = retrieve('cuaca hari ini di jakarta bagaimana', config='b')
print(f'Hasil untuk query irrelevant: {len(results)} chunk (expected: 0)')
"
```

Expected: 0 chunk.

- [ ] **Step 4.4: Verifikasi retrieval Config C (BM25)**

```bash
python -c "
from src.retriever import retrieve
results = retrieve('SKS maksimal per semester', config='c')
print(f'Config C (BM25): {len(results)} chunk ditemukan')
if results:
    print(f'Score BM25: {results[0][\"score\"]:.4f}')
"
```

Expected: BM25_K (4) chunk, selalu ada karena BM25 tidak pakai threshold.

- [ ] **Step 4.5: Commit**

```bash
git add src/retriever.py
git commit -m "feat: tambah src/retriever.py — unified retrieval interface A/B/C"
```

---

## Task 5: `src/chain.py`

**Files:**
- Create: `src/chain.py`

- [ ] **Step 5.1: Tulis `src/chain.py` lengkap**

```python
"""
src/chain.py
RAG chain: menerima query → retrieval → format konteks → LLM → response.
Mendukung streaming (st.write_stream) dan non-streaming (evaluation.py).
Mengelola memory percakapan sebagai plain Python list via st.session_state.
"""

import logging
import time
from typing import Generator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_TOP_P,
    MEMORY_K,
    MAX_RETRIES,
    RETRY_DELAYS,
    GOOGLE_API_KEY,
    SYSTEM_PROMPT,
    FALLBACK_RESPONSE,
    CATEGORY_KEYWORDS,
)
from src.retriever import retrieve

logger = logging.getLogger(__name__)

# Instance LLM — bisa di-reinisialisasi via reinitialize_llm()
_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    """Dapatkan atau buat instance LLM dengan konfigurasi default."""
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
            top_p=LLM_TOP_P,
            google_api_key=GOOGLE_API_KEY,
        )
    return _llm


def reinitialize_llm(model_name: str) -> None:
    """
    Reinisialisasi instance LLM dengan model baru.
    Dipanggil oleh app.py saat user mengganti model di sidebar.
    Tidak perlu restart server Streamlit.
    """
    global _llm
    _llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
        top_p=LLM_TOP_P,
        google_api_key=GOOGLE_API_KEY,
    )
    logger.info(f"LLM diinisialisasi ulang dengan model: {model_name}")


def detect_category(query: str) -> str:
    """
    Deteksi kategori pertanyaan berdasarkan keyword matching.
    Mengembalikan label kategori untuk tampilan UI saja.
    TIDAK mempengaruhi proses retrieval ChromaDB (lihat Design Spec D-A2).
    """
    scores: dict[str, int] = {}
    query_lower = query.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in query_lower)
        if score > 0:
            scores[category] = score

    if not scores:
        return "general"
    return max(scores, key=lambda k: scores[k])


def _format_context(chunks: list[dict]) -> str:
    """Gabungkan isi chunk menjadi satu string konteks untuk prompt LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        header = f"[Sumber {i}: {meta.get('title', '')} — {meta.get('bab', '')}]"
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def _format_history(messages: list[dict]) -> str:
    """
    Format MEMORY_K pasang Q&A terakhir sebagai string untuk prompt.
    Mengambil dari plain list (bukan LangChain memory abstraction).
    """
    recent = messages[-(MEMORY_K * 2):]
    lines = []
    for msg in recent:
        role = "Pengguna" if msg["role"] == "user" else "Asisten"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _build_prompt(
    query: str,
    context: str,
    history: str,
) -> list:
    """Bangun daftar pesan (messages) untuk dikirim ke LLM."""
    user_content = ""
    if history:
        user_content += f"Riwayat percakapan:\n{history}\n\n"
    user_content += f"Konteks dokumen:\n{context}\n\nPertanyaan: {query}"

    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]


def get_response(
    query: str,
    config: str,
    chat_history: list[dict] | None = None,
    streaming: bool = False,
) -> dict:
    """
    Dapatkan respons RAG untuk pertanyaan yang diberikan.

    PENTING — get_response() SELALU mengembalikan dict, baik streaming=True maupun False.
    Ini mencegah double API call yang menggandakan biaya dan latensi (K-01 fix).

    Args:
        query:        Pertanyaan dari pengguna.
        config:       "a", "b", atau "c" — config retrieval.
        chat_history: list[dict] riwayat chat. None berarti tidak ada history.
                      Setiap dict: {"role": "user"/"assistant", "content": str}
        streaming:    True = field "stream" berisi Generator untuk st.write_stream().
                      False = field "answer" berisi teks lengkap (untuk evaluation.py).

    Returns:
        Selalu dict dengan keys:
        {
            "answer":  str | None,     # Teks jawaban (streaming=False) atau None (streaming=True)
            "stream":  Generator | None, # Token generator (streaming=True) atau None
            "sources": list[dict],     # Chunk yang digunakan — tersedia di KEDUA mode
            "found":   bool,           # False jika tidak ada chunk lolos threshold
        }
        sources[i] memiliki keys: doc_id, title, bab, bagian, pasal, content, preview, category
    """
    if chat_history is None:
        chat_history = []

    # 1. Retrieval — dilakukan SEKALI, hasilnya digunakan untuk sources DAN LLM
    chunks = retrieve(query, config)

    # 2. Fallback jika tidak ada chunk yang relevan
    if not chunks:
        if streaming:
            def _fallback_gen():
                yield FALLBACK_RESPONSE
            return {"answer": None, "stream": _fallback_gen(), "sources": [], "found": False}
        return {"answer": FALLBACK_RESPONSE, "stream": None, "sources": [], "found": False}

    # 3. Bangun sources SEKARANG dari chunks — tersedia sebelum LLM dipanggil
    # Sources dikembalikan di KEDUA mode streaming/non-streaming tanpa API call tambahan
    sources = []
    for chunk in chunks:
        meta = chunk["metadata"]
        sources.append({
            "doc_id":   meta.get("doc_id", ""),
            "title":    meta.get("title", ""),
            "bab":      meta.get("bab", ""),
            "bagian":   meta.get("bagian", ""),
            "pasal":    meta.get("pasal", ""),
            "content":  chunk["content"],       # FULL text — untuk Ragas
            "preview":  chunk["content"][:150], # 150 char — untuk UI
            "category": meta.get("category", ""),
        })

    # 4. Format konteks dan history
    context  = _format_context(chunks)
    history  = _format_history(chat_history)
    messages = _build_prompt(query, context, history)
    llm      = _get_llm()

    # 5. Panggil LLM — SATU KALI untuk kedua mode
    if streaming:
        return {
            "answer":  None,
            "stream":  _stream_with_retry(llm, messages),
            "sources": sources,
            "found":   True,
        }
    else:
        answer = _invoke_with_retry(llm, messages)
        return {
            "answer":  answer,
            "stream":  None,
            "sources": sources,
            "found":   True,
        }


def _invoke_with_retry(llm: ChatGoogleGenerativeAI, messages: list) -> str:
    """Panggil LLM.invoke() dengan retry policy. Raises Exception jika semua attempt gagal."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            last_error = e
            logger.warning(f"LLM invoke gagal (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])

    raise RuntimeError(
        f"Gagal mendapatkan respons dari LLM setelah {MAX_RETRIES} percobaan. "
        f"Error terakhir: {last_error}"
    )


def _stream_with_retry(llm: ChatGoogleGenerativeAI, messages: list) -> Generator:
    """
    Panggil LLM.stream() dengan retry policy.
    Mengembalikan Generator token untuk st.write_stream().
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            for chunk in llm.stream(messages):
                yield chunk.content
            return
        except Exception as e:
            last_error = e
            logger.warning(f"LLM stream gagal (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])

    yield (
        f"\n\nMaaf, terjadi kesalahan saat menghubungi layanan AI "
        f"setelah {MAX_RETRIES} percobaan. Silakan coba lagi."
    )
```

- [ ] **Step 5.2: Verifikasi get_response() non-streaming**

```bash
python -c "
from src.chain import get_response
result = get_response('Berapa SKS maksimal yang bisa diambil per semester?', config='b', streaming=False)
print('found:', result['found'])
print('stream field:', result['stream'])   # harus None di non-streaming
print('answer preview:', result['answer'][:200])
print('jumlah sources:', len(result['sources']))
if result['sources']:
    src = result['sources'][0]
    print('source[0] keys:', list(src.keys()))
    print('content length:', len(src['content']))  # harus > 150
    print('preview length:', len(src['preview']))   # harus <= 150
"
```

Expected: `found: True`, `stream: None`, `content length > 150`, `preview length <= 150`.

- [ ] **Step 5.3: Verifikasi get_response() streaming — sources tersedia tanpa second call**

```bash
python -c "
from src.chain import get_response
result = get_response('Berapa SKS maksimal?', config='b', streaming=True)
print('found:', result['found'])
print('stream is generator:', hasattr(result['stream'], '__next__'))  # harus True
print('sources tersedia langsung:', len(result['sources']))  # harus > 0, TANPA second call
print('answer field:', result['answer'])  # harus None di streaming mode
# Konsumsi beberapa token saja untuk verifikasi
tokens = []
for i, token in enumerate(result['stream']):
    tokens.append(token)
    if i >= 3: break
print('streaming berjalan, token pertama:', repr(tokens[0]))
"
```

Expected: `sources` berisi chunk, `stream` adalah generator, `answer` adalah None.

- [ ] **Step 5.4: Verifikasi fallback (query tidak relevan)**

```bash
python -c "
from src.chain import get_response
result = get_response('cuaca hari ini bagaimana', config='b', streaming=False)
print('found:', result['found'])   # Expected: False
print('answer:', result['answer'][:100])  # Expected: FALLBACK_RESPONSE
print('sources:', result['sources'])      # Expected: []
"
```

Expected: `found: False`, answer dimulai dengan "Maaf, saya tidak menemukan...", sources kosong.

- [ ] **Step 5.5: Verifikasi reinitialize_llm()**

```bash
python -c "
from src.chain import reinitialize_llm, get_response
from src.config import AVAILABLE_MODELS
reinitialize_llm(AVAILABLE_MODELS[0])
print('reinitialize_llm berhasil dengan model:', AVAILABLE_MODELS[0])
"
```

Expected: tidak ada error.

- [ ] **Step 5.6: Commit**

```bash
git add src/chain.py
git commit -m "feat: tambah src/chain.py — RAG chain, single-call streaming+sources, retry, memory"
```

---

## Task 6: `app.py`

**Files:**
- Create: `app.py`

- [ ] **Step 6.1: Tulis `app.py` lengkap**

```python
"""
app.py
Entry point UI Streamlit untuk sistem chatbot RAG UNSRAT.
Dua tab: Chat (RAG real-time) dan Evaluasi (tampilkan hasil eval).
Semua logika bisnis ada di src/ — app.py hanya UI dan pemanggilan fungsi.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import (
    AVAILABLE_MODELS,
    LLM_MODEL_NAME,
    EVAL_RESULTS_DIR,
    METRICS_COLS,
    SESSION_MESSAGES,
    SESSION_ACTIVE_CONFIG,
)
from src.chain import get_response, detect_category, reinitialize_llm

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Asisten Akademik UNSRAT",
    page_icon="🎓",
    layout="wide",
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --color-primary:      #7B2D2D;
  --color-primary-alt:  #8B1A2B;
  --color-primary-dark: #5C1F1F;
  --color-primary-light:#A84545;
  --font-main: 'Inter', sans-serif;
}

html, body, [class*="css"] {
  font-family: var(--font-main);
}

.stApp {
  background: linear-gradient(135deg, #1a0a0a 0%, #2d1515 100%);
  color: #f0e6e6;
}

.main-header {
  background: linear-gradient(90deg, var(--color-primary-dark), var(--color-primary));
  padding: 1.5rem 2rem;
  border-radius: 12px;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 20px rgba(123, 45, 45, 0.4);
}

.main-header h1 {
  color: #fff;
  margin: 0;
  font-size: 1.6rem;
  font-weight: 700;
}

.main-header p {
  color: #f0c0c0;
  margin: 0.25rem 0 0 0;
  font-size: 0.85rem;
}

.category-badge {
  display: inline-block;
  background: var(--color-primary);
  color: white;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 0.72rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
}

.disclaimer {
  background: rgba(123, 45, 45, 0.15);
  border: 1px solid var(--color-primary);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  font-size: 0.78rem;
  color: #d4a0a0;
  margin-top: 1rem;
}

.stButton > button {
  background: var(--color-primary);
  color: white;
  border: none;
  border-radius: 8px;
  font-family: var(--font-main);
  font-weight: 500;
  transition: background 0.2s;
}

.stButton > button:hover {
  background: var(--color-primary-light);
}
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE INIT ────────────────────────────────────────────────────────
def _init_session_state() -> None:
    """Inisialisasi session state keys jika belum ada."""
    if SESSION_MESSAGES not in st.session_state:
        st.session_state[SESSION_MESSAGES] = []
    if SESSION_ACTIVE_CONFIG not in st.session_state:
        st.session_state[SESSION_ACTIVE_CONFIG] = "b"
    if "active_model" not in st.session_state:
        st.session_state["active_model"] = LLM_MODEL_NAME


def _reset_conversation() -> None:
    """Hapus riwayat percakapan dan reset session state."""
    st.session_state[SESSION_MESSAGES] = []


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    """Render sidebar dengan config switcher, model switcher, dan reset button."""
    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan")
        st.divider()

        # Config switcher
        config_labels = {"a": "Config A — RAG (500 char)", "b": "Config B — RAG (2000 char)", "c": "Config C — BM25"}
        prev_config = st.session_state[SESSION_ACTIVE_CONFIG]
        selected_label = st.radio(
            "Konfigurasi Retrieval",
            options=list(config_labels.values()),
            index=list(config_labels.keys()).index(prev_config),
        )
        new_config = [k for k, v in config_labels.items() if v == selected_label][0]

        if new_config != prev_config:
            st.session_state[SESSION_ACTIVE_CONFIG] = new_config
            _reset_conversation()
            st.rerun()

        st.divider()

        # Model switcher
        prev_model = st.session_state["active_model"]
        selected_model = st.selectbox(
            "Model LLM",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(prev_model) if prev_model in AVAILABLE_MODELS else 0,
        )
        if selected_model != prev_model:
            reinitialize_llm(selected_model)
            st.session_state["active_model"] = selected_model

        st.divider()

        # Reset button
        if st.button("🔄 Reset Percakapan", use_container_width=True):
            _reset_conversation()
            st.rerun()

        st.divider()

        # Info panel
        active_config = st.session_state[SESSION_ACTIVE_CONFIG]
        st.markdown(f"""
        **Config aktif:** `{active_config.upper()}`
        **Model aktif:** `{st.session_state['active_model']}`
        """)


# ── TAB 1: CHAT ───────────────────────────────────────────────────────────────
def _render_chat_tab() -> None:
    """Render tab chat utama dengan streaming response dan sitasi sumber."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Asisten Informasi Akademik UNSRAT</h1>
        <p>Didukung oleh arsitektur RAG + Google Gemini</p>
    </div>
    """, unsafe_allow_html=True)

    messages = st.session_state[SESSION_MESSAGES]
    active_config = st.session_state[SESSION_ACTIVE_CONFIG]

    # Tampilkan riwayat chat
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "category" in msg:
                st.markdown(
                    f'<span class="category-badge">📂 {msg["category"]}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 Lihat Sumber ({len(msg['sources'])} referensi)"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}] {src['title']} — {src['bab']}**")
                        st.text(src["preview"])

    # Input pengguna
    if user_input := st.chat_input("Ketik pertanyaan Anda..."):
        # Tampilkan pesan user
        with st.chat_message("user"):
            st.markdown(user_input)
        messages.append({"role": "user", "content": user_input})

        # Deteksi kategori (label UI saja)
        category = detect_category(user_input)

        # SINGLE API CALL: get_response() mengembalikan stream + sources sekaligus
        # Tidak ada second call — sources dari retrieval (sebelum LLM), bukan dari LLM.
        result = get_response(
            query=user_input,
            config=active_config,
            chat_history=messages[:-1],  # exclude pesan user yang baru saja ditambahkan
            streaming=True,
        )
        sources      = result["sources"]  # Tersedia langsung dari retrieval
        found        = result["found"]
        stream_gen   = result["stream"]   # Generator untuk st.write_stream

        # Tampilkan streaming response
        with st.chat_message("assistant"):
            st.markdown(
                f'<span class="category-badge">📂 {category}</span>',
                unsafe_allow_html=True,
            )
            full_response = st.write_stream(stream_gen)

            # Tampilkan sitasi sumber langsung setelah streaming selesai
            if sources:
                with st.expander(f"📚 Lihat Sumber ({len(sources)} referensi)"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**[{i}] {src['title']} — {src['bab']}**")
                        st.text(src["preview"])

        # Simpan ke session state
        messages.append({
            "role":     "assistant",
            "content":  full_response,
            "category": category,
            "sources":  sources,
        })

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> Sistem ini adalah prototipe penelitian.
        Informasi yang disajikan bersumber dari dokumen yang tersedia dan dapat
        memiliki keterbatasan. Untuk keputusan akademik penting, selalu verifikasi
        dengan pihak Akademik UNSRAT secara langsung.
    </div>
    """, unsafe_allow_html=True)


# ── TAB 2: EVALUASI ───────────────────────────────────────────────────────────
def _render_evaluation_tab() -> None:
    """Render tab hasil evaluasi Ragas (read-only dari eval/results/)."""
    st.markdown("## 📊 Hasil Evaluasi Ragas")
    st.markdown(
        "Halaman ini menampilkan hasil evaluasi yang sudah dijalankan via CLI. "
        "Untuk menjalankan evaluasi baru, gunakan terminal."
    )

    result_files = {
        "a": EVAL_RESULTS_DIR / "hasil_config_a.csv",
        "b": EVAL_RESULTS_DIR / "hasil_config_b.csv",
        "c": EVAL_RESULTS_DIR / "hasil_config_c.csv",
    }
    stats_file  = EVAL_RESULTS_DIR / "statistical_test.csv"
    chart_file  = EVAL_RESULTS_DIR / "perbandingan_visual.png"

    existing = {k: v for k, v in result_files.items() if v.exists()}

    if not existing:
        st.info(
            "📂 Belum ada hasil evaluasi.\n\n"
            "Jalankan evaluasi via terminal:\n"
            "```\n"
            "python evaluation.py --config a\n"
            "python evaluation.py --config b\n"
            "python evaluation.py --config c\n"
            "python evaluation.py --stats\n"
            "python evaluation.py --visualize\n"
            "```"
        )
        return

    # Tampilkan tabel per config
    dfs: dict[str, pd.DataFrame] = {}
    for config_key, filepath in existing.items():
        df = pd.read_csv(filepath)
        dfs[config_key] = df
        st.markdown(f"### Config {'ABC'[ord(config_key) - ord('a')]}: `{filepath.name}`")
        # Tampilkan aggregasi metrik
        agg_data = {}
        for metric in METRICS_COLS:
            if metric in df.columns:
                agg_data[metric] = f"{df[metric].mean():.3f} ± {df[metric].std():.3f}"
        if "response_time_seconds" in df.columns:
            agg_data["response_time"] = (
                f"{df['response_time_seconds'].mean():.2f} ± "
                f"{df['response_time_seconds'].std():.2f} detik"
            )
        if agg_data:
            st.table(pd.DataFrame(agg_data, index=["mean ± std"]).T)

    # Tampilkan tabel Wilcoxon jika ada
    if stats_file.exists():
        st.markdown("### Uji Signifikansi Wilcoxon (A vs B)")
        stats_df = pd.read_csv(stats_file)
        st.dataframe(stats_df)

    # Tampilkan chart jika ada
    if chart_file.exists():
        st.markdown("### Visualisasi Perbandingan")
        st.image(str(chart_file), use_container_width=True)
    else:
        st.info("Chart belum tersedia. Jalankan: `python evaluation.py --visualize`")

    # CLI reference
    st.markdown("---")
    st.markdown("**Perintah CLI evaluasi:**")
    st.code(
        "python evaluation.py --config a   # Evaluasi Config A\n"
        "python evaluation.py --config b   # Evaluasi Config B\n"
        "python evaluation.py --config c   # Evaluasi Config C (BM25)\n"
        "python evaluation.py --stats      # Uji Wilcoxon A vs B\n"
        "python evaluation.py --visualize  # Generate bar chart",
        language="bash",
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    """Entry point utama aplikasi Streamlit."""
    _init_session_state()
    _render_sidebar()

    tab_chat, tab_eval = st.tabs(["💬 Chatbot", "📊 Evaluasi Ragas"])

    with tab_chat:
        _render_chat_tab()

    with tab_eval:
        _render_evaluation_tab()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Jalankan Streamlit dan verifikasi UI**

```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`. Verifikasi manual:
- [ ] Header UNSRAT tampil dengan warna maroon
- [ ] Sidebar berisi config switcher, model switcher, reset button
- [ ] Tab Chat: input box tampil di bawah
- [ ] Ketik pertanyaan → response streaming (token per token)
- [ ] Category badge tampil di atas response
- [ ] Expander sumber tampil setelah response
- [ ] Disclaimer tampil di bawah chat
- [ ] Tab Evaluasi: tampilkan instruksi CLI karena belum ada CSV
- [ ] Switch config A → percakapan direset

- [ ] **Step 6.3: Commit**

```bash
git add app.py
git commit -m "feat: tambah app.py — Streamlit UI 2 tab (Chat + Evaluasi)"
```

---

## Task 7: `evaluation.py`

**Files:**
- Create: `evaluation.py`
- Create directory: `eval/dataset/` dan `eval/results/`

- [ ] **Step 7.1: Buat direktori evaluasi**

```bash
mkdir -p eval/dataset eval/results
```

Buat file placeholder agar direktori bisa di-commit:
```bash
echo "# Letakkan ground_truth.csv di direktori ini" > eval/dataset/.gitkeep
echo "# Hasil evaluasi akan tersimpan di sini" > eval/results/.gitkeep
```

- [ ] **Step 7.0 (WAJIB — sebelum menulis kode): Verifikasi Ragas API**

> ⚠️ **PRD Section 11.1 secara eksplisit memperingatkan:** nama class/import Ragas berubah
> antar versi minor. Jangan asumsikan dari memori — selalu verifikasi.

```bash
# Cek versi Ragas yang terinstall
pip show ragas
```

Catat versi yang muncul (contoh: `0.4.x`), lalu buka:
**https://docs.ragas.io/en/stable/getstarted/evals.html**

Verifikasi nama import berikut SESUAI dokumentasi versi tersebut:
- `from ragas import evaluate` → cek nama fungsi
- `from ragas import EvaluationDataset` atau cara lain membuat dataset
- `from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall`
  → verifikasi nama class/instance yang benar
- Parameter `llm=` pada `evaluate()` → verifikasi nama parameter

```bash
# Quick smoke test — verifikasi import berhasil:
python -c "
import ragas
print('Ragas version:', ragas.__version__)
# Coba import metrik — jika error, sesuaikan nama di Step 7.2
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
print('Metric imports OK')
try:
    from ragas import EvaluationDataset
    print('EvaluationDataset import OK')
except ImportError:
    print('EvaluationDataset tidak ada — cek dokumentasi untuk cara membuat dataset')
"
```

**Jika import gagal:** buka https://docs.ragas.io dan sesuaikan nama class/fungsi
di Step 7.2 sebelum menulis kode. Jangan lanjut sebelum semua import berhasil.

- [ ] **Step 7.2: Tulis `evaluation.py` lengkap**

> **CATATAN:** Kode di bawah menggunakan import Ragas yang diestimasikan untuk 0.4.x.
> **Sesuaikan dengan hasil verifikasi Step 7.0** jika nama import berbeda.

```python
"""
evaluation.py
Pipeline evaluasi Ragas 0.4.x untuk sistem RAG UNSRAT.
Mengukur faithfulness, answer_relevancy, context_precision, context_recall.
Juga mengukur latensi (response_time_seconds) per query.

CLI:
  python evaluation.py --config [a|b|c]   Evaluasi satu config
  python evaluation.py --stats            Uji Wilcoxon A vs B
  python evaluation.py --visualize        Generate bar chart
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend untuk server/CLI

from scipy.stats import wilcoxon

from src.config import (
    EVAL_DATASET_PATH,
    EVAL_RESULTS_DIR,
    METRICS_COLS,
    ERROR_ANALYSIS_N,
    EVALUATOR_MODEL_NAME,
    GOOGLE_API_KEY,
)
from src.chain import get_response

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_ground_truth() -> pd.DataFrame:
    """
    Muat dan validasi file ground_truth.csv.
    Raises FileNotFoundError atau ValueError dengan pesan Bahasa Indonesia.
    """
    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"File ground truth tidak ditemukan: {EVAL_DATASET_PATH}\n"
            "Buat file ground_truth.csv di eval/dataset/ dengan kolom: "
            "user_input, reference, category, source_doc"
        )

    df = pd.read_csv(EVAL_DATASET_PATH, encoding="utf-8-sig")

    required_cols = ["user_input", "reference", "category", "source_doc"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan di {EVAL_DATASET_PATH.name}: {missing}\n"
            f"Format yang benar: {required_cols}"
        )

    logger.info(f"Ground truth dimuat: {len(df)} pasang Q&A")
    return df


def _run_evaluation(df: pd.DataFrame, config: str) -> pd.DataFrame:
    """
    Jalankan pipeline RAG untuk setiap pertanyaan dan kumpulkan hasil.
    Setiap pertanyaan dievaluasi INDEPENDEN (chat_history di-reset per pertanyaan).
    """
    # VERIFIKASI: import di bawah harus sudah dikonfirmasi di Step 7.0
    # Jika versi Ragas berbeda, sesuaikan nama import dengan dokumentasi resmi
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_google_genai import ChatGoogleGenerativeAI

    logger.info(f"Memulai evaluasi Config {config.upper()} — {len(df)} pertanyaan...")

    rows = []
    response_times = []

    for idx, row in df.iterrows():
        logger.info(f"  [{idx + 1}/{len(df)}] {row['user_input'][:60]}...")

        chat_history = []  # WAJIB reset per pertanyaan (FR-14)

        t_start = time.time()
        try:
            result = get_response(
                query=row["user_input"],
                config=config,
                chat_history=chat_history,
                streaming=False,
            )
            response_time = time.time() - t_start

            answer = result["answer"]
            # KRITIS: gunakan s["content"] (full chunk), BUKAN s["preview"] (150 char)
            # Menggunakan preview akan merusak validitas context_precision & context_recall
            contexts = [s["content"] for s in result["sources"]]

        except Exception as e:
            logger.warning(f"  ERROR pada pertanyaan {idx}: {e}")
            response_time = time.time() - t_start
            answer = ""
            contexts = []

        rows.append({
            "user_input":         row["user_input"],
            "reference":          row["reference"],
            "response":           answer,
            "retrieved_contexts": contexts,
        })
        response_times.append(response_time)

    # Siapkan EvaluationDataset Ragas 0.4.x
    dataset = EvaluationDataset.from_list(rows)

    # Gunakan EVALUATOR_MODEL (berbeda dari LLM_MODEL — mitigasi self-eval bias D-16)
    evaluator_llm = ChatGoogleGenerativeAI(
        model=EVALUATOR_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    logger.info("Menjalankan ragas.evaluate()...")

    # Mapping nama metrik ke objek Ragas
    metric_objects = {
        "faithfulness":      faithfulness,
        "answer_relevancy":  answer_relevancy,
        "context_precision": context_precision,
        "context_recall":    context_recall,
    }
    selected_metrics = [metric_objects[m] for m in METRICS_COLS if m in metric_objects]

    results = evaluate(
        dataset=dataset,
        metrics=selected_metrics,
        llm=evaluator_llm,
    )

    results_df = results.to_pandas()

    # Tambahkan kolom tambahan
    results_df["response_time_seconds"] = response_times
    results_df["category"]  = df["category"].values
    results_df["source_doc"] = df["source_doc"].values

    return results_df


def _print_aggregation(results_df: pd.DataFrame, config: str) -> None:
    """Cetak ringkasan agregasi ke terminal."""
    print(f"\n{'='*60}")
    print(f"HASIL EVALUASI CONFIG {config.upper()}")
    print(f"{'='*60}")
    for metric in METRICS_COLS:
        if metric in results_df.columns:
            mean = results_df[metric].mean()
            std  = results_df[metric].std()
            print(f"  {metric:<25}: {mean:.3f} ± {std:.3f}")
    if "response_time_seconds" in results_df.columns:
        rt_mean = results_df["response_time_seconds"].mean()
        rt_std  = results_df["response_time_seconds"].std()
        print(f"  {'response_time (detik)':<25}: {rt_mean:.2f} ± {rt_std:.2f}")
    print(f"{'='*60}\n")


def _export_error_analysis(results_df: pd.DataFrame, config: str) -> None:
    """
    Ekspor ERROR_ANALYSIS_N sampel dengan rata-rata skor metrik terendah.
    Kolom failure_type dan failure_notes diisi manual oleh peneliti.
    """
    if config == "c":
        logger.info("Error analysis Config C dilewati (opsional, analisis terpisah).")
        return

    available_metrics = [m for m in METRICS_COLS if m in results_df.columns]
    if not available_metrics:
        logger.warning("Tidak ada kolom metrik untuk error analysis.")
        return

    results_df = results_df.copy()
    results_df["avg_metric_score"] = results_df[available_metrics].mean(axis=1)

    worst = results_df.nsmallest(ERROR_ANALYSIS_N, "avg_metric_score").copy()
    worst.insert(0, "rank", range(1, len(worst) + 1))
    worst["failure_type"]  = ""   # diisi manual oleh peneliti
    worst["failure_notes"] = ""   # diisi manual oleh peneliti

    output_cols = [
        "rank", "user_input", "reference", "response",
        "avg_metric_score", "failure_type", "failure_notes",
    ]
    output_cols = [c for c in output_cols if c in worst.columns]

    out_path = EVAL_RESULTS_DIR / f"error_analysis_config_{config}.csv"
    worst[output_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Error analysis disimpan ke: {out_path}")
    logger.info(
        "PENTING: Isi kolom 'failure_type' secara manual dengan salah satu:\n"
        "  retrieval_failure | generation_failure | chunking_failure"
    )


def run_config_evaluation(config: str) -> None:
    """Jalankan evaluasi lengkap untuk satu konfigurasi."""
    df = _load_ground_truth()
    results_df = _run_evaluation(df, config)

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_RESULTS_DIR / f"hasil_config_{config}.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Hasil disimpan ke: {out_path}")

    _print_aggregation(results_df, config)
    _export_error_analysis(results_df, config)


def run_wilcoxon_stats() -> None:
    """Jalankan uji Wilcoxon Signed-Rank antara Config A dan Config B."""
    path_a = EVAL_RESULTS_DIR / "hasil_config_a.csv"
    path_b = EVAL_RESULTS_DIR / "hasil_config_b.csv"

    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(
            "File hasil Config A dan/atau Config B tidak ditemukan!\n"
            "Jalankan terlebih dahulu:\n"
            "  python evaluation.py --config a\n"
            "  python evaluation.py --config b"
        )

    df_a = pd.read_csv(path_a, encoding="utf-8-sig")
    df_b = pd.read_csv(path_b, encoding="utf-8-sig")

    if len(df_a) != len(df_b):
        raise ValueError(
            f"Jumlah baris Config A ({len(df_a)}) dan Config B ({len(df_b)}) berbeda. "
            "Pastikan kedua evaluasi menggunakan ground truth yang sama."
        )

    stats_rows = []
    for metric in METRICS_COLS:
        if metric not in df_a.columns or metric not in df_b.columns:
            continue
        scores_a = df_a[metric].fillna(0).values
        scores_b = df_b[metric].fillna(0).values
        try:
            stat, p_val = wilcoxon(scores_a, scores_b)
            significant = bool(p_val < 0.05)
            winner = "Config B" if df_b[metric].mean() > df_a[metric].mean() else "Config A"
            if not significant:
                winner = "n.s."
        except Exception as e:
            stat, p_val, significant, winner = None, None, None, f"Error: {e}"

        stats_rows.append({
            "metric":             metric,
            "wilcoxon_statistic": stat,
            "p_value":            p_val,
            "significant_at_0.05": significant,
            "winner":             winner,
        })
        print(f"  {metric}: p={p_val:.4f} {'*' if significant else 'n.s.'} | winner: {winner}")

    stats_df = pd.DataFrame(stats_rows)
    out_path = EVAL_RESULTS_DIR / "statistical_test.csv"
    stats_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Hasil Wilcoxon disimpan ke: {out_path}")


def run_visualize() -> None:
    """
    Generate bar chart perbandingan 3 config + p-value annotation + latency line.
    Output: eval/results/perbandingan_visual.png
    """
    paths = {
        "Config A": EVAL_RESULTS_DIR / "hasil_config_a.csv",
        "Config B": EVAL_RESULTS_DIR / "hasil_config_b.csv",
        "Config C (BM25)": EVAL_RESULTS_DIR / "hasil_config_c.csv",
    }

    missing = [k for k, v in paths.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(
            f"File hasil berikut tidak ditemukan: {missing}\n"
            "Jalankan evaluasi terlebih dahulu untuk semua config."
        )

    dfs = {label: pd.read_csv(path, encoding="utf-8-sig") for label, path in paths.items()}

    # Baca p-value dari Wilcoxon jika ada
    stats_path = EVAL_RESULTS_DIR / "statistical_test.csv"
    pvalues: dict[str, str] = {}
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        for _, row in stats_df.iterrows():
            sig = "*" if row.get("significant_at_0.05") else "n.s."
            pvalues[row["metric"]] = f"p={row['p_value']:.3f}{sig}"

    available_metrics = [m for m in METRICS_COLS if all(m in df.columns for df in dfs.values())]
    n_metrics = len(available_metrics)
    configs   = list(dfs.keys())
    n_configs = len(configs)

    x         = range(n_metrics)
    bar_width = 0.22
    colors    = ["#7B2D2D", "#A84545", "#D4A0A0"]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a0a0a")
    ax1.set_facecolor("#2d1515")

    for i, (label, df) in enumerate(dfs.items()):
        means = [df[m].mean() for m in available_metrics]
        offset = (i - n_configs / 2 + 0.5) * bar_width
        bars = ax1.bar(
            [xi + offset for xi in x],
            means,
            width=bar_width,
            label=label,
            color=colors[i],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    # Anotasi p-value di atas bar Config A & B
    for i, metric in enumerate(available_metrics):
        if metric in pvalues:
            ax1.text(
                i,
                max(dfs[label][metric].mean() for label in dfs) + 0.02,
                pvalues[metric],
                ha="center",
                fontsize=8,
                color="#f0c0c0",
            )

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(available_metrics, rotation=15, ha="right", color="#f0e6e6")
    ax1.set_ylabel("Skor Rata-rata", color="#f0e6e6")
    ax1.set_ylim(0, 1.15)
    ax1.tick_params(colors="#f0e6e6")
    ax1.spines["bottom"].set_color("#7B2D2D")
    ax1.spines["left"].set_color("#7B2D2D")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Sumbu kedua: latency line
    if all("response_time_seconds" in df.columns for df in dfs.values()):
        ax2 = ax1.twinx()
        latencies = [dfs[label]["response_time_seconds"].mean() for label in configs]
        x_line = [sum(range(n_metrics)) / n_metrics] * n_configs  # posisi tengah
        ax2.plot(
            [i * bar_width * n_configs + bar_width * (n_configs / 2)
             for i in range(n_configs)],
            latencies,
            "o--",
            color="#FFD700",
            linewidth=1.5,
            markersize=6,
            label="Latensi (detik)",
        )
        ax2.set_ylabel("Latensi Rata-rata (detik)", color="#FFD700")
        ax2.tick_params(colors="#FFD700")
        ax2.spines["right"].set_color("#FFD700")

    ax1.set_title(
        "Perbandingan Performa Config A vs B vs C (BM25)\nSistem RAG Akademik UNSRAT",
        color="#f0e6e6",
        fontsize=13,
        pad=15,
    )
    ax1.legend(loc="upper right", framealpha=0.3, labelcolor="#f0e6e6")

    plt.tight_layout()
    out_path = EVAL_RESULTS_DIR / "perbandingan_visual.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Chart disimpan ke: {out_path}")


def main():
    """Entry point CLI untuk pipeline evaluasi."""
    parser = argparse.ArgumentParser(
        description="Pipeline evaluasi Ragas untuk UNSRAT RAG"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        choices=["a", "b", "c"],
        help="Evaluasi satu konfigurasi (a=RAG 500, b=RAG 2000, c=BM25)",
    )
    group.add_argument(
        "--stats",
        action="store_true",
        help="Uji Wilcoxon A vs B (jalankan setelah --config a dan --config b)",
    )
    group.add_argument(
        "--visualize",
        action="store_true",
        help="Generate bar chart (jalankan setelah semua --config selesai)",
    )
    args = parser.parse_args()

    if args.config:
        run_config_evaluation(args.config)
    elif args.stats:
        run_wilcoxon_stats()
    elif args.visualize:
        run_visualize()


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.3: Buat sample ground_truth.csv untuk testing**

Buat file `eval/dataset/ground_truth.csv` dengan minimal 2 baris untuk test:

```csv
user_input,reference,category,source_doc
"Apa visi Universitas Sam Ratulangi?","Visi UNSRAT adalah menjadi universitas unggulan...","institution_profile","02_visi_misi"
"Berapakah jumlah maksimal SKS yang dapat diambil mahasiswa per semester?","Sesuai Pasal 14 Peraturan Akademik UNSRAT 2025, mahasiswa dapat mengambil maksimal 24 SKS per semester...","academic","Peraturan_Akademik_UNSRAT_2025_RAG_REVISED"
```

- [ ] **Step 7.4: Verifikasi evaluation.py — Config B (sample)**

```bash
python evaluation.py --config b
```

Expected: muncul output log Ragas, kemudian file `eval/results/hasil_config_b.csv` terbuat.

- [ ] **Step 7.5: Verifikasi --stats (jalankan setelah Config A selesai)**

```bash
python evaluation.py --config a
python evaluation.py --stats
```

Expected: `eval/results/statistical_test.csv` terbuat.

- [ ] **Step 7.6: Verifikasi --visualize**

```bash
python evaluation.py --visualize
```

Expected: `eval/results/perbandingan_visual.png` terbuat.

- [ ] **Step 7.7: Verifikasi error handling CSV tidak valid**

```bash
python -c "
import subprocess, sys
result = subprocess.run(
    [sys.executable, 'evaluation.py', '--config', 'b'],
    capture_output=True, text=True,
    env={**__import__('os').environ, 'GROUND_TRUTH_MISSING': '1'}
)
print(result.stderr[:300])
"
```

Expected: error message dalam Bahasa Indonesia.

- [ ] **Step 7.8: Commit final**

```bash
git add evaluation.py eval/dataset/.gitkeep eval/results/.gitkeep
git commit -m "feat: tambah evaluation.py — Ragas 0.4.x + Wilcoxon + chart"
```

---

## Final Verification Checklist

Jalankan semua verifikasi ini sebelum menyatakan implementasi selesai:

- [ ] `python src/config.py` → tidak ada error
- [ ] `python src/ingestion.py --config b --rebuild` → log chunk count > 0
- [ ] `python src/ingestion.py --config a --rebuild` → log chunk count > 0
- [ ] `python src/bm25_retriever.py --rebuild` → pickle terbuat, chunk count > 9
- [ ] **Step 7.0: Verifikasi Ragas installation:** `pip show ragas` (pastikan v0.4.x)
- [ ] Config B retrieval: query relevan → 1-4 chunk
- [ ] Config B retrieval: query irrelevant → 0 chunk
- [ ] Config C retrieval: query apa pun → 4 chunk (BM25 selalu return)
- [ ] `get_response()` streaming=False → `{answer, stream: None, sources, found}`
- [ ] `get_response()` streaming=True → `{answer: None, stream: Generator, sources, found}` — SATU call
- [ ] Sources tersedia di KEDUA mode tanpa second API call
- [ ] Sources berisi field `content` (full) DAN `preview` (150 char)
- [ ] Query irrelevant → FALLBACK_RESPONSE, found=False, sources=[]
- [ ] `streamlit run app.py` → UI tampil tanpa error
- [ ] Streaming response berfungsi di UI (token per token)
- [ ] Expander sumber tampil setelah streaming (dari single call)
- [ ] Switch config → conversation reset
- [ ] `python evaluation.py --config a` → `hasil_config_a.csv` terbuat
- [ ] `python evaluation.py --config b` → `hasil_config_b.csv` terbuat
- [ ] `python evaluation.py --config c` → `hasil_config_c.csv` terbuat (baseline utama)
- [ ] `hasil_config_c.csv` berisi kolom: `faithfulness, answer_relevancy, context_precision, context_recall, response_time_seconds`
- [ ] CSV `retrieved_contexts` berisi full chunk content (bukan 150 char preview)
- [ ] **NFR-02:** Setiap file memiliki satu tanggung jawab utama. Khusus `chain.py`: verifikasi `detect_category()` tidak mengakses database/LLM (murni keyword matching — tetap boleh ada di chain.py)
- [ ] **NFR-05:** Setiap fungsi di semua file memiliki docstring

---

## Self-Review

**Spec coverage check:**
- FR-01 (Parsing Corpus) → Task 2 `parse_markdown_file()`  ✅
- FR-02 (Validasi Frontmatter) → Task 2, `REQUIRED_YAML_FIELDS` dari config ✅
- FR-03 (Two-Stage Chunking) → Task 2 `split_content()` ✅
- FR-04 (Summary Chunk) → Task 2 `create_summary_chunk()` ✅
- FR-05 (Chunk Filtering) → Task 2, filter < MIN_CHUNK_LENGTH ✅
- FR-06 (Idempotent) → Task 2 `generate_chunk_id()` + collection.get() ✅
- FR-07 (task_type) → Task 2 retrieval_document, Task 4 retrieval_query ✅
- FR-08 (Dual Collection) → Task 1/2, CHROMA_DIR_A + CHROMA_DIR_B ✅
- FR-09 (Category Detection) → Task 5/6 `detect_category()`, label UI saja ✅
- FR-10 (Similarity Threshold) → Task 4 `retrieve()`, filter > SIMILARITY_THRESHOLD ✅
- FR-11 (Fallback Response) → Task 5 `get_response()`, found=False ✅
- FR-12 (Dual-Mode Response) → Task 5, streaming=True/False ✅
- FR-13 (Manual Memory) → Task 5, `_format_history()` + `messages[-(MEMORY_K*2):]` ✅
- FR-14 (Eval Memory Reset) → Task 7, `chat_history = []` per pertanyaan ✅
- FR-15 (Ragas Evaluation) → Task 7, 4 metrik dari METRICS_COLS ✅
- FR-16 (Aggregation) → Task 7 `_print_aggregation()` ✅
- FR-17 (Visualization) → Task 7 `run_visualize()` ✅
- FR-18 (API Retry) → Task 5 `_invoke_with_retry()` + `_stream_with_retry()` ✅
- FR-19 (BM25 Indexing) → Task 3 `build_index()` + --rebuild ✅
- FR-20 (BM25 Retrieval) → Task 3 `retrieve_chunks_bm25()` ✅
- FR-21 (Latensi) → Task 7, `time.time()` sebelum/sesudah `get_response()` ✅
- FR-22 (Wilcoxon) → Task 7 `run_wilcoxon_stats()` ✅
- FR-23 (Error Analysis) → Task 7 `_export_error_analysis()` ✅
- FR-24 (Model Switcher) → Task 5 `reinitialize_llm()`, Task 6 sidebar ✅
- FR-25 (Config C Eval) → Task 7, `--config c` ✅
- NFR-01 (Modularitas) → semua logika di src/, app.py hanya UI ✅
- NFR-03 (No Magic Numbers) → semua angka dari config.py ✅
- NFR-04 (Secret Management) → `.env` + `load_dotenv()` ✅
- NFR-05 (Docstring) → setiap fungsi punya docstring ✅
- NFR-07 (Reproducibility) → LLM_TEMPERATURE=0.1, memory reset ✅
- NFR-08 (Logging) → logging di ingestion.py ✅
- NFR-09 (Graceful Degradation) → retry policy, fallback response ✅

**Placeholder scan:** Tidak ada TBD atau TODO dalam plan ini.

**Type consistency:**
- `retrieve()` mengembalikan `list[dict]` dengan keys: `content`, `metadata`, `score`
- `get_response()` mengembalikan `dict` dengan keys: `answer`, `sources`, `found`
- `sources[i]` memiliki keys: `doc_id`, `title`, `bab`, `bagian`, `pasal`, `content`, `preview`, `category`
- `retrieved_contexts` di evaluation menggunakan `s["content"]` ✅ (bukan `s["preview"]`)
