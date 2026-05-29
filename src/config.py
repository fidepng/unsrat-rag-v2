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
LOGS_DIR             = ROOT_DIR / "logs"
SYSTEM_LOG_PATH      = LOGS_DIR / "unsrat_rag.log"
CHAT_LOG_PATH        = LOGS_DIR / "transaksi_chat.csv"
INGESTION_LOG_PATH   = LOGS_DIR / "ingestion_report.csv"


# ── CHROMADB COLLECTIONS ────────────────────────────────────────────────────
CHROMA_COLLECTION_A = "unsrat_rag_config_a"
CHROMA_COLLECTION_B = "unsrat_rag_config_b"
CHROMA_DISTANCE_FN  = "cosine"

# ── MODEL ───────────────────────────────────────────────────────────────────
# Generator dan evaluator HARUS BERBEDA (mitigasi self-eval bias — D-16).
# Ganti dengan nama model yang tersedia di akun GCP Anda.
LLM_MODEL_NAME       = "qwen/qwen3.5-397b-a17b"
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EVALUATOR_MODEL_NAME = "meta/llama-3.1-70b-instruct"

# Daftar model yang bisa dipilih di UI sidebar.
AVAILABLE_MODELS: list[str] = [
    "qwen/qwen3.5-397b-a17b",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
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
LLM_MAX_OUTPUT_TOKENS = 2048
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
# Definisi ada HANYA di sini — ingestion.py and bm25_retriever.py mengimport dari sini.
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
