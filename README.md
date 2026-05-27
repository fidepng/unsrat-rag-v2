# 🎓 UNSRAT RAG — Sistem Chatbot Informasi Akademik

> **Prototipe penelitian Tugas Akhir** — Sistem tanya-jawab berbasis
> Retrieval-Augmented Generation (RAG) untuk civitas akademika Universitas Sam Ratulangi.

---

## 📖 Deskripsi Singkat

Chatbot ini menjawab pertanyaan seputar:

- Peraturan Akademik UNSRAT (Peraturan Rektor No. 01 Tahun 2025)
- Kalender Akademik Semester Genap 2025/2026
- Profil, Sejarah, Visi-Misi, dan Identitas Institusi UNSRAT

Sistem ini membandingkan **dua strategi pemotongan dokumen (chunking)**:

| Config | Chunk Size    | Overlap | Tujuan                       |
| ------ | ------------- | ------- | ---------------------------- |
| **A**  | 500 karakter  | 100     | Eksperimen (ukuran kecil)    |
| **B**  | 2000 karakter | 200     | Best practice (ukuran besar) |

---

## 📋 Prasyarat

Sebelum mulai, pastikan sudah tersedia:

- [ ] **Miniconda** atau **Anaconda** terinstall di komputer
- [ ] **API Key** dari Google AI Studio → [ai.google.dev](https://ai.google.dev)
- [ ] **GCP Budget Alert** sudah diaktifkan ($50 dan $250 threshold)
- [ ] Semua file corpus `.md` sudah ada di folder `data/corpus/`

---

## ⚡ Quick Start (Jalankan Berurutan)

### 1. Buat Conda Environment

```bash
conda env create -f environment.yml
conda activate unsrat-rag
```

### 2. Buat File `.env`

Buat file baru bernama `.env` di folder root proyek (sejajar dengan `app.py`):

```
GOOGLE_API_KEY=isi_dengan_api_key_anda_di_sini
```

> ⚠️ File `.env` TIDAK boleh di-commit ke Git. Sudah ada di `.gitignore`.

### 3. Bangun Database Vektor (Ingestion)

**Config A & B (RAG dengan ChromaDB):**

```bash
python src/ingestion.py --config a --rebuild
python src/ingestion.py --config b --rebuild
```

**Config C (BM25 — tidak membutuhkan embedding/ChromaDB):**

```bash
python src/bm25_retriever.py --rebuild
```

> ℹ️ Flag `--rebuild` menghapus database lama dan membangun ulang dari nol.
> Gunakan ini setiap kali ada perubahan pada file corpus.

### 4. Jalankan Chatbot

```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`. Gunakan sidebar untuk memilih
konfigurasi aktif (Config A atau B).

### 5. Jalankan Evaluasi (Opsional)

Pastikan `eval/dataset/ground_truth.csv` sudah diisi terlebih dahulu.

```bash
# Evaluasi masing-masing config
python evaluation.py --config a
python evaluation.py --config b
python evaluation.py --config c     # BM25 baseline

# Uji signifikansi statistik (Wilcoxon A vs B)
python evaluation.py --stats

# Visualisasi perbandingan ketiga config
python evaluation.py --visualize
```

Setelah evaluasi selesai, isi kolom `failure_type` secara manual
di file `eval/results/error_analysis_config_*.csv`.

---

## 📁 Struktur Proyek

```
unsrat-rag/                          ← Root folder proyek
│
├── data/                            ← Semua aset data
│   └── corpus/                      ← File Markdown knowledge base
│       ├── 01_sejarah.md
│       ├── 02_visi_misi.md
│       ├── 03_tujuan_sasaran_strategi.md
│       ├── 04_lambang.md
│       ├── 05_bendera.md
│       ├── 06_mars_hymne.md
│       ├── 07_akreditasi.md
│       ├── Peraturan_Akademik_UNSRAT_2025_RAG_REVISED.md
│       ├── Kalender_Akademik_UNSRAT_Genap_2025-2026.md
│       └── [PENDING] faq.md dan file tambahan
│
├── chroma_db/                       ← Database ChromaDB (auto-generated)
│   ├── config_a/                    ← Collection untuk Config A
│   └── config_b/                    ← Collection untuk Config B
│
├── bm25_index/                      ← Indeks BM25 untuk Config C (auto-generated)
│   └── bm25_index.pkl               ← Serialized BM25 index + corpus
│
├── eval/                            ← Semua aset evaluasi penelitian
│   ├── dataset/
│   │   └── ground_truth.csv         ← Dataset uji 50++ pasang Q&A
│   └── results/
│       ├── hasil_config_a.csv       ← Output evaluasi Config A
│       ├── hasil_config_b.csv       ← Output evaluasi Config B
│       ├── hasil_config_c.csv       ← Output evaluasi Config C (BM25)
│       ├── statistical_test.csv     ← Hasil uji Wilcoxon (p-values)
│       ├── error_analysis_config_a.csv  ← Analisis kegagalan Config A
│       ├── error_analysis_config_b.csv  ← Analisis kegagalan Config B
│       └── perbandingan_visual.png  ← Bar chart 3 config (via --visualize)
│
├── src/                             ← Source code modular. Semua logika bisnis (framework-agnostic)
│   ├── __init__.py                  ← File kosong (wajib ada)
│   ├── config.py                    ← SEMUA parameter konfigurasi terpusat
│   ├── ingestion.py                 ← Pipeline data ingestion → ChromaDB
│   ├── retriever.py                 ← Logika retrieval dari ChromaDB
│   ├── bm25_retriever.py            ← BM25 indexing & retrieval (Config C)
│   └── chain.py                     ← Logika RAG (retriever + LLM)
│
├── .env                             ← API Keys & secret variables (TIDAK BOLEH di-commit ke Git)
├── .gitignore
├── environment.yml                  ← Definisi Conda environment
├── app.py                           ← Entry point demo UI Streamlit (di root, bukan di src/)
└── evaluation.py                    ← Pipeline evaluasi Ragas (CLI)
```

---

## 🛠️ Perintah Berguna

```bash
# Verifikasi instalasi library
python -c "import langchain; print('LangChain OK:', langchain.__version__)"
python -c "import chromadb; print('ChromaDB OK:', chromadb.__version__)"
python -c "import ragas; print('Ragas OK:', ragas.__version__)"
python -c "import streamlit; print('Streamlit OK:', streamlit.__version__)"

# Verifikasi BM25 dan scipy
python -c "from rank_bm25 import BM25Okapi; print('rank-bm25 OK')"
python -c "from scipy import stats; print('scipy OK')"

# Cek model yang aktif (verifikasi tidak ada placeholder)
python -c "
from src.config import LLM_MODEL_NAME, EVALUATOR_MODEL_NAME
print('Generator:', LLM_MODEL_NAME)
print('Evaluator:', EVALUATOR_MODEL_NAME)
assert LLM_MODEL_NAME != EVALUATOR_MODEL_NAME, 'ERROR: Generator = Evaluator!'
assert '[ISI' not in LLM_MODEL_NAME, 'ERROR: Placeholder belum diisi!'
print('✓ Konfigurasi model valid.')
"

# Cek jumlah chunk di database (sanity check setelah ingestion)
python -c "
import chromadb
from src.config import CHROMA_DIR, CHROMA_COLLECTION_A, CHROMA_COLLECTION_B
client = chromadb.PersistentClient(path=str(CHROMA_DIR / 'config_a'))
col = client.get_collection(CHROMA_COLLECTION_A)
print('Config A chunks:', col.count())
"
```

---

## ⚠️ Catatan Penting

1. **Jangan jalankan `ingestion.py` tanpa `--config`** — akan error karena
   wajib memilih konfigurasi.

2. **Evaluasi mengabaikan memori percakapan** — setiap pertanyaan evaluasi
   diproses mandiri tanpa konteks dari pertanyaan sebelumnya. Ini disengaja
   agar hasil evaluasi murni.

3. **Biaya API** — Ingestion (~9 dokumen) dan evaluasi (30-50 Q&A × 2 config)
   akan mengonsumsi kuota API. Pantau di GCP Console.

4. **Bukan sistem produksi** — Proyek ini adalah prototipe penelitian.
   Tidak ada autentikasi, tidak ada deployment, tidak ada SLA.

5. **Isi placeholder model sebelum mulai** — `config.py` berisi `[ISI_NAMA_MODEL_*]`
   sebagai placeholder. Ganti dengan nama model nyata dari GCP/AI Studio
   sebelum menjalankan kode apapun.

6. **Generator ≠ Evaluator** — `LLM_MODEL_NAME` dan `EVALUATOR_MODEL_NAME`
   di `config.py` HARUS berbeda. Ini memitigasi self-evaluation bias.

7. **Verifikasi import Ragas** — Sebelum implementasi `evaluation.py`,
   jalankan `pip show ragas` dan baca dokumentasi versi yang terinstall.
   Nama class/instance metrik berubah antar versi minor.

---

## 📚 Dokumentasi Lebih Lanjut

- **Spesifikasi & arsitektur lengkap:** `prd+srs.md`
- **Panduan implementasi (untuk AI/dev):** `agents.md`
- **Template analisis Bab IV:** `prd+srs.md` → Section 19
- **Standar YAML metadata corpus:** `UNSRAT_YAML_STANDARD_v2.md`

> 🤖 **Untuk AI Coding Agent:** Jika Anda membutuhkan dokumentasi resmi
> library atau framework (Ragas, LangChain, rank-bm25, dsb.), **jangan
> mengasumsikan nama API dari memori**. Minta user untuk memberikan link
> dokumentasi yang relevan. Tujuan: mencegah halusinasi dan bad practice
> akibat perubahan API antar versi.
