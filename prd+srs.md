# PROJECT REQUIREMENTS & SPECIFICATION DOCUMENT (PRD + SRS)

## Sistem Chatbot Informasi Akademik UNSRAT Berbasis RAG

**Nama Mahasiswa:** Teofide W. K. Pangemanan
**NIM:** 220211060317
**Program Studi:** Informatika / Ilmu Komputer
**Universitas:** Universitas Sam Ratulangi (UNSRAT) Manado

---

## KONTROL DOKUMEN

| Versi   | Tanggal     | Perubahan                                                                                                                                                                        |
| ------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0     | 2025-12     | Draft awal                                                                                                                                                                       |
| 2.0     | 2026-01     | Revisi arsitektur                                                                                                                                                                |
| 3.0     | 2026-05     | Revisi komprehensif (PRD)                                                                                                                                                        |
| 4.0     | 2026-05     | Integrasi YAML Standard v2, perbaikan bug kritis, tambahan SRS                                                                                                                   |
| **5.0** | **2026-06** | **Tambah Config C (BM25), 5 metrik Ragas, pengukuran latensi, UI model switcher, uji Wilcoxon, analisis kegagalan, mitigasi self-eval bias, Section 19 Rencana Analisis Bab IV** |

> **PERINGATAN:** Dokumen ini adalah Sumber Kebenaran Tunggal (Single Source
> of Truth). Setiap keputusan teknis, nama file, nama variabel, dan nilai
> parameter yang tercantum di sini adalah STANDAR MUTLAK. Tidak boleh ada
> deviasi tanpa revisi tertulis pada dokumen ini terlebih dahulu.

---

## DAFTAR ISI

**BAGIAN A — PRD (Product Requirements Document)**

1. Ringkasan Eksekutif & Konteks
2. Filosofi Pengembangan
3. Tech Stack & Versi
4. Struktur Folder (Canonical)
5. Spesifikasi Corpus & Standar YAML
6. Arsitektur Sistem & Alur Kerja
7. Spesifikasi Konfigurasi & Parameter
8. Spesifikasi File `.env`
9. Format Data Evaluasi
10. Spesifikasi UI (Streamlit — Default Demo)
11. Spesifikasi Pipeline Evaluasi Ragas
12. Aturan Kode & Hard Constraints
13. Spesifikasi Error Handling
14. Panduan Setup Environment
15. Daftar Keputusan Arsitektur (Decision Log)

**BAGIAN B — SRS (Software Requirements Specification)** 16. Functional Requirements (FR) 17. Non-Functional Requirements (NFR) 18. Constraints & Assumptions

---

# BAGIAN A — PRD

---

## 1. RINGKASAN EKSEKUTIF & KONTEKS

### 1.1 Deskripsi & Tujuan

Sistem tanya-jawab (Question-Answering) berbasis RAG untuk menjawab
pertanyaan civitas akademika UNSRAT tentang:

- Sejarah, profil, visi-misi, identitas institusi
- Peraturan Akademik resmi (Peraturan Rektor No. 01 Tahun 2025)
- Kalender Akademik Semester Genap 2025/2026
- Frequently Asked Questions dalam lingkungan Universitas (Status: Pending, setelah file tersedia)
- Informasi Umum Lainnya (Status: Pending)

**Sifat Proyek:** Prototipe penelitian Tugas Akhir/Skripsi. BUKAN sistem produksi.
**Fokus Penelitian:** Implementasi RAG + evaluasi kinerja _dua dimensi_:
(1) **Kualitas jawaban** — diukur via Ragas (faithfulness, relevancy, precision, recall).
(2) **Performa sistem** — diukur via latensi respons (`response_time_seconds`).
**Metode Ilmiah:** Studi komparasi tiga konfigurasi:

- Config A: RAG chunking kecil (500 char)
- Config B: RAG chunking besar (2000 char) — _kandidat terbaik_
- Config C: BM25 murni (keyword search, tanpa embedding) — _baseline komparasi_

### 1.2 Batasan Masalah (Scope)

**DALAM RUANG LINGKUP:**

- Sistem tanya-jawab berbasis dokumen yang sudah tersedia
- Implementasi dan perbandingan dua strategi chunking
- Evaluasi kuantitatif menggunakan framework Ragas

**DI LUAR RUANG LINGKUP (DILARANG):**

- Fitur transaksional (pengisian KRS, pendaftaran, dsb.)
- Manajemen akun / sistem login
- Integrasi dengan sistem informasi akademik UNSRAT yang ada
- Deployment ke server publik / production environment
- Fitur Reranking (Cohere, Jina, dsb.) — tidak perlu untuk skala dokumen ini
- Google Search Grounding / web browsing

---

## 2. FILOSOFI PENGEMBANGAN

> **Prinsip Utama:** Setiap keputusan teknis harus melewati filter ini:
> _"Apakah kompleksitas tambahan ini secara langsung meningkatkan kualitas penelitian atau
> kemudahan debugging? Jika tidak,jangan tambahkan."_

| Fitur                     | Status           | Alasan                                                     |
| ------------------------- | ---------------- | ---------------------------------------------------------- |
| Reranking (Cohere/Jina)   | ❌ Tidak dipakai | Overkill untuk < 100 halaman                               |
| Hybrid BM25 + Vector      | ❌ Tidak dipakai | Metadata filter sudah cukup (lihat D-09)                   |
| **BM25 murni (Config C)** | **✅ Dipakai**   | **Baseline komparasi ilmiah. Satu file ~50 baris (D-09b)** |
| Persistent user sessions  | ❌ Tidak dipakai | Fokus evaluasi RAG, bukan UX                               |
| Database SQL chat history | ❌ Tidak dipakai | In-memory sudah cukup                                      |
| Google Search Grounding   | ❌ Dilarang      | Biaya tambahan + keluar dari scope RAG                     |
| Multi-language support    | ❌ Tidak dipakai | Fokus Bahasa Indonesia                                     |

---

## 3. TECH STACK & VERSI

### 3.1 Runtime & Environment

| Komponen        | Spesifikasi                | Catatan                          |
| --------------- | -------------------------- | -------------------------------- |
| Bahasa          | Python 3.11                | Wajib 3.11                       |
| Package Manager | Conda (Miniconda/Anaconda) | BUKAN venv atau pip global       |
| Nama Conda Env  | `unsrat-rag`               | Nama standar, tidak boleh diubah |
| OS Pengembangan | Windows 11 Home 64-bit     |                                  |
| RAM             | 16 GB                      |                                  |
| GPU             | NVIDIA RTX 3050 Laptop     | Tidak dipakai (semua via API)    |

### 3.2 Library Python (`environment.yml`)

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

> **Catatan:** `rank-bm25` digunakan khusus untuk Config C (BM25 baseline).
> `scipy` digunakan untuk `scipy.stats.wilcoxon` di pipeline evaluasi.
> gunakan `ragas` versi stabel (terbaru), minta dokumentasi kepada user

### 3.3 Model AI

| Komponen                    | Model                    | Provider         |
| --------------------------- | ------------------------ | ---------------- |
| LLM Utama (RAG Generation)  | `gemini-3.5-flash`       | Google AI Studio |
| Embedding                   | `gemini-embedding-001`   | Google AI Studio |
| LLM Evaluator (Ragas Judge) | `gemini-3.1-pro-preview` | Google AI Studio |

> **CATATAN MODEL:** Peneliti menggunakan `gemini-3.5-flash` sebagai model utama dan
> berencana bereksperimen dengan model lain yang tersedia di GCP untuk menemukan
> konfigurasi terbaik. Penggunaan saldo kredit GCP $300 (90 hari) memberikan
> fleksibilitas untuk eksplorasi model berbayar. Pastikan setiap pergantian model
> dicatat di jurnal penelitian untuk keperluan reproduktibilitas.
> Jika model tidak tersedia, fallback ke `gemini-2.5-flash`.

> **MITIGASI SELF-EVALUATION BIAS (D-16):** Generator dan evaluator HARUS
> menggunakan model yang berbeda. Ini mengeliminasi kelemahan metodologis
> di mana model menilai output dirinya sendiri.
> Dokumentasikan pilihan model di laporan penelitian (untuk reproduktibilitas).
> Jika model tidak tersedia, fallback ke versi flash terdekat.

### 3.4 Keamanan Biaya

- **Budget Alert:** $50 (peringatan), $250 (kritis)
- **Hard Cap:** $280 (sisakan $20 buffer)
- **Dilarang:** Google Search Grounding ($14/1000 req)

---

## 4. STRUKTUR FOLDER (CANONICAL)

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
│   │   └── ground_truth.csv         ← Dataset uji 30-50 pasang Q&A
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

### 4.1 Tanggung Jawab Tiap File

| File                    | Tanggung Jawab Tunggal                                  | Dipanggil Oleh            |
| ----------------------- | ------------------------------------------------------- | ------------------------- |
| `src/config.py`         | Menyimpan SEMUA parameter & konstanta                   | Semua file                |
| `src/ingestion.py`      | Baca .md → chunk → embed → ChromaDB (Config A & B)      | CLI langsung              |
| `src/retriever.py`      | Terima Query → ChromaDB → return chunks (Config A & B)  | `src/chain.py`            |
| `src/bm25_retriever.py` | BM25 indexing & retrieval tanpa embedding (Config C)    | `src/chain.py`, CLI       |
| `src/chain.py`          | Rangkai retriever + memory + LLM menjadi satu RAG chain | `app.py`, `evaluation.py` |
| `app.py`                | Render UI, handle interaksi pengguna                    | `streamlit run`           |
| `evaluation.py`         | Load CSV, evaluasi Ragas, statistik, simpan hasil       | CLI langsung              |

> **Prinsip Modularitas UI:** Semua logika bisnis (RAG, retrieval, LLM)
> WAJIB ada di dalam `src/`. File `app.py` hanya boleh berisi kode UI.
> Ini memungkinkan UI diganti framework lain (FastAPI, Gradio, dsb.)
> di masa depan tanpa menyentuh kode inti di `src/`.

---

## 5. SPESIFIKASI CORPUS & STANDAR YAML

### 5.1 Daftar Dokumen (Corpus Master List)

| #   | Nama File                                       | doc_id                  | category            | content_type | Priority | Status     |
| --- | ----------------------------------------------- | ----------------------- | ------------------- | ------------ | -------- | ---------- |
| 1   | `01_sejarah.md`                                 | UNSRAT-PROFILE-2020-001 | institution_profile | narrative    | 4        | ✅ Siap    |
| 2   | `02_visi_misi.md`                               | UNSRAT-PROFILE-2020-002 | institution_profile | narrative    | 4        | ✅ Siap    |
| 3   | `03_tujuan_sasaran_strategi.md`                 | UNSRAT-PROFILE-2020-003 | institution_profile | narrative    | 4        | ✅ Siap    |
| 4   | `04_lambang.md`                                 | UNSRAT-PROFILE-2020-004 | institution_profile | narrative    | 4        | ✅ Siap    |
| 5   | `05_bendera.md`                                 | UNSRAT-PROFILE-2020-005 | institution_profile | narrative    | 4        | ✅ Siap    |
| 6   | `06_mars_hymne.md`                              | UNSRAT-PROFILE-2020-006 | institution_profile | narrative    | 4        | ✅ Siap    |
| 7   | `07_akreditasi.md`                              | UNSRAT-PROFILE-2020-007 | institution_profile | narrative    | 3        | ✅ Siap    |
| 8   | `Peraturan_Akademik_UNSRAT_2025_RAG_REVISED.md` | UNSRAT-REG-2025-001     | academic            | regulation   | 1        | ✅ Siap    |
| 9   | `Kalender_Akademik_UNSRAT_Genap_2025-2026.md`   | UNSRAT-CAL-2026-001     | calendar            | calendar     | 2        | ✅ Siap    |
| 10+ | `faq.md`                                        | UNSRAT-FAQ-001          | faq                 | guide        | 2        | ⏳ PENDING |

> **Catatan:** Nilai field `category` menggunakan English enum sesuai YAML
> Standard v2.0 (Inggris). Extension khusus proyek ini: nilai `"calendar"`
> dan `"faq"` ditambahkan ke enum `category` karena diperlukan untuk
> metadata filter retrieval.

### 5.2 Standar YAML Frontmatter v2.0 (Wajib untuk Semua File .md)

Setiap file corpus WAJIB memiliki YAML frontmatter persis seperti ini.
Ikuti standar `UNSRAT_YAML_STANDARD_v2.md` untuk panduan lengkap.

**Contoh lengkap (`Peraturan_Akademik_UNSRAT_2025_RAG_REVISED.md`):**

```yaml
---
# BLOK 1 — IDENTITAS DOKUMEN
doc_id: "UNSRAT-REG-2025-001"
title: "Peraturan Rektor UNSRAT Nomor 01 Tahun 2025 tentang Peraturan Akademik"
version: "1.0"
language_primary: "id"
language_secondary: null
institution: "Universitas Sam Ratulangi"
unit_penerbit: "Rektorat"

# BLOK 2 — KLASIFIKASI KONTEN
content_type: "regulation"
category: "academic"
subcategory:
  - "perkuliahan"
  - "evaluasi"
  - "wisuda"
audience:
  - "mahasiswa_s1"
  - "dosen"
access_level: "public"

# BLOK 3 — LEGALITAS & SUMBER
nomor_sk: "01/2025"
tanggal_penetapan: "2025-02-13"
pejabat_penandatangan: "Oktovian Berty Alexander Sompie, Rektor UNSRAT"
source_document: "Peraturan-Rektor-Unsrat-Nomor-1-tahun-2025.pdf"
source_url: null
supersedes: null
superseded_by: null

# BLOK 4 — VALIDITAS TEMPORAL
valid_from: "2025-02-13"
valid_until: null
status: "active"
last_updated: "2025-12-16"
last_verified: "2025-12-16"

# BLOK 5 — METADATA RAG (KRITIS)
retrieval_summary: "Peraturan Rektor UNSRAT Nomor 01 Tahun 2025 mengatur
  seluruh aspek akademik program sarjana di Universitas Sam Ratulangi,
  mencakup sistem kredit semester (SKS), pengisian KRS, evaluasi, cuti
  akademik, yudisium, dan wisuda. Berlaku bagi seluruh mahasiswa S1 dan
  dosen mulai Februari 2025."
chunk_strategy: "by_section"
chunk_notes: "Unit atomik adalah Pasal. Jangan memotong di tengah Pasal.
  Tabel di Pasal 1 harus satu chunk. Ayat-ayat dalam satu Pasal tetap bersama."
embedding_model: "gemini-embedding-001"
priority: 1
related_docs:
  - "UNSRAT-CAL-2026-001"

# BLOK 6 — INDEXING & PENCARIAN
tags:
  - peraturan_akademik
  - beban_belajar
  - kelulusan
  - wisuda
keywords:
  - "SKS"
  - "KRS"
  - "IPK"
  - "IPS"
  - "DO"
  - "drop out"
  - "cuti akademik"
  - "yudisium"
  - "Portal INSPIRE"
entities:
  - "Oktovian Berty Alexander Sompie"
  - "Portal INSPIRE"
  - "Senat UNSRAT"
---
```

**Field Kritis yang WAJIB Selalu Terisi:**

| Field               | Wajib? | Catatan                                      |
| ------------------- | ------ | -------------------------------------------- |
| `doc_id`            | ✅     | Format: `UNSRAT-{TYPE}-{YYYY}-{NNN}`         |
| `title`             | ✅     | Sama persis dengan judul di dokumen asli     |
| `category`          | ✅     | Digunakan sebagai ChromaDB metadata filter   |
| `content_type`      | ✅     | Digunakan sebagai ChromaDB metadata filter   |
| `valid_from`        | ✅     | ISO 8601                                     |
| `status`            | ✅     | Enum: `"active"` / `"inactive"` / `"draft"`  |
| `retrieval_summary` | ✅     | KRITIS — Dijadikan chunk mandiri di ChromaDB |
| `chunk_strategy`    | ✅     | Instruksi pemotongan untuk ingestion.py      |
| `last_updated`      | ✅     | ISO 8601                                     |

> **FIELD KRITIS `retrieval_summary`:** Field ini WAJIB diisi untuk setiap dokumen.
> Isinya akan dijadikan chunk mandiri bertipe "summary" yang di-embed ke ChromaDB
> secara terpisah. Ini membantu sistem mengenali konteks umum dokumen bahkan sebelum
> menemukan Pasal/Bagian spesifik yang relevan.

### 5.3 Format Dokumen FAQ

```markdown
---
doc_id: "UNSRAT-FAQ-001"
title: "FAQ Informasi Akademik UNSRAT"
version: "1.0"
language_primary: "id"
institution: "Universitas Sam Ratulangi"
unit_penerbit: "Rektorat"
content_type: "guide"
category: "faq"
access_level: "public"
valid_from: "2025-01-01"
status: "active"
last_updated: "2025-12-16"
last_verified: "2025-12-16"
retrieval_summary: "Kumpulan pertanyaan yang sering diajukan mahasiswa
  UNSRAT beserta jawabannya. Mencakup prosedur KRS, yudisium, cuti,
  dan informasi umum kampus."
chunk_strategy: "by_section"
chunk_notes: "Unit atomik adalah satu pasang Q&A. Jangan memisahkan
  pertanyaan dari jawabannya. Setiap FAQ-NNN adalah satu chunk."
embedding_model: "gemini-embedding-001"
priority: 2
related_docs: []
tags:
  - faq
  - informasi_umum
keywords:
  - "KRS"
  - "cuti"
  - "yudisium"
entities: []
---

# FAQ Informasi Akademik UNSRAT

### FAQ-001 — Pertanyaan singkat sebagai heading

Jawaban lengkap di sini...

### FAQ-002 — Pertanyaan berikutnya

Jawaban...
```

---

## 6. ARSITEKTUR SISTEM & ALUR KERJA

### 6.1 Gambaran Keseluruhan

```
[File .md + YAML Frontmatter]
          │
          ▼
┌──────────────────────────────────────────────-------------------┐
│  PIPELINE INGESTION (src/ingestion.py)                          │
│  Dijalankan SEKALI per config via CLI                           │
│                                                                 │
│  1. Parse YAML frontmatter → ekstrak metadata                   │
│  2. Buat "summary chunk" dari field retrieval_summary           │
│  3. MarkdownHeaderTextSplitter (struktur) → potong per heading  │
│  4. RecursiveCharacterTextSplitter (ukuran) → normalisasi size  │
│  5. Filter: buang chunk < MIN_CHUNK_LENGTH                      │
│  6. Cek hash per chunk → skip jika duplikat                     │
│  7. Embed via gemini-embedding-001                              │
│     (task_type = "retrieval_document")                          │
│  8. Simpan ke ChromaDB collection terpisah                      │
│     (config_a atau config_b)                                    │
└──────────────────────────────────────────────-------------------┘
          │
          ▼
  [chroma_db/config_a/ atau chroma_db/config_b/]
          │
          ▼
┌──────────────────────────────────────────────┐
│  PIPELINE RAG (src/chain.py)                 │
│  Dijalankan setiap query                     │
│                                              │
│  Input: query + chat_history (list)          │
│       │                                      │
│       ▼                                      │
│  [Detect category dari query — keyword match]│
│       │                                      │
│       ▼                                      │
│  [ChromaDB Search: k=4, threshold≥0.35]      │
│  (task_type = "retrieval_query" untuk embed) │
│       │                                      │
│  ┌────┴────────────────────────────────────┐ │
│  │ Tidak ada chunk lolos threshold?        │ │
│  │ → Return pesan "tidak ditemukan"        │ │
│  │   TANPA memanggil LLM                   │ │
│  └─────────────────────────────────────────┘ │
│       │ Ada chunk yang lolos                 │
│       ▼                                      │
│  [Gabung chunk + trim chat_history (5 turn)] │
│  [Kirim ke Gemini → response]                │
│  [Return: response + retrieved_chunks]       │
└──────────────────────────────────────────────┘
```

### 6.2 Strategi Chunking (Two-Stage Hybrid Splitter)

**Tahap 1 — Structural Split (MarkdownHeaderTextSplitter):**

```python
# Hierarki heading yang dijadikan acuan pemotongan
headers_to_split_on = [
    ("#",    "header_1"),   # Judul dokumen utama
    ("##",   "bab"),        # Contoh: "## BAB I — KETENTUAN UMUM"
    ("###",  "bagian"),     # Contoh: "### Pasal 1 — Definisi Istilah"
    ("####", "pasal"),      # Sub-pasal jika ada
]
```

**Tahap 2 — Size Normalization (RecursiveCharacterTextSplitter):**

|                     | Config A              | Config B              |
| ------------------- | --------------------- | --------------------- |
| `chunk_size`        | 500                   | 2000                  |
| `chunk_overlap`     | 100                   | 200                   |
| Tujuan              | Eksperimen (kecil)    | Best practice (besar) |
| ChromaDB collection | `unsrat_rag_config_a` | `unsrat_rag_config_b` |
| ChromaDB folder     | `chroma_db/config_a/` | `chroma_db/config_b/` |

**Tahap Khusus — Summary Chunk:**
Untuk setiap dokumen, buat satu chunk tambahan dari field `retrieval_summary`.
Chunk ini diberi metadata `chunk_type = "summary"` dan field `bab`, `bagian`,
`pasal` diisi dengan nilai default `"Ringkasan Dokumen"`.

> **CATATAN PENTING tentang chunk_notes:** Setiap dokumen memiliki field `chunk_notes`
> di YAML yang berisi instruksi pemotongan spesifik. Contoh dari Peraturan Akademik:
> _"Unit atomic adalah Pasal. Jangan memotong di tengah Pasal."_
> Ini berarti `chunk_size: 500` pada Config A berisiko memotong tabel definisi
> di Pasal 1 yang sangat panjang. Ini adalah variabel penelitian yang DISENGAJA
> untuk menunjukkan degradasi performa pada Config A vs Config B.

### 6.3 Metadata yang Disimpan per Chunk di ChromaDB

```python
# Metadata wajib yang harus ada di setiap chunk:
{
    "doc_id":       str,   # Contoh: "UNSRAT-REG-2025-001"
    "title":        str,   # Judul dokumen
    "category":     str,   # Digunakan untuk filter: "academic", "calendar", dll.
    "content_type": str,   # Contoh: "regulation", "calendar"
    "chunk_type":   str,   # "content" atau "summary"
    "bab":          str,   # Nama BAB, atau "Ringkasan Dokumen" untuk summary chunk
    "bagian":       str,   # Nama Bagian/Pasal, atau "Ringkasan Dokumen"
    "pasal":        str,   # Nama sub-pasal, atau "Ringkasan Dokumen"
    "priority":     int,   # 1-5, dari YAML
    "chunk_id":     str,   # MD5 hash dari doc_id + content (untuk idempotency)
    "status":       str,   # "active", "inactive", dll.
}
```

### 6.4 Strategi Retrieval

**Metode:** ChromaDB similarity search dengan metadata pre-filter.

```
CATEGORY_KEYWORDS = {
    "academic": [
        "SKS", "KRS", "IPK", "IPS", "UTS", "UAS", "DO", "drop out",
        "pasal", "peraturan", "yudisium", "cuti", "mata kuliah",
        "nilai", "ujian", "wisuda", "ijazah", "skripsi", "sidang"
    ],
    "calendar": [
        "jadwal", "tanggal", "kapan", "batas", "deadline", "registrasi",
        "pengisian", "periode", "libur", "minggu ke", "kalender"
    ],
    "institution_profile": [
        "sejarah", "visi", "misi", "lambang", "bendera", "mars",
        "hymne", "akreditasi", "rektor", "berdiri", "didirikan"
    ],
}
```

**Logika:**

1. Hitung keyword score per kategori dari query
2. Pilih kategori dengan score tertinggi (jika > 0)
3. Jika ada kategori terdeteksi → tambahkan filter: `{"category": detected_category}`
4. Jika tidak ada → gunakan seluruh collection tanpa filter

**Similarity Threshold:**

- ChromaDB distance function: **cosine**
- Arah: skor mendekati **0 = lebih mirip**, mendekati 2 = tidak mirip
- Threshold: buang chunk dengan distance > `SIMILARITY_THRESHOLD = 0.65`
  (artinya: hanya pertahankan chunk yang "cukup dekat" secara semantik)

> **Penjelasan:** `SIMILARITY_THRESHOLD = 0.65` menggunakan cosine distance.
> Nilai 0.0 = identik sempurna, 1.0 = orthogonal, 2.0 = berlawanan.
> Threshold 0.65 berarti: buang semua chunk dengan jarak > 0.65.

### 6.5 Manajemen Memori Percakapan

**Komponen:** `ConversationBufferWindowMemory` dari LangChain

```python
# Parameter memori (definisikan di src/config.py)
MEMORY_K = 5    # Simpan 5 giliran percakapan terakhir (5 tanya + 5 jawab)
                # Giliran lebih lama otomatis dibuang dari memori
                # Ini mencegah context window overflow saat sesi panjang
```

**Siklus hidup memori:**

- Memori hidup selama sesi Streamlit berlangsung (`st.session_state`)
- Memori **mati** saat pengguna klik tombol "Reset Percakapan" atau menutup browser
- Memori **tidak pernah** disimpan ke database atau file apapun
- Setiap tab browser baru = sesi baru = memori kosong

---

### 6.6 Return Type Formal dari `chain.py`

Setiap fungsi di `chain.py` yang dipanggil oleh `app.py` atau `evaluation.py`
WAJIB mengembalikan tipe data berikut:

```python
# Tipe data yang dikembalikan get_response():
{
    "answer":  str,          # Teks jawaban dari LLM (atau pesan fallback)
    "sources": List[Dict],   # List chunk yang digunakan sebagai konteks
    "found":   bool,         # True jika ada chunk yang lolos threshold
}

# Setiap item dalam "sources":
{
    "doc_id":   str,   # ID dokumen
    "title":    str,   # Judul dokumen
    "bab":      str,   # Nama BAB (atau "Ringkasan Dokumen")
    "bagian":   str,   # Nama Bagian/Pasal
    "pasal":    str,   # Nama sub-pasal
    "preview":  str,   # 150 karakter pertama dari isi chunk
    "category": str,   # Kategori dokumen
}
```

---

## 7. SPESIFIKASI KONFIGURASI & PARAMETER

Semua parameter sistem wajib didefinisikan di `src/config.py`.
**Tidak boleh ada angka "ajaib" (magic number) di dalam `ingestion.py`, `chain.py`,
atau file lainnya.** Selalu impor dari `config.py`.

```python
# src/config.py — CANONICAL CONFIGURATION FILE

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

---

## 8. SPESIFIKASI FILE `.env`

```bash
# File: .env (di root proyek, sejajar dengan app.py)
# JANGAN commit file ini ke Git!

GOOGLE_API_KEY=isi_dengan_api_key_anda_di_sini
```

File `.gitignore` WAJIB berisi:

```
.env
chroma_db/
__pycache__/
*.pyc
.conda/
*.egg-info/
```

---

## 9. FORMAT DATA EVALUASI

### 9.1 Ground Truth (`eval/dataset/ground_truth.csv`)

**Nama file:** `eval/dataset/ground_truth.csv`
**Format:** CSV dengan encoding UTF-8 (dengan BOM untuk kompatibilitas Excel)
**Jumlah baris:** 30-50 pasang data (target minimum 30)

**Struktur:**

```csv
user_input,reference,category,source_doc,notes
"Apa syarat yudisium program sarjana?","Mahasiswa dapat mengikuti yudisium...","academic","Peraturan_Akademik_UNSRAT_2025","Pasal 45"
"Kapan batas pengisian KRS semester genap 2025/2026?","...","calendar","Kalender_Akademik_UNSRAT_Genap_2025-2026",""
```

**Keterangan kolom:**

| Kolom        | Tipe   | Deskripsi                                                        | Wajib?      |
| ------------ | ------ | ---------------------------------------------------------------- | ----------- |
| `user_input` | string | Pertanyaan seperti yang akan diketik pengguna                    | ✅ Wajib    |
| `reference`  | string | Jawaban benar berdasarkan dokumen asli (kunci jawaban)           | ✅ Wajib    |
| `category`   | string | Kategori dokumen sumber: `akademik`, `kalender`, `profil`, `faq` | ✅ Wajib    |
| `source_doc` | string | Nama file dokumen sumber (tanpa ekstensi .md)                    | ⭕ Opsional |
| `notes`      | string | Catatan tambahan (nomor Pasal, dll.)                             | ⭕ Opsional |

**Distribusi yang direkomendasikan:**

- 40% → Peraturan Akademik (`category = "academic"`)
- 30% → Kalender Akademik (`category = "calendar"`)
- 20% → Profil institusi (`category = "institution_profile"`)
- 10% → FAQ (`category = "faq"`, setelah file tersedia)

### 9.2 Hasil Evaluasi (`eval/results/`)

**File output:**

- `hasil_config_a.csv` — Config A (RAG chunk 500)
- `hasil_config_b.csv` — Config B (RAG chunk 2000)
- `hasil_config_c.csv` — Config C (BM25 murni)
- `statistical_test.csv` — Hasil uji Wilcoxon A vs B per metrik
- `error_analysis_config_[a|b].csv` — 10 sampel terburuk + klasifikasi kegagalan

Semua file di-overwrite setiap evaluasi baru dijalankan.

**Skema CSV hasil (`hasil_config_*.csv`):**

```csv
user_input,reference,response,retrieved_contexts,faithfulness,answer_relevancy,context_precision,context_recall,response_time_seconds
```

**Keterangan kolom tambahan:**

| Kolom                   | Tipe  | Deskripsi                                                    |
| ----------------------- | ----- | ------------------------------------------------------------ |
| `response_time_seconds` | float | Durasi `get_response()` dalam detik (diukur via `time.time`) |
| `faithfulness`          | float | Ragas: jawaban hanya dari dokumen                            |
| `answer_relevancy`      | float | Ragas: ketepatan jawaban terhadap pertanyaan                 |
| `context_precision`     | float | Ragas: chunk relevan vs chunk noise                          |
| `context_recall`        | float | Ragas: semua informasi relevan berhasil diambil              |

**Skema CSV uji statistik (`statistical_test.csv`):**

```csv
metric,wilcoxon_statistic,p_value,significant_at_0.05,winner
faithfulness,12.0,0.031,True,Config B
...
```

**Skema CSV analisis kegagalan (`error_analysis_config_*.csv`):**

```csv
rank,user_input,reference,response,avg_metric_score,failure_type,failure_notes
1,"Apa syarat cuti?","...","...",0.23,retrieval_failure,"Chunk salah diambil — kategori terdeteksi keliru"
...
```

`failure_type` diisi manual oleh peneliti dengan salah satu:
`retrieval_failure` | `generation_failure` | `chunking_failure`

**Agregasi yang dilaporkan ke terminal:**

- `mean ± std` untuk setiap metrik Ragas
- `mean ± std` untuk `response_time_seconds`
- Contoh: `Faithfulness: 0.87 ± 0.05 | Latency: 3.12 ± 0.84 detik`

---

## 10. SPESIFIKASI UI (STREAMLIT — DEFAULT DEMO)

### 10.1 Prinsip Modularitas UI

> **ATURAN INTI:** `app.py` adalah adapter UI. Ia hanya memanggil fungsi dari
> `src/chain.py` dan menampilkan hasilnya. Tidak ada logika RAG di dalam `app.py`.
>
> Jika di masa depan ingin menggunakan FastAPI, Gradio, atau framework lain,
> cukup buat file baru (misalnya `api.py`) yang memanggil fungsi yang sama
> dari `src/chain.py`. Kode inti tidak perlu diubah sama sekali.

### 10.2 Standar Nama Session State Keys

```python
# Di app.py, gunakan nama key ini secara konsisten:
SESSION_MESSAGES      = "messages"       # List[Dict] untuk tampilan chat bubble
SESSION_CHAT_HISTORY  = "chat_history"   # List[BaseMessage] untuk LangChain
SESSION_ACTIVE_CONFIG = "active_config"  # "a" atau "b"
```

### 10.3 Struktur UI (Dua Tab)

```
Tab 1: 💬 Chatbot
├── Header: "🎓 Asisten Informasi Akademik UNSRAT"
├── Caption: "Didukung oleh arsitektur RAG + Google Gemini"
├── Sidebar:
│   ├── Pilih Config: radio [A | B | C (BM25)]
│   ├── Pilih Model: selectbox dari AVAILABLE_MODELS  ← BARU
│   └── Tombol Reset Percakapan
├── Area chat (st.chat_message)
├── Streaming response (st.write_stream)
├── Sitasi sumber (lihat 10.4)
└── Disclaimer (wajib ada)

Tab 2: 📊 Evaluasi Ragas
├── Tabel hasil Config A
├── Tabel hasil Config B
├── Tabel hasil Config C (BM25)
├── Tabel uji Wilcoxon (p-values)
├── Bar chart perbandingan (baca dari perbandingan_visual.png)
└── Kesimpulan naratif otomatis
```

**Catatan model switcher:** Selectbox model di sidebar memanggil fungsi
`reinitialize_llm(model_name)` di `src/chain.py`. Ini me-reinisialisasi
instance `_llm` dengan model baru tanpa restart server. Nama model diambil
dari `AVAILABLE_MODELS` di `config.py`

### 10.4 Komponen Sitasi Sumber

```python
# Sitasi sumber (DUA OPSI — pilih satu setelah melihat UI)
# --- OPSI A: Plain text sederhana ---
st.caption(f"📄 Sumber: {doc_title} — {bab}, {pasal}")

# --- OPSI B: Expandable dengan cuplikan teks ---
# Expandable dengan cuplikan teks
with st.expander(f"📚 Lihat Sumber ({len(sources)} referensi)"):
    for i, src in enumerate(sources):
        st.markdown(f"**[{i+1}] {src['title']} — {src['bab']}**")
        st.text(src['preview'])   # 150 karakter pertama isi chunk
```

### 10.5 Respons Fallback (Tidak Ada Chunk Lolos)

Ketika `result["found"] == False`, tampilkan tanpa memanggil LLM:

```
Maaf, saya tidak menemukan informasi yang relevan mengenai pertanyaan Anda
dalam dokumen yang tersedia. Untuk informasi lebih lanjut, silakan hubungi:

• Bagian Akademik UNSRAT
• Portal INSPIRE: inspire.unsrat.ac.id
• Atau datang langsung ke Gedung Rektorat UNSRAT
```

---

## 11. SPESIFIKASI PIPELINE EVALUASI RAGAS

### 11.1 Metrik yang Diukur

**Dimensi 1 — Kualitas Jawaban (via Ragas):**

| Metrik                      | Mengukur Apa                                 | Target Min | Membutuhkan `reference`?                    |
| --------------------------- | -------------------------------------------- | ---------- | ------------------------------------------- |
| **Faithfulness**            | Jawaban hanya dari dokumen, tidak mengarang  | ≥ 0.90     | Tidak                                       |
| **Answer Relevancy**        | Jawaban tepat sasaran terhadap pertanyaan    | ≥ 0.85     | Tidak                                       |
| **Context Precision**       | Chunk yang diambil relevan (tidak ada noise) | ≥ 0.80     | Ya                                          |
| **Context Recall**          | Semua informasi relevan berhasil diambil     | ≥ 0.80     | Ya                                          |
| ~~Context Entities Recall~~ | Fraksi entitas dari reference ada di konteks | —          | Ya (opsional, aktifkan jika entitas kritis) |

> ⚠️ **PENTING — Verifikasi import Ragas sebelum coding:**
> Nama class/instance metrik Ragas berubah antar versi minor.
> Sebelum mengimplementasikan `evaluation.py`, **minta user untuk memberikan
> link dokumentasi resmi Ragas versi yang terinstall** (`pip show ragas` →
> cek versi → buka `https://docs.ragas.io/en/stable/`).
> Jangan asumsikan nama import dari memori — ini rawan halusinasi.

**Dimensi 2 — Performa Sistem (via `time`):**

| Metrik                    | Mengukur Apa                                    | Target          |
| ------------------------- | ----------------------------------------------- | --------------- |
| **response_time_seconds** | Latensi end-to-end per query (`get_response()`) | Deskriptif saja |

Laporkan sebagai `mean ± std` atas seluruh dataset evaluasi.

### 11.2 Alur Kerja `evaluation.py`

```
1. Parse argumen: --config [a|b|c] atau --visualize atau --stats atau --errors
   │
   ▼
2. Load ground_truth.csv (UTF-8)
   │
   ▼
3. Untuk setiap baris Q&A:
   a. RESET chat_history = [] (WAJIB — evaluasi harus bersih)
   b. Catat t_start = time.time()
   c. Panggil get_response(question, chat_history=[], config=args.config, streaming=False)
   d. Catat response_time = time.time() - t_start
   e. Tangkap: answer (str) + sources (List[Dict])
   │
   ▼
4. Susun dataset Ragas:
   {user_input, reference, response, retrieved_contexts} Tambahkan kolom: response_time_seconds (list[float])
   │
   ▼
5. Jalankan ragas.evaluate() dengan metrik:
   [faithfulness, answer_relevancy, context_precision, context_recall]
   (Gunakan EVALUATOR_MODEL_NAME — BERBEDA dari LLM_MODEL_NAME)
   │
   ▼
7. Gabungkan skor Ragas + response_time_seconds ke satu DataFrame
   │
   ▼
6. Hitung agregasi: mean ± std, min, max per metrik + latensi
   │
   ▼
8. Simpan ke eval/results/hasil_config_[a|b|c].csv
   │
   ▼
9. Generate error_analysis_config_[a|b].csv (10 sampel skor terendah)
   (Hanya untuk Config A & B — Config C BM25 bisa dianalisis terpisah)
   │
   ▼
10. Cetak ringkasan ke terminal
```

**Flag CLI tambahan:**

```bash
# Evaluasi satu config (termasuk latensi + error analysis otomatis)
python evaluation.py --config a
python evaluation.py --config b
python evaluation.py --config c   # BM25 baseline — tidak memanggil LLM untuk retrieval

# Uji statistik (jalankan setelah Config A & B selesai)
python evaluation.py --stats      # Wilcoxon A vs B → statistical_test.csv

# Visualisasi (jalankan setelah semua config selesai)
python evaluation.py --visualize  # Bar chart 3 config + p-value annotation
```

### 11.3 Catatan Metodologi

### 11.3 Catatan Metodologi

**Mitigasi self-evaluation bias (D-16):**
Generator (`LLM_MODEL_NAME`) dan evaluator Ragas (`EVALUATOR_MODEL_NAME`)
menggunakan model yang **berbeda**. Ini mengeliminasi risiko model menilai
output dirinya sendiri secara tidak objektif. Dokumentasikan pasangan model
yang digunakan di jurnal penelitian untuk reproduktibilitas.

**Baseline komparasi (D-09b):**
Config C (BM25 murni) tidak menggunakan LLM untuk retrieval sama sekali —
hanya keyword matching berbasis TF-IDF. Ini memberikan batas bawah performa
yang dapat diukur. Klaim penelitian menjadi: "RAG dengan Config B mengungguli
keyword search murni sebesar X% pada metrik Y (p = Z)."

**Uji signifikansi statistik:**
Wilcoxon Signed-Rank Test (`scipy.stats.wilcoxon`) dijalankan antara
Config A vs Config B untuk setiap metrik Ragas, menggunakan skor per-query
(bukan agregasi). Hasil dilaporkan sebagai p-value. Threshold: p < 0.05.
Jika p ≥ 0.05, perbedaan tidak signifikan — ini tetap kontribusi ilmiah
yang jujur dan harus didiskusikan di Bab 5.

**Evaluasi Config C (BM25):**
Karena BM25 tidak memanggil LLM untuk retrieval, `faithfulness` dan
`answer_relevancy` tetap dievaluasi (LLM masih dipakai untuk _generate_
jawaban dari teks BM25). `context_precision` dan `context_recall` mengukur
kualitas retrieval BM25 secara langsung.

**Keterbatasan yang tersisa:**

- Ground truth dibuat secara manual (bukan crowdsourced)
- Jumlah sampel (30–50) terbatas secara statistik — akui di Bab 5

### 11.4 Visualisasi & Output Statistik

```bash
# Generate bar chart 3 config + p-value annotation:
python evaluation.py --visualize
# Output: eval/results/perbandingan_visual.png

# Generate tabel uji Wilcoxon:
python evaluation.py --stats
# Output: eval/results/statistical_test.csv

# Syarat --visualize: ketiga file CSV hasil sudah ada.
# Syarat --stats: hasil_config_a.csv DAN hasil_config_b.csv sudah ada.
```

Chart yang dihasilkan:

- Grouped bar chart: 3 kelompok (Config A / B / C) × N metrik
- Anotasi p-value di atas setiap pasangan bar A vs B (format: `p=0.031*`)
- Sumbu kedua (kanan): rata-rata `response_time_seconds` sebagai line plot

---

## 12. ATURAN KODE & HARD CONSTRAINTS

### 12.1 Hal yang TIDAK BOLEH Dilakukan

```python
# ❌ DILARANG: Hardcode API Key di dalam kode
GOOGLE_API_KEY = "AIza..."

# ❌ DILARANG: Magic number di luar config.py
results = vectorstore.similarity_search(query, k=4)  # k harusnya dari config!

# ❌ DILARANG: Logika RAG di dalam app.py
# app.py hanya boleh memanggil fungsi dari src/chain.py

# ❌ DILARANG: Google Search Grounding
tools = [Tool(google_search=GoogleSearch())]

# ❌ DILARANG: Import reranking library
import cohere  # atau jina

# ❌ DILARANG: Hardcode path absolut
path = "C:/Users/Teofide/unsrat-rag/data/corpus/"  # Gunakan config.py!

# ❌ DILARANG: ConversationBufferWindowMemory (deprecated LangChain v0.3)
from langchain.memory import ConversationBufferWindowMemory

# ❌ DILARANG: Import rank_bm25 di luar src/bm25_retriever.py
# BM25 hanya boleh diakses melalui fungsi di src/bm25_retriever.py
from rank_bm25 import BM25Okapi  # ← hanya boleh ada di bm25_retriever.py
```

### 12.2 Hal yang WAJIB Dilakukan

```python
# ✅ Semua parameter dari config.py
from src.config import RETRIEVAL_K, SIMILARITY_THRESHOLD

# ✅ Setiap fungsi punya docstring
def get_response(question: str, ...) -> dict:
    """Dapatkan respons RAG untuk pertanyaan yang diberikan. ..."""

# ✅ Error handling informatif dalam Bahasa Indonesia

# ✅ Memory reset saat evaluasi
chat_history = []  # Reset sebelum setiap pertanyaan evaluasi

# ✅ task_type berbeda untuk embed dokumen vs query
# Saat ingestion: task_type="retrieval_document"
# Saat query:     task_type="retrieval_query"
```

---

## 13. SPESIFIKASI ERROR HANDLING

### 13.1 Tabel Error & Penanganannya

| Skenario                                               | Lokasi              | Penanganan                                       |
| ------------------------------------------------------ | ------------------- | ------------------------------------------------ |
| `GOOGLE_API_KEY` tidak ditemukan                       | `config.py` startup | `ValueError` dengan pesan jelas                  |
| `chroma_db/config_[a\|b]/` tidak ada                   | `chain.py` init     | Error informatif di Streamlit / terminal         |
| `ground_truth.csv` tidak ditemukan                     | `evaluation.py`     | `FileNotFoundError` + path ditampilkan           |
| File .md tidak punya frontmatter                       | `ingestion.py`      | `ValueError` + nama file, skip file tersebut     |
| Field wajib frontmatter kosong                         | `ingestion.py`      | `ValueError` + nama field + nama file            |
| Gemini API timeout (> 30 detik)                        | `chain.py`          | Retry → tampilkan error ke UI. Tidak crash.      |
| Gemini API rate limit (429)                            | `chain.py`          | Retry dengan delay → tampilkan pesan tunggu      |
| Tidak ada chunk lolos threshold                        | `chain.py`          | Return `{"found": False, ...}` tanpa panggil LLM |
| Chunk terlalu pendek (< MIN_CHUNK_LENGTH)              | `ingestion.py`      | Log warning + skip chunk tersebut                |
| Chunk duplikat (hash sama)                             | `ingestion.py`      | Log info + skip (idempotent)                     |
| `hasil_config_[a\|b].csv` tidak ada saat `--visualize` | `evaluation.py`     | Error jelas: "Jalankan evaluasi terlebih dahulu" |

### 13.2 Retry Policy

```python
MAX_RETRIES  = 3
RETRY_DELAYS = [2, 5]   # detik (dari config.py)

# Implementasi:
# - Attempt 1: Langsung
# - Jika gagal → tunggu 2 detik → Attempt 2
# - Jika gagal → tunggu 5 detik → Attempt 3
# - Jika masih gagal → tampilkan error ke pengguna
```

---

## 14. PANDUAN SETUP ENVIRONMENT

```bash
# Langkah 1: Buat environment
conda env create -f environment.yml

# Langkah 2: Aktifkan environment
conda activate unsrat-rag

# Langkah 3: Verifikasi
python --version        # Harus: Python 3.11.x
python -c "import langchain; print(langchain.__version__)"
python -c "import chromadb; print(chromadb.__version__)"
python -c "import ragas; print(ragas.__version__)"
python -c "import streamlit; print(streamlit.__version__)"
python -c "import frontmatter; print('python-frontmatter OK')"

# Langkah 4: Buat .env (manual via text editor)

# Langkah 5: Ingestion untuk kedua config
python src/ingestion.py --config a --rebuild
python src/ingestion.py --config b --rebuild

# Langkah 6: Verifikasi database
python -c "
import chromadb
from src.config import CHROMA_DIR_A, CHROMA_DIR_B, CHROMA_COLLECTION_A, CHROMA_COLLECTION_B
client_a = chromadb.PersistentClient(path=str(CHROMA_DIR_A))
client_b = chromadb.PersistentClient(path=str(CHROMA_DIR_B))
col_a = client_a.get_collection(CHROMA_COLLECTION_A)
col_b = client_b.get_collection(CHROMA_COLLECTION_B)
print(f'Config A: {col_a.count()} chunks')
print(f'Config B: {col_b.count()} chunks')
"

# Langkah 7: Jalankan chatbot
streamlit run app.py

# Langkah 8: Evaluasi (setelah ground_truth.csv siap)
python evaluation.py --config a
python evaluation.py --config b
python evaluation.py --visualize
```

---

## 15. DAFTAR KEPUTUSAN ARSITEKTUR (DECISION LOG)

| #     | Keputusan                                              | Pilihan Lain Ditolak             | Alasan                                                                                           |
| ----- | ------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------------------------------------------ |
| D-01  | Model: `gemini-3.5-flash`                              | gemini-2.0-flash                 | Rilis terbaru Mei 2026; fleksibel ganti via config                                               |
| D-02  | ChromaDB: Dua collection terpisah per config           | Satu collection, metadata filter | Isolasi penuh data antar config; tidak saling timpa                                              |
| D-03  | Retrieval: k=4, cosine distance threshold=0.65         | k=2, k=10 tanpa threshold        | Keseimbangan presisi-recall; threshold cegah halusinasi                                          |
| D-04  | Memory: Manual history list (LangChain v0.3)           | ConversationBufferWindowMemory   | Hindari deprecated API; lebih transparan untuk pemula                                            |
| D-05  | UI: Streamlit (default demo)                           | Chainlit, Gradio                 | Simpel, cepat, cukup untuk sidang; UI dipisah dari logic                                         |
| D-06  | Ground truth: CSV UTF-8                                | Excel, JSON                      | Portabel, kompatibel Ragas, mudah diedit                                                         |
| D-07  | Category detection: Keyword matching                   | LLM classifier                   | Zero API call tambahan; deterministic; mudah di-debug                                            |
| D-08  | Summary chunk: dari retrieval_summary                  | Abaikan / di system prompt       | Meningkatkan recall untuk query umum                                                             |
| D-09  | Search: Pure vector + metadata pre-filter (Config A/B) | Hybrid BM25+vector di Config A/B | Lebih simpel; BM25 sebagai Config C terpisah lebih bersih secara ilmiah                          |
| D-09b | BM25 murni sebagai Config C (baseline terpisah)        | Tidak ada baseline               | Meningkatkan bobot ilmiah: klaim "RAG mengungguli keyword search sebesar X%" dapat dipertahankan |
| D-10  | Versioning library: >= (fleksibel)                     | Pin ketat ==                     | Kemudahan update; diimbangi pengujian saat update                                                |
| D-11  | Idempotency: Chunk hash (MD5)                          | Upsert ChromaDB                  | Transparan, mudah di-debug; tidak ada side effect                                                |
| D-12  | Evaluasi agregasi: mean ± std                          | Mean saja                        | Std dev penting untuk menunjukkan konsistensi sistem                                             |
| D-13  | Konversi manual PDF → Markdown                         | PyPDF2 otomatis                  | Kualitas teks lebih bersih; bagian valid metodologi                                              |
| D-14  | Tidak ada reranking                                    | Cohere Rerank, Jina              | Overkill untuk < 100 halaman; tambah kompleksitas                                                |
| D-15  | `task_type` embedding berbeda (doc vs query)           | Satu task_type                   | Best practice Google Embedding API; kualitas lebih baik                                          |
| D-16  | Generator ≠ Evaluator model (berbeda)                  | Model sama (self-evaluation)     | Mengeliminasi self-eval bias; perubahan satu baris di `config.py`                                |
| D-17  | Wilcoxon Signed-Rank Test per-metrik (A vs B)          | t-test parametrik                | Data skor Ragas tidak diasumsikan berdistribusi normal; Wilcoxon lebih tepat untuk sampel kecil  |

---

# BAGIAN B — SRS (SOFTWARE REQUIREMENTS SPECIFICATION)

---

## 16. FUNCTIONAL REQUIREMENTS (FR)

### FR-01: Parsing Corpus

Sistem HARUS dapat membaca semua file `.md` di `data/corpus/`, mem-parsing
YAML frontmatter menggunakan library `python-frontmatter`, dan mengekstrak
metadata serta konten secara terpisah.

### FR-02: Validasi Frontmatter

Sistem HARUS memvalidasi bahwa setiap file memiliki field wajib:
`doc_id`, `title`, `category`, `content_type`, `valid_from`, `status`,
`retrieval_summary`, `chunk_strategy`, `last_updated`.
File yang gagal validasi harus di-log dan di-skip (tidak menghentikan proses).

### FR-03: Two-Stage Chunking

Sistem HARUS memotong dokumen dalam dua tahap:
(1) MarkdownHeaderTextSplitter berdasarkan heading,
(2) RecursiveCharacterTextSplitter untuk normalisasi ukuran.
Parameter chunk_size dan chunk_overlap diambil dari `config.py` sesuai config
yang dipilih.

### FR-04: Summary Chunk Generation

Sistem HARUS membuat satu chunk tambahan bertipe `"summary"` per dokumen dari
field `retrieval_summary`. Chunk ini harus memiliki metadata `bab`, `bagian`,
dan `pasal` bernilai `"Ringkasan Dokumen"`.

### FR-05: Chunk Filtering

Sistem HARUS membuang chunk yang panjangnya (setelah `.strip()`) kurang dari
`MIN_CHUNK_LENGTH` karakter.

### FR-06: Idempotent Ingestion

Sistem HARUS menghasilkan `chunk_id` unik per chunk berdasarkan MD5 hash
dari `f"{doc_id}::{chunk_content}"`. Chunk dengan ID yang sama tidak boleh
di-insert ulang ke ChromaDB. Jika sudah ada, sistem log "skipped (duplicate)"
dan lanjut.

### FR-07: Embedding dengan Task Type

Sistem HARUS menggunakan `task_type="retrieval_document"` saat embedding
chunk untuk ingestion, dan `task_type="retrieval_query"` saat embedding
query dari pengguna.

### FR-08: Dual Collection ChromaDB

Sistem HARUS menyimpan data Config A dan Config B di collection ChromaDB
yang berbeda: `CHROMA_COLLECTION_A` dan `CHROMA_COLLECTION_B`, di folder
yang berbeda: `chroma_db/config_a/` dan `chroma_db/config_b/`.

### FR-09: Category Detection

Sistem HARUS menjalankan keyword matching terhadap query untuk mendeteksi
kategori dokumen yang relevan. Hasil deteksi digunakan sebagai metadata
pre-filter sebelum vector search.

### FR-10: Similarity Threshold Filtering

Sistem HARUS membuang chunk yang memiliki cosine distance > `SIMILARITY_THRESHOLD`
sebelum chunk dikirim ke LLM.

### FR-11: Fallback Response

Sistem HARUS mengembalikan pesan fallback standar (lihat Section 10.5)
TANPA memanggil LLM jika tidak ada satu pun chunk yang lolos threshold.

### FR-12: Dual-Mode Response

Fungsi `get_response()` di `src/chain.py` HARUS mendukung dua mode:

- `streaming=True`: mengembalikan Generator (untuk Streamlit `st.write_stream`)
- `streaming=False`: mengembalikan `dict` dengan key `answer`, `sources`, `found`

### FR-13: Manual Memory Management

Sistem HARUS mengelola chat history sebagai Python list biasa (bukan memory
class LangChain). List harus di-trim ke maksimal `MEMORY_K * 2` pesan sebelum
dikirim ke LLM.

### FR-14: Evaluation Memory Reset

`evaluation.py` HARUS mereset `chat_history = []` sebelum setiap pertanyaan
evaluasi untuk memastikan setiap pertanyaan dievaluasi secara independen.

### FR-15: Ragas Evaluation

Sistem HARUS dapat menjalankan evaluasi menggunakan metrik Ragas:
`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`.
`context_entity_recall` bersifat opsional (diaktifkan via `METRICS_COLS`
di `config.py`).

> ⚠️ Verifikasi nama import Ragas dengan dokumentasi versi yang terinstall
> sebelum implementasi. Minta link dokumentasi ke user jika tidak yakin.

### FR-16: Evaluation Aggregation

Sistem HARUS menghitung dan menampilkan: mean, std, min, max untuk setiap
metrik Ragas setelah evaluasi selesai.

### FR-17: Comparison Visualization

Sistem HARUS dapat menghasilkan file `perbandingan_visual.png` (bar chart
grouped) yang membandingkan Config A vs Config B vs Config C ketika
dijalankan dengan flag `--visualize` (dan ketiga file CSV hasil sudah ada).
Chart harus mengandung anotasi p-value dari uji Wilcoxon (Config A vs B).

### FR-18: API Retry

Sistem HARUS mengimplementasikan exponential backoff dengan maksimal
`MAX_RETRIES` percobaan dan delay `RETRY_DELAYS` detik untuk semua panggilan
ke Gemini API.

### FR-19: BM25 Indexing (Config C)

`src/bm25_retriever.py` HARUS dapat membangun indeks BM25 dari seluruh
file `.md` di `CORPUS_DIR` dan menyimpannya sebagai `bm25_index.pkl`.
Harus mendukung flag `--rebuild` via CLI.

### FR-20: BM25 Retrieval

`retrieve_chunks_bm25(query)` di `src/bm25_retriever.py` HARUS mengembalikan
`BM25_K` teks chunk teratas beserta metadata (doc_id, title, preview),
dalam format yang kompatibel dengan `get_response()` di `chain.py`.

### FR-21: Latensi Per-Query

`evaluation.py` HARUS mengukur `response_time_seconds` untuk setiap query
menggunakan `time.time()` sebelum dan sesudah `get_response()`.
Kolom `response_time_seconds` HARUS disimpan di file CSV hasil evaluasi.

### FR-22: Uji Wilcoxon

`evaluation.py` HARUS menjalankan `scipy.stats.wilcoxon` antara skor
per-query Config A vs Config B untuk setiap metrik Ragas, dan menyimpan
hasilnya (statistic, p_value, significant, winner) ke `statistical_test.csv`.

### FR-23: Error Analysis Export

`evaluation.py` HARUS mengidentifikasi `ERROR_ANALYSIS_N` sampel dengan
rata-rata skor metrik Ragas terendah dan mengekspornya ke
`error_analysis_config_[a|b].csv` dengan kolom: rank, user_input, reference,
response, avg_metric_score, failure_type (diisi manual), failure_notes.

### FR-24: Model Switcher UI

`app.py` HARUS menyediakan selectbox di sidebar untuk memilih model LLM
dari daftar `AVAILABLE_MODELS`. Pilihan harus memperbarui instance LLM
yang aktif via fungsi `reinitialize_llm(model_name)` di `src/chain.py`.

### FR-25: Config C Evaluation Support

`evaluation.py` HARUS mendukung `--config c` yang memanggil
`get_response(..., config="c")` — Config C menggunakan BM25 untuk retrieval
dan LLM yang sama untuk generation.

---

## 17. NON-FUNCTIONAL REQUIREMENTS (NFR)

### NFR-01: Modularitas

Semua logika bisnis (RAG, retrieval, embedding, LLM) HARUS ada di dalam
`src/`. File `app.py` hanya boleh berisi kode UI dan pemanggilan fungsi dari
`src/`. Ini memungkinkan penggantian UI framework tanpa mengubah `src/`.

### NFR-02: Single Responsibility

Setiap file Python HARUS memiliki satu tanggung jawab utama sesuai Section 4.1.

### NFR-03: No Magic Numbers

Semua angka konfigurasi HARUS berasal dari `src/config.py`. Dilarang ada
literal angka konfigurasi di file lain.

### NFR-04: Secret Management

API Key HARUS dimuat dari file `.env` via `python-dotenv`. Dilarang hardcode
API key di kode manapun.

### NFR-05: Docstring

Setiap fungsi Python HARUS memiliki docstring minimal satu baris yang
menjelaskan apa yang dilakukan fungsi tersebut.

### NFR-06: Bahasa Pesan Error

Semua pesan error yang ditampilkan ke pengguna HARUS dalam Bahasa Indonesia
yang jelas dan informatif.

### NFR-07: Reproducibility

Evaluasi HARUS menghasilkan hasil yang konsisten untuk input yang sama.
Ini dijamin oleh `LLM_TEMPERATURE = 0.1` dan memory reset antar pertanyaan.

### NFR-08: Logging Ingestion

`ingestion.py` HARUS mencetak progress ke terminal: nama file yang diproses,
jumlah chunk yang dihasilkan, jumlah chunk yang di-skip (duplikat/terlalu
pendek), dan total chunk yang berhasil di-insert.

### NFR-09: Graceful Degradation

Sistem HARUS tetap berjalan (tidak crash) saat menghadapi error API yang
bisa di-retry. Crash hanya boleh terjadi saat kondisi yang tidak bisa
dipulihkan (API key tidak ada, database tidak ada).

---

## 18. CONSTRAINTS & ASSUMPTIONS

### 18.1 Technical Constraints

- Semua model AI diakses via API (tidak ada model lokal/offline)
- Corpus terbatas pada dokumen statis dalam format Markdown
- Tidak ada persistensi sesi percakapan antar restart aplikasi
- Chromadb berjalan sebagai embedded database (bukan server terpisah)
- Sistem hanya berjalan di localhost (tidak ada deployment publik)

### 18.2 Research Constraints

- Generator dan evaluator menggunakan model yang **berbeda** (D-16) —
  self-evaluation bias telah dimitigasi secara teknis. Tetap akui
  keterbatasan LLM-as-a-judge di Bab 5.
- Ground truth dibuat secara manual oleh peneliti (bukan crowdsourced)
- Jumlah sampel evaluasi (30–50) terbatas — uji Wilcoxon tetap valid
  untuk sampel kecil, namun power statistik terbatas; akui di Bab 5.
- Config C (BM25) tidak memiliki konteks percakapan (stateless per-query),
  konsisten dengan cara evaluasi Config A & B yang juga stateless.

### 18.3 Assumptions

- Semua file corpus sudah tersedia dalam format Markdown yang bersih
  sebelum ingestion dijalankan
- Koneksi internet tersedia saat sistem berjalan (akses ke Gemini API)
- Pengguna memiliki pemahaman dasar tentang cara menggunakan browser

---

## 19. RENCANA ANALISIS HASIL (TEMPLATE BAB IV)

> Section ini mendefinisikan secara eksplisit bagaimana Bab IV skripsi
> harus ditulis. Dengan template ini, penulisan menjadi terarah.

### 19.1 Struktur Bab IV yang Direkomendasikan

Bab IV — Hasil dan Pembahasan
4.1 Hasil Evaluasi Kuantitatif
4.1.1 Perbandingan Metrik Ragas (A vs B vs C)
4.1.2 Perbandingan Latensi Sistem
4.1.3 Uji Signifikansi Statistik (Wilcoxon A vs B)
4.2 Analisis Kegagalan Kualitatif
4.2.1 Kasus Kegagalan Config A (10 sampel terburuk)
4.2.2 Kasus Kegagalan Config B (10 sampel terburuk)
4.2.3 Klasifikasi Penyebab Kegagalan
4.3 Diskusi Trade-off
4.3.1 Chunk Size vs Token Consumption vs Latensi
4.3.2 Chunk Size vs Kualitas Konteks
4.4 Rekomendasi Konfigurasi

### 19.2 Template Tabel Komparasi Utama (Tabel 4.1)

| Metrik                    | Config A (500) | Config B (2000) | Config C (BM25) | Winner | p-value (A vs B) |
| ------------------------- | -------------- | --------------- | --------------- | ------ | ---------------- |
| Faithfulness              | X.XXX ± X.XXX  | X.XXX ± X.XXX   | X.XXX ± X.XXX   | ?      | p = X.XXX        |
| Answer Relevancy          | —              | —               | —               | ?      | p = X.XXX        |
| Context Precision         | —              | —               | —               | ?      | p = X.XXX        |
| Context Recall            | —              | —               | —               | ?      | p = X.XXX        |
| **Response Time (detik)** | X.XX ± X.XX    | X.XX ± X.XX     | X.XX ± X.XX     | ?      | —                |

> Nilai diisi dari file CSV hasil evaluasi. p-value dari `statistical_test.csv`.
> Tanda `*` jika p < 0.05. Tanda `n.s.` jika p ≥ 0.05.

### 19.3 Protokol Analisis Kegagalan Kualitatif

Setelah evaluasi selesai:

1. Buka `error_analysis_config_a.csv` dan `error_analysis_config_b.csv`
2. Untuk setiap sampel (rank 1–10), baca `user_input`, `reference`, dan `response`
3. Isi kolom `failure_type` secara manual dengan salah satu:

   | Kode                 | Definisi                                                                |
   | -------------------- | ----------------------------------------------------------------------- |
   | `retrieval_failure`  | Chunk yang salah diambil; informasi jawaban ada tapi tidak ter-retrieve |
   | `generation_failure` | Chunk yang tepat diambil, tapi LLM salah interpretasi atau merangkum    |
   | `chunking_failure`   | Informasi terpotong di antara dua chunk; tidak ada chunk yang lengkap   |

4. Tulis narasi Bab IV 4.2 berdasarkan distribusi `failure_type`:
   - Jika mayoritas `retrieval_failure` → solusi: perbaiki `SIMILARITY_THRESHOLD` atau `RETRIEVAL_K`
   - Jika mayoritas `generation_failure` → solusi: perbaiki `SYSTEM_PROMPT` atau model
   - Jika mayoritas `chunking_failure` → solusi: gunakan Config B (chunk lebih besar)

### 19.4 Diskusi Trade-off yang Wajib Dibahas

**Chunk Size vs Token Consumption:**
Estimasi rata-rata token per request = `avg_chunk_length × RETRIEVAL_K / 4`.
Config B (2000 char) mengonsumsi ~4× token input lebih banyak dari Config A.
Hitung biaya estimasi berdasarkan harga token model yang digunakan.

**Chunk Size vs Latensi:**
Lebih banyak token input → potensi latensi lebih tinggi. Bandingkan
`response_time_seconds` mean antara Config A dan B secara eksplisit.

**BM25 vs RAG:**
Config C memberikan baseline: seberapa jauh embedding vector search
mengungguli keyword matching sederhana? Jika selisihnya kecil,
ini adalah temuan menarik yang harus didiskusikan.

### 19.5 Kriteria Rekomendasi Konfigurasi

Rekomendasikan Config X jika memenuhi ≥ 3 dari 4 kriteria:

| Kriteria                 | Threshold                       |
| ------------------------ | ------------------------------- |
| Faithfulness tertinggi   | Selisih ≥ 0.05 dari config lain |
| Context Recall tertinggi | Selisih ≥ 0.05 dari config lain |
| Latensi dapat diterima   | mean response_time < 10 detik   |
| Unggul vs BM25 baseline  | Semua metrik utama > Config C   |

---

## CHECKLIST SEBELUM MULAI CODING

- [ ] File `.env` sudah dibuat dengan `GOOGLE_API_KEY` yang valid
- [ ] Conda environment `unsrat-rag` berhasil dibuat dan diaktifkan
- [ ] Semua file corpus `.md` ada di `data/corpus/` dengan YAML frontmatter yang benar
- [ ] GCP Budget Alert sudah diaktifkan ($50 dan $250)
- [ ] File `.gitignore` sudah berisi `.env` dan `chroma_db/`
- [ ] Dokumen ini dibaca dan dipahami sebelum mulai menulis kode
- [ ] `src/bm25_retriever.py` berhasil dijalankan: `python src/bm25_retriever.py --rebuild`
- [ ] `bm25_index/bm25_index.pkl` terbuat
- [ ] `python evaluation.py --config c` berjalan menghasilkan `hasil_config_c.csv`
- [ ] `python evaluation.py --stats` menghasilkan `statistical_test.csv`
- [ ] `error_analysis_config_a.csv` dan `error_analysis_config_b.csv` terbuat
- [ ] Kolom `failure_type` di error analysis CSV diisi manual oleh peneliti
- [ ] `AVAILABLE_MODELS` di `config.py` sudah diisi dengan nama model nyata
- [ ] `LLM_MODEL_NAME` ≠ `EVALUATOR_MODEL_NAME` (beda model, beda baris di config)
- [ ] `[ISI_NAMA_MODEL_*]` placeholder tidak ada lagi di `config.py`

---

_Dokumen terakhir direvisi: 26 Mei 2026 | Versi: 5.0_
