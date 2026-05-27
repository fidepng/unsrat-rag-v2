# UNSRAT RAG Chatbot — Logging System Design Spec

> **Status:** Approved by User  
> **Versi:** 1.0  
> **Tanggal:** 2026-05-27  
> **Author:** Antigravity (AI Coding Assistant)  
> **Target Skripsi:** Data Primer untuk Bab III (Metodologi) & Bab IV (Hasil & Pembahasan)  

---

## 1. Latar Belakang & Kebutuhan Akademik

Untuk keperluan penyusunan laporan skripsi/thesis, sistem chatbot RAG UNSRAT membutuhkan mekanisme perekaman data transaksi dan performa sistem secara kuantitatif dan kualitatif. Data ini bertindak sebagai **bukti empiris** untuk menganalisis efisiensi biaya (*cost efficiency*), latensi (*response time*), dan keandalan sistem (*reliability & API limit protection*).

Logging difokuskan pada meta-perubahan strategi sistem (misalnya chunking, API retry policy, data embedding, dan evaluasi Ragas), bukan pada kesalahan sintaks kecil (receh).

---

## 2. Arsitektur Logging Terpusat & Modular

Sesuai prinsip **NFR-02 (Single Responsibility)**, seluruh operasi penulisan log diisolasi ke dalam satu berkas manajer tunggal: `src/logger_manager.py`. Modul-modul lain di `src/` maupun Streamlit UI (`app.py`) berinteraksi dengan manajer ini secara modular tanpa mengurusi operasi I/O file secara langsung.

```
unsrat-rag/
├── logs/                      ← Direktori terisolasi (.gitignore)
│   ├── unsrat_rag.log         ← Log teks terstruktur (peristiwa sistem & API retry)
│   ├── transaksi_chat.csv     ← Log Q&A (latensi, similarity score, estimasi token)
│   └── ingestion_report.csv   ← Log performa ingestion & chunking per berkas
└── src/
    └── logger_manager.py      ← MODUL BARU: Manajer Logging Terpusat
```

---

## 3. Spesifikasi Tiga Jenis Log

### 3.1. Log Teks Terstruktur (`logs/unsrat_rag.log`)
Mencatat seluruh siklus hidup sistem, inisialisasi ulang model LLM, audit evaluasi Ragas, dan pertahanan API limit (retry policy).
*   **Format:** `[TIMESTAMP] [LEVEL] [KATEGORI] Pesan Log detail`
*   **Contoh Tag Kategori:** `[LLM_INIT]`, `[API_LIMIT]`, `[API_RETRY]`, `[CHROMA_DB]`, `[BM25_INDEX]`, `[EVALUATOR_INIT]`, `[EVALUATION]`.

### 3.2. Log Transaksi Chatbot (`logs/transaksi_chat.csv`)
Merekam data kuantitatif dari setiap interaksi pengguna dengan chatbot. Data ini diimpor dengan mudah ke Excel / WPS Office untuk Bab IV skripsi Anda.

| Nama Kolom | Tipe Data | Deskripsi / Manfaat Akademik |
|---|---|---|
| `timestamp` | String | Waktu transaksi (`YYYY-MM-DD HH:MM:SS`) |
| `config_retrieval` | String | Konfigurasi yang aktif (`a`, `b`, atau `c`) |
| `model_llm` | String | Model LLM aktif yang merespons (misal: `gemini-3.5-flash`) |
| `user_query` | String | Pertanyaan asli dari pengguna |
| `category_detected` | String | Hasil deteksi kategori pertanyaan (UI matching) |
| `chunks_retrieved_count`| Integer | Jumlah chunk dokumen yang berhasil ditarik |
| `retrieved_chunk_ids` | String | List ID Hash MD5 chunk terambil (dipisah koma) |
| `best_similarity_score`| Float | Cosine similarity score terbaik (untuk validitas retrieval) |
| `average_similarity_score`| Float| Rata-rata cosine similarity score dari chunk lolos threshold |
| `response_time_seconds`| Float | Durasi total respons chatbot (detik) |
| `estimated_prompt_tokens`| Integer | Estimasi token input prompt (System + History + Context) |
| `estimated_completion_tokens`| Integer| Estimasi token output jawaban chatbot |
| `estimated_total_tokens`| Integer | Total token terpakai (Prompt + Completion) |
| `found_state` | Boolean | Status keberhasilan retrieval (`True`/`False` jika fallback) |

### 3.3. Log Ingestion & Indexing (`logs/ingestion_report.csv`)
Merekam performa pemrosesan data corpus akademik:

| Nama Kolom | Tipe Data | Deskripsi / Manfaat Akademik |
|---|---|---|
| `timestamp` | String | Waktu pemrosesan |
| `config` | String | Strategi pemotongan (`a`, `b`, atau `c` untuk BM25) |
| `file_processed` | String | Nama berkas `.md` yang diproses |
| `chunks_generated` | Integer | Jumlah total chunk yang dihasilkan |
| `chunks_inserted` | Integer | Jumlah chunk baru yang disimpan ke DB |
| `chunks_duplicate_skipped`| Integer | Jumlah chunk duplikat yang disaring (idempotensi) |
| `execution_time_seconds`| Float | Durasi total pengindeksan file (detik) |

---

## 4. Mekanisme Kunci & Integrasi

### 4.1. Graceful Degradation & API Limit Logging (`src/chain.py`)
Mekanisme pertahanan API limit (RPM/TPM) dicatat secara mendalam:
*   Ketika terjadi kesalahan API limit (`ResourceExhausted` atau sejenisnya) saat memanggil LLM Gemini, modul `src/chain.py` mencatat tingkat `WARNING` pada `unsrat_rag.log` berisi detail percobaan ke-N dan durasi penundaan.
*   Sistem melakukan jeda (`time.sleep`) secara otomatis sesuai `RETRY_DELAYS`.
*   Jika limit berhasil dilewati pada percobaan berikutnya, dicatat tingkat `INFO` bertag `[API_RETRY]`. Jika semua percobaan gagal, dicatat tingkat `ERROR` sebelum melempar Exception.

### 4.2. Estimasi Token Terpakai secara Offline
Untuk menghindari biaya API tambahan, estimasi token dihitung secara offline di dalam `src/logger_manager.py` menggunakan pustaka standard industri **`tiktoken`** dengan encoding model `cl100k_base` (atau encoder standard). Ini memberikan akurasi tinggi tanpa pemanggilan API berbayar.

### 4.3. Kesiapan Dashboard Visual (Modularitas Tinggi)
Berkas log didesain untuk mudah dibaca kembali menggunakan pustaka `pandas`. Di masa depan, visualisasi grafik performa log dapat diintegrasikan ke Streamlit (`app.py`) sebagai Tab baru cukup dengan memanggil:
```python
import pandas as pd
df = pd.read_csv("logs/transaksi_chat.csv")
```

---

## 5. Rencana Pengujian Logging

Penerapan logging akan diuji dengan skenario:
1.  **Pengujian Ingestion:** Menjalankan `python src/ingestion.py --config b --rebuild` dan memverifikasi bahwa berkas `logs/ingestion_report.csv` terisi dengan baris data log yang valid.
2.  **Pengujian Chatbot & Token:** Menanyakan 1 pertanyaan akademik di Streamlit UI dan memverifikasi berkas `logs/transaksi_chat.csv` terisi dengan record transaksi lengkap beserta estimasi token.
3.  **Pengujian API Limit (Mock):** Memverifikasi bahwa upaya penanganan retry tercatat dengan tag `[API_RETRY]` di dalam `logs/unsrat_rag.log`.
