# Design Spec: Sistem Chatbot Informasi Akademik UNSRAT (RAG)

**Tanggal:** 2026-05-27
**Versi Spec:** 1.2 (perbaikan bug kritis retrieved_contexts + NFR-05 + FR-09 deviasi template)
**Status:** Disetujui — siap untuk implementation plan
**Berdasarkan:** `prd+srs.md` v5.0, `agents.md`, `UNSRAT_YAML_STANDARD_v2.md`
**Git Remote:** `https://github.com/fidepng/unsrat-rag-v2.git` (lokal only, belum push)

---

## Ringkasan

Sistem RAG (Retrieval-Augmented Generation) berbasis Streamlit untuk menjawab
pertanyaan informasi akademik Universitas Sam Ratulangi. Sistem membandingkan
tiga konfigurasi retrieval (Config A, B, C) dan menyediakan pipeline evaluasi
menggunakan Ragas 0.4.x.

**Prinsip desain utama:**
> **"Extensible by configuration, not by code."**
> Menambah dokumen, kategori, atau model baru cukup dengan mengedit corpus
> dan `config.py` lalu menjalankan ulang ingestion — tanpa menyentuh logika
> pipeline.

---

## Metodologi Pengembangan

**Metodologi:** Modified Waterfall (Fase Sequential)

Proyek ini menggunakan pendekatan **Waterfall termodifikasi**, bukan Agile,
karena:

| Alasan | Detail |
|---|---|
| Requirement sudah lengkap | PRD+SRS v5.0 mendefinisikan semua FR, NFR, dan constraint sebelum kode ditulis |
| Satu peneliti | Tidak ada tim yang perlu sinkronisasi sprint |
| Scope tetap | Dilarang menambah fitur di luar PRD+SRS (AGENTS.md Golden Rule #6) |
| Output akademik | Thesis membutuhkan metodologi yang dapat dipertanggungjawabkan dan direproduksi |

**Fase implementasi (sequential, tidak boleh dilompati):**

```
Fase 0: Setup Lingkungan
  ↓
Fase 1: src/config.py
  ↓
Fase 2: src/ingestion.py (Config A & B)
  ↓
Fase 2.5: src/bm25_retriever.py (Config C)
  ↓
Fase 3: src/retriever.py
  ↓
Fase 4: src/chain.py
  ↓
Fase 5: app.py
  ↓
Fase 6: evaluation.py
```

**Elemen Agile yang diadopsi:**
- Verifikasi per-fase sebelum lanjut (checklist di akhir spec ini)
- Design spec ini dibuat sebelum kode (spec-first development)
- Dokumentasi setiap keputusan arsitektur (decision log)

**Reprodusibilitas penelitian dijamin oleh:**
- `LLM_TEMPERATURE = 0.1` (deterministik)
- Memory reset antar pertanyaan evaluasi
- Version-pinned `environment.yml`
- `chunk_id` berbasis MD5 hash (idempotent ingestion)

---

## Keputusan Arsitektur — Deviasi dari PRD+SRS

Tabel berikut mendokumentasikan **setiap keputusan desain yang berbeda**
dari PRD+SRS atau skeleton `agents.md`. Ini adalah referensi audit.

| # | Isu | Keputusan | Alasan |
|---|---|---|---|
| D-A1 | `chunk_strategy` & `chunk_notes` | Informational only — pipeline tidak membacanya | Corpus 9 dokumen, format seragam, uniform pipeline sudah cukup. PRD sendiri mengakui ini "variabel penelitian yang disengaja." |
| D-A2 | Category detection (FR-09) | **Deviasi dari PRD:** Label UI saja, TIDAK digunakan sebagai ChromaDB pre-filter | PRD menggunakan category sebagai metadata pre-filter. Keputusan ini memilih pendekatan yang lebih sederhana: query multi-kategori terlalu fragile untuk winner-takes-all routing. ChromaDB similarity search sudah cukup tanpa pre-filter tambahan. |
| D-A3 | BM25 Config C — granularitas | **Refaktor dari agents.md:** BM25 mengindeks chunk, bukan full-doc | Full-document BM25 vs chunk-based embedding tidak dapat dibandingkan secara metodologis valid. |
| D-A4 | `ConversationBufferWindowMemory` | **Manual list** via `st.session_state` | Deprecated di LangChain v0.3. Lebih transparan dan zero external dependency. |
| D-A5 | `SIMILARITY_THRESHOLD` | 0.65 yang benar — nilai 0.35 di diagram PRD Section 6.4 adalah errata | PRD Section 6.4 sendiri menjelaskan threshold 0.65 di bawah diagram, sehingga diagram 0.35 adalah typo. |
| D-A6 | `ragas>=stable` | **Pin ke `ragas>=0.4.0`** | `stable` bukan valid pip version specifier. |
| D-A7 | Ground truth: 5 kolom PRD vs 4 kolom | **4 kolom wajib:** `user_input, reference, category, source_doc` | Kolom `notes` tetap opsional (tidak divalidasi). PRD Section 9.1 memang mendefinisikan `notes` sebagai opsional. |
| D-A8 | Evaluasi: CLI-only vs UI | **CLI untuk proses, UI untuk tampil hasil** | Evaluasi di UI tidak reproducible. Artefak CSV dibaca oleh Tab Evaluasi Streamlit. |
| D-A9 | `REQUIRED_YAML_FIELDS` duplikat | **Pindah ke `config.py`** | DRY principle. |
| D-A10 | Animasi loading | **Streaming token via `st.write_stream()`** | Native Streamlit + LangChain streaming. Mengurangi persepsi latency. |
| D-A11 | Output file naming | **Ikuti PRD:** `hasil_config_[a\|b\|c].csv`, `perbandingan_visual.png` | Konsistensi dengan PRD Section 9.2. |

---

## Arsitektur Keseluruhan

```
[Corpus Layer]
  data/corpus/*.md  (9 dokumen aktif + pending faq.md)
  YAML frontmatter (UNSRAT_YAML_STANDARD_v2)
        ↓ (offline, CLI)
[Ingestion Layer]
  src/ingestion.py       → Config A (chunk 500) & Config B (chunk 2000)
  src/bm25_retriever.py  → Config C (BM25, chunk-based)
        ↓ (artefak persisten)
[Index Layer]
  chroma_db/config_a/    → ChromaDB collection Config A
  chroma_db/config_b/    → ChromaDB collection Config B
  bm25_index/            → BM25 pickle index Config C
        ↓ (online, Streamlit runtime)
[Runtime Layer]
  src/retriever.py       → unified interface: list[{content, metadata, score}]
  src/chain.py           → RAG chain + memory + streaming + retry
  app.py                 → Streamlit UI (Tab Chat + Tab Evaluasi)
        ↓ (offline, CLI, terpisah)
[Evaluation Layer]
  evaluation.py          → jalankan pipeline, ukur dengan Ragas 0.4.x
  eval/results/          → CSV + PNG (dibaca Tab Evaluasi di app)
```

**Single point of control:** `config.py` — semua angka, path, nama model,
dan konstanta ada di sini. File lain hanya mengimport.

---

## Seksi 1 — Corpus & Data Layer

### Struktur File (Canonical)
```
data/corpus/
  01_sejarah.md                               ← institution_profile, priority 4
  02_visi_misi.md                             ← institution_profile, priority 4
  03_tujuan_sasaran_strategi.md               ← institution_profile, priority 4
  04_lambang.md                               ← institution_profile, priority 4
  05_bendera.md                               ← institution_profile, priority 4
  06_mars_hymne.md                            ← institution_profile, priority 4
  07_akreditasi.md                            ← institution_profile, priority 3
  Peraturan_Akademik_UNSRAT_2025_RAG_REVISED.md ← academic, priority 1
  Kalender_Akademik_UNSRAT_Genap_2025-2026.md  ← calendar, priority 2
  [PENDING] faq.md                            ← faq, priority 2
```

### YAML Fields Contract

**Fields yang dikonsumsi pipeline** (wajib ada, divalidasi `REQUIRED_YAML_FIELDS`):
```
doc_id, title, category, content_type, valid_from, status,
retrieval_summary, chunk_strategy, last_updated
```

**Fields yang disimpan sebagai ChromaDB metadata** (subset dari atas + extra):
```python
{
    "doc_id":       str,   # ID dokumen
    "title":        str,   # Judul dokumen
    "category":     str,   # "academic", "calendar", "institution_profile", "faq"
    "content_type": str,   # "regulation", "calendar", "narrative", "guide"
    "chunk_type":   str,   # "content" atau "summary"
    "bab":          str,   # Nama BAB, atau "Ringkasan Dokumen"
    "bagian":       str,   # Nama Bagian/Pasal, atau "Ringkasan Dokumen"
    "pasal":        str,   # Nama sub-pasal, atau "Ringkasan Dokumen"
    "priority":     int,   # 1-5, dari YAML
    "status":       str,   # "active", "inactive", "draft"
    # chunk_id TIDAK disimpan di metadata ChromaDB — digunakan sebagai
    # ChromaDB document ID, bukan sebagai metadata field
}
```

**Fields metadata dokumentasi** (tidak divalidasi pipeline, untuk manusia):
```
tags, keywords, entities, subcategory, audience, access_level,
related_docs, language_*, nomor_sk, tanggal_penetapan,
pejabat_penandatangan, source_url, supersedes, superseded_by,
valid_until, last_verified, embedding_model, chunk_notes, version
```

> **Catatan `chunk_strategy`:** Termasuk `REQUIRED_YAML_FIELDS` (divalidasi)
> tapi tidak dibaca untuk menentukan strategi split. Pipeline selalu menggunakan
> two-stage splitter. Field ini adalah instruksi dokumentasi untuk manusia dan
> sebagai variabel penelitian yang disengaja (PRD Section 6.2).

### Menambah Dokumen Baru
1. Buat file `.md` di `data/corpus/` dengan frontmatter yang valid
2. Pastikan 9 field `REQUIRED_YAML_FIELDS` terisi
3. Jalankan `python src/ingestion.py --config a --rebuild`
4. Jalankan `python src/ingestion.py --config b --rebuild`
5. Jalankan `python src/bm25_retriever.py --rebuild`
6. **Tidak ada perubahan kode apapun**

---

## Seksi 2 — Ingestion Pipeline

### `src/__init__.py`
File kosong — wajib ada agar Python mengenali `src/` sebagai package.

### Config A & B (ChromaDB)

```
File .md
  → parse_markdown_file() — validasi REQUIRED_YAML_FIELDS (9 field)
  → create_summary_chunk() — chunk dari retrieval_summary
    (chunk_type="summary", bab/bagian/pasal="Ringkasan Dokumen")
  → split_content() — MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter
  → filter chunk < MIN_CHUNK_LENGTH (50 char) — log + skip
  → generate_chunk_id() — MD5("doc_id::content") → digunakan sebagai ChromaDB ID
  → cek duplikat via collection.get(ids=[chunk_id])
  → embed via GoogleGenerativeAIEmbeddings(task_type="retrieval_document")
  → collection.add(ids, embeddings, documents, metadatas)
```

**Parameter per config:**

| Parameter | Config A | Config B |
|---|---|---|
| `CHUNK_SIZE` | 500 | 2000 |
| `CHUNK_OVERLAP` | 100 | 200 |
| ChromaDB dir | `chroma_db/config_a/` | `chroma_db/config_b/` |
| Collection | `unsrat_rag_config_a` | `unsrat_rag_config_b` |
| Distance fn | cosine | cosine |

**Heading hierarchy untuk MarkdownHeaderTextSplitter:**
```python
HEADERS_TO_SPLIT = [
    ("#",    "header_1"),
    ("##",   "bab"),
    ("###",  "bagian"),
    ("####", "pasal"),
]
```

**Logging wajib (NFR-08):** nama file, jumlah chunk insert, duplikat, terlalu pendek, total.

**Task type embedding:**
- Ingestion: `task_type="retrieval_document"`
- Query runtime: `task_type="retrieval_query"`

### Config C (BM25) — Refaktor dari agents.md

**Perubahan dari skeleton agents.md:** BM25 mengindeks chunk-chunk (bukan full dokumen).
Chunk yang diindeks: hasil split Config B (chunk_size=2000) untuk perbandingan fair.

```
File .md
  → parse frontmatter (validasi REQUIRED_YAML_FIELDS)
  → split_content() — SAMA dengan Config B
  → _tokenize() setiap chunk (lowercase + filter token < BM25_MIN_TOKEN_LEN)
  → BM25Okapi(tokenized_chunks)
  → pickle.dump({
        "index": bm25_index,
        "texts": chunks_text,       # list[str]
        "metadata": chunks_metadata # list[dict]
    })
```

### `REQUIRED_YAML_FIELDS` — Perubahan dari agents.md

**Perubahan:** Didefinisikan SEKALI di `config.py`, bukan di dua file berbeda.

```python
# src/config.py
REQUIRED_YAML_FIELDS = [
    "doc_id", "title", "category", "content_type",
    "valid_from", "status", "retrieval_summary",
    "chunk_strategy", "last_updated",
]
```

`ingestion.py` dan `bm25_retriever.py` keduanya `from src.config import REQUIRED_YAML_FIELDS`.

---

## Seksi 3 — Runtime Layer

### `src/retriever.py` — Unified Interface

Semua config mengembalikan format identik ke `chain.py`:
```python
list[dict]  # setiap dict: {"content": str, "metadata": dict, "score": float}
```

**Config A & B (ChromaDB):**
```
query
  → embed(task_type="retrieval_query")
  → collection.query(n_results=RETRIEVAL_K)
  → filter: buang chunk dengan distance > SIMILARITY_THRESHOLD (0.65)
  → return list[{content, metadata, score}]
  → jika kosong → return [] (trigger FALLBACK_RESPONSE di chain)
```

**Config C (BM25):**
```
query
  → _tokenize(query)
  → bm25.get_scores()
  → argsort descending → top BM25_K
  → return list[{content, metadata, score}]
  → BM25 TIDAK memiliki similarity threshold — selalu return BM25_K teratas
```

> **Implikasi `found` pada Config C (BM25):**
> Config C TIDAK memiliki similarity threshold, sehingga `retrieve()` selalu
> mengembalikan BM25_K chunk teratas — tidak pernah mengembalikan [].
> Akibatnya:
> (a) `found` selalu bernilai `True` untuk Config C — FALLBACK_RESPONSE tidak pernah dipicu.
> (b) Untuk query di luar domain corpus, Config C tetap memanggil LLM dengan chunk
>     yang mungkin tidak relevan, berpotensi menghasilkan jawaban yang kurang faithful.
> (c) Ini adalah karakteristik metodologis yang harus disebutkan di Bab III,
>     bukan bug. Ia menunjukkan kelemahan keyword search murni dibanding vector search
>     yang memiliki threshold-based fallback.

> **Catatan SIMILARITY_THRESHOLD:** Nilai `0.65` di `config.py` adalah benar.
> Nilai `0.35` di diagram PRD Section 6.4 adalah errata — PRD sendiri
> menjelaskan 0.65 di paragraf bawah diagram tersebut.

### `src/chain.py` — RAG Chain

**Return type formal `get_response()` (FR-12 Dual-Mode):**
```python
# streaming=False → dict
{
    "answer":  str,        # Teks jawaban LLM atau FALLBACK_RESPONSE
    "sources": list[dict], # List chunk yang digunakan sebagai konteks
    "found":   bool,       # True jika ada chunk lolos threshold
}

# Setiap item dalam "sources":
{
    "doc_id":   str,   # ID dokumen
    "title":    str,   # Judul dokumen
    "bab":      str,   # Nama BAB
    "bagian":   str,   # Nama Bagian/Pasal
    "pasal":    str,   # Nama sub-pasal
    "content":  str,   # FULL isi chunk — WAJIB untuk Ragas context_precision/recall
    "preview":  str,   # 150 karakter pertama — hanya untuk tampilan UI
    "category": str,   # Kategori dokumen
}

# CATATAN KRITIS (Bug Fix v1.2):
# evaluation.py HARUS menggunakan s["content"] bukan s["preview"] untuk
# retrieved_contexts. Menggunakan preview (150 char) akan membuat skor
# context_precision dan context_recall bias ke bawah secara sistematis
# dan merusak validitas ilmiah evaluasi.

# streaming=True → Generator (dikonsumsi st.write_stream())
```

**Fungsi `reinitialize_llm(model_name: str)` (FR-24):**
Fungsi di `chain.py` untuk reinisialisasi instance LLM dengan model baru
tanpa restart server. Dipanggil oleh `app.py` saat user ganti model di sidebar.

# Signature formal get_response() di src/chain.py
def get_response(
    question:     str,
    chat_history: list,   # list[dict] — dari SESSION_MESSAGES, atau [] untuk evaluasi
    config:       str,    # "a", "b", atau "c"
    streaming:    bool,   # True untuk UI Streamlit, False untuk evaluation.py
) -> dict | Generator:
    """
    Menjalankan RAG pipeline: retrieve → format → generate.
    streaming=False → dict {"answer", "sources", "found"}
    streaming=True  → Generator untuk st.write_stream()
    """
    
**Flow lengkap:**
```
user_query + config + model_name
  ↓
category_detection()
  → scan CATEGORY_KEYWORDS → return label (str) untuk UI saja
  → TIDAK mempengaruhi retrieval (deviasi D-A2 dari PRD FR-09)
  → Lihat: "Catatan Metodologi Deviasi FR-09" di bawah
  ↓
retriever.retrieve(query, config)
  → return list[chunk] atau []
  ↓
if not chunks: return {"answer": FALLBACK_RESPONSE, "sources": [], "found": False}
  ↓
format_context(chunks) → gabung content chunk
  ↓
format_history(st.session_state[SESSION_MESSAGES], k=MEMORY_K)
  → ambil MEMORY_K*2 pesan terakhir (= MEMORY_K pasang Q&A)
  ↓
prompt = SYSTEM_PROMPT + context + history + query
  ↓
if streaming=True:  return llm.stream(prompt)   → Generator
if streaming=False: return {"answer": llm.invoke(prompt), "sources": ..., "found": True}
```

**Memory — Session State Keys (Section 10.2 PRD):**
```python
SESSION_MESSAGES      = "messages"        # list[dict] untuk tampilan chat bubble
SESSION_ACTIVE_CONFIG = "active_config"   # "a", "b", atau "c"

# SESSION_CHAT_HISTORY (dari PRD Section 10.2) TIDAK digunakan.
# PRD mendefinisikannya untuk LangChain BaseMessage, tapi kita menggunakan
# manual list approach (deviasi D-A4). Key ini sengaja dihilangkan untuk
# menghindari kebingungan. History percakapan ada di SESSION_MESSAGES.
```

---

### Catatan Metodologi Deviasi FR-09 (Template Teks Bab III Thesis)

> **Untuk peneliti:** Teks berikut dapat digunakan sebagai dasar penjelasan
> metodologi di Bab III skripsi pada bagian implementasi pipeline retrieval.

**Template:**
> "Pada implementasi sistem, deteksi kategori pertanyaan (Category Detection)
> dirancang sebagai komponen informatif yang menampilkan label kategori
> pada antarmuka pengguna, namun tidak digunakan sebagai filter metadata
> pada proses pencarian ChromaDB. Keputusan ini berbeda dari rancangan awal
> PRD Section 6.4 (FR-09) yang menggunakan deteksi kategori sebagai
> metadata pre-filter. Pertimbangan utama adalah bahwa pertanyaan yang
> mengandung kata kunci dari dua kategori berbeda secara bersamaan
> (misalnya: 'Kapan semester genap dimulai?' yang mencakup kata kunci
> kategori 'calendar' dan 'academic') akan menghasilkan routing yang
> tidak deterministik menggunakan pendekatan winner-takes-all.
> ChromaDB cosine similarity search sudah cukup untuk menemukan chunk
> yang relevan tanpa pre-filtering kategori tambahan."

**Retry policy:**
```
Attempt 1: langsung
Gagal → tunggu RETRY_DELAYS[0] (2 detik) → Attempt 2
Gagal → tunggu RETRY_DELAYS[1] (5 detik) → Attempt 3
Gagal → tampilkan error ke pengguna (Bahasa Indonesia)
```

**Evaluation mode:** `evaluation.py` memanggil `get_response(..., streaming=False)`
dan mereset `chat_history = []` sebelum setiap pertanyaan (FR-14).

---

## Seksi 4 — Application Layer (Streamlit UI)

### Navigasi
```
app.py
  ├── Tab 1: 💬 Chatbot
  │   ├── Header: "🎓 Asisten Informasi Akademik UNSRAT"
  │   ├── Caption: "Didukung oleh arsitektur RAG + Google Gemini"
  │   ├── Sidebar
  │   │   ├── Config switcher: radio [A | B | C (BM25)]
  │   │   │     → switching config: reset st.session_state[SESSION_MESSAGES]
  │   │   ├── Model switcher: selectbox(AVAILABLE_MODELS)
  │   │   │     → memanggil reinitialize_llm(model_name) di chain.py
  │   │   ├── Reset button: "🔄 Reset Percakapan"
  │   │   └── Info panel: config aktif + model aktif
  │   ├── Area chat (st.chat_message bubble style)
  │   │   ├── Streaming response via st.write_stream()
  │   │   ├── Category badge per response (cosmetic, label UI)
  │   │   └── Sitasi sumber (lihat bawah)
  │   ├── Disclaimer (wajib ada — NFR dari PRD Section 10.3)
  │   └── Input: st.chat_input() — fixed bottom
  │
  └── Tab 2: 📊 Evaluasi Ragas
      ├── Jika eval/results/ kosong: tampilkan instruksi CLI
      ├── Tabel hasil Config A, B, C (baca dari CSV)
      ├── Tabel uji Wilcoxon (statistical_test.csv)
      ├── Bar chart (perbandingan_visual.png)
      └── Narasi kesimpulan otomatis berdasarkan data
```

**Tidak ada tombol "Run Evaluation" di UI.**

### Sitasi Sumber (Section 10.4 PRD)

Implementasi default: Expandable (Opsi B dari PRD):
```python
with st.expander(f"📚 Lihat Sumber ({len(sources)} referensi)"):
    for i, src in enumerate(sources):
        st.markdown(f"**[{i+1}] {src['title']} — {src['bab']}**")
        st.text(src['preview'])   # 150 karakter pertama isi chunk
```

### Streaming Response
```python
with st.chat_message("assistant"):
    response_text = st.write_stream(chain.get_response(query, chat_history=st.session_state[SESSION_MESSAGES],
                   config=st.session_state[SESSION_ACTIVE_CONFIG], streaming=True))
# Simpan full text setelah streaming selesai:
st.session_state[SESSION_MESSAGES].append(
    {"role": "assistant", "content": response_text}
)
```

### UI Style
```css
:root {
  --color-primary:     #7B2D2D;  /* Maroon UNSRAT (aktif) */
  --color-primary-alt: #8B1A2B;  /* Crimson UNSRAT (tersimpan, switch via var) */
  --color-primary-dark:#5C1F1F;
  --color-primary-light:#A84545;
  --font-main: 'Inter', sans-serif;
}
```
Switch ke Crimson: ubah nilai `--color-primary` menjadi `#8B1A2B`. Tidak ada perubahan logic.

---

## Seksi 5 — Evaluation Pipeline

### CLI Interface Lengkap
```bash
python evaluation.py --config a      # Evaluasi Config A
python evaluation.py --config b      # Evaluasi Config B
python evaluation.py --config c      # Evaluasi Config C (BM25 baseline)
python evaluation.py --stats         # Wilcoxon A vs B → statistical_test.csv
python evaluation.py --visualize     # Bar chart 3 config → perbandingan_visual.png
```

**Syarat `--stats`:** `hasil_config_a.csv` DAN `hasil_config_b.csv` sudah ada.
**Syarat `--visualize`:** Ketiga file CSV hasil sudah ada.

### Format `ground_truth.csv`

**Kolom wajib** (divalidasi sebelum evaluasi mulai, error Bahasa Indonesia):

| Kolom | Tipe | Keterangan |
|---|---|---|
| `user_input` | str | Pertanyaan (input Ragas 0.4.x) |
| `reference` | str | Jawaban ideal (input Ragas 0.4.x) |
| `category` | str | `academic`, `calendar`, `institution_profile`, `faq` |
| `source_doc` | str | Nama file sumber tanpa ekstensi |

**Kolom opsional** (tidak divalidasi, boleh ada):

| Kolom | Tipe | Keterangan |
|---|---|---|
| `notes` | str | Catatan tambahan (nomor Pasal, dll.) |

**Encoding:** UTF-8 (dengan BOM untuk kompatibilitas Excel)

**Jumlah target:** 30–50 pasang Q&A

**Distribusi yang direkomendasikan (PRD Section 9.1):**
- 40% → Peraturan Akademik (`category="academic"`)
- 30% → Kalender Akademik (`category="calendar"`)
- 20% → Profil institusi (`category="institution_profile"`)
- 10% → FAQ (`category="faq"`, setelah file tersedia)

### Metrik Evaluasi

**Dimensi 1 — Kualitas Jawaban (via Ragas 0.4.x):**

| Metrik | Target Min | Butuh `reference`? |
|---|---|---|
| `faithfulness` | ≥ 0.90 | Tidak |
| `answer_relevancy` | ≥ 0.85 | Tidak |
| `context_precision` | ≥ 0.80 | Ya |
| `context_recall` | ≥ 0.80 | Ya |

**Dimensi 2 — Performa Sistem:**

| Metrik | Pengukuran | Cara |
|---|---|---|
| `response_time_seconds` | Per-query | `time.time()` sebelum & sesudah `get_response()` |

Dilaporkan sebagai `mean ± std` atas seluruh dataset.

### Flow Evaluasi (Ragas 0.4.x)

```python
# evaluation.py — pseudocode
df = pd.read_csv(EVAL_DATASET_PATH, encoding="utf-8-sig")

# Validasi 4 kolom wajib
required = ["user_input", "reference", "category", "source_doc"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

rows = []
for _, row in df.iterrows():
    chat_history = []  # WAJIB reset per pertanyaan (FR-14)
    t_start = time.time()
    result = chain.get_response(row["user_input"],
                                chat_history=chat_history,
                                config=args.config,
                                streaming=False)
    response_time = time.time() - t_start

    rows.append({
        "user_input":          row["user_input"],
        "reference":           row["reference"],
        "response":            result["answer"],
        # WAJIB: gunakan s["content"] (full chunk), BUKAN s["preview"] (150 char)
        # Menggunakan preview akan merusak validitas context_precision & context_recall
        "retrieved_contexts":  [s["content"] for s in result["sources"]],
        "response_time_seconds": response_time,
    })

dataset = EvaluationDataset.from_list(rows)
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                      context_precision, context_recall],
                   llm=evaluator_llm)  # EVALUATOR_MODEL ≠ LLM_MODEL

results_df = results.to_pandas()
results_df["category"] = df["category"].values
results_df["source_doc"] = df["source_doc"].values
results_df["response_time_seconds"] = [r["response_time_seconds"] for r in rows]
results_df.to_csv(EVAL_RESULTS_DIR / f"hasil_config_{config}.csv", index=False)
```

**Agregasi yang dicetak ke terminal:**
```
Faithfulness: 0.87 ± 0.05
Answer Relevancy: 0.82 ± 0.08
Context Precision: 0.79 ± 0.11
Context Recall: 0.81 ± 0.07
Response Time: 3.12 ± 0.84 detik
```

### Output Artefak (Naming Canonical dari PRD)

```
eval/results/
  hasil_config_a.csv              ← Config A
  hasil_config_b.csv              ← Config B
  hasil_config_c.csv              ← Config C (BM25)
  statistical_test.csv            ← Wilcoxon A vs B
  error_analysis_config_a.csv     ← 10 sampel terburuk A
  error_analysis_config_b.csv     ← 10 sampel terburuk B
  perbandingan_visual.png         ← Bar chart 3 config + p-value + latency
```

**Skema `hasil_config_*.csv`:**
```
user_input, reference, response, retrieved_contexts,
faithfulness, answer_relevancy, context_precision, context_recall,
response_time_seconds, category, source_doc
```

**Skema `statistical_test.csv` (Wilcoxon A vs B):**
```
metric, wilcoxon_statistic, p_value, significant_at_0.05, winner
```

**Skema `error_analysis_config_[a|b].csv`:**
```
rank, user_input, reference, response, avg_metric_score,
failure_type, failure_notes
```

`failure_type` diisi **manual** oleh peneliti:
- `retrieval_failure` — chunk salah diambil
- `generation_failure` — chunk tepat tapi LLM salah interpretasi
- `chunking_failure` — informasi terpotong antar chunk

**Error analysis hanya untuk Config A & B.** Config C dianalisis terpisah jika diperlukan.

### Visualisasi (`--visualize`)

Bar chart `perbandingan_visual.png`:
- Grouped bar: 3 kelompok (Config A / B / C) × 4 metrik Ragas
- Anotasi p-value di atas setiap pasangan bar A vs B (`p=0.031*` atau `n.s.`)
- Sumbu kedua (kanan): rata-rata `response_time_seconds` sebagai line plot

### Uji Statistik (`--stats`)

`scipy.stats.wilcoxon` antara skor per-query Config A vs Config B, per metrik.
Threshold signifikansi: p < 0.05.

---

## Seksi 6 — Error Handling

| Skenario | Lokasi | Penanganan |
|---|---|---|
| `GOOGLE_API_KEY` tidak ditemukan | `config.py` startup | `ValueError` pesan jelas Bahasa Indonesia |
| `chroma_db/config_[a\|b]/` tidak ada | `chain.py` init | Error informatif di Streamlit/terminal |
| `ground_truth.csv` tidak ditemukan | `evaluation.py` | `FileNotFoundError` + path ditampilkan |
| Kolom wajib CSV tidak ada | `evaluation.py` | `ValueError` + daftar kolom missing |
| File .md tidak punya frontmatter | `ingestion.py` | Log warning + skip file |
| Field wajib frontmatter kosong | `ingestion.py` | Log warning + nama field + nama file |
| Gemini API timeout | `chain.py` | Retry 3x → tampilkan error ke UI |
| Gemini API rate limit (429) | `chain.py` | Retry dengan delay → pesan tunggu |
| Tidak ada chunk lolos threshold | `chain.py` | Return `FALLBACK_RESPONSE`, found=False |
| Chunk terlalu pendek | `ingestion.py` | Log warning + skip chunk |
| Chunk duplikat | `ingestion.py` | Log info + skip (idempotent) |
| CSV hasil belum ada saat `--visualize` | `evaluation.py` | Error jelas: "Jalankan evaluasi terlebih dahulu" |

---

## Seksi 7 — Dependency & Environment

### `environment.yml` (Final)

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
      - ragas>=0.4.0          # PERUBAHAN dari ragas>=stable (invalid)
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

### `.gitignore` (Canonical)

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

> **Catatan:** `bm25_index/` ditambahkan ke gitignore (auto-generated artifact).

### Git Setup
```bash
# Remote sudah dikonfigurasi (lokal only, jangan push tanpa izin peneliti):
git remote -v
# origin  https://github.com/fidepng/unsrat-rag-v2.git (fetch)
# origin  https://github.com/fidepng/unsrat-rag-v2.git (push)
```

### Budget Safety (PRD Section 3.4)
- Budget Alert: $50 (peringatan), $250 (kritis)
- Hard Cap: $280 (sisakan $20 buffer)
- Dilarang: Google Search Grounding, reranking eksternal

---

## Seksi 8 — `config.py` Additions

```python
# Tambahan yang tidak ada di skeleton agents.md:

# REQUIRED_YAML_FIELDS — dipindah dari ingestion.py ke sini (DRY)
REQUIRED_YAML_FIELDS = [
    "doc_id", "title", "category", "content_type",
    "valid_from", "status", "retrieval_summary",
    "chunk_strategy", "last_updated",
]

# PIPELINE_CONSUMED_FIELDS — subset yang benar-benar dibaca retriever/chain
PIPELINE_CONSUMED_FIELDS = [
    "doc_id", "title", "category", "content_type", "status",
]
# Catatan: retrieval_summary dikonsumsi saat ingestion (summary chunk)
# priority disimpan di ChromaDB metadata tapi tidak digunakan untuk ranking

# METRICS_COLS — urutan metrik ini dipakai di chart, CSV, dan terminal output
# evaluation.py WAJIB mengimport ini, tidak boleh hardcode list metrik (NFR-03)
METRICS_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    # "context_entity_recall",  # Opsional — aktifkan di sini jika diperlukan
]

# ERROR_ANALYSIS_N — jumlah sampel terburuk untuk analisis kegagalan kualitatif
# evaluation.py WAJIB mengimport ini, tidak boleh hardcode angka 10 (NFR-03)
ERROR_ANALYSIS_N = 10

# SESSION STATE KEYS (Section 10.2 PRD) — konstanta untuk konsistensi
# Gunakan konstanta ini di app.py, jangan hardcode string key langsung
SESSION_MESSAGES      = "messages"
SESSION_ACTIVE_CONFIG = "active_config"
# SESSION_CHAT_HISTORY (dari PRD) TIDAK digunakan — lihat Seksi 3 catatan memory

# CATEGORY_KEYWORDS — digunakan oleh category_detection() di chain.py
# Hasilnya hanya label UI (badge), TIDAK mempengaruhi retrieval (deviasi D-A2)
# WAJIB di config.py — jangan definisikan di chain.py (NFR-03)
CATEGORY_KEYWORDS: dict = {
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
    "faq": ["FAQ", "pertanyaan umum", "sering ditanya"],
}
```

---

## Peta Dependensi File

```
.env
  └── src/config.py          ← Semua file bergantung padanya
        ├── src/ingestion.py  ← Tidak bergantung pada file src/ lain
        ├── src/bm25_retriever.py ← Bergantung pada config.py
        ├── src/retriever.py  ← Bergantung pada config.py
        └── src/chain.py      ← Bergantung pada config.py + retriever.py
              ├── app.py       ← Bergantung pada chain.py SAJA
              └── evaluation.py ← Bergantung pada chain.py
```

**Urutan implementasi WAJIB mengikuti urutan ini.**

---

## Checklist Verifikasi Per Fase (Extended dari PRD)

### Fase 0 — Setup
- [ ] `conda env create -f environment.yml` sukses
- [ ] `python -c "import ragas; print(ragas.__version__)"` → `0.4.x`
- [ ] `python -c "import langchain; print(langchain.__version__)"` → `0.3.x+`
- [ ] `.env` berisi `GOOGLE_API_KEY` yang valid
- [ ] GCP Budget Alert aktif ($50 + $250)
- [ ] `src/__init__.py` ada (file kosong)

### Fase 1 — Config
- [ ] `python src/config.py` tanpa error
- [ ] `REQUIRED_YAML_FIELDS` hanya ada di `config.py`, tidak di file lain
- [ ] `LLM_MODEL_NAME` ≠ `EVALUATOR_MODEL_NAME`
- [ ] `AVAILABLE_MODELS` berisi nama model yang benar-benar tersedia
- [ ] **NFR-05:** Semua fungsi di `config.py` (jika ada) memiliki docstring

### Fase 2 — Ingestion A & B
- [ ] `python src/ingestion.py --config b --rebuild` → log jumlah chunk
- [ ] `python src/ingestion.py --config a --rebuild` → log jumlah chunk
- [ ] Verifikasi: `python -c "import chromadb; from src.config import *; c=chromadb.PersistentClient(str(CHROMA_DIR_B)); print(c.get_collection(CHROMA_COLLECTION_B).count())"`
- [ ] File .md tanpa frontmatter → di-skip (tidak crash)
- [ ] Field wajib kosong → log warning + skip
- [ ] **NFR-05:** Semua fungsi di `ingestion.py` memiliki docstring (minimal 1 baris)

### Fase 2.5 — BM25
- [ ] `python src/bm25_retriever.py --rebuild` sukses
- [ ] `bm25_index/bm25_index.pkl` terbuat
- [ ] Verifikasi BM25 mengindeks CHUNK (bukan full doc) via log count yang lebih besar dari jumlah file
- [ ] **NFR-05:** Semua fungsi di `bm25_retriever.py` memiliki docstring

### Fase 3 — Retriever
- [ ] `retriever.retrieve("SKS maksimal", config="b")` → list[dict] valid
- [ ] `retriever.retrieve("xyz123nonsense", config="b")` → `[]`
- [ ] `retriever.retrieve("SKS maksimal", config="c")` → list[dict] dari BM25
- [ ] Return type konsisten: `{"content": str, "metadata": dict, "score": float}`
- [ ] **NFR-05:** Semua fungsi di `retriever.py` memiliki docstring

### Fase 4 — Chain
- [ ] `get_response("Berapa SKS maksimal?", streaming=False)` → `{"answer", "sources", "found"}`
- [ ] Setiap item `sources` memiliki field `content` (full text) DAN `preview` (150 char)
- [ ] `get_response(..., streaming=True)` → Generator
- [ ] Query irrelevant → `FALLBACK_RESPONSE`, `found=False`
- [ ] `reinitialize_llm(model_name)` berfungsi tanpa restart
- [ ] Category detection mengembalikan label string, tidak mempengaruhi retrieval
- [ ] **NFR-05:** Semua fungsi di `chain.py` memiliki docstring

### Fase 5 — App
- [ ] `streamlit run app.py` tanpa error
- [ ] Switch config → conversation reset
- [ ] Response muncul streaming (token per token)
- [ ] Sitasi sumber muncul di expander
- [ ] Disclaimer tampil
- [ ] Tab Evaluasi: pesan instruksi jika CSV belum ada
- [ ] Model switcher berfungsi tanpa restart
- [ ] **NFR-05:** Semua fungsi di `app.py` memiliki docstring

### Fase 6 — Evaluasi
- [ ] `python evaluation.py --config b` → `hasil_config_b.csv` terbuat
- [ ] `python evaluation.py --config a` → `hasil_config_a.csv` terbuat
- [ ] `python evaluation.py --config c` → `hasil_config_c.csv` terbuat
- [ ] `python evaluation.py --stats` → `statistical_test.csv` terbuat
- [ ] `python evaluation.py --visualize` → `perbandingan_visual.png` terbuat
- [ ] `error_analysis_config_a.csv` dan `error_analysis_config_b.csv` terbuat
- [ ] CSV tanpa kolom wajib → error message Bahasa Indonesia
- [ ] Tab Evaluasi di app menampilkan semua hasil
- [ ] **Bug Fix v1.2:** `retrieved_contexts` menggunakan `s["content"]` (full text) — VERIFIKASI manual
- [ ] **NFR-05:** Semua fungsi di `evaluation.py` memiliki docstring
- [ ] **MANUAL (peneliti):** Buka `error_analysis_config_a.csv` dan `error_analysis_config_b.csv`
      → Isi kolom `failure_type` untuk setiap baris dengan salah satu:
      `retrieval_failure` | `generation_failure` | `chunking_failure`
      → Isi kolom `failure_notes` dengan penjelasan singkat kegagalan
      → Simpan file (ini adalah input narasi Bab IV 4.2)

---

## Referensi Analisis Hasil (Template Bab IV Thesis)

Setelah evaluasi selesai, output digunakan untuk:

**Tabel 4.1 — Komparasi Utama:**

| Metrik | Config A (500) | Config B (2000) | Config C (BM25) | Winner | p-value (A vs B) |
|---|---|---|---|---|---|
| Faithfulness | X.XXX ± X.XXX | X.XXX ± X.XXX | X.XXX ± X.XXX | ? | p = X.XXX |
| Answer Relevancy | — | — | — | ? | p = X.XXX |
| Context Precision | — | — | — | ? | p = X.XXX |
| Context Recall | — | — | — | ? | p = X.XXX |
| **Response Time (detik)** | X.XX ± X.XX | X.XX ± X.XX | X.XX ± X.XX | ? | — |

> Tanda `*` jika p < 0.05. Tanda `n.s.` jika p ≥ 0.05.

**Kriteria rekomendasi Config terbaik (PRD Section 19.5):**
Rekomendasikan Config X jika memenuhi ≥ 3 dari 4 kriteria:
- Faithfulness tertinggi (selisih ≥ 0.05)
- Context Recall tertinggi (selisih ≥ 0.05)
- Response time < 10 detik (mean)
- Unggul vs BM25 baseline di semua metrik utama

### Diskusi Trade-off yang Wajib Dibahas di Bab IV (PRD Section 19.4)

**4.3.1 Chunk Size vs Token Consumption:**
Estimasi token per request = `rata_rata_panjang_chunk × RETRIEVAL_K / 4`
Config B (chunk 2000 char) mengonsumsi ~4× token input lebih banyak dari Config A.
Hitung estimasi biaya API berdasarkan harga token model yang digunakan.

**4.3.2 Chunk Size vs Latensi:**
Token input lebih banyak → potensi latensi lebih tinggi.
Bandingkan `response_time_seconds` mean Config A vs B secara eksplisit.
Data ada di `hasil_config_a.csv` dan `hasil_config_b.csv`, kolom `response_time_seconds`.

**4.3.3 BM25 vs RAG (Config C sebagai Baseline):**
Pertanyaan kunci: seberapa jauh vector search mengungguli keyword search murni?
- Jika selisih besar → validasi kuat bahwa embedding memberikan nilai tambah nyata.
- Jika selisih kecil → temuan menarik: perlu diskusi mengapa (corpus bahasa formal,
  kata kunci teknis seperti SKS/KRS yang spesifik sehingga BM25 juga efektif).
Catatan: Config C tidak memiliki threshold fallback, sehingga skor faithfulness-nya
mungkin lebih rendah untuk out-of-domain query — sebutkan ini dalam diskusi.

---

## Catatan untuk Audit

### Deviasi dari PRD+SRS yang Didokumentasikan

| Deviasi | PRD Asli | Keputusan Akhir | Justifikasi |
|---|---|---|---|
| FR-09 Category pre-filter | Digunakan sebagai ChromaDB metadata filter | Label UI saja | Query multi-kategori fragile; ChromaDB similarity sudah cukup |
| BM25 granularitas | Full document per unit | Chunk-based (= Config B) | Perbandingan metodologi yang valid |
| Memory management | `ConversationBufferWindowMemory` | Manual list `st.session_state` | Deprecated API; lebih transparan |
| `ragas>=stable` | `stable` (invalid) | `ragas>=0.4.0` | Valid version specifier + API contract |
| Ground truth kolom | 5 kolom (notes wajib) | 4 wajib + notes opsional | PRD sendiri menyatakan notes opsional |

### Versi Dokumen Spec

| Versi | Tanggal | Perubahan |
|---|---|---|
| 1.0 | 2026-05-27 | Draft awal hasil brainstorming |
| 1.1 | 2026-05-27 | Cross-check lengkap PRD+SRS v5.0, tambah metodologi pengembangan, FR lengkap, git remote, error handling table, extended checklist |
| 1.2 | 2026-05-27 | **Bug fix kritis:** `retrieved_contexts` perbaiki ke full content; tambah `METRICS_COLS`/`ERROR_ANALYSIS_N` ke config section; NFR-05 docstring ke semua checklist fase; template teks Bab III untuk deviasi FR-09; komentar eksplisit SESSION_CHAT_HISTORY |

---

_Spec terakhir direvisi: 27 Mei 2026 | Versi: 1.2_
_Penulis spec: AI Design Session (Brainstorming) — untuk keperluan audit thesis_
