# Spesifikasi Desain: Transisi UI ke FastAPI + Tailwind CSS + Chart.js
## Sistem Chatbot Informasi Akademik UNSRAT (RAG)

* **Tanggal Perubahan:** 2026-05-28
* **Status Desain:** Tervalidasi (Disetujui User)
* **Arsitektur:** REST API Client-Server (FastAPI Backend + HTML/JS Frontend)

---

## 🎯 1. Latar Belakang & Tujuan
Implementasi UI berbasis Streamlit pada berkas `app.py` memiliki keterbatasan kustomisasi tata letak (grid/flexbox kaku), risiko *collisional styling* pada CSS kustom, serta performa visual yang statis (seperti grafik evaluasi PNG dari Matplotlib). 

Spesifikasi desain ini merinci transisi penuh dari Streamlit ke **FastAPI (REST Backend)** dipadu **Tailwind CSS & Chart.js (Single-Page Frontend)** guna menghadirkan antarmuka asisten akademik premium, fleksibel, responsif, dan interaktif secara visual untuk sidang skripsi Anda.

---

## 🏛️ 2. Arsitektur & Peta Berkas Baru

Transisi ini memisahkan secara tegas logika backend server dengan penyajian tampilan frontend:

```
unsrat-rag/
├── app.py                   ← Backend Server FastAPI (menggantikan Streamlit)
├── index.html               ← Frontend Single-Page App (HTML + Tailwind CDN + Chart.js)
├── static/
│   └── js/
│       └── app.js           ← Logika interaksi JS, fetch API, SSE stream parser, dan Chart.js
└── src/
    ├── config.py            ← Pusat parameter & model (tetap dipertahankan)
    ├── chain.py             ← Logika pemanggilan RAG & Gemini stream (tetap dipertahankan)
    └── logger_manager.py    ← Log logging terpusat CSV (tetap dipertahankan)
```

---

## 🌐 3. Spesifikasi API Endpoints (FastAPI Backend)

FastAPI akan diinisialisasi secara stabil dan melayani endpoints terstruktur berikut:

### A. Inisialisasi Server & Static File Mounting
FastAPI akan diatur untuk melayani folder `static` secara dinamis:
```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(title="UNSRAT Academic RAG API")

# Mount folder static untuk Javascript
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
```

### B. Chatbot RAG Stream Endpoint (`POST /api/chat`)
Endpoint ini memproses kueri masukan RAG dan mengembalikan token teks secara *real-time* via **Server-Sent Events (SSE)**.
*   **Format Request (JSON):**
    ```json
    {
      "query": "Apa visi Universitas Sam Ratulangi?",
      "config": "b",
      "model": "gemini-3.5-flash",
      "chat_history": []
    }
    ```
*   **Format Respon (Event-Stream / SSE):**
    ```
    data: {"token": "Visi"}
    data: {"token": " Universitas"}
    data: {"token": " Sam"}
    ...
    data: {"sources": [{"title": "Profil UNSRAT", "content": "..."}], "meta_logging": {...}}
    ```
    *Logika:* Backend memanggil generator stream dari `src/chain.py` sekali saja (**anti double-call**), mengalirkan token teks ke client, dan mengirimkan daftar referensi `sources` utuh di akhir streaming untuk menghindari biaya token ganda.

### C. Dashboard Evaluasi Endpoint (`GET /api/evaluation`)
Membaca berkas evaluasi offline dari `eval/results/` dan mengembalikan objek JSON bersih:
```json
{
  "configs": {
    "a": {
      "faithfulness": 0.85,
      "answer_relevancy": 0.92,
      "context_precision": 0.88,
      "context_recall": 0.90,
      "latency": 5.2
    },
    "b": { ... },
    "c": { ... }
  },
  "wilcoxon": [
    {"metric": "faithfulness", "p_value": 1.0, "significant": false, "winner": "n.s."}
  ]
}
```

---

## 🎨 4. Spesifikasi Estetika Frontend (Maroon Klasik)

Desain visual akan diimplementasikan secara elegan di dalam `index.html` dengan framework **Tailwind CSS (v3 CDN)** dan ikon **Lucide Icons**:

### A. Palet Warna Utama
*   **Background Utama:** Warm Sand Off-White (`#FAFAF8`) yang bersih, memberi kontras tinggi, dan nyaman dibaca dalam waktu lama.
*   **Sidebar Kontrol:** Gradasi Marun Klasik: `bg-gradient-to-b from-[#5B1A1A] to-[#7B2D2D]` dengan teks putih semi-transparan (`text-white/80`).
*   **Aksen Primer (Maroon):** `#7B2D2D` dengan hover `#963E3E`.
*   **Bubble Chat Bot:** Latar belakang putih bersih (`bg-white`), bayangan sangat halus (`shadow-sm`), border abu-abu pasir (`border-[#E8E4E0]`), dan teks abu-abu gelap akademik (`text-[#2D1A1A]`).
*   **Bubble Chat User:** Latar belakang merah marun primer (`bg-[#7B2D2D]`) dengan teks putih bersih rata kanan.
*   **Box Sumber/Konteks:** Warna latar warm peach (`bg-[#FFF8F6]`), border halus (`border-[#F0DDD8]`), and teks maroon (`text-[#7B2D2D]`).

### B. Transisi Tab SPA (Single Page Application)
Navigasi di sidebar kiri dapat memindah layar secara *smooth* tanpa memicu *refresh* halaman browser, bertransisi antara:
1.  **💬 Chatbot:** Area obrolan AI interaktif, kolom input chat kustom melayang di bagian bawah dengan transisi halus, dan referensi ber-expander modern.
2.  **📊 Evaluasi Ragas:** Dashboard skripsi dengan layout grid.

---

## 📊 5. Spesifikasi Dashboard Evaluasi Interaktif (Chart.js)

Sebagai pengganti gambar grafik statis PNG dari Matplotlib, Tab Evaluasi akan memuat **Chart.js** via CDN (`https://cdn.jsdelivr.net/npm/chart.js`) guna merender grafik secara dinamis dan responsif langsung di browser:

*   **Tipe Bagan:** Bar Chart (Diagram Batang) yang membandingkan 4 metrik Ragas untuk Config A vs Config B vs Config baseline C (BM25).
*   **Estetika Grafik:** Warna batang menggunakan nuansa gradasi merah marun (`rgba(123, 45, 45, 0.8)`) dengan hover effect yang berubah terang saat pointer diarahkan ke grafik.
*   **Bagan Responsif:** Menggunakan opsi `responsive: true` and `maintainAspectRatio: false` agar bagan ter-render tajam dan pas di layar laptop maupun monitor.

---

## 🧪 6. Langkah Pengujian & Validasi UI
1.  **Validasi Endpoint API:**
    FastAPI secara otomatis menyediakan Swagger UI interaktif di `/docs`. Kita dapat memvalidasi endpoint `/api/chat` dan `/api/evaluation` secara instan lewat browser.
2.  **Validasi Concurrency:**
    Memastikan stream token tersaji secara real-time tanpa penundaan buffering browser, menggunakan EventSource atau Fetch API ReadableStream Reader.
3.  **Idempotency & Session History:**
    JS memelihara list chat history lokal dan mengirimkannya sebagai payload ke backend, menjaga kelancaran memori chat tanpa penumpukan session state di server.
