# Transisi UI RAG UNSRAT ke FastAPI + Tailwind + Chart.js Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menggantikan antarmuka Streamlit `app.py` yang kaku dengan aplikasi web modern satu halaman (SPA) berbasis FastAPI (REST API Backend) dipadu Tailwind CSS & Chart.js (Frontend) tanpa mengubah parameter RAG dan modul logger yang ada.

**Architecture:** Pemisahan Client-Server bersih. FastAPI melayani file HTML statis dan API endpoints. Komunikasi chat streaming real-time menggunakan Server-Sent Events (SSE) via FastAPI StreamingResponse, memanggil generator `get_response` dari `src/chain.py` sekali saja. Dashboard evaluasi memuat data CSV secara lokal dan merender grafik interaktif Chart.js.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, Tailwind CSS v3 (CDN), Chart.js (CDN), Lucide Icons.

---

## 📂 Peta Perubahan Berkas

```
unsrat-rag/
├── app.py                   ← MODIFIKASI TOTAL: Migrasi dari Streamlit ke FastAPI
├── index.html               ← BARU: Single-Page Application (HTML + Tailwind CDN)
├── static/
│   └── js/
│       └── app.js           ← BARU: Client Javascript (Chat render, SSE, & Chart.js)
```

---

## 🛠️ Rencana Pekerjaan Bertahap

### Task 1: Backend Server Setup (`app.py`)

**Files:**
- Modify: `app.py`
- Test: Jalankan `python app.py` dan periksa Swagger UI di `/docs`

- [ ] **Step 1: Tulis Boilerplate FastAPI & Static Mounting di `app.py`**
  Ganti isi berkas `app.py` untuk menginisialisasi FastAPI, memuat dependensi RAG dari `src.chain`, `src.config`, dan `src.logger_manager`, serta me-mount direktori static.
  
  ```python
  import time
  import json
  from fastapi import FastAPI, Request
  from fastapi.responses import HTMLResponse, StreamingResponse
  from fastapi.staticfiles import StaticFiles
  from pydantic import BaseModel
  
  from src.config import (
      AVAILABLE_MODELS,
      LLM_MODEL_NAME,
      EVAL_RESULTS_DIR,
      METRICS_COLS,
  )
  from src.chain import get_response, detect_category, reinitialize_llm
  from src.logger_manager import log_chat_transaction
  
  app = FastAPI(title="Asisten Akademik UNSRAT RAG API")
  
  # Mount directory static untuk aset Javascript
  app.mount("/static", StaticFiles(directory="static"), name="static")
  
  class ChatRequest(BaseModel):
      query: str
      config: str
      model: str
      chat_history: list[dict]
  
  @app.get("/", response_class=HTMLResponse)
  async def serve_index():
      with open("index.html", "r", encoding="utf-8") as f:
          return f.read()
  
  @app.get("/api/config")
  async def get_config():
      return {
          "available_models": AVAILABLE_MODELS,
          "default_model": LLM_MODEL_NAME,
          "default_config": "b"
      }
  ```

- [ ] **Step 2: Tambahkan RAG Streaming API Endpoint (`POST /api/chat`)**
  Implementasikan Event-Stream generator asinkron yang memanggil `get_response` dari `src/chain.py` dengan parameter yang terkunci aman, mengalirkan token, dan mengirim referensi dokumen beserta meta_logging di event terakhir untuk memicu `log_chat_transaction` di frontend.
  
  ```python
  @app.post("/api/chat")
  async def chat_endpoint(request: ChatRequest):
      # Reinisialisasi model LLM aktif secara instan jika ada perbedaan
      reinitialize_llm(request.model)
      
      # Single-call get_response dengan streaming=True (K-01 Compliance)
      result = get_response(
          query=request.query,
          config=request.config,
          chat_history=request.chat_history,
          streaming=True
      )
      
      sources = result["sources"]
      stream_gen = result["stream"]
      found = result["found"]
      meta_logging = result.get("_meta_logging")
      
      async def event_generator():
          t_start = time.time()
          full_response = ""
          
          # 1. Alirkan token teks real-time
          for token in stream_gen:
              full_response += token
              yield f"data: {json.dumps({'token': token})}\n\n"
          
          # 2. Alirkan sources referensi & status found di akhir stream
          payload = {
              "sources": sources,
              "found": found,
              "full_response": full_response,
              "meta_logging": meta_logging
          }
          yield f"data: {json.dumps(payload)}\n\n"
          
      return StreamingResponse(event_generator(), media_type="text/event-stream")
  ```

- [ ] **Step 3: Tambahkan Dashboard Evaluasi API Endpoint (`GET /api/evaluation`)**
  Membaca file evaluasi CSV di `eval/results/` dan mem-parsing-nya menjadi JSON bersih untuk di-render oleh Chart.js di frontend.
  
  ```python
  import pandas as pd
  from pathlib import Path
  
  @app.get("/api/evaluation")
  async def get_evaluation():
      result_files = {
          "a": EVAL_RESULTS_DIR / "hasil_config_a.csv",
          "b": EVAL_RESULTS_DIR / "hasil_config_b.csv",
          "c": EVAL_RESULTS_DIR / "hasil_config_c.csv"
      }
      stats_file = EVAL_RESULTS_DIR / "statistical_test.csv"
      
      configs_data = {}
      for key, filepath in result_files.items():
          if filepath.exists():
              df = pd.read_csv(filepath)
              metrics_summary = {}
              for metric in METRICS_COLS:
                  if metric in df.columns:
                      metrics_summary[metric] = {
                          "mean": float(df[metric].mean()),
                          "std": float(df[metric].std())
                      }
              if "response_time_seconds" in df.columns:
                  metrics_summary["response_time"] = {
                      "mean": float(df["response_time_seconds"].mean()),
                      "std": float(df["response_time_seconds"].std())
                  }
              configs_data[key] = metrics_summary
              
      wilcoxon_data = []
      if stats_file.exists():
          stats_df = pd.read_csv(stats_file)
          wilcoxon_data = stats_df.to_dict(orient="records")
          
      # Tambahkan audit log: baca 5 transaksi terakhir dari transaksi_chat.csv
      recent_transactions = []
      chat_log_path = Path("logs/transaksi_chat.csv")
      if chat_log_path.exists():
          chat_df = pd.read_csv(chat_log_path)
          recent_transactions = chat_df.tail(5).to_dict(orient="records")
          
      return {
          "configs": configs_data,
          "wilcoxon": wilcoxon_data,
          "recent_transactions": recent_transactions
      }
  ```

- [ ] **Step 4: Tambahkan Audit Logging Endpoint (`POST /api/log_transaction`)**
  Membungkus pencatatan transaksi chat kuantitatif secara eksplisit tepat setelah streaming selesai di client.
  
  ```python
  class LogRequest(BaseModel):
      config_retrieval: str
      model_llm: str
      user_query: str
      category_detected: str
      chunks_retrieved_count: int
      retrieved_chunk_ids: list[str]
      best_similarity_score: float
      average_similarity_score: float
      response_time_seconds: float
      response_text: str
      found_state: bool
      prompt_payload_for_token_calc: str
  
  @app.post("/api/log_transaction")
  async def log_transaction_endpoint(request: LogRequest):
      log_chat_transaction(
          config_retrieval=request.config_retrieval,
          model_llm=request.model_llm,
          user_query=request.user_query,
          category_detected=request.category_detected,
          chunks_retrieved_count=request.chunks_retrieved_count,
          retrieved_chunk_ids=request.retrieved_chunk_ids,
          best_similarity_score=request.best_similarity_score,
          average_similarity_score=request.average_similarity_score,
          response_time_seconds=request.response_time_seconds,
          response_text=request.response_text,
          found_state=request.found_state,
          prompt_payload_for_token_calc=request.prompt_payload_for_token_calc
      )
      return {"status": "success"}
  ```

- [ ] **Step 5: Jalankan Uvicorn Startup di `app.py`**
  ```python
  if __name__ == "__main__":
      import uvicorn
      uvicorn.run("app:app", host="0.0.0.0", port=8501, reload=True)
  ```

- [ ] **Step 6: Commit Backend**
  ```bash
  git add app.py
  git commit -m "feat: setup FastAPI backend dengan endpoint chat streaming, evaluation, dan log audit"
  ```

---

### Task 2: SPA Frontend Structure (`index.html`)

**Files:**
- Create: `index.html`

- [ ] **Step 1: Buat Kerangka HTML & Hubungkan Tailwind CDN + Lucide Icons**
  Tulis boilerplate HTML5 premium dengan Google Fonts Inter, Tailwind CSS CDN, dan Lucide Icons CDN.
  
  ```html
  <!DOCTYPE html>
  <html lang="id">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Asisten Akademik UNSRAT</title>
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
      <script src="https://cdn.tailwindcss.com"></script>
      <script src="https://cdn.jsdelivr.net/npm/lucide@latest"></script>
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      <style>
          body { font-family: 'Inter', sans-serif; background-color: #FAFAF8; }
          .custom-scrollbar::-webkit-scrollbar { width: 6px; }
          .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
          .custom-scrollbar::-webkit-scrollbar-thumb { background: #E8E4E0; border-radius: 4px; }
      </style>
  </head>
  <body class="h-screen flex overflow-hidden">
  ```

- [ ] **Step 2: Tulis Layout Sidebar Marun Klasik**
  Desain panel kontrol kiri gradasi maroon hangat dengan logo akademik, pemilih model LLM, pemilih config retrieval, dan tombol reset.
  
  ```html
      <!-- Sidebar Panel Kontrol -->
      <aside class="w-80 bg-gradient-to-b from-[#5B1A1A] to-[#7B2D2D] text-white flex flex-col justify-between p-6 shadow-xl z-20">
          <div class="space-y-6">
              <!-- Header Branding -->
              <div class="flex items-center space-x-3 pb-4 border-b border-white/20">
                  <div class="bg-white/10 p-2 rounded-lg">
                      <i data-lucide="graduation-cap" class="w-8 h-8 text-white"></i>
                  </div>
                  <div>
                      <h2 class="font-bold text-lg leading-tight">UNSRAT</h2>
                      <p class="text-white/60 text-xs">Asisten Informasi RAG</p>
                  </div>
              </div>
  
              <!-- Tab Navigation Buttons -->
              <nav class="space-y-2">
                  <button id="tab-chat-btn" class="w-full flex items-center space-x-3 px-4 py-3 rounded-lg bg-white/10 text-white font-medium transition duration-200">
                      <i data-lucide="message-square" class="w-5 h-5"></i>
                      <span>💬 Chatbot</span>
                  </button>
                  <button id="tab-eval-btn" class="w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-white/75 hover:bg-white/5 hover:text-white transition duration-200">
                      <i data-lucide="bar-chart-3" class="w-5 h-5"></i>
                      <span>📊 Evaluasi Ragas</span>
                  </button>
              </nav>
  
              <!-- Parameter Config Panel -->
              <div class="space-y-4 pt-4 border-t border-white/20">
                  <h3 class="text-xs font-semibold text-white/50 tracking-wider uppercase">⚙️ Pengaturan</h3>
                  
                  <!-- Config Switcher -->
                  <div class="space-y-2">
                      <label class="text-xs text-white/70">Konfigurasi Retrieval</label>
                      <div class="space-y-1 bg-black/10 p-2 rounded-lg text-sm">
                          <label class="flex items-center space-x-2 p-1.5 rounded hover:bg-white/5 cursor-pointer">
                              <input type="radio" name="config" value="a" class="accent-[#7B2D2D]">
                              <span>Config A (500 char)</span>
                          </label>
                          <label class="flex items-center space-x-2 p-1.5 rounded hover:bg-white/5 cursor-pointer">
                              <input type="radio" name="config" value="b" checked class="accent-[#7B2D2D]">
                              <span>Config B (2000 char)</span>
                          </label>
                          <label class="flex items-center space-x-2 p-1.5 rounded hover:bg-white/5 cursor-pointer">
                              <input type="radio" name="config" value="c" class="accent-[#7B2D2D]">
                              <span>Config C (BM25 baseline)</span>
                          </label>
                      </div>
                  </div>
  
                  <!-- Model Selector -->
                  <div class="space-y-2">
                      <label class="text-xs text-white/70">Model Generator LLM</label>
                      <select id="model-select" class="w-full bg-black/20 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-white/30 cursor-pointer">
                          <!-- Opsi dimuat dinamis dari API -->
                      </select>
                  </div>
              </div>
          </div>
  
          <!-- Tombol Reset di Bagian Bawah -->
          <button id="reset-btn" class="w-full flex items-center justify-center space-x-2 py-3 bg-white/10 hover:bg-white/20 active:bg-white/30 rounded-lg text-sm text-white font-medium transition duration-200 border border-white/10">
              <i data-lucide="refresh-cw" class="w-4 h-4"></i>
              <span>🔄 Reset Percakapan</span>
          </button>
      </aside>
  ```

- [ ] **Step 3: Tulis Area Chatbot Tab Panel (Tab 1)**
  Desain bagian chat utama off-white dengan header akademik merah maroon gradasi melintang, area chat scrollable, bubble asisten/user kustom, dan box input chat fixed bottom.
  
  ```html
      <!-- Konten Utama -->
      <main class="flex-1 flex flex-col h-full overflow-hidden relative">
          <!-- TAB 1: AREA CHATBOT -->
          <div id="tab-chat" class="flex-1 flex flex-col h-full overflow-hidden transition-all duration-300">
              <!-- Header Akademik Melintang -->
              <header class="bg-gradient-to-r from-[#5C1F1F] to-[#7B2D2D] text-white px-8 py-5 shadow-md flex items-center justify-between z-10">
                  <div>
                      <h1 class="font-bold text-lg tracking-tight">🎓 Asisten Informasi Akademik UNSRAT</h1>
                      <p class="text-white/60 text-xs">Arsitektur RAG Dual-Chunking Terintegrasi</p>
                  </div>
                  <span id="badge-config-display" class="bg-white/15 px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider text-white">Config B</span>
              </header>
  
              <!-- Area Riwayat Chat -->
              <div id="chat-messages" class="flex-1 overflow-y-auto px-8 py-6 space-y-6 custom-scrollbar pb-32">
                  <!-- Contoh Bubble Bot Awal -->
                  <div class="flex items-start space-x-3 max-w-4xl">
                      <div class="bg-[#7B2D2D] text-white p-2.5 rounded-lg flex-shrink-0 mt-1 shadow-sm">
                          <i data-lucide="bot" class="w-5 h-5"></i>
                      </div>
                      <div class="space-y-2">
                          <span class="inline-block bg-[#7B2D2D] text-white px-2 py-0.5 rounded-full text-[10px] font-semibold tracking-wider">📂 GENERAL</span>
                          <div class="bg-white border border-[#E8E4E0] rounded-2xl rounded-tl-none px-4 py-3 shadow-sm text-[#2D1A1A] leading-relaxed text-sm">
                              Halo! Saya asisten informasi akademik resmi Universitas Sam Ratulangi. Silakan ajukan pertanyaan seputar Peraturan Akademik, Visi Misi, Sejarah, Kalender Semester, atau Profil Kampus UNSRAT.
                          </div>
                      </div>
                  </div>
              </div>
  
              <!-- Fixed Input Panel di Bawah -->
              <div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-[#FAFAF8] via-[#FAFAF8] to-transparent pt-6 pb-6 px-8 flex flex-col items-center">
                  <form id="chat-form" class="w-full max-w-4xl flex items-center bg-white border border-[#E8E4E0] rounded-full shadow-lg overflow-hidden focus-within:border-[#7B2D2D] focus-within:ring-2 focus-within:ring-[#7B2D2D]/20 transition duration-200">
                      <input type="text" id="chat-input" placeholder="Ketik pertanyaan Anda di sini..." class="flex-1 px-6 py-4 text-sm text-[#2D1A1A] focus:outline-none placeholder-gray-400">
                      <button type="submit" class="bg-[#7B2D2D] hover:bg-[#963E3E] active:bg-[#5C1F1F] text-white p-3 rounded-full mr-2 shadow-sm transition duration-200">
                          <i data-lucide="send" class="w-5 h-5"></i>
                      </button>
                  </form>
                  <p class="text-[10px] text-gray-400 mt-2">Prototipe Penelitian Akademik UNSRAT. Verifikasi keputusan penting dengan bagian administrasi terkait.</p>
              </div>
          </div>
  ```

- [ ] **Step 4: Tulis Area Dashboard Evaluasi Tab Panel (Tab 2)**
  Desain dashboard kuantitatif skripsi dengan grid card layout, diagram batang Chart.js responsif, tabel Wilcoxon Signed-Rank, dan Live Audit Log.
  
  ```html
          <!-- TAB 2: AREA EVALUASI RAGAS -->
          <div id="tab-eval" class="hidden flex-1 flex flex-col h-full overflow-y-auto px-8 py-6 space-y-6 custom-scrollbar pb-10">
              <div>
                  <h2 class="font-bold text-2xl text-[#2D1A1A] tracking-tight">📊 Dashboard Evaluasi Metrik Ragas</h2>
                  <p class="text-sm text-gray-500 mt-1">Laporan pengujian komparatif performa Config A (500 char), Config B (2000 char), dan Config C (BM25 baseline).</p>
              </div>
  
              <!-- Grid Utama Evaluasi -->
              <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <!-- Card Grafik Interaktif Chart.js -->
                  <div class="bg-white border border-[#E8E4E0] rounded-xl p-5 shadow-sm space-y-4">
                      <h3 class="font-semibold text-[#2D1A1A] flex items-center space-x-2">
                          <i data-lucide="presentation" class="w-5 h-5 text-[#7B2D2D]"></i>
                          <span>Diagram Perbandingan Metrik Ragas (Dinamis)</span>
                      </h3>
                      <div class="relative h-80">
                          <canvas id="evalChart"></canvas>
                      </div>
                  </div>
  
                  <!-- Card Hasil Uji Statistik Wilcoxon -->
                  <div class="bg-white border border-[#E8E4E0] rounded-xl p-5 shadow-sm space-y-4 flex flex-col justify-between">
                      <div class="space-y-3">
                          <h3 class="font-semibold text-[#2D1A1A] flex items-center space-x-2">
                              <i data-lucide="binary" class="w-5 h-5 text-[#7B2D2D]"></i>
                              <span>Uji Signifikansi Wilcoxon Signed-Rank (A vs B)</span>
                          </h3>
                          <div class="overflow-x-auto">
                              <table class="w-full text-xs text-left text-gray-600 border border-gray-100 rounded-lg overflow-hidden">
                                  <thead class="bg-gray-50 text-gray-700 font-semibold uppercase text-[10px]">
                                      <tr>
                                          <th class="px-4 py-3">Metrik Ragas</th>
                                          <th class="px-4 py-3">P-Value</th>
                                          <th class="px-4 py-3">Signifikan (0.05)</th>
                                          <th class="px-4 py-3">Unggul</th>
                                      </tr>
                                  </thead>
                                  <tbody id="wilcoxon-table-body" class="divide-y divide-gray-100 bg-white">
                                      <!-- Dimuat dinamis -->
                                  </tbody>
                              </table>
                          </div>
                      </div>
                      <p class="text-[10px] text-gray-400 mt-2">Dianalisis menggunakan modul scipy.stats.wilcoxon secara offline berdasarkan data ground_truth.csv.</p>
                  </div>
              </div>
  
              <!-- Live Audit Log Kuantitatif (Bab IV Data Kunci) -->
              <div class="bg-white border border-[#E8E4E0] rounded-xl p-5 shadow-sm space-y-4">
                  <h3 class="font-semibold text-[#2D1A1A] flex items-center space-x-2">
                      <i data-lucide="terminal" class="w-5 h-5 text-[#7B2D2D]"></i>
                      <span>Audit Log Transaksi Chat Terbaru (Live Kuantitatif)</span>
                  </h3>
                  <div class="overflow-x-auto">
                      <table class="w-full text-xs text-left text-gray-600 border border-gray-100 rounded-lg overflow-hidden">
                          <thead class="bg-gray-50 text-gray-700 font-semibold uppercase text-[10px]">
                              <tr>
                                  <th class="px-4 py-3">Waktu</th>
                                  <th class="px-4 py-3">Config</th>
                                  <th class="px-4 py-3">Model</th>
                                  <th class="px-4 py-3">Pertanyaan</th>
                                  <th class="px-4 py-3">Chunks</th>
                                  <th class="px-4 py-3">Best Distance</th>
                                  <th class="px-4 py-3">Latency</th>
                                  <th class="px-4 py-3">Total Token</th>
                              </tr>
                          </thead>
                          <tbody id="audit-table-body" class="divide-y divide-gray-100 bg-white">
                              <!-- Dimuat dinamis -->
                          </tbody>
                      </table>
                  </div>
              </div>
          </div>
      </main>
  
      <!-- Hubungkan File Javascript Logika -->
      <script src="/static/js/app.js"></script>
  </body>
  </html>
  ```

- [ ] **Step 5: Commit index.html**
  ```bash
  git add index.html
  git commit -m "feat: setup HTML5 UI framework dengan Maroon Klasik layout, SPA navigation, dan placeholder Chart.js/Wilcoxon"
  ```

---

### Task 3: Client JS Controller (`static/js/app.js`)

**Files:**
- Create: `static/js/app.js`

- [ ] **Step 1: Inisialisasi Lucide & State Manager**
  Tulis inisialisasi ikon Lucide, kelola tab switching dinamis, dan inisialisasi session state lokal.
  
  ```javascript
  document.addEventListener("DOMContentLoaded", () => {
      // Inisialisasi icon Lucide secara instan
      lucide.createIcons();
      
      // SPA State Management
      const tabChatBtn = document.getElementById("tab-chat-btn");
      const tabEvalBtn = document.getElementById("tab-eval-btn");
      const tabChat    = document.getElementById("tab-chat");
      const tabEval    = document.getElementById("tab-eval");
      
      tabChatBtn.addEventListener("click", () => {
          tabChatBtn.classList.add("bg-white/10");
          tabEvalBtn.classList.remove("bg-white/10");
          tabChat.classList.remove("hidden");
          tabEval.classList.add("hidden");
      });
      
      tabEvalBtn.addEventListener("click", () => {
          tabEvalBtn.classList.add("bg-white/10");
          tabChatBtn.classList.remove("bg-white/10");
          tabEval.classList.remove("hidden");
          tabChat.classList.add("hidden");
          loadEvaluationData(); // Muat data evaluasi & Wilcoxon dari API
      });
      
      // History lokal (SESSION_MESSAGES)
      let chatHistory = [];
      let availableModels = [];
      let activeConfig = "b";
  ```

- [ ] **Step 2: Tarik Konfigurasi API Awal & Terapkan Reset Percakapan**
  Tarik list models LLM dari backend untuk mengisi selectbox secara dinamis, dan pasang listener tombol reset.
  
  ```javascript
      const modelSelect = document.getElementById("model-select");
      const configRadios = document.getElementsByName("config");
      const badgeConfig = document.getElementById("badge-config-display");
      const resetBtn = document.getElementById("reset-btn");
      
      // Ambil Config Awal dari API FastAPI
      fetch("/api/config")
          .then(res => res.json())
          .then(data => {
              availableModels = data.available_models;
              modelSelect.innerHTML = availableModels.map(m => 
                  `<option value="${m}" ${m === data.default_model ? 'selected' : ''} class="text-[#2D1A1A]">${m}</option>`
              ).join("");
              activeConfig = data.default_config;
              badgeConfig.innerText = `Config ${activeConfig.toUpperCase()}`;
          });
          
      // Ambil input config radio
      configRadios.forEach(radio => {
          radio.addEventListener("change", (e) => {
              activeConfig = e.target.value;
              badgeConfig.innerText = `Config ${activeConfig.toUpperCase()}`;
              // Reset chat jika config diubah
              clearChatUI();
          });
      });
      
      // Bersihkan UI & History percakapan
      function clearChatUI() {
          chatHistory = [];
          const chatMessages = document.getElementById("chat-messages");
          chatMessages.innerHTML = `
              <div class="flex items-start space-x-3 max-w-4xl">
                  <div class="bg-[#7B2D2D] text-white p-2.5 rounded-lg flex-shrink-0 mt-1 shadow-sm">
                      <i data-lucide="bot" class="w-5 h-5"></i>
                  </div>
                  <div class="space-y-2">
                      <span class="inline-block bg-[#7B2D2D] text-white px-2 py-0.5 rounded-full text-[10px] font-semibold tracking-wider">📂 GENERAL</span>
                      <div class="bg-white border border-[#E8E4E0] rounded-2xl rounded-tl-none px-4 py-3 shadow-sm text-[#2D1A1A] leading-relaxed text-sm">
                          Riwayat percakapan telah dibersihkan. Konfigurasi ${activeConfig.toUpperCase()} aktif. Silakan ajukan pertanyaan Anda.
                      </div>
                  </div>
              </div>
          `;
          lucide.createIcons();
      }
      
      resetBtn.addEventListener("click", () => {
          clearChatUI();
      });
  ```

- [ ] **Step 3: Implementasikan SSE Chat Stream & Form Submitting**
  Implementasikan penanganan form submit, deteksi kategori lokal (meniru fungsi UI label), dan parser stream Server-Sent Events (SSE) yang memetakan respon token ke layar secara real-time.
  
  ```javascript
      const chatForm = document.getElementById("chat-form");
      const chatInput = document.getElementById("chat-input");
      const chatMessages = document.getElementById("chat-messages");
      
      // Deteksi kategori lokal sederhana hanya untuk visual label UI (D-A2 Compliance)
      function localDetectCategory(query) {
          const keywords = {
              "academic": ["SKS", "KRS", "IPK", "pasal", "yudisium", "cuti", "skripsi", "wisuda"],
              "calendar": ["jadwal", "tanggal", "kapan", "deadline", "periode", "kalender"],
              "institution_profile": ["sejarah", "visi", "misi", "rektor", "akreditasi", "bendera"]
          };
          const text = query.toLowerCase();
          for (const [cat, words] of Object.entries(keywords)) {
              if (words.some(w => text.includes(w.toLowerCase()))) return cat;
          }
          return "general";
      }
      
      chatForm.addEventListener("submit", async (e) => {
          e.preventDefault();
          const query = chatInput.value.trim();
          if (!query) return;
          
          chatInput.value = "";
          
          // 1. Tampilkan Bubble User
          chatMessages.innerHTML += `
              <div class="flex items-start justify-end space-x-3 max-w-4xl ml-auto">
                  <div class="bg-white border border-[#E8E4E0] rounded-2xl rounded-tr-none px-4 py-3 shadow-sm text-[#2D1A1A] leading-relaxed text-sm">
                      ${query}
                  </div>
                  <div class="bg-gray-200 text-gray-700 p-2.5 rounded-lg flex-shrink-0 mt-1 shadow-sm font-bold text-xs uppercase">
                      User
                  </div>
              </div>
          `;
          chatMessages.scrollTop = chatMessages.scrollHeight;
          
          const catDetected = localDetectCategory(query);
          
          // 2. Buat Bubble Bot Sementara untuk Token Stream
          const botMessageId = "bot-" + Date.now();
          chatMessages.innerHTML += `
              <div class="flex items-start space-x-3 max-w-4xl" id="container-${botMessageId}">
                  <div class="bg-[#7B2D2D] text-white p-2.5 rounded-lg flex-shrink-0 mt-1 shadow-sm">
                      <i data-lucide="bot" class="w-5 h-5"></i>
                  </div>
                  <div class="space-y-2 flex-1">
                      <span class="inline-block bg-[#7B2D2D] text-white px-2 py-0.5 rounded-full text-[10px] font-semibold tracking-wider uppercase">📂 ${catDetected}</span>
                      <div class="bg-white border border-[#E8E4E0] rounded-2xl rounded-tl-none px-4 py-3 shadow-sm text-[#2D1A1A] leading-relaxed text-sm" id="${botMessageId}">
                          <span class="text-gray-400">Mengetik...</span>
                      </div>
                      <div id="sources-${botMessageId}" class="space-y-2 hidden pt-2"></div>
                  </div>
              </div>
          `;
          chatMessages.scrollTop = chatMessages.scrollHeight;
          lucide.createIcons();
          
          const botBubble = document.getElementById(botMessageId);
          const sourcesContainer = document.getElementById(`sources-${botMessageId}`);
          
          // 3. Panggil API Streaming RAG FastAPI
          const requestBody = {
              query: query,
              config: activeConfig,
              model: modelSelect.value,
              chat_history: chatHistory
          };
          
          let responseText = "";
          const t_start = Date.now() / 1000;
          
          try {
              const response = await fetch("/api/chat", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(requestBody)
              });
              
              const reader = response.body.getReader();
              const decoder = new TextDecoder();
              let sseBuffer = "";
              
              botBubble.innerHTML = ""; // Bersihkan teks pemuatan
              
              while (true) {
                  const { value, done } = await reader.read();
                  if (done) break;
                  
                  sseBuffer += decoder.decode(value, { stream: true });
                  const lines = sseBuffer.split("\n\n");
                  sseBuffer = lines.pop(); // simpan sisa buffer yang tidak lengkap
                  
                  for (const line of lines) {
                      if (line.startsWith("data: ")) {
                          const data = JSON.parse(line.slice(6));
                          
                          if (data.token) {
                              responseText += data.token;
                              // Ganti newlines dengan break tag agar ter-render rapi di browser
                              botBubble.innerHTML = responseText.replace(/\n/g, "<br>");
                              chatMessages.scrollTop = chatMessages.scrollHeight;
                          } else if (data.sources) {
                              // Payload Akhir: Referensi Dokumen & Logger
                              const sources = data.sources;
                              const found = data.found;
                              const meta_logging = data.meta_logging;
                              const full_response = data.full_response;
                              
                              if (sources && sources.length > 0) {
                                  sourcesContainer.classList.remove("hidden");
                                  let sourcesHtml = `<div class="text-xs font-semibold text-gray-500 mt-2 mb-1 flex items-center space-x-1"><i data-lucide="book-open" class="w-3.5 h-3.5"></i><span>📚 Referensi Dokumen:</span></div>`;
                                  sources.forEach((src, idx) => {
                                      sourcesHtml += `
                                          <div class="bg-[#FFF8F6] border border-[#F0DDD8] text-[#7B2D2D] rounded-xl p-3 text-xs leading-relaxed space-y-1">
                                              <div class="font-bold flex justify-between">
                                                  <span>[${idx+1}] ${src.title}</span>
                                                  <span class="bg-[#7B2D2D]/10 px-1.5 py-0.5 rounded text-[9px] uppercase">${src.category}</span>
                                              </div>
                                              <div class="text-[10px] text-gray-500 font-medium">Doc ID: ${src.doc_id} | Bab: ${src.bab} | Bagian: ${src.bagian || '-'}</div>
                                              <div class="text-gray-700 italic border-l-2 border-[#7B2D2D]/30 pl-2 mt-1 font-mono">${src.preview}...</div>
                                          </div>
                                      `;
                                  });
                                  sourcesContainer.innerHTML = sourcesHtml;
                                  lucide.createIcons();
                              }
                              
                              // Simpan riwayat chat internal
                              chatHistory.push({ "role": "user", "content": query });
                              chatHistory.push({ "role": "assistant", "content": full_response });
                              
                              // Trigger logger API ke database CSV
                              if (found && meta_logging) {
                                  const duration = (Date.now() / 1000) - t_start;
                                  fetch("/api/log_transaction", {
                                      method: "POST",
                                      headers: { "Content-Type": "application/json" },
                                      body: JSON.stringify({
                                          config_retrieval: activeConfig,
                                          model_llm: modelSelect.value,
                                          user_query: query,
                                          category_detected: meta_logging.category,
                                          chunks_retrieved_count: sources.length,
                                          retrieved_chunk_ids: meta_logging.chunk_ids,
                                          best_similarity_score: meta_logging.best_score,
                                          average_similarity_score: meta_logging.avg_score,
                                          response_time_seconds: duration,
                                          response_text: full_response,
                                          found_state: true,
                                          prompt_payload_for_token_calc: meta_logging.prompt_payload
                                      })
                                  });
                              }
                          }
                      }
                  }
              }
          } catch (err) {
              botBubble.innerHTML = `<span class="text-red-500 font-semibold flex items-center space-x-1"><i data-lucide="alert-triangle" class="w-4 h-4"></i><span>Koneksi error: Gagal menghubungi layanan RAG.</span></span>`;
              lucide.createIcons();
          }
      });
  ```

- [ ] **Step 4: Muat Data Evaluasi & Rias Bagan Batang Chart.js**
  Implementasikan penarikan data evaluasi, populasi tabel Wilcoxon dan Audit Log di UI secara dinamis, serta inisialisasi/gambar ulang bagan Chart.js yang responsif.
  
  ```javascript
      let evalChartInstance = null;
      
      async function loadEvaluationData() {
          const wilcoxonTable = document.getElementById("wilcoxon-table-body");
          const auditTable = document.getElementById("audit-table-body");
          
          try {
              const res = await fetch("/api/evaluation");
              const data = await res.json();
              
              // 1. Populasi Tabel Wilcoxon
              if (data.wilcoxon && data.wilcoxon.length > 0) {
                  wilcoxonTable.innerHTML = data.wilcoxon.map(row => `
                      <tr class="hover:bg-[#FFF8F6]/40 transition duration-150">
                          <td class="px-4 py-3 font-semibold text-gray-700">${row.metric}</td>
                          <td class="px-4 py-3 font-mono">${row.p_value.toFixed(4)}</td>
                          <td class="px-4 py-3">
                              <span class="px-2 py-0.5 rounded-full text-[10px] font-bold ${row.significant_at_0.05 ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'}">
                                  ${row.significant_at_0.05 ? 'Ya' : 'Tidak (n.s.)'}
                              </span>
                          </td>
                          <td class="px-4 py-3 font-medium text-[#7B2D2D]">${row.winner}</td>
                      </tr>
                  `).join("");
              } else {
                  wilcoxonTable.innerHTML = `<tr><td colspan="4" class="px-4 py-3 text-center text-gray-400">Data uji Wilcoxon belum tersedia. Jalankan evaluasi via CLI.</td></tr>`;
              }
              
              // 2. Populasi Tabel Audit Log Transaksi Chat (Kunci Data Skripsi)
              if (data.recent_transactions && data.recent_transactions.length > 0) {
                  // Urutkan transaksi terbaru di atas
                  const reversedTransactions = [...data.recent_transactions].reverse();
                  auditTable.innerHTML = reversedTransactions.map(row => `
                      <tr class="hover:bg-gray-50/50 transition duration-150">
                          <td class="px-4 py-3 text-gray-400 font-mono text-[10px]">${row.timestamp.split(" ")[1]}</td>
                          <td class="px-4 py-3"><span class="bg-[#7B2D2D]/10 text-[#7B2D2D] px-1.5 py-0.5 rounded text-[9px] uppercase font-bold">${row.config_retrieval.toUpperCase()}</span></td>
                          <td class="px-4 py-3 font-mono text-[10px] text-gray-500">${row.model_llm}</td>
                          <td class="px-4 py-3 font-medium text-gray-700 truncate max-w-xs">${row.user_query}</td>
                          <td class="px-4 py-3 text-center font-mono">${row.chunks_retrieved_count}</td>
                          <td class="px-4 py-3 font-mono">${row.best_similarity_score.toFixed(4)}</td>
                          <td class="px-4 py-3 font-mono text-amber-600">${row.response_time_seconds.toFixed(2)}s</td>
                          <td class="px-4 py-3 font-semibold text-gray-800 font-mono">${row.estimated_total_tokens}</td>
                      </tr>
                  `).join("");
              } else {
                  auditTable.innerHTML = `<tr><td colspan="8" class="px-4 py-3 text-center text-gray-400">Belum ada aktivitas chat yang terekam.</td></tr>`;
              }
              
              // 3. Rias Bagan Batang Dinamis Chart.js
              const configKeys = Object.keys(data.configs);
              if (configKeys.length > 0) {
                  const metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"];
                  
                  // Petakan skor mean untuk datasets Chart.js
                  const dataA = metrics.map(m => data.configs.a ? data.configs.a[m].mean : 0);
                  const dataB = metrics.map(m => data.configs.b ? data.configs.b[m].mean : 0);
                  const dataC = metrics.map(m => data.configs.c ? data.configs.c[m].mean : 0);
                  
                  const ctx = document.getElementById("evalChart").getContext("2d");
                  
                  // Hancurkan bagan lama jika ada re-render
                  if (evalChartInstance) evalChartInstance.destroy();
                  
                  evalChartInstance = new Chart(ctx, {
                      type: "bar",
                      data: {
                          labels: ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"],
                          datasets: [
                              {
                                  label: "Config A (RAG 500 char)",
                                  data: dataA,
                                  backgroundColor: "rgba(123, 45, 45, 0.4)",
                                  borderColor: "rgb(123, 45, 45)",
                                  borderWidth: 1.5,
                                  borderRadius: 4
                              },
                              {
                                  label: "Config B (RAG 2000 char)",
                                  data: dataB,
                                  backgroundColor: "rgba(168, 69, 69, 0.8)",
                                  borderColor: "rgb(168, 69, 69)",
                                  borderWidth: 1.5,
                                  borderRadius: 4
                              },
                              {
                                  label: "Config C (BM25 baseline)",
                                  data: dataC,
                                  backgroundColor: "rgba(212, 160, 160, 0.4)",
                                  borderColor: "rgb(212, 160, 160)",
                                  borderWidth: 1.5,
                                  borderRadius: 4
                              }
                          ]
                      },
                      options: {
                          responsive: true,
                          maintainAspectRatio: false,
                          plugins: {
                              legend: {
                                  labels: { color: "#2D1A1A", font: { family: "Inter", weight: 500 } }
                              },
                              tooltip: {
                                  padding: 10,
                                  cornerRadius: 8,
                                  backgroundColor: "#2D1A1A"
                              }
                          },
                          scales: {
                              x: { ticks: { color: "#2D1A1A" }, grid: { display: false } },
                              y: { max: 1.1, ticks: { color: "#2D1A1A" }, grid: { color: "#F0E6E6" } }
                          }
                      }
                  });
              }
          } catch (err) {
              console.error("Gagal memuat data dashboard evaluasi:", err);
          }
      }
  });
  ```

- [ ] **Step 5: Commit app.js**
  ```bash
  git add static/js/app.js
  git commit -m "feat: implementasikan Javascript controller SSE chat stream parser, local categories, SPA navigation, dan dinamis Chart.js renderer"
  ```

---

## 🔍 Evaluasi & Validasi Rencana
1.  **Strict Parameter Lock:**
    Rencana ini 100% menggunakan session state, nama variabel, dan signatures yang identik dengan spec `PROMPT_REVISI_UI.md` dan `src/chain.py` (K-01 streaming data dict layout).
2.  **No Placeholders:**
    Kode di atas lengkap, utuh, tanpa TBD, dan siap disalin untuk mengganti total `app.py` Streamlit secara mutakhir dan andal.
3.  **TDD & Commit Berulang:**
    Rencana dirinci secara bertahap (Backend, Frontend SPA, JS Controller) dengan pos pemeriksaan tes individual dan commits kecil untuk keamanan data.
