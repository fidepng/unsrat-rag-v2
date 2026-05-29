"""
app.py — Entry point Backend API FastAPI, Sistem Chatbot RAG Akademik UNSRAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Menggantikan antarmuka Streamlit yang lama dengan API backend modern berbasis FastAPI.
Menyediakan REST API untuk konfigurasi, RAG streaming chat (Server-Sent Events / SSE),
dashboard evaluasi Ragas kuantitatif, dan logging transaksi audit skripsi secara eksplisit.
"""

import os
import time
import json
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
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
from src.logger_manager import log_chat_transaction, log_system_event

# Inisialisasi folder static untuk Javascript secara otomatis dan aman
os.makedirs("static/js", exist_ok=True)

# Inisialisasi FastAPI
app = FastAPI(
    title="Asisten Akademik UNSRAT RAG API",
    description="Backend API untuk Sistem Chatbot Informasi Akademik UNSRAT berbasis RAG",
    version="2.0.0"
)

# Mount directory static untuk aset Javascript/CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── PYDANTIC SCHEMAS ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    config: str
    model: str
    chat_history: list[dict]

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

# ── ROUTERS & ENDPOINTS ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """
    Melayani single page application frontend (index.html).
    Jika index.html belum dibuat (Task 2 belum dijalankan), menyajikan halaman info API.
    """
    index_path = Path("index.html")
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>UNSRAT RAG API - Standby</title>
            <style>
                body { font-family: system-ui, sans-serif; text-align: center; padding: 50px; background: #FAF9F6; color: #2D1A1A; }
                h1 { color: #7B2D2D; }
                a { color: #A84545; font-weight: bold; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>🎓 UNSRAT RAG API Backend Server</h1>
            <p>Server FastAPI berhasil diinisialisasi dan berjalan lancar!</p>
            <p>Halaman <code>index.html</code> (Frontend SPA) belum dibuat.</p>
            <p>Silakan buka Swagger UI di <a href="/docs">/docs</a> untuk menguji API.</p>
        </body>
        </html>
        """

@app.get("/api/config")
async def get_config():
    """
    Mengembalikan data konfigurasi awal sistem untuk UI frontend.
    """
    return {
        "available_models": AVAILABLE_MODELS,
        "default_model": LLM_MODEL_NAME,
        "default_config": "b"
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    RAG Streaming API Endpoint (EventSource / Server-Sent Events).
    
    Menerima request query, config, model LLM pilihan, dan history percakapan.
    Melakukan re-inisialisasi model jika berganti, memanggil get_response secara single-call
    untuk efisiensi biaya kuota API, dan mengalirkan token teks secara real-time.
    Di event akhir, mengirimkan metadata 'sources' dan 'meta_logging' untuk client audit.
    """
    try:
        # Reinisialisasi model LLM aktif secara instan jika ada perbedaan
        reinitialize_llm(request.model)
    except Exception as e:
        log_system_event("error", "LLM_INIT_FAIL", f"Gagal reinisialisasi model {request.model}: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal memuat model LLM: {str(e)}")

    # Single-call get_response dengan streaming=True (K-01 Compliance)
    result = get_response(
        query=request.query,
        config=request.config,
        chat_history=request.chat_history,
        streaming=True
    )
    
    sources = result.get("sources", [])
    stream_gen = result.get("stream")
    found = result.get("found", False)
    meta_logging = result.get("_meta_logging")

    if not stream_gen:
        fallback_text = result.get("answer", "Maaf, saya tidak menemukan informasi yang relevan dalam dokumen yang tersedia.")
        
        def fallback_generator():
            # Alirkan teks fallback sebagai satu token
            yield f"data: {json.dumps({'token': fallback_text})}\n\n"
            # Kirim payload penutup dengan status found=False
            payload = {
                "sources": [],
                "found": False,
                "full_response": fallback_text,
                "meta_logging": None
            }
            yield f"data: {json.dumps(payload)}\n\n"
            
        return StreamingResponse(fallback_generator(), media_type="text/event-stream")

    def event_generator():
        full_response = ""
        
        # 1. Alirkan token teks real-time
        try:
            for token in stream_gen:
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            log_system_event("error", "STREAM_TOKEN_ERROR", f"Error saat streaming token: {e}")
            err_token = f"\n\n[Error Stream: {str(e)}]"
            yield f"data: {json.dumps({'token': err_token})}\n\n"
        
        # 2. Alirkan sources referensi & status found di akhir stream
        payload = {
            "sources": sources,
            "found": found,
            "full_response": full_response,
            "meta_logging": meta_logging
        }
        yield f"data: {json.dumps(payload)}\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/evaluation")
async def get_evaluation():
    """
    Dashboard Evaluasi API Endpoint.
    
    Membaca hasil evaluasi offline di folder 'eval/results/' (hasil_config_a.csv,
    hasil_config_b.csv, hasil_config_c.csv) dan statistical_test.csv secara lokal,
    kemudian merangkum data statistik rata-rata (mean) dan standar deviasi (std)
    serta uji Wilcoxon untuk visualisasi Chart.js.
    Membaca juga 5 transaksi chat terbaru dari logs/transaksi_chat.csv untuk live audit log.
    """
    result_files = {
        "a": EVAL_RESULTS_DIR / "hasil_config_a.csv",
        "b": EVAL_RESULTS_DIR / "hasil_config_b.csv",
        "c": EVAL_RESULTS_DIR / "hasil_config_c.csv"
    }
    stats_file = EVAL_RESULTS_DIR / "statistical_test.csv"
    
    configs_data = {}
    for key, filepath in result_files.items():
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                metrics_summary = {}
                for metric in METRICS_COLS:
                    if metric in df.columns:
                        mean_val = df[metric].mean()
                        std_val = df[metric].std()
                        metrics_summary[metric] = {
                            "mean": float(mean_val) if pd.notna(mean_val) else 0.0,
                            "std": float(std_val) if pd.notna(std_val) else 0.0
                        }
                if "response_time_seconds" in df.columns:
                    mean_rt = df["response_time_seconds"].mean()
                    std_rt = df["response_time_seconds"].std()
                    metrics_summary["response_time"] = {
                        "mean": float(mean_rt) if pd.notna(mean_rt) else 0.0,
                        "std": float(std_rt) if pd.notna(std_rt) else 0.0
                    }
                configs_data[key] = metrics_summary
            except Exception as e:
                log_system_event("error", "EVAL_READ_FAIL", f"Gagal membaca file evaluasi {filepath.name}: {e}")
                configs_data[key] = {}
            
    wilcoxon_data = []
    if stats_file.exists():
        try:
            stats_df = pd.read_csv(stats_file)
            stats_df = stats_df.fillna("")
            wilcoxon_data = stats_df.to_dict(orient="records")
        except Exception as e:
            log_system_event("error", "STATS_READ_FAIL", f"Gagal membaca statistical_test.csv: {e}")
            
    # Live Audit Log: ambil 5 transaksi chat terbaru secara lokal
    recent_transactions = []
    chat_log_path = Path("logs/transaksi_chat.csv")
    if chat_log_path.exists():
        try:
            chat_df = pd.read_csv(chat_log_path)
            if not chat_df.empty:
                chat_df = chat_df.fillna("")
                recent_transactions = chat_df.tail(5).to_dict(orient="records")
        except Exception as e:
            log_system_event("error", "LOGS_READ_FAIL", f"Gagal membaca transaksi_chat.csv: {e}")
            
    return {
        "configs": configs_data,
        "wilcoxon": wilcoxon_data,
        "recent_transactions": recent_transactions
    }

@app.post("/api/log_transaction")
async def log_transaction_endpoint(request: LogRequest):
    """
    Audit Logging Endpoint.
    
    Menerima detail transaksi Q&A kuantitatif dari client setelah streaming selesai
    dan mencatatnya secara eksplisit ke 'logs/transaksi_chat.csv' menggunakan fungsi
    bawaan 'log_chat_transaction'.
    """
    try:
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
    except Exception as e:
        log_system_event("error", "AUDIT_LOG_FAIL", f"Gagal menulis transaksi Q&A ke CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal mencatat log transaksi: {str(e)}")

# Startup uvicorn
if __name__ == "__main__":
    import uvicorn
    
    # Custom Server untuk menangani registrasi sinyal di secondary thread secara aman
    class CustomServer(uvicorn.Server):
        def install_signal_handlers(self) -> None:
            try:
                super().install_signal_handlers()
            except ValueError:
                # signal only works in main thread of the main interpreter
                # Tangani dan bypass secara aman bila dijalankan di secondary thread/wrapper
                pass

    # Menjalankan di host 0.0.0.0 dan port 8501 (reload dimatikan demi stabilitas signal)
    config = uvicorn.Config("app:app", host="0.0.0.0", port=8501, reload=False)
    server = CustomServer(config=config)
    server.run()