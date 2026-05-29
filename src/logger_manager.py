"""
src/logger_manager.py
Manajer Logging Terpusat untuk sistem RAG UNSRAT.
Mengelola penulisan logs/unsrat_rag.log (teks terstruktur) serta
logs/transaksi_chat.csv dan logs/ingestion_report.csv (skripsi data).
Memiliki tanggung jawab tunggal (NFR-02) dan tanpa magic numbers.
"""

import logging
import os
import time
from pathlib import Path
import pandas as pd
import tiktoken

from src.config import (
    LOGS_DIR,
    SYSTEM_LOG_PATH,
    CHAT_LOG_PATH,
    INGESTION_LOG_PATH,
)

# Inisialisasi folder logs secara aman
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Konfigurasi standard logging Python ke file log teks
_file_handler = logging.FileHandler(SYSTEM_LOG_PATH, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

logger = logging.getLogger("unsrat_rag")
logger.setLevel(logging.INFO)
# Cegah duplikasi handler jika modul di-reload
if not logger.handlers:
    logger.addHandler(_file_handler)


def log_system_event(level: str, category: str, message: str) -> None:
    """
    Tulis log teks terstruktur ke berkas logs/unsrat_rag.log.

    Args:
        level:    "info", "warning", atau "error".
        category: Tag kategori (misal: "LLM_INIT", "API_RETRY", "EVALUATION").
        message:  Pesan detail log.
    """
    msg = f"[{category.upper()}] {message}"
    lvl = level.lower()
    if lvl == "warning":
        logger.warning(msg)
    elif lvl == "error":
        logger.error(msg)
    else:
        logger.info(msg)


def estimate_tokens(text) -> int:
    """
    Hitung estimasi jumlah token dari teks secara offline menggunakan tiktoken.
    Menggunakan encoding 'cl100k_base' (standard industri model GPT/Gemini).
    Sangat aman terhadap masukan bertipe list, None, atau objek non-string lainnya.

    Args:
        text: Teks, list konten, atau objek yang akan diestimasi.

    Returns:
        Jumlah token hasil estimasi (integer).
    """
    if text is None:
        return 0
        
    if isinstance(text, list):
        parts = []
        for part in text:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            else:
                parts.append(str(part))
        text = "".join(parts)
    elif not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return 0

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback kasar jika tiktoken gagal memuat encoding
        return len(text.split())


def log_chat_transaction(
    config_retrieval: str,
    model_llm: str,
    user_query: str,
    category_detected: str,
    chunks_retrieved_count: int,
    retrieved_chunk_ids: list[str],
    best_similarity_score: float,
    average_similarity_score: float,
    response_time_seconds: float,
    response_text: str,
    found_state: bool,
    prompt_payload_for_token_calc: str = "",
) -> None:
    """
    Catat transaksi Q&A kuantitatif ke logs/transaksi_chat.csv.
    Aman terhadap I/O concurrency.

    Args:
        config_retrieval: Konfigurasi RAG yang aktif ("a", "b", atau "c").
        model_llm: Nama model generator LLM yang digunakan.
        user_query: Pertanyaan asli dari pengguna.
        category_detected: Kategori pertanyaan yang terdeteksi.
        chunks_retrieved_count: Jumlah chunk dokumen yang ditarik.
        retrieved_chunk_ids: Daftar ID hash MD5 chunk terambil.
        best_similarity_score: Skor cosine distance terkecil (terbaik) terambil.
        average_similarity_score: Rata-rata skor cosine distance chunk terambil.
        response_time_seconds: Latensi total penyelesaian respons (detik).
        response_text: Teks respons jawaban lengkap chatbot.
        found_state: Apakah data relevan ditemukan (Boolean).
        prompt_payload_for_token_calc: Payload teks gabungan prompt untuk hitung token.
    """
    # 1. Hitung estimasi token secara offline
    prompt_tokens = estimate_tokens(prompt_payload_for_token_calc)
    completion_tokens = estimate_tokens(response_text)
    total_tokens = prompt_tokens + completion_tokens

    # 2. Format row data baru
    new_row = {
        "timestamp":                  time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_retrieval":          config_retrieval,
        "model_llm":                 model_llm,
        "user_query":                user_query.replace("\n", " "),
        "category_detected":         category_detected,
        "chunks_retrieved_count":    chunks_retrieved_count,
        "retrieved_chunk_ids":       ",".join(retrieved_chunk_ids),
        "best_similarity_score":     round(best_similarity_score, 4),
        "average_similarity_score":  round(average_similarity_score, 4),
        "response_time_seconds":     round(response_time_seconds, 2),
        "estimated_prompt_tokens":    prompt_tokens,
        "estimated_completion_tokens": completion_tokens,
        "estimated_total_tokens":     total_tokens,
        "found_state":               found_state,
    }

    df_new = pd.DataFrame([new_row])

    # 3. Tulis secara aman (append)
    try:
        if not CHAT_LOG_PATH.exists():
            df_new.to_csv(CHAT_LOG_PATH, index=False, encoding="utf-8-sig")
        else:
            df_new.to_csv(CHAT_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
            
        # 4. Tulis log terstruktur untuk mempermudah debugging obrolan
        clean_query = user_query.replace("\n", " ").strip()
        clean_response = response_text.replace("\n", " ").strip()
        log_system_event(
            "info",
            "CHAT_RESPONSE",
            f"Query: '{clean_query}' | Response: '{clean_response}' | Model: {model_llm} | Config: {config_retrieval}"
        )
    except Exception as e:
        log_system_event("error", "LOGGER_IO", f"Gagal menulis chat CSV: {e}")


def log_ingestion_event(
    config: str,
    file_processed: str,
    chunks_generated: int,
    chunks_inserted: int,
    chunks_duplicate_skipped: int,
    execution_time_seconds: float,
) -> None:
    """
    Catat aktivitas pemrosesan berkas corpus ke logs/ingestion_report.csv.

    Args:
        config: Strategi pemotongan yang digunakan ("a", "b", atau "c").
        file_processed: Nama berkas markdown yang diproses.
        chunks_generated: Jumlah chunk yang dihasilkan oleh text splitter.
        chunks_inserted: Jumlah chunk baru yang masuk ke database.
        chunks_duplicate_skipped: Jumlah chunk yang duplikat dan diabaikan.
        execution_time_seconds: Durasi total pengindeksan file (detik).
    """
    new_row = {
        "timestamp":                 time.strftime("%Y-%m-%d %H:%M:%S"),
        "config":                    config,
        "file_processed":            file_processed,
        "chunks_generated":          chunks_generated,
        "chunks_inserted":           chunks_inserted,
        "chunks_duplicate_skipped":  chunks_duplicate_skipped,
        "execution_time_seconds":     round(execution_time_seconds, 2),
    }

    df_new = pd.DataFrame([new_row])

    try:
        if not INGESTION_LOG_PATH.exists():
            df_new.to_csv(INGESTION_LOG_PATH, index=False, encoding="utf-8-sig")
        else:
            df_new.to_csv(INGESTION_LOG_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
    except Exception as e:
        log_system_event("error", "LOGGER_IO", f"Gagal menulis ingestion CSV: {e}")
