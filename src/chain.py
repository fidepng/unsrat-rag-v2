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
from src.logger_manager import log_system_event, log_chat_transaction


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
    log_system_event("info", "LLM_INIT", f"Model diinisialisasi ulang: {model_name}")


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

    t_start = time.time()  # 1. Mulai stopwatch

    # Retrieval
    chunks = retrieve(query, config)

    # Dapatkan status similarity score
    best_score = 0.0
    avg_score  = 0.0
    if chunks:
        scores = [c["score"] for c in chunks]
        best_score = min(scores)  # Cosine distance: semakin kecil semakin relevan (terbaik)
        avg_score  = sum(scores) / len(scores)

    # 2. Fallback jika tidak ada chunk yang relevan
    if not chunks:
        duration = time.time() - t_start
        if streaming:
            def _fallback_gen():
                yield FALLBACK_RESPONSE
            # Log fallback transaction ke CSV
            log_chat_transaction(
                config_retrieval=config,
                model_llm=LLM_MODEL_NAME,
                user_query=query,
                category_detected=detect_category(query),
                chunks_retrieved_count=0,
                retrieved_chunk_ids=[],
                best_similarity_score=1.0,  # Nilai distance terburuk
                average_similarity_score=1.0,
                response_time_seconds=duration,
                response_text=FALLBACK_RESPONSE,
                found_state=False,
                prompt_payload_for_token_calc="",
            )
            return {"answer": None, "stream": _fallback_gen(), "sources": [], "found": False}

        log_chat_transaction(
            config_retrieval=config,
            model_llm=LLM_MODEL_NAME,
            user_query=query,
            category_detected=detect_category(query),
            chunks_retrieved_count=0,
            retrieved_chunk_ids=[],
            best_similarity_score=1.0,
            average_similarity_score=1.0,
            response_time_seconds=duration,
            response_text=FALLBACK_RESPONSE,
            found_state=False,
            prompt_payload_for_token_calc="",
        )
        return {"answer": FALLBACK_RESPONSE, "stream": None, "sources": [], "found": False}

    # 3. Petakan chunk ke dalam objek sources dan kumpulkan ID
    sources = []
    chunk_ids = []
    for chunk in chunks:
        meta = chunk["metadata"]
        raw_id = f"{meta.get('doc_id', '')}::{chunk['content']}"
        md5_id = __import__("hashlib").md5(raw_id.encode("utf-8")).hexdigest()
        chunk_ids.append(md5_id)

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

    # Payload teks kasar untuk estimasi hitung token offline
    prompt_payload = f"{SYSTEM_PROMPT}\n{history}\n{context}\n{query}"

    llm = _get_llm()
    active_model_name = llm.model_name if hasattr(llm, "model_name") else LLM_MODEL_NAME

    # 5. Panggil LLM — SATU KALI untuk kedua mode
    if streaming:
        return {
            "answer":  None,
            "stream":  _stream_with_retry(llm, messages),
            "sources": sources,
            "found":   True,
            "_meta_logging": {
                "t_start":                  t_start,
                "best_score":               best_score,
                "avg_score":                avg_score,
                "chunk_ids":                chunk_ids,
                "prompt_payload":           prompt_payload,
                "active_model":             active_model_name,
                "category":                 detect_category(query),
            }
        }
    else:
        answer = _invoke_with_retry(llm, messages)
        duration = time.time() - t_start

        # Log transaksi non-streaming ke CSV
        log_chat_transaction(
            config_retrieval=config,
            model_llm=active_model_name,
            user_query=query,
            category_detected=detect_category(query),
            chunks_retrieved_count=len(chunks),
            retrieved_chunk_ids=chunk_ids,
            best_similarity_score=best_score,
            average_similarity_score=avg_score,
            response_time_seconds=duration,
            response_text=answer,
            found_state=True,
            prompt_payload_for_token_calc=prompt_payload,
        )

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
            if attempt > 0:
                log_system_event("info", "API_RETRY", f"Percobaan ke-{attempt + 1} sukses.")
            
            content = response.content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        parts.append(part["text"])
                    else:
                        parts.append(str(part))
                content = "".join(parts)
            return content
        except Exception as e:
            last_error = e
            msg = f"LLM invoke gagal (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            logger.warning(msg)
            log_system_event("warning", "API_LIMIT", f"Kendala API pada perc. ke-{attempt + 1}. Error: {e}")
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])

    log_system_event("error", "API_FAIL", f"Gagal total pemanggilan LLM setelah {MAX_RETRIES} perc. Error: {last_error}")
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
            if attempt > 0:
                log_system_event("info", "API_RETRY", f"Percobaan stream ke-{attempt + 1} sukses.")
            return
        except Exception as e:
            last_error = e
            msg = f"LLM stream gagal (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            logger.warning(msg)
            log_system_event("warning", "API_LIMIT", f"Kendala API stream pada perc. ke-{attempt + 1}. Error: {e}")
            if attempt < len(RETRY_DELAYS):
                time.sleep(RETRY_DELAYS[attempt])

    log_system_event("error", "API_FAIL", f"Gagal total stream LLM setelah {MAX_RETRIES} perc. Error: {last_error}")
    yield (
        f"\n\nMaaf, terjadi kesalahan saat menghubungi layanan AI "
        f"setelah {MAX_RETRIES} percobaan. Silakan coba lagi."
    )
