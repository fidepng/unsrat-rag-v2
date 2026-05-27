"""
app.py
Entry point UI Streamlit untuk sistem chatbot RAG UNSRAT.
Dua tab: Chat (RAG real-time) dan Evaluasi (tampilkan hasil eval).
Semua logika bisnis ada di src/ — app.py hanya UI dan pemanggilan fungsi.
"""

import streamlit as st
import pandas as pd
import time
from pathlib import Path
from src.logger_manager import log_chat_transaction


from src.config import (
    AVAILABLE_MODELS,
    LLM_MODEL_NAME,
    EVAL_RESULTS_DIR,
    METRICS_COLS,
    SESSION_MESSAGES,
    SESSION_ACTIVE_CONFIG,
)
from src.chain import get_response, detect_category, reinitialize_llm

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Asisten Akademik UNSRAT",
    page_icon="🎓",
    layout="wide",
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --color-primary:      #7B2D2D;
  --color-primary-alt:  #8B1A2B;
  --color-primary-dark: #5C1F1F;
  --color-primary-light:#A84545;
  --font-main: 'Inter', sans-serif;
}

html, body, [class*="css"] {
  font-family: var(--font-main);
}

.stApp {
  background: linear-gradient(135deg, #1a0a0a 0%, #2d1515 100%);
  color: #f0e6e6;
}

.main-header {
  background: linear-gradient(90deg, var(--color-primary-dark), var(--color-primary));
  padding: 1.5rem 2rem;
  border-radius: 12px;
  margin-bottom: 1.5rem;
  box-shadow: 0 4px 20px rgba(123, 45, 45, 0.4);
}

.main-header h1 {
  color: #fff;
  margin: 0;
  font-size: 1.6rem;
  font-weight: 700;
}

.main-header p {
  color: #f0c0c0;
  margin: 0.25rem 0 0 0;
  font-size: 0.85rem;
}

.category-badge {
  display: inline-block;
  background: var(--color-primary);
  color: white;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 0.72rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
}

.disclaimer {
  background: rgba(123, 45, 45, 0.15);
  border: 1px solid var(--color-primary);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  font-size: 0.78rem;
  color: #d4a0a0;
  margin-top: 1rem;
}

.stButton > button {
  background: var(--color-primary);
  color: white;
  border: none;
  border-radius: 8px;
  font-family: var(--font-main);
  font-weight: 500;
  transition: background 0.2s;
}

.stButton > button:hover {
  background: var(--color-primary-light);
}
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE INIT ────────────────────────────────────────────────────────
def _init_session_state() -> None:
    """Inisialisasi session state keys jika belum ada."""
    if SESSION_MESSAGES not in st.session_state:
        st.session_state[SESSION_MESSAGES] = []
    if SESSION_ACTIVE_CONFIG not in st.session_state:
        st.session_state[SESSION_ACTIVE_CONFIG] = "b"
    if "active_model" not in st.session_state:
        st.session_state["active_model"] = LLM_MODEL_NAME


def _reset_conversation() -> None:
    """Hapus riwayat percakapan dan reset session state."""
    st.session_state[SESSION_MESSAGES] = []


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> None:
    """Render sidebar dengan config switcher, model switcher, dan reset button."""
    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan")
        st.divider()

        # Config switcher
        config_labels = {"a": "Config A — RAG (500 char)", "b": "Config B — RAG (2000 char)", "c": "Config C — BM25"}
        prev_config = st.session_state[SESSION_ACTIVE_CONFIG]
        selected_label = st.radio(
            "Konfigurasi Retrieval",
            options=list(config_labels.values()),
            index=list(config_labels.keys()).index(prev_config),
        )
        new_config = [k for k, v in config_labels.items() if v == selected_label][0]

        if new_config != prev_config:
            st.session_state[SESSION_ACTIVE_CONFIG] = new_config
            _reset_conversation()
            st.rerun()

        st.divider()

        # Model switcher
        prev_model = st.session_state["active_model"]
        selected_model = st.selectbox(
            "Model LLM",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(prev_model) if prev_model in AVAILABLE_MODELS else 0,
        )
        if selected_model != prev_model:
            reinitialize_llm(selected_model)
            st.session_state["active_model"] = selected_model

        st.divider()

        # Reset button
        if st.button("🔄 Reset Percakapan", use_container_width=True):
            _reset_conversation()
            st.rerun()

        st.divider()

        # Info panel
        active_config = st.session_state[SESSION_ACTIVE_CONFIG]
        st.markdown(f"""
        **Config aktif:** `{active_config.upper()}`
        **Model aktif:** `{st.session_state['active_model']}`
        """)


# ── TAB 1: CHAT ───────────────────────────────────────────────────────────────
def _render_chat_tab() -> None:
    """Render tab chat utama dengan streaming response dan sitasi sumber."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Asisten Informasi Akademik UNSRAT</h1>
        <p>Didukung oleh arsitektur RAG + Google Gemini</p>
    </div>
    """, unsafe_allow_html=True)

    messages = st.session_state[SESSION_MESSAGES]
    active_config = st.session_state[SESSION_ACTIVE_CONFIG]

    # Tampilkan riwayat chat
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "category" in msg:
                st.markdown(
                    f'<span class="category-badge">📂 {msg["category"]}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 Lihat Sumber ({len(msg['sources'])} referensi)"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}] {src['title']} — {src['bab']}**")
                        st.text(src["preview"])

    # Input pengguna
    if user_input := st.chat_input("Ketik pertanyaan Anda..."):
        # Tampilkan pesan user
        with st.chat_message("user"):
            st.markdown(user_input)
        messages.append({"role": "user", "content": user_input})

        # Deteksi kategori (label UI saja)
        category = detect_category(user_input)

        # SINGLE API CALL: get_response() mengembalikan stream + sources sekaligus
        # Tidak ada second call — sources dari retrieval (sebelum LLM), bukan dari LLM.
        result = get_response(
            query=user_input,
            config=active_config,
            chat_history=messages[:-1],  # exclude pesan user yang baru saja ditambahkan
            streaming=True,
        )
        sources      = result["sources"]  # Tersedia langsung dari retrieval
        stream_gen   = result["stream"]   # Generator untuk st.write_stream
        found        = result["found"]
        meta_logging = result.get("_meta_logging")

        # Tampilkan streaming response
        with st.chat_message("assistant"):
            st.markdown(
                f'<span class="category-badge">📂 {category}</span>',
                unsafe_allow_html=True,
            )
            full_response = st.write_stream(stream_gen)

            # Tampilkan sitasi sumber langsung setelah streaming selesai
            if sources:
                with st.expander(f"📚 Lihat Sumber ({len(sources)} referensi)"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**[{i}] {src['title']} — {src['bab']}**")
                        st.text(src["preview"])

        # Catat transaksi setelah streaming selesai (menggunakan metadata log dari get_response)
        if found and meta_logging:
            duration = time.time() - meta_logging["t_start"]
            log_chat_transaction(
                config_retrieval=active_config,
                model_llm=st.session_state.get("active_model", meta_logging["active_model"]),
                user_query=user_input,
                category_detected=meta_logging["category"],
                chunks_retrieved_count=len(sources),
                retrieved_chunk_ids=meta_logging["chunk_ids"],
                best_similarity_score=meta_logging["best_score"],
                average_similarity_score=meta_logging["avg_score"],
                response_time_seconds=duration,
                response_text=full_response,
                found_state=True,
                prompt_payload_for_token_calc=meta_logging["prompt_payload"],
            )

        # Simpan ke session state
        messages.append({
            "role":     "assistant",
            "content":  full_response,
            "category": category,
            "sources":  sources,
        })

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> Sistem ini adalah prototipe penelitian.
        Informasi yang disajikan bersumber dari dokumen yang tersedia dan dapat
        memiliki keterbatasan. Untuk keputusan akademik penting, selalu verifikasi
        dengan pihak Akademik UNSRAT secara langsung.
    </div>
    """, unsafe_allow_html=True)


# ── TAB 2: EVALUASI ───────────────────────────────────────────────────────────
def _render_evaluation_tab() -> None:
    """Render tab hasil evaluasi Ragas (read-only dari eval/results/)."""
    st.markdown("## 📊 Hasil Evaluasi Ragas")
    st.markdown(
        "Halaman ini menampilkan hasil evaluasi yang sudah dijalankan via CLI. "
        "Untuk menjalankan evaluasi baru, gunakan terminal."
    )

    result_files = {
        "a": EVAL_RESULTS_DIR / "hasil_config_a.csv",
        "b": EVAL_RESULTS_DIR / "hasil_config_b.csv",
        "c": EVAL_RESULTS_DIR / "hasil_config_c.csv",
    }
    stats_file  = EVAL_RESULTS_DIR / "statistical_test.csv"
    chart_file  = EVAL_RESULTS_DIR / "perbandingan_visual.png"

    existing = {k: v for k, v in result_files.items() if v.exists()}

    if not existing:
        st.info(
            "📂 Belum ada hasil evaluasi.\n\n"
            "Jalankan evaluasi via terminal:\n"
            "```\n"
            "python evaluation.py --config a\n"
            "python evaluation.py --config b\n"
            "python evaluation.py --config c\n"
            "python evaluation.py --stats\n"
            "python evaluation.py --visualize\n"
            "```"
        )
        return

    # Tampilkan tabel per config
    for config_key, filepath in existing.items():
        df = pd.read_csv(filepath)
        st.markdown(f"### Config {config_key.upper()}: `{filepath.name}`")
        # Tampilkan aggregasi metrik
        agg_data = {}
        for metric in METRICS_COLS:
            if metric in df.columns:
                agg_data[metric] = f"{df[metric].mean():.3f} ± {df[metric].std():.3f}"
        if "response_time_seconds" in df.columns:
            agg_data["response_time"] = (
                f"{df['response_time_seconds'].mean():.2f} ± "
                f"{df['response_time_seconds'].std():.2f} detik"
            )
        if agg_data:
            st.table(pd.DataFrame(agg_data, index=["mean ± std"]).T)

    # Tampilkan tabel Wilcoxon jika ada
    if stats_file.exists():
        st.markdown("### Uji Signifikansi Wilcoxon (A vs B)")
        stats_df = pd.read_csv(stats_file)
        st.dataframe(stats_df)

    # Tampilkan chart jika ada
    if chart_file.exists():
        st.markdown("### Visualisasi Perbandingan")
        st.image(str(chart_file), use_container_width=True)
    else:
        st.info("Chart belum tersedia. Jalankan: `python evaluation.py --visualize`")

    # CLI reference
    st.markdown("---")
    st.markdown("**Perintah CLI evaluasi:**")
    st.code(
        "python evaluation.py --config a   # Evaluasi Config A\n"
        "python evaluation.py --config b   # Evaluasi Config B\n"
        "python evaluation.py --config c   # Evaluasi Config C (BM25)\n"
        "python evaluation.py --stats      # Uji Wilcoxon A vs B\n"
        "python evaluation.py --visualize  # Generate bar chart",
        language="bash",
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    """Entry point utama aplikasi Streamlit."""
    _init_session_state()
    _render_sidebar()

    tab_chat, tab_eval = st.tabs(["💬 Chatbot", "📊 Evaluasi Ragas"])

    with tab_chat:
        _render_chat_tab()

    with tab_eval:
        _render_evaluation_tab()


if __name__ == "__main__":
    main()
