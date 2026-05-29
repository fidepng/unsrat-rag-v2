"""
evaluation.py
Pipeline evaluasi Ragas 0.4.x untuk sistem RAG UNSRAT.
Mengukur faithfulness, answer_relevancy, context_precision, context_recall.
Juga mengukur latensi (response_time_seconds) per query.

CLI:
  python evaluation.py --config [a|b|c]   Evaluasi satu config
  python evaluation.py --stats            Uji Wilcoxon A vs B
  python evaluation.py --visualize        Generate bar chart
"""

# --- MONKEY PATCH UNTUK BYPASS VERTEXAI IMPORT DI RAGAS 0.4.x ---
import sys
import types
m = types.ModuleType("langchain_community.chat_models.vertexai")
m.ChatVertexAI = None
sys.modules["langchain_community.chat_models.vertexai"] = m
# ---------------------------------------------------------------

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend untuk server/CLI

from scipy.stats import wilcoxon

from src.config import (
    EVAL_DATASET_PATH,
    EVAL_RESULTS_DIR,
    METRICS_COLS,
    ERROR_ANALYSIS_N,
    EVALUATOR_MODEL_NAME,
    GOOGLE_API_KEY,
)
from src.chain import get_response
from src.logger_manager import log_system_event


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_ground_truth() -> pd.DataFrame:
    """
    Muat dan validasi file ground_truth.csv.
    Raises FileNotFoundError atau ValueError dengan pesan Bahasa Indonesia.
    """
    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"File ground truth tidak ditemukan: {EVAL_DATASET_PATH}\n"
            "Buat file ground_truth.csv di eval/dataset/ dengan kolom: "
            "user_input, reference, category, source_doc"
        )

    df = pd.read_csv(EVAL_DATASET_PATH, encoding="utf-8-sig")

    required_cols = ["user_input", "reference", "category", "source_doc"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan di {EVAL_DATASET_PATH.name}: {missing}\n"
            f"Format yang benar: {required_cols}"
        )

    # Batasi maksimal 15 kueri (sesuai batasan kuota API dan instruksi pengguna)
    if len(df) > 15:
        logger.info(f"Membatasi dataset ground truth dari {len(df)} menjadi 15 kueri teratas.")
        df = df.head(15)

    logger.info(f"Ground truth dimuat: {len(df)} pasang Q&A")
    return df


def _run_evaluation(df: pd.DataFrame, config: str) -> pd.DataFrame:
    """
    Jalankan pipeline RAG untuk setiap pertanyaan dan kumpulkan hasil.
    Setiap pertanyaan dievaluasi INDEPENDEN (chat_history di-reset per pertanyaan).
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from src.config import EMBEDDING_MODEL_NAME

    logger.info(f"Memulai evaluasi Config {config.upper()} — {len(df)} pertanyaan...")

    rows = []
    response_times = []

    for idx, row in df.iterrows():
        logger.info(f"  [{idx + 1}/{len(df)}] {row['user_input'][:60]}...")

        chat_history = []  # WAJIB reset per pertanyaan (FR-14)

        t_start = time.time()
        try:
            result = get_response(
                query=row["user_input"],
                config=config,
                chat_history=chat_history,
                streaming=False,
            )
            response_time = time.time() - t_start

            answer = result["answer"]
            # KRITIS: gunakan s["content"] (full chunk), BUKAN s["preview"] (150 char)
            # Menggunakan preview akan merusak validitas context_precision & context_recall
            contexts = [s["content"] for s in result["sources"]]

        except Exception as e:
            logger.warning(f"  ERROR pada pertanyaan {idx}: {e}")
            response_time = time.time() - t_start
            answer = ""
            contexts = []

        rows.append({
            "user_input":         row["user_input"],
            "reference":          row["reference"],
            "response":           answer,
            "retrieved_contexts": contexts,
        })
        response_times.append(response_time)

    # Siapkan EvaluationDataset Ragas 0.4.x
    dataset = EvaluationDataset.from_list(rows)

    # Load NVIDIA API Key secara aman dari environment (.env)
    import os
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise ValueError(
            "NVIDIA_API_KEY tidak ditemukan di file .env!\n"
            "Silakan tambahkan: NVIDIA_API_KEY=your_nvapi_key di berkas .env Anda."
        )

    # Gunakan ChatNVIDIA (Llama 3.1 70B atau Qwen) dari NVIDIA NIM sebagai evaluator Ragas
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    
    logger.info(f"Menggunakan NVIDIA NIM {EVALUATOR_MODEL_NAME} sebagai Evaluator Ragas...")
    evaluator_llm = ChatNVIDIA(
        model=EVALUATOR_MODEL_NAME,
        api_key=nvidia_api_key,
        temperature=0,  # Nol demi stabilitas dan reproduktibilitas hasil Ragas
    )
    
    # Gunakan Google embedding model yang sama
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        task_type="retrieval_document",
        google_api_key=GOOGLE_API_KEY,
    )

    logger.info("Menjalankan ragas.evaluate()...")

    # Bungkus model LLM dan Embeddings untuk Ragas 0.4.x
    wrapped_llm = LangchainLLMWrapper(evaluator_llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(evaluator_embeddings)

    # Inisialisasi masing-masing objek metrik secara eksplisit dengan llm
    metric_objects = {
        "faithfulness":      Faithfulness(llm=wrapped_llm),
        "answer_relevancy":  AnswerRelevancy(llm=wrapped_llm),
        "context_precision": ContextPrecision(llm=wrapped_llm),
        "context_recall":    ContextRecall(llm=wrapped_llm),
    }
    selected_metrics = [metric_objects[m] for m in METRICS_COLS if m in metric_objects]

    # Konfigurasi run settings untuk memitigasi kendala API rate limit (TPM/RPM)
    from ragas.run_config import RunConfig
    run_config = RunConfig(
        timeout=300,        # Batas waktu maksimal operasi per kueri (detik)
        max_retries=10,     # Jumlah percobaan ulang saat terkena limit
        max_wait=60,        # Jeda waktu tunggu antar percobaan ulang
        max_workers=1,      # Jalankan secara berurutan (sequential) untuk menghindari 429
        log_tenacity=True,  # Catat log tenacity saat melakukan retrying
    )

    results = evaluate(
        dataset=dataset,
        metrics=selected_metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    results_df = results.to_pandas()

    # Tambahkan kolom tambahan
    results_df["response_time_seconds"] = response_times
    results_df["category"]  = df["category"].values
    results_df["source_doc"] = df["source_doc"].values

    return results_df


def _print_aggregation(results_df: pd.DataFrame, config: str) -> None:
    """Cetak ringkasan agregasi ke terminal."""
    print(f"\n{'='*60}")
    print(f"HASIL EVALUASI CONFIG {config.upper()}")
    print(f"{'='*60}")
    for metric in METRICS_COLS:
        if metric in results_df.columns:
            mean = results_df[metric].mean()
            std  = results_df[metric].std()
            print(f"  {metric:<25}: {mean:.3f} ± {std:.3f}")
    if "response_time_seconds" in results_df.columns:
        rt_mean = results_df["response_time_seconds"].mean()
        rt_std  = results_df["response_time_seconds"].std()
        print(f"  {'response_time (detik)':<25}: {rt_mean:.2f} ± {rt_std:.2f}")
    print(f"{'='*60}\n")


def _export_error_analysis(results_df: pd.DataFrame, config: str) -> None:
    """
    Ekspor ERROR_ANALYSIS_N sampel dengan rata-rata skor metrik terendah.
    Kolom failure_type dan failure_notes diisi manual oleh peneliti.
    """
    if config == "c":
        logger.info("Error analysis Config C dilewati (opsional, analisis terpisah).")
        return

    available_metrics = [m for m in METRICS_COLS if m in results_df.columns]
    if not available_metrics:
        logger.warning("Tidak ada kolom metrik untuk error analysis.")
        return

    results_df = results_df.copy()
    results_df["avg_metric_score"] = results_df[available_metrics].mean(axis=1)

    worst = results_df.nsmallest(ERROR_ANALYSIS_N, "avg_metric_score").copy()
    worst.insert(0, "rank", range(1, len(worst) + 1))
    worst["failure_type"]  = ""   # diisi manual oleh peneliti
    worst["failure_notes"] = ""   # diisi manual oleh peneliti

    output_cols = [
        "rank", "user_input", "reference", "response",
        "avg_metric_score", "failure_type", "failure_notes",
    ]
    output_cols = [c for c in output_cols if c in worst.columns]

    out_path = EVAL_RESULTS_DIR / f"error_analysis_config_{config}.csv"
    worst[output_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Error analysis disimpan ke: {out_path}")
    logger.info(
        "PENTING: Isi kolom 'failure_type' secara manual dengan salah satu:\n"
        "  retrieval_failure | generation_failure | chunking_failure"
    )


def run_config_evaluation(config: str) -> None:
    """Jalankan evaluasi lengkap untuk satu konfigurasi."""
    df = _load_ground_truth()
    log_system_event("info", "EVALUATION", f"Memulai pengujian Ragas Config {config.upper()} dengan {len(df)} kueri.")
    results_df = _run_evaluation(df, config)

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_RESULTS_DIR / f"hasil_config_{config}.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Hasil disimpan ke: {out_path}")

    _print_aggregation(results_df, config)
    _export_error_analysis(results_df, config)
    log_system_event("info", "EVALUATION", f"Selesai pengujian Ragas Config {config.upper()}. Aggregasi tercetak.")


def run_wilcoxon_stats() -> None:
    """Jalankan uji Wilcoxon Signed-Rank antara Config A dan Config B."""
    path_a = EVAL_RESULTS_DIR / "hasil_config_a.csv"
    path_b = EVAL_RESULTS_DIR / "hasil_config_b.csv"

    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(
            "File hasil Config A dan/atau Config B tidak ditemukan!\n"
            "Jalankan terlebih dahulu:\n"
            "  python evaluation.py --config a\n"
            "  python evaluation.py --config b"
        )

    df_a = pd.read_csv(path_a, encoding="utf-8-sig")
    df_b = pd.read_csv(path_b, encoding="utf-8-sig")

    if len(df_a) != len(df_b):
        raise ValueError(
            f"Jumlah baris Config A ({len(df_a)}) dan Config B ({len(df_b)}) berbeda. "
            "Pastikan kedua evaluasi menggunakan ground truth yang sama."
        )

    log_system_event("info", "EVALUATION", "Memulai perhitungan uji signifikansi Wilcoxon A vs B.")
    stats_rows = []
    for metric in METRICS_COLS:
        if metric not in df_a.columns or metric not in df_b.columns:
            continue
        scores_a = df_a[metric].fillna(0).values
        scores_b = df_b[metric].fillna(0).values
        try:
            stat, p_val = wilcoxon(scores_a, scores_b)
            significant = bool(p_val < 0.05)
            winner = "Config B" if df_b[metric].mean() > df_a[metric].mean() else "Config A"
            if not significant:
                winner = "n.s."
        except Exception as e:
            stat, p_val, significant, winner = None, None, None, f"Error: {e}"

        stats_rows.append({
            "metric":             metric,
            "wilcoxon_statistic": stat,
            "p_value":            p_val,
            "significant_at_0.05": significant,
            "winner":             winner,
        })
        print(f"  {metric}: p={p_val:.4f} {'*' if significant else 'n.s.'} | winner: {winner}")

    stats_df = pd.DataFrame(stats_rows)
    out_path = EVAL_RESULTS_DIR / "statistical_test.csv"
    stats_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Hasil Wilcoxon disimpan ke: {out_path}")
    log_system_event("info", "EVALUATION", "Uji Wilcoxon Signed-Rank sukses disimpan di statistical_test.csv")


def run_visualize() -> None:
    """
    Generate bar chart perbandingan 3 config + p-value annotation + latency line.
    Output: eval/results/perbandingan_visual.png
    """
    paths = {
        "Config A": EVAL_RESULTS_DIR / "hasil_config_a.csv",
        "Config B": EVAL_RESULTS_DIR / "hasil_config_b.csv",
        "Config C (BM25)": EVAL_RESULTS_DIR / "hasil_config_c.csv",
    }

    missing = [k for k, v in paths.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(
            f"File hasil berikut tidak ditemukan: {missing}\n"
            "Jalankan evaluasi terlebih dahulu untuk semua config."
        )

    dfs = {label: pd.read_csv(path, encoding="utf-8-sig") for label, path in paths.items()}

    # Baca p-value dari Wilcoxon jika ada
    stats_path = EVAL_RESULTS_DIR / "statistical_test.csv"
    pvalues: dict[str, str] = {}
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        for _, row in stats_df.iterrows():
            sig = "*" if row.get("significant_at_0.05") else "n.s."
            pvalues[row["metric"]] = f"p={row['p_value']:.3f}{sig}"

    available_metrics = [m for m in METRICS_COLS if all(m in df.columns for df in dfs.values())]
    n_metrics = len(available_metrics)
    configs   = list(dfs.keys())
    n_configs = len(configs)

    x         = range(n_metrics)
    bar_width = 0.22
    colors    = ["#7B2D2D", "#A84545", "#D4A0A0"]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a0a0a")
    ax1.set_facecolor("#2d1515")

    for i, (label, df) in enumerate(dfs.items()):
        means = [df[m].mean() for m in available_metrics]
        offset = (i - n_configs / 2 + 0.5) * bar_width
        ax1.bar(
            [xi + offset for xi in x],
            means,
            width=bar_width,
            label=label,
            color=colors[i],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

    # Anotasi p-value di atas bar Config A & B
    for i, metric in enumerate(available_metrics):
        if metric in pvalues:
            ax1.text(
                i,
                max(dfs[label][metric].mean() for label in dfs) + 0.02,
                pvalues[metric],
                ha="center",
                fontsize=8,
                color="#f0c0c0",
            )

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(available_metrics, rotation=15, ha="right", color="#f0e6e6")
    ax1.set_ylabel("Skor Rata-rata", color="#f0e6e6")
    ax1.set_ylim(0, 1.15)
    ax1.tick_params(colors="#f0e6e6")
    ax1.spines["bottom"].set_color("#7B2D2D")
    ax1.spines["left"].set_color("#7B2D2D")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Sumbu kedua: latency line
    if all("response_time_seconds" in df.columns for df in dfs.values()):
        ax2 = ax1.twinx()
        latencies = [dfs[label]["response_time_seconds"].mean() for label in configs]
        ax2.plot(
            [i * bar_width * n_configs + bar_width * (n_configs / 2)
             for i in range(n_configs)],
            latencies,
            "o--",
            color="#FFD700",
            linewidth=1.5,
            markersize=6,
            label="Latensi (detik)",
        )
        ax2.set_ylabel("Latensi Rata-rata (detik)", color="#FFD700")
        ax2.tick_params(colors="#FFD700")
        ax2.spines["right"].set_color("#FFD700")

    ax1.set_title(
        "Perbandingan Performa Config A vs B vs C (BM25)\nSistem RAG Akademik UNSRAT",
        color="#f0e6e6",
        fontsize=13,
        pad=15,
    )
    ax1.legend(loc="upper right", framealpha=0.3, labelcolor="#f0e6e6")

    plt.tight_layout()
    out_path = EVAL_RESULTS_DIR / "perbandingan_visual.png"
    log_system_event("info", "EVALUATION", "Memulai pembangunan grafik visualisasi performa skripsi (Matplotlib).")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Chart disimpan ke: {out_path}")
    log_system_event("info", "EVALUATION", "Visualisasi diagram berhasil disimpan di perbandingan_visual.png")


def main():
    """Entry point CLI untuk pipeline evaluasi."""
    parser = argparse.ArgumentParser(
        description="Pipeline evaluasi Ragas untuk UNSRAT RAG"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        choices=["a", "b", "c"],
        help="Evaluasi satu konfigurasi (a=RAG 500, b=RAG 2000, c=BM25)",
    )
    group.add_argument(
        "--stats",
        action="store_true",
        help="Uji Wilcoxon A vs B (jalankan setelah --config a dan --config b)",
    )
    group.add_argument(
        "--visualize",
        action="store_true",
        help="Generate bar chart (jalankan setelah semua --config selesai)",
    )
    args = parser.parse_args()

    if args.config:
        run_config_evaluation(args.config)
    elif args.stats:
        run_wilcoxon_stats()
    elif args.visualize:
        run_visualize()


if __name__ == "__main__":
    main()
