/**
 * static/js/app.js
 * Client Controller Javascript untuk Sistem Chatbot Informasi Akademik UNSRAT (RAG)
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Mengelola antarmuka pengguna satu halaman (SPA):
 * 1. Navigasi tab (Chatbot vs Evaluasi Ragas).
 * 2. Pemilihan parameter model LLM & konfigurasi retrieval.
 * 3. Parser streaming real-time Server-Sent Events (SSE) menggunakan Fetch Streams.
 * 4. Pengiriman data audit log skripsi (POST /api/log_transaction) pasca-chat.
 * 5. Visualisasi metrik Ragas interaktif via Chart.js, tabel Wilcoxon, dan Live Audit Log.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Fungsi pembungkus Lucide agar aman dari ReferenceError / crash jika gagal load CDN
    function safeCreateIcons() {
        try {
            if (typeof lucide !== "undefined" && lucide.createIcons) {
                lucide.createIcons();
            } else {
                console.warn("Pustaka Lucide tidak terdeteksi di window.");
            }
        } catch (e) {
            console.error("Gagal merender ikon Lucide:", e);
        }
    }

    // Inisialisasi ikon Lucide secara aman di awal load
    safeCreateIcons();

    // ── STATE MANAGEMENT CLIENT ──────────────────────────────────────────────
    let chatHistory = [];
    let availableModels = [];
    let activeConfig = "b";
    let evalChartInstance = null;

    // Element Selector Utama
    const tabChatBtn = document.getElementById("tab-chat-btn");
    const tabEvalBtn = document.getElementById("tab-eval-btn");
    const tabChat    = document.getElementById("tab-chat");
    const tabEval    = document.getElementById("tab-eval");
    const modelSelect = document.getElementById("model-select");
    const configRadios = document.getElementsByName("config");
    const badgeConfig = document.getElementById("badge-config-display");
    const resetBtn = document.getElementById("reset-btn");
    const refreshEvalBtn = document.getElementById("refresh-eval-btn");

    // Form Chat & Input
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatMessages = document.getElementById("chat-messages");

    // ── SPA TAB NAVIGATION ───────────────────────────────────────────────────
    tabChatBtn.addEventListener("click", () => {
        // Update styling tombol navigasi
        tabChatBtn.className = "w-full flex items-center space-x-3 px-4 py-3.5 rounded-xl font-medium transition duration-200 tab-btn-active";
        tabEvalBtn.className = "w-full flex items-center space-x-3 px-4 py-3.5 rounded-xl font-medium transition duration-200 tab-btn-inactive hover:bg-white/5 hover:text-white";
        
        // Toggle area konten
        tabChat.classList.remove("hidden");
        tabEval.classList.add("hidden");
    });

    tabEvalBtn.addEventListener("click", () => {
        // Update styling tombol navigasi
        tabEvalBtn.className = "w-full flex items-center space-x-3 px-4 py-3.5 rounded-xl font-medium transition duration-200 tab-btn-active";
        tabChatBtn.className = "w-full flex items-center space-x-3 px-4 py-3.5 rounded-xl font-medium transition duration-200 tab-btn-inactive hover:bg-white/5 hover:text-white";
        
        // Toggle area konten
        tabEval.classList.remove("hidden");
        tabChat.classList.add("hidden");
        
        // Muat data kuantitatif dari API
        loadEvaluationData();
    });

    refreshEvalBtn.addEventListener("click", () => {
        loadEvaluationData();
    });

    // ── LOAD INITIAL SYSTEM CONFIG ───────────────────────────────────────────
    async function loadSystemConfig() {
        try {
            const res = await fetch("/api/config");
            if (!res.ok) throw new Error("Gagal mengambil konfigurasi awal");
            const data = await res.json();
            
            // Populasi model LLM dropdown
            availableModels = data.available_models || [];
            modelSelect.innerHTML = availableModels.map(model => 
                `<option value="${model}" ${model === data.default_model ? 'selected' : ''} class="text-[#2D1A1A] text-xs font-medium">${model}</option>`
            ).join("");
            
            // Set default active config
            activeConfig = data.default_config || "b";
            
            // Pastikan radio button yang sesuai terpilih di sidebar
            configRadios.forEach(radio => {
                if (radio.value === activeConfig) {
                    radio.checked = true;
                }
            });
            
            // Update badge config display di header
            updateConfigBadge();
        } catch (err) {
            console.error("Error loading system config:", err);
        }
    }

    // Update tampilan badge config
    function updateConfigBadge() {
        if (badgeConfig) {
            const configName = activeConfig === "c" ? "BM25 Baseline" : `Config ${activeConfig.toUpperCase()}`;
            badgeConfig.innerText = configName;
        }
    }

    // Event Listener perubahan config radio di sidebar
    configRadios.forEach(radio => {
        radio.addEventListener("change", (e) => {
            activeConfig = e.target.value;
            updateConfigBadge();
            // Clear chat history secara lokal & UI demi idempotensi data per config
            clearChatUI("Konfigurasi diubah menjadi " + activeConfig.toUpperCase());
        });
    });

    // Bersihkan chat UI
    function clearChatUI(messagePrefix = "Riwayat percakapan telah dibersihkan.") {
        chatHistory = [];
        chatMessages.innerHTML = `
            <div class="flex items-start space-x-4 max-w-4xl animate-fade-in">
                <div class="bg-[#7B2D2D] text-white p-3 rounded-xl flex-shrink-0 mt-1 shadow-md flex items-center justify-center">
                    <i data-lucide="sparkles" class="w-5 h-5"></i>
                </div>
                <div class="space-y-2 flex-1">
                    <span class="inline-block bg-[#7B2D2D] text-white px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wider shadow-sm">📂 SYSTEM</span>
                    <div class="bg-white border border-[#EBE7E1] rounded-2xl rounded-tl-none px-5 py-4 shadow-sm text-[#2D1A1A] leading-relaxed text-sm">
                        ${messagePrefix} Konfigurasi ${activeConfig.toUpperCase()} aktif. Silakan ajukan pertanyaan regulasi akademik Anda.
                    </div>
                </div>
            </div>
        `;
        safeCreateIcons();
    }

    resetBtn.addEventListener("click", () => {
        clearChatUI();
    });

    // ── LOCAL CATEGORY DETECTOR ──────────────────────────────────────────────
    /**
     * Deteksi kategori lokal sederhana hanya untuk memberikan label warna-warni 
     * visual pada bubble UI secara real-time berdasarkan kata kunci pertanyaan (D-A2).
     */
    function localDetectCategory(query) {
        const keywords = {
            "academic": ["SKS", "KRS", "IPK", "IPS", "UTS", "UAS", "pasal", "yudisium", "cuti", "skripsi", "wisuda", "do", "drop out", "mata kuliah", "nilai", "ujian", "ijazah", "sidang", "kurikulum", "semester", "beban belajar"],
            "calendar": ["jadwal", "tanggal", "kapan", "deadline", "periode", "kalender", "registrasi", "pengisian", "batas", "libur", "minggu ke", "dimulai"],
            "institution_profile": ["sejarah", "visi", "misi", "rektor", "akreditasi", "bendera", "mars", "hymne", "lambang", "berdiri", "didirikan", "profil"]
        };
        
        const text = query.toLowerCase();
        for (const [cat, words] of Object.entries(keywords)) {
            if (words.some(word => text.includes(word))) {
                return cat;
            }
        }
        return "general";
    }

    // Label format visual untuk kategori
    function getCategoryLabel(category) {
        switch (category.toLowerCase()) {
            case "academic":
                return "📂 REGULASI AKADEMIK";
            case "calendar":
                return "📅 KALENDER AKADEMIK";
            case "institution_profile":
                return "🏛️ PROFIL INSTITUSI";
            default:
                return "📂 INFORMASI UMUM";
        }
    }

    // ── STREAMING CHAT SSE GENERATOR ─────────────────────────────────────────
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        // Kosongkan input field segera
        chatInput.value = "";

        // 1. Tampilkan Bubble Chat User (Maroon Klasik)
        chatMessages.innerHTML += `
            <div class="flex items-start justify-end space-x-4 max-w-4xl ml-auto animate-fade-in">
                <div class="space-y-1.5 flex flex-col items-end">
                    <span class="inline-block bg-gray-200 text-gray-700 px-2.5 py-0.5 rounded-full text-[9px] font-bold tracking-wider">MAHASISWA</span>
                    <div class="bg-gradient-to-r from-[#6B2222] to-[#7B2D2D] text-white rounded-2xl rounded-tr-none px-5 py-3.5 shadow-md leading-relaxed text-sm">
                        ${query}
                    </div>
                </div>
                <div class="bg-gray-100 border border-gray-200 text-gray-500 p-3 rounded-xl flex-shrink-0 mt-1 shadow-sm flex items-center justify-center font-bold text-xs w-11 h-11">
                    U
                </div>
            </div>
        `;
        // Scroll otomatis ke bawah
        chatMessages.scrollTop = chatMessages.scrollHeight;

        const category = localDetectCategory(query);
        const botMessageId = "bot-msg-" + Date.now();

        // 2. Tampilkan Bubble Chat Bot Kosong (Typing Indicator)
        chatMessages.innerHTML += `
            <div class="flex items-start space-x-4 max-w-4xl animate-fade-in" id="container-${botMessageId}">
                <div class="bg-[#7B2D2D] text-white p-3 rounded-xl flex-shrink-0 mt-1 shadow-md flex items-center justify-center w-11 h-11">
                    <i data-lucide="bot" class="w-6 h-6 text-white"></i>
                </div>
                <div class="space-y-2.5 flex-1">
                    <span class="inline-block bg-[#7B2D2D] text-white px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wider shadow-sm uppercase" id="cat-label-${botMessageId}">
                        ${getCategoryLabel(category)}
                    </span>
                    <div class="bg-white border border-[#EBE7E1] rounded-2xl rounded-tl-none px-5 py-4 shadow-sm text-[#2D1A1A] leading-relaxed text-sm parsed-markdown" id="${botMessageId}">
                        <div class="flex items-center space-x-2 text-gray-400 font-medium">
                            <i data-lucide="loader" class="w-4 h-4 animate-spin text-[#7B2D2D]/60"></i>
                            <span>Menghubungkan ke server akademik...</span>
                        </div>
                    </div>
                    <!-- Wadah Dokumen Rujukan -->
                    <div id="sources-${botMessageId}" class="space-y-2 hidden.pt-2 transition-all duration-300"></div>
                </div>
            </div>
        `;
        chatMessages.scrollTop = chatMessages.scrollHeight;
        safeCreateIcons();

        const botBubble = document.getElementById(botMessageId);
        const sourcesContainer = document.getElementById(`sources-${botMessageId}`);
        const categoryBadge = document.getElementById(`cat-label-${botMessageId}`);

        let fullResponseText = "";
        const timeStart = Date.now() / 1000;

        // 3. Panggil API streaming FastAPI via POST Fetch Stream
        const requestPayload = {
            query: query,
            config: activeConfig,
            model: modelSelect.value,
            chat_history: chatHistory
        };

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(requestPayload)
            });

            if (!response.ok) {
                throw new Error(`HTTP Error ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let chunkBuffer = "";

            botBubble.innerHTML = ""; // Bersihkan teks typing/pemuatan

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                // Decode chunk biner ke teks
                chunkBuffer += decoder.decode(value, { stream: true });
                const lines = chunkBuffer.split("\n\n");
                chunkBuffer = lines.pop(); // Simpan baris yang belum utuh di akhir buffer

                for (const line of lines) {
                    if (line.trim().startsWith("data: ")) {
                        const rawData = line.slice(6).trim();
                        try {
                            const parsed = JSON.parse(rawData);

                            // Kasus 1: Token Teks Stream Real-Time
                            if (parsed.token !== undefined) {
                                fullResponseText += parsed.token;
                                // Render markdown menggunakan marked.js dengan fallback rapi
                                botBubble.innerHTML = typeof marked !== "undefined" 
                                    ? marked.parse(fullResponseText) 
                                    : fullResponseText.replace(/\n/g, "<br>");
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            }
                            
                            // Kasus 2: Penutup Stream (Metadata & Sources Referensi)
                            else if (parsed.sources !== undefined) {
                                const sources = parsed.sources;
                                const found = parsed.found;
                                const metaLogging = parsed.meta_logging;
                                const finalResponse = parsed.full_response;

                                // Update label kategori berdasarkan prediksi matang backend
                                if (metaLogging && metaLogging.category) {
                                    categoryBadge.innerHTML = getCategoryLabel(metaLogging.category);
                                }

                                // Render dokumen rujukan di bawah bubble teks sebagai collapsible accordion click dropdown
                                if (sources && sources.length > 0) {
                                    let sourcesHtml = `
                                        <!-- Tombol Collapsible Accordion Rujukan -->
                                        <button id="btn-toggle-sources-${botMessageId}" class="w-full flex items-center justify-between px-4 py-3 bg-[#FAF9F6] border border-[#EBE7E1] hover:bg-[#7B2D2D]/5 hover:border-[#7B2D2D]/20 text-[#7B2D2D] rounded-xl text-xs font-bold transition duration-200 cursor-pointer mt-4 shadow-sm active:scale-[0.99] focus:outline-none">
                                            <div class="flex items-center space-x-2">
                                                <i data-lucide="book-open" class="w-4 h-4 text-[#7B2D2D]/75"></i>
                                                <span>📚 Lihat ${sources.length} Rujukan Dokumen SK / Peraturan</span>
                                            </div>
                                            <i data-lucide="chevron-down" id="icon-toggle-${botMessageId}" class="w-4 h-4 transition-transform duration-200"></i>
                                        </button>
                                        
                                        <!-- Konten rujukan dokumen (awal tersembunyi/hidden) -->
                                        <div id="content-sources-${botMessageId}" class="hidden space-y-2.5 mt-2.5 transition-all duration-300">
                                            <div class="grid grid-cols-1 gap-2.5">
                                    `;
                                    
                                    sources.forEach((src, index) => {
                                        sourcesHtml += `
                                            <div class="bg-[#FAF9F6] border border-[#EBE7E1] hover:border-[#7B2D2D]/20 rounded-xl p-3.5 text-xs transition duration-150 shadow-sm space-y-1.5">
                                                <div class="font-bold text-[#7B2D2D] flex items-center justify-between">
                                                    <span class="truncate max-w-[200px]">Doc [${index + 1}]: ${src.title}</span>
                                                    <span class="bg-[#7B2D2D]/10 text-[#7B2D2D] border border-[#7B2D2D]/15 px-2 py-0.5 rounded text-[8px] uppercase tracking-wider font-extrabold">${src.category}</span>
                                                </div>
                                                <div class="text-[9px] text-gray-400 font-semibold uppercase tracking-wider flex items-center space-x-2">
                                                    <span>Doc ID: ${src.doc_id}</span>
                                                    <span>•</span>
                                                    <span>Bab: ${src.bab || '-'}</span>
                                                    <span>•</span>
                                                    <span>Pasal: ${src.pasal || '-'}</span>
                                                </div>
                                                <div class="text-gray-600 bg-white border border-gray-50 rounded-lg p-2.5 italic pl-3 border-l-2 border-l-[#7B2D2D]/40 font-mono text-[10px] leading-normal">
                                                    "...${src.preview}..."
                                                </div>
                                            </div>
                                        `;
                                    });

                                    sourcesHtml += `
                                            </div>
                                        </div>
                                    `;
                                    sourcesContainer.innerHTML = sourcesHtml;
                                    sourcesContainer.classList.remove("hidden");
                                    safeCreateIcons();

                                    // Daftarkan event toggle collapsible click dropdown
                                    const toggleBtn = document.getElementById(`btn-toggle-sources-${botMessageId}`);
                                    const toggleContent = document.getElementById(`content-sources-${botMessageId}`);
                                    const toggleIcon = document.getElementById(`icon-toggle-${botMessageId}`);
                                    
                                    if (toggleBtn && toggleContent && toggleIcon) {
                                        toggleBtn.addEventListener("click", () => {
                                            const isHidden = toggleContent.classList.contains("hidden");
                                            if (isHidden) {
                                                toggleContent.classList.remove("hidden");
                                                toggleContent.classList.add("animate-fade-in");
                                                toggleIcon.style.transform = "rotate(180deg)";
                                                toggleBtn.classList.add("bg-[#7B2D2D]/5", "border-[#7B2D2D]/20");
                                            } else {
                                                toggleContent.classList.add("hidden");
                                                toggleContent.classList.remove("animate-fade-in");
                                                toggleIcon.style.transform = "rotate(0deg)";
                                                toggleBtn.classList.remove("bg-[#7B2D2D]/5", "border-[#7B2D2D]/20");
                                            }
                                            // Scroll ke bawah agar rujukan yang terbuka terlihat
                                            chatMessages.scrollTop = chatMessages.scrollHeight;
                                        });
                                    }
                                }

                                // Update riwayat chat internal
                                chatHistory.push({ "role": "user", "content": query });
                                chatHistory.push({ "role": "assistant", "content": finalResponse });

                                // Simpan log transaksi kuantitatif (log audit skripsi)
                                if (found && metaLogging) {
                                    const responseTime = (Date.now() / 1000) - timeStart;
                                    fetch("/api/log_transaction", {
                                        method: "POST",
                                        headers: {
                                            "Content-Type": "application/json"
                                        },
                                        body: JSON.stringify({
                                            config_retrieval: activeConfig,
                                            model_llm: modelSelect.value,
                                            user_query: query,
                                            category_detected: metaLogging.category,
                                            chunks_retrieved_count: sources.length,
                                            retrieved_chunk_ids: metaLogging.chunk_ids,
                                            best_similarity_score: parseFloat(metaLogging.best_score) || 0.0,
                                            average_similarity_score: parseFloat(metaLogging.avg_score) || 0.0,
                                            response_time_seconds: parseFloat(responseTime) || 0.0,
                                            response_text: finalResponse,
                                            found_state: true,
                                            prompt_payload_for_token_calc: metaLogging.prompt_payload
                                        })
                                    }).catch(err => console.error("Gagal mencatat log transaksi audit:", err));
                                }
                            }
                        } catch (err) {
                            console.error("Gagal parsing baris stream SSE:", err, line);
                        }
                    }
                }
            }

            // Flush sisa data di chunkBuffer jika masih ada setelah stream ditutup
            if (chunkBuffer.trim()) {
                const lines = chunkBuffer.split("\n\n");
                for (const line of lines) {
                    if (line.trim().startsWith("data: ")) {
                        const rawData = line.slice(6).trim();
                        try {
                            const parsed = JSON.parse(rawData);
                            if (parsed.token !== undefined) {
                                fullResponseText += parsed.token;
                                botBubble.innerHTML = typeof marked !== "undefined" 
                                    ? marked.parse(fullResponseText) 
                                    : fullResponseText.replace(/\n/g, "<br>");
                            }
                        } catch (e) {
                            console.error("Gagal parsing leftover stream buffer:", e);
                        }
                    }
                }
            }

        } catch (err) {
            console.error("Gagal melakukan chat stream:", err);
            botBubble.innerHTML = `
                <div class="flex items-center space-x-2 text-red-600 font-semibold p-2.5 bg-red-50 border border-red-100 rounded-xl text-xs">
                    <i data-lucide="alert-triangle" class="w-4 h-4 flex-shrink-0"></i>
                    <span>Koneksi error: Gagal menghubungi server RAG UNSRAT. Pastikan backend aktif.</span>
                </div>
            `;
            safeCreateIcons();
        }
    });

    // ── LOAD EVALUATION & RAGAS METRICS (TAB 2) ──────────────────────────────
    async function loadEvaluationData() {
        const wilcoxonTable = document.getElementById("wilcoxon-table-body");
        const auditTable = document.getElementById("audit-table-body");

        try {
            const res = await fetch("/api/evaluation");
            if (!res.ok) throw new Error("Gagal mengambil data evaluasi");
            const data = await res.json();

            // 1. Populasi Tabel Wilcoxon Signed-Rank
            if (data.wilcoxon && data.wilcoxon.length > 0) {
                wilcoxonTable.innerHTML = data.wilcoxon.map(row => {
                    const isSig = row["significant_at_0.05"];
                    const sigBadge = isSig 
                        ? `<span class="bg-green-50 border border-green-200 text-green-700 px-2 py-0.5 rounded-full text-[9px] font-extrabold uppercase">Signifikan</span>` 
                        : `<span class="bg-gray-50 border border-gray-200 text-gray-500 px-2 py-0.5 rounded-full text-[9px] font-bold">Tidak Sig. (n.s)</span>`;
                    
                    return `
                        <tr class="hover:bg-[#FAF9F6] transition duration-150">
                            <td class="px-4 py-3.5 font-bold text-gray-800 text-xs">${row.metric}</td>
                            <td class="px-4 py-3.5 font-mono text-[10px] text-gray-500">${row.p_value.toFixed(5)}</td>
                            <td class="px-4 py-3.5">${sigBadge}</td>
                            <td class="px-4 py-3.5 text-right font-extrabold text-[#7B2D2D] text-xs">${row.winner}</td>
                        </tr>
                    `;
                }).join("");
            } else {
                wilcoxonTable.innerHTML = `
                    <tr>
                        <td colspan="4" class="px-4 py-6 text-center text-gray-400 font-medium">
                            <i data-lucide="info" class="w-4 h-4 inline mr-1 text-gray-300"></i>
                            <span>Data Wilcoxon tidak tersedia. Silakan jalankan evaluasi offline via CLI.</span>
                        </td>
                    </tr>
                `;
                safeCreateIcons();
            }

            // 2. Populasi Tabel Audit Log Transaksi Chat (Kunci Data Skripsi Bab IV)
            if (data.recent_transactions && data.recent_transactions.length > 0) {
                // Balik urutan agar transaksi paling baru (indeks terbesar) berada di atas tabel
                const reversedTransactions = [...data.recent_transactions].reverse();
                
                auditTable.innerHTML = reversedTransactions.map(row => {
                    const timestampStr = row.timestamp ? row.timestamp.split(" ")[1] || row.timestamp : "-";
                    const configName = row.config_retrieval ? row.config_retrieval.toUpperCase() : "-";
                    
                    return `
                        <tr class="hover:bg-gray-50/50 transition duration-150">
                            <td class="px-4 py-3 text-gray-400 font-mono text-[10px]">${timestampStr}</td>
                            <td class="px-4 py-3">
                                <span class="bg-[#7B2D2D]/10 text-[#7B2D2D] px-2 py-0.5 rounded text-[8px] font-extrabold border border-[#7B2D2D]/15">${configName}</span>
                            </td>
                            <td class="px-4 py-3 font-mono text-[9px] text-gray-400 truncate max-w-[80px]">${row.model_llm || '-'}</td>
                            <td class="px-4 py-3 font-semibold text-gray-700 truncate max-w-[150px]" title="${row.user_query}">${row.user_query}</td>
                            <td class="px-4 py-3 text-center font-mono font-bold text-gray-600">${row.chunks_retrieved_count}</td>
                            <td class="px-4 py-3 font-mono font-semibold text-[#7B2D2D]">${parseFloat(row.best_similarity_score).toFixed(4)}</td>
                            <td class="px-4 py-3 font-mono text-amber-600">${parseFloat(row.response_time_seconds).toFixed(2)}s</td>
                            <td class="px-4 py-3 text-right font-mono font-bold text-gray-800">${row.estimated_total_tokens || 0}</td>
                        </tr>
                    `;
                }).join("");
            } else {
                auditTable.innerHTML = `
                    <tr>
                        <td colspan="8" class="px-4 py-6 text-center text-gray-400 font-medium">
                            <i data-lucide="info" class="w-4 h-4 inline mr-1 text-gray-300"></i>
                            <span>Belum ada transaksi chat yang terekam di logs/transaksi_chat.csv.</span>
                        </td>
                    </tr>
                `;
                safeCreateIcons();
            }

            // 3. Render Chart.js Dinamis Kuantitatif Ragas
            renderRagasChart(data.configs);

        } catch (err) {
            console.error("Gagal memuat data evaluasi:", err);
            wilcoxonTable.innerHTML = `<tr><td colspan="4" class="px-4 py-4 text-center text-red-500 font-semibold">Error: Gagal memuat data evaluasi.</td></tr>`;
            auditTable.innerHTML = `<tr><td colspan="8" class="px-4 py-4 text-center text-red-500 font-semibold">Error: Gagal memuat log transaksi.</td></tr>`;
        }
    }

    // Fungsi Render Chart.js Dinamis
    function renderRagasChart(configs) {
        const ctx = document.getElementById("evalChart");
        if (!ctx) return;

        const metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"];
        const labels = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"];

        // Ekstrak nilai rata-rata (mean) dari data API
        const dataA = metrics.map(m => configs.a && configs.a[m] ? configs.a[m].mean : 0.0);
        const dataB = metrics.map(m => configs.b && configs.b[m] ? configs.b[m].mean : 0.0);
        const dataC = metrics.map(m => configs.c && configs.c[m] ? configs.c[m].mean : 0.0);

        // Hancurkan instance chart lama untuk mencegah penumpukan canvas (bugs canvas reuse)
        if (evalChartInstance) {
            evalChartInstance.destroy();
        }

        // Terapkan skema warna premium bertema Maroon Klasik
        evalChartInstance = new Chart(ctx.getContext("2d"), {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Config A (RAG 500 char)",
                        data: dataA,
                        backgroundColor: "rgba(123, 45, 45, 0.45)", // Maroon transparan
                        borderColor: "rgb(123, 45, 45)",
                        borderWidth: 1.5,
                        borderRadius: 6,
                        barPercentage: 0.8,
                        categoryPercentage: 0.8
                    },
                    {
                        label: "Config B (RAG 2000 char)",
                        data: dataB,
                        backgroundColor: "rgba(168, 69, 69, 0.9)", // Maroon Solid premium
                        borderColor: "rgb(168, 69, 69)",
                        borderWidth: 1.5,
                        borderRadius: 6,
                        barPercentage: 0.8,
                        categoryPercentage: 0.8
                    },
                    {
                        label: "Config C (BM25 baseline)",
                        data: dataC,
                        backgroundColor: "rgba(212, 160, 160, 0.4)", // Maroon tipis
                        borderColor: "rgb(212, 160, 160)",
                        borderWidth: 1.5,
                        borderRadius: 6,
                        barPercentage: 0.8,
                        categoryPercentage: 0.8
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "top",
                        labels: {
                            color: "#2D1A1A",
                            font: {
                                family: "Inter",
                                size: 10,
                                weight: 600
                            },
                            boxWidth: 12,
                            padding: 12
                        }
                    },
                    tooltip: {
                        backgroundColor: "#2D1A1A",
                        titleColor: "#FFFFFF",
                        bodyColor: "#FFFFFF",
                        titleFont: { family: "Inter", weight: 700, size: 11 },
                        bodyFont: { family: "Inter", size: 11 },
                        padding: 10,
                        cornerRadius: 8,
                        displayColors: true
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: "#2D1A1A",
                            font: { family: "Inter", size: 10, weight: 500 }
                        },
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        min: 0.0,
                        max: 1.0,
                        ticks: {
                            color: "#2D1A1A",
                            font: { family: "Inter", size: 10 },
                            stepSize: 0.2
                        },
                        grid: {
                            color: "#FAF9F6",
                            borderColor: "#FAF9F6"
                        }
                    }
                }
            }
        });
    }

    // ── INITIALIZATION ───────────────────────────────────────────────────────
    loadSystemConfig();
});
