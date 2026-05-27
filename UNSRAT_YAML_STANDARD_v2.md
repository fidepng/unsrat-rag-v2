# UNSRAT RAG — Standar YAML Frontmatter v2.0

# Penulis: [Anda] | Revisi dari: pedoman_yaml.txt

---

# ============================================================

# BLOK 1 — IDENTITAS DOKUMEN

# Wajib: semua field. Null jika tidak berlaku, BUKAN "".

# ============================================================

doc_id: "UNSRAT-REG-2025-001"

# Format: UNSRAT-{TYPE}-{YYYY}-{NNN} ← semua UPPERCASE, konsisten

# TYPE enum (pilih SATU, selalu Inggris):

# REG → Peraturan, SK, Keputusan Rektor/Dekan

# CAL → Kalender akademik, jadwal resmi

# PROFILE → Sejarah, profil institusi

# GUIDE → Panduan, SOP, petunjuk teknis

# ANNOUNCE → Pengumuman, edaran

# CURRIC → Kurikulum, silabus, RPS

# FORM → Formulir, template

# Contoh: UNSRAT-REG-2025-001, UNSRAT-CAL-2026-001

title: ""

# Judul lengkap dokumen dalam Bahasa Indonesia,

# sama persis dengan judul di konten asli.

version: "1.0"

# Versi dokumen. Gunakan semantic versioning sederhana.

# "1.0" untuk dokumen baru. "1.1", "2.0" untuk revisi.

language_primary: "id"

# ISO 639-1 bahasa utama dokumen.

language_secondary: null

# ISO 639-1 jika ada bagian substansial dalam bahasa lain.

# null jika tidak ada.

institution: "Universitas Sam Ratulangi"

# Selalu nama lengkap ini, tidak pernah singkatan.

unit_penerbit: ""

# Unit yang menerbitkan. Contoh: "Rektorat", "Fakultas Teknik"

# ============================================================

# BLOK 2 — KLASIFIKASI KONTEN

# ============================================================

content_type: ""

# Enum (pilih SATU, Inggris):

# "regulation" → Peraturan, SK, Keputusan resmi

# "calendar" → Kalender akademik, jadwal resmi

# "narrative" → Sejarah, profil, artikel deskriptif

# "guide" → Panduan, SOP, petunjuk teknis

# "announcement" → Pengumuman, edaran

# "curriculum" → Kurikulum, silabus, RPS

# "form" → Formulir, template

category: ""

# Enum (pilih SATU, Inggris):

# "academic", "student_affairs", "finance",

# "institution_profile", "admission", "research"

subcategory:

- ""

# LIST — dokumen bisa mencakup lebih dari satu sub-topik.

# Contoh: ["perkuliahan", "evaluasi", "mbkm", "wisuda"]

# null jika tidak relevan.

audience:

- ""

# LIST. Enum:

# "mahasiswa_s1", "mahasiswa_s2", "mahasiswa_s3",

# "mahasiswa_profesi", "dosen", "tendik",

# "pimpinan", "publik"

access_level: "public"

# Enum: "public" | "internal" | "restricted"

# ============================================================

# BLOK 3 — LEGALITAS & SUMBER

# Wajib untuk content_type: regulation & calendar.

# Gunakan null (bukan "") untuk field yang tidak berlaku.

# ============================================================

nomor_sk: null

# String jika ada. null jika bukan SK/regulasi resmi.

# Contoh: "01/2025" atau "2079/UN12/PP/2025"

tanggal_penetapan: null

# ISO 8601: "YYYY-MM-DD". null jika tidak ada.

pejabat_penandatangan: null

# Nama lengkap + jabatan, atau null.

# Contoh: "Oktovian Berty Alexander Sompie, Rektor UNSRAT"

source_document: ""

# Nama file PDF sumber asli.

# Contoh: "Peraturan-Rektor-Unsrat-Nomor-1-tahun-2025.pdf"

source_url: null

# URL resmi jika tersedia. null jika tidak ada.

supersedes: null

# doc_id dokumen yang DIGANTIKAN oleh file ini.

# Harus berupa doc_id valid (UNSRAT-REG-YYYY-NNN) atau null.

# JANGAN gunakan judul bebas — harus machine-resolvable.

# Jika dokumen lama belum di-index, tetap null.

superseded_by: null

# doc_id dokumen yang MENGGANTIKAN file ini.

# Diisi saat dokumen ini tidak berlaku lagi.

# null untuk dokumen aktif.

# ============================================================

# BLOK 4 — VALIDITAS TEMPORAL

# ============================================================

valid_from: ""

# ISO 8601: "YYYY-MM-DD". WAJIB diisi.

# Untuk narasi/profil: tanggal publikasi atau last_updated.

valid_until: null

# ISO 8601: "YYYY-MM-DD", atau null jika tidak ada batas.

# Untuk kalender: akhir semester (contoh: "2026-07-31").

# Untuk regulasi permanen: null.

status: ""

# Enum: "active" | "inactive" | "draft" | "under_revision"

# "active" → Berlaku dan sah

# "inactive" → Sudah diganti atau dicabut

# "draft" → Belum resmi

# "under_revision" → Sedang direvisi

last_updated: ""

# ISO 8601. Tanggal file .md ini terakhir diperbarui.

last_verified: ""

# ISO 8601. Tanggal terakhir konten diverifikasi keakuratannya.

# ============================================================

# BLOK 5 — METADATA RAG (KRITIS)

# ============================================================

retrieval_summary: ""

# !! WAJIB DIISI — TIDAK BOLEH KOSONG !!

# Chunk ini akan di-embed SECARA MANDIRI di ChromaDB sebagai

# chunk bertipe "summary" untuk cold-start retrieval.

# Tulis 3–5 kalimat padat dalam Bahasa Indonesia yang mencakup:

# 1. Jenis & nomor dokumen

# 2. Topik utama yang dicakup

# 3. Siapa yang terdampak

# 4. Periode berlaku (jika relevan)

# JANGAN copy-paste judul — tulis narasi informatif.

chunk_strategy: ""

# Instruksi untuk chunker. Enum:

# "by_chapter" → Potong per BAB

# "by_section" → Potong per Pasal atau heading

# "by_table" → Setiap tabel = satu chunk

# "by_paragraph" → Potong per paragraf

# "fixed_tokens" → Fixed window (narasi panjang)

# "hybrid" → Kombinasi (jelaskan di chunk_notes)

chunk_notes: null

# String atau null. Instruksi tambahan untuk chunker.

# Contoh: "Setiap Pasal adalah unit atomik — jangan potong

# di tengah Pasal. Tabel di Pasal 1 harus satu chunk."

embedding_model: "text-embedding-001"

# Model yang digunakan saat indexing ke ChromaDB.

# Catat ini agar bisa deteksi chunk stale saat ganti model.

# Contoh: "text-embedding-001", "text-embedding-ada-2"

priority: 3

# Integer 1–5. Bobot retrieval relatif.

# 1 → Regulasi aktif (Peraturan Rektor, SK)

# 2 → Kalender akademik aktif

# 3 → Panduan, SOP, kurikulum

# 4 → Profil, sejarah, narasi

# 5 → Arsip / tidak berlaku

related_docs: []

# List doc_id dokumen berkaitan. [] jika tidak ada.

# Contoh: ["UNSRAT-CAL-2026-001", "UNSRAT-GUIDE-2025-003"]

# Harus menggunakan doc_id valid, bukan judul bebas.

# ============================================================

# BLOK 6 — INDEXING & PENCARIAN

# ============================================================

tags:

- ""

# Lowercase, underscore untuk spasi, tanpa quotes.

# Gunakan untuk kategori tematik luas.

# Contoh:

# - peraturan_akademik

# - kalender_akademik

# - krs

# - uts

# - uas

# - wisuda

# - beban_belajar

# - kelulusan

# - mbkm

# - sanksi

keywords:

- ""

# Kata kunci spesifik, akronim, singkatan yang sering dicari.

# String (boleh dengan spasi dan huruf besar).

# Contoh:

# - "UTS"

# - "UAS"

# - "KRS"

# - "SKS"

# - "IPK"

# - "IPS"

# - "DO"

# - "drop out"

# - "cuti akademik"

# - "Portal INSPIRE"

# - "tugas akhir"

entities:

- ""

# Named entities: orang, sistem, program, regulasi yang dirujuk.

# Contoh:

# - "Oktovian Berty Alexander Sompie"

# - "Portal INSPIRE"

# - "Senat UNSRAT"

# - "LPPM"

# - "LPM"

# - "LP3"

---

# ============================================================

# CONTOH IMPLEMENTASI — Peraturan Akademik UNSRAT 2025

# ============================================================

---

doc_id: "UNSRAT-REG-2025-001"
title: "Peraturan Rektor UNSRAT Nomor 01 Tahun 2025 tentang Peraturan Akademik"
version: "1.0"
language_primary: "id"
language_secondary: null
institution: "Universitas Sam Ratulangi"
unit_penerbit: "Rektorat"

content_type: "regulation"
category: "academic"
subcategory:

- "perkuliahan"
- "evaluasi"
- "mbkm"
- "wisuda"
- "sanksi"
  audience:
- "mahasiswa_s1"
- "mahasiswa_s2"
- "mahasiswa_s3"
- "mahasiswa_profesi"
- "dosen"
- "tendik"
  access_level: "public"

nomor_sk: "01/2025"
tanggal_penetapan: "2025-02-13"
pejabat_penandatangan: "Oktovian Berty Alexander Sompie, Rektor UNSRAT"
source_document: "Peraturan-Rektor-Unsrat-Nomor-1-tahun-2025-tentang-Peraturan-Akademik.pdf"
source_url: "https://storage.googleapis.com/unsrat-web/www.unsrat.ac.id/2025/04/91e73dcb-peraturan-rektor-nomor-1-tahun-2025-tentang-peraturan-akademik-__fin.pdf"
supersedes: null

# ↑ null karena Peraturan 2019 belum di-index sebagai doc_id valid.

# Update ke "UNSRAT-REG-2019-001" jika dokumen itu di-index nanti.

superseded_by: null

valid_from: "2025-02-13"
valid_until: null
status: "active"
last_updated: "2025-12-16"
last_verified: "2025-12-16"

retrieval_summary: "Peraturan Rektor UNSRAT Nomor 01 Tahun 2025 adalah regulasi akademik utama yang mengatur seluruh aspek penyelenggaraan pendidikan di UNSRAT, menggantikan peraturan tahun 2019. Mencakup kurikulum, beban belajar (SKS), sistem penilaian (KRS, UTS, UAS, IPK), evaluasi kelulusan, cuti akademik, program luar prodi (MBKM), penelitian, pengabdian masyarakat, hingga sanksi pelanggaran kode etik. Berlaku untuk semua jenjang: Sarjana, Magister, Doktor, Vokasi, dan Profesi."

chunk_strategy: "by_section"
chunk_notes: "Unit atomik adalah Pasal. Jangan memotong di tengah Pasal. Tabel definisi di Pasal 1 dipertahankan sebagai satu chunk. Setiap BAB boleh dijadikan satu chunk jika jumlah Pasal sedikit."
embedding_model: "text-embedding-001"
priority: 1

related_docs:

- "UNSRAT-CAL-2026-001"

tags:

- peraturan_akademik
- kurikulum
- krs
- beban_belajar
- evaluasi
- kelulusan
- cuti_akademik
- uts
- uas
- ipk
- ips
- mbkm
- sanksi

keywords:

- "UTS"
- "UAS"
- "KRS"
- "SKS"
- "IPK"
- "IPS"
- "DO"
- "drop out"
- "cuti akademik"
- "Portal INSPIRE"
- "tugas akhir"
- "skripsi"
- "tesis"
- "disertasi"
- "yudisium"

entities:

- "Oktovian Berty Alexander Sompie"
- "Portal INSPIRE"
- "Senat UNSRAT"
- "LPM"
- "LPPM"
- "LP3"

---
