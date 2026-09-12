import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# ── Konfigurasi halaman ──────────────────────────────────────
st.set_page_config(
    page_title="Peringkasan Berita Indonesia",
    page_icon="📰",
    layout="centered"
)

# ── Judul dan deskripsi ──────────────────────────────────────
st.title("📰 Peringkasan Teks Berita Otomatis")
st.markdown("**Model:** IndoBART-v2 | **Dataset:** IndoSUM")
st.markdown("---")

# ── Load model (cache agar tidak reload setiap kali) ─────────
MODEL_PATH = MODEL_PATH = "Lyenseptryanti/indobart-summarizer"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32   # CPU pakai float32
    )
    model.eval()
    return tokenizer, model

with st.spinner("Memuat model... (hanya sekali saat pertama dibuka)"):
    tokenizer, model = load_model()

st.success("Model siap digunakan!")
st.markdown("---")

# ── Input teks ───────────────────────────────────────────────
st.subheader("Masukkan Artikel Berita")
input_text = st.text_area(
    label="Teks artikel:",
    placeholder="Tempelkan artikel berita bahasa Indonesia di sini...",
    height=300,
    label_visibility="collapsed"
)

# ── Tombol sumarisasi ────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    summarize_btn = st.button(
        "🔍 Sumarisasi",
        use_container_width=True,
        type="primary"
    )

# ── Proses sumarisasi ────────────────────────────────────────
if summarize_btn:
    if not input_text.strip():
        st.warning("Silakan masukkan teks artikel terlebih dahulu.")
    else:
        with st.spinner("Sedang meringkas..."):
            # Tokenisasi input
            inputs = tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )

            # Generate ringkasan
            with torch.no_grad():
                output_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=64,
                    num_beams=2,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

            # Decode hasil
            summary = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

        # ── Tampilkan hasil ──────────────────────────────────
        st.markdown("---")
        st.subheader("Hasil Ringkasan")
        st.success(summary)

        # Info tambahan
        input_words  = len(input_text.split())
        output_words = len(summary.split())
        rasio        = round((1 - output_words / input_words) * 100, 1) if input_words > 0 else 0

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Panjang Artikel", f"{input_words} kata")
        col_b.metric("Panjang Ringkasan", f"{output_words} kata")
        col_c.metric("Kompresi", f"{rasio}%")

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.caption("Tugas Akhir — Peringkasan Teks Otomatis Artikel Berita Bahasa Indonesia")
