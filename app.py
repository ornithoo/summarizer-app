import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import torch

# ── Konfigurasi halaman ──────────────────────────────────────────
st.set_page_config(
    page_title="Peringkasan Berita Indonesia",
    page_icon="📰",
    layout="centered"
)

# ── CSS tambahan ─────────────────────────────────────────────────
st.markdown("""
<style>
.judul-app {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: #1f3d7a;
    margin-bottom: 4px;
}
.subjudul-app {
    text-align: center;
    font-size: 14px;
    color: #555;
    margin-bottom: 20px;
}
.kotak-hasil {
    background-color: #f0f7f0;
    border-left: 5px solid #2e7d32;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 16px;
    color: #1a1a1a;
    margin-top: 8px;
}
.badge-kategori {
    display: inline-block;
    background-color: #1f3d7a;
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 10px;
}
.label-rouge {
    font-size: 13px;
    color: #444;
    margin-bottom: 2px;
}
.nilai-rouge {
    font-size: 22px;
    font-weight: bold;
    color: #1f3d7a;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────
st.markdown('<div class="judul-app">Peringkasan Teks Berita Otomatis</div>', unsafe_allow_html=True)
st.markdown('<div class="subjudul-app">Model: IndoBART-v2 &nbsp;|&nbsp; Dataset: IndoSUM &nbsp;|&nbsp; Evaluasi: ROUGE</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Load model ────────────────────────────────────────────────────
MODEL_ID = "Lyenseptryanti/indobart-summarizer"

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32
    )
    model.eval()
    return tokenizer, model

with st.spinner("Memuat model, harap tunggu..."):
    tokenizer, model = load_model()

# ── Daftar kategori berita IndoSUM ────────────────────────────────
KATEGORI_LIST = [
    "Otomotif", "Bola", "Bisnis", "Hiburan",
    "Gaya Hidup", "Internasional", "Nasional",
    "Olahraga", "Sains", "Tekno"
]

# ── Form input ────────────────────────────────────────────────────
st.subheader("Input Artikel Berita")

kategori = st.selectbox(
    "Kategori Berita",
    options=KATEGORI_LIST,
    index=6,
    help="Pilih kategori yang sesuai dengan artikel yang Anda masukkan"
)

input_teks = st.text_area(
    "Teks Artikel",
    placeholder="Tempelkan artikel berita berbahasa Indonesia di sini...",
    height=280,
)

referensi_teks = st.text_area(
    "Ringkasan Referensi (opsional — untuk hitung ROUGE)",
    placeholder="Masukkan ringkasan manual jika ingin melihat skor ROUGE...",
    height=100,
)

st.markdown("")

# ── Tombol ────────────────────────────────────────────────────────
col_kiri, col_tengah, col_kanan = st.columns([1.5, 2, 1.5])
with col_tengah:
    tombol = st.button("Sumarisasi", use_container_width=True, type="primary")

# ── Proses ───────────────────────────────────────────────────────
if tombol:
    if not input_teks.strip():
        st.warning("Masukkan teks artikel terlebih dahulu.")
    else:
        with st.spinner("Sedang meringkas artikel..."):

            # Tokenisasi
            inputs = tokenizer(
                input_teks,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )

            # Generate ringkasan
            with torch.no_grad():
                output_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=256,
                    num_beams=2,
                    no_repeat_ngram_size=4,
                    early_stopping=False,
                    length_penalty=2.0,
                )

            ringkasan = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

        # ── Hasil ─────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Hasil Peringkasan")

        # Badge kategori
        st.markdown(f'<div class="badge-kategori">Kategori: {kategori}</div>', unsafe_allow_html=True)

        # Kotak ringkasan
        st.markdown(f'<div class="kotak-hasil">{ringkasan}</div>', unsafe_allow_html=True)
        st.markdown("")

        # Statistik
        jml_kata_artikel  = len(input_teks.split())
        jml_kata_ringkasan = len(ringkasan.split())
        rasio_kompresi = round((1 - jml_kata_ringkasan / jml_kata_artikel) * 100, 1) if jml_kata_artikel > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Panjang Artikel",   f"{jml_kata_artikel} kata")
        col2.metric("Panjang Ringkasan", f"{jml_kata_ringkasan} kata")
        col3.metric("Rasio Kompresi",    f"{rasio_kompresi}%")

        # ── Skor ROUGE (jika ada referensi) ───────────────────────
        if referensi_teks.strip():
            st.markdown("")
            st.subheader("Evaluasi ROUGE")
            st.caption("Skor dihitung berdasarkan perbandingan ringkasan model dengan ringkasan referensi.")

            sc = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"],
                use_stemmer=False
            )
            skor = sc.score(referensi_teks.strip(), ringkasan)

            r1 = round(skor["rouge1"].fmeasure * 100, 2)
            r2 = round(skor["rouge2"].fmeasure * 100, 2)
            rl = round(skor["rougeL"].fmeasure * 100, 2)

            col_r1, col_r2, col_rl = st.columns(3)

            with col_r1:
                st.markdown('<div class="label-rouge">ROUGE-1</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="nilai-rouge">{r1}%</div>', unsafe_allow_html=True)
                st.progress(min(r1 / 100, 1.0))

            with col_r2:
                st.markdown('<div class="label-rouge">ROUGE-2</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="nilai-rouge">{r2}%</div>', unsafe_allow_html=True)
                st.progress(min(r2 / 100, 1.0))

            with col_rl:
                st.markdown('<div class="label-rouge">ROUGE-L</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="nilai-rouge">{rl}%</div>', unsafe_allow_html=True)
                st.progress(min(rl / 100, 1.0))

        else:
            st.info("Masukkan ringkasan referensi di kolom atas untuk melihat skor ROUGE.")

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Tugas Akhir — Peringkasan Teks Otomatis Artikel Berita Bahasa Indonesia | IndoBART-v2 + IndoSUM")
