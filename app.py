import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import torch
import re

st.set_page_config(
    page_title="Peringkasan Berita Indonesia",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
<style>
.judul-app {
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: #1f3d7a;
    margin-bottom: 4px;
}
.subjudul-app {
    text-align: center;
    font-size: 13px;
    color: #666;
    margin-bottom: 16px;
}
.kotak-hasil {
    background-color: #f0f7f0;
    border-left: 5px solid #2e7d32;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 15px;
    color: #1a1a1a;
    line-height: 1.6;
    margin-top: 6px;
}
.badge-kategori {
    display: inline-block;
    background-color: #1f3d7a;
    color: white;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 12px;
}
.label-rouge {
    font-size: 13px;
    color: #555;
    margin-bottom: 2px;
    font-weight: 600;
}
.nilai-rouge {
    font-size: 26px;
    font-weight: bold;
    color: #1f3d7a;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="judul-app">Peringkasan Teks Berita Otomatis</div>', unsafe_allow_html=True)
st.markdown('<div class="subjudul-app">Model: IndoBART-v2 &nbsp;|&nbsp; Dataset: IndoSUM &nbsp;|&nbsp; Evaluasi: ROUGE</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Load model ───────────────────────────────────────────────────
MODEL_ID = "Lyenseptryanti/indobart-summarizer"

@st.cache_resource(show_spinner=False)
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    mdl.eval()
    return tok, mdl

with st.spinner("Memuat model, harap tunggu..."):
    tokenizer, model = load_model()

# ── Deteksi kategori otomatis dari teks ─────────────────────────
KATA_KUNCI_KATEGORI = {
    "Olahraga": [
        "gol", "liga", "pertandingan", "pemain", "klub", "sepak bola",
        "bola basket", "badminton", "tenis", "olimpiade", "juara", "latihan",
        "pelatih", "skuad", "turnamen", "atletik", "renang", "voli",
        "tinju", "motogp", "f1", "sirkuit", "balap", "medali"
    ],
    "Showbiz": [
        "artis", "aktor", "aktris", "film", "sinetron", "lagu", "album",
        "konser", "penyanyi", "band", "musisi", "gosip", "selebriti",
        "drama", "serial", "sutradara", "produser", "spotify", "youtube",
        "instagram", "viral", "idol", "kpop"
    ],
    "Teknologi": [
        "aplikasi", "smartphone", "iphone", "android", "laptop", "komputer",
        "internet", "artificial intelligence", "ai", "robot", "startup",
        "digital", "software", "hardware", "gadget", "samsung", "apple",
        "google", "microsoft", "data", "siber", "hack", "teknologi",
        "jaringan", "5g", "satelit", "chip", "processor"
    ],
    "Hiburan": [
        "wisata", "liburan", "kuliner", "restoran", "cafe", "makanan",
        "fashion", "gaya hidup", "hobi", "pameran", "festival", "konser",
        "bioskop", "streaming", "netflix", "game", "esport"
    ],
    "Inspirasi": [
        "motivasi", "sukses", "pengusaha", "prestasi", "beasiswa",
        "pendidikan", "mahasiswa", "pelajar", "inovasi", "kreasi",
        "wirausaha", "umkm", "karir", "tips", "produktif", "inspirasi"
    ],
    "Tajuk Utama": [
        "presiden", "menteri", "pemerintah", "dpr", "mpr", "polisi",
        "hukum", "korupsi", "kpk", "sidang", "anggaran", "kebijakan",
        "pilkada", "pemilu", "partai", "gubernur", "bupati", "walikota",
        "bencana", "gempa", "banjir", "kebakaran", "kecelakaan",
        "ekonomi", "inflasi", "saham", "rupiah", "ekspor", "impor",
        "covid", "kesehatan", "rumah sakit", "vaksin", "militer",
        "perang", "diplomasi", "pbb", "asean"
    ],
}

def deteksi_kategori(teks: str) -> str:
    teks_lower = teks.lower()
    skor = {kat: 0 for kat in KATA_KUNCI_KATEGORI}
    for kat, kata_list in KATA_KUNCI_KATEGORI.items():
        for kata in kata_list:
            if kata in teks_lower:
                skor[kat] += 1
    kategori_terpilih = max(skor, key=skor.get)
    if skor[kategori_terpilih] == 0:
        return "Tajuk Utama"
    return kategori_terpilih

# ── Input ────────────────────────────────────────────────────────
st.subheader("Input Artikel Berita")

input_teks = st.text_area(
    "Teks Artikel",
    placeholder="Tempelkan artikel berita berbahasa Indonesia di sini...",
    height=280,
    label_visibility="collapsed"
)

referensi_teks = st.text_area(
    "Ringkasan Referensi (opsional — untuk menghitung skor ROUGE)",
    placeholder="Masukkan ringkasan referensi jika ingin melihat skor ROUGE...",
    height=90,
)

st.markdown("")
kol1, kol2, kol3 = st.columns([1.5, 2, 1.5])
with kol2:
    tombol = st.button("Sumarisasi", use_container_width=True, type="primary")

# ── Proses ───────────────────────────────────────────────────────
if tombol:
    if not input_teks.strip():
        st.warning("Masukkan teks artikel terlebih dahulu.")
    else:
        with st.spinner("Sedang meringkas artikel..."):
            inputs = tokenizer(
                input_teks,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
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

            # Deteksi kategori otomatis
            kategori = deteksi_kategori(input_teks)

        # ── Output ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Hasil Peringkasan")

        st.markdown(
            f'<div class="badge-kategori">Kategori: {kategori}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="kotak-hasil">{ringkasan}</div>',
            unsafe_allow_html=True
        )
        st.markdown("")

        jml_kata_artikel   = len(input_teks.split())
        jml_kata_ringkasan = len(ringkasan.split())
        rasio = round((1 - jml_kata_ringkasan / jml_kata_artikel) * 100, 1) if jml_kata_artikel > 0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Panjang Artikel",   f"{jml_kata_artikel} kata")
        c2.metric("Panjang Ringkasan", f"{jml_kata_ringkasan} kata")
        c3.metric("Rasio Kompresi",    f"{rasio}%")

        # ── ROUGE ─────────────────────────────────────────────────
        if referensi_teks.strip():
            st.markdown("")
            st.subheader("Evaluasi ROUGE")
            st.caption("Perbandingan ringkasan model dengan ringkasan referensi.")

            sc   = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
            skor = sc.score(referensi_teks.strip(), ringkasan)

            r1 = round(skor["rouge1"].fmeasure * 100, 2)
            r2 = round(skor["rouge2"].fmeasure * 100, 2)
            rl = round(skor["rougeL"].fmeasure * 100, 2)

            cr1, cr2, crl = st.columns(3)

            with cr1:
                st.markdown('<div class="label-rouge">ROUGE-1</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="nilai-rouge">{r1}%</div>', unsafe_allow_html=True)
                st.progress(min(r1 / 100, 1.0))
            with cr2:
                st.markdown('<div class="label-rouge">ROUGE-2</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="nilai-rouge">{r2}%</div>', unsafe_allow_html=True)
                st.progress(min(r2 / 100, 1.0))
            with crl:
                st.markdown('<div class="label-rouge">ROUGE-L</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="nilai-rouge">{rl}%</div>', unsafe_allow_html=True)
                st.progress(min(rl / 100, 1.0))
        else:
            st.info("Masukkan ringkasan referensi di atas untuk melihat skor ROUGE.")

st.markdown("---")
st.caption("Tugas Akhir — Peringkasan Teks Otomatis Artikel Berita Bahasa Indonesia | IndoBART-v2 + IndoSUM")
