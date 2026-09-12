import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="judul-app">Peringkasan Teks Berita Otomatis</div>', unsafe_allow_html=True)
st.markdown('<div class="subjudul-app">Model: IndoBART-v2 &nbsp;|&nbsp; Dataset: IndoSUM</div>', unsafe_allow_html=True)
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

# ── Fungsi Pembersihan dan Perapihan Teks (Pre & Post-Processing) ─
def bersihkan_input(teks: str) -> str:
    # 1. Hapus dateline / nama media di awal teks (misal: "KOMPAS.com - ", "JAKARTA, CNN - ", dll)
    teks = re.sub(r'^[A-Za-z0-9\.\s/,-]+?\s*[-–—]\s*', '', teks.strip())
    
    # 2. Hapus URL atau link jika ada yang tertinggal
    teks = re.sub(r'https?://\S+|www\.\S+', '', teks)
    
    # 3. Normalkan spasi dan samakan ke huruf kecil sesuai format training model
    teks = re.sub(r'\s+', ' ', teks)
    return teks.strip().lower()

def format_hasil_ringkasan(teks: str) -> str:
    # 1. Bersihkan jika ada nama portal/ekstensi situs yang lolos di awal kalimat (misal: "Bas.Com -", "kompas.com -")
    teks = re.sub(r'^[A-Za-z0-9\.]+(?:\.com|\.co\.id|\.id|\.net)\s*[-–—:]*\s*', '', teks, flags=re.IGNORECASE).strip()
    
    # 2. Hapus tanda tanya berulang (artefak token rusak)
    teks = re.sub(r'\?+', '', teks).strip()
    
    # 3. Rapikan tanda baca ganda atau tabrakan tanda baca (misal: ",." -> ".")
    teks = re.sub(r'[,;:]+\s*([.!?])', r'\1', teks)
    teks = re.sub(r'\s+([,.:;!?])', r'\1', teks)
    teks = re.sub(r'([(\[{])\s+', r'\1', teks)
    teks = re.sub(r'\s+([)\]}])', r'\1', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    
    # 4. Kapitalisasi huruf pertama di setiap awal kalimat
    kalimat = re.split(r'([.!?]\s*)', teks)
    formatted = "".join([k.capitalize() for k in kalimat])
    
    # 5. Pastikan kalimat terakhir memiliki tanda baca penutup
    if formatted and not formatted.endswith(('.', '!', '?')):
        formatted += '.'
        
    return formatted

# ── Deteksi kategori otomatis ─────────────────────────────────────
KATA_KUNCI = {
    "Olahraga": [
        "gol", "liga", "pertandingan", "pemain", "klub", "sepak bola",
        "bola basket", "badminton", "tenis", "olimpiade", "juara",
        "pelatih", "turnamen", "atletik", "renang", "medali", "balap",
        "motogp", "f1", "sirkuit", "tinju", "voli"
    ],
    "Showbiz": [
        "artis", "aktor", "aktris", "film", "sinetron", "lagu", "album",
        "konser", "penyanyi", "band", "musisi", "selebriti", "drama",
        "serial", "sutradara", "kpop", "idol", "viral", "spotify"
    ],
    "Teknologi": [
        "aplikasi", "smartphone", "iphone", "android", "laptop", "komputer",
        "internet", "artificial intelligence", "ai", "robot", "startup",
        "digital", "software", "hardware", "gadget", "samsung", "apple",
        "google", "microsoft", "5g", "satelit", "chip", "siber", "hack"
    ],
    "Hiburan": [
        "wisata", "liburan", "kuliner", "restoran", "cafe", "makanan",
        "fashion", "gaya hidup", "hobi", "pameran", "festival",
        "bioskop", "streaming", "netflix", "game", "esport"
    ],
    "Inspirasi": [
        "motivasi", "sukses", "pengusaha", "prestasi", "beasiswa",
        "pendidikan", "mahasiswa", "inovasi", "wirausaha", "umkm",
        "karir", "produktif", "inspirasi", "kreasi"
    ],
    "Tajuk Utama": [
        "presiden", "menteri", "pemerintah", "dpr", "polisi", "hukum",
        "korupsi", "kpk", "sidang", "kebijakan", "pilkada", "pemilu",
        "partai", "gubernur", "bencana", "gempa", "banjir", "kebakaran",
        "gunung", "erupsi", "lahar", "vulkanik", "tsunami", "longsor",
        "ekosistem", "habitat", "konservasi", "vegetasi",
        "ekonomi", "inflasi", "saham", "rupiah", "ekspor", "impor",
        "kesehatan", "rumah sakit", "vaksin", "militer", "perang"
    ],
}

def deteksi_kategori(teks: str) -> str:
    teks_lower = teks.lower()
    skor = {kat: 0 for kat in KATA_KUNCI}
    for kat, kata_list in KATA_KUNCI.items():
        for kata in kata_list:
            if kata in teks_lower:
                skor[kat] += 1
    best = max(skor, key=skor.get)
    return best if skor[best] > 0 else "Tajuk Utama"

# ── Input ─────────────────────────────────────────────────────────
st.subheader("Input Artikel Berita")

input_teks = st.text_area(
    "Teks Artikel",
    placeholder="Tempelkan artikel berita berbahasa Indonesia di sini...",
    height=300,
    label_visibility="collapsed"
)

st.markdown("")
kol1, kol2, kol3 = st.columns([1.5, 2, 1.5])
with kol2:
    tombol = st.button("Sumarisasi", use_container_width=True, type="primary")

# ── Proses ────────────────────────────────────────────────────────
if tombol:
    if not input_teks.strip():
        st.warning("Masukkan teks artikel terlebih dahulu.")
    else:
        with st.spinner("Sedang meringkas artikel..."):
            # 1. Bersihkan dan normalkan input sebelum masuk ke tokenizer
            teks_siap = bersihkan_input(input_teks)
            
            inputs = tokenizer(
                teks_siap,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            
            # 2. Parameter generasi teks yang seimbang agar tidak memicu halusinasi/tanda tanya
            with torch.no_grad():
                output_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=150,
                    min_new_tokens=20,
                    num_beams=3,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    length_penalty=1.0,
                )
                
            ringkasan_mentah = tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            # 3. Rapikan hasil teks dan kembalikan kapitalisasi awal kalimat
            ringkasan = format_hasil_ringkasan(ringkasan_mentah)
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

st.markdown("---")
st.caption("Tugas Akhir — Peringkasan Teks Otomatis Artikel Berita Bahasa Indonesia | IndoBART-v2 + IndoSUM")
