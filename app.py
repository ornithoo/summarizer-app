import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import torch
import json
import os
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Peringkasan Berita Indonesia",
    page_icon="📰",
    layout="wide"
)

st.markdown("""
<style>
.judul { text-align:center; font-size:26px; font-weight:bold; color:#5B7FA6; margin-bottom:4px; }
.subjudul { text-align:center; font-size:13px; color:#666; margin-bottom:16px; }
.kotak-hasil {
    background:#f0f7f0; border-left:5px solid #2e7d32;
    border-radius:8px; padding:16px 20px;
    font-size:15px; color:#1a1a1a; line-height:1.6; margin-top:6px;
}
.badge {
    display:inline-block; background:#1f3d7a; color:white;
    padding:4px 16px; border-radius:20px;
    font-size:13px; font-weight:600; margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="judul">Peringkasan Teks Berita Otomatis</div>', unsafe_allow_html=True)
st.markdown('<div class="subjudul">Model: IndoBART-v2 &nbsp;|&nbsp; Dataset: IndoSUM &nbsp;|&nbsp; Evaluasi: ROUGE</div>', unsafe_allow_html=True)
st.markdown("---")

# ================================================================
# DAFTAR 10 MODEL — sesuaikan nama repo dengan yang Anda upload
# ================================================================
MODELS = {
    "Model 1  — LR=5e-6, BS=4  | ROUGE-L F1: 69.20%": "Lyenseptryanti/indobart-lr5e6-bs4",
    "Model 2  — LR=1e-5, BS=4  | ROUGE-L F1: 69.68%  [Terbaik]": "Lyenseptryanti/indobart-lr1e5-bs4",
    "Model 3  — LR=5e-5, BS=4  | ROUGE-L F1: 68.95%": "Lyenseptryanti/indobart-lr5e5-bs4",
    "Model 4  — LR=1e-4, BS=4  | ROUGE-L F1: 67.82%": "Lyenseptryanti/indobart-lr1e4-bs4",
    "Model 5  — LR=1e-3, BS=4  | ROUGE-L F1: 2.18%":  "Lyenseptryanti/indobart-lr1e3-bs4",
    "Model 6  — LR=5e-6, BS=8  | ROUGE-L F1: 69.14%": "Lyenseptryanti/indobart-lr5e6-bs8",
    "Model 7  — LR=1e-5, BS=8  | ROUGE-L F1: 69.04%": "Lyenseptryanti/indobart-lr1e5-bs8",
    "Model 8  — LR=5e-5, BS=8  | ROUGE-L F1: 68.96%": "Lyenseptryanti/indobart-lr5e5-bs8",
    "Model 9  — LR=1e-4, BS=8  | ROUGE-L F1: 68.62%": "Lyenseptryanti/indobart-lr1e4-bs8",
    "Model 10 — LR=1e-3, BS=8  | ROUGE-L F1: 9.37%":  "Lyenseptryanti/indobart-lr1e3-bs8",
}

# Skor F1 hasil training untuk grafik perbandingan
SKOR_TRAINING = {
    "M1\n5e-6/BS4":  {"R1":72.21,"R2":65.15,"RL":69.20},
    "M2\n1e-5/BS4":  {"R1":72.63,"R2":65.68,"RL":69.68},
    "M3\n5e-5/BS4":  {"R1":71.95,"R2":64.79,"RL":68.95},
    "M4\n1e-4/BS4":  {"R1":70.98,"R2":63.40,"RL":67.82},
    "M5\n1e-3/BS4":  {"R1":2.62, "R2":0.37, "RL":2.18},
    "M6\n5e-6/BS8":  {"R1":72.15,"R2":65.04,"RL":69.14},
    "M7\n1e-5/BS8":  {"R1":72.11,"R2":64.90,"RL":69.04},
    "M8\n5e-5/BS8":  {"R1":72.04,"R2":64.88,"RL":68.96},
    "M9\n1e-4/BS8":  {"R1":71.66,"R2":64.45,"RL":68.62},
    "M10\n1e-3/BS8": {"R1":13.38,"R2":2.21, "RL":9.37},
}

# ── Deteksi kategori otomatis ─────────────────────────────────────
KATA_KUNCI = {
    "Olahraga":    ["gol","liga","pertandingan","pemain","klub","sepak bola","juara","turnamen","medali","olimpiade","balap"],
    "Showbiz":     ["artis","aktor","aktris","film","sinetron","lagu","album","konser","penyanyi","selebriti","kpop","viral"],
    "Teknologi":   ["aplikasi","smartphone","android","laptop","internet","ai","robot","startup","digital","5g","samsung","apple"],
    "Hiburan":     ["wisata","kuliner","restoran","festival","bioskop","streaming","netflix","game","esport"],
    "Inspirasi":   ["motivasi","sukses","beasiswa","pendidikan","inovasi","wirausaha","prestasi","mahasiswa"],
    "Tajuk Utama": ["presiden","menteri","pemerintah","polisi","hukum","korupsi","bencana","gempa","banjir","ekonomi","pilkada"],
}

def deteksi_kategori(teks):
    teks = teks.lower()
    skor = {k: sum(1 for w in v if w in teks) for k, v in KATA_KUNCI.items()}
    best = max(skor, key=skor.get)
    return best if skor[best] > 0 else "Tajuk Utama"

# ── Load model (cache agar tidak reload setiap klik) ─────────────
@st.cache_resource(show_spinner=False)
def load_model(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)
    mdl.eval()
    return tok, mdl

# ── Load data uji IndoSum dari file CSV (article, summary, category) ─
NAMA_FILE_DATA = "indosum_test01.csv"

@st.cache_data
def load_test_data():
    if not os.path.exists(NAMA_FILE_DATA):
        return []
    df = pd.read_csv(NAMA_FILE_DATA)
    df = df.dropna(subset=["article", "summary"])
    data = []
    for _, row in df.iterrows():
        article = str(row["article"]).strip()
        summary = str(row["summary"]).strip()
        if article and summary:
            data.append({
                "article":  article,
                "summary":  summary,
                "category": str(row.get("category", "")).strip(),
            })
    return data

test_data = load_test_data()

# ================================================================
# SIDEBAR — Pilih Model
# ================================================================
with st.sidebar:
    st.header("Pilih Model")
    pilihan_model = st.selectbox(
        "Skenario Hyperparameter",
        options=list(MODELS.keys()),
        index=1,
    )
    model_id = MODELS[pilihan_model]
    st.caption(f"Model ID: `{model_id}`")

    st.markdown("---")
    st.header("Mode Input")
    mode = st.radio(
        "",
        ["Berita Online (tanpa ROUGE)", "Data Uji IndoSum (dengan ROUGE)"],
        label_visibility="collapsed"
    )

# ================================================================
# AREA UTAMA
# ================================================================
col_input, col_output = st.columns([1, 1], gap="large")

artikel_input   = ""
referensi_input = ""
kategori_fix    = ""

# ── INPUT ─────────────────────────────────────────────────────────
with col_input:
    st.subheader("Input Artikel")

    if mode == "Berita Online (tanpa ROUGE)":
        artikel_input = st.text_area(
            "Tempel artikel berita di sini:",
            placeholder="Tempelkan artikel berita berbahasa Indonesia dari sumber mana saja...",
            height=320,
            label_visibility="visible"
        )
        st.caption("Skor ROUGE tidak ditampilkan karena tidak ada ringkasan referensi.")

    else:
        if not test_data:
            st.error(f"File '{NAMA_FILE_DATA}' tidak ditemukan di folder app.")
        else:
            idx = st.selectbox(
                "Pilih artikel dari dataset IndoSum:",
                options=range(len(test_data)),
                format_func=lambda i: (
                    f"[{test_data[i]['category']}] {test_data[i]['article'][:70]}..."
                )
            )
            artikel_input   = test_data[idx]["article"]
            referensi_input = test_data[idx]["summary"]
            kategori_fix    = test_data[idx]["category"]

            with st.expander("Artikel lengkap"):
                st.write(artikel_input)
            with st.expander("Ringkasan referensi"):
                st.write(referensi_input)

    st.markdown("")
    tombol = st.button("Sumarisasi", use_container_width=True, type="primary")

# ── OUTPUT ────────────────────────────────────────────────────────
with col_output:
    st.subheader("Hasil Peringkasan")

    if tombol:
        if not artikel_input.strip():
            st.warning("Masukkan artikel terlebih dahulu.")
        else:
            with st.spinner("Memuat model dan meringkas artikel..."):
                tokenizer, model = load_model(model_id)
                inputs = tokenizer(
                    artikel_input,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    output_ids = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=85,
                        num_beams=5,
                        length_penalty=1.0,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                    )
                ringkasan = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                ).strip()
                kategori = kategori_fix if kategori_fix else deteksi_kategori(artikel_input)

            # Tampilkan ringkasan
            st.markdown(f'<div class="badge">Kategori: {kategori}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kotak-hasil">{ringkasan}</div>', unsafe_allow_html=True)
            st.markdown("")

            # Statistik
            jml_art  = len(artikel_input.split())
            jml_ring = len(ringkasan.split())
            rasio    = round((1 - jml_ring/jml_art)*100, 1) if jml_art > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Panjang Artikel",   f"{jml_art} kata")
            c2.metric("Panjang Ringkasan", f"{jml_ring} kata")
            c3.metric("Rasio Kompresi",    f"{rasio}%")

            # ── ROUGE (hanya mode IndoSum) ────────────────────────
            if referensi_input.strip():
                st.markdown("---")
                st.subheader("Skor ROUGE")
                sc   = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=False)
                skor = sc.score(referensi_input.strip(), ringkasan)
                r1   = round(skor["rouge1"].fmeasure * 100, 2)
                r2   = round(skor["rouge2"].fmeasure * 100, 2)
                rl   = round(skor["rougeL"].fmeasure * 100, 2)

                cr1, cr2, crl = st.columns(3)
                cr1.metric("ROUGE-1", f"{r1}%")
                cr2.metric("ROUGE-2", f"{r2}%")
                crl.metric("ROUGE-L", f"{rl}%")

                # Grafik perbandingan artikel ini vs skor training
                idx_model   = list(MODELS.keys()).index(pilihan_model)
                skor_train  = list(SKOR_TRAINING.values())[idx_model]
                label_short = f"M{idx_model+1}"

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Artikel Ini", x=["ROUGE-1","ROUGE-2","ROUGE-L"],
                    y=[r1,r2,rl], marker_color="#2e7d32",
                    text=[f"{v}%" for v in [r1,r2,rl]], textposition="outside"
                ))
                fig.add_trace(go.Bar(
                    name=f"Rata-rata Training {label_short}",
                    x=["ROUGE-1","ROUGE-2","ROUGE-L"],
                    y=[skor_train["R1"],skor_train["R2"],skor_train["RL"]],
                    marker_color="#1f3d7a",
                    text=[f"{v}%" for v in [skor_train["R1"],skor_train["R2"],skor_train["RL"]]],
                    textposition="outside"
                ))
                fig.update_layout(
                    title="Artikel Ini vs Rata-rata Training",
                    barmode="group",
                    yaxis=dict(title="Skor (%)", range=[0,100]),
                    legend=dict(orientation="h", y=-0.25),
                    height=320, margin=dict(t=40,b=60)
                )
                st.plotly_chart(fig, use_container_width=True)

# ================================================================
# GRAFIK PERBANDINGAN 10 MODEL — full width
# ================================================================
st.markdown("---")
st.subheader("Perbandingan Performa 10 Skenario Model")

label_x = list(SKOR_TRAINING.keys())
fig_all = go.Figure()
for metric, color in [("R1","#1f3d7a"),("R2","#2e7d32"),("RL","#c0392b")]:
    vals = [v[metric] for v in SKOR_TRAINING.values()]
    nama = {"R1":"ROUGE-1","R2":"ROUGE-2","RL":"ROUGE-L"}[metric]
    fig_all.add_trace(go.Bar(
        name=nama, x=label_x, y=vals, marker_color=color,
        text=[f"{v}%" for v in vals], textposition="outside"
    ))

fig_all.add_vline(x=1, line_dash="dash", line_color="orange",
                   annotation_text="Model Terbaik (M2)", annotation_position="top right")
fig_all.update_layout(
    barmode="group",
    yaxis=dict(title="Skor F1 (%)", range=[0,100]),
    xaxis=dict(title="Skenario Model"),
    legend=dict(orientation="h", y=1.08),
    height=430, margin=dict(t=20,b=40)
)
st.plotly_chart(fig_all, use_container_width=True)

# Tabel ringkasan
st.markdown("**Tabel Ringkasan Semua Skenario**")
rows = []
for i,(label,skor) in enumerate(SKOR_TRAINING.items()):
    lr  = list(MODELS.keys())[i].split("LR=")[1].split(",")[0]
    bs  = list(MODELS.keys())[i].split("BS=")[1].split(" ")[0].replace("|","").strip()
    rows.append({
        "No": i+1, "Learning Rate": lr, "Batch Size": bs,
        "ROUGE-1 (F1)": f"{skor['R1']}%",
        "ROUGE-2 (F1)": f"{skor['R2']}%",
        "ROUGE-L (F1)": f"{skor['RL']}%",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Peringkasan Teks Otomatis Artikel Berita Bahasa Indonesia | IndoBART-v2")
