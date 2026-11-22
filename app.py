"""Aplikasi Web SPK UMKM Ayam Geprek — Editorial Minimalist UI (DM Sans)."""

from pathlib import Path

import pandas as pd
import streamlit as st

from algorithms import (
    CATEGORICAL_FEATURES,
    SAW_CRITERIA_DEFAULT,
    calculate_saw,
    enrich_menu_row,
    evaluate_classifier,
    extract_c45_rules,
    predict_c45,
    predict_naive_bayes,
    train_c45,
    train_naive_bayes,
)

DATA_PATH = Path(__file__).parent / "umkm_ayam_geprek.csv"

REQUIRED_COLUMNS = [
    "kode", "nama_barang", "kategori", "harga_beli", "harga_jual", "stok_saat_ini",
    "jual_minggu1", "jual_minggu2", "jual_minggu3", "jual_minggu4", "rata_jual_mingguan",
    "lead_time_hari", "margin_persen", "rentang_harga", "level_margin",
    "frekuensi_kebutuhan", "label",
]

FEATURE_LABELS = {
    "kategori": "Kategori",
    "rentang_harga": "Rentang Harga",
    "level_margin": "Level Margin",
    "frekuensi_kebutuhan": "Frekuensi Kebutuhan",
}

CRITERIA_HELP = {
    "rata_jual_mingguan": "Semakin tinggi penjualan mingguan, semakin tinggi prioritas restock.",
    "margin_persen": "Semakin besar margin, semakin layak diprioritaskan.",
    "stok_saat_ini": "Stok yang lebih sedikit lebih mendesak untuk segera diisi ulang.",
    "lead_time_hari": "Waktu tunggu pasokan dari pemesanan hingga siap dijual.",
}

WEIGHT_PRESETS = {
    "Seimbang": {"rata_jual_mingguan": 0.35, "margin_persen": 0.25, "stok_saat_ini": 0.20, "lead_time_hari": 0.20},
    "Fokus menu laris": {"rata_jual_mingguan": 0.50, "margin_persen": 0.15, "stok_saat_ini": 0.25, "lead_time_hari": 0.10},
    "Fokus margin": {"rata_jual_mingguan": 0.25, "margin_persen": 0.45, "stok_saat_ini": 0.15, "lead_time_hari": 0.15},
    "Fokus stok menipis": {"rata_jual_mingguan": 0.25, "margin_persen": 0.15, "stok_saat_ini": 0.45, "lead_time_hari": 0.15},
}

st.set_page_config(
    page_title="SPK UMKM Ayam Geprek",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300..700;1,9..40,300..700&display=swap');

    html, body, [class*="css"], .stApp, button, input, select, textarea {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        letter-spacing: -0.018em;
        font-feature-settings: "cv02", "cv03", "cv04", "cv11";
    }

    .block-container {
        max-width: 1080px;
        padding-top: 2.2rem;
        padding-bottom: 3.5rem;
    }

    #MainMenu, footer, header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'DM Sans', sans-serif !important;
        letter-spacing: -0.03em !important;
        font-weight: 600 !important;
    }

    .page-kicker {
        font-size: 0.78rem;
        font-weight: 500;
        opacity: 0.55;
        letter-spacing: -0.01em;
        margin-bottom: 0.2rem;
    }

    .page-title {
        font-size: 1.95rem;
        font-weight: 600;
        letter-spacing: -0.04em;
        line-height: 1.15;
        margin: 0;
    }

    .page-sub {
        font-size: 0.94rem;
        opacity: 0.65;
        letter-spacing: -0.015em;
        margin: 0.35rem 0 1.4rem 0;
    }

    .section-head {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: -0.025em;
        margin: 0.2rem 0 0.15rem 0;
    }

    .section-note {
        font-size: 0.86rem;
        opacity: 0.62;
        letter-spacing: -0.012em;
        margin: 0 0 1rem 0;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 8px;
        padding: 14px 16px;
        background: transparent;
    }

    [data-testid="stMetricLabel"] p {
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        opacity: 0.62;
        letter-spacing: -0.01em !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.035em !important;
        font-variant-numeric: tabular-nums;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.18);
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.65rem 0.1rem;
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: -0.018em;
    }

    .plain-box {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 8px;
        padding: 16px 18px;
        background: transparent;
        height: 100%;
    }

    .plain-kicker {
        font-size: 0.76rem;
        font-weight: 500;
        opacity: 0.55;
        letter-spacing: -0.01em;
    }

    .plain-title {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: -0.028em;
        margin: 4px 0 6px 0;
    }

    .plain-meta {
        font-size: 0.86rem;
        opacity: 0.72;
        letter-spacing: -0.014em;
        line-height: 1.45;
        font-variant-numeric: tabular-nums;
    }

    .pred-value {
        font-size: 1.8rem;
        font-weight: 600;
        letter-spacing: -0.04em;
        line-height: 1.15;
        margin: 6px 0 8px 0;
    }

    .rule-item {
        font-size: 0.88rem;
        padding: 8px 12px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.14);
        letter-spacing: -0.015em;
    }
    .rule-item:last-child {
        border-bottom: none;
    }
</style>
"""


def load_dataset(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file is not None else pd.read_csv(DATA_PATH)
    except Exception as exc:  # noqa: BLE001
        return None, f"File tidak bisa dibaca: {exc}"
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, "Kolom wajib tidak ditemukan: " + ", ".join(missing)
    if df.empty:
        return None, "CSV tidak berisi data."
    return df, None


def days_left(row: pd.Series) -> float:
    daily = row["rata_jual_mingguan"] / 7
    return round(row["stok_saat_ini"] / daily, 1) if daily > 0 else 999.0


def stock_status(row: pd.Series) -> str:
    left = days_left(row)
    if left <= row["lead_time_hari"] + 1:
        return "Segera restock"
    if left <= row["lead_time_hari"] + 4:
        return "Pantau"
    return "Aman"


def plain_box(kicker: str, title: str, meta: str) -> None:
    st.markdown(
        f'<div class="plain-box">'
        f'<div class="plain-kicker">{kicker}</div>'
        f'<div class="plain-title">{title}</div>'
        f'<div class="plain-meta">{meta}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    head_left, head_right = st.columns([3.2, 1], vertical_alignment="bottom")
    with head_left:
        st.markdown('<div class="page-kicker">Sistem Pendukung Keputusan</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-title">UMKM Ayam Geprek</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="page-sub">Klasifikasi perputaran menu (Naive Bayes, C.45) dan urutan prioritas restock (SAW).</div>',
            unsafe_allow_html=True,
        )
    with head_right:
        with st.popover("Sumber data", width="stretch"):
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            st.caption("Biarkan kosong untuk memakai umkm_ayam_geprek.csv.")
            st.download_button(
                "Unduh template CSV",
                data=DATA_PATH.read_bytes(),
                file_name="umkm_ayam_geprek.csv",
                mime="text/csv",
                width="stretch",
            )

    df, error = load_dataset(uploaded)
    if error:
        st.error(error)
        st.stop()

    df = df.copy()
    df["sisa_hari_stok"] = df.apply(days_left, axis=1)
    df["status_stok"] = df.apply(stock_status, axis=1)

    tab_data, tab_clf, tab_saw, tab_rumus = st.tabs(
        ["Ringkasan", "Klasifikasi Menu", "Prioritas Restock", "Rumus"]
    )
    with tab_data:
        render_dashboard(df, is_default=(uploaded is None))
    with tab_clf:
        render_classification(df)
    with tab_saw:
        render_saw(df)
    with tab_rumus:
        render_formulas()


def render_dashboard(df: pd.DataFrame, is_default: bool) -> None:
    urgent_df = df[df["status_stok"] == "Segera restock"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total menu", len(df))
    c2.metric("Fast moving", int((df["label"] == "Fast").sum()))
    c3.metric("Slow moving", int((df["label"] == "Slow").sum()))
    c4.metric("Perlu restock", len(urgent_df))

    if not urgent_df.empty:
        names = ", ".join(urgent_df["nama_barang"].tolist())
        st.warning(f"Stok mendekati batas waktu pasokan: {names}.")

    st.write("")
    f1, f2, f3 = st.columns([2.1, 1.2, 1.1])
    search = f1.text_input("Cari menu", placeholder="Cari nama menu...", label_visibility="collapsed")
    kategori = f2.selectbox(
        "Kategori",
        ["Semua kategori"] + sorted(df["kategori"].unique().tolist()),
        label_visibility="collapsed",
    )
    label = f3.segmented_control("Label", ["Semua", "Fast", "Slow"], default="Semua", label_visibility="collapsed")

    view = df
    if search:
        view = view[view["nama_barang"].str.contains(search, case=False, na=False)]
    if kategori != "Semua kategori":
        view = view[view["kategori"] == kategori]
    if label and label != "Semua":
        view = view[view["label"] == label]

    if view.empty:
        st.info("Tidak ada menu yang sesuai dengan filter.")
    else:
        st.dataframe(
            view[
                [
                    "kode", "nama_barang", "kategori", "harga_jual", "stok_saat_ini",
                    "rata_jual_mingguan", "sisa_hari_stok", "margin_persen", "label", "status_stok",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "kode": st.column_config.TextColumn("Kode", width="small"),
                "nama_barang": st.column_config.TextColumn("Nama menu", width="medium"),
                "kategori": "Kategori",
                "harga_jual": st.column_config.NumberColumn("Harga jual", format="Rp %d"),
                "stok_saat_ini": st.column_config.NumberColumn("Stok", format="%d"),
                "rata_jual_mingguan": st.column_config.ProgressColumn(
                    "Jual / minggu", format="%.1f", min_value=0, max_value=float(df["rata_jual_mingguan"].max())
                ),
                "sisa_hari_stok": st.column_config.NumberColumn("Estimasi stok", format="%.1f hari"),
                "margin_persen": st.column_config.NumberColumn("Margin", format="%.1f%%"),
                "label": "Kelas",
                "status_stok": "Status",
            },
        )

    with st.expander("Tambah menu baru"):
        with st.form("add_menu_form", border=False, clear_on_submit=True):
            a1, a2, a3 = st.columns([1, 2, 1.5])
            kode = a1.text_input("Kode", value=f"B{len(df) + 1:03d}")
            nama = a2.text_input("Nama menu", placeholder="Contoh: Ayam Geprek Mentega")
            kat = a3.selectbox("Kategori", sorted(df["kategori"].unique().tolist()))

            b1, b2, b3, b4 = st.columns(4)
            harga_beli = b1.number_input("Harga beli (Rp)", min_value=500, value=10000, step=500)
            harga_jual = b2.number_input("Harga jual (Rp)", min_value=500, value=14000, step=500)
            stok = b3.number_input("Stok saat ini", min_value=0, value=20, step=1)
            lead_time = b4.number_input("Lead time (hari)", min_value=1, value=2, step=1)

            st.caption("Penjualan 4 minggu terakhir dan frekuensi kebutuhan")
            m = st.columns(5)
            m1 = m[0].number_input("Minggu 1", min_value=0, value=15, step=1)
            m2 = m[1].number_input("Minggu 2", min_value=0, value=16, step=1)
            m3 = m[2].number_input("Minggu 3", min_value=0, value=14, step=1)
            m4 = m[3].number_input("Minggu 4", min_value=0, value=15, step=1)
            freq = m[4].selectbox("Frekuensi", ["Harian", "Mingguan", "Bulanan"])

            if st.form_submit_button("Simpan ke dataset", type="primary"):
                if not nama.strip():
                    st.error("Nama menu wajib diisi.")
                elif harga_jual <= harga_beli:
                    st.error("Harga jual harus lebih besar dari harga beli.")
                elif kode.strip() in df["kode"].astype(str).tolist():
                    st.error(f"Kode {kode} sudah digunakan.")
                elif not is_default:
                    st.warning("Penyimpanan permanen hanya berlaku untuk dataset bawaan.")
                else:
                    new_row = enrich_menu_row({
                        "kode": kode.strip(), "nama_barang": nama.strip(), "kategori": kat,
                        "harga_beli": harga_beli, "harga_jual": harga_jual, "stok_saat_ini": stok,
                        "jual_minggu1": m1, "jual_minggu2": m2, "jual_minggu3": m3, "jual_minggu4": m4,
                        "lead_time_hari": lead_time, "frekuensi_kebutuhan": freq,
                    })
                    base = df[REQUIRED_COLUMNS]
                    pd.concat([base, pd.DataFrame([new_row])[REQUIRED_COLUMNS]], ignore_index=True).to_csv(
                        DATA_PATH, index=False
                    )
                    st.toast(f"Menu '{nama}' tersimpan ({new_row['label']}).")
                    st.rerun()


def render_classification(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-head">Simulasi klasifikasi perputaran menu</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Ubah parameter di kiri untuk memprediksi kelas Fast atau Slow secara langsung.</div>',
        unsafe_allow_html=True,
    )

    algo = st.segmented_control("Algoritma", ["Naive Bayes", "C.45"], default="Naive Bayes")
    algo = algo or "Naive Bayes"
    model = train_naive_bayes(df) if algo == "Naive Bayes" else train_c45(df)

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        with st.container(border=True):
            kat = st.selectbox("Kategori", sorted(df["kategori"].unique().tolist()))
            rh = st.pills(
                "Rentang harga",
                ["Murah", "Sedang", "Mahal"],
                default="Sedang",
                help="Murah < Rp10.000, Sedang Rp10.000-30.000, Mahal > Rp30.000.",
            )
            lm = st.pills(
                "Level margin",
                ["Rendah", "Sedang", "Tinggi"],
                default="Tinggi",
                help="Rendah < 10%, Sedang 10-20%, Tinggi > 20%.",
            )
            freq = st.pills("Frekuensi kebutuhan", ["Harian", "Mingguan", "Bulanan"], default="Harian")

    sample = {
        "kategori": kat,
        "rentang_harga": rh or "Sedang",
        "level_margin": lm or "Tinggi",
        "frekuensi_kebutuhan": freq or "Harian",
    }

    with right:
        if algo == "Naive Bayes":
            pred, probs = predict_naive_bayes(model, sample)
            conf_line = f"Probabilitas Fast {probs['Fast'] * 100:.1f}% / Slow {probs['Slow'] * 100:.1f}%"
        else:
            pred = predict_c45(model, sample)
            probs = None
            conf_line = "Ditentukan dari aturan pohon keputusan C.45"

        action = (
            "Perputaran cepat. Pastikan ketersediaan bahan dipantau harian."
            if pred == "Fast"
            else "Perputaran lambat. Batasi jumlah stok agar modal tidak tertahan."
        )
        st.markdown(
            f'<div class="plain-box">'
            f'<div class="plain-kicker">Hasil prediksi / {algo}</div>'
            f'<div class="pred-value">{pred} Moving</div>'
            f'<div class="plain-meta">{conf_line}<br><br>{action}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if probs is not None:
            st.write("")
            st.progress(probs["Fast"], text=f"Peluang Fast: {probs['Fast'] * 100:.1f}%")

    st.divider()
    acc, cm = evaluate_classifier(df, algo)
    e1, e2 = st.columns([1, 2.2], gap="large")
    with e1:
        st.metric("Akurasi pada dataset", f"{acc}%")
    with e2:
        with st.expander("Confusion matrix"):
            st.dataframe(cm, width="stretch")

    with st.expander("Detail perhitungan model"):
        if algo == "Naive Bayes":
            st.caption("Probabilitas Prior")
            st.json({k: round(v, 4) for k, v in model["priors"].items()})
            cols = st.columns(2)
            for idx, feat in enumerate(CATEGORICAL_FEATURES):
                with cols[idx % 2]:
                    st.caption(f"Likelihood: {FEATURE_LABELS[feat]}")
                    st.dataframe(pd.DataFrame(model["likelihoods"][feat]).round(4), width="stretch")
        else:
            rules = extract_c45_rules(model["tree"])
            html_rules = "".join(f'<div class="rule-item">{i}. {r}</div>' for i, r in enumerate(rules, 1))
            st.markdown(f'<div class="plain-box">{html_rules}</div>', unsafe_allow_html=True)


def render_saw(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-head">Urutan prioritas restock (SAW)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Pilih skenario pembobotan atau atur bobot sendiri untuk menghitung nilai preferensi V.</div>',
        unsafe_allow_html=True,
    )

    preset = st.segmented_control(
        "Skenario bobot",
        list(WEIGHT_PRESETS) + ["Atur sendiri"],
        default="Seimbang",
    )
    preset = preset or "Seimbang"

    weights: dict[str, float] = {}
    types: dict[str, str] = {k: v["type"] for k, v in SAW_CRITERIA_DEFAULT.items()}

    if preset == "Atur sendiri":
        with st.container(border=True):
            cols = st.columns(len(SAW_CRITERIA_DEFAULT))
            for idx, (key, cfg) in enumerate(SAW_CRITERIA_DEFAULT.items()):
                with cols[idx]:
                    weights[key] = (
                        st.slider(
                            cfg["label"], 0, 100, int(cfg["weight"] * 100), 5,
                            format="%d%%", help=CRITERIA_HELP[key], key=f"w_{key}",
                        )
                        / 100
                    )
                    types[key] = st.radio(
                        "Sifat",
                        ["Benefit", "Cost"],
                        horizontal=True,
                        index=0 if cfg["type"] == "Benefit" else 1,
                        key=f"t_{key}",
                        label_visibility="collapsed",
                    )
            total = sum(weights.values())
            if total == 0:
                st.error("Minimal satu kriteria harus memiliki bobot lebih dari 0%.")
                return
            if abs(total - 1) > 0.001:
                st.caption(f"Total bobot {total * 100:.0f}% otomatis dinormalisasi ke 100%.")
    else:
        weights = WEIGHT_PRESETS[preset]
        st.caption(
            "Bobot aktif: "
            + " · ".join(f"{SAW_CRITERIA_DEFAULT[k]['label']} {v * 100:.0f}% ({types[k]})" for k, v in weights.items())
        )

    decision_df, norm_df, result_df = calculate_saw(df, weights, types)
    result_df = result_df.merge(df[["kode", "sisa_hari_stok", "status_stok"]], on="kode", how="left")

    st.write("")
    top3 = result_df.head(3)
    t_cols = st.columns(3)
    for idx, (_, row) in enumerate(top3.iterrows()):
        with t_cols[idx]:
            plain_box(
                f"Prioritas 0{int(row['peringkat'])}",
                row["nama_barang"],
                f"Nilai V {row['nilai_preferensi']:.4f} · Stok {int(row['stok_saat_ini'])} · Estimasi {row['sisa_hari_stok']:.0f} hari",
            )

    st.write("")
    st.dataframe(
        result_df[
            [
                "peringkat", "nama_barang", "kategori", "stok_saat_ini",
                "rata_jual_mingguan", "status_stok", "nilai_preferensi",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "peringkat": st.column_config.NumberColumn("Rank", width="small"),
            "nama_barang": st.column_config.TextColumn("Nama menu", width="medium"),
            "kategori": "Kategori",
            "stok_saat_ini": st.column_config.NumberColumn("Stok", format="%d"),
            "rata_jual_mingguan": st.column_config.NumberColumn("Jual / minggu", format="%.1f"),
            "status_stok": "Status",
            "nilai_preferensi": st.column_config.ProgressColumn(
                "Nilai preferensi (V)", format="%.4f", min_value=0.0, max_value=1.0
            ),
        },
    )

    st.download_button(
        "Unduh hasil peringkat (CSV)",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="hasil_prioritas_restock_saw.csv",
        mime="text/csv",
    )

    with st.expander("Matriks keputusan (X) dan normalisasi (R)"):
        st.caption("Matriks Keputusan (X)")
        st.dataframe(decision_df, width="stretch", hide_index=True)
        st.caption("Matriks Normalisasi (R)")
        st.dataframe(norm_df, width="stretch", hide_index=True)


def render_formulas() -> None:
    st.markdown('<div class="section-head">Referensi rumus perhitungan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Ringkasan persamaan yang digunakan pada sistem untuk keperluan laporan.</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("**01 / Metode SAW**")
            st.caption("Normalisasi dan nilai preferensi")
            st.latex(r"r_{ij} = \frac{x_{ij}}{\max_i x_{ij}} \quad (\text{Benefit})")
            st.latex(r"r_{ij} = \frac{\min_i x_{ij}}{x_{ij}} \quad (\text{Cost})")
            st.latex(r"V_i = \sum_{j} w_j \, r_{ij}")
    with c2:
        with st.container(border=True):
            st.markdown("**02 / Naive Bayes**")
            st.caption("Klasifikasi probabilistik dengan Laplace smoothing")
            st.latex(r"P(C \mid X) \propto P(C) \prod_i P(x_i \mid C)")
            st.latex(r"P(x_i \mid C) = \frac{\text{count}(x_i, C) + 1}{\text{count}(C) + |V_i|}")
    with c3:
        with st.container(border=True):
            st.markdown("**03 / Algoritma C.45**")
            st.caption("Pemilihan atribut berbasis Gain Ratio")
            st.latex(r"\text{Entropy}(S) = -\sum p_i \log_2 p_i")
            st.latex(r"\text{GainRatio}(S, A) = \frac{\text{Gain}(S, A)}{\text{SplitInfo}(S, A)}")


if __name__ == "__main__":
    main()
