import streamlit as st

# === Atur tampilan jadi wide ===
st.set_page_config(page_title="AgroForecast", layout="wide")

# === Judul Aplikasi ===
st.markdown("<h1 style='text-align:center;'>AGROFORECAST</h1>", unsafe_allow_html=True)

# === Kalender Bulanan dengan Kategori Warna ===
st.markdown("### Kalender Musim Tanam (Basah - Lembab - Kering)")

months = [
    "Januari","Februari","Maret","April","Mei","Juni",
    "Juli","Agustus","September","Oktober","November","Desember"
]

# Warna kategori (contoh: biru=basah, hijau=lembab, kuning=kering)
colors = [
    "#42A5F5", "#42A5F5", "#42A5F5",   # Basah (biru)
    "#81C784", "#81C784",              # Lembab (hijau)
    "#FFD54F", "#FFD54F", "#FFD54F", "#FFD54F", # Kering (kuning)
    "#81C784",                         # Lembab (hijau)
    "#42A5F5", "#42A5F5"               # Basah (biru)
]

# Tampilkan kotak bulan
cols = st.columns(12, gap="small")
for i, month in enumerate(months):
    with cols[i]:
        st.markdown(
            f"""
            <div style='background-color:{colors[i]};
                        padding:20px;
                        text-align:center;
                        border-radius:8px;
                        color:black;
                        font-weight:bold;
                        font-size:14px;
                        min-height:60px;'>
                {month}
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

# === Input Data Curah Hujan & Lahan ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Curah Hujan")
    st.caption("Pastikan data curah hujan dalam bentuk Excel")
    uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx", "xls"])

with col2:
    st.subheader("Luas Lahan")
    st.caption("Masukkan luas lahan anda dalam hektar")
    luas_lahan = st.number_input("Input luas lahan (Ha)", min_value=0.0, step=0.1)

st.markdown("---")

# === Tombol Hasil Periode Tanam ===
col3, col4 = st.columns(2)

with col3:
    st.subheader("Padi")
    if st.button("Hasil periode tanam - Padi", use_container_width=True):
        st.success("Periode tanam padi berhasil dihitung!")

with col4:
    st.subheader("Palawija")
    if st.button("Hasil periode tanam - Palawija", use_container_width=True):
        st.success("Periode tanam palawija berhasil dihitung!")

st.markdown("---")

# === Rekomendasi Subsidi ===
st.markdown(
    "<h3 style='text-align:center;'>Rekomendasi subsidi bibit yang dapat diberikan : 12 ton</h3>",
    unsafe_allow_html=True
)
