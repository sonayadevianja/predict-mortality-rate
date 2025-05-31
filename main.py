import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from streamlit_echarts import st_echarts
from tensorflow.keras.models import load_model
from pandas.tseries.offsets import DateOffset

# Load scaler dan model
scaler_x = joblib.load('scaler_x.save')
scaler_y = joblib.load('scaler_y.save')
final_model = load_model('best_model.keras')

# Fungsi prediksi
def make_predictions(df, n_months=3):
    feature_cols = ['Suhu/Temperature (0C)', 'Kelembapan (%)',
                    'Kecepatan Angin (m/det)', 'Curah Hujan (mm)', 
                    'Penyinaran Matahari (jam)']
    window_size = 3
    if len(df) < window_size:
        raise ValueError(f"Minimal {window_size} bulan data historis diperlukan.")

    last_window = df.iloc[-window_size:][feature_cols].copy()
    last_window = last_window.fillna(last_window.mean())

    scaled_features = scaler_x.transform(last_window)
    current_window = scaled_features.reshape(1, window_size, len(feature_cols))

    predictions = []
    for _ in range(n_months):
        pred_scaled = final_model.predict(current_window, verbose=0)[0,0]
        predictions.append(pred_scaled)
        next_step = current_window[0, -1].copy()
        current_window = np.roll(current_window, -1, axis=1)
        current_window[0, -1] = next_step

    pred_array = np.array(predictions).reshape(-1,1)
    pred_inverse = scaler_y.inverse_transform(pred_array)[:,0]
    pred_rounded = np.round(pred_inverse).astype(int)

    last_date = pd.to_datetime(df['Waktu'].iloc[-1])
    prediction_months = [last_date + DateOffset(months=i) for i in range(1, n_months+1)]

    return prediction_months, pred_rounded

# Judul aplikasi
st.markdown("""
<div style='text-align: center;'>
    <h1>🐔 GROWCHICK</h1>
    <h3 style='font-size: 20px;'>Chicken Mortality Prediction</h3>
</div>
""", unsafe_allow_html=True)

# Data dummy sebelum upload file
temporary_data = {
    'Waktu': pd.date_range(start='2025-01-01', periods=6, freq='M'),
    'Jumlah Ayam Mati': [0]*6,
    'Suhu/Temperature (0C)': [0]*6,
    'Kelembapan (%)': [0]*6,
    'Kecepatan Angin (m/det)': [0]*6,
    'Curah Hujan (mm)': [0]*6,
    'Penyinaran Matahari (jam)': [0]*6,
}
df = pd.DataFrame(temporary_data)

# Upload file
uploaded_file = st.file_uploader("📂 Upload Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, header=0)
        df.columns = df.columns.str.strip()
        df = df.fillna(df.mean())

        required_columns = ['Waktu', 'Jumlah Ayam Mati', 'Suhu/Temperature (0C)',
                            'Kelembapan (%)', 'Kecepatan Angin (m/det)', 'Curah Hujan (mm)',
                            'Penyinaran Matahari (jam)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Kolom yang diperlukan tidak ada: {', '.join(missing_columns)}")
            st.stop()

        df['Waktu'] = pd.to_datetime(df['Waktu'], errors='coerce')
    except Exception as e:
        st.error(f"Error membaca file: {str(e)}")
        st.stop()

# Tombol download template
template_path = "DATA ASLI_GANTIAVERAGE.xlsx"
if os.path.exists(template_path):
    with open(template_path, "rb") as file:
        st.download_button(
            label="📅 Unduh Template Excel",
            data=file,
            file_name="DATA ASLI_GANTIAVERAGE.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.warning("⚠️ File template Excel tidak ditemukan. Pastikan file 'DATA ASLI_GANTIAVERAGE.xlsx' ada di folder proyek.")

# Tampilkan data
st.subheader("📊 Data Peternakan")
st.write(df)

# Prediksi dan visualisasi
if len(df) >= 3:
    try:
        if df['Jumlah Ayam Mati'].eq(0).all():
            predictions = [0, 0, 0]
            prediction_months = [df['Waktu'].iloc[-1] + DateOffset(months=i) for i in range(1, 4)]
        else:
            prediction_months, predictions = make_predictions(df)

        bulan = pd.to_datetime(df['Waktu'])
        bulan_prediksi = prediction_months

        all_months = bulan.tolist() + bulan_prediksi
        actual_values = df['Jumlah Ayam Mati'].tolist()

        pred_values = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions

        options = {
            "title": {
                "text": "Prediksi Kematian Ayam per Bulan",
                "left": "center",
                "textStyle": {"fontSize": 20}
            },
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Data Aktual", "Prediksi"], "top": 30},
            "xAxis": {
                "type": "category",
                "data": [d.strftime('%b %Y') for d in all_months],
                "axisLabel": {"rotate": 45, "interval": 0}
            },
            "yAxis": {
                "type": "value",
                "name": "Jumlah Ayam Mati",
                "min": 0,
                "max": max(actual_values + pred_values + [10]),
                "interval": 2
            },
            "series": [
                {
                    "name": "Data Aktual",
                    "type": "line",
                    "data": actual_values + [None]*3,
                    "itemStyle": {"color": "#3498db"},
                    "lineStyle": {"width": 3},
                    "symbol": "circle",
                    "symbolSize": 8
                },
                {
                    "name": "Prediksi",
                    "type": "line",
                    "data": [None]*len(actual_values) + pred_values,
                    "itemStyle": {"color": "#e74c3c"},
                    "lineStyle": {"width": 3},
                    "symbol": "circle",
                    "symbolSize": 8
                },
                {
                    "name": "Garis Penghubung",
                    "type": "line",
                    "data": [None]*(len(actual_values)-1) + [actual_values[-1], pred_values[0]] + [None]*2,
                    "itemStyle": {"color": "#3498db"},
                    "lineStyle": {"width": 3},
                    "symbol": "none",
                    "showSymbol": False
                }
            ],
            "grid": {"containLabel": True}
        }

        st_echarts(options=options, height="500px")

        # ======= Tabel Rekomendasi =======
        st.markdown("📋 Prediksi Tingkat Kematian Ayam 3 Bulan Berikutnya")

        if uploaded_file is not None and len(df) >= 3:
            bulan_prediksi_str = [d.strftime('%B %Y') for d in bulan_prediksi]

            rec_df = pd.DataFrame({
                'Bulan': bulan_prediksi_str,
                'Prediksi': pred_values
            })

            def generate_rekomendasi(pred, month_index):
                base = "AI H5N1, ND Lasota, ND IB" if pred > 10 else "ND Lasota, ND IB"
                if month_index % 3 == 0:
                    base += ", Vitamin"
                return base

            rec_df['Rekomendasi'] = [
                generate_rekomendasi(pred, idx) for idx, pred in enumerate(rec_df['Prediksi'])
            ]

            rec_df['Prediksi'] = rec_df['Prediksi'].apply(lambda x: f"{int(x)} ekor")
        else:
            rec_df = pd.DataFrame({
                'Bulan': [np.nan]*3,
                'Prediksi': [np.nan]*3,
                'Rekomendasi': [np.nan]*3
            })

        st.dataframe(rec_df)

    except Exception as e:
        st.error(f"Prediksi gagal: {str(e)}")
else:
    st.warning("⚠️ Data tidak cukup untuk memberikan rekomendasi. Diperlukan minimal 3 bulan data historis.")
