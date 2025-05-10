import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_echarts import st_echarts
from tensorflow.keras.models import load_model
from pandas.tseries.offsets import DateOffset

# Load model dan scaler
model = load_model('best_model.keras')
scaler_x = joblib.load('scaler_x.save')
scaler_y = joblib.load('scaler_y.save')

st.markdown("""
<div style='text-align: center;'>
    <h1>🐔 GROWCHICK</h1>
    <h3 style='font-size: 20px;'>Chicken Mortality Prediction</h3>
</div>
""", unsafe_allow_html=True)

# Fungsi prediksi yang dioptimalkan
# Fungsi prediksi yang dioptimalkan
def make_predictions(df, n_months=3):
    feature_cols = ['Suhu/Temperature (0C)', 'Kelembapan (%)',
                   'Kecepatan Angin (m/det)', 'Curah Hujan (mm)', 'Penyinaran Matahari (jam)']
    
    latest_window = df[feature_cols].iloc[-3:]
    
    if latest_window.shape[0] < 3:
        raise ValueError("Data tidak cukup. Minimal 3 bulan untuk prediksi.")
    
    scaled_input = scaler_x.transform(latest_window)

    x_input = scaled_input.reshape(1, 3, len(feature_cols))
    
    predictions = []
    current_input = x_input.copy()
    
    for _ in range(n_months):
        pred_scaled = model.predict(current_input, verbose=0)
        predictions.append(pred_scaled[0, 0])
        
        # Update input untuk prediksi berikutnya
        new_input = np.roll(current_input, -1, axis=1)
        new_input[0, -1, :] = current_input[0, -1, :]  # Pertahankan fitur lain
        current_input = new_input
    
    pred_array = np.array(predictions).reshape(-1, 1)
    pred_inverse = scaler_y.inverse_transform(pred_array)[:, 0]
    
    # Pembulatan sesuai aturan: nilai 0.5 dibulatkan ke bawah, di atas 0.5 dibulatkan ke atas
    pred_final = np.floor(pred_inverse + 0.5).astype(int)  # Pembulatan custom
    
    return pred_final.tolist()


# Data dummy dengan nilai 0 semua (untuk ketika tidak ada file diunggah)
dummy_data = {
    'Bulan': pd.date_range(start='2025-01-01', periods=6, freq='M'),  # 6 bulan data dummy
    'Jumlah Ayam Mati': [0, 0, 0, 0, 0, 0],
    'Suhu/Temperature (0C)': [0, 0, 0, 0, 0, 0],
    'Kelembapan (%)': [0, 0, 0, 0, 0, 0],
    'Kecepatan Angin (m/det)': [0, 0, 0, 0, 0, 0],
    'Curah Hujan (mm)': [0, 0, 0, 0, 0, 0],
    'Penyinaran Matahari (jam)': [0, 0, 0, 0, 0, 0],
}
df = pd.DataFrame(dummy_data)

# Upload file
uploaded_file = st.file_uploader("📂 Upload Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, header=0)
        df.columns = df.columns.str.strip()
        df = df.fillna(0)
        
        required_columns = ['Bulan', 'Jumlah Ayam Mati', 'Suhu/Temperature (0C)',
                          'Kelembapan (%)', 'Kecepatan Angin (m/det)', 'Curah Hujan (mm)',
                          'Penyinaran Matahari (jam)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Kolom yang diperlukan tidak ada: {', '.join(missing_columns)}")
            st.stop()
        
        df['Bulan'] = pd.to_datetime(df['Bulan'], errors='coerce')
    except Exception as e:
        st.error(f"Error membaca file: {str(e)}")
        st.stop()

# Tampilkan data setelah file diunggah atau data dummy
st.subheader("📊 Data Peternakan")
st.write(df)

# Prediksi hanya jika data cukup
if len(df) >= 3:
    try:
        # Gunakan data dummy dengan prediksi 0 jika data dummy digunakan
        if df['Jumlah Ayam Mati'].eq(0).all():
            predictions = [0, 0, 0]  # Jika data dummy, prediksi semua bulan adalah 0
        else:
            predictions = make_predictions(df)
        
        bulan = pd.to_datetime(df['Bulan'])
        bulan_prediksi = [bulan.iloc[-1] + DateOffset(months=i) for i in range(1, 4)]
        
        # Siapkan data untuk grafik
        all_months = bulan.tolist() + bulan_prediksi
        actual_values = df['Jumlah Ayam Mati'].tolist()
        pred_values = predictions
        
        options = {
            "title": {
                "text": "Prediksi Kematian Ayam per Bulan",
                "left": "center",
                "textStyle": {"fontSize": 20}
            },
            "tooltip": {"trigger": "axis"},
            "legend": {
                "data": ["Data Aktual", "Prediksi"],
                "top": 30
            },
            "xAxis": {
                "type": "category",
                "data": [d.strftime('%b %Y') for d in all_months],
                "axisLabel": {"rotate": 45, "interval": 0}
            },
            "yAxis": {
                "type": "value",
                "name": "Jumlah Ayam Mati",
                "min": 0,
                "max": max(actual_values + pred_values + [10]) or 10,
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
                    "data": [None]*(len(actual_values)) + pred_values,
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
        
        # Rekomendasi obat-obatan dalam format teks
        st.markdown("📋 Prediksi Tingkat Kematian Ayam 3 Bulan Berikutnya")
        
        bulan_prediksi_str = [d.strftime('%B %Y') for d in bulan_prediksi]
        
        # Buat tabel rekomendasi
        recommendation_table = pd.DataFrame({
            'Bulan': bulan_prediksi_str,
            'Prediksi': [f"{int(p)} ekor" if p % 1 == 0 else f"{p:.2f} ekor" for p in predictions],
            'Vaksin': [
                "AI H5N1, ND Lasota, ND IB, Vitamin" if i == 0 else "ND Lasota, ND IB, Vitamin" 
                for i in range(3)
            ]
        })
        
        st.table(recommendation_table)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam memproses data: {str(e)}")
else:
    st.warning("⚠️ Data tidak cukup untuk memberikan rekomendasi. Diperlukan minimal 3 bulan data historis.")
