import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os

from utils.data_utils import create_example_data

def show_upload_page():
    """
    Fungsi untuk menampilkan halaman upload dan preprocessing
    """
    st.markdown('<p class="section-title">Upload Customer Data</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Excel file with customer data", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        st.session_state.uploaded_file_name = uploaded_file.name

        try:
            data = pd.read_excel(uploaded_file, dtype=str)  # Baca semua kolom sebagai string (aman untuk tanggal)

            st.success(f"File '{uploaded_file.name}' successfully loaded with {data.shape[0]} rows and {data.shape[1]} columns!")
            st.markdown('<p class="section-title">Data Preview</p>', unsafe_allow_html=True)
            st.dataframe(data.head())

            # Deteksi otomatis kolom dengan 'DATE' di namanya
            date_cols = [col for col in data.columns if 'DATE' in col.upper()]
            st.markdown(f"🔍 Auto-detected date columns: `{', '.join(date_cols)}`")

            if st.button("Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    processed_data = preprocess_data(data, date_cols)
                    
                    st.success("✅ Data preprocessing completed!")
                    st.dataframe(processed_data.head())

                    st.session_state.data = processed_data
                    
                    # Pastikan folder temp ada
                    if not os.path.exists("temp"):
                        os.makedirs("temp")
                        
                    processed_data.to_excel("temp/processed_data.xlsx", index=False)
                    st.session_state.eda_completed = True

                    st.markdown("### Next Steps")
                    st.info("You can now proceed to the Exploratory Data Analysis section.")

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.warning("Please check your file and try again.")
    
    else:
        if st.button("Use Example Data"):
            example_data = create_example_data()
            st.success("✅ Example data loaded successfully!")
            st.session_state.data = example_data
            st.session_state.uploaded_file_name = "example_data.xlsx"
            
            # Pastikan folder temp ada
            if not os.path.exists("temp"):
                os.makedirs("temp")
                
            example_data.to_excel("temp/processed_data.xlsx", index=False)
            st.dataframe(example_data.head())
            st.session_state.eda_completed = True
            st.markdown("### Next Steps")
            st.info("You can now proceed to the Exploratory Data Analysis section.")

def preprocess_data(data, date_cols):
    """
    Fungsi untuk memproses data sebelum analisis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data mentah yang akan diproses
    date_cols : list
        Daftar kolom yang berisi tanggal
    
    Returns:
    --------
    pandas.DataFrame
        Data yang telah diproses
    """
    processed_data = data.copy()

    # Konversi tanggal dari format YYYYMMDD
    for col in date_cols:
        processed_data[col] = (
            processed_data[col]
            .astype(str)
            .str.replace(r"[^\d]", "", regex=True)  # Hapus koma/titik
            .apply(lambda x: pd.to_datetime(x, format="%Y%m%d", errors="coerce"))
        )

    # Hitung usia jika ada kolom BIRTH_DATE
    if 'BIRTH_DATE' in processed_data.columns:
        # Hitung usia sebagai nilai numerik (penting untuk analisis distribusi)
        processed_data['Usia'] = datetime.datetime.now().year - processed_data['BIRTH_DATE'].dt.year
        
        # Buat kategori usia (untuk visualisasi dan segmentasi)
        bins = [0, 25, 35, 45, 55, 100]
        labels = ['<25', '25-35', '35-45', '45-55', '55+']
        processed_data['Usia_Kategori'] = pd.cut(processed_data['Usia'], bins=bins, labels=labels, right=False)

    # Handle missing values (default: median/mode)
    # Konversi kolom numerik ke tipe numerik
    numeric_cols = ['TOTAL_AMOUNT_MPF', 'TOTAL_PRODUCT_MPF', 'MAX_MPF_AMOUNT', 'MIN_MPF_AMOUNT', 
                    'LAST_MPF_AMOUNT', 'LAST_MPF_INST', 'LAST_MPF_TOP', 'AVG_MPF_INST',
                    'PRINCIPAL', 'GRS_DP', 'JMH_CON_SBLM_MPF', 'JMH_PPC']
    
    for col in numeric_cols:
        if col in processed_data.columns:
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
    
    # Identifikasi kolom numerik dan kategorikal
    numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = processed_data.select_dtypes(include=['object']).columns

    # Handle missing values
    for col in numeric_cols:
        if processed_data[col].isnull().sum() > 0:
            processed_data[col].fillna(processed_data[col].median(), inplace=True)

    for col in categorical_cols:
        if processed_data[col].isnull().sum() > 0:
            processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)

    # Hapus kolom yang tidak diperlukan jika ada
    if 'JMH_CON_NON_MPF' in processed_data.columns:
        processed_data.drop(columns=['JMH_CON_NON_MPF'], inplace=True)

    # Tambahkan fitur tambahan
    processed_data['PROCESSING_DATE'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "TOTAL_PRODUCT_MPF" in processed_data.columns:
        processed_data["Multi-Transaction_Customer"] = processed_data["TOTAL_PRODUCT_MPF"].astype(int).apply(lambda x: 1 if x > 1 else 0)

    return processed_data
