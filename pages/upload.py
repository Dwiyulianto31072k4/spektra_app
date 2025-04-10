import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import re

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

                    # Tampilkan informasi kolom Usia jika ada
                    if 'Usia' in processed_data.columns:
                        st.markdown("### Age Information")
                        
                        # Cek keberadaan nilai
                        valid_ages = processed_data['Usia'].notna().sum()
                        total_rows = len(processed_data)
                        st.write(f"Valid age values: {valid_ages} out of {total_rows} rows ({valid_ages/total_rows*100:.1f}%)")
                        
                        # Tampilkan statistik jika ada nilai valid
                        if valid_ages > 0:
                            age_stats = processed_data['Usia'].describe()
                            st.write(f"Age statistics: Min={age_stats['min']:.1f}, Max={age_stats['max']:.1f}, Mean={age_stats['mean']:.1f}")
                            
                            # Tampilkan distribusi kategori usia
                            if 'Usia_Kategori' in processed_data.columns:
                                st.write("Age category distribution:")
                                st.write(processed_data['Usia_Kategori'].value_counts())
                    else:
                        st.warning("Age calculation failed. Age-based analysis will not be available.")

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
        st.write("Or use our example data to explore the application:")
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

def clean_date_string(date_string):
    """
    Membersihkan dan menstandardisasi string tanggal
    
    Parameters:
    -----------
    date_string : str
        String tanggal yang akan dibersihkan
    
    Returns:
    --------
    str
        String tanggal yang sudah dibersihkan
    """
    if pd.isna(date_string) or date_string is None or date_string == '':
        return date_string
    
    # Hapus waktu (00.00.00 atau time component)
    date_string = str(date_string)
    date_string = re.sub(r'\s*\d{1,2}[:.]\d{1,2}[:.]\d{1,2}.*$', '', date_string)
    date_string = re.sub(r'\s+00\.00\.00.*$', '', date_string)
    
    # Bersihkan character tidak standar
    date_string = date_string.replace('/', '-').replace('.', '-')
    
    return date_string.strip()

def preprocess_data(data, date_cols):
    """
    Fungsi untuk memproses data sebelum analisis dengan penanganan khusus untuk BIRTH_DATE
    
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
    
    # Konversi kolom tanggal
    for col in date_cols:
        if col in processed_data.columns:
            # Simpan nilai original untuk debugging
            original_values = processed_data[col].copy()
            
            # Bersihkan nilai tanggal
            processed_data[col] = processed_data[col].apply(clean_date_string)
            
            # Coba beberapa format tanggal umum
            date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d', '%d/%m/%Y']
            converted = False
            
            # Coba semua format
            for fmt in date_formats:
                try:
                    temp_dates = pd.to_datetime(processed_data[col], format=fmt, errors='coerce')
                    # Hitung keberhasilan konversi
                    success_rate = temp_dates.notna().sum() / len(processed_data)
                    
                    # Jika >30% berhasil, gunakan format ini
                    if success_rate > 0.3:
                        processed_data[col] = temp_dates
                        st.write(f"Converted {col} using format {fmt}: {temp_dates.notna().sum()} valid dates ({success_rate:.1%})")
                        converted = True
                        break
                except Exception as e:
                    continue
            
            # Jika format spesifik gagal, gunakan pandas automatic detection
            if not converted:
                processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
                success_rate = processed_data[col].notna().sum() / len(processed_data)
                st.write(f"Converted {col} using pandas automatic detection: {processed_data[col].notna().sum()} valid dates ({success_rate:.1%})")
    
    # Khusus untuk BIRTH_DATE, gunakan pendekatan yang lebih agresif jika masih gagal
    if 'BIRTH_DATE' in processed_data.columns:
        # Cek keberhasilan konversi
        if processed_data['BIRTH_DATE'].notna().sum() == 0:
            st.warning("All BIRTH_DATE values failed to convert. Trying more aggressive approach...")
            
            # Tampilkan sampel untuk debugging
            st.write("Sample BIRTH_DATE values before cleanup:")
            st.write(data['BIRTH_DATE'].dropna().head(5).tolist())
            
            # Coba pendekatan ekstraksi komponen tanggal dengan regex
            def extract_date_components(date_str):
                if pd.isna(date_str):
                    return None
                
                date_str = str(date_str).strip()
                
                # Coba ekstrak komponen dengan regex
                # Format MM/DD/YYYY atau DD/MM/YYYY
                match = re.search(r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4}|\d{2})', date_str)
                if match:
                    day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    # Fix tahun 2 digit
                    if year < 100:
                        year = year + 1900 if year >= 50 else year + 2000
                    
                    # Validasi tanggal
                    if month > 12:  # Kemungkinan format MM/DD/YYYY
                        month, day = day, month
                    
                    try:
                        return pd.Timestamp(year=year, month=month, day=day)
                    except:
                        return None
                
                # Format YYYY-MM-DD
                match = re.search(r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', date_str)
                if match:
                    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    try:
                        return pd.Timestamp(year=year, month=month, day=day)
                    except:
                        return None
                
                return None
            
            # Konversi menggunakan fungsi ekstraksi komponen
            processed_data['BIRTH_DATE'] = processed_data['BIRTH_DATE'].apply(extract_date_components)
            
            # Cek keberhasilan
            success_rate = processed_data['BIRTH_DATE'].notna().sum() / len(processed_data)
            st.write(f"Extracted dates with component extraction: {processed_data['BIRTH_DATE'].notna().sum()} valid dates ({success_rate:.1%})")
        
        # Jika sudah berhasil konversi, hitung usia
        if processed_data['BIRTH_DATE'].notna().sum() > 0:
            st.write("Calculating age from birth dates...")
            
            # Hitung usia
            current_date = pd.Timestamp.now()
            processed_data['Usia'] = ((current_date - processed_data['BIRTH_DATE']).dt.days / 365.25)
            
            # Cek dan filter untuk usia yang valid
            processed_data['Usia'] = processed_data['Usia'].round()
            valid_age = (processed_data['Usia'] >= 18) & (processed_data['Usia'] <= 100)
            processed_data.loc[~valid_age, 'Usia'] = np.nan
            
            # Konversi ke tipe numerik
            processed_data['Usia'] = pd.to_numeric(processed_data['Usia'], errors='coerce')
            
            # Tampilkan statistik usia
            if processed_data['Usia'].notna().sum() > 0:
                st.success(f"Successfully calculated age for {processed_data['Usia'].notna().sum()} customers")
                st.write(f"Age statistics: Min={processed_data['Usia'].min():.1f}, Max={processed_data['Usia'].max():.1f}, Mean={processed_data['Usia'].mean():.1f}")
                
                # Buat kategori usia
                bins = [0, 25, 35, 45, 55, 100]
                labels = ['<25', '25-35', '35-45', '45-55', '55+']
                processed_data['Usia_Kategori'] = pd.cut(processed_data['Usia'], bins=bins, labels=labels, right=False)
                processed_data['Usia_Kategori'] = processed_data['Usia_Kategori'].astype(str)
                processed_data.loc[processed_data['Usia_Kategori'] == 'nan', 'Usia_Kategori'] = 'Unknown'
                
                # Tampilkan distribusi kategori
                st.write("Age category distribution:")
                st.write(processed_data['Usia_Kategori'].value_counts())
            else:
                st.warning("No valid ages could be calculated (between 18-100 years)")
        else:
            st.warning("No valid birth dates could be converted")
    else:
        st.warning("BIRTH_DATE column not found in the data")
    
    # Konversi kolom numerik
    numeric_cols = ['TOTAL_AMOUNT_MPF', 'TOTAL_PRODUCT_MPF', 'MAX_MPF_AMOUNT', 'MIN_MPF_AMOUNT', 
                  'LAST_MPF_AMOUNT', 'LAST_MPF_INST', 'LAST_MPF_TOP', 'AVG_MPF_INST',
                  'PRINCIPAL', 'GRS_DP', 'JMH_CON_SBLM_MPF', 'JMH_PPC']
    
    for col in numeric_cols:
        if col in processed_data.columns:
            # Bersihkan data terlebih dahulu
            if processed_data[col].dtype == 'object':
                processed_data[col] = processed_data[col].astype(str).str.replace(',', '').str.replace(r'[^\d.]', '', regex=True)
            
            # Konversi ke numerik
            original_count = len(processed_data)
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            null_count = processed_data[col].isnull().sum()
            
            # Isi nilai yang hilang
            if null_count > 0 and null_count < len(processed_data):
                median_val = processed_data[col].median()
                processed_data[col] = processed_data[col].fillna(median_val)
    
    # Tambahkan fitur tambahan
    processed_data['PROCESSING_DATE'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "TOTAL_PRODUCT_MPF" in processed_data.columns:
        processed_data["Multi-Transaction_Customer"] = processed_data["TOTAL_PRODUCT_MPF"].astype(float).apply(lambda x: 1 if x > 1 else 0)
    
    return processed_data
