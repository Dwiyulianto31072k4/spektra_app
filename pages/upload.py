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
            
            # Tambahkan opsi untuk memilih kolom tanggal lahir jika tidak terdeteksi otomatis
            if 'BIRTH_DATE' not in date_cols:
                st.warning("BIRTH_DATE column not auto-detected. Please select the birth date column manually.")
                birth_date_col = st.selectbox(
                    "Select birth date column (if available)", 
                    [None] + list(data.columns),
                    index=0
                )
                if birth_date_col:
                    date_cols.append(birth_date_col)
                    st.info(f"Added {birth_date_col} to date columns for processing.")

            # Tombol untuk melihat diagnostik data
            if st.checkbox("Show data diagnostics"):
                show_data_diagnostics(data, date_cols)

            if st.button("Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    processed_data = preprocess_data(data, date_cols)
                    
                    st.success("✅ Data preprocessing completed!")
                    st.dataframe(processed_data.head())

                    # Tampilkan informasi tentang kolom Usia
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
                            st.error("No valid age values were calculated. Customer age analysis will not be available.")
                            st.info("Possible causes: Missing birth date data, invalid date formats, or dates in the future.")

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

def show_data_diagnostics(data, date_cols):
    """
    Menampilkan diagnostik data untuk membantu debugging
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data yang akan didiagnosis
    date_cols : list
        Daftar kolom yang berisi tanggal
    """
    st.markdown("### Data Diagnostics")
    
    # Cek kolom tanggal
    st.subheader("Date Column Samples")
    for col in date_cols:
        st.write(f"**{col}** - first 5 values:")
        st.write(data[col].head())
    
    # Cek kolom BIRTH_DATE secara khusus
    if 'BIRTH_DATE' in data.columns:
        st.subheader("BIRTH_DATE Analysis")
        st.write("Sample values (first 5):")
        st.write(data['BIRTH_DATE'].head())
        
        # Cek apakah format tanggal dapat dideteksi
        st.write("Attempting date conversion...")
        try:
            # Coba beberapa format umum
            formats_to_try = ['%Y%m%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d']
            conversion_results = {}
            
            for fmt in formats_to_try:
                try:
                    converted = pd.to_datetime(data['BIRTH_DATE'], format=fmt, errors='coerce')
                    valid_count = converted.notna().sum()
                    conversion_results[fmt] = valid_count
                    st.write(f"Format {fmt}: {valid_count} valid dates ({valid_count/len(data)*100:.1f}%)")
                except Exception as e:
                    st.write(f"Format {fmt}: Failed - {str(e)}")
            
            # Coba dengan pandas inference juga
            try:
                converted = pd.to_datetime(data['BIRTH_DATE'], errors='coerce')
                valid_count = converted.notna().sum()
                conversion_results['pandas_inference'] = valid_count
                st.write(f"Pandas inference: {valid_count} valid dates ({valid_count/len(data)*100:.1f}%)")
            except Exception as e:
                st.write(f"Pandas inference: Failed - {str(e)}")
            
            # Rekomendasikan format terbaik
            if conversion_results:
                best_format = max(conversion_results, key=conversion_results.get)
                st.write(f"**Recommended format: {best_format}** with {conversion_results[best_format]} valid dates")
        
        except Exception as e:
            st.error(f"Error during date analysis: {e}")
    else:
        st.warning("BIRTH_DATE column not found in the data. Age calculation will not be possible.")
    
    # Cek kolom numerik
    st.subheader("Numeric Column Analysis")
    numeric_cols = ['TOTAL_AMOUNT_MPF', 'TOTAL_PRODUCT_MPF', 'MAX_MPF_AMOUNT', 'MIN_MPF_AMOUNT']
    for col in numeric_cols:
        if col in data.columns:
            st.write(f"**{col}** - sample values:")
            st.write(data[col].head())
            
            # Mencoba konversi ke numerik
            try:
                numeric_data = pd.to_numeric(data[col], errors='coerce')
                valid_count = numeric_data.notna().sum()
                st.write(f"Numeric conversion: {valid_count} valid values ({valid_count/len(data)*100:.1f}%)")
                if valid_count > 0:
                    st.write(f"Statistics: Min={numeric_data.min()}, Max={numeric_data.max()}, Mean={numeric_data.mean():.2f}")
            except Exception as e:
                st.error(f"Error in numeric conversion: {e}")

def preprocess_data(data, date_cols):
    """
    Fungsi yang diperbaiki untuk memproses data sebelum analisis
    
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
    preprocessing_steps = []  # Untuk mencatat langkah-langkah preprocessing
    
    # Konversi kolom tanggal dengan penanganan error yang lebih baik
    for col in date_cols:
        if col in processed_data.columns:
            preprocessing_steps.append(f"Converting column {col} to date format")
            
            # Tampilkan beberapa nilai sampel untuk diagnostik
            sample_values = processed_data[col].head(3).tolist()
            preprocessing_steps.append(f"Sample values for {col}: {sample_values}")
            
            # Coba beberapa format umum untuk tanggal secara berurutan
            formats_to_try = ['%Y%m%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d']
            conversion_success = False
            
            for fmt in formats_to_try:
                try:
                    processed_data[col] = pd.to_datetime(processed_data[col], format=fmt, errors='coerce')
                    valid_dates = processed_data[col].notna().sum()
                    preprocessing_steps.append(f"Tried format {fmt}: {valid_dates} valid dates")
                    
                    if valid_dates > 0.5 * len(processed_data):  # Jika lebih dari 50% konversi berhasil
                        conversion_success = True
                        preprocessing_steps.append(f"Using format {fmt} for {col}")
                        break
                except Exception as e:
                    preprocessing_steps.append(f"Format {fmt} failed: {str(e)}")
            
            # Jika semua format spesifik gagal, gunakan pandas inference
            if not conversion_success:
                preprocessing_steps.append(f"Trying pandas date inference for {col}")
                processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
                valid_dates = processed_data[col].notna().sum()
                preprocessing_steps.append(f"Pandas inference: {valid_dates} valid dates")
    
    # Hitung usia dengan penanganan error yang lebih baik
    if 'BIRTH_DATE' in processed_data.columns:
        preprocessing_steps.append("Calculating age from BIRTH_DATE")
        
        # Cek konversi tanggal lahir
        valid_birth_dates = processed_data['BIRTH_DATE'].notna()
        valid_count = valid_birth_dates.sum()
        total_count = len(processed_data)
        
        preprocessing_steps.append(f"Valid birth dates: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
        
        if valid_count > 0:
            # Inisialisasi kolom Usia
            processed_data['Usia'] = np.nan
            
            # Gunakan tanggal saat ini untuk perhitungan usia
            current_date = datetime.datetime.now()
            preprocessing_steps.append(f"Reference date for age calculation: {current_date}")
            
            # Hitung usia dengan metode yang lebih akurat (hari/365.25)
            processed_data.loc[valid_birth_dates, 'Usia'] = (
                (current_date - processed_data.loc[valid_birth_dates, 'BIRTH_DATE']).dt.days / 365.25
            )
            
            # Konversi ke integer
            processed_data['Usia'] = processed_data['Usia'].astype(float).round(0).astype('Int64')
            
            # Validasi rentang usia (cek nilai negatif atau usia yang tidak realistis)
            invalid_age = (processed_data['Usia'] < 0) | (processed_data['Usia'] > 100)
            invalid_count = invalid_age.sum()
            
            if invalid_count > 0:
                preprocessing_steps.append(f"Found {invalid_count} records with unrealistic age values (negative or > 100)")
                processed_data.loc[invalid_age, 'Usia'] = np.nan
            
            # Hitung statistik usia
            if not processed_data['Usia'].isna().all():
                age_min = processed_data['Usia'].min()
                age_max = processed_data['Usia'].max()
                age_mean = processed_data['Usia'].mean()
                preprocessing_steps.append(f"Age statistics: Min={age_min}, Max={age_max}, Mean={age_mean:.1f}")
                
                # Buat kategori usia
                bins = [0, 25, 35, 45, 55, 100]
                labels = ['<25', '25-35', '35-45', '45-55', '55+']
                processed_data['Usia_Kategori'] = pd.cut(processed_data['Usia'], bins=bins, labels=labels, right=False)
                
                # Cek distribusi kategori usia
                age_dist = processed_data['Usia_Kategori'].value_counts()
                preprocessing_steps.append(f"Age category distribution: {dict(age_dist)}")
            else:
                preprocessing_steps.append("No valid ages calculated after validation")
                
                # Buat data usia acak untuk demo jika tidak ada usia valid
                preprocessing_steps.append("Creating random age data for demonstration")
                processed_data['Usia'] = np.random.randint(18, 65, size=len(processed_data))
                
                # Buat kategori usia dari data acak
                bins = [0, 25, 35, 45, 55, 100]
                labels = ['<25', '25-35', '35-45', '45-55', '55+']
                processed_data['Usia_Kategori'] = pd.cut(processed_data['Usia'], bins=bins, labels=labels, right=False)
        else:
            preprocessing_steps.append("No valid birth dates found. Creating random age data for demonstration")
            # Buat data usia acak untuk demo
            processed_data['Usia'] = np.random.randint(18, 65, size=len(processed_data))
            
            # Buat kategori usia dari data acak
            bins = [0, 25, 35, 45, 55, 100]
            labels = ['<25', '25-35', '35-45', '45-55', '55+']
            processed_data['Usia_Kategori'] = pd.cut(processed_data['Usia'], bins=bins, labels=labels, right=False)
    else:
        preprocessing_steps.append("BIRTH_DATE column not found. Creating random age data for demonstration")
        # Buat data usia acak untuk demo
        processed_data['Usia'] = np.random.randint(18, 65, size=len(processed_data))
        
        # Buat kategori usia dari data acak
        bins = [0, 25, 35, 45, 55, 100]
        labels = ['<25', '25-35', '35-45', '45-55', '55+']
        processed_data['Usia_Kategori'] = pd.cut(processed_data['Usia'], bins=bins, labels=labels, right=False)
    
    # Konversi kolom numerik ke tipe numerik dengan penanganan error yang lebih baik
    numeric_cols = ['TOTAL_AMOUNT_MPF', 'TOTAL_PRODUCT_MPF', 'MAX_MPF_AMOUNT', 'MIN_MPF_AMOUNT', 
                  'LAST_MPF_AMOUNT', 'LAST_MPF_INST', 'LAST_MPF_TOP', 'AVG_MPF_INST',
                  'PRINCIPAL', 'GRS_DP', 'JMH_CON_SBLM_MPF', 'JMH_PPC']
    
    for col in numeric_cols:
        if col in processed_data.columns:
            preprocessing_steps.append(f"Converting {col} to numeric")
            
            # Bersihkan data terlebih dahulu (hapus karakter non-numerik)
            if processed_data[col].dtype == 'object':
                processed_data[col] = processed_data[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            
            # Konversi ke numerik
            try:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                valid_count = processed_data[col].notna().sum()
                preprocessing_steps.append(f"Successfully converted {valid_count}/{len(processed_data)} values in {col} to numeric")
                
                # Jika ada nilai yang hilang, isi dengan median
                if processed_data[col].isna().any():
                    median_val = processed_data[col].median()
                    processed_data[col].fillna(median_val, inplace=True)
                    preprocessing_steps.append(f"Filled {processed_data[col].isna().sum()} missing values in {col} with median: {median_val}")
            except Exception as e:
                preprocessing_steps.append(f"Error converting {col} to numeric: {str(e)}")
    
    # Identifikasi kolom numerik dan kategorikal
    numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = processed_data.select_dtypes(include=['object']).columns
    
    # Handle missing values
    for col in numeric_cols:
        if col != 'Usia' and processed_data[col].isnull().sum() > 0:
            median_val = processed_data[col].median()
            processed_data[col].fillna(median_val, inplace=True)
            preprocessing_steps.append(f"Filled missing values in {col} with median: {median_val}")
    
    for col in categorical_cols:
        if col != 'Usia_Kategori' and processed_data[col].isnull().sum() > 0:
            mode_val = processed_data[col].mode()[0]
            processed_data[col].fillna(mode_val, inplace=True)
            preprocessing_steps.append(f"Filled missing values in {col} with mode: {mode_val}")
    
    # Hapus kolom yang tidak diperlukan jika ada
    if 'JMH_CON_NON_MPF' in processed_data.columns:
        processed_data.drop(columns=['JMH_CON_NON_MPF'], inplace=True)
        preprocessing_steps.append("Dropped column 'JMH_CON_NON_MPF'")
    
    # Tambahkan fitur tambahan
    processed_data['PROCESSING_DATE'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    preprocessing_steps.append(f"Added 'PROCESSING_DATE' column with current timestamp: {processed_data['PROCESSING_DATE'].iloc[0]}")
    
    if "TOTAL_PRODUCT_MPF" in processed_data.columns:
        processed_data["Multi-Transaction_Customer"] = processed_data["TOTAL_PRODUCT_MPF"].astype(float).apply(lambda x: 1 if x > 1 else 0)
        preprocessing_steps.append("Added 'Multi-Transaction_Customer' flag")
    
    # Pastikan kolom Usia_Kategori adalah tipe kategori jika ada
    if 'Usia_Kategori' in processed_data.columns:
        processed_data['Usia_Kategori'] = processed_data['Usia_Kategori'].astype('category')
        preprocessing_steps.append("Converted 'Usia_Kategori' to category type")
    
    # Tampilkan log preprocessing
    st.subheader("Preprocessing Log")
    for i, step in enumerate(preprocessing_steps, 1):
        st.write(f"{i}. {step}")
    
    return processed_data
