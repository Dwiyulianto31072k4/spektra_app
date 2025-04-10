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
            # Baca semua kolom sebagai string untuk menjaga konsistensi (namun jika format numeric seperti Excel serial, nantinya akan dideteksi)
            data = pd.read_excel(uploaded_file, dtype=str)
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
                        valid_ages = processed_data['Usia'].notna().sum()
                        total_rows = len(processed_data)
                        st.write(f"Valid age values: {valid_ages} out of {total_rows} rows ({valid_ages/total_rows*100:.1f}%)")

                        if valid_ages > 0:
                            age_stats = processed_data['Usia'].describe()
                            st.write(f"Age statistics: Min={age_stats['min']:.1f}, Max={age_stats['max']:.1f}, Mean={age_stats['mean']:.1f}")

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

    # Hapus waktu (contoh: "00.00.00" atau komponen waktu lain)
    date_string = str(date_string)
    date_string = re.sub(r'\s*\d{1,2}[:.]\d{1,2}[:.]\d{1,2}.*$', '', date_string)
    date_string = re.sub(r'\s+00\.00\.00.*$', '', date_string)

    # Bersihkan karakter pemisah yang tidak standar
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
            # Cek apakah kolom tersebut merupakan tipe numerik (mis. Excel serial number)
            if pd.api.types.is_numeric_dtype(processed_data[col]) or processed_data[col].str.isnumeric().all():
                try:
                    processed_data[col] = pd.to_datetime(processed_data[col], origin='1899-12-30', unit='d', errors='coerce')
                    st.write(f"Converted {col} from numeric (Excel serial) to datetime: {processed_data[col].notna().sum()} valid dates")
                    continue
                except Exception as e:
                    st.warning(f"Numeric conversion failed for {col} with error: {e}")

            # Jika bukan numeric, lakukan clean-up terlebih dahulu
            original_values = processed_data[col].copy()
            processed_data[col] = processed_data[col].apply(clean_date_string)

            # Coba beberapa format tanggal umum
            date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d', '%d/%m/%Y', '%d.%m.%Y', '%Y.%m.%d']
            converted = False

            for fmt in date_formats:
                try:
                    temp_dates = pd.to_datetime(processed_data[col], format=fmt, errors='coerce')
                    success_rate = temp_dates.notna().sum() / len(processed_data)
                    if success_rate > 0.3:
                        processed_data[col] = temp_dates
                        st.write(f"Converted {col} using format {fmt}: {temp_dates.notna().sum()} valid dates ({success_rate:.1%})")
                        converted = True
                        break
                except Exception:
                    continue

            if not converted:
                processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
                success_rate = processed_data[col].notna().sum() / len(processed_data)
                st.write(f"Converted {col} using pandas automatic detection: {processed_data[col].notna().sum()} valid dates ({success_rate:.1%})")

    # Khusus untuk BIRTH_DATE dengan penanganan lebih agresif
    if 'BIRTH_DATE' in processed_data.columns:
        if processed_data['BIRTH_DATE'].notna().sum() == 0:
            st.warning("All BIRTH_DATE values failed to convert. Trying more aggressive approach...")

            st.write("Sample BIRTH_DATE values before aggressive cleanup:")
            st.write(data['BIRTH_DATE'].dropna().head(5).tolist())

            def extract_date_components(date_str):
                if pd.isna(date_str):
                    return None
                date_str = str(date_str).strip()
                # Coba ekstrak dengan regex: MM/DD/YYYY atau DD/MM/YYYY
                match = re.search(r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4}|\d{2})', date_str)
                if match:
                    first, second, third = match.groups()
                    first, second, third = int(first), int(second), int(third)
                    # Jika tahun hanya dua digit
                    if third < 100:
                        third = third + 1900 if third >= 50 else third + 2000

                    # Jika nilai kedua lebih besar dari 12, asumsikan format MM/DD/YYYY
                    if second > 12:
                        day, month = first, second
                    else:
                        # Jika ambigu, gunakan format default: first sebagai day
                        day, month = first, second

                    try:
                        return pd.Timestamp(year=third, month=month, day=day)
                    except Exception:
                        return None

                # Coba format YYYY-MM-DD
                match = re.search(r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', date_str)
                if match:
                    year, month, day = map(int, match.groups())
                    try:
                        return pd.Timestamp(year=year, month=month, day=day)
                    except Exception:
                        return None
                return None

            processed_data['BIRTH_DATE'] = processed_data['BIRTH_DATE'].apply(extract_date_components)
            success_rate = processed_data['BIRTH_DATE'].notna().sum() / len(processed_data)
            st.write(f"Extracted BIRTH_DATE with component extraction: {processed_data['BIRTH_DATE'].notna().sum()} valid dates ({success_rate:.1%})")

        # Jika sudah ada tanggal yang valid, hitung usia
        if processed_data['BIRTH_DATE'].notna().sum() > 0:
            st.write("Calculating age from BIRTH_DATE...")
            current_date = pd.Timestamp.now()
            processed_data['Usia'] = ((current_date - processed_data['BIRTH_DATE']).dt.days / 365.25).round()

            # Filter usia yang valid (antara 18 dan 100 tahun)
            valid_age = (processed_data['Usia'] >= 18) & (processed_data['Usia'] <= 100)
            processed_data.loc[~valid_age, 'Usia'] = np.nan

            processed_data['Usia'] = pd.to_numeric(processed_data['Usia'], errors='coerce')

            if processed_data['Usia'].notna().sum() > 0:
                st.success(f"Successfully calculated age for {processed_data['Usia'].notna().sum()} customers")
                st.write(f"Age statistics: Min={processed_data['Usia'].min():.1f}, Max={processed_data['Usia'].max():.1f}, Mean={processed_data['Usia'].mean():.1f}")

                # Buat kategori usia
                bins = [0, 25, 35, 45, 55, 100]
                labels = ['<25', '25-35', '35-45', '45-55', '55+']
                processed_data['Usia_Kategori'] = pd.cut(processed_data['Usia'], bins=bins, labels=labels, right=False)
                processed_data['Usia_Kategori'] = processed_data['Usia_Kategori'].astype(str)
                processed_data.loc[processed_data['Usia_Kategori'] == 'nan', 'Usia_Kategori'] = 'Unknown'

                st.write("Age category distribution:")
                st.write(processed_data['Usia_Kategori'].value_counts())
            else:
                st.warning("No valid ages could be calculated (between 18-100 years)")
        else:
            st.warning("No valid BIRTH_DATE values could be converted")
    else:
        st.warning("BIRTH_DATE column not found in the data")

    # Konversi kolom numerik
    numeric_cols = ['TOTAL_AMOUNT_MPF', 'TOTAL_PRODUCT_MPF', 'MAX_MPF_AMOUNT', 'MIN_MPF_AMOUNT', 
                    'LAST_MPF_AMOUNT', 'LAST_MPF_INST', 'LAST_MPF_TOP', 'AVG_MPF_INST',
                    'PRINCIPAL', 'GRS_DP', 'JMH_CON_SBLM_MPF', 'JMH_PPC']
    for col in numeric_cols:
        if col in processed_data.columns:
            if processed_data[col].dtype == 'object':
                processed_data[col] = processed_data[col].astype(str)\
                                            .str.replace(',', '')\
                                            .str.replace(r'[^\d\.-]', '', regex=True)
            original_count = len(processed_data)
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            null_count = processed_data[col].isnull().sum()
            if null_count > 0 and null_count < len(processed_data):
                median_val = processed_data[col].median()
                processed_data[col] = processed_data[col].fillna(median_val)

    # Tambahkan fitur tambahan
    processed_data['PROCESSING_DATE'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "TOTAL_PRODUCT_MPF" in processed_data.columns:
        processed_data["Multi-Transaction_Customer"] = processed_data["TOTAL_PRODUCT_MPF"].astype(float).apply(lambda x: 1 if x > 1 else 0)

    return processed_data
