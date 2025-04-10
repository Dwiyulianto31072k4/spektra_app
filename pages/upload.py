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

                    # Tampilkan informasi kolom Usia jika ada
                    if 'Usia' in processed_data.columns:
                        st.write("Age statistics:")
                        stats = processed_data['Usia'].describe()
                        st.write(f"- Min: {stats['min']:.0f}, Max: {stats['max']:.0f}, Mean: {stats['mean']:.1f}")
                        
                        # Tampilkan distribusi kategori usia jika ada
                        if 'Usia_Kategori' in processed_data.columns:
                            st.write("Age category distribution:")
                            st.write(processed_data['Usia_Kategori'].value_counts())

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

def preprocess_data(data, date_cols):
    """
    Fungsi preprocessing data - fokus pada pembuatan kolom Usia dan Usia_Kategori
    hanya jika BIRTH_DATE tersedia
    """
    processed_data = data.copy()
    
    # Konversi kolom tanggal dengan multi-format handling
    for col in date_cols:
        if col in processed_data.columns:
            try:
                # Ambil sampel untuk melihat format yang mungkin
                samples = processed_data[col].dropna().head(3).tolist()
                st.write(f"Sample dates in {col}: {samples}")
                
                # Coba beberapa format standar
                for fmt in ['%Y%m%d', '%d/%m/%Y', '%m/%d/%Y', '%Y-%m-%d']:
                    try:
                        test_convert = pd.to_datetime(processed_data[col], format=fmt, errors='coerce')
                        if test_convert.notna().sum() > processed_data.shape[0] * 0.5:  # Jika > 50% berhasil
                            processed_data[col] = test_convert
                            st.write(f"Converted {col} using format {fmt}")
                            break
                    except:
                        continue
                
                # Fallback ke pandas automatic detection
                if pd.api.types.is_datetime64_dtype(processed_data[col]) == False:
                    processed_data[col] = pd.to_datetime(processed_data[col], errors='coerce')
                    st.write(f"Converted {col} using pandas automatic detection")
                
                # Laporan hasil konversi
                valid_dates = processed_data[col].notna().sum()
                st.write(f"Successfully converted {valid_dates}/{len(processed_data)} dates in {col}")
                
            except Exception as e:
                st.error(f"Error converting {col}: {str(e)}")
    
    # Hitung usia HANYA jika BIRTH_DATE tersedia dan valid
    if 'BIRTH_DATE' in processed_data.columns:
        valid_birth_dates = processed_data['BIRTH_DATE'].notna()
        valid_count = valid_birth_dates.sum()
        
        if valid_count > 0:
            st.write(f"Calculating age from {valid_count} valid birth dates...")
            
            # Hitung usia
            current_date = pd.Timestamp.now()
            processed_data['Usia'] = np.nan  # Initialize with NaN
            
            # Hanya hitung untuk tanggal lahir yang valid
            processed_data.loc[valid_birth_dates, 'Usia'] = (
                (current_date - processed_data.loc[valid_birth_dates, 'BIRTH_DATE']).dt.days / 365.25
            )
            
            # Round dan filter usia yang masuk akal (18-100)
            processed_data['Usia'] = processed_data['Usia'].round()
            valid_age = (processed_data['Usia'] >= 18) & (processed_data['Usia'] <= 100)
            
            # Hanya gunakan usia yang valid
            if valid_age.sum() > 0:
                # Convert to numeric and integer
                processed_data['Usia'] = pd.to_numeric(processed_data['Usia'], errors='coerce')
                
                # Create age categories
                bins = [0, 25, 35, 45, 55, 100]
                labels = ['<25', '25-35', '35-45', '45-55', '55+']
                
                processed_data['Usia_Kategori'] = pd.cut(
                    processed_data['Usia'], 
                    bins=bins, 
                    labels=labels, 
                    right=False
                )
                
                # Convert to string for consistency
                processed_data['Usia_Kategori'] = processed_data['Usia_Kategori'].astype(str)
                
                # Report success
                st.write(f"Successfully created age data for {valid_age.sum()} customers")
                st.write(f"Age statistics: Min={processed_data['Usia'].min():.0f}, Max={processed_data['Usia'].max():.0f}, Mean={processed_data['Usia'].mean():.1f}")
            else:
                st.warning("No valid ages (18-100) could be calculated from birth dates.")
        else:
            st.warning("No valid birth dates found in the BIRTH_DATE column.")
    else:
        st.info("BIRTH_DATE column not found. Age data won't be available.")
    
    # Konversi kolom numerik
    numeric_cols = ['TOTAL_AMOUNT_MPF', 'TOTAL_PRODUCT_MPF', 'MAX_MPF_AMOUNT', 'MIN_MPF_AMOUNT', 
                   'LAST_MPF_AMOUNT', 'LAST_MPF_INST', 'LAST_MPF_TOP', 'AVG_MPF_INST',
                   'PRINCIPAL', 'GRS_DP', 'JMH_CON_SBLM_MPF', 'JMH_PPC']
    
    for col in numeric_cols:
        if col in processed_data.columns:
            if processed_data[col].dtype == 'object':
                processed_data[col] = processed_data[col].astype(str).str.replace(',', '').str.replace(r'[^\d.]', '', regex=True)
            processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            if processed_data[col].isna().any():
                median_val = processed_data[col].median()
                processed_data[col] = processed_data[col].fillna(median_val)
    
    # Tambahkan fitur tambahan
    processed_data['PROCESSING_DATE'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "TOTAL_PRODUCT_MPF" in processed_data.columns:
        processed_data["Multi-Transaction_Customer"] = processed_data["TOTAL_PRODUCT_MPF"].astype(float).apply(lambda x: 1 if x > 1 else 0)
    
    return processed_data
