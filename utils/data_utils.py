import pandas as pd
import numpy as np
import datetime

def create_example_data(n=500):
    """
    Fungsi untuk membuat data contoh
    
    Parameters:
    -----------
    n : int
        Jumlah baris data yang akan dibuat
    
    Returns:
    --------
    pandas.DataFrame
        Dataframe berisi data contoh
    """
    np.random.seed(42)
    cust_ids = [f"1010000{i:05d}" for i in range(1, n+1)]
    product_categories = ['GADGET', 'ELECTRONIC', 'FURNITURE', 'OTHER']
    product_weights = [0.4, 0.3, 0.2, 0.1]
    ppc_types = ['MPF', 'REFI', 'NMC']
    genders = ['M', 'F']
    education = ['SD', 'SMP', 'SMA', 'S1', 'S2', 'S3']
    house_status = ['H01', 'H02', 'H03', 'H04', 'H05']
    marital_status = ['M', 'S', 'D']
    areas = ['JATA 1', 'JATA 2', 'JATA 3', 'SULSEL', 'KALSEL', 'SUMSEL']
    
    start_date = pd.Timestamp('2018-01-01')
    end_date = pd.Timestamp('2023-12-31')
    date_range = (end_date - start_date).days
    
    df = pd.DataFrame({
        'CUST_NO': cust_ids,
        'FIRST_PPC': np.random.choice(ppc_types, size=n, p=[0.6, 0.3, 0.1]),
        'FIRST_PPC_DATE': [start_date + pd.Timedelta(days=np.random.randint(0, date_range)) for _ in range(n)],
        'FIRST_MPF_DATE': [start_date + pd.Timedelta(days=np.random.randint(0, date_range)) for _ in range(n)],
        'LAST_MPF_DATE': [start_date + pd.Timedelta(days=np.random.randint(0, date_range)) for _ in range(n)],
        'JMH_CON_SBLM_MPF': np.random.randint(0, 5, size=n),
        'MAX_MPF_AMOUNT': np.random.randint(1000000, 20000000, size=n),
        'MIN_MPF_AMOUNT': np.random.randint(1000000, 10000000, size=n),
        'AVG_MPF_INST': np.random.randint(100000, 2000000, size=n),
        'MPF_CATEGORIES_TAKEN': [np.random.choice(product_categories, p=product_weights) for _ in range(n)],
        'LAST_MPF_PURPOSE': [np.random.choice(product_categories, p=product_weights) for _ in range(n)],
        'LAST_MPF_AMOUNT': np.random.randint(1000000, 15000000, size=n),
        'LAST_MPF_TOP': np.random.choice([6, 9, 12, 18, 24, 36], size=n),
        'LAST_MPF_INST': np.random.randint(100000, 1500000, size=n),
        'JMH_PPC': np.random.randint(1, 6, size=n),
        'PRINCIPAL': np.random.randint(2000000, 20000000, size=n),
        'GRS_DP': np.random.randint(0, 5000000, size=n),
        'BIRTH_DATE': [pd.Timestamp('1970-01-01') + pd.Timedelta(days=np.random.randint(0, 365*40)) for _ in range(n)],
        'CUST_SEX': np.random.choice(genders, size=n),
        'EDU_TYPE': np.random.choice(education, size=n, p=[0.05, 0.1, 0.4, 0.35, 0.08, 0.02]),
        'OCPT_CODE': np.random.randint(1, 25, size=n),
        'HOUSE_STAT': np.random.choice(house_status, size=n),
        'MARITAL_STAT': np.random.choice(marital_status, size=n, p=[0.7, 0.2, 0.1]),
        'NO_OF_DEPEND': np.random.randint(0, 5, size=n),
        'BRANCH_ID': np.random.randint(10000, 99999, size=n),
        'AREA': np.random.choice(areas, size=n),
        'TOTAL_AMOUNT_MPF': np.random.randint(1000000, 50000000, size=n),
        'TOTAL_PRODUCT_MPF': np.random.randint(1, 5, size=n)
    })
    
    # Adjust values for realism
    for i in range(len(df)):
        if df.loc[i, 'MIN_MPF_AMOUNT'] > df.loc[i, 'MAX_MPF_AMOUNT']:
            df.loc[i, 'MIN_MPF_AMOUNT'], df.loc[i, 'MAX_MPF_AMOUNT'] = df.loc[i, 'MAX_MPF_AMOUNT'], df.loc[i, 'MIN_MPF_AMOUNT']
    
    for i in range(len(df)):
        if df.loc[i, 'FIRST_MPF_DATE'] > df.loc[i, 'LAST_MPF_DATE']:
            df.loc[i, 'FIRST_MPF_DATE'], df.loc[i, 'LAST_MPF_DATE'] = df.loc[i, 'LAST_MPF_DATE'], df.loc[i, 'FIRST_MPF_DATE']
        if df.loc[i, 'FIRST_PPC_DATE'] > df.loc[i, 'FIRST_MPF_DATE']:
            df.loc[i, 'FIRST_PPC_DATE'] = df.loc[i, 'FIRST_MPF_DATE'] - pd.Timedelta(days=np.random.randint(1, 100))
    
    # Hitung usia sebagai variabel NUMERIK (penting untuk analisis distribusi)
    df['Usia'] = 2024 - df['BIRTH_DATE'].dt.year
    
    # Buat kategori usia sebagai variabel KATEGORIK
    bins = [0, 25, 35, 45, 55, 100]
    labels = ['<25', '25-35', '35-45', '45-55', '55+']
    df['Usia_Kategori'] = pd.cut(df['Usia'], bins=bins, labels=labels, right=False)
    
    # Tambahkan flag multi-transaction
    df['Multi-Transaction_Customer'] = df['TOTAL_PRODUCT_MPF'].apply(lambda x: 1 if x > 1 else 0)
    
    return df

def calculate_rfm(df, recency_col, freq_col, mon_col):
    """
    Menghitung metrik RFM (Recency, Frequency, Monetary)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data customer
    recency_col : str
        Nama kolom untuk recency (tanggal transaksi terakhir)
    freq_col : str
        Nama kolom untuk frequency (jumlah transaksi/produk)
    mon_col : str
        Nama kolom untuk monetary (total nilai transaksi)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame yang berisi metrik RFM
    """
    # Pastikan kolom tanggal dalam format yang benar
    if pd.api.types.is_datetime64_any_dtype(df[recency_col]):
        now = df[recency_col].max() + pd.Timedelta(days=1)
    else:
        raise ValueError(f"Kolom {recency_col} bukan tipe datetime")
    
    # Group by customer ID
    rfm = df.groupby('CUST_NO').agg({
        recency_col: 'max',
        freq_col: 'sum',
        mon_col: 'sum',
    }).reset_index()
    
    # Hitung metrik RFM
    rfm['Recency'] = (now - rfm[recency_col]).dt.days
    rfm['Frequency'] = rfm[freq_col]
    rfm['Monetary'] = rfm[mon_col]
    
    # Tambahkan transformasi log untuk monetary dan frequency
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
    
    # Transfer kolom usia (jika ada)
    if 'Usia' in df.columns:
        # Ambil nilai usia untuk setiap pelanggan (biasanya ambil nilai terakhir)
        usia_data = df.groupby('CUST_NO')['Usia'].last().reset_index()
        rfm = rfm.merge(usia_data, on='CUST_NO', how='left')
    
    # Transfer kolom kategori usia (jika ada)
    if 'Usia_Kategori' in df.columns:
        # Ambil kategori usia untuk setiap pelanggan (biasanya ambil nilai terakhir)
        usia_kategori_data = df.groupby('CUST_NO')['Usia_Kategori'].last().reset_index()
        rfm = rfm.merge(usia_kategori_data, on='CUST_NO', how='left')
    
    return rfm

def normalize_data(df, columns):
    """
    Normalisasi data menggunakan Z-score
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data yang akan dinormalisasi
    columns : list
        Daftar kolom yang akan dinormalisasi
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame dengan kolom yang sudah dinormalisasi
    """
    from scipy.stats import zscore
    
    df_norm = df.copy()
    for col in columns:
        if col in df.columns:
            df_norm[col] = zscore(df[col], nan_policy='omit')
    
    # Handle NaN values after normalization
    df_norm = df_norm.fillna(df_norm.median())
    
    return df_norm

def get_cluster_info(df, cluster_col='Cluster'):
    """
    Mendapatkan informasi tentang cluster
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data yang sudah di-cluster
    cluster_col : str
        Nama kolom cluster
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame yang berisi informasi cluster
    """
    cluster_info = df.groupby(cluster_col).agg({
        'Recency': ['mean', 'median', 'count'],
        'Frequency': ['mean', 'median'],
        'Monetary': ['mean', 'median'],
    }).reset_index()
    
    # Flatten multi-index
    cluster_info.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in cluster_info.columns]
    
    # Hitung persentase pelanggan di setiap cluster
    total_customers = df.shape[0]
    cluster_info['Percentage'] = (cluster_info['Recency_count'] / total_customers * 100).round(2)
    
    # Urutkan berdasarkan nilai monetary
    cluster_info = cluster_info.sort_values('Monetary_mean', ascending=False)
    
    return cluster_info

def generate_promo_recommendations(df, cluster_col='Cluster'):
    """
    Menghasilkan rekomendasi promo berdasarkan cluster
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data customer yang telah di-cluster
    cluster_col : str
        Nama kolom cluster
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame yang berisi rekomendasi promo untuk setiap cluster
    """
    # Buat DataFrame untuk rekomendasi
    promo_df = pd.DataFrame()
    
    # Dapatkan info cluster
    cluster_info = get_cluster_info(df, cluster_col)
    
    # Sortir cluster berdasarkan nilai rata-rata monetary
    high_value_clusters = cluster_info.sort_values('Monetary_mean', ascending=False)['Cluster'].tolist()
    
    # Sortir cluster berdasarkan recency (semakin kecil semakin baru)
    recent_clusters = cluster_info.sort_values('Recency_mean')['Cluster'].tolist()
    
    # Sortir cluster berdasarkan frequency
    frequent_clusters = cluster_info.sort_values('Frequency_mean', ascending=False)['Cluster'].tolist()
    
    # Inisialisasi DataFrame untuk menyimpan rekomendasi promo
    promo_recommendations = []
    
    # Iterasi untuk semua cluster
    for cluster in df[cluster_col].unique():
        cluster_df = df[df[cluster_col] == cluster]
        
        # Ambil statistik cluster
        avg_recency = cluster_df['Recency'].mean()
        avg_frequency = cluster_df['Frequency'].mean()
        avg_monetary = cluster_df['Monetary'].mean()
        count = cluster_df.shape[0]
        
        # Tentukan kategori customer berdasarkan posisi cluster
        customer_value = "High Value" if cluster == high_value_clusters[0] else (
            "Medium Value" if cluster in high_value_clusters[1:3] else "Low Value")
        
        recency_status = "Active" if cluster in recent_clusters[:3] else (
            "Lapsed" if avg_recency > 180 else "Inactive")
        
        loyalty_status = "Loyal" if cluster in frequent_clusters[:2] else (
            "Occasional" if avg_frequency >= 2 else "One-time")
        
        # Tentukan rekomendasi promo berdasarkan kategori
        if customer_value == "High Value" and recency_status == "Active":
            promo_type = "VIP Exclusive Program"
            promo_desc = "Akses prioritas untuk produk baru, voucher diskon eksklusif, dan benefit spesial"
            channel = "Direct Call, WhatsApp Business"
        elif customer_value == "High Value" and recency_status != "Active":
            promo_type = "Win-back Premium"
            promo_desc = "Penawaran khusus untuk kembali, diskon eksklusif dan hadiah menarik"
            channel = "Direct Call, Email Personal, WhatsApp"
        elif customer_value == "Medium Value" and recency_status == "Active":
            promo_type = "Loyalty Reward"
            promo_desc = "Program reward berdasarkan transaksi, poin loyalty, diskon bertingkat"
            channel = "SMS, Email, Push Notification"
        elif customer_value == "Medium Value" and recency_status != "Active":
            promo_type = "Win-back Standard"
            promo_desc = "Penawaran untuk kembali, diskon terbatas, dan promosi bundle produk"
            channel = "Email, SMS"
        elif recency_status == "Active" and loyalty_status == "Occasional":
            promo_type = "Upgrade Program"
            promo_desc = "Insentif untuk meningkatkan jumlah transaksi, promosi produk terkait"
            channel = "Push Notification, SMS"
        elif recency_status != "Active" and loyalty_status == "One-time":
            promo_type = "Re-engagement Basic"
            promo_desc = "Penawaran dasar untuk kembali berbelanja, diskon kecil"
            channel = "Email Blast, SMS Bulk"
        else:
            promo_type = "General Promotion"
            promo_desc = "Promosi umum seperti cashback, diskon seasonal, atau bundling produk"
            channel = "Email Blast, SMS, Social Media"
        
        # Tentukan budget berdasarkan nilai customer
        if customer_value == "High Value":
            budget_allocation = "High (30-40% dari total budget)"
        elif customer_value == "Medium Value":
            budget_allocation = "Medium (20-30% dari total budget)"
        else:
            budget_allocation = "Low (10-20% dari total budget)"
        
        # Tambahkan ke list rekomendasi
        promo_recommendations.append({
            'Cluster': cluster,
            'Customer_Count': count,
            'Avg_Recency_Days': round(avg_recency),
            'Avg_Frequency': round(avg_frequency, 1),
            'Avg_Monetary': int(avg_monetary),
            'Customer_Value': customer_value,
            'Recency_Status': recency_status,
            'Loyalty_Status': loyalty_status,
            'Promo_Type': promo_type,
            'Promo_Description': promo_desc,
            'Channel': channel,
            'Budget_Allocation': budget_allocation
        })
    
    return pd.DataFrame(promo_recommendations)
