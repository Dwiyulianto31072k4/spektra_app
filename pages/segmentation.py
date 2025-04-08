import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy.stats import zscore

from utils.data_utils import calculate_rfm, normalize_data, get_cluster_info

def show_segmentation_page():
    """
    Fungsi untuk menampilkan halaman segmentasi pelanggan
    """
    st.markdown('<p class="section-title">Segmentation Analysis (RFM + K-Means)</p>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("Please upload and preprocess your data first.")
        return

    df = st.session_state.data.copy()

    # Pastikan kolom 'Multi-Transaction_Customer' ada
    if 'Multi-Transaction_Customer' not in df.columns and 'TOTAL_PRODUCT_MPF' in df.columns:
        df["Multi-Transaction_Customer"] = df["TOTAL_PRODUCT_MPF"].apply(lambda x: 1 if pd.to_numeric(x, errors='coerce') > 1 else 0)

    # Pilih kolom untuk RFM
    st.markdown("### Select RFM Columns and Clustering Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if date_cols:
            recency_col = st.selectbox(
                "Recency (last transaction date)", 
                date_cols, 
                index=date_cols.index("LAST_MPF_DATE") if "LAST_MPF_DATE" in date_cols else 0
            )
        else:
            st.error("No date columns found in the data. Please preprocess the data first.")
            return
    
    with col2:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        freq_col = st.selectbox(
            "Frequency (number of products)", 
            numeric_cols, 
            index=numeric_cols.index("TOTAL_PRODUCT_MPF") if "TOTAL_PRODUCT_MPF" in numeric_cols else 0
        )
    
    with col3:
        mon_col = st.selectbox(
            "Monetary (total amount)", 
            numeric_cols, 
            index=numeric_cols.index("TOTAL_AMOUNT_MPF") if "TOTAL_AMOUNT_MPF" in numeric_cols else 0
        )

    # Pilih jumlah cluster
    cluster_k = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=4)
    
    # Pilih kolom tambahan
    st.markdown("### Additional Features for Clustering")
    col1, col2 = st.columns(2)
    
    with col1:
        use_age = st.checkbox("Include Age", value=True)
        use_multi_transaction = st.checkbox("Include Multi-Transaction Flag", value=True)
    
    with col2:
        use_log_transform = st.checkbox("Apply Log Transform to Monetary & Frequency", value=True)
        use_zscore = st.checkbox("Apply Z-Score Normalization", value=True)

    if st.button("Perform Segmentation"):
        with st.spinner("Processing segmentation..."):
            try:
                # Hitung metrik RFM
                rfm = calculate_rfm(df, recency_col, freq_col, mon_col)
                
                # Cek nilai NaN dalam RFM
                if rfm.isnull().values.any():
                    st.info("Terdapat nilai NaN dalam data RFM. Mengisi dengan nilai median.")
                    rfm = rfm.fillna(rfm.median())
                
                # Tambahkan kolom tambahan jika dipilih
                features = ['Recency']
                
                if use_log_transform:
                    # Pastikan nilai positif sebelum log transform
                    # Tambahkan epsilon kecil untuk menghindari log(0)
                    epsilon = 1e-6
                    rfm['Frequency_log'] = np.log1p(np.maximum(rfm['Frequency'] - 1, 0) + epsilon)
                    rfm['Monetary_log'] = np.log1p(np.maximum(rfm['Monetary'], 0) + epsilon)
                    features.extend(['Frequency_log', 'Monetary_log'])
                else:
                    features.extend(['Frequency', 'Monetary'])
                
                # Tambahkan fitur usia jika dipilih
                if use_age and 'BIRTH_DATE' in df.columns:
                    # Group by customer ID to get birth date
                    birth_data = df.groupby('CUST_NO').agg({'BIRTH_DATE': 'max'}).reset_index()
                    
                    # Merge with RFM data
                    rfm = rfm.merge(birth_data, on='CUST_NO', how='left')
                    
                    # Calculate age
                    current_year = pd.Timestamp.now().year
                    rfm['Usia'] = current_year - rfm['BIRTH_DATE'].dt.year
                    
                    # Periksa dan tangani nilai NaN dalam Usia
                    if rfm['Usia'].isnull().any():
                        st.info("Beberapa nilai Usia kosong. Mengisi dengan median.")
                        rfm['Usia'] = rfm['Usia'].fillna(rfm['Usia'].median())
                    
                    # Create age segment (1 for prime age 25-50, 0 otherwise)
                    rfm['Usia_Segment'] = rfm['Usia'].apply(lambda x: 1 if 25 <= x <= 50 else 0)
                    
                    features.append('Usia_Segment')
                
                # Tambahkan flag multi-transaction jika dipilih
                if use_multi_transaction:
                    if 'Multi-Transaction_Customer' not in rfm.columns:
                        # Group by customer ID to get multi-transaction flag
                        multi_trans_data = df.groupby('CUST_NO').agg({'Multi-Transaction_Customer': 'max'}).reset_index()
                        
                        # Merge with RFM data
                        rfm = rfm.merge(multi_trans_data, on='CUST_NO', how='left')
                        
                        # Periksa dan tangani nilai NaN
                        if rfm['Multi-Transaction_Customer'].isnull().any():
                            st.info("Beberapa nilai Multi-Transaction kosong. Mengisi dengan nilai 0.")
                            rfm['Multi-Transaction_Customer'] = rfm['Multi-Transaction_Customer'].fillna(0)
                    
                    features.append('Multi-Transaction_Customer')
                
                # Pastikan tidak ada NaN sebelum normalisasi
                for feat in features:
                    if rfm[feat].isnull().any():
                        rfm[feat] = rfm[feat].fillna(rfm[feat].median())
                        st.info(f"Nilai yang hilang terdeteksi di kolom {feat} dan diisi dengan median.")
                
                # Normalisasi fitur
                if use_zscore:
                    # Gunakan try-except untuk menangani zscore
                    try:
                        rfm_scaled = rfm[features].apply(zscore)
                    except:
                        st.warning("Gagal menerapkan Z-score normalization. Menggunakan metode alternatif.")
                        # Alternatif normalisasi manual
                        rfm_scaled = rfm[features].copy()
                        for col in rfm_scaled.columns:
                            mean = rfm_scaled[col].mean()
                            std = rfm_scaled[col].std()
                            if std > 0:  # Hindari pembagian dengan 0
                                rfm_scaled[col] = (rfm_scaled[col] - mean) / std
                            else:
                                rfm_scaled[col] = 0  # Jika std=0, semua nilai sama
                else:
                    rfm_scaled = rfm[features].copy()
                
                # Tangani missing value di rfm_scaled
                if rfm_scaled.isnull().values.any():
                    st.warning("There are NaN values in the clustering data. Filling NaN with median to avoid losing data.")
                    rfm_scaled = rfm_scaled.fillna(rfm_scaled.median())
                
                # Lakukan clustering
                kmeans = KMeans(n_clusters=cluster_k, random_state=42, n_init=10)
                rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
                
                # Evaluasi hasil cluster
                cluster_info = get_cluster_info(rfm)
                
                # Hitung skor untuk setiap cluster
                cluster_score = rfm.groupby('Cluster').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean',
                    'Monetary': 'mean'
                }).reset_index()
                
                # Semakin kecil recency semakin baik (baru transaksi)
                cluster_score['Recency_Score'] = 1 / (cluster_score['Recency'] + 1)
                
                # Hitung total skor (recency_score + frequency + monetary)
                cluster_score['Total_Score'] = (
                    cluster_score['Recency_Score'] + 
                    (cluster_score['Frequency'] / cluster_score['Frequency'].max()) +
                    (cluster_score['Monetary'] / cluster_score['Monetary'].max())
                )
                
                # Tentukan cluster mana yang akan diundang berdasarkan skor total
                n_invited = int(np.ceil(cluster_k / 2))  # 50% dari jumlah cluster
                top_clusters = cluster_score.sort_values('Total_Score', ascending=False).head(n_invited)['Cluster'].tolist()
                
                # Tandai pelanggan yang layak diundang
                rfm['Invitation_Status'] = rfm['Cluster'].apply(lambda x: '✅ Invited' if x in top_clusters else '❌ Not Invited')
                
                # Simpan hasil segmentasi
                st.session_state.segmented_data = rfm
                st.session_state.segmentation_completed = True
                
                # Tampilkan hasil segmentasi
                st.success("Segmentation completed successfully!")
                
                # Tampilkan hasil
                display_segmentation_results(rfm, cluster_info, top_clusters)
                
            except Exception as e:
                st.error(f"Error during segmentation: {e}")
                st.warning("Please check your data and selections, then try again.")
                
                # Tampilkan saran untuk mengatasi error
                if "NaN" in str(e):
                    st.warning("""
                    Error terkait nilai NaN (Not a Number). Solusi:
                    1. Periksa data input Anda untuk nilai yang hilang
                    2. Pastikan semua kolom numerik berisi nilai yang valid
                    3. Coba gunakan data contoh untuk melihat hasil yang diharapkan
                    """)
                    
                    # Tambahkan saran untuk menggunakan scikit-learn alternative
                    st.info("""
                    Untuk supervised learning, Anda mungkin ingin mempertimbangkan sklearn.ensemble.HistGradientBoostingClassifier atau 
                    sklearn.ensemble.HistGradientBoostingRegressor yang dapat menangani nilai NaN secara native. 
                    Atau, Anda dapat menggunakan sklearn.impute.SimpleImputer untuk mengisi nilai yang hilang.
                    
                    Lihat: https://scikit-learn.org/stable/modules/impute.html
                    """)

def display_segmentation_results(rfm, cluster_info, top_clusters):
    """
    Menampilkan hasil segmentasi
    
    Parameters:
    -----------
    rfm : pandas.DataFrame
        Data hasil segmentasi dengan kolom Cluster
    cluster_info : pandas.DataFrame
        Informasi tentang setiap cluster
    top_clusters : list
        Daftar cluster yang direkomendasikan untuk diundang
    """
    st.markdown("### Segmentation Results")
    
    # Overview
    st.write(f"Total customers: {rfm.shape[0]}")
    st.write(f"Number of clusters: {rfm['Cluster'].nunique()}")
    
    # Preview data hasil segmentasi
    st.markdown("#### Customer Segmentation Preview")
    st.dataframe(rfm[['CUST_NO', 'Recency', 'Frequency', 'Monetary', 'Cluster', 'Invitation_Status']].head(10))
    
    # Visualisasi cluster
    st.markdown("#### Cluster Visualization")
    
    tab1, tab2, tab3 = st.tabs(["Recency vs Monetary", "Recency vs Frequency", "3D Visualization"])
    
    with tab1:
        fig = px.scatter(
            rfm, x='Recency', y='Monetary',
            color='Cluster',
            title="Clusters by Recency and Monetary",
            hover_data=['CUST_NO', 'Frequency', 'Invitation_Status'],
            size='Frequency',
            size_max=15,
            color_continuous_scale=px.colors.sequential.Blues,
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.scatter(
            rfm, x='Recency', y='Frequency',
            color='Cluster',
            title="Clusters by Recency and Frequency",
            hover_data=['CUST_NO', 'Monetary', 'Invitation_Status'],
            size='Monetary',
            size_max=15,
            color_continuous_scale=px.colors.sequential.Blues,
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.scatter_3d(
            rfm, x='Recency', y='Frequency', z='Monetary',
            color='Cluster',
            title="3D Visualization of RFM Clusters",
            opacity=0.7
        )
        fig.update_layout(scene=dict(
            xaxis_title='Recency',
            yaxis_title='Frequency',
            zaxis_title='Monetary'
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Karakteristik cluster
    st.markdown("#### Cluster Characteristics")
    
    # Format tabel informasi cluster
    cluster_stats = cluster_info.copy()
    cluster_stats['Recency_mean'] = cluster_stats['Recency_mean'].round(0).astype(int)
    cluster_stats['Frequency_mean'] = cluster_stats['Frequency_mean'].round(1)
    cluster_stats['Monetary_mean'] = cluster_stats['Monetary_mean'].round(0).astype(int)
    
    # Tambahkan kolom is_top_cluster
    cluster_stats['Invitation_Status'] = cluster_stats['Cluster'].apply(
        lambda x: '✅ Invited' if x in top_clusters else '❌ Not Invited'
    )
    
    # Tampilkan statistik cluster
    st.dataframe(cluster_stats[['Cluster', 'Recency_mean', 'Frequency_mean', 'Monetary_mean', 
                               'Recency_count', 'Percentage', 'Invitation_Status']])
    
    # Visualisasi karakteristik cluster
    fig = go.Figure()
    
    # Tambahkan bar untuk setiap cluster
    for i, cluster in enumerate(sorted(rfm['Cluster'].unique())):
        cluster_data = rfm[rfm['Cluster'] == cluster]
        
        invitation_status = '✅ Invited' if cluster in top_clusters else '❌ Not Invited'
        marker_color = '#003366' if cluster in top_clusters else '#999999'
        
        fig.add_trace(go.Bar(
            x=['Recency (days)', 'Frequency', 'Monetary (in millions)'],
            y=[
                cluster_data['Recency'].mean(),
                cluster_data['Frequency'].mean(),
                cluster_data['Monetary'].mean() / 1000000  # Convert to millions
            ],
            name=f'Cluster {cluster} ({invitation_status})',
            marker_color=marker_color
        ))
    
    fig.update_layout(
        title="Cluster Characteristics Comparison",
        xaxis_title="Metrics",
        yaxis_title="Average Value",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download link
    csv = rfm.to_csv(index=False)
    st.download_button(
        label="Download Segmentation Results",
        data=csv,
        file_name="segmentation_results.csv",
        mime="text/csv"
    )
    
    # Hints for next steps
    st.info("You can now proceed to the Promo Mapping section to create targeted promotional strategies based on these segments.")
