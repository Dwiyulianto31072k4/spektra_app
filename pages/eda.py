import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def show_eda_page():
    """
    Fungsi untuk menampilkan halaman exploratory data analysis
    """
    st.markdown('<p class="section-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload and preprocess your data first.")
        return
    
    data = st.session_state.data
    st.markdown("### Select Analysis Options")
    
    tab1, tab2, tab3 = st.tabs(["Distributions", "Customer Demographics", "Transaction Patterns"])
    
    with tab1:
        show_distribution_analysis(data)
    
    with tab2:
        show_demographic_analysis(data)
    
    with tab3:
        show_transaction_analysis(data)

def show_distribution_analysis(data):
    """
    Menampilkan analisis distribusi variabel
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data customer untuk dianalisis
    """
    st.subheader("Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Numeric column distribution
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns available for analysis.")
            return
            
        selected_num_col = st.selectbox(
            "Select numeric column for distribution analysis", 
            options=numeric_cols,
            index=numeric_cols.index('TOTAL_AMOUNT_MPF') if 'TOTAL_AMOUNT_MPF' in numeric_cols else 0
        )
        
        log_transform = st.checkbox("Apply log transformation", value=True)
        
        # Check if data is valid for visualization
        if data[selected_num_col].isnull().all():
            st.error(f"Column '{selected_num_col}' contains only null values.")
            return
            
        # Create histogram
        if log_transform and (data[selected_num_col] > 0).all():
            fig = px.histogram(
                data, 
                x=np.log1p(data[selected_num_col]), 
                title=f"Log Distribution of {selected_num_col}",
                nbins=50,
                color_discrete_sequence=['#003366']
            )
            fig.update_layout(xaxis_title=f"Log({selected_num_col})")
        else:
            fig = px.histogram(
                data, 
                x=selected_num_col, 
                title=f"Distribution of {selected_num_col}",
                nbins=50,
                color_discrete_sequence=['#003366']
            )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        st.write("**Statistics:**")
        stats = data[selected_num_col].describe()
        st.write(f"- Mean: {stats['mean']:,.2f}")
        st.write(f"- Median: {stats['50%']:,.2f}")
        st.write(f"- Min: {stats['min']:,.2f}")
        st.write(f"- Max: {stats['max']:,.2f}")
        st.write(f"- Standard Deviation: {stats['std']:,.2f}")
    
    with col2:
        # Categorical column distribution
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Tambahkan Usia_Kategori jika ada (bisa jadi tipe kategori atau object)
        if 'Usia_Kategori' in data.columns and 'Usia_Kategori' not in categorical_cols:
            categorical_cols.append('Usia_Kategori')
        
        if not categorical_cols:
            st.info("No categorical columns available for analysis.")
            return
            
        selected_cat_col = st.selectbox(
            "Select categorical column for distribution analysis", 
            options=categorical_cols,
            index=categorical_cols.index('MPF_CATEGORIES_TAKEN') if 'MPF_CATEGORIES_TAKEN' in categorical_cols else 0
        )
        
        # Count values and sort
        value_counts = data[selected_cat_col].value_counts().reset_index()
        value_counts.columns = [selected_cat_col, 'Count']
        
        # Show top N categories only
        top_n = st.slider("Show top N categories", min_value=5, max_value=30, value=10)
        
        # Create bar chart
        fig = px.bar(
            value_counts.head(top_n), 
            x=selected_cat_col, 
            y='Count',
            title=f"Top {top_n} values of {selected_cat_col}",
            color='Count',
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top categories
        st.write("**Top Categories:**")
        for i, row in value_counts.head(5).iterrows():
            st.write(f"- {row[selected_cat_col]}: {row['Count']} ({row['Count']/data.shape[0]*100:.1f}%)")

def show_demographic_analysis(data):
    """
    Menampilkan analisis demografi pelanggan
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data customer untuk dianalisis
    """
    st.subheader("Customer Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution (both numerical and categorical)
        if 'Usia' in data.columns:
            # Tab untuk pilihan tipe visualisasi usia
            age_tabs = st.tabs(["Age Categories", "Age Distribution"])
            
            with age_tabs[0]:
                # Pastikan Usia_Kategori ada
                if 'Usia_Kategori' not in data.columns:
                    # Buat kategori usia jika belum ada
                    bins = [0, 25, 35, 45, 55, 100]
                    labels = ['<25', '25-35', '35-45', '45-55', '55+']
                    data['Usia_Kategori'] = pd.cut(data['Usia'], bins=bins, labels=labels, right=False)
                
                # Plot distribusi kategori usia
                age_counts = data['Usia_Kategori'].value_counts().reset_index()
                age_counts.columns = ['Age Group', 'Count']
                
                fig = px.pie(
                    age_counts, 
                    values='Count', 
                    names='Age Group',
                    title="Customer Age Distribution",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    paper_bgcolor="white",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with age_tabs[1]:
                # Plot distribusi usia numerik
                fig = px.histogram(
                    data, 
                    x='Usia',
                    title="Customer Age Distribution",
                    nbins=20,
                    color_discrete_sequence=['#003366']
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="white",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show age statistics
            st.write("**Age Statistics:**")
            st.write(f"- Average Age: {data['Usia'].mean():.1f} years")
            st.write(f"- Median Age: {data['Usia'].median():.1f} years")
            st.write(f"- Youngest: {data['Usia'].min():.0f} years")
            st.write(f"- Oldest: {data['Usia'].max():.0f} years")
        else:
            st.info("Age data not available.")
    
    with col2:
        # Gender distribution
        if 'CUST_SEX' in data.columns:
            gender_counts = data['CUST_SEX'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            
            fig = px.pie(
                gender_counts, 
                values='Count', 
                names='Gender',
                title="Customer Gender Distribution",
                hole=0.4,
                color_discrete_sequence=['#003366', '#66b3ff']
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                paper_bgcolor="white",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show gender statistics
            st.write("**Gender Distribution:**")
            for i, row in gender_counts.iterrows():
                gender_label = "Male" if row['Gender'] == 'M' else "Female"
                st.write(f"- {gender_label}: {row['Count']} ({row['Count']/data.shape[0]*100:.1f}%)")
        else:
            st.info("Gender data not available.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Education distribution
        if 'EDU_TYPE' in data.columns:
            edu_counts = data['EDU_TYPE'].value_counts().reset_index()
            edu_counts.columns = ['Education', 'Count']
            
            fig = px.bar(
                edu_counts, 
                x='Education', 
                y='Count',
                title="Customer Education Level",
                color='Count',
                color_continuous_scale=px.colors.sequential.Blues
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="white",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Education data not available.")
    
    with col2:
        # Marital status distribution
        if 'MARITAL_STAT' in data.columns:
            marital_counts = data['MARITAL_STAT'].value_counts().reset_index()
            marital_counts.columns = ['Marital Status', 'Count']
            
            marital_labels = {
                'M': 'Married',
                'S': 'Single',
                'D': 'Divorced/Separated'
            }
            
            marital_counts['Status'] = marital_counts['Marital Status'].map(marital_labels)
            
            fig = px.pie(
                marital_counts, 
                values='Count', 
                names='Status' if 'Status' in marital_counts.columns else 'Marital Status',
                title="Customer Marital Status",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Blues
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                paper_bgcolor="white",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Marital status data not available.")

def show_transaction_analysis(data):
    """
    Menampilkan analisis pola transaksi pelanggan
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data customer untuk dianalisis
    """
    st.subheader("Transaction Patterns")
    col1, col2 = st.columns(2)
    
    with col1:
        # Product category distribution
        if 'MPF_CATEGORIES_TAKEN' in data.columns:
            # Count product categories
            product_counts = data['MPF_CATEGORIES_TAKEN'].value_counts().reset_index()
            product_counts.columns = ['Product Category', 'Count']
            
            fig = px.pie(
                product_counts, 
                values='Count', 
                names='Product Category',
                title="Product Category Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Blues
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                paper_bgcolor="white",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Product category data not available.")
    
    with col2:
        # Transaction amount vs age
        if 'TOTAL_AMOUNT_MPF' in data.columns and 'Usia' in data.columns:
            # Convert to numeric if needed
            if not pd.api.types.is_numeric_dtype(data['TOTAL_AMOUNT_MPF']):
                data['TOTAL_AMOUNT_MPF'] = pd.to_numeric(data['TOTAL_AMOUNT_MPF'], errors='coerce')
                
            # Create scatter plot
            fig = px.scatter(
                data, 
                x='Usia', 
                y='TOTAL_AMOUNT_MPF',
                title="Total Transaction Amount vs Age",
                color='TOTAL_AMOUNT_MPF',
                color_continuous_scale=px.colors.sequential.Blues,
                opacity=0.7
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="white",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Transaction amount or age data not available.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Number of products purchased distribution
        if 'TOTAL_PRODUCT_MPF' in data.columns:
            # Convert to numeric if needed
            if not pd.api.types.is_numeric_dtype(data['TOTAL_PRODUCT_MPF']):
                data['TOTAL_PRODUCT_MPF'] = pd.to_numeric(data['TOTAL_PRODUCT_MPF'], errors='coerce')
                
            # Count products purchased
            product_counts = data['TOTAL_PRODUCT_MPF'].value_counts().reset_index()
            product_counts.columns = ['Products Purchased', 'Count']
            product_counts = product_counts.sort_values('Products Purchased')
            
            fig = px.bar(
                product_counts, 
                x='Products Purchased', 
                y='Count',
                title="Number of Products Purchased Distribution",
                color='Count',
                color_continuous_scale=px.colors.sequential.Blues
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="white",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display key metrics
            st.write("**Transaction Metrics:**")
            st.write(f"- Average products per customer: {data['TOTAL_PRODUCT_MPF'].mean():.2f}")
            st.write(f"- Percentage of multi-product customers: {(data['TOTAL_PRODUCT_MPF'] > 1).mean()*100:.1f}%")
        else:
            st.info("Number of products data not available.")
    
    with col2:
        # Transaction amount by product category
        if 'TOTAL_AMOUNT_MPF' in data.columns and 'MPF_CATEGORIES_TAKEN' in data.columns:
            # Convert to numeric if needed
            if not pd.api.types.is_numeric_dtype(data['TOTAL_AMOUNT_MPF']):
                data['TOTAL_AMOUNT_MPF'] = pd.to_numeric(data['TOTAL_AMOUNT_MPF'], errors='coerce')
                
            # Group by product category
            product_amount = data.groupby('MPF_CATEGORIES_TAKEN')['TOTAL_AMOUNT_MPF'].mean().reset_index()
            product_amount.columns = ['Product Category', 'Average Amount']
            
            fig = px.bar(
                product_amount, 
                x='Product Category', 
                y='Average Amount',
                title="Average Transaction Amount by Product Category",
                color='Average Amount',
                color_continuous_scale=px.colors.sequential.Blues
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="white",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Required columns not available for this visualization.")
