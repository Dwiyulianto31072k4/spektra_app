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
        if df.loc[
