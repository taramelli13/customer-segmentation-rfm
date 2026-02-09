import os
import sys
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def sample_raw_df():
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2011-01-01', periods=365, freq='D')
    df = pd.DataFrame({
        'InvoiceNo': [str(500000 + i) for i in range(n)],
        'StockCode': np.random.choice(['85123A', '71053', '84406B', '22752', '21730'], n),
        'Description': ['Widget'] * n,
        'Quantity': np.random.randint(1, 50, n),
        'InvoiceDate': np.random.choice(dates, n),
        'UnitPrice': np.round(np.random.uniform(0.5, 20.0, n), 2),
        'CustomerID': np.random.choice([12345.0, 12346.0, 12347.0, 12348.0, 12349.0,
                                        12350.0, 12351.0, 12352.0, 12353.0, 12354.0], n),
        'Country': ['United Kingdom'] * n
    })
    # Add some NaN CustomerIDs
    df.loc[0:4, 'CustomerID'] = np.nan
    # Add a cancelled invoice
    df.loc[5, 'InvoiceNo'] = 'C500005'
    df.loc[5, 'Quantity'] = -3
    # Add an invalid price
    df.loc[6, 'UnitPrice'] = 0.0
    # Add a zero quantity
    df.loc[7, 'Quantity'] = 0
    # Add a duplicate
    df = pd.concat([df, df.iloc[10:11]], ignore_index=True)
    return df


@pytest.fixture
def sample_clean_df(sample_raw_df):
    from src.data.data_cleaner import clean_data
    return clean_data(sample_raw_df)


@pytest.fixture
def sample_rfm_df(sample_clean_df):
    from src.features.rfm_calculator import calculate_rfm_features
    analysis_date = sample_clean_df['InvoiceDate'].max() + pd.Timedelta(days=1)
    return calculate_rfm_features(sample_clean_df, analysis_date=analysis_date)


@pytest.fixture
def sample_scaled_array(sample_rfm_df):
    from sklearn.preprocessing import StandardScaler
    features = ['Recency', 'Frequency', 'Monetary']
    X = sample_rfm_df[features].copy()
    X_log = X.apply(np.log1p)
    scaler = StandardScaler()
    return scaler.fit_transform(X_log)


@pytest.fixture
def tmp_output_dir(tmp_path):
    models_dir = tmp_path / 'models'
    models_dir.mkdir()
    figures_dir = tmp_path / 'figures'
    figures_dir.mkdir()
    return tmp_path
