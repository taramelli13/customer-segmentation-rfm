import pytest
import numpy as np
import pandas as pd
from src.features.rfm_calculator import (
    calculate_rfm_features,
    safe_qcut,
    add_rfm_scores,
)


def _make_transactions(n_customers=10, n_per_customer=20):
    np.random.seed(42)
    dates = pd.date_range('2011-01-01', periods=365, freq='D')
    rows = []
    for cid in range(1, n_customers + 1):
        for _ in range(n_per_customer):
            rows.append({
                'CustomerID': float(cid),
                'InvoiceNo': str(np.random.randint(500000, 600000)),
                'InvoiceDate': np.random.choice(dates),
                'TotalPrice': round(np.random.uniform(5, 200), 2),
                'Quantity': np.random.randint(1, 30),
                'StockCode': np.random.choice(['A', 'B', 'C', 'D', 'E']),
            })
    return pd.DataFrame(rows)


class TestCalculateRFMFeatures:
    def test_output_columns(self):
        df = _make_transactions()
        analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        result = calculate_rfm_features(df, analysis_date=analysis_date)
        expected_cols = [
            'CustomerID', 'Recency', 'Frequency', 'Monetary',
            'AvgOrderValue', 'OrderValueStd', 'TotalUnits',
            'ProductDiversity', 'CustomerTenure', 'AvgPurchaseInterval',
            'R_Score', 'F_Score', 'M_Score', 'RFM_Score',
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_one_customer(self):
        df = pd.DataFrame({
            'CustomerID': [1.0, 1.0, 1.0],
            'InvoiceNo': ['500001', '500002', '500003'],
            'InvoiceDate': pd.to_datetime(['2011-01-01', '2011-02-01', '2011-03-01']),
            'TotalPrice': [100.0, 200.0, 150.0],
            'Quantity': [5, 10, 7],
            'StockCode': ['A', 'B', 'A'],
        })
        result = calculate_rfm_features(df, analysis_date=pd.Timestamp('2011-04-01'))
        assert len(result) == 1
        assert result.iloc[0]['Recency'] == 31  # days from March 1 to April 1

    def test_single_purchase_customer(self):
        df = pd.DataFrame({
            'CustomerID': [1.0],
            'InvoiceNo': ['500001'],
            'InvoiceDate': pd.to_datetime(['2011-06-01']),
            'TotalPrice': [50.0],
            'Quantity': [3],
            'StockCode': ['A'],
        })
        result = calculate_rfm_features(df, analysis_date=pd.Timestamp('2011-07-01'))
        assert len(result) == 1
        assert result.iloc[0]['Frequency'] == 1
        # AvgPurchaseInterval should be 0 (fillna(0) for single purchase)
        assert result.iloc[0]['AvgPurchaseInterval'] == 0

    def test_analysis_date_none_uses_max(self):
        df = _make_transactions(n_customers=3, n_per_customer=5)
        result = calculate_rfm_features(df, analysis_date=None)
        assert (result['Recency'] >= 0).all()

    def test_analysis_date_as_datetime(self):
        df = _make_transactions(n_customers=3, n_per_customer=5)
        result = calculate_rfm_features(df, analysis_date=pd.Timestamp('2012-01-01'))
        assert (result['Recency'] >= 0).all()

    def test_n_customers_matches(self):
        df = _make_transactions(n_customers=10, n_per_customer=5)
        result = calculate_rfm_features(df)
        assert len(result) == 10


class TestSafeQcut:
    def test_normal_distribution_5_bins(self):
        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 1, 1000))
        result = safe_qcut(series, 5, ascending=True)
        assert set(result.unique()).issubset({1, 2, 3, 4, 5})

    def test_ties_handled(self):
        series = pd.Series([1, 1, 1, 1, 2, 2, 3, 3, 4, 5])
        result = safe_qcut(series, 5, ascending=True)
        assert len(result) == len(series)
        # All values should be valid integers
        assert result.dtype in [np.int32, np.int64, int]

    def test_descending_order(self):
        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 1, 1000))
        result = safe_qcut(series, 5, ascending=False)
        # Higher values in series should get lower scores
        assert result[series.idxmax()] <= result[series.idxmin()]

    def test_constant_values_fallback(self):
        series = pd.Series([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        result = safe_qcut(series, 5, ascending=True)
        assert len(result) == 10


class TestAddRFMScores:
    def test_scores_range_1_to_5(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'Recency': np.random.randint(1, 365, 100),
            'Frequency': np.random.randint(1, 50, 100),
            'Monetary': np.random.uniform(10, 5000, 100),
        })
        result = add_rfm_scores(df, quantiles=5)
        for col in ['R_Score', 'F_Score', 'M_Score']:
            assert result[col].min() >= 1
            assert result[col].max() <= 5

    def test_rfm_score_concatenation(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'Recency': np.random.randint(1, 365, 100),
            'Frequency': np.random.randint(1, 50, 100),
            'Monetary': np.random.uniform(10, 5000, 100),
        })
        result = add_rfm_scores(df, quantiles=5)
        assert 'RFM_Score' in result.columns
        assert 'RFM_Score_Sum' in result.columns
        # RFM_Score should be 3-digit string
        assert all(len(s) == 3 for s in result['RFM_Score'])

    def test_all_same_frequency(self):
        df = pd.DataFrame({
            'Recency': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'Frequency': [5] * 10,
            'Monetary': np.arange(100, 1100, 100, dtype=float),
        })
        result = add_rfm_scores(df, quantiles=5)
        assert 'F_Score' in result.columns
        assert len(result) == 10
