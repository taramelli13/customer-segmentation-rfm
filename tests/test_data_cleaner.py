import pytest
import numpy as np
import pandas as pd
from src.data.data_cleaner import (
    clean_data,
    remove_missing_customerid,
    remove_cancelled_invoices,
    remove_invalid_prices,
    handle_invalid_quantities,
    create_total_price,
    handle_outliers,
    extract_date_features,
    remove_duplicates,
    get_data_quality_report,
    cap_outliers_iqr,
)


class TestCleanData:
    def test_pipeline_reduces_rows(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        assert len(result) < len(sample_raw_df)

    def test_pipeline_no_nan_customerid(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        assert result['CustomerID'].isna().sum() == 0

    def test_pipeline_no_cancelled_invoices(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        assert not result['InvoiceNo'].astype(str).str.startswith('C').any()

    def test_pipeline_has_totalprice(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        assert 'TotalPrice' in result.columns

    def test_pipeline_has_date_features(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        for col in ['Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 'Date']:
            assert col in result.columns


class TestRemoveMissingCustomerID:
    def test_removes_nan(self):
        df = pd.DataFrame({
            'CustomerID': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        result = remove_missing_customerid(df)
        assert len(result) == 3
        assert result['CustomerID'].isna().sum() == 0

    def test_no_nan_unchanged(self):
        df = pd.DataFrame({'CustomerID': [1.0, 2.0, 3.0]})
        result = remove_missing_customerid(df)
        assert len(result) == 3


class TestRemoveCancelledInvoices:
    def test_filters_c_prefix(self):
        df = pd.DataFrame({
            'InvoiceNo': ['500001', 'C500002', '500003', 'C500004']
        })
        result = remove_cancelled_invoices(df, prefix='C')
        assert len(result) == 2
        assert not result['InvoiceNo'].str.startswith('C').any()

    def test_no_cancellations_unchanged(self):
        df = pd.DataFrame({'InvoiceNo': ['500001', '500002']})
        result = remove_cancelled_invoices(df)
        assert len(result) == 2


class TestRemoveInvalidPrices:
    def test_removes_zero_price(self):
        df = pd.DataFrame({'UnitPrice': [0.0, 1.5, 0.01, -0.5, 5.0]})
        result = remove_invalid_prices(df, price_floor=0.01)
        assert len(result) == 3

    def test_boundary_at_floor(self):
        df = pd.DataFrame({'UnitPrice': [0.01, 0.009, 0.011]})
        result = remove_invalid_prices(df, price_floor=0.01)
        assert len(result) == 2  # 0.01 and 0.011


class TestHandleInvalidQuantities:
    def test_removes_zero_and_negative(self):
        df = pd.DataFrame({'Quantity': [5, 0, -2, 10, -1]})
        result = handle_invalid_quantities(df)
        assert len(result) == 2
        assert (result['Quantity'] > 0).all()


class TestCreateTotalPrice:
    def test_correct_calculation(self):
        df = pd.DataFrame({'Quantity': [2, 5, 10], 'UnitPrice': [3.0, 2.0, 1.5]})
        result = create_total_price(df)
        expected = [6.0, 10.0, 15.0]
        np.testing.assert_array_almost_equal(result['TotalPrice'].values, expected)


class TestHandleOutliers:
    def test_percentile_capping(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'Quantity': np.concatenate([np.random.randint(1, 20, 98), [1000, 2000]]).astype(float),
            'UnitPrice': np.concatenate([np.random.uniform(1, 10, 98), [500.0, 1000.0]]),
            'TotalPrice': np.zeros(100)
        })
        result = handle_outliers(df, method='percentile', quantity_cap_percentile=99)
        assert result['Quantity'].max() < 2000

    def test_iqr_capping(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'Quantity': np.concatenate([np.random.randint(1, 20, 98), [1000, 2000]]),
            'UnitPrice': np.concatenate([np.random.uniform(1, 10, 98), [500.0, 1000.0]]),
            'TotalPrice': np.zeros(100)
        })
        result = handle_outliers(df, method='iqr')
        assert result['Quantity'].max() < 2000


class TestExtractDateFeatures:
    def test_creates_six_columns(self):
        df = pd.DataFrame({
            'InvoiceDate': pd.to_datetime(['2011-01-01 10:30', '2011-06-15 14:00'])
        })
        result = extract_date_features(df)
        assert result['Year'].tolist() == [2011, 2011]
        assert result['Month'].tolist() == [1, 6]
        assert result['Day'].tolist() == [1, 15]
        assert result['Hour'].tolist() == [10, 14]
        assert 'DayOfWeek' in result.columns
        assert 'Date' in result.columns


class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({'A': [1, 1, 2, 3], 'B': ['x', 'x', 'y', 'z']})
        result = remove_duplicates(df)
        assert len(result) == 3

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        result = remove_duplicates(df)
        assert len(result) == 3


class TestGetDataQualityReport:
    def test_returns_dict_with_keys(self, sample_raw_df):
        report = get_data_quality_report(sample_raw_df)
        assert isinstance(report, dict)
        assert 'total_rows' in report
        assert 'total_columns' in report
        assert 'missing_values' in report
        assert 'duplicate_rows' in report

    def test_total_rows_correct(self, sample_raw_df):
        report = get_data_quality_report(sample_raw_df)
        assert report['total_rows'] == len(sample_raw_df)


class TestEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame({
            'InvoiceNo': pd.Series(dtype=str),
            'StockCode': pd.Series(dtype=str),
            'Description': pd.Series(dtype=str),
            'Quantity': pd.Series(dtype=int),
            'InvoiceDate': pd.Series(dtype='datetime64[ns]'),
            'UnitPrice': pd.Series(dtype=float),
            'CustomerID': pd.Series(dtype=float),
            'Country': pd.Series(dtype=str),
        })
        result = clean_data(df)
        assert len(result) == 0
