import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def clean_data(
    df,
    cancelled_prefix='C',
    price_floor=0.01,
    outlier_method='percentile',
    quantity_cap_percentile=99
):
    # Regras de config/config.yaml (secao preprocessing)
    initial_rows = len(df)
    logger.info(f"Starting data cleaning: {initial_rows:,} rows")

    df = df.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    df = remove_missing_customerid(df)
    df = remove_cancelled_invoices(df, prefix=cancelled_prefix)
    df = remove_invalid_prices(df, price_floor=price_floor)
    df = handle_invalid_quantities(df)
    df = remove_duplicates(df)
    df = handle_outliers(df, method=outlier_method,
                         quantity_cap_percentile=quantity_cap_percentile)
    df = create_total_price(df)
    df = extract_date_features(df)

    df = df.reset_index(drop=True)

    retained = len(df) / initial_rows * 100 if initial_rows else 0
    logger.info(f"Cleaning finished: {initial_rows:,} -> {len(df):,} rows ({retained:.1f}% retained)")

    return df


def remove_missing_customerid(df, customer_id_col='CustomerID'):
    n_missing = df[customer_id_col].isna().sum()
    logger.info(f"Removing {n_missing:,} rows without {customer_id_col}")
    return df.dropna(subset=[customer_id_col])


def remove_cancelled_invoices(df, invoice_col='InvoiceNo', prefix='C'):
    df = df.copy()
    df[invoice_col] = df[invoice_col].astype(str)
    cancelled = df[invoice_col].str.startswith(prefix)
    logger.info(f"Removing {cancelled.sum():,} cancelled invoices (prefix '{prefix}')")
    return df[~cancelled]


def remove_invalid_prices(df, price_col='UnitPrice', price_floor=0.01):
    invalid = df[price_col] < price_floor
    logger.info(f"Removing {invalid.sum():,} rows with {price_col} < {price_floor}")
    return df[~invalid]


def handle_invalid_quantities(df, quantity_col='Quantity'):
    invalid = df[quantity_col] <= 0
    logger.info(f"Removing {invalid.sum():,} rows with {quantity_col} <= 0")
    return df[~invalid]


def remove_duplicates(df):
    n_duplicates = df.duplicated().sum()
    logger.info(f"Removing {n_duplicates:,} duplicate rows")
    return df.drop_duplicates()


def handle_outliers(df, method='percentile', quantity_cap_percentile=99,
                    columns=('Quantity', 'UnitPrice')):
    df = df.copy()

    for col in columns:
        if method == 'percentile':
            upper_q = quantity_cap_percentile / 100
            lower, upper = df[col].quantile([1 - upper_q, upper_q])
            df[col] = df[col].astype(float).clip(lower, upper)
        elif method == 'iqr':
            df[col] = cap_outliers_iqr(df[col])
        else:
            raise ValueError(f"Unknown outlier method: {method}")

    logger.info(f"Outliers capped in {list(columns)} using '{method}' method")

    return df


def cap_outliers_iqr(series, factor=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return series.astype(float).clip(q1 - factor * iqr, q3 + factor * iqr)


def create_total_price(df, quantity_col='Quantity', price_col='UnitPrice'):
    df = df.copy()
    df['TotalPrice'] = df[quantity_col] * df[price_col]
    return df


def extract_date_features(df, date_col='InvoiceDate'):
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    df['Year'] = dates.dt.year
    df['Month'] = dates.dt.month
    df['Day'] = dates.dt.day
    df['DayOfWeek'] = dates.dt.day_of_week
    df['Hour'] = dates.dt.hour
    df['Date'] = dates.dt.date
    return df


def get_data_quality_report(df):
    missing = df.isna().sum()
    return {
        'total_rows': len(df),
        'total_columns': df.shape[1],
        'missing_values': missing[missing > 0].to_dict(),
        'missing_pct': (missing[missing > 0] / max(len(df), 1) * 100).round(2).to_dict(),
        'duplicate_rows': int(df.duplicated().sum()),
        'dtypes': df.dtypes.astype(str).to_dict(),
    }
