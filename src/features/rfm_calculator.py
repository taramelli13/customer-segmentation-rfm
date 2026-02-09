import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def calculate_rfm_features(
    df,
    customer_id_col='CustomerID',
    invoice_date_col='InvoiceDate',
    invoice_no_col='InvoiceNo',
    total_price_col='TotalPrice',
    quantity_col='Quantity',
    product_col='StockCode',
    analysis_date=None
):
    logger.info("Starting RFM feature calculation...")
    logger.info(f"Input data shape: {df.shape}")
    logger.info(f"Unique customers: {df[customer_id_col].nunique():,}")

    # Set analysis date
    if analysis_date is None:
        analysis_date = df[invoice_date_col].max()
        logger.info(f"Using max date as analysis date: {analysis_date}")
    elif isinstance(analysis_date, str):
        analysis_date = pd.to_datetime(analysis_date)

    # Calculate RFM metrics
    rfm_df = df.groupby(customer_id_col).agg({
        invoice_date_col: ['min', 'max', 'count'],
        invoice_no_col: 'nunique',
        total_price_col: ['sum', 'mean', 'std'],
        quantity_col: ['sum', 'mean'],
        product_col: 'nunique'
    })

    # Flatten column names
    rfm_df.columns = ['_'.join(col).strip() for col in rfm_df.columns.values]
    rfm_df = rfm_df.reset_index()

    # Rename columns
    column_mapping = {
        f'{invoice_date_col}_min': 'FirstPurchaseDate',
        f'{invoice_date_col}_max': 'LastPurchaseDate',
        f'{invoice_date_col}_count': 'TransactionCount',
        f'{invoice_no_col}_nunique': 'Frequency',
        f'{total_price_col}_sum': 'Monetary',
        f'{total_price_col}_mean': 'AvgOrderValue',
        f'{total_price_col}_std': 'OrderValueStd',
        f'{quantity_col}_sum': 'TotalUnits',
        f'{quantity_col}_mean': 'AvgUnitsPerTransaction',
        f'{product_col}_nunique': 'ProductDiversity'
    }
    rfm_df = rfm_df.rename(columns=column_mapping)

    # Calculate Recency (days since last purchase)
    rfm_df['Recency'] = (analysis_date - rfm_df['LastPurchaseDate']).dt.days

    # Calculate Customer Tenure (days since first purchase)
    rfm_df['CustomerTenure'] = (analysis_date - rfm_df['FirstPurchaseDate']).dt.days

    # Calculate Purchase Intervals (vectorized)
    rfm_df = calculate_purchase_intervals(df, rfm_df, customer_id_col, invoice_date_col)

    # Calculate derived metrics
    rfm_df['AvgItemsPerTransaction'] = rfm_df['TotalUnits'] / rfm_df['Frequency']

    # Fill NaN values in OrderValueStd (for customers with only one transaction)
    rfm_df['OrderValueStd'] = rfm_df['OrderValueStd'].fillna(0)

    # Add RFM scoring
    rfm_df = add_rfm_scores(rfm_df)

    # Sort by Monetary value descending
    rfm_df = rfm_df.sort_values('Monetary', ascending=False).reset_index(drop=True)

    logger.info(f"RFM features calculated for {len(rfm_df):,} customers")
    logger.info(f"Feature columns: {list(rfm_df.columns)}")

    return rfm_df


def calculate_purchase_intervals(df, rfm_df, customer_id_col, invoice_date_col):
    # Vectorized: sort by customer and date, compute diff within each group
    sorted_df = df[[customer_id_col, invoice_date_col]].sort_values(
        [customer_id_col, invoice_date_col]
    )

    # Compute date diff within each customer group
    sorted_df['_date_diff'] = sorted_df.groupby(customer_id_col)[invoice_date_col].diff().dt.days

    # Aggregate per customer
    intervals = sorted_df.groupby(customer_id_col)['_date_diff'].agg(
        AvgPurchaseInterval='mean',
        StdPurchaseInterval='std'
    ).reset_index()

    # Fill NaN (single-purchase customers or single-date customers)
    intervals['AvgPurchaseInterval'] = intervals['AvgPurchaseInterval'].fillna(0)
    intervals['StdPurchaseInterval'] = intervals['StdPurchaseInterval'].fillna(0)

    rfm_df = rfm_df.merge(intervals, on=customer_id_col, how='left')

    return rfm_df


def safe_qcut(series, q, ascending=True):
    try:
        n_bins = q
        bins = pd.qcut(series, q=n_bins, duplicates='drop')
        n_actual = bins.nunique()
        if ascending:
            labels = list(range(1, n_actual + 1))
        else:
            labels = list(range(n_actual, 0, -1))
        return pd.qcut(series, q=n_bins, labels=labels, duplicates='drop').astype(int)
    except Exception:
        # Fallback: rank percentile
        pct = series.rank(pct=True)
        if ascending:
            return pd.cut(pct, bins=q, labels=list(range(1, q + 1)), include_lowest=True).astype(int)
        else:
            return pd.cut(pct, bins=q, labels=list(range(q, 0, -1)), include_lowest=True).astype(int)


def add_rfm_scores(rfm_df, quantiles=5):
    rfm_df['R_Score'] = safe_qcut(rfm_df['Recency'], quantiles, ascending=False)
    rfm_df['F_Score'] = safe_qcut(rfm_df['Frequency'], quantiles, ascending=True)
    rfm_df['M_Score'] = safe_qcut(rfm_df['Monetary'], quantiles, ascending=True)

    # Combined RFM Score
    rfm_df['RFM_Score'] = (
        rfm_df['R_Score'].astype(str) +
        rfm_df['F_Score'].astype(str) +
        rfm_df['M_Score'].astype(str)
    )

    # Numeric RFM Score (sum)
    rfm_df['RFM_Score_Sum'] = rfm_df['R_Score'] + rfm_df['F_Score'] + rfm_df['M_Score']

    logger.info("RFM scores added (1-5 scale per metric)")

    return rfm_df
