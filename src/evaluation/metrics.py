"""
Evaluation metrics and cluster profiling for clustering
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.metrics.pairwise import pairwise_distances
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    metric_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate clustering quality using multiple metrics.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature data
    labels : np.ndarray
        Cluster labels
    metric_names : list, optional
        Specific metrics to calculate. Default: all

    Returns
    -------
    dict
        Dictionary of metric scores

    Examples
    --------
    >>> scores = evaluate_clustering(X_scaled, labels)
    >>> print(f"Silhouette: {scores['silhouette']:.3f}")
    """
    if metric_names is None:
        metric_names = ['silhouette', 'davies_bouldin', 'calinski_harabasz']

    scores = {}

    # Check for valid clustering
    n_clusters = len(np.unique(labels[labels >= 0]))  # Exclude noise points (-1)

    if n_clusters < 2:
        logger.warning("Less than 2 clusters found. Cannot calculate metrics.")
        return {metric: np.nan for metric in metric_names}

    # Calculate each metric
    if 'silhouette' in metric_names:
        try:
            scores['silhouette'] = silhouette_score(X, labels)
        except Exception as e:
            logger.error(f"Error calculating silhouette score: {e}")
            scores['silhouette'] = np.nan

    if 'davies_bouldin' in metric_names:
        try:
            scores['davies_bouldin'] = davies_bouldin_score(X, labels)
        except Exception as e:
            logger.error(f"Error calculating Davies-Bouldin score: {e}")
            scores['davies_bouldin'] = np.nan

    if 'calinski_harabasz' in metric_names:
        try:
            scores['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        except Exception as e:
            logger.error(f"Error calculating Calinski-Harabasz score: {e}")
            scores['calinski_harabasz'] = np.nan

    logger.info("Evaluation scores:")
    for metric, score in scores.items():
        logger.info(f"  {metric}: {score:.4f}")

    return scores


def calculate_silhouette_per_cluster(
    X: np.ndarray,
    labels: np.ndarray
) -> pd.DataFrame:
    """
    Calculate silhouette scores per cluster.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature data
    labels : np.ndarray
        Cluster labels

    Returns
    -------
    pd.DataFrame
        Silhouette scores per cluster
    """
    silhouette_vals = silhouette_samples(X, labels)

    results = []
    for cluster in np.unique(labels):
        if cluster >= 0:  # Exclude noise
            cluster_silhouette = silhouette_vals[labels == cluster]
            results.append({
                'Cluster': cluster,
                'Mean_Silhouette': cluster_silhouette.mean(),
                'Std_Silhouette': cluster_silhouette.std(),
                'Min_Silhouette': cluster_silhouette.min(),
                'Max_Silhouette': cluster_silhouette.max(),
                'Size': len(cluster_silhouette)
            })

    df = pd.DataFrame(results)
    return df


def profile_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
    label_col: str = 'Cluster'
) -> pd.DataFrame:
    """
    Create detailed profiles for each cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Original feature dataframe
    labels : np.ndarray
        Cluster labels
    feature_cols : list
        Features to profile
    label_col : str
        Name for cluster label column

    Returns
    -------
    pd.DataFrame
        Cluster profiles with statistics
    """
    df_with_labels = df.copy()
    df_with_labels[label_col] = labels

    # Calculate statistics per cluster
    profiles = df_with_labels.groupby(label_col)[feature_cols].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ])

    return profiles


def get_cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get high-level summary statistics per cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe
    labels : np.ndarray
        Cluster labels
    feature_cols : list, optional
        Features to summarize. If None, uses RFM columns.

    Returns
    -------
    pd.DataFrame
        Summary statistics per cluster
    """
    if feature_cols is None:
        feature_cols = ['Recency', 'Frequency', 'Monetary']

    df_with_labels = df.copy()
    df_with_labels['Cluster'] = labels

    summary = df_with_labels.groupby('Cluster')[feature_cols].agg(['mean', 'median', 'count'])

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Calculate percentage of total customers
    summary['Percentage'] = (summary['Recency_count'] / summary['Recency_count'].sum() * 100)

    logger.info(f"\nCluster Summary:\n{summary}")

    return summary


def identify_cluster_characteristics(
    summary: pd.DataFrame,
    feature_prefix: str = ''
) -> pd.DataFrame:
    """
    Identify key characteristics of each cluster for business interpretation.

    Parameters
    ----------
    summary : pd.DataFrame
        Cluster summary from get_cluster_summary()
    feature_prefix : str
        Prefix for feature columns (e.g., 'mean_' or '')

    Returns
    -------
    pd.DataFrame
        Cluster characteristics
    """
    characteristics = []

    for idx, row in summary.iterrows():
        cluster = row['Cluster']

        # Determine characteristics based on RFM values
        r_col = f'{feature_prefix}Recency_mean' if f'{feature_prefix}Recency_mean' in row else 'Recency'
        f_col = f'{feature_prefix}Frequency_mean' if f'{feature_prefix}Frequency_mean' in row else 'Frequency'
        m_col = f'{feature_prefix}Monetary_mean' if f'{feature_prefix}Monetary_mean' in row else 'Monetary'

        recency = row[r_col]
        frequency = row[f_col]
        monetary = row[m_col]

        # Get quartiles for comparison
        r_q75 = summary[r_col].quantile(0.75)
        r_q25 = summary[r_col].quantile(0.25)
        f_q75 = summary[f_col].quantile(0.75)
        f_q25 = summary[f_col].quantile(0.25)
        m_q75 = summary[m_col].quantile(0.75)
        m_q25 = summary[m_col].quantile(0.25)

        # Classify
        r_class = 'Low' if recency <= r_q25 else ('High' if recency >= r_q75 else 'Medium')
        f_class = 'Low' if frequency <= f_q25 else ('High' if frequency >= f_q75 else 'Medium')
        m_class = 'Low' if monetary <= m_q25 else ('High' if monetary >= m_q75 else 'Medium')

        # Suggest segment name
        if r_class == 'Low' and f_class == 'High' and m_class == 'High':
            segment_name = 'Champions'
        elif r_class in ['Low', 'Medium'] and f_class == 'High':
            segment_name = 'Loyal Customers'
        elif m_class == 'High' and f_class == 'Low':
            segment_name = 'Big Spenders'
        elif r_class == 'High' and f_class == 'High':
            segment_name = 'At Risk'
        elif r_class == 'Low' and f_class == 'Low':
            segment_name = 'New Customers'
        elif r_class == 'High' and f_class == 'Low':
            segment_name = 'Hibernating'
        else:
            segment_name = 'Others'

        characteristics.append({
            'Cluster': cluster,
            'Recency_Level': r_class,
            'Frequency_Level': f_class,
            'Monetary_Level': m_class,
            'Suggested_Name': segment_name,
            'Size': row.get('Recency_count', 0),
            'Percentage': row.get('Percentage', 0)
        })

    return pd.DataFrame(characteristics)


def calculate_cluster_stability(
    X: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int,
    n_runs: int = 10,
    random_state_base: int = 42
) -> Dict:
    """
    Calculate cluster stability using multiple runs with different random states.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data
    feature_cols : list
        Features to use
    n_clusters : int
        Number of clusters
    n_runs : int
        Number of runs to test stability
    random_state_base : int
        Base random state

    Returns
    -------
    dict
        Stability metrics
    """
    from src.models.clustering import CustomerSegmentation
    from sklearn.metrics import adjusted_rand_score

    logger.info(f"Testing cluster stability over {n_runs} runs...")

    all_labels = []
    all_scores = []

    for run in range(n_runs):
        model = CustomerSegmentation(
            algorithm='kmeans',
            n_clusters=n_clusters,
            random_state=random_state_base + run
        )
        model.fit(X, feature_cols)

        all_labels.append(model.labels_)

        # Get scaled data for scoring
        X_scaled = model.scaler.transform(X[feature_cols])
        sil_score = silhouette_score(X_scaled, model.labels_)
        all_scores.append(sil_score)

    # Calculate pairwise ARI (Adjusted Rand Index)
    ari_scores = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            ari = adjusted_rand_score(all_labels[i], all_labels[j])
            ari_scores.append(ari)

    stability = {
        'mean_ari': np.mean(ari_scores),
        'std_ari': np.std(ari_scores),
        'min_ari': np.min(ari_scores),
        'max_ari': np.max(ari_scores),
        'mean_silhouette': np.mean(all_scores),
        'std_silhouette': np.std(all_scores)
    }

    logger.info(f"Cluster Stability (ARI): {stability['mean_ari']:.4f} ± {stability['std_ari']:.4f}")
    logger.info(f"Silhouette Stability: {stability['mean_silhouette']:.4f} ± {stability['std_silhouette']:.4f}")

    return stability


def validate_clustering(
    scores: Dict[str, float],
    min_silhouette: float = 0.4,
    max_davies_bouldin: float = 1.5
) -> bool:
    """
    Validate clustering quality against thresholds.

    Parameters
    ----------
    scores : dict
        Evaluation scores
    min_silhouette : float
        Minimum acceptable silhouette score
    max_davies_bouldin : float
        Maximum acceptable Davies-Bouldin score

    Returns
    -------
    bool
        True if clustering meets quality thresholds
    """
    checks = []

    # Silhouette check
    if 'silhouette' in scores:
        sil_pass = scores['silhouette'] >= min_silhouette
        checks.append(sil_pass)
        status = "✓" if sil_pass else "✗"
        logger.info(f"{status} Silhouette: {scores['silhouette']:.4f} (threshold: {min_silhouette})")

    # Davies-Bouldin check
    if 'davies_bouldin' in scores:
        db_pass = scores['davies_bouldin'] <= max_davies_bouldin
        checks.append(db_pass)
        status = "✓" if db_pass else "✗"
        logger.info(f"{status} Davies-Bouldin: {scores['davies_bouldin']:.4f} (threshold: {max_davies_bouldin})")

    valid = all(checks) if checks else False

    if valid:
        logger.info("✓ Clustering validation PASSED")
    else:
        logger.warning("✗ Clustering validation FAILED - consider different parameters")

    return valid


def generate_evaluation_report(
    X: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
    feature_cols: List[str],
    output_path: str = 'outputs/reports/evaluation_report.txt'
) -> None:
    """
    Generate comprehensive evaluation report.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature data
    labels : np.ndarray
        Cluster labels
    df : pd.DataFrame
        Original feature dataframe
    feature_cols : list
        Features used
    output_path : str
        Output file path
    """
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CLUSTERING EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Overall metrics
        f.write("OVERALL METRICS\n")
        f.write("-" * 70 + "\n")
        scores = evaluate_clustering(X, labels)
        for metric, value in scores.items():
            f.write(f"{metric:25s}: {value:.4f}\n")

        # Cluster summary
        f.write("\n" + "=" * 70 + "\n")
        f.write("CLUSTER SUMMARY\n")
        f.write("-" * 70 + "\n")
        summary = get_cluster_summary(df, labels, feature_cols)
        f.write(summary.to_string())

        # Cluster characteristics
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("CLUSTER CHARACTERISTICS\n")
        f.write("-" * 70 + "\n")
        chars = identify_cluster_characteristics(summary)
        f.write(chars.to_string())

        # Per-cluster silhouette
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("PER-CLUSTER SILHOUETTE SCORES\n")
        f.write("-" * 70 + "\n")
        sil_per_cluster = calculate_silhouette_per_cluster(X, labels)
        f.write(sil_per_cluster.to_string())

    logger.info(f"Evaluation report saved to: {output_path}")


