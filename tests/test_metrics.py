import pytest
import numpy as np
import pandas as pd
from src.evaluation.metrics import (
    evaluate_clustering,
    profile_clusters,
    get_cluster_summary,
    identify_cluster_characteristics,
    calculate_cluster_stability,
    validate_clustering,
)


def _make_clustered_data(n=100, k=3):
    np.random.seed(42)
    X = np.random.randn(n, 3)
    labels = np.repeat(np.arange(k), n // k)
    if len(labels) < n:
        labels = np.concatenate([labels, np.zeros(n - len(labels), dtype=int)])
    # Shift cluster centers so metrics are meaningful
    for i in range(k):
        X[labels == i] += i * 3
    return X, labels


class TestEvaluateClustering:
    def test_returns_all_metrics(self):
        X, labels = _make_clustered_data()
        scores = evaluate_clustering(X, labels)
        assert 'silhouette' in scores
        assert 'davies_bouldin' in scores
        assert 'calinski_harabasz' in scores

    def test_silhouette_in_range(self):
        X, labels = _make_clustered_data()
        scores = evaluate_clustering(X, labels)
        assert -1 <= scores['silhouette'] <= 1

    def test_less_than_2_clusters_returns_nan(self):
        X = np.random.randn(50, 3)
        labels = np.zeros(50, dtype=int)
        scores = evaluate_clustering(X, labels)
        assert np.isnan(scores['silhouette'])

    def test_specific_metrics(self):
        X, labels = _make_clustered_data()
        scores = evaluate_clustering(X, labels, metric_names=['silhouette'])
        assert 'silhouette' in scores
        assert 'davies_bouldin' not in scores


class TestProfileClusters:
    def test_aggregates_by_cluster(self):
        df = pd.DataFrame({
            'Recency': [10, 20, 100, 200],
            'Frequency': [50, 40, 5, 3],
            'Monetary': [1000, 900, 100, 50],
        })
        labels = np.array([0, 0, 1, 1])
        result = profile_clusters(df, labels, ['Recency', 'Frequency', 'Monetary'])
        assert len(result) == 2  # 2 clusters


class TestGetClusterSummary:
    def test_percentages_sum_to_100(self):
        df = pd.DataFrame({
            'Recency': np.random.randint(1, 365, 100),
            'Frequency': np.random.randint(1, 50, 100),
            'Monetary': np.random.uniform(50, 5000, 100),
        })
        labels = np.random.choice([0, 1, 2], 100)
        summary = get_cluster_summary(df, labels)
        assert abs(summary['Percentage'].sum() - 100.0) < 0.1


class TestIdentifyClusterCharacteristics:
    def test_valid_segment_names(self):
        summary = pd.DataFrame({
            'Cluster': [0, 1, 2],
            'Recency_mean': [10, 200, 50],
            'Frequency_mean': [50, 2, 10],
            'Monetary_mean': [5000, 100, 500],
            'Recency_count': [100, 200, 150],
            'Percentage': [22.2, 44.4, 33.3],
        })
        result = identify_cluster_characteristics(summary)
        valid_names = {'Champions', 'Loyal Customers', 'Big Spenders',
                       'At Risk', 'New Customers', 'Hibernating', 'Others'}
        for name in result['Suggested_Name']:
            assert name in valid_names, f"Invalid segment name: {name}"


class TestCalculateClusterStability:
    def test_ari_and_silhouette(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'Recency': np.random.randint(1, 365, 100),
            'Frequency': np.random.randint(1, 50, 100),
            'Monetary': np.random.uniform(50, 5000, 100),
        })
        result = calculate_cluster_stability(df, ['Recency', 'Frequency', 'Monetary'],
                                             n_clusters=3, n_runs=5, random_state_base=42)
        assert 'mean_ari' in result
        assert 'mean_silhouette' in result
        assert 0 <= result['mean_ari'] <= 1


class TestValidateClustering:
    def test_pass(self):
        scores = {'silhouette': 0.5, 'davies_bouldin': 1.0}
        assert validate_clustering(scores, min_silhouette=0.4, max_davies_bouldin=1.5) is True

    def test_fail_silhouette(self):
        scores = {'silhouette': 0.2, 'davies_bouldin': 1.0}
        assert validate_clustering(scores, min_silhouette=0.4, max_davies_bouldin=1.5) is False

    def test_fail_davies_bouldin(self):
        scores = {'silhouette': 0.5, 'davies_bouldin': 2.0}
        assert validate_clustering(scores, min_silhouette=0.4, max_davies_bouldin=1.5) is False

    def test_empty_scores(self):
        assert validate_clustering({}) is False


class TestEdgeCases:
    def test_labels_with_noise(self):
        X = np.random.randn(60, 3)
        labels = np.array([0]*20 + [1]*20 + [-1]*20)
        scores = evaluate_clustering(X, labels)
        # Should still compute for the 2 valid clusters
        assert 'silhouette' in scores
