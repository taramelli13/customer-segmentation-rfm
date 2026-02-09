import pytest
import os
import numpy as np
import pandas as pd
from src.models.clustering import CustomerSegmentation, find_optimal_k, compare_algorithms


def _make_rfm_df(n=100):
    np.random.seed(42)
    return pd.DataFrame({
        'Recency': np.random.randint(1, 365, n),
        'Frequency': np.random.randint(1, 50, n),
        'Monetary': np.random.uniform(50, 5000, n),
    })


FEATURES = ['Recency', 'Frequency', 'Monetary']


class TestCustomerSegmentationKMeans:
    def test_fit_predict(self):
        df = _make_rfm_df()
        model = CustomerSegmentation(algorithm='kmeans', n_clusters=3, random_state=42)
        labels = model.fit_predict(df, feature_cols=FEATURES)
        assert len(labels) == len(df)
        assert len(np.unique(labels)) == 3

    def test_predict_after_fit(self):
        df = _make_rfm_df()
        model = CustomerSegmentation(algorithm='kmeans', n_clusters=3, random_state=42)
        model.fit(df, feature_cols=FEATURES)
        labels = model.predict(df)
        assert len(labels) == len(df)

    def test_predict_before_fit_raises(self):
        model = CustomerSegmentation(algorithm='kmeans', n_clusters=3)
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(_make_rfm_df())


class TestCustomerSegmentationHierarchical:
    def test_fit_predict(self):
        df = _make_rfm_df()
        model = CustomerSegmentation(algorithm='hierarchical', n_clusters=4, random_state=42)
        labels = model.fit_predict(df, feature_cols=FEATURES)
        assert len(labels) == len(df)
        assert len(np.unique(labels)) == 4


class TestCustomerSegmentationDBSCAN:
    def test_returns_labels(self):
        df = _make_rfm_df()
        model = CustomerSegmentation(algorithm='dbscan', eps=1.5, min_samples=5, random_state=42)
        labels = model.fit_predict(df, feature_cols=FEATURES)
        assert len(labels) == len(df)


class TestScalerTypes:
    @pytest.mark.parametrize("method", ["standard", "robust", "minmax"])
    def test_scaler_initialization(self, method):
        model = CustomerSegmentation(scaling_method=method)
        assert model.scaler is not None

    def test_invalid_scaler_raises(self):
        with pytest.raises(ValueError, match="Unknown scaling method"):
            CustomerSegmentation(scaling_method='invalid')


class TestPCA:
    def test_pca_components(self):
        df = _make_rfm_df()
        model = CustomerSegmentation(algorithm='kmeans', n_clusters=3, pca_components=2, random_state=42)
        labels = model.fit_predict(df, feature_cols=FEATURES)
        assert len(labels) == len(df)
        assert model.pca is not None
        assert model.pca.n_components == 2


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_output_dir):
        df = _make_rfm_df()
        model = CustomerSegmentation(algorithm='kmeans', n_clusters=3, random_state=42)
        model.fit(df, feature_cols=FEATURES)
        original_labels = model.labels_.copy()

        model_path = os.path.join(str(tmp_output_dir / 'models'), 'test_model.pkl')
        scaler_path = os.path.join(str(tmp_output_dir / 'models'), 'test_scaler.pkl')
        model.save_model(model_path, scaler_path)

        loaded = CustomerSegmentation.load_model(model_path, scaler_path)
        assert loaded.model is not None
        assert loaded.scaler is not None


class TestFindOptimalK:
    def test_returns_optimal_k_and_scores(self):
        df = _make_rfm_df()
        results = find_optimal_k(df, FEATURES, k_range=(2, 5), method='silhouette', random_state=42)
        assert 'optimal_k' in results
        assert 2 <= results['optimal_k'] <= 5
        assert len(results['scores']) == 4
        assert len(results['k_values']) == 4


class TestCompareAlgorithms:
    def test_returns_comparison_dataframe(self):
        df = _make_rfm_df()
        result = compare_algorithms(df, FEATURES, n_clusters=3, random_state=42)
        assert isinstance(result, pd.DataFrame)
        assert 'Algorithm' in result.columns
        assert 'Silhouette' in result.columns
        assert len(result) == 2  # kmeans + hierarchical


class TestEdgeCases:
    def test_invalid_algorithm_raises(self):
        model = CustomerSegmentation(algorithm='invalid')
        with pytest.raises(ValueError, match="Unknown algorithm"):
            model.fit(_make_rfm_df(), feature_cols=FEATURES)

    def test_k_greater_than_samples(self):
        df = _make_rfm_df(n=5)
        model = CustomerSegmentation(algorithm='kmeans', n_clusters=10, random_state=42)
        # sklearn raises ValueError when n_clusters > n_samples
        with pytest.raises((ValueError, Exception)):
            model.fit_predict(df, feature_cols=FEATURES)
