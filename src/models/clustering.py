"""
Clustering algorithms for customer segmentation

Implements K-Means, Hierarchical Clustering, and DBSCAN
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Optional, Tuple, List, Dict
import joblib
import logging

logger = logging.getLogger(__name__)


class CustomerSegmentation:
    """
    Customer segmentation using clustering algorithms.

    Supports K-Means, Hierarchical Clustering, and DBSCAN.
    """

    def __init__(
        self,
        algorithm: str = 'kmeans',
        n_clusters: int = 5,
        scaling_method: str = 'standard',
        pca_components: Optional[int] = None,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize CustomerSegmentation model.

        Parameters
        ----------
        algorithm : str
            Clustering algorithm: 'kmeans', 'hierarchical', or 'dbscan'
        n_clusters : int
            Number of clusters (for kmeans and hierarchical)
        scaling_method : str
            Feature scaling method: 'standard', 'robust', or 'minmax'
        pca_components : int, optional
            Number of PCA components for dimensionality reduction
        random_state : int
            Random seed for reproducibility
        **kwargs
            Additional parameters for clustering algorithms
        """
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.scaling_method = scaling_method
        self.pca_components = pca_components
        self.random_state = random_state
        self.kwargs = kwargs

        # Initialize components
        self.scaler = None
        self.pca = None
        self.model = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.feature_names_ = None

        # Initialize scaler
        self._init_scaler()

        logger.info(f"Initialized {algorithm} clustering with {n_clusters} clusters")

    def _init_scaler(self):
        """Initialize feature scaler based on scaling_method."""
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

    def fit(self, X: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> 'CustomerSegmentation':
        """
        Fit clustering model to data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data
        feature_cols : list, optional
            Specific columns to use for clustering. If None, uses all numeric columns.

        Returns
        -------
        self
        """
        # Select features
        if feature_cols is not None:
            X_features = X[feature_cols].copy()
            self.feature_names_ = feature_cols
        else:
            X_features = X.select_dtypes(include=[np.number]).copy()
            self.feature_names_ = list(X_features.columns)

        logger.info(f"Fitting model with features: {self.feature_names_}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)

        # Apply PCA if specified
        if self.pca_components is not None:
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            X_scaled = self.pca.fit_transform(X_scaled)
            logger.info(f"Applied PCA: {X_scaled.shape[1]} components, "
                       f"explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")

        # Fit clustering model
        if self.algorithm == 'kmeans':
            self._fit_kmeans(X_scaled)
        elif self.algorithm == 'hierarchical':
            self._fit_hierarchical(X_scaled)
        elif self.algorithm == 'dbscan':
            self._fit_dbscan(X_scaled)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        logger.info(f"Model fitted successfully. Unique labels: {len(np.unique(self.labels_))}")

        return self

    def _fit_kmeans(self, X_scaled: np.ndarray):
        """Fit K-Means clustering."""
        n_init = self.kwargs.get('n_init', 10)
        max_iter = self.kwargs.get('max_iter', 300)

        self.model = KMeans(
            n_clusters=self.n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=self.random_state
        )
        self.model.fit(X_scaled)
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_

    def _fit_hierarchical(self, X_scaled: np.ndarray):
        """Fit Hierarchical clustering."""
        linkage = self.kwargs.get('linkage', 'ward')

        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=linkage
        )
        self.labels_ = self.model.fit_predict(X_scaled)

    def _fit_dbscan(self, X_scaled: np.ndarray):
        """Fit DBSCAN clustering."""
        eps = self.kwargs.get('eps', 0.5)
        min_samples = self.kwargs.get('min_samples', 5)

        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = self.model.fit_predict(X_scaled)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data

        Returns
        -------
        np.ndarray
            Cluster labels
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Select and scale features
        X_features = X[self.feature_names_].copy()
        X_scaled = self.scaler.transform(X_features)

        # Apply PCA if used during training
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)

        # Predict
        if self.algorithm in ['kmeans']:
            labels = self.model.predict(X_scaled)
        elif self.algorithm in ['hierarchical', 'dbscan']:
            # These algorithms don't support predict, need to refit or use nearest neighbor
            # For simplicity, we'll refit (in production, use nearest neighbor approach)
            logger.warning(f"{self.algorithm} doesn't support predict. Using fit_predict.")
            labels = self.model.fit_predict(X_scaled)
        else:
            raise ValueError(f"Prediction not implemented for {self.algorithm}")

        return labels

    def fit_predict(self, X: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit model and return cluster labels.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data
        feature_cols : list, optional
            Specific columns to use

        Returns
        -------
        np.ndarray
            Cluster labels
        """
        self.fit(X, feature_cols)
        return self.labels_

    def get_cluster_centers(self, original_scale: bool = True) -> Optional[pd.DataFrame]:
        """
        Get cluster centers in original or scaled space.

        Parameters
        ----------
        original_scale : bool
            If True, inverse transform to original scale

        Returns
        -------
        pd.DataFrame or None
            Cluster centers (None for DBSCAN which doesn't have centers)
        """
        if self.cluster_centers_ is None:
            logger.warning(f"{self.algorithm} doesn't have cluster centers")
            return None

        centers = self.cluster_centers_.copy()

        # Inverse PCA if used
        if self.pca is not None and original_scale:
            centers = self.pca.inverse_transform(centers)

        # Inverse scaling if requested
        if original_scale:
            centers = self.scaler.inverse_transform(centers)

        # Create dataframe
        centers_df = pd.DataFrame(centers, columns=self.feature_names_)
        centers_df.insert(0, 'Cluster', range(len(centers_df)))

        return centers_df

    def save_model(
        self,
        model_path: str = 'outputs/models/clustering_model.pkl',
        scaler_path: str = 'outputs/models/scaler.pkl'
    ):
        """
        Save model and scaler to disk.

        Parameters
        ----------
        model_path : str
            Path to save model
        scaler_path : str
            Path to save scaler
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        if self.pca is not None:
            pca_path = model_path.replace('.pkl', '_pca.pkl')
            joblib.dump(self.pca, pca_path)

        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Scaler saved to: {scaler_path}")

    @classmethod
    def load_model(
        cls,
        model_path: str = 'outputs/models/clustering_model.pkl',
        scaler_path: str = 'outputs/models/scaler.pkl'
    ) -> 'CustomerSegmentation':
        """
        Load model and scaler from disk.

        Parameters
        ----------
        model_path : str
            Path to model file
        scaler_path : str
            Path to scaler file

        Returns
        -------
        CustomerSegmentation
            Loaded model instance
        """
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Create instance
        instance = cls(algorithm='kmeans')  # Will be overwritten
        instance.model = model
        instance.scaler = scaler

        # Try to load PCA
        pca_path = model_path.replace('.pkl', '_pca.pkl')
        try:
            instance.pca = joblib.load(pca_path)
        except FileNotFoundError:
            instance.pca = None

        logger.info(f"Model loaded from: {model_path}")

        return instance


def find_optimal_k(
    X: pd.DataFrame,
    feature_cols: List[str],
    k_range: Tuple[int, int] = (2, 10),
    method: str = 'silhouette',
    scaling_method: str = 'standard',
    random_state: int = 42
) -> Dict:
    """
    Find optimal number of clusters using elbow method or silhouette score.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data
    feature_cols : list
        Columns to use for clustering
    k_range : tuple
        Range of K values to test (min, max)
    method : str
        'silhouette', 'elbow', 'davies_bouldin', or 'calinski_harabasz'
    scaling_method : str
        Feature scaling method
    random_state : int
        Random seed

    Returns
    -------
    dict
        Results for each K value
    """
    results = {
        'k_values': [],
        'scores': [],
        'inertias': []
    }

    logger.info(f"Finding optimal K using {method} method for K in {k_range}")

    for k in range(k_range[0], k_range[1] + 1):
        # Fit model
        model = CustomerSegmentation(
            algorithm='kmeans',
            n_clusters=k,
            scaling_method=scaling_method,
            random_state=random_state
        )
        model.fit(X, feature_cols)

        # Get scaled data for scoring
        X_scaled = model.scaler.transform(X[feature_cols])

        # Calculate metrics
        if method == 'silhouette':
            score = silhouette_score(X_scaled, model.labels_)
        elif method == 'davies_bouldin':
            score = davies_bouldin_score(X_scaled, model.labels_)
        elif method == 'calinski_harabasz':
            score = calinski_harabasz_score(X_scaled, model.labels_)
        elif method == 'elbow':
            score = model.model.inertia_
        else:
            raise ValueError(f"Unknown method: {method}")

        results['k_values'].append(k)
        results['scores'].append(score)

        if hasattr(model.model, 'inertia_'):
            results['inertias'].append(model.model.inertia_)

        logger.info(f"K={k}: {method} score = {score:.4f}")

    # Find optimal K
    if method in ['silhouette', 'calinski_harabasz']:
        optimal_k = results['k_values'][np.argmax(results['scores'])]
    elif method in ['davies_bouldin', 'elbow']:
        optimal_k = results['k_values'][np.argmin(results['scores'])]

    results['optimal_k'] = optimal_k
    logger.info(f"Optimal K: {optimal_k}")

    return results


def compare_algorithms(
    X: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 5,
    scaling_method: str = 'standard',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compare different clustering algorithms.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data
    feature_cols : list
        Columns to use for clustering
    n_clusters : int
        Number of clusters
    scaling_method : str
        Feature scaling method
    random_state : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    algorithms = ['kmeans', 'hierarchical']
    results = []

    for algo in algorithms:
        logger.info(f"Evaluating {algo}...")

        # Fit model
        model = CustomerSegmentation(
            algorithm=algo,
            n_clusters=n_clusters,
            scaling_method=scaling_method,
            random_state=random_state
        )
        model.fit(X, feature_cols)

        # Get scaled data
        X_scaled = model.scaler.transform(X[feature_cols])

        # Calculate metrics
        sil_score = silhouette_score(X_scaled, model.labels_)
        db_score = davies_bouldin_score(X_scaled, model.labels_)
        ch_score = calinski_harabasz_score(X_scaled, model.labels_)

        n_labels = len(np.unique(model.labels_))

        results.append({
            'Algorithm': algo,
            'N_Clusters': n_labels,
            'Silhouette': sil_score,
            'Davies_Bouldin': db_score,
            'Calinski_Harabasz': ch_score
        })

    results_df = pd.DataFrame(results)
    logger.info("\nAlgorithm Comparison:")
    logger.info(f"\n{results_df}")

    return results_df


