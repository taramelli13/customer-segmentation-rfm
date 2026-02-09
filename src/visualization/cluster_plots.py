"""
Cluster visualization functions for customer segmentation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Optional, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)


def plot_clusters_pca_2d(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot clusters in 2D using PCA projection.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature data
    labels : np.ndarray
        Cluster labels
    feature_names : list, optional
        Names of features
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Create plot
    plt.figure(figsize=figsize)

    # Plot each cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points (for DBSCAN)
            color = 'gray'
            marker = 'x'
            cluster_name = 'Noise'
        else:
            marker = 'o'
            cluster_name = f'Cluster {label}'

        mask = labels == label
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[color],
            label=cluster_name,
            marker=marker,
            s=50,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title('Customer Segments (PCA Projection)', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.close()


def plot_clusters_pca_3d_interactive(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot clusters in 3D using PCA projection (interactive plotly).

    Parameters
    ----------
    X : np.ndarray
        Scaled feature data
    labels : np.ndarray
        Cluster labels
    feature_names : list, optional
        Names of features
    save_path : str, optional
        Path to save figure (HTML)
    """
    # Apply PCA
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)

    # Create dataframe for plotly
    df_plot = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2],
        'Cluster': [f'Cluster {l}' if l != -1 else 'Noise' for l in labels]
    })

    # Create 3D scatter plot
    fig = px.scatter_3d(
        df_plot,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Cluster',
        title='Customer Segments (3D PCA Projection)',
        labels={
            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
            'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
        },
        opacity=0.7
    )

    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Interactive plot saved to: {save_path}")

    pass


def plot_rfm_3d_scatter(
    df: pd.DataFrame,
    labels: np.ndarray,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
):
    """
    Plot 3D scatter of RFM values colored by cluster.

    Parameters
    ----------
    df : pd.DataFrame
        RFM feature dataframe
    labels : np.ndarray
        Cluster labels
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    from mpl_toolkits.mplot3d import Axes3D

    df_plot = df.copy()
    df_plot['Cluster'] = labels

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = df_plot['Cluster'] == label
        cluster_data = df_plot[mask]

        ax.scatter(
            cluster_data['Recency'],
            cluster_data['Frequency'],
            cluster_data['Monetary'],
            c=[color],
            label=f'Cluster {label}' if label != -1 else 'Noise',
            s=50,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel('Recency (days)', fontsize=10)
    ax.set_ylabel('Frequency (count)', fontsize=10)
    ax.set_zlabel('Monetary ($)', fontsize=10)
    ax.set_title('Customer Segments (RFM Space)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.close()


def plot_rfm_3d_interactive(
    df: pd.DataFrame,
    labels: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot interactive 3D scatter of RFM values using plotly.

    Parameters
    ----------
    df : pd.DataFrame
        RFM feature dataframe
    labels : np.ndarray
        Cluster labels
    save_path : str, optional
        Path to save figure (HTML)
    """
    df_plot = df.copy()
    df_plot['Cluster'] = [f'Cluster {l}' if l != -1 else 'Noise' for l in labels]

    fig = px.scatter_3d(
        df_plot,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Cluster',
        title='Customer Segments (RFM Space)',
        labels={
            'Recency': 'Recency (days)',
            'Frequency': 'Frequency (count)',
            'Monetary': 'Monetary ($)'
        },
        hover_data=['Recency', 'Frequency', 'Monetary'],
        opacity=0.7
    )

    fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color='DarkSlateGrey')))

    if save_path:
        fig.write_html(save_path)
        logger.info(f"Interactive plot saved to: {save_path}")

    pass


def plot_cluster_profiles(
    summary: pd.DataFrame,
    features: List[str] = ['Recency', 'Frequency', 'Monetary'],
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
):
    """
    Plot cluster profiles showing mean values of features.

    Parameters
    ----------
    summary : pd.DataFrame
        Cluster summary dataframe
    features : list
        Features to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    # Extract mean values
    mean_cols = [f'{feat}_mean' for feat in features]

    # Check if columns exist
    available_cols = [col for col in mean_cols if col in summary.columns]

    if not available_cols:
        logger.warning("No mean columns found in summary")
        return

    plot_data = summary[['Cluster'] + available_cols].set_index('Cluster')

    # Normalize for better comparison
    plot_data_norm = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min())

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Absolute values
    plot_data.plot(kind='bar', ax=ax1)
    ax1.set_title('Cluster Profiles (Absolute Values)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Mean Value')
    ax1.legend(title='Features', labels=features)
    ax1.grid(True, alpha=0.3, axis='y')

    # Normalized values
    plot_data_norm.plot(kind='bar', ax=ax2)
    ax2.set_title('Cluster Profiles (Normalized)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Normalized Value (0-1)')
    ax2.legend(title='Features', labels=features)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.close()


def plot_cluster_sizes(
    labels: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot cluster size distribution.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    unique, counts = np.unique(labels, return_counts=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar chart
    ax1.bar(unique, counts, color=sns.color_palette('Set2', len(unique)))
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Number of Customers')
    ax1.set_title('Cluster Sizes (Count)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Pie chart
    cluster_labels = [f'Cluster {l}' if l != -1 else 'Noise' for l in unique]
    ax2.pie(counts, labels=cluster_labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Cluster Sizes (Percentage)', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.close()


def plot_elbow_curve(
    results: dict,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot elbow curve for K-Means.

    Parameters
    ----------
    results : dict
        Results from find_optimal_k() function
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    plt.figure(figsize=figsize)

    if 'inertias' in results and results['inertias']:
        plt.plot(results['k_values'], results['inertias'], 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Inertia (WSS)', fontsize=12)
        plt.title('Elbow Curve for Optimal K', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Mark optimal K if available
        if 'optimal_k' in results:
            optimal_idx = results['k_values'].index(results['optimal_k'])
            plt.scatter(
                [results['optimal_k']],
                [results['inertias'][optimal_idx]],
                c='red',
                s=200,
                marker='*',
                label=f'Optimal K={results["optimal_k"]}',
                zorder=5
            )
            plt.legend()

    elif 'scores' in results:
        plt.plot(results['k_values'], results['scores'], 'go-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (K)', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Score vs Number of Clusters', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.close()


def plot_silhouette_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    figsize: Tuple[int, int] = (10, 7),
    save_path: Optional[str] = None
):
    """
    Plot silhouette analysis for cluster evaluation.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature data
    labels : np.ndarray
        Cluster labels
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    from sklearn.metrics import silhouette_samples, silhouette_score

    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    fig, ax = plt.subplots(figsize=figsize)

    y_lower = 10
    unique_labels = np.unique(labels[labels >= 0])  # Exclude noise

    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for i, (label, color) in enumerate(zip(unique_labels, colors)):
        # Get silhouette values for this cluster
        cluster_silhouette_values = sample_silhouette_values[labels == label]
        cluster_silhouette_values.sort()

        size_cluster = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )

        # Label clusters
        ax.text(-0.05, y_lower + 0.5 * size_cluster, f'C{label}')

        y_lower = y_upper + 10

    ax.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster Label', fontsize=12)

    # Vertical line for average silhouette score
    ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2,
               label=f'Average Score: {silhouette_avg:.3f}')

    ax.legend(loc='best')
    ax.set_xlim([-0.1, 1])
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    plt.close()


