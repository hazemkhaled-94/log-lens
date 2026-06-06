"""Clustering and dimensionality reduction engine for log embeddings."""

import logging

import numpy as np
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.mixture import (  # type: ignore[import-untyped]
    BayesianGaussianMixture as GaussianMixture,
)

from classifier.utils.config import N_CLUSTERS, RANDOM_STATE

logger = logging.getLogger(__name__)


class LogEngine:
    """Engine for clustering and dimensionality reduction of log embeddings."""

    def __init__(self) -> None:
        """Initialize K-means, Bayesian GMM, and a lazy PCA reducer."""
        self.kmeans = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=RANDOM_STATE,
            n_init="auto",
            init="k-means++",
        )

        self.gmm = GaussianMixture(
            n_components=N_CLUSTERS,
            covariance_type="diag",
            random_state=RANDOM_STATE,
            init_params="k-means++",
            n_init=5,
            verbose=1,
        )

        self.pca: PCA | None = None

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit the GMM and return per-sample cluster labels."""
        return self.gmm.fit_predict(embeddings)

    def reduce_dims(
        self,
        embeddings: np.ndarray,
        n_dims: int = 2,
    ) -> np.ndarray:
        """Reduce embeddings to ``n_dims`` principal components via PCA."""
        logger.info("Running PCA reduction to %d dimensions...", n_dims)
        self.pca = PCA(n_components=n_dims, random_state=RANDOM_STATE)
        return self.pca.fit_transform(embeddings)
