"""3D scatter-plot visualizer for GMM-clustered log embeddings."""

import logging
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class ClusterVisualizer:
    """3D scatter plot with covariance ellipsoids for GMM cluster output."""

    def plot_gmm_clusters_3d(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        cluster_names: list[str] | None = None,
        save_path: str = "gmm_plot_3d.png",
        show: bool = False,
    ) -> None:
        """Render a 3D scatter plot with per-cluster covariance ellipsoids.

        Args:
            coords: Shape (n_samples, 3) — must be exactly 3D.
            labels: Shape (n_samples,) cluster assignments.
            cluster_names: Legend labels in sorted unique-label order.
            save_path: Output image path.
            show: Display the plot interactively after saving.

        Raises:
            ValueError: If coords does not have exactly 3 columns.
        """
        if coords.shape[1] != 3:
            raise ValueError(
                f"Expected 3D coordinates, got shape {coords.shape}"
            )

        logger.info("Initializing 3D GMM density plot...")

        fig = plt.figure(figsize=(12, 10))
        ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))

        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],  # type: ignore
            c=labels,
            cmap="coolwarm",
            s=40,
            alpha=0.8,
            edgecolors="w",
            linewidth=0.5,
        )

        ax.set_title("3D GMM Log Clustering")

        handles, default_labels = scatter.legend_elements()
        if cluster_names and len(cluster_names) == len(handles):
            legend_labels = cluster_names
        else:
            legend_labels = [
                f"Cluster {int(lbl)}" for lbl in np.unique(labels)
            ]

        legend = ax.legend(handles, legend_labels, title="Log Categories")
        ax.add_artist(legend)

        unique_labels = np.unique(labels)
        for label in unique_labels:
            self._plot_covariance_ellipsoid(ax, coords, labels, label)

        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Plot saved successfully to {save_path}")

        if show:
            logger.info("Displaying interactive plot...")
            plt.show()

    def _plot_covariance_ellipsoid(
        self,
        ax: Axes3D,
        coords: np.ndarray,
        labels: np.ndarray,
        target_label: int,
    ) -> None:
        """Draw a 2-std wireframe ellipsoid for one cluster."""
        cluster_points = coords[labels == target_label]

        # Fewer than 4 points can't yield a meaningful 3D covariance matrix
        if len(cluster_points) < 4:
            return

        mean = np.mean(cluster_points, axis=0)
        cov = np.cov(cluster_points, rowvar=False)

        # Eigen-decompose for ellipsoid rotation and scale
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        sphere_coords = np.vstack((x.flatten(), y.flatten(), z.flatten()))

        n_std = 2.0
        # np.maximum guards against tiny negative values from float precision
        radii = n_std * np.sqrt(np.maximum(eigenvalues, 0))
        transform = eigenvectors @ np.diag(radii)

        ellipsoid_coords = (transform @ sphere_coords).T + mean

        x_ellipsoid = ellipsoid_coords[:, 0].reshape(x.shape)
        y_ellipsoid = ellipsoid_coords[:, 1].reshape(y.shape)
        z_ellipsoid = ellipsoid_coords[:, 2].reshape(z.shape)

        ax.plot_wireframe(
            x_ellipsoid,
            y_ellipsoid,
            z_ellipsoid,
            color="black",
            alpha=0.1,
            linewidth=0.5,
        )
