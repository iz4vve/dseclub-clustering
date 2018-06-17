from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def silhouette(data):
    range_n_clusters = range(2, 11)
    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(data)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(data, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

    #         color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                                alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
    #     colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(data[:, 0], data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()

def plot_correlations(df, T, cmap=None):
    """
    This functions plots the correlations in a dataset above a given threshold
    :param df: dataset on which to calculate correlations
    :param T: Threshold of correlations to show
    :param cmap: color map for visualization. Default uses two colours from Optum palette
    :type cmap: matplotlib.color.LinearSegmentedColormap
    """

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list('optum', ["#E87722", "#078576"])

    c = df.corr()
    c = c.where(np.triu(np.ones(c.shape)).astype(np.bool))  # filter tril
    c = c.where((abs(c) > T)).where(c != 1)  # filter diag

    # get column/row names of interesting corrs
    cols = [(n, col) for n, col in enumerate(c.columns) if any(abs(i) > T for i in c[col])]
    indices, cols = list(zip(*cols))

    row_cols = [(n, row) for n, row in enumerate(c.index) if any(abs(i) > T for i in c.loc[row, :])]
    row_indices, rows = list(zip(*row_cols))

    plt.imshow(c, cmap=cmap)
    plt.xticks(indices, cols, rotation="vertical")
    plt.yticks(row_indices, rows)

    plt.colorbar()
    plt.title(f"High correlations map: Threshold={T}")
    plt.show()