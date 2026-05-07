import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def extract_latent_vectors(encoder,X_test):
    latent_vectors= encoder.predict(X_test)
    return latent_vectors

def reduce_dimensions(latent_vectors):
    pca= PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    return latent_2d

def perform_clustering(latent_vectors, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(latent_vectors)
    return clusters

def plot_true_labels(latent_2d, test_df):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        latent_2d[:,0],
        latent_2d[:,1],
        c = pd.factorize(test_df['label'])[0],
        alpha= 0.6,
        s= 10
    )
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Autoencoder Latent Space')
    plt.savefig("outputs/plots/latent_space_pca.png")
    plt.show()

def plot_clusters(latent_2d, clusters):
    plt.figure(figsize=(8, 6))

    plt.scatter(
        latent_2d[:,0],
        latent_2d[:,1],
        c = clusters,
        alpha= 0.5
    )

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('K-Means Clusters in Latent Space')

    plt.savefig("outputs/plots/latent_space_clusters.png")
    plt.show()

def compute_silhouette_scores(latent_vectors):
    cluster_range = range(2, 15)

    scores = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(latent_vectors)
        score = silhouette_score(latent_vectors, cluster_labels)
        scores.append(score)

        print(f'K={k}, Silhouette Score={score:.4f}')

    return cluster_range, scores

def plot_silhouette_scores(cluster_range, scores):
    plt.figure(figsize=(8, 5))

    plt.plot(cluster_range, scores, marker='o')

    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores vs Number of Clusters')
    plt.savefig("outputs/plots/silhouette_scores.png")
    plt.show()