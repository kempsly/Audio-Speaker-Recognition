"""
    Fit and Evaluate a gaussian mixture model

    several metrics can be used, including:

        - Log-Likelihood: This measures how well the model explains the data. Higher values indicate a better fit.
        - Bayesian Information Criterion (BIC): This is used for model selection among a finite set of models; 
        it balances the model fit with the number of parameters, penalizing more complex models. 
        Lower values indicate a better model.
        - Akaike Information Criterion (AIC): Similar to BIC, it also balances model fit with complexity 
        but is less stringent in penalizing complexity. Lower values indicate a better model.
        - Silhouette Score: This measures how similar a point is to its own cluster compared to other clusters.
        Values range from -1 to 1, with higher values indicating better-defined clusters.
        - Adjusted Rand Index (ARI): This compares the similarity of the clustering with a ground truth clustering,
        adjusting for chance. Values range from -1 to 1, with higher values indicating better clustering.
"""

import os
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Create a subfolder for saving plots
plot_folder = 'gmm_performance_plots'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

# Load the data
with open('data.pickle', mode='rb') as f:
    samples = pickle.load(f)

# Function to plot ellipsoids representing the covariance matrices
def plot_ellipsoid(ax, mean, cov, color):
    u, s, vt = np.linalg.svd(cov)
    radii = np.sqrt(s)
    u = u * radii
    for i in range(len(radii)):
        u[:, i] = np.sqrt(radii[i]) * u[:, i]
    
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))
    
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot(u, [x[i, j], y[i, j], z[i, j]]) + mean

    ax.plot_wireframe(x, y, z, color=color, alpha=0.2)

# Fit GMM with diagonal covariance matrices
gmm_diagonal = GaussianMixture(n_components=3, covariance_type='diag', random_state=42)
gmm_diagonal.fit(samples)
labels_diagonal = gmm_diagonal.predict(samples)

# Fit GMM with full covariance matrices
gmm_full = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm_full.fit(samples)
labels_full = gmm_full.predict(samples)

# Plot results for diagonal covariance
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=labels_diagonal, s=1, cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.set_title('GMM with Diagonal Covariance')
for mean, cov in zip(gmm_diagonal.means_, gmm_diagonal.covariances_):
    plot_ellipsoid(ax, mean, np.diag(cov), 'red')
plt.savefig(os.path.join(plot_folder, 'gmm_diagonal.png'))
plt.close()

# Plot results for full covariance
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=labels_full, s=1, cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.set_title('GMM with Full Covariance')
for mean, cov in zip(gmm_full.means_, gmm_full.covariances_):
    plot_ellipsoid(ax, mean, cov, 'red')
plt.savefig(os.path.join(plot_folder, 'gmm_full.png'))
plt.close()

# Performance Metrics
# Log-Likelihood
log_likelihood_diagonal = gmm_diagonal.score(samples)
log_likelihood_full = gmm_full.score(samples)

# BIC
bic_diagonal = gmm_diagonal.bic(samples)
bic_full = gmm_full.bic(samples)

# AIC
aic_diagonal = gmm_diagonal.aic(samples)
aic_full = gmm_full.aic(samples)

# Silhouette Score
silhouette_score_diagonal = silhouette_score(samples, labels_diagonal)
silhouette_score_full = silhouette_score(samples, labels_full)

# Print metrics
print(f"Log-Likelihood (Diagonal): {log_likelihood_diagonal}")
print(f"Log-Likelihood (Full): {log_likelihood_full}")
print(f"BIC (Diagonal): {bic_diagonal}")
print(f"BIC (Full): {bic_full}")
print(f"AIC (Diagonal): {aic_diagonal}")
print(f"AIC (Full): {aic_full}")
print(f"Silhouette Score (Diagonal): {silhouette_score_diagonal}")
print(f"Silhouette Score (Full): {silhouette_score_full}")

# If ground truth labels are available
# ground_truth_labels = ... (your ground truth labels here)
# adjusted_rand_index_diagonal = adjusted_rand_score(ground_truth_labels, labels_diagonal)
# adjusted_rand_index_full = adjusted_rand_score(ground_truth_labels, labels_full)

# print(f"Adjusted Rand Index (Diagonal): {adjusted_rand_index_diagonal}")
# print(f"Adjusted Rand Index (Full): {adjusted_rand_index_full}")
