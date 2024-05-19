import os
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a subfolder for saving plots
plot_folder = 'gmm_plots'
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
