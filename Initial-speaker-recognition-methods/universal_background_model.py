import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load the samples from the data.pickle file
with open('data.pickle', mode='rb') as f:
    samples = pickle.load(f)

# Fit a Universal Background Model (UBM) with N=3 components and diagonal covariance matrices
ubm = GaussianMixture(n_components=3, covariance_type='diag')
ubm.fit(samples)

# Evaluate the performance of the UBM using the silhouette score
labels = ubm.predict(samples)
silhouette = silhouette_score(samples, labels)

# Print the silhouette score
print("Silhouette Score:", silhouette)

