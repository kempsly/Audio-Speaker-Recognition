"""
    Fit a Gaussian mixture model to these samples, where the number 
    of Gaussian components is N=3, and covariance matrices are all diagonal.

    We gonna find the weights, means, and covariance matrices of the Gaussian mixture model
    
    STEPS:
        - Load the data from the pickle file.
        - Fit a Gaussian mixture model with three components (N=3) and diagonal covariance matrices.
        - Retrieve the weights, means, and covariance matrices of the fitted GMM.
"""

import pickle
import numpy as np
from sklearn.mixture import GaussianMixture

# Load the data from the pickle file
with open('data.pickle', mode='rb') as f:
    samples = pickle.load(f)

# Fit a Gaussian mixture model with three components and diagonal covariance matrices
gmm_diagonal = GaussianMixture(n_components=3, covariance_type='diag', random_state=42)
gmm_diagonal.fit(samples)

# Retrieve the weights, means, and covariance matrices of the GMM
weights_diagonal = gmm_diagonal.weights_
means_diagonal = gmm_diagonal.means_
covariances_diagonal = gmm_diagonal.covariances_

print("Weights of the Gaussian mixture model (with diagonal covariance matrices):")
print(weights_diagonal)
print("\nMeans of the Gaussian mixture model (with diagonal covariance matrices):")
print(means_diagonal)
print("\nCovariance matrices of the Gaussian mixture model (with diagonal covariance matrices):")
print(covariances_diagonal)

# Fit a Gaussian mixture model with three components and full covariance matrices
gmm_full = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm_full.fit(samples)

# Retrieve the weights, means, and covariance matrices of the GMM with full covariance matrices
weights_full = gmm_full.weights_
means_full = gmm_full.means_
covariances_full = gmm_full.covariances_

print("\nWeights of the Gaussian mixture model (with full covariance matrices):")
print(weights_full)
print("\nMeans of the Gaussian mixture model (with full covariance matrices):")
print(means_full)
print("\nCovariance matrices of the Gaussian mixture model (with full covariance matrices):")
print(covariances_full)
