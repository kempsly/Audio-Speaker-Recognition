import pickle
import numpy as np
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

# Load the data
with open('data.pickle', mode='rb') as f:
    samples = pickle.load(f)

# Fit Factor Analysis model
factor_analysis = FactorAnalysis(n_components=3, random_state=42)
factor_analysis.fit(samples)

# Transform the data
transformed_data = factor_analysis.transform(samples)

# Visualize the transformed data
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], c='b', marker='o', alpha=0.5)
ax.set_xlabel('Factor 1')
ax.set_ylabel('Factor 2')
ax.set_zlabel('Factor 3')
ax.set_title('Factor Analysis')
plt.show()
