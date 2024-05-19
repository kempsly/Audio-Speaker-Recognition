import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load the data
with open('data.pickle', mode='rb') as f:
    samples = pickle.load(f)

# Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(samples)

# Get cluster assignments
labels = gmm.predict(samples)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2, random_state=42)

# Fit SVM
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# Predict on test set
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_mat)

# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Plot clusters in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=labels, s=1, cmap='viridis')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
ax.set_title('GMM Clusters')
plt.show()
