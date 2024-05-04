import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA


X = np.array([[0.1, 0.1], [0.9, 0.8], [2, 2.1], [3.1, 2.9], [4, 4.1], [5.1, 5.0], [6, 5.8]])
y = [0, 1, 0, 1, 0, 1, 0] 

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X)

# Plotting
plt.figure(figsize=(18, 6))

# Plot original data
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)

# Plot PCA-transformed data
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('PCA Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)

# Plot Kernel PCA-transformed data
plt.subplot(1, 3, 3)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('Kernel PCA Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)

plt.tight_layout()
plt.show()

