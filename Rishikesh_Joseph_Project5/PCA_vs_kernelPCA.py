import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

# synthetic data
X, _ = make_circles(n_samples=500, factor=0.2, noise=0.08, random_state=42)

# applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# applying Kernel PCA (rbf)
kernel_pca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kernel_pca = kernel_pca.fit_transform(X)

# plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.scatter(X[:, 0], X[:, 1], c=_)
ax1.set_title('Original Data')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=_)
ax2.set_title('PCA')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

ax3.scatter(X_kernel_pca[:, 0], X_kernel_pca[:, 1], c=_)
ax3.set_title('Kernel PCA')
ax3.set_xlabel('Principal Component 1')
ax3.set_ylabel('Principal Component 2')

plt.tight_layout()
plt.show()




