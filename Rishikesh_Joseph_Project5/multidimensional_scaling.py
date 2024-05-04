import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

# Generate a small synthetic dataset
X = np.array([[0, 0], [1, 1], [2, 1], [4, 4], [5, 5]])

# Create a distance matrix more efficiently
distances = squareform(pdist(X))

# Apply MDS
mds = MDS(n_components=2, dissimilarity='precomputed')
X_mds = mds.fit_transform(distances)

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.scatter(X[:, 0], X[:, 1])
ax1.set_title('Original Data')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

ax2.scatter(X_mds[:, 0], X_mds[:, 1])
ax2.set_title('MDS')
ax2.set_xlabel('Dimension 1')
ax2.set_ylabel('Dimension 2')

for i, txt in enumerate(range(len(X))):
    ax1.annotate(txt, (X[i, 0], X[i, 1]))
    ax2.annotate(txt, (X_mds[i, 0], X_mds[i, 1]))

plt.tight_layout()
plt.show()


