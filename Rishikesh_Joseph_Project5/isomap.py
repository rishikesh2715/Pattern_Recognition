# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import Isomap
# from sklearn.datasets import make_s_curve

# # Generate a synthetic dataset (S-curve)
# X, _ = make_s_curve(n_samples=500, noise=0.1, random_state=42)

# # Apply Isomap
# isomap = Isomap(n_components=2, n_neighbors=10)
# X_isomap = isomap.fit_transform(X)

# # Plot the results
# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122)

# ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 2], cmap='viridis')
# ax1.set_title('Original Data')
# ax1.set_xlabel('Feature 1')
# ax1.set_ylabel('Feature 2')
# ax1.set_zlabel('Feature 3')

# ax2.scatter(X_isomap[:, 0], X_isomap[:, 1], c=X[:, 2], cmap='viridis')
# ax2.set_title('Isomap Embedding')
# ax2.set_xlabel('Component 1')
# ax2.set_ylabel('Component 2')

# plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax2)
# plt.tight_layout()
# plt.show()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.datasets import make_swiss_roll

# Generate a synthetic dataset (Swiss Roll)
X, _ = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

# Apply Isomap
isomap = Isomap(n_components=1, n_neighbors=10)
X_isomap = isomap.fit_transform(X)

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(X[:, 0], X[:, 1], c=X[:, 2], cmap='viridis')
ax1.set_title('Original Data')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

ax2.scatter(X_isomap[:, 0], np.zeros_like(X_isomap[:, 0]), c=X[:, 2], cmap='viridis')
ax2.set_title('Isomap Embedding')
ax2.set_xlabel('Component 1')
ax2.set_yticks([])

plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax2)
plt.tight_layout()
plt.show()