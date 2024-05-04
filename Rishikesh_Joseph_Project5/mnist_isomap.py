import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import Isomap
from sklearn.preprocessing import scale
import numpy as np

# Load MNIST dataset
digits = datasets.load_digits()

# Scale data and select only a subset to speed up processing
X = scale(digits.data[:300])
y = digits.target[:300]

# Create an Isomap instance to reduce dimensionality
n_neighbors = 5
n_components = 2
isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)

# Fit the model and transform the data
X_isomap = isomap.fit_transform(X)

# plt
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y, cmap=plt.cm.Spectral, edgecolor='k')
plt.colorbar(scatter)
unique = np.unique(y, return_index=True)
legend_handles = [plt.Line2D([0], [0], linestyle="none", marker='o', alpha=scatter.get_alpha(),
                             color=scatter.cmap(scatter.norm(y[i])), label=str(y[i])) for i in unique[1]]

plt.legend(handles=legend_handles, title="Digits")

plt.title('Isomap on MNIST')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

