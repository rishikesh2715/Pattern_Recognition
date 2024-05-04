import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Create the figure and axes
fig = plt.figure(figsize=(18, 8))

# 3D scatter plot of the original data
ax1 = fig.add_subplot(121, projection='3d', aspect='auto')
ax1.set_title('Original Iris Data')
sc1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor='k', s=50)
ax1.set_xlabel('Sepal length')
ax1.set_ylabel('Sepal width')
ax1.set_zlabel('Petal length')
plt.colorbar(sc1, ax=ax1, pad=0.1, label='Species')

# Add class names to the 3D plot
for i, class_name in enumerate(class_names):
    class_center = X[y == i].mean(axis=0)
    ax1.text(class_center[0], class_center[1], class_center[2], class_name, color='black', fontsize=12, weight='bold')

# Isomap for dimensionality reduction to 2D
isomap = Isomap(n_neighbors=5, n_components=2)
X_isomap = isomap.fit_transform(X)

# 2D scatter plot of the Isomap results
ax2 = fig.add_subplot(122)
ax2.set_title('2D Isomap of Iris Data')
sc2 = ax2.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y, cmap=plt.cm.nipy_spectral, edgecolor='k', s=50)
ax2.set_xlabel('Component 1')
ax2.set_ylabel('Component 2')
plt.colorbar(sc2, ax=ax2, pad=0.1, label='Species')

# Add class names to the 2D plot
for i, class_name in enumerate(class_names):
    class_center = X_isomap[y == i].mean(axis=0)
    ax2.text(class_center[0], class_center[1], class_name, color='black', fontsize=12, weight='bold')

plt.show()
