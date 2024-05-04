import numpy as np
import matplotlib.pyplot as plt


X = np.array([[4, 8, 13, 7], 
              [11, 4, 5, 14]])

mu = np.mean(X, axis=1)

X_centered = X - mu[:, np.newaxis]

cov_matrix = np.cov(X_centered)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sorted_indices]
top_eigenvector = eigenvectors[:, 0]

top_eigenvector = top_eigenvector / np.linalg.norm(top_eigenvector)

line_x = np.linspace(-10, 10, 100)
line_y = top_eigenvector[1] / top_eigenvector[0] * line_x

proj_scalars = np.dot(X_centered.T, top_eigenvector)
proj_points = np.outer(proj_scalars, top_eigenvector).T + mu[:, np.newaxis]

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.scatter(X[0, :], X[1, :], color='blue', label='Original Data')
for i in range(X.shape[1]):
    plt.annotate(f'{X[0, i]}, {X[1, i]}', (X[0, i], X[1, i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.xlabel('Feature X1')
plt.ylabel('Feature X2')
plt.title('Original Data Points')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(proj_points[0, :], proj_points[1, :], color='red', label='PCA Projection')
plt.plot(line_x + mu[0], line_y + mu[1], color='purple', label='PCA Line')
for i in range(X.shape[1]):
    plt.plot([X[0, i], proj_points[0, i]], [X[1, i], proj_points[1, i]], 'ro-')
plt.xlabel('Feature X1')
plt.title('Projection onto PCA Line')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
PCA_components = proj_scalars  # These are the coordinates of the projected points on the line
plt.scatter(PCA_components, np.zeros_like(PCA_components), color='green', label='PCA Components')
for i in range(X.shape[1]):
    plt.annotate(f'{PCA_components[i]:.2f}', (PCA_components[i], 0), textcoords="offset points", xytext=(0,-15), ha='center')
plt.xlabel('Principal Component 1')
plt.title('Transformed Data Points')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
