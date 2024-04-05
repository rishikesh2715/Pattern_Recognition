import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_excel("Proj2Dataset.xlsx", header=None)
X = df.iloc[:, :2].values
y = df.iloc[:, 2].values

# Gaussian kernel
def gaussian_kernel(x, y, sigma=1.75):
    return np.exp(-np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], axis=2)**2 / (2 * sigma**2))

# Solve the QP problem to get alphas
def solve_dual_soft_margin_svm(X, y, C, kernel):
    n_samples, n_features = X.shape
    K = kernel(X, X)
    print('K:', K)
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(n_samples))
    G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
    h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))
    A = matrix(y, (1, n_samples), 'd')
    b = matrix(0.0)
    
    solvers.options['show_progress'] = True
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(solution['x'])
    
    return alphas

# Function to calculate decision values for any input point
def decision_function(X_train, y_train, X, alphas, b, kernel):
    return np.dot((alphas * y_train), kernel(X_train, X)) + b

# Estimate b using support vectors
def compute_b(X, y, alphas, kernel):
    y_pred = decision_function(X, y, X, alphas, 0, kernel)
    return np.mean(y - y_pred)

# Run for two C values and plot
for C_value in [10, 100]:
    alphas = solve_dual_soft_margin_svm(X, y, C_value, gaussian_kernel)
    b = compute_b(X, y, alphas, gaussian_kernel)

    # Support Vectors
    support_vectors = (alphas > 1e-4)

    print(f"Number of support vectors for C = {C_value}: {sum(support_vectors)}")
    
    # Plot
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', marker='s', edgecolors='k', s=50)

    plt.title(f"Gaussian Kernel SVM with C = {C_value}")
    
    # Decision boundary and margin (approximate visualization)
    ax = plt.gca()
    # print(plt.gca())
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    Z = np.array([decision_function(X, y, np.array([xxi, yyi]).reshape(1, -1), alphas, b, gaussian_kernel) for xxi, yyi in zip(np.ravel(xx), np.ravel(yy))])
    Z = Z.reshape(xx.shape)
    contour = plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=1, linestyles=['--', '-', '--'], colors='black')
    
    plt.show()

    