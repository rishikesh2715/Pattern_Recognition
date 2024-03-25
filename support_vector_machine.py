# Rishikesh
# Project II
# Pattern Recognition
# python 3.9.5
# import matplotlib.pyplot as plt
# import numpy as np
# from cvxopt import matrix, solvers
# from scipy.spatial import distance
# import pandas as pd
# from sklearn import svm
# import time


# plt.ion()

# solvers.options['show_progress'] = False

# X = pd.read_excel('Proj2DataSet(2).xlsx', header=None)
# X = X.to_numpy()

# def rbf_kernel(x1, x2, gamma=1.75):
#     return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

# C = [10, 100]

# for C in C:
#   plt.figure(figsize=(10, 8))
#   y=X[:,2]

#   # Construct the matrices for the QP problem
#   N = len(X) # number of training examples
#   dist_matrix = distance.cdist(X[:,:2], X[:,:2], 'euclidean') 
#   K = np.exp(-np.square(dist_matrix)) / ((2 * 1.75**2)) # kernel matrix with Gaussian kernel
#   P = matrix(np.outer(y, y) * K)
#   q = matrix(-np.ones(N))
#   G = matrix(np.vstack((-np.eye(N), np.eye(N))))
#   h = matrix(np.hstack((np.zeros(N), C*np.ones(N))))
#   A = matrix(y.reshape(1, -1))
#   b = matrix(0.0)
#   # Solve the QP problem
#   sol = solvers.qp(P, q, G, h, A, b)

#   # Extract the solution
#   lambdas = np.array(sol['x']).reshape(-1)
#   sv = []
#   w=np.zeros((1,2))
#   for i in range(N):
#       w+=lambdas[i]*y[i]*X[i,:2]
#   wl = 0

#   for i in range(N):
#     if lambdas[i] > 10e-6 and lambdas[i] < C:
#       wl+=(1/y[i]-w[0].dot(X[i,:2]))
#       sv.append(i)

#   wl /= len(sv)
#   # print(lambdas[sv])

#   # Get the misclassified points by putting them into the decision boundary and checking if they are supposed to be pos or neg based on their y
#   svValues = w.dot(X[sv,:2].T) + wl
#   misclassified = []
#   for i in range(len(sv)):
#     if (y[sv][i] > 0 and svValues[0,i] < 0) or (y[sv][i] < 0 and svValues[0,i] > 0):
#       misclassified.append(i)
#   sv = np.array(sv)
#   misclassified = sv[misclassified]


#   plt.scatter(X[:60,0], X[:60,1], label='Class 1')
#   plt.scatter(X[60:,0], X[60:,1], c='red', marker=',', label='Class 2')
#   plt.scatter(X[sv,0], X[sv,1], c='green', marker='x', label='Support Vectors')
#   plt.scatter(X[misclassified,0], X[misclassified,1], c='orange', marker='.', label='Misclassified')
#   x0 = np.linspace(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1, 100)
#   decisionBoundary = (-wl - w[0, 0] * x0) / w[0, 1]
#   upperMargin = (1 - wl - w[0, 0] * x0) / w[0, 1]
#   lowerMargin = (-1 - wl - w[0, 0] * x0) / w[0, 1]
#   plt.plot(x0, decisionBoundary, 'k-', label='Decision Boundary')
#   plt.plot(x0, upperMargin, 'r--', label='Upper Margin')
#   plt.plot(x0, lowerMargin, 'g--', label='Lower Margin')
#   plt.xlabel('Feature 1')
#   plt.ylabel('Feature 2')
#   plt.title('SVM with C={}. Support Vectors: {}. Misclassifications: {}.'.format(C, len(sv), len(misclassified)))
#   plt.legend()
#   plt.show()

# input('Press Enter to exit')


import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
from scipy.spatial import distance
import pandas as pd


plt.ion()
def rbf_kernel(x1, x2, gamma=1.75):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

# Load data
X = pd.read_excel('Proj2DataSet(2).xlsx', header=None)
X = X.to_numpy()

C_values = [10, 100]

solvers.options['show_progress'] = False

for C in C_values:
    plt.figure(figsize=(10, 8))
    y = X[:, 2]
    N = len(X)
    
    # Construct the kernel matrix
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = rbf_kernel(X[i, :2], X[j, :2])
            
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(N))
    G = matrix(np.vstack((-np.eye(N), np.eye(N))))
    h = matrix(np.hstack((np.zeros(N), C * np.ones(N))))
    A = matrix(y, (1, N))
    b = matrix(0.0)
    
    # Solve QP problem
    solution = solvers.qp(P, q, G, h, A, b)
    lambdas = np.array(solution['x']).flatten()
    
    # Extract support vectors
    sv = lambdas > 1e-5
    index = np.arange(len(lambdas))[sv]
    sv_lambdas = lambdas[sv]
    sv_X = X[sv, :2]
    sv_y = y[sv]

    # Define function to predict using support vectors
    def predict(x):
        pred = sum(sv_lambdas[i] * sv_y[i] * rbf_kernel(sv_X[i], x) for i in range(len(sv_X)))
        return pred

    # Create a grid to plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    # Predict for each point in the grid
    Z = np.array([predict(np.array([x, y])) for x, y in zip(np.ravel(xx), np.ravel(yy))])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.contourf(xx, yy, Z > 0, alpha=0.4)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.scatter(sv_X[:, 0], sv_X[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, label='Data Points')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'SVM with RBF Kernel | C = {C}')
    plt.legend()
    plt.show()

input('Press Enter to exit')