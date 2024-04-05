# """
# Rishikesh
# Project III
# Pattern Recognition
# python 3.9.5

# """


import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
import pandas as pd

# gaussian kernel
def rbf_kernel(X1, X2, gamma=1.75):
    gamma = 1/(2*gamma**2)
    sq_dist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dist)

# Adjusted predict function to include bias term 'b'
def predict(X, sv_X, sv_y, sv_lambdas, b, gamma=1.75):
    K = rbf_kernel(X, sv_X, gamma)
    return np.dot(K, sv_lambdas * sv_y) + b  # Use of bias 'b' here

# load data
X = pd.read_excel('Proj2DataSet.xlsx', header=None)
X = X.to_numpy()

C_values = [10, 100]
solvers.options['show_progress'] = False

for C in C_values:
    plt.figure(figsize=(10, 8))
    y = X[:, 2]
    N = len(X)
    
    K = rbf_kernel(X[:, :2], X[:, :2])
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(N))
    G = matrix(np.vstack((-np.eye(N), np.eye(N))))
    h = matrix(np.hstack((np.zeros(N), C * np.ones(N))))
    A = matrix(y, (1, N), 'd')
    b_qp = matrix(0.0)  # Renamed to b_qp to distinguish from bias 'b'
    
    # solve QP problem
    solution = solvers.qp(P, q, G, h, A, b_qp)
    # print(solution)
    lambdas = np.array(solution['x']).flatten()
    
    # extract support vectors
    sv = lambdas > 1e-4
    index = np.arange(len(lambdas))[sv]
    sv_lambdas = lambdas[sv]
    sv_X = X[sv, :2]
    sv_y = y[sv]
  
    # # Calculate b using support vectors
    # b_sum = 0  # Initialize sum for calculating b

    # for i in range(len(sv_lambdas)):
    #     b_sum += sv_lambdas[i] * sv_y[i] * K[index[i], sv]
        
    # # Average over the support vectors
    # b = np.mean(sv_y - b_sum)

    # # Adjust the bias term to correct its calculation
    # K_sv_sv = rbf_kernel(sv_X, sv_X, gamma=1.75)
    # b_sv = y[sv] - np.dot(K_sv_sv, sv_lambdas * sv_y)

    # # Since b could potentially be calculated from each support vector,
    # # taking the mean is the approach used for stabilization    
    # b = np.mean(b_sv)
    # print(b)
    # Calculate bias 'b' using support vectors
    # It's calculated as the average over the support vectors
    b = np.mean(sv_y - np.dot(K[sv][:, sv], sv_lambdas * sv_y))
    print(b)

    # define function to predict using support vectors and include bias 'b'
    predictions = predict(X[:, :2], sv_X, sv_y, sv_lambdas, b).flatten()
    predictions_class = np.sign(predictions)
    misclassified = predictions_class != y

    # no. of support vectors and misclassifications
    num_support_vectors = len(sv_X)
    num_misclass = sum(misclassified)

    # grid points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    
    # predict for each point in the grid
    grid_X = np.vstack([xx.ravel(), yy.ravel()]).T  
    Z = predict(grid_X, sv_X, sv_y, sv_lambdas, b).reshape(xx.shape)  

    # plot decision boundary and margins
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])
    
    # plot data points
    class1_mask = y == 1
    class2_mask = y == -1
    plt.scatter(X[class1_mask, 0], X[class1_mask, 1], c='red', marker='o', label='Class 1', edgecolors='none')
    plt.scatter(X[class2_mask, 0], X[class2_mask, 1], c='green', marker='s', label='Class 2', edgecolors='none')
    plt.scatter(X[misclassified, 0], X[misclassified, 1], c='blue', marker='x', s=50, label='Misclassified')
    plt.scatter(sv_X[:, 0], sv_X[:, 1], facecolors='none', edgecolors='k', marker='o', s=100, label='Support Vectors')
    plt.text(0.05, 0.95, f'C={C}; Sup.Vec.={num_support_vectors}; Misclass={num_misclass}',
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.5))
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'SVM with Gaussian Kernel | C = {C}, No. of support vectors = {num_support_vectors}, No. of misclassifications = {num_misclass}')
    plt.legend()
    plt.show()
