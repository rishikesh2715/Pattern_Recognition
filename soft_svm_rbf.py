import numpy as np
from cvxopt import matrix, solvers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import time

solvers.options['show_progress'] = False

X = pd.read_excel('Proj2DataSet.xlsx', header=None)
X = X.to_numpy()

C = [10, 100]

def kernel(Xi, Xj):
  return np.exp(-0.1632653 * np.linalg.norm(Xi - Xj) ** 2)

for C in C:
  y=X[:,2]

  N = len(X) 
  K = np.zeros((N, N))

  for i in range(N):
      for j in range(N):
          K[i, j] = kernel(X[i,:2],X[j,:2])
  P = matrix(np.outer(y, y) * K)
  q = matrix(-np.ones(N))
  G = matrix(np.vstack((-np.eye(N), np.eye(N))))
  h = matrix(np.hstack((np.zeros(N), C*np.ones(N))))
  A = matrix(y.reshape(1, -1))
  b = matrix(0.0)
  sol = solvers.qp(P, q, G, h, A, b)

  lambdas = np.array(sol['x']).reshape(-1)
  sv = lambdas > 1e-4  
  svLambdas = lambdas[sv]
  svX = X[sv,:2]
  svY= y[sv]

  wl = 0
  for i in range(len(svLambdas)):
    temp = 0
    for j in range(len(svLambdas)):
      temp+= svLambdas[j]*svY[j]*kernel(svX[i], svX[j])
    wl += (svY[i]-temp)

  wl /= len(svLambdas)
  # print(svLambdas)

  def predict(x):
    prediction = 0
    for i in range(len(svLambdas)):
      prediction += svLambdas[i] * svY[i] * kernel(x, svX[i])
    prediction += wl
    return np.sign(prediction)

  misclassified = []
  for i in range(N):
    if predict(X[i, :2]) != y[i]:
      misclassified.append(i)

  def decision_function(X_grid):
      Z = np.zeros(X_grid.shape[0])
      for i in range(X_grid.shape[0]):
          Z[i] = predict(X_grid[i])
      return Z

  # Create a mesh grid for the contour plot
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))

  # Predict the output for each point in the mesh grid
  Z = decision_function(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  plt.contour(xx, yy, Z, levels=[0], alpha=0.8, colors='black', linestyles='-')
  plt.contour(xx, yy, Z, levels=[-1], alpha=0.8, colors='blue', linestyles='--')
  plt.contour(xx, yy, Z, levels=[1], alpha=0.8, colors='blue', linestyles='--')
  plt.scatter(X[:60,0], X[:60,1], label='Class 1')
  plt.scatter(X[60:,0], X[60:,1], c='red', marker=',', label='Class 2')
  plt.scatter(svX[:,0], svX[:,1], c='green', marker='x', label='Support Vectors')
  plt.scatter(X[misclassified,0], X[misclassified,1], c='orange', marker='.', label='Misclassified')
  
  plt.title('Gaussian Kernel Dual Form Soft Margin SVM. C: {}. \nNumber of Support Vectors: {}. Misclassifications: {}.'.format(C, len(sv), len(misclassified)))
  plt.legend()
  plt.show()
