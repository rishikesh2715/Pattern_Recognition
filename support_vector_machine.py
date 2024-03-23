# Rishikesh
# Project II
# Pattern Recognition
# python 3.9.5
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import matrix, solvers
from scipy.spatial import distance
import pandas as pd
from sklearn import svm
import time


plt.ion()

solvers.options['show_progress'] = False

X = pd.read_excel('Proj2DataSet(2).xlsx', header=None)
X = X.to_numpy()

C = [10, 100]

for C in C:
  plt.figure(figsize=(10, 8))
  y=X[:,2]

  # Construct the matrices for the QP problem
  N = len(X) # number of training examples
  dist_matrix = distance.cdist(X[:,:2], X[:,:2], 'euclidean') 
  K = np.exp(-np.square(dist_matrix)) / ((2 * 1.75**2)) # kernel matrix with Gaussian kernel
  P = matrix(np.outer(y, y) * K)
  q = matrix(-np.ones(N))
  G = matrix(np.vstack((-np.eye(N), np.eye(N))))
  h = matrix(np.hstack((np.zeros(N), C*np.ones(N))))
  A = matrix(y.reshape(1, -1))
  b = matrix(0.0)
  # Solve the QP problem
  sol = solvers.qp(P, q, G, h, A, b)

  # Extract the solution
  lambdas = np.array(sol['x']).reshape(-1)
  sv = []
  w=np.zeros((1,2))
  for i in range(N):
      w+=lambdas[i]*y[i]*X[i,:2]
  wl = 0

  for i in range(N):
    if lambdas[i] > 10e-6 and lambdas[i] < C:
      wl+=(1/y[i]-w[0].dot(X[i,:2]))
      sv.append(i)

  wl /= len(sv)
  # print(lambdas[sv])

  # Get the misclassified points by putting them into the decision boundary and checking if they are supposed to be pos or neg based on their y
  svValues = w.dot(X[sv,:2].T) + wl
  misclassified = []
  for i in range(len(sv)):
    if (y[sv][i] > 0 and svValues[0,i] < 0) or (y[sv][i] < 0 and svValues[0,i] > 0):
      misclassified.append(i)
  sv = np.array(sv)
  misclassified = sv[misclassified]


  plt.scatter(X[:60,0], X[:60,1], label='Class 1')
  plt.scatter(X[60:,0], X[60:,1], c='red', marker=',', label='Class 2')
  plt.scatter(X[sv,0], X[sv,1], c='green', marker='x', label='Support Vectors')
  plt.scatter(X[misclassified,0], X[misclassified,1], c='orange', marker='.', label='Misclassified')
  x0 = np.linspace(-2,8,len(sv))

  decisionBoundary = (-wl-w[0,0]*x0)/w[0,1]
  upperMargin = (1-wl-w[0,0]*x0)/w[0,1]
  lowerMargin = (-1-wl-w[0,0]*x0)/w[0,1]
  # plt.figure()
  plt.xlim([-2, 8])
  plt.ylim([-2, 6])
  plt.plot(x0, decisionBoundary, label='Decision Boundary')
  plt.plot(x0, upperMargin, 'r--', label='Upper Margin Hyperplane')
  plt.plot(x0, lowerMargin, 'g--', label='Lower Margin Hyperplane')
  plt.title('Dual Form Soft Margin SVM. C: {}. \nNumber of Support Vectors: {}. Misclassifications: {}.'.format(C, len(sv), len(misclassified)))
  plt.legend()
  plt.show()



nIter = 10
nSampleGroupsList = list(range(1,nIter+1))
timeQuadProg = []
timeSk = []
for nSampleGroups in nSampleGroupsList:
  mean1 = [1,3]
  cov1 = [[1,0], [0,1]]
  mean2 = [4,1]
  cov2 = [[2,0], [0,2]]
  X = np.vstack((np.random.multivariate_normal(mean1, cov1, size=60*nSampleGroups),np.random.multivariate_normal(mean2, cov2, size=40*nSampleGroups)))


  y=np.vstack((np.ones((60*nSampleGroups,1)), -np.ones((40*nSampleGroups,1)))).ravel()

  quadProgStart = time.time()
  # Construct the matrices for the QP problem
  N = len(X) # number of training examples
  dist_matrix = distance.cdist(X, X, 'euclidean')
  K = np.exp(-np.square(dist_matrix) / (2 * 1.75**2)) # kernel matrix with Gaussian kernel
  P = matrix(np.outer(y, y) * K)
  q = matrix(-np.ones(N))
  G = matrix(np.vstack((-np.eye(N), np.eye(N))))
  h = matrix(np.hstack((np.zeros(N), 100*np.ones(N))))
  A = matrix(y.reshape(1, -1))
  b = matrix(0.0)

  
  sol = solvers.qp(P, q, G, h, A, b)

  lambdas = np.array(sol['x']).reshape(-1)

  quadProgStop = time.time() - quadProgStart 
  timeQuadProg.append(quadProgStop)
  # print(f"CVXOPT Execution Time for C={C}: {quadProgStop:.4f} seconds")

  skStart = time.time()
  clf = svm.SVC(kernel='rbf', gamma=1/(2*1.75**2))
  clf.fit(X, y)
  skStop = time.time() - skStart 
  timeSk.append(skStop)

nSampleGroupsList = [x * 100 for x in nSampleGroupsList]
plt.figure(figsize=(10, 8))
plt.xlim([100, 1000])
plt.ylim([0, 1.5])
plt.plot(nSampleGroupsList, timeQuadProg, label='Time For Quad Prog Implementation')
plt.plot(nSampleGroupsList, timeSk, label='Time For Sklearn SMO Implementation')
plt.xlabel('Number of Samples')
plt.ylabel('Time in Seconds')

plt.legend()
plt.show()


input("press a key to close script")



