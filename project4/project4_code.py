import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_excel('project4\Proj4Train100.xlsx', header=None)
# print(df)
X100, y100 = df.to_numpy()[:,:-1],df.to_numpy()[:,-1]
# print(X100.shape)
# print(len(y100))
df = pd.read_excel('project4\Proj4Train1000.xlsx', header=None)
X1000, y1000 = df.to_numpy()[:,:-1],df.to_numpy()[:,-1]
# print(X)
# print(y)
df = pd.read_excel('project4\Proj4Test.xlsx', header=None)
Xtest, yTest = df.to_numpy()[:,:-1],df.to_numpy()[:,-1]
# print(len(Xtest), len(yTest))


class NaiveBayes:
    def __init__(self):
        self.means = None
        self.variances = None
        self.priors = None

    def fit(self, X, y):
        nSamples, nFeatures = X.shape
        nClasses = len(np.unique(y))

        self.means = np.zeros((nClasses, nFeatures))
        self.variances = np.zeros((nClasses, nFeatures))
        self.priors = np.zeros(nClasses)

        for c in range(nClasses):
            Xc = X[y==(c+1)]
            # print(Xc)
            self.means[c, :] = Xc.mean(axis=0)
            self.variances[c, :] = Xc.var(axis = 0)
            self.priors[c] = Xc.shape[0] / float(nSamples)
            # print(self.means)
            print(self.variances)
            # print(self.priors)

    def predict(self, X):
        predictedYs = np.log(self.priors) + np.sum(-0.5 * np.log(2 * np.pi * self.variances) - (X[:, np.newaxis] - self.means)**2 / (2 * self.variances), axis=2)
        return np.argmax(predictedYs, axis=1)
    
class BayesMLE:
    def __init__(self):
        self.means = None
        self.covariances = None
        self.priors = None
    
    def fit(self, X, y):
        nSamples, nFeatures = X.shape
        nClasses = len(np.unique(y))

        self.means = np.zeros((nClasses, nFeatures))
        self.covariances = np.zeros((nClasses, nFeatures, nFeatures))
        self.priors = np.zeros(nClasses)

        for c in range(nClasses):
            Xc = X[y==(c+1)]
            nSamplesC = Xc.shape[0]

            self.means[c, :] = Xc.mean(axis=0)

            self.covariances[c,:,:] = np.cov(Xc.T)

            self.priors[c] = nSamplesC / float(nSamples)

    def predict(self, X):
        nClasses = self.means.shape[0]
        predictedYs = np.zeros((X.shape[0], nClasses))

        for c in range(nClasses):
            predictedYs[:, c] = np.log(self.priors[c]) + mvLogPDF(X, self.means[c], self.covariances[c])

        return np.argmax(predictedYs, axis = 1)

class BayesTrue:
    def __init__(self, means, covariances, priors):
        self.means = means
        self.covariances = covariances
        self.priors = priors

    def predict(self, X):
        nClasses = self.means.shape[0]
        predictedYs = np.zeros((X.shape[0], nClasses))

        for c in range(nClasses):
            predictedYs[:, c] = np.log(self.priors[c]) + mvLogPDF(X, self.means[c], self.covariances[c])

        return np.argmax(predictedYs, axis = 1)

def mvLogPDF(X, mean, cov):
    nFeatures = mean.shape[0]
    covDet = np.linalg.det(cov)
    covInv = np.linalg.inv(cov)
    centeredX = X - mean

    logPDF = -0.5 * np.log(covDet) - 0.5 * np.sum(centeredX @ covInv * centeredX, axis=1) - 0.5 * nFeatures * np.log(2 * np.pi)

    return logPDF    

for X, y in [(X100, y100), (X1000, y1000)]:
  print('\nDataset size: {}'.format(len(y)))

  nb = NaiveBayes()
  nb.fit(X,y)
  yPreds = nb.predict(Xtest)
  misclass = 0
  for i in range(len(yPreds)):
      if (yPreds[i]+1) != yTest[i]:
          misclass += 1

  print("Naive Baye's Accuracy: {:.4f}".format((misclass/len(yPreds))))

  bMLE = BayesMLE()
  bMLE.fit(X, y)
  yPreds = bMLE.predict(Xtest)
  misclass = 0
  for i in range(len(yPreds)):
      if (yPreds[i]+1) != yTest[i]:
          misclass += 1

  print("Baye's MLE Accuracy: {:.4f}".format((misclass/len(yPreds))))

  means = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
  covariances = np.array([
      [
          [0.8, 0.2, 0.1, 0.05, 0.01],
          [0.2, 0.7, 0.1, 0.03, 0.02],
          [0.1, 0.1, 0.8, 0.02, 0.01],
          [0.05, 0.03, 0.02, 0.9, 0.01],
          [0.01, 0.02, 0.01, 0.01, 0.8]
      ],
      [
          [0.9, 0.1, 0.05, 0.02, 0.01],
          [0.1, 0.8, 0.1, 0.02, 0.02],
          [0.05, 0.1, 0.7, 0.02, 0.01],
          [0.02, 0.02, 0.02, 0.6, 0.02],
          [0.01, 0.02, 0.01, 0.02, 0.7]
      ]
  ])
  priors = np.array([0.5, 0.5])

  trueBayes = BayesTrue(means, covariances, priors)
  yPreds = trueBayes.predict(Xtest)
  misclass = 0
  for i in range(len(yPreds)):
      if (yPreds[i]+1) != yTest[i]:
          misclass += 1

  print("Baye's True Parameters Accuracy: {:.4f}".format((misclass/len(yPreds))))