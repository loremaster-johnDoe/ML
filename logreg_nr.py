# Binary-response Logistic Regression model using Newton-Raphson method 
# also known as Iteratively Reweighted Least Squares or IRLS

import numpy as np

class LogisticRegression():
    
    def __init__(self, X, y, p=0.5, add_intercept=True, normalise_req=True):
        self.X = X
        self.y = y
        self.p = p
        self.add_intercept = add_intercept
        self.normalise_req = normalise_req
        if add_intercept:
            self.X = np.hstack((np.ones((self.X.shape[0],1)), self.X))
            self.features = self.X.shape[1] - 1
        else:
            self.features = self.X.shape[1]
        self.n_rows = self.X.shape[0]
        self.coeffs_ = np.zeros((self.X.shape[1]))
        
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def loss(self, y, y_hat):
        return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
    
    def normalise(self, X):
        if self.normalise_req:
            m, n = X.shape
            if not self.add_intercept:
                for i in range(n):
                    X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
            else:
                for i in range(1,n):
                    X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
            return X
        else:
            return X
        
    def fit(self, iter_=10000):
        for i in range(iter_):
            z = np.dot(self.X, self.coeffs_)
            prob = self.sigmoid(z)
            hessian = np.dot(np.dot(self.X.T, np.diag(prob*(1-prob))), self.X)
            self.coeffs_ += np.dot(np.dot(np.linalg.inv(hessian), self.X.T), (self.y-prob))
        return self
    
    def y_hat(self):
        return self.sigmoid(self.X*self.coeffs_)
    
    def log_odds(self):
        odds = self.y_hat()/(1-self.y_hat())
        return np.log(odds)