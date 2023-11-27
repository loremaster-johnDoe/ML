# Linear Discriminant Analysis model
# source : Hastie, Tibshirani and Friedman - The Elements of Statistical Learning
# Assumption : different classes of y have the same covariance matrix Σ
# Assumption : Gaussian probability density has been used

import numpy as np

class LDA():
    
    def __init__(self, X, y, add_intercept=True):
        self.X = X
        self.y = y
        self.add_intercept = add_intercept
        if add_intercept:
            self.X = np.hstack((np.ones((self.X.shape[0],1)), self.X))
            self.features = self.X.shape[1] - 1
        else:
            self.features = self.X.shape[1]
            pass
        
    def generate_params(self):
        """
        we generate here, the parameters used in the gaussian density function
        namely, 1. the prior probability of each class, 2. the mean for each x variable and 3. the covariance matrix
        """
        unique, counts = np.unique(self.y, return_counts=True)
        prior_prob = [i/counts.sum() for i in counts]
        
        # mean
        means = np.array([self.X[np.where(self.y==i)].sum(axis=0)/tuple(self.y).count(i) for i in np.unique(self.y)])
        
        # covariance
        x_centered = self.X
        x_centered = x_centered.astype('float')
        for i in np.unique(self.y):
            z = np.zeros(self.X.shape)
            z[np.where(self.y==i)] = np.array([self.X[np.where(self.y==i)].sum(axis=0)/tuple(self.y).count(i)])
            x_centered -= z
        
        cov = np.dot(x_centered.T, x_centered)/(self.X.shape[0]-len(np.unique(self.y)))
        
        return prior_prob, means, cov
    
    def log_likelihood_ratio(self, class1 = i, class2 = j, x = None):
        """
        We calculate here, the pairwise log-likelihood ratio between two different classes.
        log(P(class i)/ P(class j)) classifies to class i if the log value > 0, otherwise class j
        x : the X row for which the class is being predicted
        returns the llr value and the recommended class.
        """
        distinct_y = np.unique(self.y)
        idx_i = np.where(distinct_y==class1)[0][0]
        idx_j = np.where(distinct_y==class2)[0][0]
        
        prior_prob, means, cov = self.generate_params()
        
        prior_i = prior_prob[idx_i]
        prior_j = prior_prob[idx_j]
        
        means_i = means[idx_i]
        means_j = means[idx_j]
        
        llr = np.log(prior_i/prior_j) - 0.5*np.dot(np.dot((means_i+means_j).T, np.linalg.inv(cov)), means_i-means_j) + np.dot(np.dot(x.T, np.linalg.inv(cov)), means_i-means_j)
        
        if llr > 0:
            recommended_class = i
            print(f"recommended class is : {i}")
        elif llr < 0:
            recommended_class = j
            print(f"recommended class is : {j}")
        else:
            recommended_class = None
            print("No recommendation, since llr is 0")
        
        return llr, recommended_class
