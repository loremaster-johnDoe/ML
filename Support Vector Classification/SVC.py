import numpy as np
import random

class SVC():
    
    def __init__(self, X, y, C=1.0, add_intercept=True, iter_=1000, learning_rate=0.001):
        """
        X : matrix of covariates (numpy array) of order n*k
        y : dependent binary var having values in the set {-1,1} (numpy array)
        C : regularization parameter
        add_intercept : adds an intercept column to the matrix of covariates, X
        iter_ : no. of iterations
        learning_rate : the learning rate parameter for updating beta coefficient
        Note : Categorical variables will have to be treated exogenously prior to using this code.
        """
        self.X = X
        self.y = y
        self.C = C
        self.add_intercept = add_intercept
        if add_intercept:
            self.X = np.hstack((np.ones((self.X.shape[0],1)), self.X))
            self.features = self.X.shape[1] - 1
        else:
            self.features = self.X.shape[1]
            pass
        self.beta = np.zeros((self.X.shape[1], 1))
        self.iter_ = iter_
        self.learning_rate = learning_rate
        
    def loss_function(self, X, y, b):
        """
        loss = 0.5*||b||^2 + C*Σξi
        where ξ = max(0, 1-y*(Xb+b0))
        """
        part1 = 0.5*np.dot(b[1:].T,b[1:])
        yxb_ = y*np.dot(X,b)
        xi = max(0, 1-yxb_)
        part2 = self.C*np.sum(xi)
        return part1+part2, xi
    
    def fit(self):
        """
        with the loss function being convex, the gradient for beta will eventually converge to a minima.
        gradient_beta = Σ αi*yi*xi
        where αi = C for points that have ξi > 0 (well-behaving points)
        while, 0 < αi < C for points that have ξi = 0 (points that either lie on the margin or exceed beyond into the "other" territory.
        """
        for i in range(self.iter_):
            loss, xi = self.loss_function(self.X, self.y, self.beta)
            mult_factor = np.where(xi==0.0, random.random(), 1)
            cyx_ = self.C*self.y*self.X
            gradient = mult_factor*cyx_
            self.beta += (1 - self.learning_rate)*gradient
        return self
    
    def predict(self, X):
        """
        pred = sign(Xb)
        """
        pred = np.dot(X, self.beta)
        sgn_pred = np.where(pred >= 0, 1, -1)
        return sgn_pred
