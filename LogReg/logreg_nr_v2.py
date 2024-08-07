#uses Newton-Raphson method to build a binary-response Logistic Regression model (IRLS)

import numpy as np

class LogisticRegression():
    
    def __init__(self, X, y, p=0.5, add_intercept=True, max_iter=10000, tol=1e-6, hessian_tweak=1e-5):
        """
        X : matrix of covariates (numpy array)
        y : dependent var (numpy array)
        p : classification benchmark (y_pred > p then 1 else 0)
        add_intercept : whether an intercept column needs to be added to X
        max_iter : maximum number of iterations
        tol : value used as stopping criterion i.e. when successive iterations don't change the weights "sufficiently"
        hessian_tweak : value used to tweak Hessian diagonals slightly to ensure non-singularity of the resulting matrix
        """
        self.X = X
        self.y = y
        self.p = p
        self.add_intercept = add_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.hessian_tweak = hessian_tweak
        if add_intercept:
            self.X = np.hstack((np.ones((self.X.shape[0],1)), self.X))
        self.coeffs_ = None
    
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def logistic_regression_newton(self):
        
        """
        Returns updated weights/ coefficients
        """

        n_samples, n_features = self.X.shape
        self.coeffs_ = np.zeros(n_features)

        for i in range(self.max_iter):
            z = np.dot(self.X, self.coeffs_)
            h = self.sigmoid(z)

            # Gradient
            gradient = np.dot(self.X.T, (h - self.y)) / n_samples

            # Hessian
            R = np.diag(h * (1 - h))
            Hessian = np.dot(np.dot(self.X.T, R), self.X) / n_samples + self.hessian_tweak * np.eye(n_features)

            try:
                # Update weights using np.linalg.solve
                delta = np.linalg.solve(Hessian, gradient)
            except np.linalg.LinAlgError:
                # If Hessian is singular, use pseudo-inverse
                delta = np.dot(np.linalg.pinv(Hessian), gradient)

            self.coeffs_ -= delta

            # Check for convergence
            if np.linalg.norm(delta, ord=1) < self.tol:
                break

        return self.coeffs_


    def y_pred(self):

        """
        Returns predicted raw scores (predicted y values in [0,1])
        """

        y_ = self.sigmoid(np.dot(self.X,self.coeffs_))
        return y_

    
    def log_odds(self):
        odds = self.y_pred()/(1-self.y_pred())
        return np.log(odds)
    
    
    def y_hat(self):
        
        """
        Returns the classified y values based on p value set initially
        """
        
        hat = np.where(self.y_pred() >= self.p, 1, 0)
        return hat
    
    
    def confusion_matrix(self):
        
        """
        Returns the confusion matrix in the form :
        
               | Predicted_0 | Predicted_1 |
        ------------------------------------
        True_0 |             |             |
        ------------------------------------
        True_1 |             |             |
        ------------------------------------
        
        """
        
        cm00 = (self.y_hat()[np.where(self.y == 0)] == 0).sum()
        cm01 = (self.y_hat()[np.where(self.y == 0)] == 1).sum()
        cm10 = (self.y_hat()[np.where(self.y == 1)] == 0).sum()
        cm11 = (self.y_hat()[np.where(self.y == 1)] == 1).sum()
        cm = np.array([[cm00,cm01],[cm10,cm11]])
        return cm
        

    def sensitivity(self):
        
        """
        Sensitivity = TP / (TP + FN)
        Also known as Recall
        """
        
        se = self.confusion_matrix()[1][1] / (self.confusion_matrix()[1][1] + self.confusion_matrix()[1][0])
        return se
    
    
    def specificity(self):
        
        """
        Specificity = TN / (FP + TN)
        """
        
        sp = self.confusion_matrix()[0][0] / (self.confusion_matrix()[0][1] + self.confusion_matrix()[0][0])
        return sp
    
    
    def accuracy(self):
        
        """
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        """
        
        acc = (self.confusion_matrix()[1][1] + self.confusion_matrix()[0][0]) / self.X.shape[0]
        return acc
    
    
    def precision(self):
        
        """
        Precision = TP / (TP + FP)
        """
        
        pr = self.confusion_matrix()[1][1] / (self.confusion_matrix()[1][1] + self.confusion_matrix()[0][1])
        return pr
    
    
    def f1_score(self):
        
        """
        F1 score = 2*precision*recall/(precision + recall)
        """
        
        f1 = 2*self.precision()*self.sensitivity() / (self.precision() + self.sensitivity())
        return f1
    
    
    def roc_auc(self):
        
        """
        Calculates the ROC-AUC score for the model
        """
        
        roc = 0
        p_list = [0.01*i for i in range(1001)]
        coordinates = []
        for i in p_list[::-1]:
            
            roc_yhat = np.where(self.y_pred() >= i, 1, 0)
            
            tp = (roc_yhat[np.where(self.y == 1)] == 1).sum()
            fp = (roc_yhat[np.where(self.y == 0)] == 1).sum()
            tn = (roc_yhat[np.where(self.y == 0)] == 0).sum()
            fn = (roc_yhat[np.where(self.y == 1)] == 0).sum()
            
            fpr = fp/(fp+tn)
            tpr = tp/(tp+fn)
            coordinates.append((fpr,tpr))
        
        for i in range(len(coordinates)-1):
            area = (coordinates[i+1][0] - coordinates[i][0])*(coordinates[i+1][1] + coordinates[i][1])/2
            roc += area
        return roc
