import numpy as np
import scipy.stats

class LinearRegression():
    
    def __init__(self, X, y, add_intercept=True):
        """
        X : matrix of covariates (numpy array) of order n*k
        y : dependent var (numpy array)
        add_intercept : True if you need to perform a regression with a constant and the input X doesn't already contain a column of 1's. Otherwise False
        Note : Categorical variables will have to be treated exogenously prior to using this module.
        """
        self.X = X
        self.y = y
        self.add_intercept = add_intercept
        if add_intercept:
            self.X = np.hstack((np.ones((self.X.shape[0],1)), self.X))
            self.features = self.X.shape[1] - 1
        else:
            self.features = self.X.shape[1]
            pass
        self.n_rows = self.X.shape[0]
        self.coeffs_ = None
        self.intercept_ = None
        
        
    def fit(self):
        """
        β = (X'X)^(-1)X'y
        """
        self.coeffs_ = np.dot(np.linalg.inv(np.dot(self.X.T, self.X)), np.dot(self.X.T, self.y))
        self.intercept_ = self.coeffs_[0]
        return self
    
    def get_params(self):
        return self.coeffs_
    
    def y_hat(self):
        return np.dot(self.X, self.get_params())
    
    def r2(self):
        """
        R-squared value
        """
        return (1 - np.sum((self.y - self.y_hat())**2)/ (np.sum((self.y - np.mean(self.y))**2)))
    
    def adjusted_r2(self):
        """
        Adjusted R-squared value
        """
        n = self.n_rows
        return (1-(1-self.r2()**2)*(n-1)/ (n-self.X.shape[1]))
    
    def var_estimator(self):
        """
        s^2 = e'e/(n-K)
        where e = error vector
        n = # of rows
        K = # of covariates (including intercept)
        """
        e = self.y - self.y_hat()
        return (1/(self.n_rows - self.X.shape[1]))*np.dot(e.T, e)
    
    def std_errors(self):
        """
        Σ = s^2*(X'X)^(-1)
        Formula for the var-covar matrix
        the diagonal elements give the individual std. errors of the coefficients
        """
        XX = np.linalg.inv(np.dot(self.X.T, self.X))
        return np.array([self.var_estimator()*XX[i][i] for i in range(self.X.shape[1])])
    
    def t_values(self):
        """
        t-statistic values for each coefficient
        where H0: β = 0
        H1: β != 0
        """
        return self.get_params()/self.std_errors()
    
    def p_values(self):
        """
        p-values of the coefficients under a two-tailed test assumption.
        Exact probability of committing a Type-1 error
        """
        return 2*(1-scipy.stats.t.cdf(np.abs(self.t_values()), df = self.n_rows - self.X.shape[1]))
    
    def F_stat(self):
        """
        Calculates F-statistic of overall joint significance of the regression model
        H0 : Intercept only model is a better fit (i.e. each β is not statistically distinct from 0)
        H1 : At least one variable is statistically distinct from 0
        Formula : F = (ESS/df)/(RSS/df)
        Derived here using it's relationship with R^2
        Returns F-statistic value and corresponding p-value
        """
        if self.add_intercept == False:
            print("Not calculated because regression without intercept was used.")
            return None
        else:
            f = (self.r2()/(1-self.r2())) * ((self.n_rows - self.features - 1)/self.features)
            p = 1 - scipy.stats.f.cdf(f, self.n_rows - self.features - 1, self.features)
            return np.array([f, p])
        
