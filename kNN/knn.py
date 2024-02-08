# IMPORTANT - This code uses the Euclidean metric for all distances. Kindly use a different code for other requirements.

import numpy as np
from scipy.stats import mode

def euclidean(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

def predict(x_train, y, x_input, k):
    """
    x_train - covariates
    y - dependent var
    x_input - data points to be classified
    k - number of neighbours to be analysed
    """
    
    op_labels = []
    for item in x_input:
        point_dist = []
        for j in range(len(x_train)):
            distances = euclidean(np.array(x_train[j,:]), item)
            point_dist.append(distances)
        point_dist = np.array(point_dist)
        
        # Sorting the array while preserving the index
        # keeping the first K datapoints
        dist = np.argsort(point_dist)[:k]
        labels = y[dist]
        
        # Majority voting
        lab = mode(labels)
        lab = lab.mode[0]
        op_labels.append(lab)
        
    return op_labels, dist
