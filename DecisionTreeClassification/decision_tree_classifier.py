import numpy as np
import pandas as pd

class DecisionTree():
    
    def __init__(self, table, y, min_samples=100, max_depth=5):
        """
        table : pd dataframe
        y : name of dependent var
        min_samples : minimum number of samples at the leaf node
        max_depth : max depth allowed in the tree
        """
        self.table = table
        self.y = y
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.perfect_split = False
        
    def gini(self, df, col):
        """
        df : pd dataframe
        col : column for which gini is being evaluated
        returns the gini impurity for the given column
        """
        if col not in df.columns:
            raise Exception(f"{col} is not a column in the given dataframe")
        else:
            pass
        total_gini = 0
        unique_vals = df[col].value_counts().index.to_list()
        dict_pop = {}
        dict_gini = {}
        
        if df[col].dtype == "O":
            for i in unique_vals:
                dict_pop[i] = df[df[col]==i].shape[0]
                n1 = df[df[col]==i].loc[df[y]==1,].shape[0]
                n0 = df[df[col]==i].loc[df[y]==0,].shape[0]
                prob1 = n1/(n1+n0)
                prob0 = n0/(n1+n0)
                dict_gini[i] = 1 - prob1**2 - prob0**2
                if dict_gini[i] == 0.0:
                    print(f"perfect split achieved for column : {col}, value : {i}")
                total_gini += (dict_pop[i]/df.shape[0])*dict_gini[i]
                best_split = None
        
        if df[col].dtype in ["int", "float"]:
            if len(unique_vals)==1:
                n1 = df.loc[df[y]==1,].shape[0]
                n0 = df.loc[df[y]==0,].shape[0]
                prob1 = n1/(n1+n0)
                prob0 = n0/(n1+n0)
                total_gini = 1 - prob1**2 - prob0**2
                if total_gini == 0.0:
                    print(f"perfect split achieved for column : {col}")
            else:
                pop_lt = {}
                pop_gt = {}
                unique_vals.sort()
                unique_means = [(unique_vals[i]+unique_vals[i+1])/2 for i in range(len(unique_vals)-1)]
                for i in unique_means:
                    pop_lt[i] = df[df[col] < i].shape[0]
                    pop_gt[i] = df[df[col] > i].shape[0]
                    n_lt1 = df[df[col] < i].loc[df[y]==1,].shape[0]
                    n_lt0 = df[df[col] < i].loc[df[y]==0,].shape[0]
                    n_gt1 = df[df[col] > i].loc[df[y]==1,].shape[0]
                    n_gt0 = df[df[col] > i].loc[df[y]==0,].shape[0]
                    prob_lt1 = n_lt1/(n_lt1 + n_lt0)
                    prob_lt0 = 1 - prob_lt1
                    prob_gt1 = n_gt1/(n_gt1 + n_gt0)
                    prob_gt0 = 1 - prob_gt1
                    gini_lt = 1 - prob_lt1**2 - prob_lt0**2
                    gini_gt = 1 - prob_gt1**2 - prob_gt0**2
                    dict_gini[i] = (pop_lt[i]/df.shape[0])*gini_lt + (pop_gt[i]/df.shape[0])*gini_gt
                total_gini = min(dict_gini.values())
                best_split = list(dict_gini.keys())[list(dict_gini.values()).index(total_gini)]
                best_left_pop1 = df[df[col] < best_split].loc[df[y]==1,].shape[0]
                best_left_pop0 = df[df[col] < best_split].loc[df[y]==0,].shape[0]
                best_right_pop1 = df[df[col] > best_split].loc[df[y]==1,].shape[0]
                best_right_pop0 = df[df[col] > best_split].loc[df[y]==0,].shape[0]
                if best_left_pop1 == 0 or best_left_pop0 == 0:
                    print(f"perfect lt split achieved for column : {col}, value : {best_split}")
                if best_right_pop1 == 0 or best_right_pop0 == 0:
                    print(f"perfect gt split achieved for column : {col}, value : {best_split}")
                print(f"best split for {col} occurs at {best_split}")
                
        return total_gini, best_split
    
    def fit(self, df, depth=0):
        """
        """
        if depth > self.max_depth:
            print("max depth exceeded")
            continue
        elif df.shape[0] < self.min_samples:
            print("fewer than min_samples obs left in the node at this stage")
            continue
        else:
            pass
        
        x_cols = [i for i in self.table.columns if i != self.y]

        gini_dict = {}
        best_split_dict = {}
        for i in x_cols:
            gini_dict[i] = self.gini(df, i)[0]
            best_split_dict[i] = self.gini(df, i)[1]
        min_gini = min(gini_dict.values())
        best_var = list(gini_dict.keys())[list(gini_dict.values()).index(min_gini)]
        best_split = best_split_dict[best_var]
        print(f"Split info at depth = {depth+1} : Best var = {best_var}, best split value = {best_split}, gini = {min_gini}")
        if best_split == None:
            if min_gini == 0.0:
                print("perfect split achieved in this node")
                continue
            unique_vals = df[best_var].value_counts().index.to_list()
            df_split = {}
            n_splits = len(unique_vals)
            depth += 1
            for i in range(n_splits):
                df_split[i] = df[df[best_var]==unique_vals[i]]
                self.fit(df_split[i], depth)
        else:
            df_split_left = df[df[best_var] < best_split]
            df_split_right = df[df[best_var] >= best_split]
            depth += 1
            self.fit(df_split_left, depth)
            self.fit(df_split_right, depth)