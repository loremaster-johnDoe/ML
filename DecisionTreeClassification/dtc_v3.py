import numpy as np
import pandas as pd

class DecisionTree:
    
    def __init__(self, table, y, min_samples=20, max_depth=5):
        self.table = table
        self.y = y
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.tree = {}
    
    def gini(self, df, col):
        if col not in df.columns:
            raise Exception(f"{col} is not a column in the given DataFrame")
        
        total_gini = 0
        unique_vals = df[col].value_counts().index.to_list()
        dict_gini = {}
        best_split = None

        if df[col].dtype == "O":  # Categorical variable
            for val in unique_vals:
                sub_df = df[df[col] == val]
                prob1 = sub_df[self.y].mean()
                prob0 = 1 - prob1
                gini_val = 1 - prob1**2 - prob0**2
                dict_gini[val] = gini_val
                total_gini += (len(sub_df) / len(df)) * gini_val
            return total_gini, best_split
        
        else:  # Numerical variable
            unique_vals.sort()
            unique_means = [(unique_vals[i] + unique_vals[i+1]) / 2 for i in range(len(unique_vals)-1)]
            for mean in unique_means:
                left_split = df[df[col] < mean]
                right_split = df[df[col] >= mean]
                
                prob_left1 = left_split[self.y].mean()
                prob_left0 = 1 - prob_left1
                gini_left = 1 - prob_left1**2 - prob_left0**2
                
                prob_right1 = right_split[self.y].mean()
                prob_right0 = 1 - prob_right1
                gini_right = 1 - prob_right1**2 - prob_right0**2
                
                weighted_gini = (len(left_split) / len(df)) * gini_left + (len(right_split) / len(df)) * gini_right
                dict_gini[mean] = weighted_gini
            
            total_gini = min(dict_gini.values())
            best_split = min(dict_gini, key=dict_gini.get)
            return total_gini, best_split
    
    def fit(self, df=None, depth=0, tree=None):
        if df is None:
            df = self.table
        
        if tree is None:
            tree = self.tree
        
        if depth > self.max_depth or len(df) < self.min_samples:
            return
        
        x_cols = [col for col in df.columns if col != self.y]
        
        gini_dict = {}
        best_split_dict = {}
        
        for col in x_cols:
            gini, split = self.gini(df, col)
            gini_dict[col] = gini
            best_split_dict[col] = split
        
        min_gini = min(gini_dict.values())
        best_var = min(gini_dict, key=gini_dict.get)
        best_split = best_split_dict[best_var]
        
        print(f"Split info at depth = {depth+1}: Best var = {best_var}, best split value = {best_split}, Gini = {min_gini}")
        
        if min_gini == 0.0:
            print("Perfect split achieved.")
            return
        
        if best_split is None:  # For categorical variables
            tree[f"{best_var}"] = {}
            unique_vals = df[best_var].value_counts().index.to_list()
            for val in unique_vals:
                sub_df = df[df[best_var] == val]
                tree[f"{best_var}"][f"=={val}"] = {}
                self.fit(sub_df, depth + 1, tree[f"{best_var}"][f"=={val}"])
        else:  # For numerical variables
            tree[f"{best_var} < {best_split}"] = {}
            tree[f"{best_var} >= {best_split}"] = {}
            left_split = df[df[best_var] < best_split]
            right_split = df[df[best_var] >= best_split]
            self.fit(left_split, depth + 1, tree[f"{best_var} < {best_split}"])
            self.fit(right_split, depth + 1, tree[f"{best_var} >= {best_split}"])
    
    def print_tree(self, tree=None, indent="  "):
        if tree is None:
            tree = self.tree
        
        for key, value in tree.items():
            print(indent + str(key))
            if isinstance(value, dict):
                self.print_tree(value, indent + "  ")
