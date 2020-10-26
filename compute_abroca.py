from utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics 
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

def compute_abroca(df, pred_col, label_col, protected_attr_col,
                           majority_protected_attr_val, n_grid = 10000,
                           plot_slices = True, lb=0, ub=1, limit = 10000, image_dir = "./",
                           identifier = 'filename.png'):
    #Compute the value of the abroca statistic.
    if df[pred_col].between(0, 1,inclusive=True).any():
        pass
    else:
        print('predictions must be in range [0,1]')
        exit(1)
    if len(df[label_col].value_counts) == 2:
        pass
    else:
        print("The label column should be binary")
        exit(1)
    if len(df[protected_attr_col].value_counts) == 2:
        pass
    else:
        print("The protected attribute column should be binary")
        exit(1)
    # initialize data structures
    # slice_score = 0
    prot_attr_values = df[protected_attr_col].value_counts().index.values
    fpr_tpr_dict = {}
    
    # compute roc within each group of pa_values
    for pa_value in prot_attr_values:
        pa_df = df[df[protected_attr_col] == pa_value]
        fpr_tpr_dict[pa_value] = compute_roc(pa_df[pred_col],pa_df[label_col])
        
    # compare minority to majority class; accumulate absolute difference btw ROC curves to slicing statistic
    
    majority_roc_x, majority_roc_y = interpolate_roc_fun(fpr_tpr_dict[1][0],fpr_tpr_dict[1][1],n_grid)
    minority_roc_x, minority_roc_y = interpolate_roc_fun(fpr_tpr_dict[0][0],fpr_tpr_dict[0][1],n_grid)
    
    # use function approximation to compute slice statistic via piecewise linear function
    if (majority_roc_x.tolist()==minority_roc_x.tolist()):
        f1 = interpolate.interp1d(x = majority_roc_x, y = (majority_roc_y-minority_roc_y))
        f2 = lambda x: abs(f1(x))
        slice, _ = integrate.quad(f2, lb, ub, limit)
    else:
        print("Majority and minority FPR are different")
        exit(1)
    
    if (plot_slices == True):
        slice_plot(majority_roc_x, minority_roc_x,majority_roc_y, minority_roc_y, majority_group_name = 'baseline', minority_group_name = 'comparison', fout = './slice_plot.png')
        
    return slice
    
        