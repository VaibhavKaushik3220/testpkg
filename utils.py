
def compute_roc(y_scores,y_true):
    fpr,tpr,_ = metrics.roc_curve(y_true, y_scores)
    return fpr,tpr

def compute_auc(y_scores,y_true):
    auc = metrics.roc_auc_score(y_true,y_scores)
    return auc

def interpolate_roc_fun(fpr,tpr,n_grid):
    roc_approx = interpolate.interp1d(x = fpr, y = tpr)
    x_new = np.linspace(0,1,num = n_grid)
    y_new = roc_approx(x_new)
    return x_new,y_new
    
def slice_plot(majority_roc_x, minority_roc_x,
               majority_roc_y, minority_roc_y, majority_group_name = 'baseline', minority_group_name = 'comparison', fout = './slice_plot.png'):
    plt.figure(1, figsize=(6,5))
    plt.title('ABROCA - Slice Plot')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(majority_roc_x, majority_roc_y, label = "{o} - Baseline".format(o=majority_group_name), linestyle='-', color='r')
    plt.plot(minority_roc_x, minority_roc_y, label = "{o} - Comparison".format(o=minority_group_name), linestyle='-', color='b')
    plt.fill(majority_roc_x.tolist()+np.flipud(minority_roc_x).tolist(), majority_roc_y.tolist()+np.flipud(minority_roc_y).tolist(),"y")
    plt.legend()
    plt.savefig(fout)
    plt.show()