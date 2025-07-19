from util.Visualization import plot_multilabel_cf
import numpy as np
cnf_matrix = np.array([[[127,1],[2,126]],
                       [[128,0],[0,128]],
                       [[128,0],[0,128]],
                       [[126,2],[4,124]],
                       [[126,2],[3,125]],
                      [[128,0],[0,128]]])
plot_multilabel_cf(cnf_matrix)