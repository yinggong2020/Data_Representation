import sys
import numpy as np


if len(sys.argv) != 2 and len(sys.argv) != 3 :
    print('usage : ', sys.argv[0], 'please specify a data file')
    sys.exit()


# centeredPCA1 function 
def centeredPCA(X, r):
    mu = np.mean(X, axis=0)
    X = X - mu
    
    e,U = np.linalg.eigh(X.T@X)
    sorted_idxes = np.argsort(-e)
    e = e[sorted_idxes]
    U = U[:, sorted_idxes]
    Ur = U[:,:r]
    W = X@Ur
    return W

# generate the matrix
X = np.genfromtxt(sys.argv[1],delimiter=',',autostrip=True)
# main 
r = 2
X2d = centeredPCA(X,r)
# Write array to a file
temp = sys.argv[1]
fileName = temp.split('.')[0] + '2d_pca2.data'
np.savetxt(fileName , X2d, delimiter=',')