import numpy as np
import sys

# Generate distance matrix: D
def distMatrix(X,alpha):
    n = X.shape[0]
    D = np.zeros((n,n))
    for i in range(n):
       for j in range(n):
           D[i][j] = np.power(np.linalg.norm(np.array(X[j]) - np.array(X[i])), a)
    return D
    

# MDS function base on D
def MDS(D,d):
    D = np.asarray(D)
    DSquare = np.power(D, 2)
    totalMean = np.mean(DSquare)
    columnMean = np.mean(DSquare, axis = 0)
    rowMean = np.mean(DSquare, axis = 1)
    G = np.zeros(DSquare.shape)
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            G[i][j] = -0.5 * (DSquare[i][j] - rowMean[i] - columnMean[j] + totalMean)
    
    # EVD of G
    eigVal, eigVec = np.linalg.eigh(G)
    #sort eigVal
    sorted_idxes = np.argsort(-eigVal)
    #get d largest eigVec
    eigVal = eigVal[sorted_idxes]
    eigVec = eigVec[:,sorted_idxes]
    eigVec = eigVec[:,:d]
    eigVal = eigVal[0:d]
    eigVal[eigVal < 0]=0 # if eigVal < 0 replace with 0
    eigVal = np.power(eigVal,0.5) 
    
    #X = np.dot(np.diag(eigVal),eigVec.T)
    X = np.real(eigVec * eigVal.T)
    return X

# Parameters 
if len(sys.argv) != 3:
    print('usage : ', sys.argv[0], 'file_2dpoints [file_labels]')
    sys.exit()


# Input
X = np.genfromtxt(sys.argv[1], delimiter=',', autostrip=True)
a = 0.1
if len(sys.argv) == 3:
    a = float(sys.argv[2])
  
# main 
D = distMatrix(X, a)
d = 2
D2d = MDS(D,d)
a1 = ''.join(str(a).split('.'))
# Write array to a file
temp = sys.argv[1]
fileName = temp.split('.')[0] + '2d_mds'+ a1 +'.data'
np.savetxt(fileName , D2d, delimiter=',')
print(fileName,a)