import sys
import numpy as np
# data Xt   group y
Xt = np.genfromtxt(sys.argv[1],delimiter=',',autostrip=True)
y = np.genfromtxt(sys.argv[2],delimiter=',',autostrip=True)
n,m = Xt.shape
# print("n=",n)
# print("m=",m)
#mu
mu=np.mean(Xt,axis=0) # mean of column
#print("mu=",mu)
#The mixture (total) scatter matrix: M
M = np.zeros((m,m))
for xt in (Xt) :
    M += np.outer((xt - mu),(xt - mu).T)
M = np.mat(M)
# print("M:",M)

# 1: minimize the mixture-class scatter.
# EVD Of M
e,U = np.linalg.eigh(M)
# print("e=",e)
# print("U=",U)
sorted_idxes = np.argsort(e)
e = e[sorted_idxes]
U = U[:, sorted_idxes]
# print("e=",e)
# print("U=",U)

# maximize the mixture-class scatter.
eMax = e[-2:]
Umax = U[:,-2:]
# print("eMax=",eMax)
# print("Umax=",Umax)

X2d_2 = Xt@Umax
fileName = sys.argv[3]
np.savetxt(fileName, X2d_2, delimiter=',')