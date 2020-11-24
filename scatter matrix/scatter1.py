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
#print("M:",M)

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
#minimize the mixture-class scatter . value
eMin = e[:2]
Umin = U[:,:2]
n1,m1 = Umin.shape
# print("n1=",n1)
# print("m1=",m1)
#print("eMin=",eMin)
#print("Umin=",Umin)

X2d_1 =Xt@Umin
fileName = sys.argv[3]
np.savetxt(fileName, X2d_1, delimiter=',')