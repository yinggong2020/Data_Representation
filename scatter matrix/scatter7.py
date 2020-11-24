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
# print("M:",M)
M = np.mat(M)
# mu1,mu2,mu3
## mu1
#B1 = np.zeros((m,m))
s1 = np.zeros((1,m))
m1 = 0
for i, xt in enumerate(Xt) :
    if y[i] == 1 :
        s1 += xt
        m1 += 1
#print("B1=", B1, "\n s1=", s1, "\n m1=", m1)
mu1 = s1/m1
# print("mu1=",mu1)

## mu2
#B2 = np.zeros((m,m))
s2 = np.zeros((1,m))
m2 = 0
for i, xt in enumerate(Xt) :
    if y[i] == 2 :
        s2 += xt
        m2 += 1
mu2 = s2/m2
#print("mu2=",mu2)

## mu3
#B3 = np.zeros((m,m))
s3 = np.zeros((1,m))
m3 = 0
for i, xt in enumerate(Xt) :
    if y[i] == 3 :
        s3 += xt
        m3 += 1
#print("B3=", B3, "\n s3=", s3, "\n m3=", m3)
mu3 = s3/m3
#print("mu3=",mu3)

# B
##compute B = X X'
##B = B1+B2+B3
#B = m1*(mu1-mu).T*(mu1-mu) + m2*(mu2-mu).T*(mu2-mu) + m3*(mu3-mu).T*(mu3-mu)
B = m1 * (mu1-mu).T.dot(mu1-mu) + m2 * (mu2-mu).T.dot(mu2-mu) + m3* (mu3-mu).T.dot(mu3-mu)
B = np.mat(B)
# print("B=", B)
# W
W = M-B
W = np.mat(W)
# print("W:",W)

# 7: minimize the ratio of between-class scatter and within-class scatter
# the ratio of between-class scatter and within-class scatter
W_1 = np.linalg.inv(W)
C = W_1.dot(B)
# EVD of C
e7,U7 = np.linalg.eig(C)
sorted_idxes = np.argsort(e7)
e7 = e7[sorted_idxes]
U7 = U7[:, sorted_idxes]
#print("e7=",e7)
#print("U7=",U7)
# minimize the within-class scatter
eMin7 = e7[:2]
Umin7 = U7[:,:2]
#print("eMin7=",eMin7)
#print("Umin7=",Umin7)

X2d_7 = np.real(Xt@Umin7)

fileName = sys.argv[3]
np.savetxt(fileName, X2d_7, delimiter=',')