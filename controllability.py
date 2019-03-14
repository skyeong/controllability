from scipy.linalg import svd,schur
from numpy.matlib import repmat 
import numpy as np
import networkx as nx
from utils import read_csv

def modal_control(A):
# This method returns values of MODAL CONTROLLABILITY for each node 
# in a network, given the adjacency matrix for that network. Modal
# controllability indicates the ability of that node to steer the
# system into difficult-to-reach states, given input at that node.
#
# INPUT:
#     A is the structural (NOT FUNCTIONAL) network adjacency matrix,
# 	such that the simple linear model of dynamics outlined in the
# 	reference is an accurate estimate of brain state fluctuations.
# 	Assumes all values in the matrix are positive, and that the
# 	matrix is symmetric.
#
# OUTPUT:
#     Vector of modal controllability values for each node
#
# Reference: Gu, Pasqualetti, Cieslak, Telesford, Yu, Kahn, Medaglia,
#            Vettel, Miller, Grafton & Bassett, Nature Communications
#            6:8414, 2015. 
    u, s, vt = svd(A)      # singluar value decomposition
    A = A/(1+s[0])         # s is a eigen-value
    T, U = schur(A,'real') # Schur stability
    eigVals = np.diag(T)
    N = A.shape[0]
    phi = np.zeros(N,dtype=float)
    for i in range(N):
        A_left = U[i,]*U[i,] # element-wise multiplication
        A_right = (1.0-np.power(eigVals,2)).transpose()
        phi[i] = np.matmul(A_left, A_right)
    return phi


def ave_control(A):
# This program is a Python version of average controllability
#     This function returns values of AVERAGE CONTROLLABILITY for 
#     each node in a network, given the adjacency matrix for that network. 
#     Average controllability measures the ease by which input at 
#     that node can steer the system into many easily-reachable states.
#
# INPUT:
#     A is the structural (NOT FUNCTIONAL) network adjacency matrix,
#     such that the simple linear model of dynamics outlined in the%	      reference is an accurate estimate of brain state fluctuations.
#     Assumes all values in the matrix are positive, and that the
#     matrix is symmetric.
#
# OUTPUT:
#     Vector of average controllability values for each node
#
#     Reference: Gu, Pasqualetti, Cieslak, Telesford, Yu, Kahn, Medaglia,
#             Vettel, Miller, Grafton & Bassett, Nature Communications
#             6:8414, 2015.
    #%%
    N = A.shape[0]
    u, s, vt = svd(A)      # singluar value decomposition
    A = A/(1+s[0])         # s is a eigen-value
    T, U = schur(A,'real') # Schur stability
    midMat = np.multiply(U,U).transpose()           # element-wise multiplication
    v = np.matrix(np.diag(T)).transpose()
    
    #%%
    P = np.diag(1-np.matmul(v,v.transpose()))
    P = repmat(P.reshape([82,1]), 1, N)
    values = sum(np.divide(midMat,P))
    return values


if __name__ == "__main__":
    # Load sample DTI data
    test_filename = '/Users/skyeong/pythonwork/controllability/data/testgraph.edgelist'
    G=nx.read_edgelist(test_filename)
    nodelist = [str(i+1) for i in range(82)]
    A=nx.to_numpy_matrix(G,nodelist=nodelist)
    A[A>0]=1
    # A = np.array([[0, 1, 0, 1, 1, 1],[0, 0, 0, 1, 0, 0],[1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0]])
    mm = modal_control(A)
    aa = ave_control(A)
    print(mm)
    print(aa)
    