import numpy as np
A=np.array([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]]) # adjecency matrix
i=np.where(A[:,1]==1)[0] # get all neighbors of 1
print(i)
pos=np.ix_(i,i) # get a tuple ([[i0],[i1],[i2],....,[ik]], [[i0,i1,i2,...,ik]])
print(pos[0])
A[pos[0], pos[1]]=1 # create a bypass for each couple of neighbors of 1
A[:,1]=0
A[1,:]=0
print(A)


B=np.array([[1,2,3],[4,5,6]])
B[[[0],[1]],[1,2]]=37       
print(B)