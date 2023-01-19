import numpy as np
# basic operations on ndarray:
x=np.array([[1,2],[3,4]])
y=np.array([ [ [1,2,3] , [4,5,6] ]  ,  [ [7,8,9] , [10,11,12] ] ])
np.shape(x) #(2,2)
np.shape(y) #(2,2,3)
np.sum(x) #1+2+3+4=10
np.sum(x, axis=1) #[3,7]
np.sum(y,axis=(1,2)) #[21,57]
np.sum(y, axis=1) #[[5,7,9],[17,19,21]]
x.dtype #default - int64 (for integers)
np.size(x) #4
np.ndim(y) #3
# additional operations: mean, sqrt, pow...

# different ways to create ndarray:
np.zeros((2,2,3)) #[[[0.,0.,0.],[0.,0.,0.]],[[0.,0.,0.],[0.,0.,0.]]] (floats)
np.ones((2,3)) #[[1.,1.,1.],[1.,1.,1.]] (floats)
np.full((2,2), 5) #[[5,5],[5,5]] (ints)
np.eye(3) #identity matrix of size 3
np.random.random((2,2)) #random vals
np.arange(1,11,3) #[1,4,7,10]

#optionally specify dtype on creation:
np.ones((2,3), dtype=np.int32) #[[1,1,1],[1,1,1]] (floats)


# indexing

# slicing:
#   - return a subarray, values are shared.
#   - slice each axis separately.
#   - does not change the rank.
# integer indexing:
#   - return a copy - vals are duplicated
#   - change the rank.
# mix of integer indexing and slicing:
#   - return a copy - vals are duplicated
#   - does not change the rank.
a=np.array([[1,2],[3,4]])
slice_a=a[:,:1] #slice_a=[[1],[3]]
index_a=a[[0,1],0] #index_a=[1,3]
index_slice_a=a[[0,1], :1] #index_slice_a=[[1],[3]]
index_a[1]=50 #doesn't effect a
index_slice_a[1,0]=50 #doesn't effect a
slice_a[0,0]=100 #causes a to be a=[[100,2],[3,4]]
a[0:2:2,::-1] #[[2,100]]
a[[0,0],[1,1]] #eq to [a[0,1],a[0,1] eq to [a[0][1],a[0][1]]. [2,2]

#int indexing may use to change the original array:
a=np.array([[1,2],[3,4]])
a[[0,1],[0]]+=5 #a=[[6,2],[8,4]]

# Filtering (boolean indexing)
a=np.array([[1,2],[3,4]])
x=a%2==0 #x=[[False,True],[False, True]]
y=a[x] #y=[2,4] #1 dimentional!
y=a[a>1] #[2,3,4]

# Array Math: add, substract, multiply, devide, sqrt, pow
# for vectors inner product and matrices multiply use dot function (@ operator):
v = np.array([9,10])
w = np.array([11, 12])
v.dot(w) #219
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
x@y #[[19,22],[43,50]]

x.T #transpose of x, works with matrices too:)

#can use array math between different #dim arrays:
print(x+v) #[[10 12] [12 14]]
z=np.array([5])
print(x+z) #[[6 7] [8 9]] works only because z has size one on every incompatible dimention


