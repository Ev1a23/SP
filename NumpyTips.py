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
print(np.linspace(0,1,5)) #[0.0,0.25,.0.5,0.75,1.]

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

#linear algebra: np.linalg common functions:
x = np.array([[1,2],[3,4]])
print(np.linalg.inv(x))
np.linalg.eigvals(x)

v=np.array([1,2,3,4])
print(np.tile(v,(3,1))) #[v,v,v]

print(np.reshape(v,(2,2)))


np.random.seed(593)
np.random.rand(3) #3 val between 0 and 1


#Data Sience:
from scipy.optimize import minimize_scalar
func= lambda x: 3*x**4-3*x+1
res=minimize_scalar(func)
print(res)


#Pandas:
import pandas as pd
X=pd.Series(["A1","B1"]) #indexes are default: 0,1,2,3...
Y=pd.Series(["A2","B2","abc"])
D=pd.DataFrame({"C1": X, "C2":Y}, index=['a','b']) #WRONG!! result is NaN instead of data.
print(D) #  C1  C2
        # a  NaN  NaN
        # b  NaN  NaN
D=pd.DataFrame({"C1": X, "C2":Y}) #RIGHT!!!
D.index=["","b",""]#RIGHT!!!
print(D) #  C1  C2
        # a  A  A2
        # b  B  B2


#df=pd.read_csv(filename) # read the csv into df



# Accessing data in pandas (reading and editing):
df=pd.DataFrame({"Name":["Or", "Evia"], "Last Name":["Shemesh", "Shemesh"], "City":["Hod Hasharon","Batsra"]}, index=["first", "second"])
# - accessing directly by col name and then specifing rows using slicing or a single row index:
# df[["col1_name","col2_name",...]]["row_index"]
# from a Series we can select multiple rows
# when col name contains no spaces can use: df.col_name[]
# when selecting only one row/col, can remove one pair of brackets
print(df.Name)
                # first       Or
                # second    Evia
print(df.Name[1:])
                # first     Or
                # second    Evia
print(df["Name"][["first", "second"]]) # df["Name"] is a Series! note the double brackets
                # first    Shemesh
                # second   Shemesh

print("\n\n\n\n")

print(df.shape) # (2,2)
print(df.describe()) # statistical data of the df
print(df.sample(1)) # sample 1 row of the df
print(df["Last Name"].unique()) # ['Shemesh']


#Bolean filtering, like numpy:
print(df["Name"]=="Or")
                # first      True
                # second    False
print(df[df["Name"]=="Or"]) #first   Or   Shemesh

# what if we want to select specific row and cols? use loc and iloc! row first, col second
#print(df[-2]) #doesn't work
print(df.iloc[-2])
                    # Name                   Or
                    # Last Name         Shemesh
                    # City         Hod Hasharon

print(df.iloc[:,0:2])
                    #         Name Last Name
                    # first     Or   Shemesh
                    # second  Evia   Shemesh

print(df.loc["first":"second","City"]) # slicing works because indices are ordered
                                       # note that the slicing takes the last boundary as well, unlike iloc

# Changing index:
df=df.set_index("Name") # now 'Name' is not a col anymore
print(df)
            #      Last Name          City  age
            # Name
            # Or     Shemesh  Hod Hasharon
            # Evia   Shemesh        Batsra

# Arithmetic works on Series (such as X*=2 etc)
# Adding new columns:
df["age"]=pd.Series([25,22], index=["Or", "Evia"]) # note that updated index values is critical
print(df)
            #      Last Name          City  age
            # Name
            # Or     Shemesh  Hod Hasharon   25
            # Evia   Shemesh        Batsra   22


df1=pd.DataFrame({"id1": [1,2,3,4,5,6], "id2":[10,20,30,40,40,30], "count":[100,200,300,400,500,600]})

print(df1)

df2=pd.DataFrame({"id1": [1,2,5], "id2":[10,20,30], "count":[10,20,30]})

df1["count"]+=pd.merge(df1, df2, how='left', on=['id1', 'id2']).fillna(0)["count_y"]
print(df1)


x=np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]])
print(np.apply_along_axis(func1d=lambda y:y.reshape(2,2),axis=1,arr=x)) # apply func1D to each row
a=np.array([[1,2],[3,4]]) #/
a[[0,1],[0]]+=5 #a=[[6,2],[8,4]]
D=pd.DataFrame({"C1": X, "C2":Y}, index=['a','b']) #/
#WRONG!! result is NaN instead of data.
#note that in loc the slicing takes the last boundary as well, unlike iloc
df1=pd.DataFrame({"id1": [1,2,3,4,5,6],
"id2":[10,20,30,40,40,30], "count":[100,200,300,400,500,600]}) #/
df2=pd.DataFrame({"id1": [1,2,5], "id2":[10,20,30], "count":[10,20,30]}) #/
df1["count"]+=pd.merge(df1, df2, how='left', on=['id1', 'id2']).fillna(0)["count_y"] # merge example
pca_features=pca_dim.fit_transform(features_mat) # fitting
#choose model->select relevant params->creeate feature mat + target vec->fit->predict
pos=np.ix_(i,i) # get a tuple ([[i0],[i1],[i2],....,[ik]], [[i0,i1,i2,...,ik]])
