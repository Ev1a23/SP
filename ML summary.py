import numpy as np
# General:
# np.random.rand(N)*100 # create N random values 0<=val<100
# np.random.normal(loc=mu, scale=std, size=size)

# scikit sklearn:
# choose model->select relevant params->creeate feature mat + target vec->fit->predict
from sklearn import datasets #datasets has some prepared datasets
features_mat=np.array([[22,45,30],[13,17,60],[1,23,42],[90,20,3],[30,5,46],[30,35,34]])
target_vec=np.array([10,20,30,40,15,90])
# Regression:
from sklearn import linear_model # for regressions
model = linear_model.LinearRegression() #choose model, no relevant params
model.fit(features_mat, target_vec) # fit
model.score(features_mat, target_vec) # value between 0 and 1, here will be 1, works only after fitting
model.predict(np.array([[22,43,30], [13,12,11]])) # get target values based on fitted model


# Classification: when data is classified and we want to predict class of given inputs
from sklearn.model_selection import train_test_split # used for many ML methods
# select data for train, data for test, target for train and target for test
feat_train, feat_test, tar_train, tar_test=train_test_split(features_mat, target_vec, test_size=0.25, random_state=0)
from sklearn.neighbors import KNeighborsClassifier # choose model
model=KNeighborsClassifier(3) # select relevant params (classify by 3 closet neighbors)
model.fit(feat_train, tar_train) # fit
model.predict(feat_test) # get target values based on fitted model
np.mean(model.predict(feat_test)==tar_test) # Accuracy, how TN and TP we got
import sklearn.metrics as metrics
metrics.accuracy_score(model.predict(feat_test), tar_test) #same as before only builtin
print(metrics.classification_report(model.predict(feat_test),tar_test)) #full report including recall and precision
#recall=TP/(TP+FP)      precision=TP/(TP+FN)
# If we want to test same data multiple times:
from sklearn.model_selection import cross_val_score
#scores=cross_val_score(model,features_mat, target_vec, cv=5) # cv is train/test split strategy

# Clustering - when data is not classified - we want to classify it and predict class for new data
from sklearn.cluster import KMeans #choose model
model=KMeans(n_clusters=3, random_state=0) # select relevant params
model.fit(features_mat) # now fit takes only data, no target because data is not classified
model.labels_ # label for each data vector
model.predict([[10,30,40]]) # predict the class of [10,30,40]

# Dimensionality Reduction (PCA) - goal is to reduce dim and keep distance between vectors
#                                   in order to ease future proccessing
from sklearn.preprocessing import StandardScaler # this is for normalize
features_mat=StandardScaler().fit_transform(features_mat) # normalizing
from sklearn.decomposition import PCA #choose model
pca_dim= PCA(n_components=2) # select relevant params
pca_features=pca_dim.fit_transform(features_mat) # fitting



