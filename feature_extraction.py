import argparse
import numpy as np
import matplotlib.pyplot as pl

import collections
import  math
import copy
import itertools
import operator

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from numpy import linalg
import pylab

from sklearn import linear_model, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from sklearn import cross_validation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import metrics


def pfa():
    prc = 0.0
    max_prc = 0.0
    feature_to_add = 1
    trees = 50

    out = open('log.txt','w')
    train_data = np.loadtxt('spam.train.txt')
    test_data  = np.loadtxt('spam.test.txt')

    total_data = train_data[0:7000,1::]
    x_mean = np.array(np.mean(total_data,axis=0))

    n,m = total_data.shape
    R = np.cov(total_data.T)
    print R.shape

    lbd,vectors = linalg.eig(R)
    lbd_sum = sum(np.abs(i) for i in lbd)

    summ = 0.0
    best_dimm = 0

    for i in xrange(len(lbd)):
        summ += np.abs(lbd[i])
        best_dimm += 1
        if summ/lbd_sum > 0.99:
	       break

    print 'best dimm:' + str(best_dimm)

    B = vectors[:,0:best_dimm]
    C = np.array(B.real)
    m,n = B.shape

    kmeans = KMeans(init='random', n_clusters=best_dimm, n_init=10,n_jobs=4)
    kmeans.fit(C)
   
    features = []
    vector = 0
    min_dst = 1e5
    for items in kmeans.cluster_centers_ :
        for i in xrange(0,m):
            if dist(items,B[i,:].real) < min_dst:
                min_dst = dist(items,B[i,:])
                vector = i
        min_dst = 1e5
        features.append(vector)
    if features.count(0) > 0:
        features.remove(0) #Remove answer from set of features

    forest = RandomForestClassifier(n_estimators = trees,n_jobs=4)
    logreg = linear_model.LogisticRegression(C=1e5)

    selected_forest_features = [features[1],features[2]]
    selected_logreg_features = []
    num_forest_features = []
    num_logreg_features = []
    best_forest_perf = []
    best_logreg_perf = []

    for j in xrange(0,len(features)):
        max_forest_prc = 0.0
        max_logreg_prc = 0.0

        max_forest_prc_index = 0
        max_logreg_prc_index = 0

        for i in xrange(0,len(features)):
            if selected_forest_features.count(features[i]) == 0:
                selected_forest_features.append(features[i])
                train_data_ = train_data[0::,selected_forest_features]
                test_data_ = test_data[0::,selected_forest_features]
                selected_forest_features.pop()
            
                forest_prc = 0.0
                forest = forest.fit(train_data_, train_data[0::, 0])
                forest_prc = forest.score(test_data_,test_data[:,0])
                if forest_prc > max_forest_prc :
                    max_forest_prc = forest_prc
                    max_forest_prc_index = i

            if selected_logreg_features.count(features[i]) == 0:
                selected_logreg_features.append(features[i])
                train_data_ = train_data[0::,selected_logreg_features]
                test_data_ = test_data[0::,selected_logreg_features]
                selected_logreg_features.pop()

                logreg_prc = 0.0
                logreg = logreg.fit(train_data_,train_data[:,0])
                logreg_prc = logreg.score(test_data_,test_data[:,0])
                if logreg_prc > max_logreg_prc :
                    max_logreg_prc = logreg_prc
                    max_logreg_prc_index = i

        if selected_forest_features.count(features[max_forest_prc_index]) == 0:
            selected_forest_features.append(features[max_forest_prc_index])
            num_forest_features.append(len(selected_forest_features))
            best_forest_perf.append(max_forest_prc)
        if selected_logreg_features.count(features[max_logreg_prc_index]) == 0:
            selected_logreg_features.append(features[max_logreg_prc_index])
            num_logreg_features.append(len(selected_logreg_features))  
            best_logreg_perf.append(max_logreg_prc)

    print best_forest_perf
    print num_forest_features
    print best_logreg_perf
    print num_logreg_features
    pl.plot(num_forest_features,best_forest_perf)
    pl.show()

    pl.plot(num_logreg_features,best_logreg_perf)
    pl.show()

def cfs_method():
    trees = 50
    out = open('log.txt','w')
    train_data = np.loadtxt('spam.train.txt')
    test_data  = np.loadtxt('spam.test.txt')

    total_data = train_data[0:7000,1::]
    num_features = len(train_data[0,1::])

    n,m = total_data.shape
    C = np.corrcoef(total_data.T)
    selected_features = []
    max_cfs_perf = []
    forest_prc = []
    logreg_prc = []
    
    forest = RandomForestClassifier(n_estimators = trees,n_jobs=4)
    logreg = linear_model.LogisticRegression(C=1e5)

    print num_features

    for i in xrange(1,num_features):
        max_cfs = 0.0
        max_cfs_index = 1
        for j in xrange(1,num_features):
            if selected_features.count(i) > 0:
                continue
            selected_features.append(i)
            prc = cfs(selected_features,C)
            selected_features.pop()
            if prc > max_cfs:
                max_cfs = prc
                max_cfs_index = i
        if selected_features.count(max_cfs_index) == 0:
            selected_features.append(max_cfs_index)
            max_cfs_perf.append(max_cfs)

            train_data_ = train_data[0::,selected_features]
            test_data_ = test_data[0::,selected_features]

            forest = forest.fit(train_data_, train_data[0::, 0])
            forest_prc.append(forest.score(test_data_,test_data[:,0]))   
        
            logreg = logreg.fit(train_data_,train_data[:,0])
            logreg_prc.append(logreg.score(test_data_,test_data[:,0]))   
        
    pl.plot(xrange(0,len(max_cfs_perf)),max_cfs_perf,color='red')
    pl.show()
    pl.plot(xrange(0,len(forest_prc)),forest_prc,color='blue')
    pl.show()
    pl.plot(xrange(0,len(logreg_prc)),logreg_prc,color='green')
    pl.show()
    pl.plot(xrange(0,len(max_cfs_perf)),max_cfs_perf,'r',xrange(0,len(forest_prc)),forest_prc,'b',xrange(0,len(logreg_prc)),logreg_prc,'g')
    pl.show()

def mRMR():
    out = open('log.txt','w')
    eps = 0.0000001
    train_data = np.loadtxt('machine.data',delimiter=',',usecols=[i for i in xrange(2,10)])
    total_data = train_data[:,:-1]
    num_features = len(total_data[0,:])
    mean = np.mean(total_data,axis=0)
    variance = np.var(total_data,axis=0)
    total_data = (total_data-mean)/np.sqrt(variance)
    P = np.zeros(total_data.shape)
    for i in xrange(0,num_features-1):
        for j in xrange(0,num_features-1):
            P[i][j] = p_x_gauss(total_data[i][j],0.05,total_data,j)
  
    I = np.zeros((num_features,num_features))
    for i in xrange(0,num_features-1):
        for j in xrange(0,num_features-1):
            for k in xrange(0,len(total_data[:,i])):
                for l in xrange(0,len(total_data[:,j])):
                    p_x_ = P[k][i]
                    p_y_ = P[l][j]
                    p_x_y_ = p_x_y_gauss(total_data[k][i],total_data[j][j],0.05,total_data,i,j)
                    if abs(p_x_y_ ) > eps and abs(p_x_) > np.sqrt(eps) and abs(p_y_)> np.sqrt(eps):
                        I[i][j] += p_x_y_*( np.log(p_x_y_/(p_x_ * p_y_)))
    
  
    

def main():
    pfa()
    cfs_method()
    №mRMR()

def cfs(subset,C):
    answ = 0.0
    answ = sum(C[0,subset])
    fract = 0.0
    for i in subset:
        for j in subset:
            fract += 2*C[i,j]
    fract += len(subset)
    fract = np.sqrt(fract)
    return float(answ)/fract

def p_x(x,h,dataset,num):
    ans = 0.0
    ans = 1.0/len(dataset[:,num])*(1.0/h)*0.5*sum(int(np.absolute(x - i) < h) for i in dataset[:,num])
    return ans

def p_x_y(x,y,h,dataset,num_x,num_y):
    ans = 0.0
    ans = 1.0/len(dataset[:,num_x])*(1.0/h**2)*0.25*sum(int(np.absolute(x - dataset[i,num_x]) < h)*\
                                                      int(np.absolute(y - dataset[i,num_y]) < h)\
                                                      for i in xrange(0,len(dataset[:,num_x])))
    return ans

def p_x_gauss(x,h,dataset,num):
    ans = 0.0
    ans = 1.0/len(dataset[:,num])*(1.0/h)*sum(gauss_1d_func((x-i)/h,0.0,1.0) for i in dataset[:,num])
    return ans

def p_x_y_gauss(x,y,h,dataset,num_x,num_y):
    ans = 0.0
    ans = 1.0/len(dataset[:,num_x])*(1.0/h)*sum(gauss_1d_func((x-dataset[i,num_x])/h,0.0,1.0)*\
                                              gauss_1d_func((y-dataset[i,num_y])/h,0.0,1.0)\
                                              for i in xrange(0,len(dataset[:,num_x])))
    return ans

def gauss_func(x,means,cov):
    ans = 1.0/(np.sqrt(2*(np.pi**len(means)*np.absolute(np.linalg.det(cov))))) * np.exp(-1.0/2.0*np.dot(np.dot(np.array((np.array(x) - means)),np.linalg.inv(cov)),np.array((np.array(x) - means))))
    return ans

def gauss_1d_func(x,mean,var):
    return 1.0/np.sqrt(2*np.pi*var)*np.exp(-0.5 * (x - mean)**2 / var)


def dist(x,y):
    distance = sum( (x[i]-y[i])**2 for i in xrange(0,len(x)))
    return np.sqrt(distance)

if __name__ == "__main__":
    main()
