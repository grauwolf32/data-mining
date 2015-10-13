import argparse
import numpy as np
import matplotlib.pyplot as pl

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from numpy import linalg
from random import randint
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

def rsm(data_len,feature_len,data_prc,feature_prc): 
    features = []
    sample   = []

    p_sample   = np.random.uniform(0,1,data_len)
    p_features = np.random.uniform(0,1,feature_len)
 
    for i in xrange(0,data_len-1):
        if p_sample[i] < data_prc:
            sample.append(i)
    for i in xrange(0,feature_len-1):
        if p_features[i] < feature_prc:
            features.append(i)
    return sample, features
    

def main():    
    num_of_models = 50
    n_clusters = 200
    n_periods = 35
    n_start_sample = 2000
    max_entropy = 0.0

    c = 1
    data_prc = 0.3
    feature_prc = 0.5

    out = open('log.txt','w')
    train_data = np.loadtxt('spam.train.txt')
    test_data_1  = np.loadtxt('spam.test.txt')
    
    m,n = train_data.shape

    total_data = []
    total_answ = []

    o = 0
    z = 0

    for i in xrange(0,m):
        if train_data[i,0] == 0 :
            if z <= n_start_sample/2 :
                total_data.append(train_data[i,1::])
                total_answ.append(train_data[i,0])
                z += 1
        if train_data[i,0] == 1 :
            if o <= n_start_sample/2 :
                total_data.append(train_data[i,1::])
                total_answ.append(train_data[i,0])
                o += 1

    print len(total_data)
    total_data = np.array(total_data)
    total_answ = np.array(total_answ)
    m,n = total_data.shape

    test_data = test_data_1[:,1::]
    
    logreg = []

    p1 = []
    p2 = []
    p3 = []
    p4 = []

    num_of_element = []

    selected_features = []
    selected_samples = []

    features = []
    samples  = []
    #### ---------Education of commitet on first n_start_sample elements--------####
    for i in xrange(0,num_of_models-1):
        logreg.append(linear_model.LogisticRegression(C=1e5))
        samples, features = rsm(m,n,data_prc,feature_prc)
        selected_samples.append(samples)
        selected_features.append(features)
        
        train_data_    = total_data[selected_samples[i],:]
        train_data_    = train_data_[:,selected_features[i]]

        train_answers_ = total_answ[selected_samples[i]]
        logreg[i].fit(train_data_,train_answers_)
    ####------------------------------------------------------------------------####   
    main_model = linear_model.LogisticRegression(C=1e5)

    total_data = train_data[:,1::] 
    total_answ = train_data[:,0] 

    m,n = total_data.shape

    main_train_data = []
    main_train_answ = []
    main_train_indexes = []

    models_answ  = []
    index_to_add = []

    index = 0

    for t in xrange(0,n_periods-1):
        index_to_add = []
        values = list(np.zeros(n_clusters))
        for i in xrange(0,n_clusters):
            index_to_add.append(0)
        for j in xrange(0,m-1):
            models_answ = []
            for k in xrange(0,num_of_models-1):
                res = int(logreg[k].predict(total_data[j,selected_features[k]]))
                models_answ.append(res)
            curr_entropy = vote_entropy(models_answ,num_of_models,2)
            if main_train_indexes.count(j) < c:
                add_max_n(index_to_add,values,j,curr_entropy,n_clusters)     
            

        main_train_indexes = main_train_indexes + index_to_add
        #main_train_indexes = list(set(main_train_indexes))
        main_train_data = total_data[main_train_indexes,:]
        main_train_answ = total_answ[main_train_indexes]
        main_model.fit(main_train_data,main_train_answ)

        predicted = main_model.predict(test_data)
        labels = unique_labels(test_data_1[:,0], predicted)
        p, r, f1, s = precision_recall_fscore_support(test_data_1[:,0],predicted,labels=labels,average=None)

        p1.append(p)
        p2.append(r)
        p3.append(f1)
        p4.append(s)

        num_of_element.append(len(main_train_indexes))

        for i in xrange(0,num_of_models-1):
            selected_samples[i] = selected_samples[i] + index_to_add
            #selected_samples[i] = list(set(selected_samples[i])) 
            train_data_    = total_data[selected_samples[i],:]
            train_data_    = train_data_[:,selected_features[i]]
            train_answers_ = total_answ[selected_samples[i]]
            logreg[i].fit(train_data_,train_answers_)

        all_set = list(xrange(0,m-1))

    p1_smpl = []
    p2_smpl = []
    p3_smpl = []
    p4_smpl = []

    sel_smpl = []
    smpl_mdl = linear_model.LogisticRegression(C=1e5)
    num_of_el_smpl = []
    n_elements = 50

    spam_data = []
    not_spam_data = []

    for i in xrange(0,m-1):
        if total_answ[i] == 1:
            spam_data.append(i)
        else:
            not_spam_data.append(i)

    for i in xrange(0,m-50,n_elements):
        for j in xrange(0,n_elements/2):
             l = len(spam_data)
             if l > 1:
                 t = randint(0,len(spam_data)-1)
                 sel_smpl.append(spam_data.pop(t))
             
             else:
                 l = len(not_spam_data) 
                 if l > 1:
                     t = randint(0,len(not_spam_data)-1)
                     sel_smpl.append(not_spam_data.pop(t))
                 
             l = len(not_spam_data) 
             if l > 1:
                t = randint(0,len(not_spam_data)-1)
                sel_smpl.append(not_spam_data.pop(t))
             else:
                l = len(spam_data)
                if l > 1:
                    t = randint(0,len(spam_data)-1)
                    sel_smpl.append(spam_data.pop(t))
    
        smpl_mdl.fit(total_data[sel_smpl,:],total_answ[sel_smpl])

        predicted = smpl_mdl.predict(test_data)
        labels = unique_labels(test_data_1[:,0], predicted)
        p, r, f1, s = precision_recall_fscore_support(test_data_1[:,0],predicted,labels=labels,average=None)
        num_of_el_smpl.append(len(sel_smpl))
        p1_smpl.append(p)
        p2_smpl.append(r)
        p3_smpl.append(f1)
        p4_smpl.append(s)  

    f1 = [(i[0] + i[1])/2.0 for i in p3]
    f1_smpl = [(i[0] + i[1])/2.0 for i in p3_smpl]

    pl.plot(num_of_element,f1,'r',num_of_el_smpl,f1_smpl,'g')
    pl.title('F1')
    pl.show()

 
def add_max_n(array,values,i,value,n):

    if values[n-1] >= value:
        return

    values.pop()
    array.pop()

    values.append(value)
    array.append(i)

    j = n-1
    temp = 0
    while j > 0:
        if values[j-1] <= values[j]:
            temp = values[j]
            values[j] = values[j-1]
            values[j-1] = temp

            temp = array[j]
            array[j] = array[j-1]
            array[j-1] = temp
            j = j-1
        else :
            break

def vote_entropy(answ,T,num_of_class):
    a = 0.0
    eps = 0.000000001
    for i in xrange(0,num_of_class-1):
        n = float(answ.count(i))/T + eps
        a += -1.0*(n*np.log(n))
    return a
        

if __name__ == "__main__":
    main()  
