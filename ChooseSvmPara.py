import cPickle as pickle
from sklearn import *
import time
import numpy as np

num_pos = 36499 
num_neg = 93565

def logReg():

    #time1=time.time()
    #read files
    filename = 'magic04'
    features = pickle.load(open(filename + '_features.p'))
    label = pickle.load(open(filename + '_labels.p'))
    print features.shape
    print label.shape
    #time2=time.time()
    #print "reading time", time2-time1

    #Normalization
    row, coln = features.shape
    for j in range(coln):
        features[:,j] = (features[:,j] - features[:,j].mean()) / features[:,j].std()
    
    num_total = row
    num_pos = sum(label)
    num_neg = num_total - num_pos
    
    a = num_pos/2
    b = num_neg/2
    train_fs = np.vstack((features[:a,:],features[-b:,:]))
    train_lab = np.concatenate((label[:a,],label[-b:,]),axis = 0)
    test_fs = features[a:-b,:]
    test_lab = label[a:-b,]
    print train_fs.shape,train_lab.shape,test_fs.shape,test_lab.shape,sum(test_lab),sum(train_lab)


    ###############
    ##SVM
    ###############

    C_range = np.logspace(8, 10, 3)
    gamma_range = np.logspace(-2, 3, 6)

    for C in C_range:
        for gamma in gamma_range:
            clf = svm.SVC(kernel = 'rbf', C= C, gamma = gamma)
            clf.fit(train_fs, train_lab)
            i,accuracy,t = 0,0,0
            while i < 10:
                time4 = time.time()
                tmp = clf.score(test_fs,test_lab)
                time5 = time.time()
                accuracy += tmp
                t += (time5-time4)
                i +=1
            print C,gamma,clf.n_support_,t,accuracy

    # j = 0
    # ret = []
    # rea = []
    # while(j < 5):
    #     #default training
    #     i = 0
    #     t = 0
    #     accuracy = 0
    #     while(i < 50):
    #         time4=time.time()
    #         tmp = clf.score(test_fs,test_lab)
    #         time5=time.time()
    #         accuracy += tmp
    #         ret.append(time5-time4)
    #         rea.append(tmp)
    #         t += (time5-time4)
    #         i+=1
    #     print "average test time", t/i
    #     print "average accuracy", accuracy/i
    #     accuracy1 = clf.score(train_fs,train_lab)
    #     print "self", accuracy1
    #     time.sleep(20)
    #     j+=1
    # res = np.vstack((ret,rea))
    # pickle.dump(res, open('SVM_rbf.p','wb+'))
    # time.sleep(30)

if __name__ == '__main__':
    logReg()