
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
    ##LogisticRegression
    ###############
    print "LogisticRegression"
    time3=time.time()
    clf = linear_model.LogisticRegression()
    clf.fit (train_fs,train_lab)
    time4=time.time()
    print "training time", time4-time3
    
    j = 0
    ret = []
    rea = []
    while(j < 5):
        #default training
        i = 0
        t = 0
        accuracy = 0
        while(i < 3000):
            time4=time.time()
            tmp = clf.score(test_fs,test_lab)
            time5=time.time()
            accuracy += tmp
            ret.append(time5-time4)
            rea.append(tmp)
            t += (time5-time4)
            i+=1
        print "average test time", t/i
        print "average accuracy", accuracy/i
        accuracy1 = clf.score(train_fs,train_lab)
        print "self", accuracy1
        time.sleep(20)
        j+=1
    res = np.vstack((ret,rea))
    pickle.dump(res, open('LogisticRegression.p','wb+'))
    time.sleep(30)


    ###############
    ##SVM
    ###############
    print "SVM_linear"
    time3=time.time()
    clf = svm.SVC(kernel = 'linear')
    clf.fit (train_fs,train_lab)
    time4=time.time()
    print "training time", time4-time3
    
    j = 0
    ret = []
    rea = []
    while(j < 5):
        #default training
        i = 0
        t = 0
        accuracy = 0
        while(i < 50):
            time4=time.time()
            tmp = clf.score(test_fs,test_lab)
            time5=time.time()
            accuracy += tmp
            ret.append(time5-time4)
            rea.append(tmp)
            t += (time5-time4)
            i+=1
        print "average test time", t/i
        print "average accuracy", accuracy/i
        accuracy1 = clf.score(train_fs,train_lab)
        print "self", accuracy1
        time.sleep(20)
        j+=1
    res = np.vstack((ret,rea))
    pickle.dump(res, open('SVM.p','wb+'))
    time.sleep(30)

    ###############
    ##SVM
    ###############
    print "SVM_rbf"
    time3=time.time()
    #default kernel rbf
    clf = svm.SVC()
    clf.fit (train_fs,train_lab)
    time4=time.time()
    print "training time", time4-time3
    
    j = 0
    ret = []
    rea = []
    while(j < 5):
        #default training
        i = 0
        t = 0
        accuracy = 0
        while(i < 50):
            time4=time.time()
            tmp = clf.score(test_fs,test_lab)
            time5=time.time()
            accuracy += tmp
            ret.append(time5-time4)
            rea.append(tmp)
            t += (time5-time4)
            i+=1
        print "average test time", t/i
        print "average accuracy", accuracy/i
        accuracy1 = clf.score(train_fs,train_lab)
        print "self", accuracy1
        time.sleep(20)
        j+=1
    res = np.vstack((ret,rea))
    pickle.dump(res, open('SVM_rbf.p','wb+'))
    time.sleep(30)


    ###############
    ##LinearSVM
    ###############
    print "LinearSVM"
    time3=time.time()
    clf = svm.LinearSVC()
    clf.fit (train_fs,train_lab)
    time4=time.time()
    print "training time", time4-time3
    
    j = 0
    ret = []
    rea = []
    while(j < 5):
        #default training
        i = 0
        t = 0
        accuracy = 0
        while(i < 3000):
            time4=time.time()
            tmp = clf.score(test_fs,test_lab)
            time5=time.time()
            accuracy += tmp
            ret.append(time5-time4)
            rea.append(tmp)
            t += (time5-time4)
            i+=1
        print "average test time", t/i
        print "average accuracy", accuracy/i
        accuracy1 = clf.score(train_fs,train_lab)
        print "self", accuracy1
        time.sleep(20)
        j+=1
    res = np.vstack((ret,rea))
    pickle.dump(res, open('LinearSVM.p','wb+'))
    time.sleep(30)


    # ##############
    # #Random Forest
    # ##############

    # # clf = RandomForestClassifier(n_estimators=10, max_depth=None,
    # # min_samples_split=1, random_state=0)
    # # scores = cross_val_score(clf, X, y)
    # # scores.mean()   

    # print "RandomForest"
    # time3=time.time()
    # clf = ensemble.RandomForestClassifier()
    # clf.fit (train_fs,train_lab)
    # time4=time.time()
    # print "training time", time4-time3
    
    # j = 0
    # ret = []
    # rea = []
    # while(j < 5):
    #     #default training
    #     i = 0
    #     t = 0
    #     accuracy = 0
    #     while(i < 300):
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
    #     time.sleep(120)
    #     j+=1
    # res = np.vstack((ret,rea))
    # pickle.dump(res, open('RandomForest.p','wb+'))
    # time.sleep(60)

    # ###############
    # ##KNeighborsClassifier
    # ###############

    # print "KNeighborsClassifier"
    # time3=time.time()
    # clf = neighbors.KNeighborsClassifier()
    # clf.fit (train_fs,train_lab)
    # time4=time.time()
    # print "training time", time4-time3
    
    # j = 0
    # ret = []
    # rea = []
    # while(j < 5):
    #     #default training
    #     i = 0
    #     t = 0
    #     accuracy = 0
    #     while(i < 30):
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
    #     time.sleep(120)
    #     j+=1
    # res = np.vstack((ret,rea))
    # pickle.dump(res, open('KNeighborsClassifier.p','wb+'))
    # time.sleep(60)


    # ###############
    # ##GaussianNB
    # ###############

    # print "GaussianNB"
    # time3=time.time()
    # clf = naive_bayes.GaussianNB()
    # clf.fit (train_fs,train_lab)
    # time4=time.time()
    # print "training time", time4-time3
    
    # j = 0
    # ret = []
    # rea = []
    # while(j < 5):
    #     #default training
    #     i = 0
    #     t = 0
    #     accuracy = 0
    #     while(i < 300):
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
    #     time.sleep(120)
    #     j+=1
    # res = np.vstack((ret,rea))
    # pickle.dump(res, open('GaussianNB.p','wb+'))
    # time.sleep(60)




if __name__ == '__main__':
    logReg()
