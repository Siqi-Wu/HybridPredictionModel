
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


    LR = linear_model.LogisticRegression()
    SVC = svm.SVC(kernel = 'rbf')
    LR.fit (train_fs,train_lab)
    SVC.fit (train_fs,train_lab)
    partial_fs = test_fs[:10,:]
    partial_lab = test_lab[:10,]


    test_fs_100 = test_fs
    test_lab_100 = test_lab
    for i in range(99):
        test_fs_100 = np.vstack((test_fs_100[:,:],test_fs[:,:]))
        test_lab_100 = np.concatenate((test_lab_100[:,],test_lab[:,]),axis = 0)

    print test_lab_100.shape, test_fs_100.shape

    # time2 = time.time()
    # d = LR.score(test_fs_100,test_lab_100)
    # time3 = time.time()
    # print time3-time2, d
    # time2 = time.time()
    # d = LR.score(test_fs_100,test_lab_100)
    # time3 = time.time()
    # print time3-time2, d
    # j,accuracy,t,ret,rea = 0,0,0,[],[]
    # while j<10:
    #     time4 = time.time()
    #     acc_tmp = LR.score(test_fs_100,test_lab_100)
    #     time5 = time.time()
    #     time_tmp = time5 - time4
    #     print time_tmp
    #     accuracy += acc_tmp
    #     t += time_tmp
    #     ret.append(time_tmp)
    #     rea.append(acc_tmp)
    #     j +=1
    #     time.sleep(5)
    
    # print "average test time", t/j
    # print "average accuracy", accuracy/j
    # time2 = time.time()
    # d = LR.score(test_fs_100,test_lab_100)
    # time3 = time.time()
    # print time3-time2, d

    time2 = time.time()
    b = LR.predict_proba(test_fs_100)
    final_res = LR.predict(test_fs_100)
    time3 = time.time()
    c = abs(b[:,0]-0.5)
    time4 = time.time()
    d = np.argpartition(c,95100)
    time5 = time.time()
    svm_fs = test_fs_100[d[:95100]]
    time6 = time.time()
    svm_res = SVC.predict(svm_fs)
    time7 = time.time()
    #final_res = int(b[:,0]<0.5)
    final_res[d[:95100]] = svm_res
    time8 = time.time()
    print time3-time2,time4-time3,time5-time4,time6-time5,time7-time6,svm_fs.shape,svm_res.shape
    print time8-time2,time3-time2+time7-time6

    count = 0
    for i,j in zip(list(test_lab_100[:,]),final_res):
        if i == j:
            count += 1

    #print result
    print "acc",count*1.0/len(final_res)

    d = SVC.score(test_fs_100,test_lab_100)
    time3 = time.time()
    print time3-time2, d
    time2 = time.time()
    d = LR.score(test_fs_100,test_lab_100)
    time3 = time.time()
    print time3-time2, d
    time2 = time.time()
    d = LR.score(test_fs_100,test_lab_100)
    time3 = time.time()
    print time3-time2, d

    print "2"
    time2 = time.time()
    b = LR.predict_proba(test_fs_100)
    final_res = LR.predict(test_fs_100)
    time3 = time.time()
    c = abs(b[:,0]-0.5)
    time4 = time.time()
    d = np.argpartition(c,190100)
    time5 = time.time()
    svm_fs = test_fs_100[d[:190100]]
    time6 = time.time()
    svm_res = SVC.predict(svm_fs)
    time7 = time.time()
    #final_res = int(b[:,0]<0.5)
    final_res[d[:190100]] = svm_res
    time8 = time.time()
    print time3-time2,time4-time3,time5-time4,time6-time5,time7-time6,svm_fs.shape,svm_res.shape
    print time8-time2,time3-time2+time7-time6

    count = 0
    for i,j in zip(list(test_lab_100[:,]),final_res):
        if i == j:
            count += 1

    #print result
    print "acc",count*1.0/len(final_res)

    d = SVC.score(test_fs_100,test_lab_100)
    time3 = time.time()
    print time3-time2, d
    time2 = time.time()
    d = LR.score(test_fs_100,test_lab_100)
    time3 = time.time()
    print time3-time2, d
    time2 = time.time()
    d = LR.score(test_fs_100,test_lab_100)
    time3 = time.time()
    print time3-time2, d

if __name__ == '__main__':
    logReg()
