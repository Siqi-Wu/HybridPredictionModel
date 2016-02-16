import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import time

def AverageStd():
	###
	# Order: LR, LinSVM, SVM, RF, KNN, NB, NN
	###
	time_ave, time_std, accuracy_ave, accuracy_std = [],[],[],[]

	filename = "LogisticRegression.p"
	X = pickle.load(open(filename))
	time_ave.append(np.mean(X[0,:]))
	time_std.append(np.std(X[0,:]))
	accuracy_ave.append(np.mean(X[1,:]))
	accuracy_std.append(np.std(X[1,:]))

	filename = "LinearSVM.p"
	X = pickle.load(open(filename))
	time_ave.append(np.mean(X[0,:]))
	time_std.append(np.std(X[0,:]))
	accuracy_ave.append(np.mean(X[1,:]))
	accuracy_std.append(np.std(X[1,:]))

	filename = "SVM.p"
	X = pickle.load(open(filename))
	time_ave.append(np.mean(X[0,:]))
	time_std.append(np.std(X[0,:]))
	accuracy_ave.append(np.mean(X[1,:]))
	accuracy_std.append(np.std(X[1,:]))
	
	filename = "SVM_rbf.p"
	X = pickle.load(open(filename))
	time_ave.append(np.mean(X[0,:]))
	time_std.append(np.std(X[0,:]))
	accuracy_ave.append(np.mean(X[1,:]))
	accuracy_std.append(np.std(X[1,:]))

	# filename = "RandomForest.p"
	# X = pickle.load(open(filename))
	# time_ave.append(np.mean(X[0,:]))
	# time_std.append(np.std(X[0,:]))
	# accuracy_ave.append(np.mean(X[1,:]))
	# accuracy_std.append(np.std(X[1,:]))

	# filename = "KNeighborsClassifier.p"
	# X = pickle.load(open(filename))
	# time_ave.append(np.mean(X[0,:]))
	# time_std.append(np.std(X[0,:]))
	# accuracy_ave.append(np.mean(X[1,:]))
	# accuracy_std.append(np.std(X[1,:]))	

	# filename = "GaussianNB.p"
	# X = pickle.load(open(filename))
	# time_ave.append(np.mean(X[0,:]))
	# time_std.append(np.std(X[0,:]))
	# accuracy_ave.append(np.mean(X[1,:]))
	# accuracy_std.append(np.std(X[1,:]))

	# filename = "NeutralNetwork.p"
	# X = pickle.load(open(filename))
	# time_ave.append(np.mean(X[0,:]))
	# time_std.append(np.std(X[0,:]))
	# accuracy_ave.append(1-np.mean(X[1,:]))
	# accuracy_std.append(np.std(X[1,:]))

	print time_std,time_ave,accuracy_ave,accuracy_std
	res = np.vstack((time_ave, time_std, accuracy_ave, accuracy_std))
	print res.shape
	pickle.dump(res, open('Average&Std.p','wb+'))

	
def Plot():

	filename = "Average&Std.p"
	X = pickle.load(open(filename))
	time_ave = X[0,:]
	time_std = X[1,:]
	accuracy_ave = X[2,:]
	accuracy_std = X[3,:]
	fig, ax1 = plt.subplots()

	ax2 = ax1.twinx()
	# s2 = np.sin(2*np.pi*t)
	# ax2.plot(t, s2, 'r.')
	# ax2.set_ylabel('sin', color='r')
	# for tl in ax2.get_yticklabels():
	#     tl.set_color('r')
	# plt.show()



	##Plot
	groups = 4
	index = np.arange(groups)
	bar_width = 0.35

	opacity = 0.4
	error_config = {'ecolor': '0.3'}
	

	rects1 = ax1.bar(index, time_ave, bar_width,
    	            alpha=opacity,
        	        color='b',
                 	yerr=time_std,
                 	error_kw=error_config,
                 	label='Time',
                 	log = True)

	rects2 = ax2.bar(index + bar_width, accuracy_ave, bar_width,
                 	alpha=opacity,
                 	color='r',
                 	yerr=accuracy_std,
                 	error_kw=error_config,
                 	label='Accuracy')

	ax1.set_xlabel('Different Algorithm')
	ax2.set_ylabel('Accuracy')
	ax1.set_ylabel('Time(s)')
	plt.title('MiniBooNE')
	plt.xticks(index + bar_width, ('LR', 'LinSVM', 'SVM', 'SVM_rbf'))
	#plt.xticks(index + bar_width, ('LR', 'LinSVM', 'SVM', 'RF', 'KNN','NB','NN'))
	ax1.legend(['Time'],loc = 'upper right', bbox_to_anchor=(0.9, 1.0))
	ax2.legend(['Accuracy'],loc = 'upper right', bbox_to_anchor=(0.9, 0.9))

	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
    #AverageStd()
    Plot()
