import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from imblearn.under_sampling import NearMiss


if __name__ == '__main__':
	df = pd.read_csv('creditcard.csv')

	Y_data = []
	Y_data = df['Class'].tolist()
	df = df.drop('Class',axis=1)
	df = (df - df.mean()) / ((df.max() - df.min()))
	X_data = df.as_matrix()

	X_train, X_test, y_train, y_test = train_test_split(X_data,
		Y_data,
		test_size=0.1,
		random_state=0)

	X_train, X_val, y_train, y_val = train_test_split(X_train,
		y_train,
		test_size=0.1,
		random_state=0)

	#Oversampling (SMOTE)
	sm = SMOTE()
	X_smote, y_smote = sm.fit_sample(X_train, y_train)

	#Undersampling (Distance-based Near Miss 1,2,3)
	nm1 = NearMiss(version = 1)
	X_miss1, y_miss1 = nm1.fit_sample(X_train, y_train)
	nm2 = NearMiss(version = 2)
	X_miss2, y_miss2 = nm2.fit_sample(X_train, y_train)
	nm3 = NearMiss(version = 3)
	X_miss3, y_miss3 = nm3.fit_sample(X_train, y_train)

	#Undersampling (EasyEnsemble)
	ee = EasyEnsemble(n_subsets=30)
	X_resampled, y_resampled = ee.fit_sample(X_train, y_train)	


	print "Naive Bayes"
	naive_clf = GaussianNB()
	naive_clf.fit (X_train, y_train)
	y_pred = naive_clf.predict(X_test)
	print "initial: ",f1_score (y_test, y_pred)

	naive_clf.fit (X_smote, y_smote)
	y_pred = naive_clf.predict(X_test)
	print "smote: ",f1_score (y_test, y_pred)

	naive_clf.fit (X_miss1, y_miss1)
	y_pred = naive_clf.predict(X_test)
	print "near miss-1: ",f1_score (y_test, y_pred)
	naive_clf.fit (X_miss2, y_miss2)
	y_pred = naive_clf.predict(X_test)
	print "near miss-2: ",f1_score (y_test, y_pred)
	naive_clf.fit (X_miss3, y_miss3)
	y_pred = naive_clf.predict(X_test)
	print "near miss-3: ",f1_score (y_test, y_pred)

	NBclassifiers = []		
	for i in range(0,10,1):
		NBclassifiers.append(GaussianNB().fit(X_resampled[i], y_resampled[i]))
	
	y_pred = np.asarray([clf.predict(X_test) for clf in NBclassifiers]).T
	y_pred = np.apply_along_axis(lambda x:
		np.argmax(np.bincount(x)),
		axis=1,
		arr=y_pred.astype('int'))
	print "easy ensemle: ",f1_score (y_test, y_pred)



	print "Random Forest"
	forest_clf = RandomForestClassifier(n_estimators=50,
				max_depth=10, 
				random_state=0)
	forest_clf.fit(X_train, y_train)
	y_pred = forest_clf.predict(X_test)
	print "initial: ", f1_score (y_test, y_pred)

	forest_clf.fit (X_smote, y_smote)
	y_pred = forest_clf.predict(X_test)
	print "smote: ",f1_score (y_test, y_pred)

	forest_clf.fit (X_miss1, y_miss1)
	y_pred = forest_clf.predict(X_test)
	print "near miss-1: ",f1_score (y_test, y_pred)
	forest_clf.fit (X_miss2, y_miss2)
	y_pred = forest_clf.predict(X_test)
	print "near miss-2: ",f1_score (y_test, y_pred)
	forest_clf.fit (X_miss3, y_miss3)
	y_pred = forest_clf.predict(X_test)
	print "near miss-3: ",f1_score (y_test, y_pred)

	forests = []
	for i in range(0,10,1):
		forests.append(RandomForestClassifier(n_estimators=20, max_depth=5,
		random_state=0).fit(X_resampled[i], y_resampled[i]))

	y_pred = np.asarray([clf.predict(X_test) for clf in forests]).T
	y_pred = np.apply_along_axis(lambda x:
		np.argmax(np.bincount(x)),
		axis=1,
		arr=y_pred.astype('int'))
	print "easy ensemle: ",f1_score (y_test, y_pred)



	print "SVM"
	svc_clf = LinearSVC(random_state=0)
	svc_clf.fit(X_train,y_train)
	y_pred = svc_clf.predict(X_test)
	print "initial: ",f1_score (y_test, y_pred)

	svc_clf.fit (X_smote, y_smote)
	y_pred = svc_clf.predict(X_test)
	print "smote: ",f1_score (y_test, y_pred)

	svc_clf.fit (X_miss1, y_miss1)
	y_pred = svc_clf.predict(X_test)
	print "near miss-1: ",f1_score (y_test, y_pred)
	svc_clf.fit (X_miss2, y_miss2)
	y_pred = svc_clf.predict(X_test)
	print "near miss-2: ",f1_score (y_test, y_pred)
	svc_clf.fit (X_miss3, y_miss3)
	y_pred = svc_clf.predict(X_test)
	print "near miss-3: ",f1_score (y_test, y_pred)

	svms = []
	for i in range(0,10,1):
		svms.append(LinearSVC(random_state=0).fit(X_resampled[i], y_resampled[i]))
	
	y_pred = np.asarray([clf.predict(X_test) for clf in svms]).T
	y_pred = np.apply_along_axis(lambda x:
		np.argmax(np.bincount(x)),
		axis=1,
		arr=y_pred.astype('int'))
	print "easy ensemle: ",f1_score (y_test, y_pred)
