import pandas as pd
import numpy as np

from keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

def read_adult():
	"""
	"""
	
	df = pd.read_csv('data/adult.data',
		header=None)

	test_df = pd.read_csv('data/adult.test',
		skiprows=1,
		header=None)

	test_df[14] = test_df[14].map(lambda x: x[:-1])
	df = df[df[1]!=' ?']
	test_df = test_df[test_df[1]!=' ?']

	categorical_vars = [1,3,5,6,7,8,9,13,14]
	numeric_vars = [var for var in df.columns if var not in categorical_vars]

	for i in categorical_vars:
		cat2id = dict((k,i) for i,k in enumerate(df[i].unique()))
		df[i] = df[i].map(lambda x: cat2id[x])
		test_df[i] = test_df[i].map(lambda x: cat2id[x])
		#df[i] = df[i].map(lambda x: to_categorical(x, num_classes=len(cat2id)))

	y_train = df[14].as_matrix()
	y_test = test_df[14].as_matrix()


	df = df.drop([14], axis=1)
	df = (df - df.mean()) / (df.max() - df.min())
	test_df = test_df.drop([14], axis=1)
	test_df = (test_df - df.mean()) / (df.max() - df.min())


	X_train = df.as_matrix()
	X_test = test_df.as_matrix()


	return X_train, y_train, X_test, y_test

if __name__ == '__main__':

	X_train, y_train, X_test, y_test = read_adult()
	X_train, X_val, y_train, y_val = train_test_split(X_train,
		y_train,
		test_size=0.2,
		random_state=0)

	clf = DecisionTreeClassifier(random_state=0)
	clf.fit(X_train, y_train)
	print 'Single:',clf.score(X_val,y_val)

	bagg_clf = BaggingClassifier(clf,
		max_samples=0.7,
		max_features=1.0,
		random_state=0)
	bagg_clf.fit(X_train, y_train)
	print 'Bagging:',bagg_clf.score(X_val,y_val)

	ada_clf = AdaBoostClassifier(clf,
		n_estimators=50,
		learning_rate=0.5,
		random_state=0)
	ada_clf.fit(X_train, y_train)
	print 'AdaBoost:',ada_clf.score(X_val,y_val)

	gb_clf = GradientBoostingClassifier(n_estimators=100,
		max_depth=2,
		learning_rate=0.5,
		random_state=0)
	gb_clf.fit(X_train, y_train)
	print 'Gradient Boosting:',gb_clf.score(X_val,y_val)

	forest_clf = RandomForestClassifier(n_estimators=20,
		max_depth=5,
		random_state=0)
	forest_clf.fit(X_train, y_train)
	print 'Random Forest:',forest_clf.score(X_val,y_val)