#!/bin/env python

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle, resample


def cost_score(y_pred,y_true,costs=[0,5,1,0]):
	"""
	"""
	tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()

	cost_loss = (tn*costs[0] + fp*costs[1] + fn*costs[2] + tp*costs[3])
	
	return cost_loss


def read_data():
	"""
	"""

	df = pd.read_csv('data/heart.dat',
		header=None,
		delimiter=' ')

	y_data = df[13].as_matrix()

	df = df.drop(13,axis=1)
	df = (df - df.mean()) / (df.max() - df.min())

	X_data = df.as_matrix()

	X_train, X_test, y_train, y_test = train_test_split(X_data,
		y_data,
		test_size=0.1,
		random_state=0)

	X_train, X_val, y_train, y_val = train_test_split(X_train,
		y_train,
		test_size=0.1,
		random_state=0)

	return X_train, y_train, X_val, y_val, X_test, y_test


def default_scores(X_train, y_train, X_val, y_val):
	"""
	"""


	svc_model = LinearSVC(random_state=0).fit(X_train,y_train)

	y_pred = svc_model.predict(X_val)
	print 'SVC loss:',cost_score(y_pred,y_val)

	rf_model = RandomForestClassifier(random_state=0).fit(X_train,y_train)

	y_pred = rf_model.predict(X_val)
	print 'Random Forest loss:',cost_score(y_pred,y_val)

	nb_model = GaussianNB().fit(X_train,y_train)

	y_pred = nb_model.predict(X_val)
	print 'Naive Bayes loss:',cost_score(y_pred,y_val)

	return 


def class_weighting(X_train, y_train, X_val, y_val):
	"""
	"""

	svc_model = LinearSVC(random_state=0,
		class_weight={1:5.,2:1.}).fit(X_train,y_train)
	
	y_pred = svc_model.predict(X_val)
	print 'SVC with class weighting loss:',cost_score(y_pred,y_val)

	rf_model = RandomForestClassifier(random_state=0,
		class_weight={1:5.,2:1.}).fit(X_train,y_train)
	
	y_pred = rf_model.predict(X_val)
	print 'Random Forest with class weighting loss:',cost_score(y_pred,y_val)


	sample_weights = []
	for y in y_train:
		if y == 1:
			sample_weights.append(5)
		elif y == 2:
			sample_weights.append(1)

	nb_model = GaussianNB().fit(X_train,y_train,sample_weight=sample_weights)
	

	y_pred = nb_model.predict(X_val)
	print 'Naive Bayes with class weighting loss:',cost_score(y_pred,y_val)

	return 


def class_oversampling(X_train, y_train, X_val, y_val):
	"""
	"""

	positives = np.where( y_train == 1)
	X_positives = np.repeat(X_train[positives],4,axis=0)
	y_positives = np.repeat(y_train[positives],4)

	X_train_new = np.zeros(((X_train.shape[0]+X_positives.shape[0]),X_train.shape[1]))
	y_train_new = np.zeros(((y_train.shape[0]+y_positives.shape[0]),))

	X_train_new[:X_train.shape[0]] = X_train
	X_train_new[X_train.shape[0]:] = X_positives
	y_train_new[:y_train.shape[0]] = y_train
	y_train_new[y_train.shape[0]:] = y_positives

	X_train, y_train = shuffle(X_train_new, y_train_new, random_state=0)

	svc_model = LinearSVC(random_state=0).fit(X_train,y_train)

	y_pred = svc_model.predict(X_val)
	print 'SVC after oversampling loss:',cost_score(y_pred,y_val)

	rf_model = RandomForestClassifier(random_state=0).fit(X_train,y_train)

	y_pred = rf_model.predict(X_val)
	print 'Random Forest after oversampling loss:',cost_score(y_pred,y_val)


	nb_model = GaussianNB().fit(X_train,y_train)

	y_pred = nb_model.predict(X_val)
	print 'Naive Bayes after oversampling loss:',cost_score(y_pred,y_val)
	
	return 


def rejection_sampling(X_train,
	y_train,
	c=[5.,1.],
	zeta=5.,
	random_state=0):
	"""
	"""

	X_sample = []
	y_sample = []
	for x,y in zip(X_train,y_train):
		if y == 1:
			prob = c[0] / zeta
		elif y == 2:
			prob = c[1] / zeta

		sample_item = np.random.choice([True,False], p=[prob, 1-prob])

		if sample_item:
			X_sample.append(x)
			y_sample.append(y)

	return np.array(X_sample),np.array(y_sample)


def votting(clf_list,
	X_val):
	"""
	"""

	#For hard voting:
	pred = np.asarray([clf.predict(X_val) for clf in clf_list]).T
	pred = np.apply_along_axis(lambda x:
		np.argmax(np.bincount(x)),
		axis=1,
		arr=pred.astype('int'))

	return pred

def costing(X_train, y_train, X_val, y_val):
	"""
	"""

	svc_models = []
	rf_models = []
	nb_models = []
	for i in range(10):
		X_train_sample, y_train_sample = rejection_sampling(X_train, y_train, random_state=0)
		svc_models.append(LinearSVC(random_state=0).fit(X_train_sample,y_train_sample))
		rf_models.append(RandomForestClassifier(random_state=0).fit(X_train_sample,y_train_sample))
		nb_models.append(GaussianNB().fit(X_train_sample,y_train_sample))

	
	y_pred = votting(svc_models,X_val)
	print 'SVC with costing loss:',cost_score(y_pred,y_val)

	y_pred = votting(rf_models,X_val)
	print 'Random Forest with costing loss:',cost_score(y_pred,y_val)


	y_pred = votting(nb_models,X_val)
	print 'Naive Bayes with costing loss:',cost_score(y_pred,y_val)
	
	return 


if __name__ == '__main__':

	X_train, y_train, X_val, y_val, X_test, y_test = read_data()

	default_scores(X_train, y_train, X_val, y_val)
	class_weighting(X_train, y_train, X_val, y_val)
	class_oversampling(X_train, y_train, X_val, y_val)
	costing(X_train,y_train,X_val,y_val)


