#!/bin/env python

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle


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

	nb_model = GaussianNB().fit(X_train,y_train)
	
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


if __name__ == '__main__':

	X_train, y_train, X_val, y_val, X_test, y_test = read_data()

	default_scores(X_train, y_train, X_val, y_val)
	class_weighting(X_train, y_train, X_val, y_val)
	class_oversampling(X_train, y_train, X_val, y_val)



