import sys

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from loaders import load_20news,load_imdb,load_sms,load_amazon,load_paper_reviews
from loaders import load_yelp,load_youtube,load_reuters8,load_reuters52,load_webkb
	

if __name__ == '__main__':
	"""
	20 newsgroup: 
		Bagging: max_samples=0.8, max_features=0.7, n_estimators=50
		AdaBoost: n_estimators=300, learning_rates=1.7
		GradientBoostingClassifier: estimator_nums=100, learning_rates=0.5, max_depths=5
		RandomForestClassifier: estimator_nums=100, max_depths=7


	IMDB:
		Bagging: max_samples=0.7, max_features=0.95, n_estimators=40
		AdaBoost: n_estimators=200, learning_rates=1.0
		GradientBoostingClassifier: estimator_nums=, learning_rates=, max_depths=
		RandomForestClassifier: estimator_nums=, max_depths=


	SMSSpamCollection:
		Bagging: max_samples=0.4, max_features=0.6, n_estimators=60
		AdaBoost: n_estimators=30, learning_rates=1.5
		GradientBoostingClassifier: estimator_nums=100, learning_rates=0.5, max_depths=3
		RandomForestClassifier: estimator_nums=, max_depths=


	paper reviews:
		Bagging: max_samples=0.4, max_features=0.8, n_estimators=20
		AdaBoost: n_estimators=10, learning_rates=0.3
		GradientBoostingClassifier: estimator_nums=100, learning_rates=1.0, max_depths=2
		RandomForestClassifier: estimator_nums=, max_depths=


	yelp:
		Bagging: max_samples=0.3, max_features=0.95, n_estimators=70
		AdaBoost: n_estimators=60, learning_rates=1.5
		GradientBoostingClassifier: estimator_nums=, learning_rates=, max_depths=
		RandomForestClassifier: estimator_nums=, max_depths=


	amazon:
		Bagging: max_samples=0.3, max_features=0.5, n_estimators=20
		AdaBoost: n_estimators=30, learning_rates=0.5
		GradientBoostingClassifier: estimator_nums=50, learning_rates=0.5, max_depths=7
		RandomForestClassifier: estimator_nums=, max_depths=


	youtube:
		Bagging: max_samples=0.3, max_features=0.6, n_estimators=10
		AdaBoost: n_estimators=10, learning_rates=0.5
		GradientBoostingClassifier: estimator_nums=50, learning_rates=0.7, max_depths=2
		RandomForestClassifier: estimator_nums=, max_depths=


	reuters8:
		Bagging: max_samples=0.5, max_features=0.9, n_estimators=100
		AdaBoost: n_estimators=100, learning_rates=1.2
		GradientBoostingClassifier: estimator_nums=100, learning_rates=0.5, max_depths=5
		RandomForestClassifier: estimator_nums=, max_depths=


	reuters52:
		Bagging: max_samples=0.95, max_features=0.95, n_estimators=50
		AdaBoost: n_estimators=250, learning_rates=1.0
		GradientBoostingClassifier: estimator_nums=, learning_rates=, max_depths=
		RandomForestClassifier: estimator_nums=, max_depths=


	reuterswebkb:
		Bagging: max_samples=0.7, max_features=0.5, n_estimators=100
		AdaBoost: n_estimators=50, learning_rates=0.95
		GradientBoostingClassifier: estimator_nums=, learning_rates=, max_depths=
		RandomForestClassifier: estimator_nums=, max_depths=
	"""

	dataset = sys.argv[1]

	if dataset == '20news':
		X_train, y_train, X_val, y_val, X_test, y_test = load_20news()
	elif dataset == 'imdb':
		X_train, y_train, X_val, y_val, X_test, y_test = load_imdb()
	elif dataset == 'sms':
		X_train, y_train, X_val, y_val, X_test, y_test = load_sms()
	elif dataset == 'p_reviews':
		X_train, y_train, X_val, y_val, X_test, y_test = load_paper_reviews()
	elif dataset == 'yelp':
		X_train, y_train, X_val, y_val, X_test, y_test = load_yelp()
	elif dataset == 'amazon':
		X_train, y_train, X_val, y_val, X_test, y_test = load_amazon()
	elif dataset == 'youtube':
		X_train, y_train, X_val, y_val, X_test, y_test = load_youtube()
	elif dataset == 'r8':
		X_train, y_train, X_val, y_val, X_test, y_test = load_reuters8()
	elif dataset == 'r52':
		X_train, y_train, X_val, y_val, X_test, y_test = load_reuters52()
	elif dataset == 'webkb':
		X_train, y_train, X_val, y_val, X_test, y_test = load_webkb()


	clf = MultinomialNB(alpha=.01)


	clf.fit(X_train, y_train)
	preds = clf.predict(X_train)
	val_preds = clf.predict(X_test)
	print 'NB training f-score:',metrics.f1_score(y_train, preds, average='macro')
	print 'NB test f-score:',metrics.f1_score(y_test, val_preds, average='macro')
	
	estimator_nums = [100]
	max_samps = [0.7]
	max_feats = [0.5]
	best_fscore = 0.0
	for m in max_samps:
		for n in estimator_nums:
			for f in max_feats:
				bagg_clf = BaggingClassifier(clf,
					n_estimators=n,
					max_samples=m,
					max_features=f,
					random_state=0)
				bagg_clf.fit(X_train, y_train)

				val_preds = bagg_clf.predict(X_val)
				val_score = metrics.f1_score(y_val, val_preds, average='macro')
				if val_score > best_fscore:
					best_fscore = val_score
					best_params = (m,n,f)
					best_clf = bagg_clf

	print 'best parameters:',best_params
	preds = best_clf.predict(X_train)
	val_preds = best_clf.predict(X_test)
	print 'Bagging training f-score:',metrics.f1_score(y_train, preds, average='macro')
	print 'Bagging test f-score:',metrics.f1_score(y_test, val_preds, average='macro')
	
	
	estimator_nums = [50]
	learning_rates = [0.95]
	best_fscore = 0.0
	for n in estimator_nums:
		for lr in learning_rates:
			ada_clf = AdaBoostClassifier(clf,
				n_estimators=n,
				learning_rate=lr,
				random_state=0)
			ada_clf.fit(X_train, y_train)
			
			val_preds = ada_clf.predict(X_val)
			val_score = metrics.f1_score(y_val, val_preds, average='macro')
			if val_score > best_fscore:
				best_fscore = val_score
				best_params = (n,lr)
				best_clf = ada_clf

	print 'best parameters:',best_params
	preds = best_clf.predict(X_train)
	val_preds = best_clf.predict(X_test)
	print 'AdaBoost training f-score:',metrics.f1_score(y_train, preds, average='macro')
	print 'AdaBoost test f-score:',metrics.f1_score(y_test, val_preds, average='macro')
	
	
	estimator_nums = [100]
	learning_rates = [0.5]
	max_depths = [5]
	best_fscore = 0.0
	for n in estimator_nums:
		for lr in learning_rates:
			for d in max_depths:
				gb_clf = GradientBoostingClassifier(n_estimators=n,
					max_depth=d,
					learning_rate=lr,
					random_state=0)
				gb_clf.fit(X_train, y_train)
				
				val_preds = gb_clf.predict(X_val)
				val_score = metrics.f1_score(y_val, val_preds, average='macro')
				if val_score > best_fscore:
					best_fscore = val_score
					best_params = (n,lr,d)
					best_clf = gb_clf

	print 'best parameters:',best_params
	preds = best_clf.predict(X_train)
	val_preds = best_clf.predict(X_test)
	print 'Gradient Boosting training f-score:',metrics.f1_score(y_train, preds, average='macro')
	print 'Gradient Boosting test f-score:',metrics.f1_score(y_test, val_preds, average='macro')


	estimator_nums = [50]
	max_depths = [7]
	best_fscore = 0.0
	for n in estimator_nums:
		for d in max_depths:
			forest_clf = RandomForestClassifier(n_estimators=n,
				max_depth=d,
				random_state=0)
			forest_clf.fit(X_train, y_train)
			
			val_preds = forest_clf.predict(X_val)
			val_score = metrics.f1_score(y_val, val_preds, average='macro')
			if val_score > best_fscore:
				best_fscore = val_score
				best_params = (n,d)
				best_clf = forest_clf

	print 'best parameters:',best_params
	preds = best_clf.predict(X_train)
	val_preds = best_clf.predict(X_test)
	print 'Random Forest training f-score:',metrics.f1_score(y_train, preds, average='macro')
	print 'Random Forest test f-score:',metrics.f1_score(y_test, val_preds, average='macro')
	
	

