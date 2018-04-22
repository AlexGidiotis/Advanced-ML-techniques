import sys

import numpy as np

from sklearn.datasets import fetch_20newsgroups ,fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from sklearn.svm import SVC
from keras.datasets import imdb


def load_20news():
	"""
	"""
	print 'Loading data...'
	newsgroups_train = fetch_20newsgroups(subset='train',
		remove=('headers', 'footers', 'quotes'),
		shuffle=True)
	newsgroups_test = fetch_20newsgroups(subset='test',
		remove=('headers', 'footers', 'quotes'),
		shuffle=True)

	print 'Preprocessing...'
	vectorizer = TfidfVectorizer(strip_accents='unicode',
		lowercase=True,
		stop_words='english',
		ngram_range=(1, 2),
		max_df=0.5,
		min_df=5,
		max_features=20000,
		norm='l2',
		use_idf=True,
		smooth_idf=True,
		sublinear_tf=False)

	vectorizer.fit(newsgroups_train.data)

	X_train = vectorizer.transform(newsgroups_train.data)
	y_train = newsgroups_train.target
	X_test = vectorizer.transform(newsgroups_test.data)
	y_test = newsgroups_test.target

	X_train, X_val, y_train, y_val = train_test_split(X_train,
		y_train,
		test_size=0.2,
		random_state=0)

	return X_train, y_train, X_val, y_val, X_test, y_test


def load_imdb():
	"""
	"""
	print 'Loading data...'

	word_to_index = imdb.get_word_index()
	index_to_word = [None] * (max(word_to_index.values()) + 1)
	for w, i in word_to_index.items():
		index_to_word[i] = w

	(X_train, y_train), (X_test, y_test) = imdb.load_data()

	print 'Preprocessing...'
	X_train = [
		' '.join(index_to_word[i]
			for i in X_train[i]
			if i < len(index_to_word))
		for i in range(X_train.shape[0])
	]

	X_test = [
		' '.join(index_to_word[i]
			for i in X_test[i]
			if i < len(index_to_word)) 
		for i in range(X_test.shape[0])
	]

	vectorizer = TfidfVectorizer(strip_accents='unicode',
		lowercase=True,
		stop_words='english',
		ngram_range=(1, 2),
		max_df=0.5,
		min_df=5,
		max_features=50000,
		norm='l2',
		use_idf=True,
		smooth_idf=True,
		sublinear_tf=False)

	vectorizer.fit(X_train)

	X_train = vectorizer.transform(X_train)
	X_test = vectorizer.transform(X_test)


	X_train, X_val, y_train, y_val = train_test_split(X_train,
		y_train,
		test_size=0.2,
		random_state=0)

	return X_train, y_train, X_val, y_val, X_test, y_test



if __name__ == '__main__':
	"""
	20 newsgroup: 
		Bagging: max_samples=0.8, max_features=0.7
		AdaBoost: n_estimators=300, learning_rates=1.7

	IMDB:
		Bagging: max_samples=0.7, max_features=0.95, n_estimators=40
		AdaBoost: n_estimators=200, learning_rates=1.0
	"""

	dataset = sys.argv[1]

	if dataset == '20news':
		X_train, y_train, X_val, y_val, X_test, y_test = load_20news()
	elif dataset == 'imdb':
		X_train, y_train, X_val, y_val, X_test, y_test = load_imdb()


	clf = MultinomialNB(alpha=.01)


	clf.fit(X_train, y_train)
	preds = clf.predict(X_train)
	val_preds = clf.predict(X_val)
	print 'NB training f-score:',metrics.f1_score(y_train, preds, average='macro')
	print 'NB test f-score:',metrics.f1_score(y_val, val_preds, average='macro')

	estimator_nums = [40]
	max_samps = [0.7]
	max_feats = [0.95]
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
	val_preds = best_clf.predict(X_val)
	print 'Bagging training f-score:',metrics.f1_score(y_train, preds, average='macro')
	print 'Bagging test f-score:',metrics.f1_score(y_val, val_preds, average='macro')

	estimator_nums = [200]
	learning_rates = [1.0]
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
	val_preds = best_clf.predict(X_val)
	print 'AdaBoost training f-score:',metrics.f1_score(y_train, preds, average='macro')
	print 'AdaBoost test f-score:',metrics.f1_score(y_val, val_preds, average='macro')

	


