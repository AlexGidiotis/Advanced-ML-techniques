import sys

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups ,fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

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


def load_sms():
	"""
	"""
	print 'Loading data...'

	df = pd.read_csv('data/SMSSpamCollection',
		header=None,
		delimiter='\t')

	classes = dict((k,idx) for idx,k in enumerate(df[0].unique()))
	y_data = df[0].map(lambda x: classes[x]).tolist()
	X_data = df[1].tolist()

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

	vectorizer.fit(X_data)
	X_data = vectorizer.transform(X_data)

	X_train, X_test, y_train, y_test = train_test_split(X_data,
		y_data,
		test_size=0.1,
		random_state=0)

	X_train, X_val, y_train, y_val = train_test_split(X_train,
		y_train,
		test_size=0.2,
		random_state=0)

	return X_train, y_train, X_val, y_val, X_test, y_test


def load_paper_reviews():
	"""
	"""
	df = pd.read_json('data/reviews.json')
	class2id = dict((k,idx) for idx,k in enumerate(df['preliminary_decision'].unique()))

	y_data = df['preliminary_decision'].map(lambda x: class2id[x]).tolist()


	X_data = df['review']
	X_list = []
	y_list = []

	for i,(review,lab) in enumerate(zip(X_data,y_data)):
		try:
			X_list.append(review[0]['text'])
			y_list.append(lab)
		except:
			continue
	
	y_data = y_list

	print 'Preprocessing...'
	vectorizer = TfidfVectorizer(strip_accents='unicode',
		lowercase=True,
		stop_words='english',
		ngram_range=(1, 2),
		max_df=0.5,
		min_df=5,
		max_features=10000,
		norm='l2',
		use_idf=True,
		smooth_idf=True,
		sublinear_tf=False)

	vectorizer.fit(X_list)

	X_data = vectorizer.transform(X_list)
	
	X_train, X_test, y_train, y_test = train_test_split(X_data,
		y_data,
		test_size=0.1,
		random_state=0)

	X_train, X_val, y_train, y_val = train_test_split(X_train,
		y_train,
		test_size=0.2,
		random_state=0)

	return X_train, y_train, X_val, y_val, X_test, y_test


def load_yelp():
	"""
	"""
	df = pd.read_json('data/yelp.json',orient='records',lines=True, encoding='utf-8')

	X_data = df['text'].tolist()
	y_data = df['stars'].tolist()

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

	vectorizer.fit(X_data)

	X_data = vectorizer.transform(X_data)
	
	X_train, X_test, y_train, y_test = train_test_split(X_data,
		y_data,
		test_size=0.1,
		random_state=0)

	X_train, X_val, y_train, y_val = train_test_split(X_train,
		y_train,
		test_size=0.2,
		random_state=0)

	return X_train, y_train, X_val, y_val, X_test, y_test


def load_amazon():
	"""
	"""
	df = pd.read_csv('data/amazon.txt',
		header=None,
		delimiter='\t')

	X_data = df[0].tolist()
	y_data = df[1].tolist()

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

	vectorizer.fit(X_data)

	X_data = vectorizer.transform(X_data)
	
	X_train, X_test, y_train, y_test = train_test_split(X_data,
		y_data,
		test_size=0.1,
		random_state=0)

	X_train, X_val, y_train, y_val = train_test_split(X_train,
		y_train,
		test_size=0.2,
		random_state=0)

	return X_train, y_train, X_val, y_val, X_test, y_test


def load_youtube():
	"""
	"""
	df = pd.read_csv('data/untitled1/Youtube01-Psy.csv')
	df = df.append(pd.read_csv('data/untitled1/Youtube02-KatyPerry.csv'))
	df = df.append(pd.read_csv('data/untitled1/Youtube03-LMFAO.csv'))
	df = df.append(pd.read_csv('data/untitled1/Youtube04-Eminem.csv'))
	df = df.append(pd.read_csv('data/untitled1/Youtube05-Shakira.csv'))

	X_data = df["CONTENT"].tolist()
	y_data = df["CLASS"].tolist()


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

	vectorizer.fit(X_data)

	X_data = vectorizer.transform(X_data)
	
	X_train, X_test, y_train, y_test = train_test_split(X_data,
		y_data,
		test_size=0.1,
		random_state=0)

	X_train, X_val, y_train, y_val = train_test_split(X_train,
		y_train,
		test_size=0.2,
		random_state=0)

	return X_train, y_train, X_val, y_val, X_test, y_test


def load_reuters8():
	"""
	"""
	
	df = pd.read_csv('data/r8-train-all-terms.txt',
		header=None,
		delimiter='\t')

	test_df = pd.read_csv('data/r8-test-all-terms.txt',
		header=None,
		delimiter='\t')

	class2id = dict((k,idx) for idx,k in enumerate(df[0].unique()))

	X_train = df[1].tolist()
	X_test = test_df[1].tolist()

	y_train = df[0].map(lambda x: class2id[x]).tolist()
	y_test = test_df[0].map(lambda x: class2id[x]).tolist()

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

	vectorizer.fit(X_train)

	X_train = vectorizer.transform(X_train)
	X_test = vectorizer.transform(X_test)

	X_train, X_val, y_train, y_val = train_test_split(X_train,
		y_train,
		test_size=0.2,
		random_state=0)

	return X_train, y_train, X_val, y_val, X_test, y_test


def load_reuters52():
	"""
	"""
	
	df = pd.read_csv('data/r52-train-all-terms.txt',
		header=None,
		delimiter='\t')

	test_df = pd.read_csv('data/r52-test-all-terms.txt',
		header=None,
		delimiter='\t')

	class2id = dict((k,idx) for idx,k in enumerate(df[0].unique()))

	X_train = df[1].tolist()
	X_test = test_df[1].tolist()

	y_train = df[0].map(lambda x: class2id[x]).tolist()
	y_test = test_df[0].map(lambda x: class2id[x]).tolist()

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

	vectorizer.fit(X_train)

	X_train = vectorizer.transform(X_train)
	X_test = vectorizer.transform(X_test)

	X_train, X_val, y_train, y_val = train_test_split(X_train,
		y_train,
		test_size=0.2,
		random_state=0)

	return X_train, y_train, X_val, y_val, X_test, y_test


def load_webkb():
	"""
	"""
	
	df = pd.read_csv('data/web.txt',
		header=None,
		delimiter='\t')

	class2id = dict((k,idx) for idx,k in enumerate(df[0].unique()))

	y_data = df[0].map(lambda x: class2id[x]).tolist()
	X_list = df[1].tolist()


	X_data = []
	y_list = []
	for x,y in zip(X_list,y_data):
		try:
			if np.isnan(x):
				continue
		except:
			pass
		X_data.append(x)
		y_list.append(y)

	y_data = y_list

	print 'Preprocessing...'
	vectorizer = TfidfVectorizer(strip_accents='unicode',
		lowercase=True,
		stop_words='english',
		ngram_range=(1, 2),
		max_df=0.5,
		min_df=5,
		max_features=10000,
		norm='l2',
		use_idf=True,
		smooth_idf=True,
		sublinear_tf=False)

	vectorizer.fit(X_data)

	X_data = vectorizer.transform(X_data)
	
	X_train, X_test, y_train, y_test = train_test_split(X_data,
		y_data,
		test_size=0.1,
		random_state=0)

	X_train, X_val, y_train, y_val = train_test_split(X_train,
		y_train,
		test_size=0.2,
		random_state=0)

	return X_train, y_train, X_val, y_val, X_test, y_test