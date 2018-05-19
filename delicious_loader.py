import pandas as pd
import numpy as np
import re
import time


from keras.preprocessing import sequence





def read_data(file,
	lab_file):
	"""
	"""

	X_data = pd.read_csv(file,header=None)
	y_data = pd.read_csv(lab_file,header=None)

	X_data = X_data[0].map(lambda x: re.sub('<\d+>','',x) \
		.strip() \
		.split())
	X_data = X_data.map(lambda x: [int(tok.strip()) for tok in x])
	y_data = y_data[0].map(lambda x: np.array([int(lab) for lab in x.split()]))

	return X_data.tolist(),np.array(y_data.tolist())







def read_data_sentences(file,
	lab_file,
	maxlen,
	max_sentence_len):
	"""
	"""

	X_data = pd.read_csv(file,header=None)
	y_data = pd.read_csv(lab_file,header=None)

	X_data = X_data[0].map(lambda x: x.strip())

	X_data = X_data.map(lambda x: re.findall('<\d+>([^<]+)',x)[1:])

	X_data = X_data.map(lambda x: [[int(tok.strip()) for tok in sent.strip().split()] for sent in x ])

	y_data = y_data[0].map(lambda x: np.array([int(lab) for lab in x.split()]))

	X_data = X_data.tolist()
	X_data_int = np.zeros((len(X_data),maxlen,max_sentence_len))
	for idx,text_bag in enumerate(X_data):
		sentences_batch = np.zeros((maxlen,max_sentence_len))
		sentences =  sequence.pad_sequences(text_bag,
			maxlen=max_sentence_len,
			padding='post',
			truncating='post',
			dtype='int32')
		for j,sent in enumerate(sentences):
			if j >= max_sentence_len:
				break
			sentences_batch[j,:] = sent
		X_data_int[idx,:,:] = sentences_batch

	X_data = X_data_int

	return X_data,np.array(y_data.tolist())







def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))







def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences







def load_dataset(maxlen,
	ngram_range=1):
	"""
	"""
	train_data = 'data/delicious/train-data.dat'
	train_labels = 'data/delicious/train-label.dat'
	val_data = 'data/delicious/valid-data.dat'
	val_labels = 'data/delicious/valid-label.dat'
	test_data = 'data/delicious/test-data.dat'
	test_labels = 'data/delicious/test-label.dat'
	vocab_file = 'data/delicious/vocabs.txt'


	print('Loading data...')
	X_train, y_train = read_data(train_data,train_labels)
	X_val, y_val = read_data(val_data,val_labels)
	X_test, y_test = read_data(test_data,test_labels)
	print(len(X_train), 'train sequences')
	print(len(X_test), 'test sequences')
	print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
	print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))


	word_index = {}
	with open(vocab_file,'r') as vf:
		for line in vf:
			line = line.strip().split(', ')
			key = line[0]
			value = int(line[1])
			word_index[key] = value

	max_features = len(word_index)

	if ngram_range > 1:
	    print('Adding {}-gram features'.format(ngram_range))
	    # Create set of unique n-gram from the training set.
	    ngram_set = set()
	    for input_list in X_train:
	        for i in range(2, ngram_range + 1):
	            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
	            ngram_set.update(set_of_ngram)

	    # Dictionary mapping n-gram token to a unique integer.
	    # Integer values are greater than max_features in order
	    # to avoid collision with existing features.
	    start_index = max_features + 1
	    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
	    indice_token = {token_indice[k]: k for k in token_indice}

	    # max_features is the highest integer that could be found in the dataset.
	    max_features = np.max(list(indice_token.keys())) + 1

	    # Augmenting x_train and x_test with n-grams features
	    X_train = add_ngram(X_train, token_indice, ngram_range)
	    X_val = add_ngram(X_val, token_indice, ngram_range)
	    X_test = add_ngram(X_test, token_indice, ngram_range)
	    print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
	    print('Average val sequence length: {}'.format(np.mean(list(map(len, X_val)), dtype=int)))
	    print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))
	

	print('Pad sequences (samples x time)')
	X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
	X_val = sequence.pad_sequences(X_val, maxlen=maxlen)
	X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
	print('X_train shape:', X_train.shape)
	print('X_val shape:', X_val.shape)
	print('X_test shape:', X_test.shape)


	return X_train,y_train,X_val,y_val,X_test,y_test,word_index






def load_dataset_hierarchical(maxlen,
	max_sentence_len):
	"""
	"""
	train_data = 'data/delicious/train-data.dat'
	train_labels = 'data/delicious/train-label.dat'
	val_data = 'data/delicious/valid-data.dat'
	val_labels = 'data/delicious/valid-label.dat'
	test_data = 'data/delicious/test-data.dat'
	test_labels = 'data/delicious/test-label.dat'
	vocab_file = 'data/delicious/vocabs.txt'


	print('Loading data...')
	X_train, y_train = read_data_sentences(train_data,train_labels,maxlen,max_sentence_len)
	X_val, y_val = read_data_sentences(val_data,val_labels,maxlen,max_sentence_len)
	X_test, y_test = read_data_sentences(test_data,test_labels,maxlen,max_sentence_len)
	

	word_index = {}
	with open(vocab_file,'r') as vf:
		for line in vf:
			line = line.strip().split(', ')
			key = line[0]
			value = int(line[1])
			word_index[key] = value

	max_features = len(word_index)

	print('X_train shape:', X_train.shape)
	print('X_val shape:', X_val.shape)
	print('X_test shape:', X_test.shape)


	return X_train,y_train,X_val,y_val,X_test,y_test,word_index