from delicious_loader import load_dataset


import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Embedding, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import regularizers

import tensorflow as tf






def f1_score(y_true, y_pred):
	"""
	Compute the micro f(b) score with b=1.
	"""
	y_true = tf.cast(y_true, "float32")
	y_pred = tf.cast(tf.round(y_pred), "float32") # implicit 0.5 threshold via tf.round
	y_correct = y_true * y_pred


	sum_true = tf.reduce_sum(y_true, axis=1)
	sum_pred = tf.reduce_sum(y_pred, axis=1)
	sum_correct = tf.reduce_sum(y_correct, axis=1)


	precision = sum_correct / sum_pred
	recall = sum_correct / sum_true
	f_score = 2 * precision * recall / (precision + recall)
	f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)


	return tf.reduce_mean(f_score)






def build_model(num_features,
	num_classes,
	embedding_dims,
	maxlen):
	"""
	"""

	input_layer = Input(shape=(maxlen,),
		dtype='int32')


	embeddings = Embedding(num_features,
		embedding_dims,
		input_length=maxlen,
		embeddings_regularizer=regularizers.l1(10e-7))(input_layer)

	avg_layer = GlobalAveragePooling1D()(embeddings)
	predictions = Dense(num_classes, activation='sigmoid')(avg_layer)

	model = Model(inputs=input_layer, outputs=predictions)
	model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[f1_score])

	model.summary()

	return model






def load_model():
	"""
	"""

	json_file = open('multilabel_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	model.load_weights('multilabel_model.h5')
	print("Loaded model from disk")

	model.summary()

	model.compile(loss='binary_crossentropy',
		optimizer='adam',
		metrics=[f1_score])


	return model








if __name__ == '__main__':

	ngram_range = 1
	maxlen = 200
	batch_size = 32
	embedding_dims = 50
	epochs = 500
	num_classes = 20 



	X_train,y_train,X_val,y_val,X_test,y_test,word_index = load_dataset(ngram_range=ngram_range,maxlen=maxlen)

	num_features = len(word_index)
	print('Found %d words' % num_features)

	
	model = build_model(num_features,num_classes,embedding_dims,maxlen)

	model_json = model.to_json()
	with open("multilabel_model.json", "w") as json_file:
		json_file.write(model_json)


	early_stopping =EarlyStopping(monitor='val_f1_score',
		patience=15,
		mode='max')
	bst_model_path = 'multilabel_model.h5'
	model_checkpoint = ModelCheckpoint(bst_model_path,
		monitor='val_f1_score',
		verbose=1,
		save_best_only=True,
		mode='max',
		save_weights_only=True)

	model.fit(X_train, y_train,
		batch_size=batch_size,
		epochs=epochs,
		validation_data=(X_val, y_val),
		callbacks=[model_checkpoint,early_stopping])
	
	
	model = load_model()
	y_pred = model.predict(X_test)

	print 'AUC:',roc_auc_score(y_test, y_pred)
	y_pred[y_pred > 0.24] = 1
	y_pred[y_pred <= 0.24] = 0
	

	for i in range(10):
		pred,lab = y_pred[i],y_test[i]
		print np.where(pred == 1), np.where(lab == 1)
    	

 	print precision_recall_fscore_support(y_test, y_pred, average='micro')
	print precision_recall_fscore_support(y_test, y_pred, average='macro')