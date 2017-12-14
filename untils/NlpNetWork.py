import keras.backend as K
from keras.layers import Embedding,Input, Convolution1D, MaxPooling1D, Flatten, Dense, Bidirectional,concatenate,GRU,Dropout,LSTM
from keras.models import Model

def precision(y_true,y_pred):
	"""Precision metric.

	Only computes a batch-wise average of precision.

	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred,0,1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision


def recall(y_true,y_pred):
	"""Recall metric.

	Only computes a batch-wise average of recall.

	Computes the recall, a metric for multi-label classification of
	how many relevant items are selected.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))
	possible_positives = K.sum(K.round(K.clip(y_true,0,1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall


def f1(y_true,y_pred):
	def recall(y_true,y_pred):
		"""Recall metric.

		Only computes a batch-wise average of recall.

		Computes the recall, a metric for multi-label classification of
		how many relevant items are selected.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))
		possible_positives = K.sum(K.round(K.clip(y_true,0,1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall
	
	def precision(y_true,y_pred):
		"""Precision metric.

		Only computes a batch-wise average of precision.

		Computes the precision, a metric for multi-label classification of
		how many selected items are relevant.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred,0,1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred,0,1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision
	
	precisions = precision(y_true,y_pred)
	recalls = recall(y_true,y_pred)
	return 2 * ((precisions * recalls) / (precisions + recalls))

def RCNN(vocab_size, embedding_matrix,ckpt_path,max_len,embedding_size):
	main_input = Input(shape=(max_len,))
	embed = Embedding(vocab_size,embedding_size,input_length=max_len, weights=[embedding_matrix],trainable=False)(main_input)
	cnn = Convolution1D(256,3,padding='same',strides=1,activation='relu')(embed)
	cnn = MaxPooling1D(pool_size=4)(cnn)
	cnn = Flatten()(cnn)
	cnn = Dense(256)(cnn)
	rnn = Bidirectional(LSTM(256))(embed)
	rnn = Dense(256)(rnn)
	con = concatenate([cnn,rnn],axis=-1)
	den = Dense(128)(con)
	den = Dropout(0.5)(den)
	den = Dense(32)(den)
	main_output = Dense(1,activation='sigmoid')(den)
	model = Model(inputs=main_input,outputs=main_output)
	print(model.summary())
	try:
		model.load_weights(ckpt_path)
		print("load weights finish....")
	except:
		print("no pre-weights...")
	model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy",f1])
	return model
