import keras.backend as K
from keras.layers import Embedding,Input, Convolution1D, MaxPooling1D, Flatten, Dense, Bidirectional,concatenate,GRU,Dropout,LSTM
from keras.models import Model,Sequential
from keras.engine import Layer, InputSpec
from keras.layers import Flatten,BatchNormalization
from keras.layers import Conv1D,Activation,MaxPool1D
import tensorflow as tf

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

def MLP(vocab_size, embedding_matrix,ckpt_path,max_len,embedding_size):
	"""
	Multilayer perceptron
	:param vocab_size:
	:param embedding_matrix:
	:param ckpt_path:
	:param max_len:
	:param embedding_size:
	:return:
	"""
	seq = Input(shape=(max_len,))
	den = Dense(512, activation="relu",input_shape=(len(vocab_size) +1,))(seq)
	dro = Dropout(0.5)(den)
	out = Dense(1, activation="sigmoid")(dro)
	model = Model(inputs=seq, outputs=out)
	try:
		model.load_weights(ckpt_path)
		print("load weights finish...")
	except:
		print("no pre weights")
	model.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy",f1])
	return model

def CNN(vocab_size, embedding_matrix,ckpt_path,max_len,embedding_size):
	"""
	look like LeNet-5
	:param vocab_size:
	:param embedding_matrix:
	:param ckpt_path:
	:param max_len:
	:param embedding_size:
	:return:
	"""
	model = Sequential()
	model.add(Embedding(vocab_size,embedding_size, weights=[embedding_matrix],input_length=max_len))
	model.add(Convolution1D(256, 3, padding="same"))
	model.add(MaxPooling1D(3,3,padding="same"))
	model.add(Convolution1D(128,3, padding="same"))
	model.add(MaxPooling1D(3,3,padding="same"))
	model.add(Convolution1D(64,3, padding="same"))
	model.add(MaxPooling1D(3,3,padding="same"))
	model.add(Flatten())
	model.add(Dropout(0.1))
	model.add(BatchNormalization())
	model.add(Dense(256, activation="relu"))
	model.add(Dropout(0.1))
	model.add(Dense(1,activation="sigmoid"))
	try:
		model.load_weights(ckpt_path)
		print("load weights finish....")
	except:
		print("no pre weights...")
	model.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy", f1])
	return model

def RNN(vocab_size, embedding_matrix,ckpt_path,max_len,embedding_size):
	"""
	simple rnn + LSTM
	:param vocab_size:
	:param embedding_matrix:
	:param ckpt_path:
	:param max_len:
	:param embedding_size:
	:return:
	"""
	model = Sequential()
	model.add(Embedding(len(vocab_size),embedding_size,input_length=max_len))
	model.add(LSTM(256,dropout=0.2,recurrent_dropout=0.1))
	model.add(Dense(1,activation='sigmoid'))
	try:
		model.load_weights(ckpt_path)
		print("load weights finish....")
	except:
		print("no pre weights...")
	model.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy", f1])
	return model

def Bi_RNN(vocab_size, embedding_matrix,ckpt_path,max_len,embedding_size):
	"""
	bi-RNN + GRU
	:param vocab_size:
	:param embedding_matrix:
	:param ckpt_path:
	:param max_len:
	:param embedding_size:
	:return:
	"""
	model = Sequential()
	model.add(Embedding(len(vocab_size),embedding_size,input_length=max_len))
	model.add(Bidirectional(GRU(256,dropout=0.2,recurrent_dropout=0.1,return_sequences=True)))
	model.add(Bidirectional(GRU(256,dropout=0.2,recurrent_dropout=0.1)))
	model.add(Dense(1,activation='sigmoid'))
	try:
		model.load_weights(ckpt_path)
		print("load weights finish....")
	except:
		print("no pre weights...")
	model.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy", f1])
	return model

def C_RNN_series(vocab_size, embedding_matrix,ckpt_path,max_len,embedding_size):
	"""
	CNN+ RNN in series
	:param vocab_size:
	:param embedding_matrix:
	:param ckpt_path:
	:param max_len:
	:param embedding_size:
	:return:
	"""
	model = Sequential()
	model.add(Embedding(len(vocab_size),embedding_size,input_length=max_len))
	model.add(Convolution1D(256,3,padding='same',strides=1))
	model.add(Activation('relu'))
	model.add(MaxPool1D(pool_size=2))
	model.add(GRU(256,dropout=0.2,recurrent_dropout=0.1,return_sequences=True))
	model.add(GRU(256,dropout=0.2,recurrent_dropout=0.1))
	model.add(Dense(1,activation='sigmoid'))
	try:
		model.load_weights(ckpt_path)
		print("load weights finish....")
	except:
		print("no pre weights...")
	model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy",f1])
	return model

def C_RNN_parallel(vocab_size, embedding_matrix,ckpt_path,max_len,embedding_size):
	"""
	CNN+ RNN in parallel
	:param vocab_size:
	:param embedding_matrix:
	:param ckpt_path:
	:param max_len:
	:param embedding_size:
	:return:
	"""
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


def textCNN(vocab_size, embedding_matrix,ckpt_path,max_len,embedding_size):
	"""
	textCNN
	:param vocab_size:
	:param ckpt_path: 权重路径
	:param could_ckpt_path:
	:return: model
	"""
	seq = Input(shape=(max_len,),name="input",dtype='int32')
	textCNN = Embedding(vocab_size,embedding_size,input_length=max_len,trainable=False, weights=[embedding_matrix])(seq)
	cnn1 = Conv1D(filters=128,kernel_size=2,activation="relu",padding="same")(textCNN)
	cnn1 = MaxPooling1D(pool_size=4)(cnn1)
	cnn2 = Conv1D(filters=128,kernel_size=3,activation="relu",padding="same")(textCNN)
	cnn2 = MaxPooling1D(pool_size=4)(cnn2)
	cnn3 = Conv1D(filters=128,kernel_size=5,activation="relu",padding="same")(textCNN)
	cnn3 = MaxPooling1D(pool_size=4)(cnn3)
	cnn = concatenate([cnn1,cnn2,cnn3],axis=-1)
	flt = Flatten()(cnn)
	drop = Dropout(0.5)(flt)
	d1 = Dense(64,activation="relu")(drop)
	output = Dense(1,activation="sigmoid")(d1)
	model = Model(inputs=seq,outputs=output)
	print(model.summary())
	try:
		model.load_weights(ckpt_path)
		print("load weights finish....")
	except:
		print("no pre-weights...")
	model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy",f1])
	return model


class KMaxPooling(Layer):
	"""
	K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
	TensorFlow backend.
	"""
	
	def __init__(self,k=1,**kwargs):
		
		super().__init__(**kwargs)
		self.input_spec = InputSpec(ndim=3)
		self.k = k
	
	def compute_output_shape(self,input_shape):
		return (input_shape[0],(input_shape[2] * self.k))
	
	def call(self, inputs):
		# swap last two dimensions since top_k will be applied along the last dimension
		shifted_input = tf.transpose(inputs,[0,2,1])
		
		# extract top_k, returns two tensors [values, indices]
		top_k = tf.nn.top_k(shifted_input,k=self.k,sorted=True,name=None)[0]
		
		# return flattened output
		return Flatten()(top_k)


def text_dcnn(vocab_size, embedding_matrix,ckpt_path,max_len,embedding_size):
	seq = Input(shape=(max_len,),name="input",dtype='int32')
	dcnn = Embedding(vocab_size,embedding_size,input_length=max_len,trainable=True)(seq)
	cnn = Convolution1D(filters=500,activation="relu",kernel_size=3,padding="same")(dcnn)
	cnn1 = Convolution1D(filters=500,kernel_size=3,activation="relu",padding="same")(dcnn)
	maxpool = KMaxPooling(k=5)(cnn)
	maxpool1 = KMaxPooling(k=5)(cnn1)
	# cnn = Convolution1D(filters=128,kernel_size=2,activation="relu",padding="same")(maxpool)
	# cnn1 = Convolution1D(filters=128,kernel_size=2,activation="relu",padding="same")(maxpool1)
	# maxpool = KMaxPooling(k=5)(cnn)
	# maxpool1 = KMaxPooling(k=5)(cnn1)
	cnn = concatenate([maxpool,maxpool1],axis=-1)
	flt = Flatten()(cnn)
	drop = Dropout(0.5)(flt)
	d1 = Dense(32,activation="relu")(drop)
	output = Dense(1,activation="sigmoid")(d1)
	model = Model(inputs=seq,outputs=output)
	print(model.summary())
	try:
		model.load_weights(ckpt_path)
		print("load weights finish....")
	except:
		print("no pre-weights...")
		pass
	model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy",f1])
	return model

