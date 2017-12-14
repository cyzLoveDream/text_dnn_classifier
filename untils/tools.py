from gensim.models.word2vec import Word2Vec
import gensim
from tqdm import tqdm
from untils import DataProcessing
import time
import numpy as np
import pandas as pd
import jieba
import os
import re
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
from sklearn.model_selection import train_test_split
from configparser import ConfigParser
from untils.NlpNetWork import RCNN
from sklearn.metrics import f1_score
class tools():
	def strQ2B(self,ustring):
		"""
		全角转半角
		:param ustring: 需要转换的字符串
		:return: 半角字符串
		"""
		rstring = ""
		for uchar in ustring:
			inside_code = ord(uchar)
			if inside_code == 12288:  # 全角空格直接转换
				inside_code = 32
			elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
				inside_code -= 65248
			rstring += chr(inside_code)
		return rstring
	
	def trainingForWords(self,sentences,model_filename,fnum=500,NoUpdate=False):
		"""
		训练词向量的模型
		:param sentences: 一个列表，分完词之后的[[w1,w2],[w1,w2]]
		:param model_filename:
		:param fnum:
		:param isNoUpdate: 以后是否更新此训练的词向量的模型
		:return:
		"""
		
		# 设定词向量参数
		print("begin train word2vec model...")
		num_features = fnum  # 词向量的维度
		min_word_count = 4  # 词频数最低阈值
		num_workers = 12  # 线程数
		context = 10  # 上下文窗口大小
		downsampling = 1e-3  # 与自适应学习率有关
		num_iter = 10
		hs = 0
		sg = 1  # 是否使用skip-gram模型
		
		model = Word2Vec(sentences,workers=num_workers,hs=hs,
		                 size=num_features,min_count=min_word_count,iter=num_iter,
		                 window=context,sample=downsampling,sg=sg)
		if NoUpdate == True:
			model.init_sims(replace=True)  # 锁定训练好的word2vec,之后不能对其进行更新
		model.save(model_filename)  # 讲训练好的模型保存到文件中
		print('finish train')
		return model
	
	def generate_word2index(self, model_file):
		"""
		根据词向量模型产生word2index, 和词向量矩阵
		:param model_file: # 词向量模型的路径
		:return:
		"""
		embeddings_index = {}
		word2index = {}
		model = gensim.models.Word2Vec.load(model_file)
		word_vectors = model.wv
		print("get word2index...")
		for word,vocab_obj in tqdm(model.wv.vocab.items()):
			word2index[word] = vocab_obj.index + 2
			embeddings_index[word] = word_vectors[word]
		del model,word_vectors  # 删掉gensim模型释放内存
		word2index["PAD"] = 0
		word2index["UNK"] = 1
		index2word = {v : k for k,v in word2index.items()}
		print("found %s word vectors." % len(embeddings_index.get(index2word.get(2))))
		print("found %s words. " % len(word2index))
		embedding_matrix = np.zeros((len(word2index),len(embeddings_index.get(index2word.get(5)))))
		print("begin generate word matrix\n")
		for word,i in tqdm(word2index.items()):
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				# 文本数据中的词在词向量字典中没有，向量为取0；如果有则取词向量中该词的向量
				embedding_matrix[i] = embedding_vector
		return embedding_matrix, len(word2index), word2index, index2word
	
	def load_train(self, in_path, out_path, word2index, max_len, user_dict, sample_num):
		"""
		载入训练数据
		:param in_path: 训练数据的路径
		:param out_path: 中间转化tfrecords的路径
		:param word2index: word2index的词典
		:param max_len: 输入到神经网络的初始tensor长度
		:return:
		"""
		print("covert train raw data to tfrecords...")
		covert_start = time.time()
		if os.path.exists(out_path):
			os.remove(out_path)
		train_shape,data_path = DataProcessing.DataProcessing(in_path, out_path).train_txt2tfrecords()
		print("finish covert train raw data to tfrecords in time: ",time.time() - covert_start)
		data_pd = DataProcessing.DataProcessing(in_path=data_path,out_path= False).load_raw_train_data(train_shape)
		contents = data_pd.txt_content.values
		data_pd["txt_label"] = data_pd["txt_label"].apply(lambda x: 0 if x == "NEGATIVE" else 1)
		labels = data_pd.txt_label.values
		X = np.empty(data_pd.shape[0],dtype=list)
		print("the train2index's shape is: ", X.shape)
		jieba.load_userdict(user_dict)
		label = []
		print("\nbegin covert text to index\n")
		for i in tqdm(range(len(contents))):
			cut_list = jieba.lcut(tools().strQ2B(contents[i]),HMM=False)
			seg_word_list_without_biaodian = []
			for fl in cut_list:
				pattern = re.compile(u"[\u4e00-\u9fa5]+")
				result = re.findall(pattern,fl)
				if result != "":
					seg_word_list_without_biaodian.extend(result)
			seqs = []
			for w in seg_word_list_without_biaodian:
				if w in word2index:
					seqs.append(word2index[w])
				else:
					seqs.append(word2index["UNK"])
			X[i] = seqs
			if i < len(labels):
				label.append(labels[i])
		print("finish covert text to index..")
		labels = np.array(label)
		
		print("the label's shape: ",labels.shape)
		trains = sequence.pad_sequences(X,maxlen=max_len)
		print("split data")
		print("the train's shape is {0}".format(trains.shape))
		x_train,x_test,y_train,y_test = train_test_split(trains,labels,test_size=0.2,random_state=42)
		print('Shape of x_train tensor:',x_train.shape)
		print('Shape of y_train tensor:',y_train.shape)
		print('Shape of x_test tensor:',x_test.shape)
		print('Shape of y_test tensor:',y_test.shape)
		# y_train = np_utils.to_categorical(y_train,2)
		# y_test = np_utils.to_categorical(y_test,2)
		return x_train,x_test,y_train,y_test
	
	def load_test(self,in_path, out_path, word2index, max_len,user_dict, sample_num):
		"""
		载入训练数据
		:param in_path: 测试数据的路径
		:param out_path: 中间转化tfrecords的路径
		:param word2index: word2index的词典
		:param max_len: 输入到神经网络的初始tensor长度
		:return:
		"""
		print("covert test raw data to tfrecords...")
		covert_start = time.time()
		if os.path.exists(out_path):
			os.remove(out_path)
			print("remove the already exists cache...")
		test_shape,data_path = DataProcessing.DataProcessing(in_path,out_path).test_txt2tfrecords()
		print("finish covert test raw data to tfrecords in time: ",time.time() - covert_start)
		data_pd = DataProcessing.DataProcessing(in_path=data_path,out_path=False).load_raw_test_data(test_shape)
		contents = data_pd.txt_content.values
		id_list = list(data_pd.txt_id.values)
		X = np.empty(data_pd.shape[0],dtype=list)
		print("the test2index's shape is: ",X.shape)
		jieba.load_userdict(user_dict)
		print("\nbegin covert text to index\n")
		for i in tqdm(range(len(contents))):
			cut_list = jieba.lcut(tools().strQ2B(contents[i]),HMM=False)
			seg_word_list_without_biaodian = []
			for fl in cut_list:
				pattern = re.compile(u"[\u4e00-\u9fa5]+")
				result = re.findall(pattern,fl)
				if result != "":
					seg_word_list_without_biaodian.extend(result)
			seqs = []
			for w in seg_word_list_without_biaodian:
				if w in word2index:
					seqs.append(word2index[w])
				else:
					seqs.append(word2index["UNK"])
			X[i] = seqs
		print("finish covert text to index..")
		tests = sequence.pad_sequences(X,maxlen=max_len)
		return tests, id_list
	
	def train_model(self, vocab_size, x_train, y_train, x_test, y_test, embedding_matrix,
	                batch_size,epoch,ckpt,model_file,max_len,embedding_size):
		"""
		训练模型
		:param vocab_size:
		:param x_train:
		:param y_train:
		:param x_test:
		:param y_test:
		:param embedding_matrix:
		:return:
		"""
		callback = ModelCheckpoint(ckpt, verbose=1,save_best_only=True,save_weights_only=True)
		early_stop = EarlyStopping(monitor='val_loss',patience=5,verbose=1)
		reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
		                              patience=5,min_lr=0.001)
		callback_list = [callback,early_stop,reduce_lr]
		try:
			model = load_model(model_file)
			meritcs = model.evaluate(x_test,y_test)
			print('\ntest meritcs',meritcs)
			print("load model finish...")
		except:
			# model = textCNN(vocab_size,ckpt_path,cloud_ckpt_path,embedding_matrix)
			# model = text_dcnn(vocab_size,ckpt_path,cloud_ckpt_path,embedding_matrix)
			model = RCNN(vocab_size,embedding_matrix,ckpt, max_len,embedding_size)
			print("create model finish...")
		model.fit(x_train,y_train,
		          batch_size=batch_size,
		          callbacks=callback_list,
		          shuffle=True,
		          epochs=epoch,
		          verbose=1,
		          validation_data=(x_test,y_test))
		model.save(model_file)
	
	def predict(self,tests, tests_id, vocab_size, embedding_matrix,
	            batch_size,ckpt,max_len,embedding_size,sub_path):
		model = RCNN(vocab_size,embedding_matrix,ckpt,max_len,embedding_size)
		print("Created model and loaded weights from file")
		print("begin predict")
		pred = model.predict(tests,verbose=1,batch_size=batch_size)
		print("the pred's shape: ",pred.shape)
		test_pd = pd.DataFrame(tests_id,columns=["id"])
		test_pd["pred"] = pred
		print(list(pred)[0:100])
		print("test data's shape: ",test_pd.shape)
		print("predict finish...")
		test_pd.to_csv(sub_path, index=False)
		print("finish predict...")
		test_pd["pred"] = test_pd["pred"].apply(lambda x: 1 if x > 0.5 else 1)
		pred = test_pd.pred.values
		return pred
	
	def load_params(self):
		config_txt = "../config.txt"
		config = ConfigParser()
		config.read(config_txt)
		train_path = config.get("data_path","train_path")
		test_path = config.get("data_path","test_path")
		user_dict = config.get("data_path","user_dict")
		w2v_path = config.get("data_path","w2v_path")
		sub_path = config.get("data_path","sub_path")
		eval_test = config.get("data_path","eval_test")
		
		train_number = config.getint("data_number","train")
		test_number = config.getint("data_number","test")
		
		max_len = config.getint("net_word_params","max_len")
		batch_size = config.getint("net_word_params","batch_size")
		epoch = config.getint("net_word_params","epoch")
		ckpt = config.get("net_word_params","ckpt")
		model_file = config.get("net_word_params","model_file")
		embedding_size = config.getint("net_word_params","embedding_size")
		return train_path,test_path,user_dict,w2v_path,sub_path,train_number,\
		       test_number,max_len,batch_size,epoch,ckpt,model_file,embedding_size,eval_test
	
	def eval_test(self, label, pred):
		return f1_score(label, pred)