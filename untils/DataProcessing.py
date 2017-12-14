import timeit
import tensorflow as tf
import pandas as pd
from tqdm import tqdm

class DataProcessing():
	def __init__(self, in_path, out_path):
		if in_path == False:
			self.out_path = out_path
		elif out_path == False:
			self.in_path = in_path
		elif in_path == False and out_path == False:
			pass
		else:
			self.in_path = in_path
			self.out_path = out_path
	
	def train_txt2tfrecords(self):
		"""
		将train转化为tfrecords格式的文件
		:param in_path:
		:param out_path:
		:return:
		"""
		print("\nStart to convert {} to {}\n".format(self.in_path,self.out_path))
		start_time = timeit.default_timer()
		writer = tf.python_io.TFRecordWriter(self.out_path)
		num = 0
		with open(self.in_path,mode="r",encoding="utf-8") as rf:
			lines = rf.readlines()
			for line in tqdm(lines):
				num += 1
				data = line.split("\t")
				try:
					txt_id = [bytes(data[0],"utf-8")]
					txt_title = [bytes(data[1],"utf-8")]
					txt_content = [bytes(data[2],"utf-8")]
					txt_label = [bytes(data[3][0:-1],"utf-8")]
				except:
					txt_id = [bytes(str(data[0])),"utf-8"]
					txt_title = [bytes(str(" ").strip()),"utf-8"]
					txt_content = [bytes(str(" ").strip(),"utf-8")]
					txt_label = [bytes(str(data[3][0:-1]),"utf-8")]
				example = tf.train.Example(features=tf.train.Features(feature={
					"txt_id":
						tf.train.Feature(bytes_list=tf.train.BytesList(value=txt_id)),
					"txt_title":
						tf.train.Feature(bytes_list=tf.train.BytesList(value=txt_title)),
					"txt_content":
						tf.train.Feature(bytes_list=tf.train.BytesList(value=txt_content)),
					"txt_label":
						tf.train.Feature(bytes_list=tf.train.BytesList(value=txt_label))
					}))
				writer.write(example.SerializeToString())  # 序列化为字符串
			writer.close()
			print("Successfully convert {} to {}".format(self.in_path,self.out_path))
			end_time = timeit.default_timer()
			print("\nThe pretraining process ran for {0} minutes\n".format((end_time - start_time) / 60))
			print("the count is: ",num)
			return num, self.out_path
			
	def test_txt2tfrecords(self):
		"""
		将测试集转化为tfrecords文件格式
		:param in_path:
		:param out_path:
		:return:
		"""
		print("\nStart to convert {} to {}\n".format(self.in_path,self.out_path))
		start_time = timeit.default_timer()
		writer = tf.python_io.TFRecordWriter(self.out_path)
		num = 0
		with open(self.in_path,mode="r",encoding="gbk") as rf:
			lines = rf.readlines()
			for line in tqdm(lines):
				num += 1
				data = line.strip().split("\t")
				try:
					txt_id = [bytes(data[0],"utf-8")]
					txt_title = [bytes(data[1],"utf-8")]
					txt_content = [bytes(data[2],"utf-8")]
				except:
					txt_id = [bytes(str(data[0]),"utf-8")]
					txt_title = [bytes(str(" "),"utf-8")]
					txt_content = [bytes(str(" "),"utf-8")]
				# 将数据转化为原生 bytes
				example = tf.train.Example(features=tf.train.Features(feature={
					"txt_id":
						tf.train.Feature(bytes_list=tf.train.BytesList(value=txt_id)),
					"txt_title":
						tf.train.Feature(bytes_list=tf.train.BytesList(value=txt_title)),
					"txt_content":
						tf.train.Feature(bytes_list=tf.train.BytesList(value=txt_content))
					}))
				writer.write(example.SerializeToString())  # 序列化为字符串
			writer.close()
			print("Successfully convert {} to {}".format(self.in_path,self.out_path))
			end_time = timeit.default_timer()
			print("\nThe pretraining process ran for {0} minutes\n".format((end_time - start_time) / 60))
			print("the count is: ",num)
			return num,self.out_path
		
	def parase_tfrecords_to_dataFrame(self, data_shape):
		"""
		解析预测完的tfrecords,并且生成需要提交的文件
		:param filename:
		:param data_shape:
		:return:
		"""
		data_list = []
		with tf.Session() as sess:
			filename_queue = tf.train.string_input_producer([self.in_path],shuffle=False)
			read = tf.TFRecordReader()
			_,serialized_example = read.read(filename_queue)
			
			features = tf.parse_single_example(serialized_example,
			                                   features={
				                                   "txt_id": tf.FixedLenFeature([],tf.string),
				                                   "label": tf.FixedLenFeature([],tf.float32),
				                                   })
			txt_id = features['txt_id']
			label = features["label"]
			init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
			sess.run(init_op)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			for i in tqdm(range(data_shape)):
				content_list = sess.run([txt_id,label])
				c_l = []
				c_l.append(str(content_list[0],"utf-8"))
				c_l.append(content_list[1])
				data_list.append(c_l)
			coord.request_stop()
			coord.join(threads)
			sess.close()
		data_pd = pd.DataFrame(data_list,columns=["txt_id","label"])
		data_pd["label"] = data_pd["label"].apply(
			lambda x: "POSITIVE" if x > 0.50 else "NEGATIVE")
		data_pd.to_csv(self.out_path,header=False,index=False)
	
	def load_raw_train_data(self, data_shape):
		"""
		载入训练数据
		:param filename:
		:return:
		"""
		print("\nbegin load train data\n")
		data_list = []
		with tf.Session() as sess:
			filename_queue = tf.train.string_input_producer([self.in_path],shuffle=False,seed=0)
			read = tf.TFRecordReader()
			_,serialized_example = read.read(filename_queue)
			features = tf.parse_single_example(serialized_example,
			                                   features={
				                                   "txt_id": tf.FixedLenFeature([],tf.string),
				                                   "txt_title": tf.FixedLenFeature([],tf.string),
				                                   "txt_content": tf.FixedLenFeature([],tf.string),
				                                   "txt_label": tf.FixedLenFeature([],tf.string)
				                                   })
			txt_id = features["txt_id"]
			txt_title = features["txt_title"]
			txt_content = features["txt_content"]
			txt_label = features["txt_label"]
			init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
			sess.run(init_op)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			for i in tqdm(range(data_shape)):
				content_list = sess.run([txt_id,txt_title,txt_content,txt_label])
				c_l = []
				for d in content_list:
					c_l.append(str(d,"utf-8"))
				data_list.append(c_l)
			coord.request_stop()
			coord.join(threads)
			sess.close()
		data_pd = pd.DataFrame(data_list,columns=["txt_id","txt_title","txt_content","txt_label"])
		return data_pd
	
	def load_raw_test_data(self, data_shape):
		"""
		验证转存之后的数据是否相对应
		:param input_filename:
		:param data_shape: 数据总共有多少条
		:return:
		"""
		print("\nbegin load test data\n")
		data_list = []
		with tf.Session() as sess:
			filename_queue = tf.train.string_input_producer([self.in_path],shuffle=False,seed=0)
			read = tf.TFRecordReader()
			_,serialized_example = read.read(filename_queue)
			
			features = tf.parse_single_example(serialized_example,
			                                   features={
				                                   "txt_id": tf.FixedLenFeature([],tf.string),
				                                   "txt_title": tf.FixedLenFeature([],tf.string),
				                                   "txt_content": tf.FixedLenFeature([],tf.string),
				                                   # "txt_label": tf.FixedLenFeature([],tf.string)
				                                   })
			txt_id = features["txt_id"]
			txt_title = features["txt_title"]
			txt_content = features["txt_content"]
			# txt_label = features["txt_label"]
			init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
			sess.run(init_op)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			for i in tqdm(range(data_shape)):
				content_list = sess.run([txt_id,txt_title,txt_content])
				c_l = []
				for d in content_list:
					c_l.append(str(d,"utf-8"))
				data_list.append(c_l)
			coord.request_stop()
			coord.join(threads)
			sess.close()
		data_pd = pd.DataFrame(data_list,columns=["txt_id","txt_title","txt_content"])
		return data_pd

	def train_feature_txt2tfrecords(self, data):
		"""
		转化训练特征数据为tfrecords格式
		:param data:
		:return:
		"""
		print("\nStart to convert to {}\n".format(self.out_path))
		start_time = timeit.default_timer()
		writer = tf.python_io.TFRecordWriter(self.out_path)
		for line in tqdm(data):
			label = [int(line[0])]
			feature = [bytes(line[1],"utf-8")]
			# print(name)
			#  将数据转化为原生 bytes
			example = tf.train.Example(features=tf.train.Features(feature={
				"label":
					tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
				"feature":
					tf.train.Feature(bytes_list=tf.train.BytesList(value=feature))
				}))
			writer.write(example.SerializeToString())  # 序列化为字符串
		writer.close()
		print("Successfully convert to {}".format(self.out_path))
		end_time = timeit.default_timer()
		print("\nThe pretraining process ran for {0} minutes\n".format((end_time - start_time) / 60))
		
	def test_featuure_txt2tfrecords(self,data):
		"""
		转化测试特征数据为tfrecords格式
		:param data:
		:return:
		"""
		print("\nStart to convert to {}\n".format(self.out_path))
		start_time = timeit.default_timer()
		writer = tf.python_io.TFRecordWriter(self.out_path)
		for line in tqdm(data):
			txt_id = [bytes(line[0],"utf-8")]
			feature = [bytes(line[1],"utf-8")]
			# print(name)
			#  将数据转化为原生 bytes
			example = tf.train.Example(features=tf.train.Features(feature={
				"txt_id":
					tf.train.Feature(bytes_list=tf.train.BytesList(value=txt_id)),
				"feature":
					tf.train.Feature(bytes_list=tf.train.BytesList(value=feature))
				}))
			writer.write(example.SerializeToString())  # 序列化为字符串
		writer.close()
		print("Successfully convert to {}".format(self.out_path))
		end_time = timeit.default_timer()
		print("\nThe pretraining process ran for {0} minutes\n".format((end_time - start_time) / 60))
	
	def load_tfrecords_train_feature_data_train(self,data_shape):
		"""
		载入训练数据
		:param filename:
		:return:
		"""
		data_list = []
		with tf.Session() as sess:
			filename_queue = tf.train.string_input_producer([self.in_path],shuffle=False,seed=0)
			read = tf.TFRecordReader()
			_,serialized_example = read.read(filename_queue)
			
			features = tf.parse_single_example(serialized_example,
			                                   features={
				                                   "label": tf.FixedLenFeature([],tf.int64),
				                                   "feature": tf.FixedLenFeature([],tf.string)
				                                   })
			label = features["label"]
			feature = features["feature"]
			init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
			sess.run(init_op)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			for i in tqdm(range(data_shape)):
				content_list = sess.run([label,feature])
				c_l = []
				c_l.append([content_list[0],eval(str(content_list[1],"utf-8"))])
				data_list.extend(c_l)
			coord.request_stop()
			coord.join(threads)
			sess.close()
		print("have been data's count is: ",data_shape)
		return data_list
	
	def load_tfrecords_test_feature_data_train(self,data_shape):
		"""
		载入训练数据
		:param filename:
		:return:
		"""
		data_list = []
		with tf.Session() as sess:
			filename_queue = tf.train.string_input_producer([self.in_path],shuffle=False,seed=0)
			read = tf.TFRecordReader()
			_,serialized_example = read.read(filename_queue)
			
			features = tf.parse_single_example(serialized_example,
			                                   features={
				                                   "txt_id": tf.FixedLenFeature([],tf.string),
				                                   "feature": tf.FixedLenFeature([],tf.string)
				                                   })
			feature = features["feature"]
			txt_id = features["txt_id"]
			init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
			sess.run(init_op)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			for i in tqdm(range(data_shape)):
				feature_data = sess.run([txt_id,feature])
				f_l = str(feature_data[1],"utf-8")
				data_list.append([str(feature_data[0],"utf-8"),eval(f_l)])
			coord.request_stop()
			coord.join(threads)
			sess.close()
		print("have been data's count is: ",data_shape)
		return data_list


def main():
	"""
	测试当前类的方法的主方法入口
	:return:
	"""
	# DataProcessing("../data_example/test.tsv","../data_example/test.tfrecords").test_txt2tfrecords()
	# DataProcessing("../data_example/train.tsv","../data_example/train.tfrecords").train_txt2tfrecords()
	# DataProcessing("../feature_data/tsz_submission_12.tfrecords","../submission/sub.csv").parase_tfrecords_to_dataFrame(1500)
	# data = DataProcessing("../data_example/train.tfrecords",False).load_train_data(2000)
	# data = DataProcessing("../data_example/test.tfrecords",False).load_test_data(1500)
	# print(data.head())
if __name__ == '__main__':
	main()