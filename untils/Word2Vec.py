import pandas as pd
import jieba
import time
import re
from tqdm import tqdm
from untils.tools import tools
class Word2Vec():
	
	def train_w2v_model(self,train_path,test_path,model_path, user_dict):
		"""
		分割词语，读取数据，搜索模式的分词结果
		:param train_filename:
		:param test_filename:
		:return: 分词后的列表，[[word1,word2],[w1,w2]]
		"""
		start = time.time()
		train_data = pd.read_csv(train_path,sep="\t",names=["txt_id","txt_title","txt_content","txt_label"])
		test_data = pd.read_csv(test_path,sep="\t",names=["txt_id","txt_title","txt_content"])
		print("the train data's count is:",train_data.shape[0])
		print("the test data's count is:",test_data.shape[0])
		test_data.fillna("missing",inplace=True)
		train_data.fillna("missing",inplace=True)
		print("finish load data in time {0}".format(time.time() - start))
		all_data = pd.concat([train_data.loc[:,["txt_id","txt_content"]],test_data.loc[:,["txt_id","txt_content"]]])
		print("merge data's count is {0}".format(all_data.shape[0]))
		all_data.fillna("missing",inplace=True)
		segment_word_list = []
		count = 0
		max_len = 0
		jieba.load_userdict(user_dict)
		for text in tqdm(all_data.txt_content.values):
			if len(text) > max_len:
				max_len = len(text)
			count += 1
			seg_word_list = jieba.lcut(tools().strQ2B(text),HMM=False)
			seg_word_list_without_puncte = []
			for fl in seg_word_list:
				pattern = re.compile(u"[\u4e00-\u9fa5]+")
				result = re.findall(pattern,fl)
				if result != "":
					seg_word_list_without_puncte.extend(result)
			segment_word_list.append(seg_word_list_without_puncte)
		print("\nfinish segment words in time {0}, the sample's count is {1}".format(time.time() - start,len(segment_word_list)))
		print("My opinion is that max_len is set to: ", int(max_len / 100) * 100)
		tools().trainingForWords(segment_word_list,model_path)
		return model_path
	
# Word2Vec().train_model("../data_example/train.tsv","../data_example/test.tsv","../model/w2v.pkl")
# tools().generate_word2index("../model/w2v.pkl")