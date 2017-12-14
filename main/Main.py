import time
from untils import tools
from untils import Word2Vec
import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
def main():
	
	train_path,test_path,user_dict,w2v_path,\
	sub_path,train_number,test_number,max_len,\
	batch_size,epoch,ckpt,model_file,embedding_size,eval_test = tools.tools().load_params()
	
	start = time.time()
	# first train word2vec model
	print("begin train w2v model...")
	w2v_time_start = time.time()
	if not os.path.exists(w2v_path):
		w2v_path = Word2Vec.Word2Vec().train_w2v_model(train_path, test_path,w2v_path,user_dict)
	else:
		print("the w2v model is trained...")
	print("finish w2v model in time: ",time.time() - w2v_time_start)
	# second generate word2index, index2word, embeddings_index
	print("generate word2index, index2word, embeddings_index...")
	w2i_time_start = time.time()
	embedding_matrix,vocab_size,word2index,index2word = tools.tools().generate_word2index(w2v_path)
	print("finish word2index, index2word, embeddings_index: ",time.time() - w2i_time_start)
	print("begin load data...")
	start_load = time.time()
	x_train,x_test,y_train,y_test = tools.tools().load_train(train_path,
	                                                         "../data_example/train.tfrecords",
	                                                         word2index,
	                                                         max_len= max_len,
	                                                         user_dict= user_dict,
	                                                         sample_num=train_number)
	tests,id_list = tools.tools().load_test(test_path,
	                                "../data_example/test.tfrecords",
	                                word2index,
	                                max_len=max_len,
	                                user_dict=user_dict,
	                                sample_num=test_number)
	print("finish load data in time: ", time.time() - start_load)
	tools.tools().train_model(vocab_size, x_train, y_train, x_test, y_test, embedding_matrix,
	                          batch_size, epoch, ckpt, model_file,max_len,embedding_size)
	
	pred = tools.tools().predict(tests,id_list,vocab_size,embedding_matrix,batch_size,
	                      ckpt,max_len,embedding_size,sub_path)
	
	test_pd = pd.read_csv(eval_test)
	test_pd["txt_label"] = test_pd["txt_label"].apply(lambda x: 1 if x == "POSITIVE" else 0)
	f1_score = tools.tools().eval_test(test_pd.txt_label.values, pred)
	print("the final result f1_score is: ", f1_score)
	print("finish the all precessing in time: ",time.time() - start)

if __name__ == '__main__':
	main()