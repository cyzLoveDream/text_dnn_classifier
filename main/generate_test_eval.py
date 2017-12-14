import pandas as pd
train_data = pd.read_csv("../data_example/eval_test.tsv",sep="\t",names=["txt_id","txt_title","txt_content","txt_label"])
train_data[["txt_id","txt_title","txt_content"]].to_csv("../data_example/test.tsv",sep="\t",header=False,index=False)
train_data[["txt_id","txt_label"]].to_csv("../data_example/eval_test.csv",header=True, index=False)