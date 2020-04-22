import sys
sys.path.append("utils/recommenders-master/")

import logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)-8s %(message)s')
import numpy as np
import pandas as pd
import papermill as pm
import time
import os 
os.environ['NUMEXPR_MAX_THREADS'] = '64'

from reco_utils.common.timer import Timer
from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_stratified_split
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from reco_utils.recommender.sar import SAR

from load_data import load_data, load_item, gen_full_data, gen_train_eval
from conf import opt

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))

if __name__ == "__main__":
	train_data = load_data(opt.data.base_dir+opt.data.train_file, opt.phrase)
	test_data = load_data(opt.data.base_dir+opt.data.test_file, opt.phrase)
	qtime_data = load_data(opt.data.base_dir+opt.data.predict_file, opt.phrase, qtime=True)
	whole_click = gen_full_data([train_data, test_data])
	whole_click['rating'] = 1
	train_data, eval_data = gen_train_eval(whole_click)
	
	model = SAR(
		col_user="user_id",
		col_item="item_id",
		col_rating="qtime",
		col_timestamp="timestamp",
		similarity_type="cooccurrence", 
		time_decay_coefficient=1, 
		timedecay_formula=True
	)
	with Timer() as train_time:
		model.fit(whole_click)

	logging.info("Took {} seconds for training.".format(train_time.interval))
	with Timer() as test_time:
		top_k = model.recommend_k_items(qtime_data, remove_seen=True, top_k=50)	
	
	sub = top_k.groupby('user_id')['item_id'].apply(lambda x: ','.join(
        [str(i) for i in x])).str.split(',', expand=True).reset_index()
	sub.to_csv("sar_for_test.csv", header=None, index=False)
	'''
	TOP_K = 50
	args = [eval_data, top_k]
	kwargs = dict(col_user='user_id', 
				col_item='item_id', 
				col_rating='qtime', 
				col_prediction='prediction', 
				relevancy_method='top_k', 
				k=TOP_K)
	with Timer() as test_time:
		eval_map = map_at_k(*args, **kwargs)
		eval_ndcg = ndcg_at_k(*args, **kwargs)
		eval_precision = precision_at_k(*args, **kwargs)
		eval_recall = recall_at_k(*args, **kwargs)
	print(f"Model:",
      f"Top K:\t\t {TOP_K}",
      f"MAP:\t\t {eval_map:f}",
      f"NDCG:\t\t {eval_ndcg:f}",
      f"Precision@K:\t {eval_precision:f}",
      f"Recall@K:\t {eval_recall:f}", sep='\n')
	'''
