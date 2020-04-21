from bunch import bunchify

conf_dict = {
    "phrase": 2,
	"pred": {
		"model_dir": "saved_models",
		"model": "10.h5",
		"output": "underexpose_submit-1.csv",
		"pred_size": 50
	},
	"data": {
		"base_dir": "data/",
		"submit_dir": "data/submit/",
		"local_eval_file": "data/local_answer.csv",
		"item_feat": "underexpose_train/underexpose_item_feat.csv",
		"user_feat": "underexpose_train/underexpose_user_feat.csv",
        "user_cols": ['user_id','age','gender','city'],
		"train_file": "underexpose_train/underexpose_train_click-%d.csv",
		"test_file": "underexpose_test/underexpose_test_click-%d.csv",
		"predict_file": "underexpose_test/underexpose_test_qtime-%d.csv",

	},
}

opt = bunchify(conf_dict)
