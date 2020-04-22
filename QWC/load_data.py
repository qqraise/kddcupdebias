import pandas as pd
from conf import opt
import time


def load_item(filename):
    train_item_df = pd.read_csv(filename, header=None)
    txt_cols = ['txt_vec'+str(i) for i in range(128)]
    img_cols = ['img_vec'+str(i) for i in range(128)]
    train_item_df.columns = ['item_id'] + txt_cols + img_cols
    train_item_df['txt_vec0'] = train_item_df['txt_vec0'].apply(
        lambda x: float(x[1:]))
    train_item_df['txt_vec127'] = train_item_df['txt_vec127'].apply(
        lambda x: float(x[:-1]))
    train_item_df['img_vec0'] = train_item_df['img_vec0'].apply(
        lambda x: float(x[1:]))
    train_item_df['img_vec127'] = train_item_df['img_vec127'].apply(
        lambda x: float(x[:-1]))
    train_item_df['txt_vec'] = train_item_df[txt_cols].values.tolist()
    train_item_df['img_vec'] = train_item_df[img_cols].values.tolist()
    return train_item_df[['item_id', 'txt_vec', 'img_vec']]

def gen_degree_file(data, filename):
	# opt.data.base_dir+"item_count.csv"
	item_count_df = data['item_id'].value_counts().sort_values(ascending=False).reset_index()
	item_count_df.columns = ['item_id', 'item_degree']
	item_count_df.to_csv(filename, index=False)

def gen_train_eval(data):
    gp = data.groupby(['user_id'])
    train_data = data[gp.cumcount(ascending=False) > 0]
    eval_data = data[gp.cumcount(ascending=False) == 0]
    return train_data, eval_data

def gen_full_data(data_list):
	t = (2020, 4, 10, 0, 0, 0, 0, 0, 0)
	time_end = time.mktime(t)
	data = pd.concat(data_list)
	data.sort_values(by=['user_id', 'qtime'], inplace=True)
	data['timestamp'] = data['qtime'] * time_end
	data['time_diff'] = data['qtime'].diff() * time_end
	first_rec = data.time_diff.isnull() | (data.time_diff < 0)
	data.loc[first_rec, 'time_diff'] = -1

	return data[data['time_diff'] != 0]

def load_data(data_path, phase, qtime=False):
    li = []
    for i in range(phase+1):
        filename = data_path % i
        print(filename)
        if not qtime:
            names = ['user_id', 'item_id', 'qtime']
        else:
            names = ['user_id', 'qtime']
        df = pd.read_csv(filename, header=None, names=names)
        li.append(df)

    df_full = pd.concat(li, axis=0, ignore_index=True)
    df_full.sort_values(by=['user_id', 'qtime'], inplace=True)
    return df_full


if __name__ == '__main__':
    train_data = load_data(opt.data.base_dir+opt.data.train_file, opt.phase)
