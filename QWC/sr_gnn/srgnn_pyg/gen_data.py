#encoding: utf-8
import pandas as pd
import os
import pickle
import time
from sklearn.model_selection import train_test_split

def arrange_all_list(li):
    X, y = [], []
    for i in range(1, len(li)):
        X.append(li[:i])
        y.append(li[i])
    return X[::-1], y[::-1]

def filter_entity_row(data):
    """删除完全一致的行"""
    t = (2020, 4, 10, 0, 0, 0, 0, 0, 0)
    time_end = time.mktime(t)
    data['time_diff'] = data['qtime'].diff() * time_end
    first_rec = data.time_diff.isnull() | (data.time_diff < 0)
    data.loc[first_rec, 'time_diff'] = -1
    return data[data['time_diff'] != 0]

def gen_data(dir_name, predict=False, drop_dup=True):    
    all_files = [dir_name+f for f in os.listdir(dir_name) if "csv" in f and 'click' in f]
    li = []
    for filename in all_files:
        print(filename)
        df = pd.read_csv(filename, header=None, names=['user_id', 'item_id', 'qtime'])
        li.append(df)

    df_full = pd.concat(li, axis=0, ignore_index=True)
    df_full.sort_values(by=['user_id', 'qtime'], inplace=True)
    print("before drop length:", len(df_full))
    if drop_dup:
        df_full = filter_entity_row(df_full)
        print("after drop entity rowlength:", len(df_full))
    X_all, y_all = [], []
    gp = df_full.groupby('user_id')['item_id'].apply(list)
    for i, v in gp.items():
        if predict:
            X_all.append(v)
            y_all.append(i)
        else:
            X, y = arrange_all_list(v)
            X_all += X
            y_all += y
    return (X_all, y_all)

if __name__ == '__main__':
    train_dir = "../../../data/underexpose_train/"
    test_dir = "../../../data/underexpose_test/"
    dest_dir = "datasets/debias/raw/"
    delete_dir = "datasets/debias/processed/"
    if os.path.exists(delete_dir):
        os.system("rm -rf "+delete_dir)

    train_data = gen_data(train_dir)
    test_data = gen_data(test_dir)
    predict_data = gen_data(test_dir, predict=True)
    
    X = train_data[0] + test_data[0]
    y = train_data[1] + test_data[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    train_data = (X_train, y_train)
    test_data = (X_test, y_test)

    print("train_data:{}, test_data:{}, predict_data:{}".format( len(train_data[0]), len(test_data[0]), len(predict_data[0])))
    save_names = ['train.txt', 'test.txt', 'predict.txt']
    for data, file in zip([train_data, test_data, predict_data], save_names):
        with open(dest_dir+file, 'wb') as handle:
            pickle.dump(data, handle)