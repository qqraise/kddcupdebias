#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
  
import pandas as pd  
from tqdm import tqdm  
from collections import defaultdict  
import math  
  
  
def get_sim_item(df, user_col, item_col, use_iif=False):  
    user_item_ = df.groupby(user_col)[item_col].agg(set).reset_index()  
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))  
  
    sim_item = {}  
    item_cnt = defaultdict(int)  
    for user, items in tqdm(user_item_dict.items()):  
        for i in items:  
            item_cnt[i] += 1  
            sim_item.setdefault(i, {})  
            for relate_item in items:  
                if i == relate_item:  
                    continue  
                sim_item[i].setdefault(relate_item, 0)  
                if not use_iif:  
                    sim_item[i][relate_item] += 1  
                else:  
                    sim_item[i][relate_item] += 1 / math.log(1 + len(items))  
    sim_item_corr = sim_item.copy()  
    for i, related_items in tqdm(sim_item.items()):  
        for j, cij in related_items.items():  
            sim_item_corr[i][j] = cij/math.sqrt(item_cnt[i]*item_cnt[j])  
  
    return sim_item_corr, user_item_dict  
  
  
def recommend(sim_item_corr, user_item_dict, user_id, top_k, item_num):  
    rank = {}  
    interacted_items = user_item_dict[user_id]  
    for i in interacted_items:  
        for j, wij in sorted(sim_item_corr[i].items(), reverse=True)[0:top_k]:  
            if j not in interacted_items:  
                rank.setdefault(j, 0)  
                rank[j] += wij  
    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]  
  
  
# fill user to 50 items  
def get_predict(df, pred_col, top_fill):  
    top_fill = [int(t) for t in top_fill.split(',')]  
    scores = [-1 * i for i in range(1, len(top_fill) + 1)]  
    ids = list(df['user_id'].unique())  
    fill_df = pd.DataFrame(ids * len(top_fill), columns=['user_id'])  
    fill_df.sort_values('user_id', inplace=True)  
    fill_df['item_id'] = top_fill * len(ids)  
    fill_df[pred_col] = scores * len(ids)  
    df = df.append(fill_df)  
    df.sort_values(pred_col, ascending=False, inplace=True)  
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')  
    df['rank'] = df.groupby('user_id')[pred_col].rank(method='first', ascending=False)  
    df = df[df['rank'] <= 50]  
    df = df.groupby('user_id')['item_id'].apply(lambda x: ','.join([str(i) for i in x])).str.split(',', expand=True).reset_index()  
    return df  
  
  
if __name__ == "__main__":  
    now_phase = 2  
    train_path = 'data/underexpose_train'  
    test_path = 'data/underexpose_test'  
    recom_item = []  
  
    whole_click = pd.DataFrame()  
    for c in range(now_phase + 1):  
        print('phase:', c)  
        click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
        click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
  
        all_click = click_train.append(click_test)  
        whole_click = whole_click.append(all_click)  
        item_sim_list, user_item = get_sim_item(all_click, 'user_id', 'item_id', use_iif=True)  
  
        for i in tqdm(click_test['user_id'].unique()):  
            rank_item = recommend(item_sim_list, user_item, i, 500, 50)  
            for j in rank_item:  
                recom_item.append([i, j[0], j[1]])  
    # find most popular items  
    top50_click = whole_click['item_id'].value_counts().index[:50].values  
    top50_click = ','.join([str(i) for i in top50_click])  
  
    recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])  
    result = get_predict(recom_df, 'sim', top50_click)  
    result.to_csv('baseline.csv', index=False, header=None)
