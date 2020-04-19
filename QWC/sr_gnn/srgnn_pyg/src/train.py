# -*- coding: utf-8 -*-
"""
Created on 5/4/2019
@author: RuihongQiu
"""


import numpy as np
import pandas as pd
import logging
import torch


def forward(model, loader, device, writer, epoch, lr_scheduler=None, top_k=20, optimizer=None, train_flag=True):
    if train_flag:
        model.train()
    else:
        model.eval()
        hit, mrr = [], []
    mean_loss = 0.0
    updates_per_epoch = len(loader)
    # target_list = []
    for i, batch in enumerate(loader):
        if train_flag:
            optimizer.zero_grad()
        scores = model(batch.to(device))
        targets = batch.y - 1
        loss = model.loss_function(scores, targets)

        if train_flag:
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * updates_per_epoch + i)
        else:
            sub_scores = scores.topk(top_k)[1]    # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                # target_list.append(target)
                hit.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == target)[0][0] + 1))

        mean_loss += loss / batch.num_graphs

    if train_flag:
        writer.add_scalar('loss/train_loss', mean_loss.item(), epoch)
    else:
        writer.add_scalar('loss/test_loss', mean_loss.item(), epoch)
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        print("@@@:", hit, mrr)
        writer.add_scalar('index/hit', hit, epoch)
        writer.add_scalar('index/mrr', mrr, epoch)
        import pickle
        # pickle.dump(target_list, open("../test_target.txt", 'wb') )

def predict(model_path, loader, save_path, device, top_k=50):
    """(seq, user_id)"""
    model = torch.load(model_path) 
    model.eval()
    res = []
    for i, batch in enumerate(loader):
        scores = model(batch.to(device))
        targets = batch.y
        sub_scores = scores.topk(top_k)[1]    # batch * top_k
        for score, user_id in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
            score = score+1
            record = [int(user_id)] + score.tolist()
            res.append(record)
    sub_df = pd.DataFrame.from_records(res)
    sub_df.to_csv(save_path, header=None, index=False)
            
    
