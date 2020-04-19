# -*- coding: utf-8 -*-
"""
Created on 4/4/2019
@author: RuihongQiu
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import argparse
import logging
from tqdm import tqdm
from dataset import MultiSessionsGraph
from torch_geometric.data import DataLoader
from model import *
from train import forward, predict
from tensorboardX import SummaryWriter
import numpy as np
import datetime

# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='debias', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--top_k', type=int, default=50, help='top K indicator for evaluation')
parser.add_argument('--predict', type=bool, default=False, help='predict task')
parser.add_argument('--model_path', default=20, help='model to load')
opt = parser.parse_args()
logging.warning(opt)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cur_dir = os.getcwd()
    if opt.predict:
        save_dir = cur_dir + '/../result/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + ".csv"
        test_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset, phrase='predict')
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
        predict(opt.model_path, test_loader, save_path, device)
        return

    
    train_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset, phrase='train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset, phrase='test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    need_feild = {"lr", "epoch", "batch_size"}
    log_name = "".join([k+"_"+str(v) for k,v in opt.__dict__.items() if k in need_feild])
    log_dir = cur_dir + '/../log/' + str(opt.dataset) + '/' + log_name
    model_dir = cur_dir + '/../model/' + str(opt.dataset)
    model_path = cur_dir + '/../model/' + str(opt.dataset) + '/' + log_name + '.pth'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logging.warning('model save to {}'.format(log_dir))
    writer = SummaryWriter(log_dir)
    
    node_d = {'diginetica': 43097,  'yoochoose1_64': 37483,  'yoochoose1_4': 37483, 'debias': 117538}
    n_node =  node_d.get(opt.dataset, 309)
    model = GNNModel(hidden_size=opt.hidden_size, n_node=n_node).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    logging.warning(model)
    
    for epoch in tqdm(range(opt.epoch)):
        #scheduler.step()
        forward(model, train_loader, device, writer, epoch, scheduler, top_k=opt.top_k, optimizer=optimizer, train_flag=True)
        with torch.no_grad():
            forward(model, test_loader, device, writer, epoch, top_k=opt.top_k, train_flag=False)
    torch.save(model, model_path) 



if __name__ == '__main__':
    main()
