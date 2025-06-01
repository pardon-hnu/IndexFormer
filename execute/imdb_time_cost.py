#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import pandas as pd

print(torch.cuda.is_available())

from model.database_util import *
from model.util import Normalizer
from model.database_util import get_hist_file, get_job_table_sample
from model.model import IndexFormer
from model.dataset import PlanTreeDataset
from model.trainer import eval_workload, train,evaluate,predict
from model.src.encoding_predicates import *
from model.src.internal_parameters import *
from model.src.meta_info import *
import time
class Args:
    bs = 256
    lr = 0.0005
    epochs = 100
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    dropout_index = 0.3
    sch_decay = 0.6
    device = 'cuda:0'
    newpath = './results/cost/best_checkpoints_imdb/'
    to_predict = 'cost'
    dataset = 'imdb'
def imdb_time_cost_run(toe,pt_file):
    args = Args()
    if not os.path.exists(args.newpath):
        os.makedirs(args.newpath)
    hist_file = pd.DataFrame()
    cost_norm = Normalizer(-3.61192, 12.290855)
    card_norm = Normalizer(1,100)

    encoding_ckpt = torch.load('checkpoints/encoding.pt')
    encoding = encoding_ckpt['encoding']

    word_vectors = load_dictionary('checkpoints/wordvectors_updated.kv')

    from model.util import seed_everything
    seed_everything()

    model = IndexFormer(emb_size = args.embed_size ,
                        ffn_dim = args.ffn_dim, 
                        head_size = args.head_size, 
                        dropout = args.dropout, 
                        dropout_index = args.dropout_index,
                        n_layers = args.n_layers,
                        use_sample = True,
                        use_hist = True,
                        pred_hid = args.pred_hid
                    )

    _ = model.to(args.device)

    to_predict = 'cost'

    full_train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    for i in range(1,32):
        file = 'data/imdb_10/imdb_idx/output{}_withidx1.csv'.format(i)
        df = pd.read_csv(file)
        # 按比例拆分为训练集和测试集
        train_size = int(0.9 * len(df))
        full_train_df = full_train_df.append(df.iloc[:train_size], ignore_index=True)  # 前90%加入训练集
        val_df = val_df.append(df.iloc[train_size:], ignore_index=True)


    for i in range(1,32):
        file = 'data/imdb_10/imdb_query_plan/output{}.csv'.format(i)
        df = pd.read_csv(file)
        # 按比例拆分为训练集和测试集
        train_size = int(0.9 * len(df))
        full_train_df = full_train_df.append(df.iloc[:train_size], ignore_index=True)  # 前90%加入训练集
        val_df = val_df.append(df.iloc[train_size:], ignore_index=True)

    table_sample = list()
    train_ds = PlanTreeDataset(full_train_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample,word_vectors,args.dataset)
    val_ds = PlanTreeDataset(val_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample,word_vectors,args.dataset)

    val = False
    if toe == 'Training':
        val = False
    elif toe == 'Evaluation':
        val = True
        if pt_file == '':
            raise FileNotFoundError("no xxx.pt")

    if val == False:
        crit = nn.MSELoss()
        model, best_path = train(model, train_ds, val_ds, crit, cost_norm, args)
    else:
        model.load_state_dict(torch.load(pt_file)['model'],strict=False)
        _,_=evaluate(model, val_ds, args.bs, cost_norm, args.device, args.dataset,True)
        # 计算执行时间
        print("Done")
    methods = {
        'get_sample' : get_job_table_sample,
        'encoding': encoding,
        'cost_norm': cost_norm,
        'hist_file': hist_file,
        'model': model,
        'device': args.device,
        'bs': 512,
    }
