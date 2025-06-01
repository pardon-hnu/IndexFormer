#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import pandas as pd

print(torch.cuda.is_available())

# In[2]:
# import Query_Encode as qe
# import Load_Data as ld
# import Plan_Encode as pe
# import EncodeUtils as eu

from model.database_util import *
from model.util import Normalizer
from model.database_util import get_hist_file, get_job_table_sample,Batch
from model.model import QueryFormer
from model.dataset import PlanTreeDataset
from model.trainer import eval_workload, train,evaluate
from model.src.encoding_predicates import *
from model.src.internal_parameters import *
from model.src.meta_info import *

data_path = './data/imdb/'



class Args:
    bs = 256
    lr = 0.0005
    epochs = 150
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
    newpath = './results/full/cost/'
    to_predict = 'cost'
    dataset = 'tpcds'
args = Args()

import os
if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)

hist_file = get_hist_file(data_path + 'histogram_string.csv')
cost_norm = Normalizer(-3.61192, 12.290855)
card_norm = Normalizer(1,100)

encoding_ckpt = torch.load('checkpoints/encoding.pt')
encoding = encoding_ckpt['encoding']
# query_loader=ld.QueryLoader("./data/imdb/train")
# query_loader.load_query()
# query_loader.load_minmax("./data/imdb/column_min_max_vals")
# query_loader.load_bitmap()
# query_loader.load_query_plan("./data/imdb/")
# query_loader.load_hist("./data/imdb/histogram_string.csv")
# query_encoder=qe.QueryEncoder(query_loader)
# query_encoder.encode()
# query_encoder.column2idx['NA']=0
# print(query_encoder.column2idx)
# encoding = Encoding(query_loader.column_min_max_vals,query_encoder.column2idx)

# checkpoint = torch.load('checkpoints/cost_model.pt', map_location='cpu')
word_vectors = load_dictionary('checkpoints/wordvectors_updated.kv')

from model.util import seed_everything
seed_everything()

model = QueryFormer(emb_size = args.embed_size ,ffn_dim = args.ffn_dim, head_size = args.head_size,                  dropout = args.dropout, dropout_index = args.dropout_index,n_layers = args.n_layers,                  use_sample = True, use_hist = True,                  pred_hid = args.pred_hid
                )

_ = model.to(args.device)

to_predict = 'cost'


imdb_path = './data/imdb/'
tpch_path = '/home/hnu/Disk0/MaYiProject/lzy_database_test/test_fordata/data/'
template_files = [f"/home/hnu/Disk0/MaYiProject/QueryFormer-test/QueryFormer/QueryFormer/data/tpcds/tpcds_idx_recommend/index{i}_with_query.csv" for i in range(1, 20)]
# 打开一个输出文件
output_file = '/home/hnu/Disk0/MaYiProject/QueryFormer-test/QueryFormer/QueryFormer/data/tpcds/tpcds_idx_recommend/predicted_best_templates.txt'
table_sample = get_job_table_sample(imdb_path+'train')
best_path="/home/hnu/Disk0/MaYiProject/QueryFormer-test/QueryFormer/QueryFormer/results/full/cost/7153848688270536440.pt"

model.load_state_dict(torch.load(best_path)['model'], strict=False)
model = model.to('cuda')

# 初始化 sql_times 字典
sql_times = {}

with open(output_file, 'w') as f:
    # 对每个模板执行预测
    for template_file in template_files:
        df = pd.read_csv(template_file)
        # 获取每条 SQL 的执行计划
        sql_ids = df['id']
        queries = df
        queries_input = PlanTreeDataset(queries, None, encoding, hist_file, card_norm, cost_norm, to_predict,
                                        table_sample, word_vectors, args.dataset)
        real_time = queries_input.costs
        # 记录每条 SQL 在当前模板的执行时间
        for idx,val in sql_ids.items():
            with torch.no_grad():
                y = real_time[idx]
                collated_dicts,_,matrix = queries_input[idx]
                xs = collated_dicts['x']
                attn_bias= collated_dicts['attn_bias']
                rel_pos= collated_dicts['rel_pos']
                heights= collated_dicts['heights']
                batch = Batch(attn_bias, rel_pos, heights, xs, matrix)
                batch = batch.to(args.device)
                cost_preds, _ = model(batch, args.dataset)
                predicted_time = cost_norm.unnormalize_labels(cost_preds.cpu()).item()
            # 在 sql_times 字典中为每条 SQL 记录时间
            if sql_ids[idx] not in sql_times:
                sql_times[sql_ids[idx]] = []
            # 存储模板文件名和预测时间
            sql_times[sql_ids[idx]].append((template_file, predicted_time, y))

    # 对每条 SQL 查询找到最短时间的模板
    for sql_id, times in sql_times.items():
        # 找到最短时间对应的模板
        best_template = min(times, key=lambda x: x[1])
        f.write(f"SQL {sql_id}时间最短的模板是 {best_template[0]}，预测时间是 {float(best_template[1]):.4f} 微秒,真是时间是 {float(best_template[2]):.4f} 微秒\n")
print("预测完成！")
