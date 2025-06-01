import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

class Prediction(nn.Module):
    def __init__(self, in_feature=69, hid_units=512, contract=1, mid_layers=True, res_con=True):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con

        self.out_mlp1 = nn.Linear(in_feature, hid_units)

        self.mid_mlp1 = nn.Linear(hid_units, hid_units // contract)
        self.mid_mlp2 = nn.Linear(hid_units // contract, hid_units)

        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features):

        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = torch.sigmoid(self.out_mlp2(hid))

        return out


class FeatureEmbed(nn.Module):
    def __init__(self, embed_size=32, tables=10, types=50, joins=40, columns=30, \
                 ops=4, use_sample=True, use_hist=False, bin_number=50):
        super(FeatureEmbed, self).__init__()

        self.use_sample = use_sample
        self.embed_size = embed_size

        self.use_hist = use_hist
        self.bin_number = bin_number

        self.typeEmbed = nn.Embedding(types, embed_size)
        self.tableEmbed = nn.Embedding(tables, embed_size)

        self.columnEmbed = nn.Embedding(columns, embed_size)
        self.opEmbed = nn.Embedding(ops, embed_size // 8)

        self.linearFilter2 = nn.Linear(embed_size + embed_size // 8 + 1, embed_size + embed_size // 8 + 1)
        # self.linearFilter = nn.Linear(embed_size + embed_size // 8 + 1, embed_size + embed_size // 8 + 1)
        self.linearFilter = nn.Linear(1000, embed_size + embed_size // 8 + 1)

        self.linearType = nn.Linear(embed_size, embed_size)

        self.linearJoin = nn.Linear(embed_size, embed_size)

        self.linearSample = nn.Linear(1000, embed_size)

        self.linearHist = nn.Linear(bin_number, embed_size)

        self.joinEmbed = nn.Embedding(joins, embed_size)

        # if use_hist:
        #     self.project = nn.Linear(embed_size * 5 + embed_size // 8 + 1, embed_size * 5 + embed_size // 8 + 1)
        # else:
        #     # self.project = nn.Linear(embed_size * 4 + embed_size // 8 + 1, embed_size * 4 + embed_size // 8 + 1)
        self.project = nn.Linear(265, 265)

    # input: B by 14 (type, join, f1, f2, f3, mask1, mask2, mask3)
    def forward(self, feature):

        # typeId, joinId, filtersId, filtersMask, hists, table_sample = torch.split(feature, (
        # 1, 1, 1000*10, 1000, self.bin_number * 1000, 1001), dim=-1)
        typeId, joinId, filtersId, filtersMask,table_sample = torch.split(feature, (
        1, 1, 1000*10, 10,1001), dim=-1)
        typeEmb = self.getType(typeId)
        joinEmb = self.getJoin(joinId)
        filterEmbed = self.getFilter(filtersId, filtersMask)

        # histEmb = self.getHist(hists, filtersMask)
        tableEmb = self.getTable(table_sample)

        # if self.use_hist:
        #     final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb, histEmb), dim=1)
        # else:
        #     final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb), dim=1)
        final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb), dim=1)

        final = F.leaky_relu(self.project(final))

        return final

    def getType(self, typeId):
        emb = self.typeEmbed(typeId.long())

        return emb.squeeze(1)

    def getTable(self, table_sample):
        table, sample = torch.split(table_sample, (1, 1000), dim=-1)
        emb = self.tableEmbed(table.long()).squeeze(1)

        if self.use_sample:
            emb += self.linearSample(sample)
        return emb

    def getJoin(self, joinId):

        emb = self.joinEmbed(joinId.long())

        return emb.squeeze(1)

    def getHist(self, hists, filtersMask):
        # batch * 50 * 3
        histExpand = hists.view(-1, self.bin_number, 3).transpose(1, 2)

        emb = self.linearHist(histExpand)
        emb[~filtersMask.bool()] = 0.  # mask out space holder

        ## avg by # of filters
        num_filters = torch.sum(filtersMask, dim=1)
        total = torch.sum(emb, dim=1)
        avg = total / num_filters.view(-1, 1)

        return avg

    def getFilter(self, filtersId, filtersMask):
        ## get Filters, then apply mask
        # filterExpand = filtersId.view(-1, 3, 3).transpose(1, 2)
        filterExpand = filtersId.reshape(-1, 1000, 10).transpose(1, 2)
        # colsId = filterExpand[:, :, 0].long()
        # opsId = filterExpand[:, :, 1].long()
        # vals = filterExpand[:, :, 2].unsqueeze(-1)  # b by 3 by 1
        #  colsId 是一个形状为 (batch_size, 3) 的 long 类型张量。
        # opsId 是一个形状为 (batch_size, 3) 的 long 类型张量。
        # vals 是一个形状为 (batch_size, 3, 1) 的张量。
        # b by 3 by embed_dim

        # col = self.columnEmbed(colsId)
        # op = self.opEmbed(opsId)
        # #
        # concat = torch.cat((col, op, vals), dim=-1)
        # # (batch_size, 3, 37)
        concat = F.leaky_relu(self.linearFilter(filterExpand))
        concat = F.leaky_relu(self.linearFilter2(concat))

        ## apply mask
        concat[~filtersMask.bool()] = 0.

        ## avg by # of filters
        num_filters = torch.sum(filtersMask, dim=1)
        total = torch.sum(concat, dim=1)
        avg = total / num_filters.view(-1, 1)

        return avg


#     def get_output_size(self):
#         size = self.embed_size * 5 + self.embed_size // 8 + 1
#         return size


class IndexFormer(nn.Module):
    def __init__(self, emb_size=32, ffn_dim=32, head_size=8, \
                 dropout=0.1,dropout_index=0.1, attention_dropout_rate=0.1, n_layers=8, \
                 use_sample=True, use_hist=False, bin_number=50, \
                 pred_hid=256
                 ):

        super(IndexFormer, self).__init__()
        if use_hist:
            hidden_dim = emb_size * 5 + emb_size // 8 + 1
        else:
            hidden_dim = emb_size * 4 + emb_size // 8 + 1
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist

        self.rel_pos_encoder = nn.Embedding(64, head_size, padding_idx=0)

        # self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        self.height_encoder = nn.Embedding(64, 265, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        self.input_dropout_index = nn.Dropout(dropout_index)
        # encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
        #             for _ in range(n_layers)]
        encoders = [EncoderLayer(265, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]

        ecoders_foridx = [EncoderLayer_foridx(265, ffn_dim ,dropout, attention_dropout_rate)
                          for _ in range(n_layers)]

        ecoders_foridx_fn = [EncoderLayer_foridx(265*2, ffn_dim ,dropout, attention_dropout_rate)
                          for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.layers_foridx = nn.ModuleList(ecoders_foridx)
        self.layers_foridx_fn = nn.ModuleList(ecoders_foridx_fn)
        # self.final_ln = nn.LayerNorm(hidden_dim)
        self.final_ln = nn.LayerNorm(265)

        self.final_ln_foridx = nn.LayerNorm(265)
        self.final_ln_foridx_fn = nn.LayerNorm(265*2)
        # self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token = nn.Embedding(1, 265)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)

        self.embbed_layer = FeatureEmbed(emb_size, use_sample=use_sample, use_hist=use_hist, bin_number=bin_number)

        # self.pred = Prediction(hidden_dim, pred_hid)
        self.pred = Prediction(265*2, pred_hid)
        self.simple_nn = SimpleNN()
        self.simple_nn_imdb = SimpleNN_imdb()
        self.simple_nn_tpcds = SimpleNN_tpcds()
        # if multi-task
        # self.pred2 = Prediction(hidden_dim, pred_hid)
        self.pred2 = Prediction(265*2, pred_hid)

    def forward(self, batched_data,dataset_name):
        matrix,  attn_bias, rel_pos, x =batched_data.matrix, batched_data.attn_bias, batched_data.rel_pos, batched_data.x

        heights = batched_data.heights

        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)

        # rel pos
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1,
                                                             2)  # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias

        # reset rel pos here
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t

        x_view = x.view(-1,11013)
        # node_feature = self.embbed_layer(x_view).view(n_batch, -1, self.hidden_dim)
        node_feature = self.embbed_layer(x_view).view(n_batch, -1, 265)
        # -1 is number of dummy

        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)

        matrix = torch.stack(matrix, dim=0)
        # if dataset_name == 'tpch':
        #     matrix_feature = matrix.view(-1,64*138)
        #     nnmodel = SimpleNN(64*138)
        #     nnmodel.to("cuda:0")
        #     output_idx = nnmodel(matrix_feature)
        # elif dataset_name == 'imdb':
        #     matrix_feature = matrix.view(-1,64*129*2)
        #     nnmodel = SimpleNN(64*129*2)
        #     nnmodel.to("cuda:0")
        #     output_idx = nnmodel(matrix_feature)
        if dataset_name == 'tpch':
            matrix_feature = matrix.view(-1, 64*138)
            output_idx = self.simple_nn(matrix_feature)
        elif dataset_name == 'imdb':
            matrix_feature = matrix.view(-1, 64*129*2)
            output_idx = self.simple_nn_imdb(matrix_feature)
        elif dataset_name == 'tpcds':
            matrix_feature = matrix.view(-1, 64*449*2)
            output_idx = self.simple_nn_tpcds(matrix_feature)
        output_idx = output_idx.unsqueeze(1)

        # transfomrer encoder
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)

        # output_idx = output_idx.unsqueeze(1)
        # # output_idx = self.input_dropout(output_idx)
        # # # 尝试一下drop out是否可行
        # for enc_foridx_layer in self.layers_foridx:
        #     output_idx = enc_foridx_layer(output_idx)
        # output_idx = self.final_ln_foridx(output_idx)

        output_concat = torch.cat([output[:, 0, :], output_idx[:, 0, :]], dim=1)
        # return self.pred(output[:, 0, :]), self.pred2(output[:, 0, :])

        output_concat = output_concat.unsqueeze(1)
        output_concat = self.input_dropout_index(output_concat)
        for enc_foridx_layer in self.layers_foridx_fn:
            output_concat = enc_foridx_layer(output_concat)
        output_concat = self.final_ln_foridx_fn(output_concat)
        
        return self.pred(output_concat), self.pred2(output_concat)

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class Self_Attention(nn.Module):

    def __init__(self, dim_q, dim_k, dim_v):
        super(Self_Attention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_q
        # 根据文本获得相应的维度

        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        # q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        # 归一化获得attention的相关系数
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        # attention系数和v相乘，获得最终的得分
        att = torch.bmm(dist, v)
        return att


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class EncoderLayer_foridx(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate,attention_dropout_rate):
        super(EncoderLayer_foridx, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = Self_Attention(hidden_size, hidden_size, hidden_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        y = self.self_attention_norm(x)
        y = self.self_attention(y)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(64*138, 64*256)
        self.fc2 = nn.Linear(64*256, 32*32)
        self.fc3 = nn.Linear(32*32, 265)

    def forward(self, x):
        x = x.view(-1, 64*138)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

class SimpleNN_imdb(nn.Module):
    def __init__(self):
        super(SimpleNN_imdb, self).__init__()
        self.fc1 = nn.Linear(64*129*2, 64*256)
        self.fc2 = nn.Linear(64*256, 32*32)
        self.fc3 = nn.Linear(32*32, 265)

    def forward(self, x):
        x = x.view(-1, 64*129*2)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

class SimpleNN_tpcds(nn.Module):
    def __init__(self):
        super(SimpleNN_tpcds, self).__init__()
        self.fc1 = nn.Linear(64*449*2, 64*256)
        self.fc2 = nn.Linear(64*256, 32*32)
        self.fc3 = nn.Linear(32*32, 265)

    def forward(self, x):
        x = x.view(-1, 64*449*2)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x