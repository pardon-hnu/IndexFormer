from torch.utils.data import Dataset
import json
from collections import deque
from .database_util import *
import re
import ast
import numpy as np
import torch
import torch.nn.functional as F

def concatenate_matrix(matrix1, matrix2):
    matrix = []
    if matrix1 is not None and matrix2 is not None:
        matrix1_list = ast.literal_eval(matrix1)
        matrix2_list = ast.literal_eval(matrix2)
        matrix1_m = np.array(matrix1_list)
        matrix2_m = np.array(matrix2_list)
        matrix_m = np.concatenate((matrix1_m, matrix2_m), axis=1)
        matrix_m = matrix_m.astype(np.float32)
        matrix = torch.from_numpy(matrix_m)
    else:
        matrix_m= np.zeros((64, 138), dtype=np.float32)
        matrix = torch.from_numpy(matrix_m)

    return matrix

def concatenate_matrix_imdb(matrix1, matrix2):
    matrix = []
    if matrix1 is not None and matrix2 is not None:
        matrix1_list = ast.literal_eval(matrix1)
        matrix2_list = ast.literal_eval(matrix2)
        matrix1_m = np.array(matrix1_list)
        matrix2_m = np.array(matrix2_list)
        matrix_m = np.concatenate((matrix1_m, matrix2_m), axis=1)
        matrix_m = matrix_m.astype(np.float32)
        matrix = torch.from_numpy(matrix_m)
        # 检查并补零
        target_shape = (64, 258)
        current_shape = matrix.shape

        if current_shape != torch.Size(target_shape):
            padding_cols = target_shape[1] - current_shape[1]  # 需要补的列数
            if padding_cols > 0:
                matrix = F.pad(matrix, (0, padding_cols))  # 在最后一列方向补零
    else:
        matrix_m= np.zeros((64, 129*2), dtype=np.float32)
        matrix = torch.from_numpy(matrix_m)

    return matrix

def concatenate_matrix_tpcds(matrix1, matrix2):
    matrix = []
    if matrix1 is not None and matrix2 is not None:
        matrix1_list = ast.literal_eval(matrix1)
        matrix2_list = ast.literal_eval(matrix2)
        matrix1_m = np.array(matrix1_list)
        matrix2_m = np.array(matrix2_list)
        matrix_m = np.concatenate((matrix1_m, matrix2_m), axis=1)
        matrix_m = matrix_m.astype(np.float32)
        matrix = torch.from_numpy(matrix_m)
        # 检查并补零
        target_shape = (64, 449*2)
        current_shape = matrix.shape

        if current_shape != torch.Size(target_shape):
            padding_cols = target_shape[1] - current_shape[1]  # 需要补的列数
            if padding_cols > 0:
                matrix = F.pad(matrix, (0, padding_cols))  # 在最后一列方向补零
    else:
        matrix_m= np.zeros((64, 449*2), dtype=np.float32)
        matrix = torch.from_numpy(matrix_m)

    return matrix

class PlanTreeDataset(Dataset):
    def __init__(self, json_df: pd.DataFrame, train: pd.DataFrame, encoding, hist_file, card_norm, cost_norm,
                 to_predict, table_sample, word_vectors,dataset_name):

        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file

        self.length = len(json_df)
        # train = train.loc[json_df['id']]
        self.costs2 = [json.loads(plan)['Plan']['EST.TIME(us)'] for plan in json_df['json']]
        nodes = [json.loads(plan)['Plan'] for plan in json_df['json']]
        # self.cards = [node['Actual Rows'] for node in nodes]
        self.cards = [node['EST.ROWS'] for node in nodes]
        # self.costs = [json.loads(plan)['Execution Time'] for plan in json_df['json']]
        self.costs = [json.loads(plan)['time'] for plan in json_df['json']]
        self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards))
        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))
        # get matrix

        self.matrix = []
        for data in json_df['json']:
            plan_data =json.loads(data)
            matrix1=plan_data.get('matrix1',None)
            matrix2=plan_data.get('matrix2',None)
            if dataset_name == 'tpch':
                matrix_m = concatenate_matrix(matrix1, matrix2)
            elif dataset_name == 'imdb':
                matrix_m = concatenate_matrix_imdb(matrix1, matrix2)
            elif dataset_name == 'tpcds':
                matrix_m = concatenate_matrix_tpcds(matrix1, matrix2)
            self.matrix.append(matrix_m)

        self.to_predict = to_predict
        if to_predict == 'cost':
            self.gts = self.costs
            self.labels = self.cost_labels
        elif to_predict == 'card':
            self.gts = self.cards
            self.labels = self.card_labels
        elif to_predict == 'both':  ## try not to use, just in case
            self.gts = self.costs
            self.labels = self.cost_labels
        else:
            raise Exception('Unknown to_predict type')

        idxs = list(json_df['id'])

        self.treeNodes = []  ## for mem collection
        self.collated_dicts = [self.js_node2dict(i, node, word_vectors) for i, node in zip(idxs, nodes)]

    def js_node2dict(self, idx, node, word_vectors):
        treeNode = self.traversePlan(node, idx, self.encoding, word_vectors)
        _dict = self.node2dict(treeNode)
        # 字典包含了三个元素'features','heights','adjacency_list'
        collated_dict = self.pre_collate(_dict)

        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        return self.collated_dicts[idx], (self.cost_labels[idx], self.card_labels[idx]), self.matrix[idx]

    def old_getitem(self, idx):
        return self.dicts[idx], (self.cost_labels[idx], self.card_labels[idx])

    ## pre-process first half of old collator
    def pre_collate(self, the_dict, max_node=40, rel_pos_max=30):

        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)

        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N, N], dtype=torch.bool)
            adj[edge_index[0, :], edge_index[1, :]] = True

            shortest_path_result = floyd_warshall_rewrite(adj.numpy())

        rel_pos = torch.from_numpy((shortest_path_result)).long()

        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')

        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)

        return {
            'x': x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }

    def node2dict(self, treeNode):

        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features': torch.FloatTensor(features),
            'heights': torch.LongTensor(heights),
            'adjacency_list': torch.LongTensor(np.array(adj_list)),

        }
    # features应该是代码中主要修改的部分，heights包含了树中每个节点的高度，adjacency_list相当于访问顺序，表示从什么结点开始接下来是什么

    def topo_sort(self, root_node):
        #        nodes = []
        adj_list = []  # from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0, root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
            #            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id, child))
                adj_list.append((idx, next_id))
                next_id += 1

        return adj_list, num_child, features

    def traversePlan(self, plan, idx, encoding, word_vectors):  # bfs accumulate plan

        # nodeType = plan['Node Type']
        nodeType = plan['OPERATOR']
        typeId = encoding.encode_type(nodeType)
        card = None  # plan['Actual Rows']
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, word_vectors, alias)

        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)

        self.treeNodes.append(root)

        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx

        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample)
        pattern = re.compile(r'CHILD_\d+')
        for key, value in plan.items():
            if pattern.match(key):
                value['parent'] = plan
                node = self.traversePlan(value, idx, encoding, word_vectors)
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list, tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:, 0]
        child_nodes = adj_list[:, 1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order


def node2feature(node, encoding, hist_file, table_sample):
    # type, join, filter123, mask123
    # 1, 1, 3x3 (9), 3
    # TODO: add sample (or so-called table)
    # num_filter = len(node.filterDict['colId'])
    num_filter = node.filterDict.shape[1]
    if num_filter > 10:
        filts = np.array(node.filterDict)[:, :10]  # 截取前10列
    else:
        pad = np.zeros((1000, 10 - num_filter))  # 填充
        filts = np.array(node.filterDict)
        filts = np.concatenate((filts, pad), axis=1)

    filts = filts.flatten()  # 将其展平成一维数组
    mask = np.zeros(10)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])
    # 
    # hists = filterDict2Hist(hist_file, node.filterDict, encoding)

    # table, bitmap, 1 + 1000 bits
    table = np.array([node.table_id])
    # if node.table_id == 0:
    #     sample = np.zeros(1000)
    # else:
    #     sample = table_sample[node.query_id][node.table]
    sample = np.zeros(1000)
    # return np.concatenate((type_join,filts,mask))
    # return np.concatenate((type_join, filts, mask, hists, table, sample))
    return np.concatenate((type_join, filts, mask, table, sample))
