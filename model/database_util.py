import numpy as np
import pandas as pd
import csv
import torch
import re
from .src.encoding_predicates import *
from .src.internal_parameters import *
from .src.meta_info import *
## bfs shld be enough
def floyd_warshall_rewrite(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    M = adjacency_matrix.copy().astype('long')
    for i in range(nrows):
        for j in range(ncols):
            if i == j: 
                M[i][j] = 0
            elif M[i][j] == 0: 
                M[i][j] = 60
    
    for k in range(nrows):
        for i in range(nrows):
            for j in range(nrows):
                M[i][j] = min(M[i][j], M[i][k]+M[k][j])
    return M

def get_job_table_sample(workload_file_name, num_materialized_samples = 1000):

    tables = []
    samples = []

    # Load queries
    with open(workload_file_name + ".csv", 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))

            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)

    print("Loaded queries with len ", len(tables))
    
    # Load bitmaps
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(workload_file_name + ".bitmaps", 'rb') as f:
        for i in range(len(tables)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
            bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps_curr_query):
                # Read bitmap
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps")
    table_sample = []
    for ts, ss in zip(tables,samples):
        d = {}
        for t, s in zip(ts,ss):
            tf = t.split(' ')[0] # remove alias
            d[tf] = s
        table_sample.append(d)
    
    return table_sample


def get_hist_file(hist_path, bin_number = 50):
    hist_file = pd.read_csv(hist_path)
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        freq_np = np.frombuffer(bytes.fromhex(freq), dtype=np.float)
        hist_file['freq'][i] = freq_np

    table_column = []
    for i in range(len(hist_file)):
        table = hist_file['table'][i]
        col = hist_file['column'][i]
        table_alias = ''.join([tok[0] for tok in table.split('_')])
        if table == 'movie_info_idx': table_alias = 'mi_idx'
        combine = '.'.join([table_alias,col])
        table_column.append(combine)
    hist_file['table_column'] = table_column

    for rid in range(len(hist_file)):
        hist_file['bins'][rid] = \
            [int(i) for i in hist_file['bins'][rid][1:-1].split(' ') if len(i)>0]

    if bin_number != 50:
        hist_file = re_bin(hist_file, bin_number)

    return hist_file

def re_bin(hist_file, target_number):
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        bins = freq2bin(freq,target_number)
        hist_file['bins'][i] = bins
    return hist_file

def freq2bin(freqs, target_number):
    freq = freqs.copy()
    maxi = len(freq)-1
    
    step = 1. / target_number
    mini = 0
    while freq[mini+1]==0:
        mini+=1
    pointer = mini+1
    cur_sum = 0
    res_pos = [mini]
    residue = 0
    while pointer < maxi+1:
        cur_sum += freq[pointer]
        freq[pointer] = 0
        if cur_sum >= step:
            cur_sum -= step
            res_pos.append(pointer)
        else:
            pointer += 1
    
    if len(res_pos)==target_number: res_pos.append(maxi)
    
    return res_pos



class Batch():
    def __init__(self, attn_bias, rel_pos, heights, x, matrix,y=None):
        super(Batch, self).__init__()

        self.heights = heights
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.rel_pos = rel_pos
        self.matrix = matrix
        
    def to(self, device):

        self.heights = self.heights.to(device)
        self.x = self.x.to(device)

        self.attn_bias, self.rel_pos = self.attn_bias.to(device), self.rel_pos.to(device)
        self.matrix = [tensor.to(device) for tensor in self.matrix]

        return self

    def __len__(self):
        return self.in_degree.size(0)


def pad_1d_unsqueeze(x, padlen):
    x = x + 1 # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    elif xlen > padlen:
        x = x[:padlen]
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    # dont know why add 1, comment out first
#    x = x + 1 # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) + 1
        new_x[:xlen, :] = x
        x = new_x
    elif xlen > padlen:
        x = x[:padlen, :]
    return x.unsqueeze(0)

def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    elif xlen > padlen:
        x = x[:padlen, :padlen]
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    elif xlen > padlen:
        x = x[:padlen, :padlen]
    return x.unsqueeze(0)


def collator(small_set):
    y = small_set[1]
    xs = [s['x'] for s in small_set[0]]

    num_graph = len(y)
    x = torch.cat(xs)
    attn_bias = torch.cat([s['attn_bias'] for s in small_set[0]])
    rel_pos = torch.cat([s['rel_pos'] for s in small_set[0]])
    heights = torch.cat([s['heights'] for s in small_set[0]])
    matrix = small_set[2]

    return Batch(attn_bias, rel_pos, heights, x, matrix), y

def filterDict2Hist(hist_file, filterDict, encoding):
    buckets = len(hist_file['bins'][0]) 
    empty = np.zeros(buckets - 1)
    ress = np.zeros((3, buckets-1))
    for i in range(len(filterDict['colId'])):
        # 相当于看有多少个filter
        colId = filterDict['colId'][i]
        col = encoding.idx2col[colId]
        if col == 'NA':
            ress[i] = empty
            continue
        bins = hist_file.loc[hist_file['table_column']==col,'1'].item()
        # 应该是取第i个才对
        # opId = filterDict['opId'][0]
        opId = filterDict['opId'][i]
        op = encoding.idx2op[opId]
        
        # val = filterDict['val'][0]
        val = filterDict['val'][i]
        mini, maxi = encoding.column_min_max_vals[col]
        val_unnorm = val * (maxi-mini) + mini
        
        left = 0
        right = len(bins)-1
        for j in range(len(bins)):
            if bins[j]<val_unnorm:
                left = j
            if bins[j]>val_unnorm:
                right = j
                break

        res = np.zeros(len(bins)-1)

        if op == '=':
            res[left:right] = 1
        elif op == '<':
            res[:left] = 1
        elif op == '>':
            res[right:] = 1
        ress[i] = res
    
    ress = ress.flatten()
    return ress
# 就是计算这个值在哪个范围内然后计算频次返回一个扁平化的矩阵



def formatJoin(json_node):
    join = None
    if 'Hash Cond' in json_node:
        join = json_node['Hash Cond']
    # elif 'Join Filter' in json_node:
    #     join = json_node['Join Filter']
    elif 'filters' in json_node:
        join = json_node['filters']
    ## TODO: index cond
    elif 'Index Cond' in json_node and not json_node['Index Cond'][-2].isnumeric():
        join = json_node['Index Cond']
    ## sometimes no alias, say t.id 
    ## remove repeat (both way are the same)
    if join is not None:
        pattern = re.compile(r'\[(.*?)\]')
        # 使用正则表达式分割字符串
        filter_segments = pattern.findall(join)
        for segment in filter_segments:
            segment=segment.strip()
            # 构造正则表达式模式
            pattern = re.compile(r'\b([a-zA-Z_]+\.[a-zA-Z_]+)\s*=\s*([a-zA-Z_]+\.[a-zA-Z_]+)\b')
            # 查找匹配的子表达式
            matches = pattern.findall(segment)
            if len(matches) != 0:
                for match in matches:
                   join = match[0]+" = "+match[1]
                   return join
    join = None
    return join
    
def formatFilter(plan):
    alias = None
    # if 'Alias' in plan:
    #     alias = plan['Alias']
    # else:
    #     pl = plan
    #     while 'parent' in pl:
    #         pl = pl['parent']
    #         if 'Alias' in pl:
    #             alias = pl['Alias']
    #             break
    if 'NAME' in plan and plan['NAME']:
        alias = plan['NAME']
    else:
        pl = plan
        while 'parent' in pl:
            pl = pl['parent']
            if 'NAME' in pl and pl['NAME']:
                alias = pl['NAME']
                break
    
    filters = []
    # if 'Filter' in plan:
    #     filters.append(plan['Filter'])
    if 'filters' in plan:
        filters.append(plan['filters'])
    if 'Index Cond' in plan and plan['Index Cond'][-2].isnumeric():
        filters.append(plan['Index Cond'])
    if 'Recheck Cond' in plan:
        filters.append(plan['Recheck Cond'])
        
    
    
    return filters, alias

class TreeNodeforfilter:
    def __init__(self, value):
        self.value = value
        self.children = []

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.value) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret


def find_matching_parenthesis(expression, start_index):
    count = 0
    for i in range(start_index, len(expression)):
        if expression[i] == '(':
            count += 1
        elif expression[i] == ')':
            count -= 1
        if count == 0:
            return i
    raise ValueError("No matching parenthesis found")


def split_expressions(expression):
    parts = []
    bracket_level = 0
    current_part = ""
    for char in expression:
        if char == '(':
            current_part += char
            bracket_level += 1
            continue
        elif char == ')':
            current_part += char
            bracket_level -= 1
            continue
        if char == ',' and bracket_level == 0:
            parts.append(current_part.strip())
            current_part = ""
            continue
        current_part += char
    parts.append(current_part.strip())
    return parts

def encode_condition_op(right_value, word_vectors):
    right_value_vec = get_str_representation(right_value, word_vectors).tolist()
    #     print 'condition op: ', result
    return right_value_vec

def split_and_expressions(expression):
    return [part.strip() for part in expression.split(' AND ')]

def split_or_expressions(expression):
    return [part.strip() for part in expression.split(' OR ')]

def build_tree(expression,word_vectors):
    expression = expression.strip()
    sub_expressions = []
    current_node = TreeNodeforfilter('None')

    # 仅去除首尾的第一个括号
    if expression[0] == '(' and expression[-1] == ')':
       expression = expression[1:-1].strip()
    # 按照第一个逗号分割运算符和剩余表达式
       operator, rest = expression.split(',', 1)
       operator = operator.strip()
    # 创建当前节点
       current_node = TreeNodeforfilter(operator)

    # 解析剩余表达式
       rest = rest.strip()
       sub_expressions = split_expressions(rest)
    elif ' AND ' in expression:
       current_node  = TreeNodeforfilter('AND')
       sub_expressions = split_and_expressions(expression)
    elif ' OR ' in expression:
       current_node = TreeNodeforfilter('OR')
       sub_expressions = split_or_expressions(expression)
    # 分割子表达式

    for sub_expression in sub_expressions:
        # 检查子表达式是否是嵌套的逻辑表达式
        if sub_expression.startswith('('):
            child_node = build_tree(sub_expression,word_vectors)
            current_node.children.append(child_node)
        elif ' AND ' in sub_expression:
            # 创建AND节点
            and_node = TreeNodeforfilter('AND')
            and_sub_expressions = split_and_expressions(sub_expression)
            for and_sub_expression in and_sub_expressions:
                if and_sub_expression.startswith('('):
                    child_node = build_tree(and_sub_expression, word_vectors)
                    current_node.children.append(child_node)
                else:
                    and_sub_expression = encode_condition_op(and_sub_expression,word_vectors)
                    and_child_node = TreeNodeforfilter(and_sub_expression)
                    and_node.children.append(and_child_node)
            current_node.children.append(and_node)
        elif ' OR ' in sub_expression:
            # 创建AND节点
            or_node = TreeNodeforfilter('OR')
            or_sub_expressions = split_or_expressions(sub_expression)
            for or_sub_expression in or_sub_expressions:
                if or_sub_expression.startswith('('):
                    child_node = build_tree(or_sub_expression, word_vectors)
                    current_node.children.append(child_node)
                else:
                    or_sub_expression = encode_condition_op(or_sub_expression,word_vectors)
                    or_child_node = TreeNodeforfilter(or_sub_expression)
                    or_node.children.append(or_child_node)
            current_node.children.append(or_node)
        else:
            sub_expression_new = encode_condition_op(sub_expression,word_vectors)
            child_node = TreeNodeforfilter(sub_expression_new)
            current_node.children.append(child_node)
            # child_node = TreeNode(sub_expression)


    return current_node

def bitand(bit1, bit2):
    if len(bit1) > 0 and len(bit2) > 0:
        return [min(bit1[i], bit2[i]) for i in range(len(bit1))]
    elif len(bit1) > 0:
        return bit1
    elif len(bit2) > 0:
        return bit2
    else:
        return []

def bitor(bit1, bit2):
    if len(bit1) > 0 and len(bit2) > 0:
        return [max(bit1[i], bit2[i]) for i in range(len(bit1))]
    elif len(bit1) > 0:
        return bit1
    elif len(bit2) > 0:
        return bit2
    else:
        return []
def merge_tree(node):
    if node is None:
        return []

    # 如果是叶子节点，直接返回其向量
    if isinstance(node.value, list):
        return node.value

    # 递归处理所有子树
    child_vectors = [merge_tree(child) for child in node.children]

    # 根据操作符融合向量
    if node.value in ['T_OP_AND', 'AND']:
        result = child_vectors[0]
        for vec in child_vectors[1:]:
            result = bitand(result, vec)
    elif node.value in ['T_OP_OR', 'OR']:
        result = child_vectors[0]
        for vec in child_vectors[1:]:
            result = bitor(result, vec)
    else:
        result = np.zeros(1000)

    return result

class Encoding:
    def __init__(self, column_min_max_vals, 
                 col2idx, op2idx={'>':0, '=':1, '<':2, 'NA':3}):
        self.column_min_max_vals = column_min_max_vals
        self.col2idx = col2idx
        self.op2idx = op2idx
        
        idx2col = {}
        for k,v in col2idx.items():
            idx2col[v] = k
        self.idx2col = idx2col
        # self.idx2op = {0:'>', 1:'=', 2:'<', 3:'NA'}
        self.idx2op = {0: '>', 1: '=', 2: '<', 3: 'NA', 4: '>=', 5: '<='}

        self.type2idx = {}
        self.idx2type = {}
        self.join2idx = {}
        self.idx2join = {}
        
        self.table2idx = {'NA':0}
        self.idx2table = {0:'NA'}
    
    def normalize_val(self, column, val, log=False):
        mini, maxi = self.column_min_max_vals[column]
        
        val_norm = 0.0
        if maxi > mini:
            val_norm = (val-mini) / (maxi-mini)
        return val_norm
    
#     def encode_filters(self, filters=[], alias=None):
#         ## filters: list of dict
#
# #        print(filt, alias)
#         if len(filters) == 0:
#             return {'colId':[self.col2idx['NA']],
#                    'opId': [self.op2idx['NA']],
#                    'val': [0.0]}
#         res = {'colId':[],'opId': [],'val': []}
#         for filt in filters:
#             filt = ''.join(c for c in filt if c not in '()')
#             fs = filt.split(' AND ')
#             for f in fs:
#      #           print(filters)
#                 col, op, num = f.split(' ')
#                 column = alias + '.' + col
#     #            print(f)
#
#                 res['colId'].append(self.col2idx[column])
#                 res['opId'].append(self.op2idx[op])
#                 res['val'].append(self.normalize_val(column, float(num)))
#         return res
    def encode_filters(self, filters=[], word_vectors=None, alias=None):
        ## filters: list of dict
        if len(filters) == 0:
            return np.zeros((1000, 1))
        res = np.zeros((1000, 0))
        # 定义匹配中括号内内容的正则表达式
        pattern = re.compile(r'\[(.*?)\]')
        # 直接存入列表
        for filt in filters:
            # 使用正则表达式分割字符串
            filter_segments = pattern.findall(filt)

            for segment in filter_segments:
                segment=segment.strip()
                if " OR " not in segment and " AND " not in segment and "T_OP_OR" not in segment and "T_OP_AND" not in segment:
                    merged_vector = encode_condition_op(segment, word_vectors)
                else:
                    tree = build_tree(segment, word_vectors)
                    merged_vector = merge_tree(tree)

                # 将 merged_vector 添加为 res 的新列
                merged_vector = np.array(merged_vector)
                res = np.hstack((res, merged_vector.reshape(1000, 1)))
        return res
    
    def encode_join(self, join):
        if join not in self.join2idx:
            self.join2idx[join] = len(self.join2idx)
            self.idx2join[self.join2idx[join]] = join
        return self.join2idx[join]
    
    def encode_table(self, table):
        if table not in self.table2idx:
            self.table2idx[table] = len(self.table2idx)
            self.idx2table[self.table2idx[table]] = table
        return self.table2idx[table]

    def encode_type(self, nodeType):
        if nodeType not in self.type2idx:
            self.type2idx[nodeType] = len(self.type2idx)
            self.idx2type[self.type2idx[nodeType]] = nodeType
        return self.type2idx[nodeType]


class TreeNode:
    def __init__(self, nodeType, typeId, filt, card, join, join_str, filterDict):
        self.nodeType = nodeType
        self.typeId = typeId
        self.filter = filt
        
        self.table = 'NA'
        self.table_id = 0
        self.query_id = None ## so that sample bitmap can recognise
        
        self.join = join
        self.join_str = join_str
        self.card = card #'Actual Rows'
        self.children = []
        self.rounds = 0
        
        self.filterDict = filterDict
        
        self.parent = None
        
        self.feature = None
        
    def addChild(self,treeNode):
        self.children.append(treeNode)
    
    def __str__(self):
#        return TreeNode.print_nested(self)
        return '{} with {}, {}, {} children'.format(self.nodeType, self.filter, self.join_str, len(self.children))

    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def print_nested(node, indent = 0): 
        print('--'*indent+ '{} with {} and {}, {} childs'.format(node.nodeType, node.filter, node.join_str, len(node.children)))
        for k in node.children: 
            TreeNode.print_nested(k, indent+1)
        





