import operator
from collections import defaultdict
import pickle
import sys
import logging
import numpy as np
import json
import argparse
parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-param_file', action='store', dest='param_file',
                    help='param file file')
parser.add_argument('-output_path', action='store', dest='output_path',
                    help='input file path')
parser.add_argument('-input_path', action='store', dest='input_path',
                    help='input file path')
parser.add_argument('-edge_file', action='store', dest='edge_file',
                    help='input file path')     
parser.add_argument('-input_path1', action='store', dest='input_path1',
                    help='input file path')

args = parser.parse_args()
# args = parser.parse_args([
#     '-edge_file', './data/edges.npy',
#     '-input_path', './raw_data/vertex.csv',
#     '-input_path1', './raw_data/edge.csv',
#     '-output', './data/vertex.npy',
#     '-param', './data/vertex.params.json'
# ])

logger = logging.getLogger("message")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('./message.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 获取所有的 key:value 值, 同时保存特定type 的最小值
def get_dict_allkeys_values(dict_a,values,mins):
        for x in range(len(dict_a)):
            temp_key = list(dict_a.keys())[x]
            temp_value = dict_a[temp_key]
            if temp_key=='timestamp':
                if int(dict_a[temp_key])<=mins[0]:
                    mins[0]=int(dict_a[temp_key])
            elif temp_key=='src' or temp_key=='dst':
                temp_key = 'uuid'
                if temp_key not in values.keys():
                    values[temp_key]={}
                if str(temp_value) not in values[temp_key].keys():
                    values[temp_key][str(temp_value)]=len(values[temp_key].keys())
            elif temp_key=='operation':
                if temp_key not in values.keys():
                    values[temp_key]={}
                if str(temp_value) not in values[temp_key].keys():
                    values[temp_key][str(temp_value)]=len(values[temp_key].keys())
        return values
from collections import Counter
import re
import time
import sys

id2char_dict={}
char2id_dict={}
import csv
import copy
import numpy as np
def count():
    reader=[]
    data=[]
    mins=[sys.maxsize]
    values={}
    with open(args.input_path1, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i in reader:
            data.append(i)
    key=data[0]
    data=data[1:]
    for i in range(len(data)):
        json_obj={}
        tmpdata=data[i]
        for j in range(len(key)):  # -1 -> -2，不再记录 src,dst（已经在 uuid 中记录）
            if tmpdata[j]!='':
                json_obj[key[j]]=tmpdata[j]
        values=get_dict_allkeys_values(json_obj,values,mins)
    print('end_edge')
    return values,mins

key_template_dict={}
# 对边进行编码
def handle_normal(json_obj,char2id_dict,id2char_dict,mins,re_values,key_template_dict,edges,flag=0):
    data_processed_=[]
    temp_value=''
    tmplist=list(json_obj.keys())
    if 'operation' in tmplist:  # event
        temp_key='operation'
        temp_value='eventid:'+str(len(edges[0]))
        child=re_values['uuid'][json_obj['dst']]
        parent=re_values['uuid'][json_obj['src']]
        edges[0].append(child)  # 如果是边就存在 edges[0][1]中
        edges[1].append(parent)
    elif 'uuid' in tmplist:      # node_hash
        temp_key='uuid'
        temp_value='verteid:'+str(re_values[temp_key][json_obj[temp_key]]) # 该 json_obj 对应的 hash 对应的 index，并构建 verteid:index

    for temp_char in str(temp_value):   # 根据字符编码
        if temp_char not in char2id_dict:
            end=len(char2id_dict)+2
            char2id_dict[temp_char]=end
            id2char_dict[end]=temp_char
            data_processed_.append(end)
        else:
            data_processed_.append(char2id_dict[temp_char])
    data_processed_.append(0)  # 0作为分隔符？
    # 属性组合模板字典（不同json 可能有的属性组合不同，因为数据丢失）
    if ','.join(tmplist) not in key_template_dict.keys(): 
        key_template_dict[','.join(tmplist)]=len(key_template_dict.keys())
    key_indx=str(key_template_dict[','.join(tmplist)])
    if key_indx not in char2id_dict:
        end=len(char2id_dict)+2
        char2id_dict[key_indx]=end
        id2char_dict[end]=key_indx
        data_processed_.append(end)
    else:
        data_processed_.append(char2id_dict[key_indx])
    data_processed_.append(0)

    for temp_key in tmplist:
        if temp_key=='uuid' or temp_key=='src' or temp_key=='dst':  # 上面处理过了, 传入的 src,dst 不需要编码，因为已经存在 edges 中了
            continue  
        if temp_key =='timestamp':
            temp_value=str(int(json_obj[temp_key])-mins[0])
            # temp_value=str(re_values['uuid'][json_obj[temp_key]])
        else:
            temp_value=re_values[temp_key][json_obj[temp_key]]
        for temp_char in str(temp_value):
            if temp_char not in char2id_dict:
                end=len(char2id_dict)+2
                char2id_dict[temp_char]=end
                id2char_dict[end]=temp_char
                data_processed_.append(end)
            else:
                data_processed_.append(char2id_dict[temp_char])
        data_processed_.append(0)
    data_processed_.append(1)  # 1 作为结尾符
    return data_processed_,edges
import os
import pickle
start_time = time.time()

edges=[]   # [[], []]
edges.append([])
edges.append([])
error=0
re_values,mins=count()
print('finished')
data_processed=[]
reader=[]
data=[]
# with open(args.input_path, 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     for i in reader:
#         data.append(i)
# key=data[0]
# data=data[1:]
# for i in range(len(data)):
#     json_obj={}
#     tmpdata=data[i]
#     for j in range(len(key)):
#         if tmpdata[j]!='':
#             json_obj[key[j]]=tmpdata[j]
#     tmp_strr,edges=handle_normal(json_obj,char2id_dict,id2char_dict,mins,re_values,key_template_dict,edges,flag=0)
#     if tmp_strr=='':
#         error=error+1
#     else:
#         data_processed.append(tmp_strr) 
reader=[]
data=[]
with open(args.input_path1, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i in reader:
        data.append(i)
key=data[0]
data=data[1:]
for i in range(len(data)):
    json_obj={}
    tmpdata=data[i]
    for j in range(len(key)): # 这里要将 src 和 dst 传入来构建 edges
        if tmpdata[j]!='':
            json_obj[key[j]]=tmpdata[j]
    tmp_strr,edges=handle_normal(json_obj,char2id_dict,id2char_dict,mins,re_values,key_template_dict,edges,flag=0)
    if tmp_strr=='':
        error=error+1
    else:
        data_processed.append(tmp_strr) 


for i in re_values.keys(): # 转换re_values
    tmp_dict=re_values[i]
    tmp_list=list(range(len(tmp_dict.keys())))
    for j in tmp_dict.keys():
        tmp_list[tmp_dict[j]]=j
    re_values[i]=tmp_list

edges1=''  
for i in edges:
    edges1=edges1+str(i)+'\n'
f1=open(args.edge_file,'w')  
f1.write(edges1)
f1.close()
# exit()
np.save(args.edge_file,edges)  # edges.npy
out = [c for item in data_processed for c in item]  # data_processed -> flatten
integer_encoded = np.array(out)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
np.save(args.output_path, integer_encoded)  # vertex.npy
# params = {'id2char_dict':id2char_dict,'char2id_dict':char2id_dict,'mins':mins,'re_values_dict':re_values,'key_template_dict':key_template_dict}
params = {'id2char_dict':id2char_dict,'char2id_dict':char2id_dict,'mins':mins,'re_values_dict':re_values}
with open(args.param_file, 'w') as f:  # 保存超参（映射表）到 params.json 中
    json.dump(params, f, indent=4)

end_time = time.time()
logger.info(f'Coding csv file, cost time:{(end_time - start_time)}')
