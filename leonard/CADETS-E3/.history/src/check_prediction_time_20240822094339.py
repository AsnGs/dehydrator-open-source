import os 
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.models import load_model
# from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import numpy as np
import argparse
import contextlib
import json
import struct
import models
import tempfile
import shutil
import logging
from time import *

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-model', action='store', dest='model_weights_file',
                    help='model file')
parser.add_argument('-model_name', action='store', dest='model_name',
                    help='model file')
parser.add_argument('-model_path', action='store', dest='model_path',
                    help='model file')
parser.add_argument('-data', action='store', dest='sequence_npy_file',
                    help='data file')
parser.add_argument('-data_params', action='store', dest='params_file',
                    help='params file')
parser.add_argument('-table_file', action='store', dest='table_file',
                    help='table_file')
parser.add_argument('-gpu', action='store', dest='gpu',
                            help='gpu')
# args = parser.parse_args()
args = parser.parse_args([
    '-model', './src/vertex.hdf5',
    '-model_name', 'LSTM_multi',
    '-model_path', './src/lite.h52048vertex.hdf5',
    '-data', './data/vertex.npy',
    '-data_params', './data/vertex.params.json',
    '-table_file', './src/table.params.json',
    '-gpu', '0'
])

logger = logging.getLogger("message")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('./message.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

from keras import backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
tf.compat.v1.keras.backend.set_session(sess)

# Window len = L, Stride len/stepsize = S
def strided_app(a, L, S):  
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(
        a, shape=(nrows, L), strides=(S * n, n), writeable=False)

#  不仅进行预测，还收集了预测错误的信息
def predict_lstm(data_x,data_y, inds,timesteps, alphabet_size, model_name,batch_size):
        X=np.array(data_x)
        Y=data_y
        num_iter=int(len(X)/batch_size)  
        prob=[]
        addd=0
        begin1=time()
        if len(X)>batch_size*int(len(X)/batch_size):
            addd=1
        for i in range(num_iter+addd):
         #   begin_time_tmp=time()
            if addd==1 and i==num_iter:
                batch_x=X[batch_size*int(len(X)/batch_size):]
                for inde in range(batch_size-len(batch_x)):
                    batch_x=np.concatenate((batch_x,np.array([X[0]])))
            else:
                batch_x=X[i*batch_size:(i+1)*batch_size]
            batch_x=batch_x.astype(np.float32)  #  [batch_size, timesteps]
            begin_time_=time()
            interpreter = tf.lite.Interpreter(model_path=args.model_path) # 创建TFLite解释器，加载指定路径的模型文件
            interpreter.allocate_tensors() # 为模型的张量分配内存
            input_details = interpreter.get_input_details() # 获取模型的输入和输出细节，包括张量的形状和类型等信息
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]["index"], batch_x) # 将数据输入到模型的张量中
            interpreter.invoke()  # 运行模型进行推理
            #print(batch_x)
            prob_tmp = interpreter.get_tensor(output_details[0]["index"])
            #print(prob_tmp)
            end_=time()
            #print(end_-begin_time_)
            for pro in prob_tmp:
                prob.append(pro)
        table_item=[]  # 记录错误信息列表，其中每个元素是一个包含两个字符串的列表，一个是str(j-begin) 是错误在当前序列中的相对位置，另一个str(Y[j]) 是正确的标签
        #prob = model.predict(X, batch_size=len(Y))
        for i in range(len(inds)):
            table_item_=[]
            if i ==0:
                begin = 0
                end=inds[i]
            else:
                begin=inds[i-1]
                end=inds[i]
            for j in range(begin,end):
                if np.argmax(prob[j])!=Y[j]: # 比较预测与真实标签
                    table_item_.append([str(j-begin),str(Y[j])])
            table_item.append(table_item_)
        #print('each iter time')
        #print(time()-begin1)
        return table_item

# 对 data根据分隔符进行切片
def get_slice(data,flag):
    add=0
    if flag==1:
        add=1
    ind=np.where(np.array(data)==flag)[0]
    finaldata=[]
    for i in range(len(ind)):
        if i==0:
            finaldata.append(data[:ind[i]+add])
        else:
            finaldata.append(data[ind[i-1]+1:ind[i]+add])
    return finaldata

# 获取 data 中 flag 分隔符的对应索引
def get_slice_index(data,flag):
    ind=np.where(np.array(data)==flag)[0]
    return ind

# id -> char， 从编码转回原字符串, 一般是将编码后数据的检索头部分再转回到可理解的字符串
def translate(data, key=''):  
    data_str=''
    for i in data:
        data_str=data_str+id2char_dict[str(i)]
    if key!='':
        data_str=re_values[key][int(data_str)]
    return data_str


def main():
        begin_time=time()
        args.temp_dir = tempfile.mkdtemp()  # 创建临时文件夹
        args.temp_file_prefix = args.temp_dir + "/compressed"
        tf.compat.v1.set_random_seed(42)  # 需要设置随机种子吗？
        np.random.seed(0)
        series = np.load(args.sequence_npy_file) # npy
        series = series.reshape(-1, 1)        
        onehot_encoder = OneHotEncoder(sparse=False) # onhot-encoder
        onehot_encoded = onehot_encoder.fit(series)
        timesteps = 10  #?  =sequenceLength?
        with open(args.params_file, 'r') as f:
            params = json.load(f)  # params json
        params['len_series'] = len(series)
        params['timesteps'] = timesteps
        global id2char_dict
        id2char_dict=params['id2char_dict']
        global char2id_dict
        char2id_dict=params['char2id_dict']
        global re_values
        re_values=params['re_values_dict']
        alphabet_size = len(params['id2char_dict'])+2
        series = series.reshape(-1)
        series=get_slice(series,1)  # 对节点/事件间进行分割
        data_len=[]
        error_=[]
        order_=[]
        table={}
        batch_size=2048
        batchsize=2048

        for j in range(int(len(series)/batch_size)):
            timestamp=[]
            data_sub_x=[]
            data_sub_y=[]
            inds=[]

            for i in range(batch_size):
                timestamp_=get_slice(series[batch_size*j+i],0)[0][8:]  # 返回某个 vertex/edge 的检索输入信息
                if series[batch_size*j+i][0]==2:  # e : 2，对应 edge开头  #! 找 char2id
                    timestamp.append('e:'+translate(timestamp_))
                else:   # 其他的就是v开头
                    timestamp.append('v:'+translate(timestamp_))
                index_of_time=get_slice_index(series[batch_size*j+i],0)[0] # 第一个事件的分隔符索引

                data_tmp=strided_app(series[batch_size*j+i][index_of_time-timesteps+1:], timesteps+1, 1) # 根据时间步切片
                if i==0:
                    inds.append(len(data_tmp))
                else:
                    inds.append(len(data_tmp)+inds[i-1])
                data_len.append(len(data_tmp))
                for data_index in range(len(data_tmp)):  # 构建对应的输入和标签
                    data_sub_x.append(data_tmp[:,:-1][data_index])
                    data_sub_y.append(data_tmp[:,-1:][data_index][0])
            table_item=predict_lstm(data_sub_x,data_sub_y,inds, timesteps, alphabet_size, args.model_name,batchsize)
            for i in range(batch_size):
                error=len(table_item[i]) # 每行的错误个数
                if error>0:  # 如果有错误，就添加到列表中，分别是错误索引和正确值
                    table[timestamp[i]]=table_item[i]  # table : {'v:index':[errorOffset, trurValue]}
                    error_.append(error)
                    order_.append(batch_size*j+i) # 不同 batch 下的值
        timestamp=[]
        data_sub_x=[]
        data_sub_y=[]
        inds=[]
#        print('whole process time')
#        print(time()-begin_time)
        # 对末尾部分进行一下处理
        for j in range(int(len(series)/batch_size)*batch_size,len(series)):
            timestamp_=get_slice(series[j],0)[0][8:]
            if series[j][0]==2:
                timestamp.append('e:'+translate(timestamp_))
            else:
                timestamp.append('v:'+translate(timestamp_))
            index_of_time=get_slice_index(series[j],0)[0]
            data_tmp=strided_app(series[j][index_of_time-timesteps+1:], timesteps+1, 1)
            if len(inds)==0:
                inds.append(len(data_tmp))
            else:
                inds.append(len(data_tmp)+inds[j-int(len(series)/batch_size)*batch_size-1])
            data_len.append(len(data_tmp))
            for data_index in range(len(data_tmp)):
                data_sub_x.append(data_tmp[:,:-1][data_index])
                data_sub_y.append(data_tmp[:,-1:][data_index][0])
        table_item=predict_lstm(data_sub_x,data_sub_y,inds, timesteps, alphabet_size, args.model_name,batchsize)
        for i in range(len(series)-int(len(series)/batch_size)*batch_size):
            error=len(table_item[i])
            if error>0:
                table[timestamp[i]]=table_item[i]
                error_.append(error)
                order_.append(i+int(len(series)/batch_size)*batch_size)
        additional_save_time=time()
        print('additional save time')
        print(time()-additional_save_time)
        write_time=time()
        print('write time')
        # 最后将错误存到json中，table200m.params.json, 即纠错表
        with open(args.table_file, 'w') as f:
            json.dump(table, f, indent=4)
        end_time=time()
        print(end_time-write_time)
        print('time is')
        print(end_time-begin_time)
        print('read time is')
        logger.info(f'Check prediction, cost time:{(end_time - begin_time)}')

                        
if __name__ == "__main__":
    main()

'''
主要流程：

数据准备：加载 NPY 文件中的序列数据，并进行必要的预处理。
批处理：将数据分成多个批次，每个批次 2048 个样本。
预测：对每个批次使用 TFLite 模型进行预测。
错误分析：收集预测错误的信息，包括错误位置和正确标签。
结果保存：将错误信息保存到 JSON 文件中。
'''
