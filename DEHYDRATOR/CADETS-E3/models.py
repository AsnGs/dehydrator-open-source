import os
import torch
import numpy as np
import torch.nn as nn
from typing import Iterator
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from sklearn.preprocessing import OneHotEncoder

from config import *
from utils import *

def getDeliIndexList(data, flag):
    deliIndexList = np.where(data == flag)[0]
    return deliIndexList

def getAvoidIndexList(codedArray, sequenceLength):
    deliIndexList = getDeliIndexList(codedArray, 1)
    avoidIndexList = []
    for i in range(len(deliIndexList)):
        for index_ in range(sequenceLength):
            avoidIndexList.append(deliIndexList[i] + index_ + 1)
    return avoidIndexList

def getOnehotEncoder(codedArray):
    oneHotEncoder = OneHotEncoder()
    oneHotEncoder.fit(codedArray)
    return oneHotEncoder

def getChoiceIndex(codedArray, deliIndexList, vCodedArrayMap, batchsize, shuffle):
    differences = [deliIndexList[i + 1] - deliIndexList[i] for i in range(len(deliIndexList) - 1)]
    differences.append(deliIndexList[0]+1)
    min_difference = min(differences) if differences else None  
    sequenceLength = min_difference - 1  
    ret = []  # [endIndex]
    retMapDict = {}  # endIndex : leftPositionOffset
    deliIndexList = np.insert(deliIndexList, 0, -1)

    # preIndex = deliIndexList[0]  
    dataDeliIndex = 1  
    preIndex = getDeliIndexList(codedArray[deliIndexList[dataDeliIndex-1] + 1:deliIndexList[dataDeliIndex] + 1], 0)[0]  # Find the index of the first field separator in the data, then increment it by 1 to obtain the starting position for the slice.
    for i in range(deliIndexList[0], len(codedArray) + 1):
        if i <= preIndex : continue
        else:
            ret.append(i)
            if i - sequenceLength < deliIndexList[dataDeliIndex-1]:
                retMapDict[i] = i - deliIndexList[dataDeliIndex-1] - 1
            else:
                retMapDict[i] = sequenceLength - 1
            if codedArray[i] == 1: 
                vIndex = codedArray[deliIndexList[dataDeliIndex-1] + 1:preIndex]
                vCodedArrayMap[len(retMapDict)] = vIndex  
                dataDeliIndex += 1
                if dataDeliIndex == len(deliIndexList): break
                preIndex = getDeliIndexList(codedArray[deliIndexList[dataDeliIndex-1] + 1:deliIndexList[dataDeliIndex] + 1], 0)[0] + deliIndexList[dataDeliIndex-1] + 1  # Index value of the input separator in the next data

    if shuffle:  
        choice = np.random.choice(ret, len(ret), replace=False)
    else:
        choice = np.array(ret)
    choice = choice[:int(len(choice)/batchsize) * batchsize] 
    return choice, retMapDict, sequenceLength
        
class MyDataset(IterableDataset):
    def __init__(self, batchSize, filePath, shuffle=False) -> None:
        self.batchsize = batchSize
        self.codedArray = np.load(os.path.join(sc_dir, filePath))  
        print(self.codedArray.shape)
        self.codedArray = self.codedArray.reshape(-1) 
        self.deliIndexList = getDeliIndexList(self.codedArray, 1)
        self.vCodedArrayMap = {}  #{absloc of split data : vIndexCodedArray}
        self.shuffle = shuffle
        self.index = 0
        try:
            self.choice, self.mapDict, self.sequenceLength = getChoiceIndex(self.codedArray, self.deliIndexList, self.vCodedArrayMap, self.batchsize, self.shuffle)
            print(self.sequenceLength)
        except ValueError as e:
            print(repr(e))
            sys.exit()
    
    def __len__(self):
        return len(self.choice)
    
    def reset(self):
        self.index = 0
        if self.shuffle:
            self.choice = np.random.choice(self.choice, len(self.choice), replace=False)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  
            self.start_index = 0
            self.end_index = len(self.choice)
        else:  
            per_worker = len(self.choice) // worker_info.num_workers
            worker_id = worker_info.id
            self.start_index = worker_id * per_worker
            self.end_index = (worker_id + 1) * per_worker if worker_id != worker_info.num_workers - 1 else len(self.choice)

        if self.shuffle:
            np.random.shuffle(self.choice[self.start_index:self.end_index])

        self.index = self.start_index
        return self  
    
    def __pad_to_length__(self, lst):  #!
        if len(lst) == (self.sequenceLength):
            return  torch.from_numpy(lst).long() + 1
            # return torch.tensor(lst, dtype=torch.long)
        elif len(lst) < (self.sequenceLength):
            tensor = torch.from_numpy(lst).long() + 1
            pad_length = self.sequenceLength - len(lst)
            # padded_tensor = F.pad(tensor, (0, pad_length), value=-1) 
            padded_tensor = F.pad(tensor, (pad_length, 0), value= 0)  # Select left padding #! Change padding value to 0 to avoid embedding layer errors when using -1
            return padded_tensor   
        else:
            raise ValueError(f"Sequence Length Wrong")

    def __next__(self):
        if self.index >= self.end_index:
            raise StopIteration
        data = self.codedArray[self.choice[self.index] - self.mapDict[self.choice[self.index]] : self.choice[self.index] + 1]
        self.index += 1
        return self.__pad_to_length__(data)

    def getvCodedArrayMap(self):
        return self.vCodedArrayMap

class BigSingleLayerDecoderTF(nn.Module):
    def __init__(self, vocab_size, max_length, d_model=512, nhead=8, dim_feedforward=2048):
        super(BigSingleLayerDecoderTF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        device = x.device
        positional_encoding = self.positional_encoding[:, :x.size(1), :].to(device)
        embedded = self.embedding(x) + positional_encoding
        embedded = embedded.transpose(0, 1)  # Transformer expects (seq_len, batch_size, d_model)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        decoded = self.transformer_decoder_layer(embedded, embedded, tgt_mask)
        output = self.fc_out(decoded.transpose(0, 1))  
        return output[:, -1, :] 
    
class MiddleSingleLayerDecoderTF(nn.Module):
    def __init__(self, vocab_size, max_length, d_model=256, nhead=8, dim_feedforward=1024):
        super(MiddleSingleLayerDecoderTF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        device = x.device
        positional_encoding = self.positional_encoding[:, :x.size(1), :].to(device)
        embedded = self.embedding(x) + positional_encoding
        embedded = embedded.transpose(0, 1)  # Transformer expects (seq_len, batch_size, d_model)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        decoded = self.transformer_decoder_layer(embedded, embedded, tgt_mask)
        output = self.fc_out(decoded.transpose(0, 1))   
        return output[:, -1, :] 

class SingleLayerDecoderTF(nn.Module):
    def __init__(self, vocab_size, max_length, d_model=64, nhead=2, dim_feedforward=256):
        super(SingleLayerDecoderTF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        device = x.device
        positional_encoding = self.positional_encoding[:, :x.size(1), :].to(device)
        embedded = self.embedding(x) + positional_encoding
        embedded = embedded.transpose(0, 1)  # Transformer expects (seq_len, batch_size, d_model)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        decoded = self.transformer_decoder_layer(embedded, embedded, tgt_mask)
        output = self.fc_out(decoded.transpose(0, 1))   
        return output[:, -1, :] 

class SingleLayerDecoderTFLevel1(nn.Module):
    def __init__(self, vocab_size, max_length, d_model=64, nhead=2, dim_feedforward=256):
        super(SingleLayerDecoderTFLevel1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        device = x.device
        positional_encoding = self.positional_encoding[:, :x.size(1), :].to(device)
        embedded = self.embedding(x) + positional_encoding
        embedded = embedded.transpose(0, 1)  # Transformer expects (seq_len, batch_size, d_model)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        decoded = self.transformer_decoder_layer(embedded, embedded, tgt_mask)
        output = self.fc_out(decoded.transpose(0, 1))   
        return output[:, -1, :] 
    
class SingleLayerDecoderTFLevel2(nn.Module):
    def __init__(self, vocab_size, max_length, d_model=128, nhead=4, dim_feedforward=512):
        super(SingleLayerDecoderTFLevel2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        device = x.device
        positional_encoding = self.positional_encoding[:, :x.size(1), :].to(device)
        embedded = self.embedding(x) + positional_encoding
        embedded = embedded.transpose(0, 1)  # Transformer expects (seq_len, batch_size, d_model)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        decoded = self.transformer_decoder_layer(embedded, embedded, tgt_mask)
        output = self.fc_out(decoded.transpose(0, 1))   
        return output[:, -1, :]

class SingleLayerDecoderTFLevel3(nn.Module):
    def __init__(self, vocab_size, max_length, d_model=256, nhead=8, dim_feedforward=1024):
        super(SingleLayerDecoderTFLevel3, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        device = x.device
        positional_encoding = self.positional_encoding[:, :x.size(1), :].to(device)
        embedded = self.embedding(x) + positional_encoding
        embedded = embedded.transpose(0, 1)  # Transformer expects (seq_len, batch_size, d_model)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        decoded = self.transformer_decoder_layer(embedded, embedded, tgt_mask)
        output = self.fc_out(decoded.transpose(0, 1))   
        return output[:, -1, :]

class SingleLayerDecoderTFLevel4(nn.Module):
    def __init__(self, vocab_size, max_length, d_model=512, nhead=16, dim_feedforward=2048):
        super(SingleLayerDecoderTFLevel4, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        device = x.device
        positional_encoding = self.positional_encoding[:, :x.size(1), :].to(device)
        embedded = self.embedding(x) + positional_encoding
        embedded = embedded.transpose(0, 1)  # Transformer expects (seq_len, batch_size, d_model)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        decoded = self.transformer_decoder_layer(embedded, embedded, tgt_mask)
        output = self.fc_out(decoded.transpose(0, 1))   
        return output[:, -1, :]   

class SmallSingleLayerDecoderTF(nn.Module):
    def __init__(self, vocab_size, max_length, d_model=16, nhead=4, dim_feedforward=64):
        super(SmallSingleLayerDecoderTF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        device = x.device
        positional_encoding = self.positional_encoding[:, :x.size(1), :].to(device)
        embedded = self.embedding(x) + positional_encoding
        embedded = embedded.transpose(0, 1)  # Transformer expects (seq_len, batch_size, d_model)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        decoded = self.transformer_decoder_layer(embedded, embedded, tgt_mask)
        output = self.fc_out(decoded.transpose(0, 1))   
        return output[:, -1, :] 

class MultiLayerLSTM(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim=32, hidden_dim=64, num_layers=2):
        super(MultiLayerLSTM, self).__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.size(1) > self.max_length:
            x = x[:, :self.max_length]  # 截断超过 max_length 的部分
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_lstm_out = lstm_out[:, -1, :]
        fc1_out = self.relu(self.fc1(last_lstm_out))
        output = self.fc2(fc1_out)
        return output 
    

class BigMultiLayerLSTM(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim=128, hidden_dim=256, num_layers=4):
        super(BigMultiLayerLSTM, self).__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.size(1) > self.max_length:
            x = x[:, :self.max_length]
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_lstm_out = lstm_out[:, -1, :]
        fc1_out = self.relu(self.fc1(last_lstm_out))
        output = self.fc2(fc1_out)
        return output 
    
class MultiLayerGRU(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim=32, hidden_dim=64, num_layers=2):
        super(MultiLayerGRU, self).__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.size(1) > self.max_length:
            x = x[:, :self.max_length]  
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        last_gru_out = gru_out[:, -1, :]
        fc1_out = self.relu(self.fc1(last_gru_out))
        output = self.fc2(fc1_out)
        return output

class BigMultiLayerGRU(nn.Module):
    def __init__(self, vocab_size, max_length, embed_dim=128, hidden_dim=256, num_layers=4):
        super(BigMultiLayerGRU, self).__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.size(1) > self.max_length:
            x = x[:, :self.max_length]  
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        last_gru_out = gru_out[:, -1, :]
        fc1_out = self.relu(self.fc1(last_gru_out))
        output = self.fc2(fc1_out)
        return output

class FastOverfitMiddleSingleLayerDecoderMultiFFNTF(nn.Module):
    def __init__(self, vocab_size, max_length, d_model=256, nhead=8, dim_feedforward=1024, ff_iterations=2):
        super(FastOverfitMiddleSingleLayerDecoderMultiFFNTF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        decoder_layer = FlexibleFeedForwardDecoderLayer(
            d_model, nhead, dim_feedforward, dropout= 0.0, ff_iterations=ff_iterations
        )  # dropout 设置为 0
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        device = x.device
        positional_encoding = self.positional_encoding[:, :x.size(1), :].to(device)
        embedded = self.embedding(x) + positional_encoding
        embedded = embedded.transpose(0, 1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        decoded = self.transformer_decoder(embedded, embedded, tgt_mask=tgt_mask)
        output = self.fc_out(decoded.transpose(0, 1))   
        return output[:, -1, :]

class FlexibleFeedForwardDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu", ff_iterations=2):
        super(FlexibleFeedForwardDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.ff_iterations = ff_iterations
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            for _ in range(self.ff_iterations):
                x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            for _ in range(self.ff_iterations):
                x = self.norm3(x + self._ff_block(x))
        return x    
    
class MiddleSingleLayerDecoderMultiFFNTF(nn.Module):
    def __init__(self, vocab_size, max_length, d_model=256, nhead=8, dim_feedforward=1024, ff_iterations=2):
        super(MiddleSingleLayerDecoderMultiFFNTF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.transformer_decoder_layer = FlexibleFeedForwardDecoderLayer(
            d_model, nhead, dim_feedforward, ff_iterations=ff_iterations
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        device = x.device
        positional_encoding = self.positional_encoding[:, :x.size(1), :].to(device)
        embedded = self.embedding(x) + positional_encoding
        embedded = embedded.transpose(0, 1)  # Transformer expects (seq_len, batch_size, d_model)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        decoded = self.transformer_decoder_layer(embedded, embedded, tgt_mask)
        output = self.fc_out(decoded.transpose(0, 1))   
        return output[:, -1, :]
    

class MiddleSingleLayerDecoderMultiFFN4TF(nn.Module):
    def __init__(self, vocab_size, max_length, d_model=256, nhead=8, dim_feedforward=1024, ff_iterations=4):
        super(MiddleSingleLayerDecoderMultiFFN4TF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, d_model))
        self.transformer_decoder_layer = FlexibleFeedForwardDecoderLayer(
            d_model, nhead, dim_feedforward, ff_iterations=ff_iterations
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        device = x.device
        positional_encoding = self.positional_encoding[:, :x.size(1), :].to(device)
        embedded = self.embedding(x) + positional_encoding
        embedded = embedded.transpose(0, 1)  # Transformer expects (seq_len, batch_size, d_model)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        decoded = self.transformer_decoder_layer(embedded, embedded, tgt_mask)
        output = self.fc_out(decoded.transpose(0, 1))   
        return output[:, -1, :]