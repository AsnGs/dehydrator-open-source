import os
import json
import time
import copy
import csv
import torch
import logging
import argparse
import numpy as np

from models import *
from config import *

# Setting for logging
logger = logging.getLogger("queryLogger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_dir + queryLoggerFile)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('--model_path', default="./artifact/model/edge[23, 4096, 'SingleLayerDecoderTF', 0].pth", action='store', dest='modelPath')
parser.add_argument('--id', default='35C9F5FA-7396-A850-9673-243D00A81A46', action='store', dest='queryUUID')
parser.add_argument('--depth', default=3, action='store', dest='depth')

args = parser.parse_args()

def left_pad_sequences(sequences, sequence_length):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < sequence_length:
            pad_length = sequence_length - len(seq)
            padded_seq = [0] * pad_length + seq
        else:
            padded_seq = seq[-sequence_length:]
        padded_sequences.append(padded_seq)
    return torch.tensor(padded_sequences, dtype=torch.long)

def add_one_to_sequences(sequences):
    return [[x + 1 for x in seq] for seq in sequences]

def remove_padding_and_subtract_one(sequences):
    return [[x - 1 for x in seq if x != 0] for seq in sequences]

def generate_sequences_parallel(model, initial_sequences, sequence_length, ects):
    batch_size = len(initial_sequences)
    current_sequences = add_one_to_sequences(initial_sequences)
    generated_sequences = [[] for _ in range(batch_size)]
    active_sequences = [True] * batch_size
    loc_sequences = [0 for _ in range(batch_size)]
    ectime_sequences = [0 for _ in range(batch_size)]
    ectloc_sequences = [[] for _ in range(batch_size)]
    ectvalue_sequences = [[] for _ in range(batch_size)]
    
    for i in range(batch_size):
        for t in ects[i]: 
           ectloc_sequences[i].append(t[0])
           ectvalue_sequences[i].append(t[1])

    while any(active_sequences):
        try:
            padded_sequences = left_pad_sequences(current_sequences, sequence_length)
            input_tensor = padded_sequences.to(device)

            with torch.no_grad():
                output = model(input_tensor)
            next_tokens = output.argmax(dim=-1)
            
            for i in range(batch_size):
                if active_sequences[i]:
                    next_token = next_tokens[i].item()
                    if loc_sequences[i] in ectloc_sequences[i]:  
                        next_token = ectvalue_sequences[i][ectime_sequences[i]]
                        ectime_sequences[i] += 1
                    generated_sequences[i].append(next_token)
                    if next_token == 2:
                        active_sequences[i] = False
                    else:
                        current_sequences[i].append(next_token)
                        if len(current_sequences[i]) > sequence_length:
                            current_sequences[i] = current_sequences[i][-sequence_length:]
                        loc_sequences[i] += 1
        except Exception as e:
            logger.error(f"Error in sequence generation: {str(e)}", exc_info=True)
            raise
    
    generated_sequences = remove_padding_and_subtract_one(generated_sequences)
    return generated_sequences

def codeIndexList(indexList, char2listDict):
    codedList = []
    for index in indexList:
        tmpList = []
        for c in index: 
            tmpList.append(char2listDict[c])
        tmpList.append(0) 
        codedList.append(tmpList)
    return codedList


if __name__ == '__main__':
    with open(os.path.join(sc_dir, wholeMapDictFile), 'r') as f:
        wholeMapDict = json.load(f)
    vocab_size = len(wholeMapDict['id2char_dict']) + 3  # 0, 1, padding
    with open(os.path.join(sc_dir, vertexMapDictFile), 'r') as f:
        vertexMapDict = json.load(f)

    mergedIndex2us = np.load(os.path.join(sc_dir, codedVertexFile), allow_pickle=True)

    correlationTableFileName = args.modelPath.split('.')[1].split('/')[-1]
    with open(os.path.join(ect_dir, correlationTableFileName + '.json'), 'r') as f:
        ect = json.load(f)

    model = load_model(args.modelPath, vocab_size, logger=logger)
    sequence_length = int(args.modelPath.split('edge')[1].split('.')[0].strip('[]').split(', ')[0])
    set_seed(42)
    device = "cuda:2"
    model = model.to(device)

    uuids = [args.queryUUID]
    final_responses = []
    depth = 0
    modelTime = 0
    start_time = time.time()
    while (depth < int(args.depth)):
        uuidIndexList = [str(vertexMapDict['uuid'][uuid]) for uuid in uuids]
        initial_sequences = codeIndexList(uuidIndexList, wholeMapDict['char2id_dict'])
        ects = []
        for index in uuidIndexList:
            if index in ect.keys():  
                ects.append(ect[index])  
            else:
                initial_sequences.pop(uuidIndexList.index(index)) 
        
        try:
            if len(initial_sequences) == 0:break
            start2_time = time.time()
            generated_sequences = generate_sequences_parallel(model, initial_sequences, sequence_length - 1, ects) #! sequence_length -1
            end2_time = time.time()
            modelTime += (end2_time - start2_time)

            final_sequence = [initial_sequences[i] + generated_sequences[i] for i in range(len(generated_sequences))]
            strings = List2strings(final_sequence, wholeMapDict['id2char_dict'])
            responses = decodeString2Response(strings, minsTime, mergedIndex2us, vertexMapDict)
            tmpset = set()
            if len(responses) == 0: break  
            for res in responses:
                tmpset.add(res[0])
            uuids = list(tmpset)
            final_responses.extend(responses)
            depth += 1
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break
    end_time = time.time()        
    decodeTime = end_time - start_time - modelTime
    logger.info(f"The query ID is {args.queryUUID}, query depth is {args.depth}, final return num is: {len(final_responses)}")
    logger.info(f"The whole Time is: {end_time - start_time}")
    logger.info(f"Model generating time: {modelTime}")
    logger.info(f"Encoding time: {decodeTime}")
    # print(f"The query results of ID {args.queryUUID}:")
    # for response in final_responses[:args.maxNum]:
    #     print(response)
    # print(f'{modelTime} {decodeTime}')
    with open(os.path.join(csv_dir, 'query.txt'), mode='a', newline='') as file:
        file.write(f'{args.queryUUID} {args.depth} {len(final_responses)} {modelTime:.2f} {decodeTime:.2f}\n')

    