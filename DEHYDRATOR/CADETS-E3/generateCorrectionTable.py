import os
import json
import time
import torch
import logging
import argparse
from tqdm import tqdm
from collections import defaultdict

from models import *
from config import *
from utils import *

# Setting for logging
logger = logging.getLogger("generateErrorTableLogger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_dir + generateErrorTableLoggerFile)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('--model_path', default="./artifact/model/edge[23, 512, 'SingleLayerDecoderTF', 0].pth", action='store', dest='modelPath')
parser.add_argument('--batch_size', default=4096, action='store', dest='batch_size')

args = parser.parse_args()

def transAbsIndex2RelIndex(abstable, vCodedArrayMap):
    relTable = defaultdict(list)
    vIndexIndex = 0
    vIndexList = list(vCodedArrayMap.keys())
    # vIndexList.insert(0, 0)
    for loc, value in abstable.items():
        tmp = vIndexList[vIndexIndex]
        while not loc < tmp:   
            vIndexIndex += 1
            if vIndexIndex == len(vIndexList) : 
                break
            tmp = vIndexList[vIndexIndex]
        if vIndexIndex == len(vIndexList) : break
        vIndex = decode2vIndex(vCodedArrayMap[vIndexList[vIndexIndex]])  # codeArrayList -> vIndex 
        relTable[vIndex].append((loc - vIndexList[vIndexIndex - 1], value))
    return relTable 

def decode2vIndex(coded):
    tmpString = ''
    for c in coded:
        tmpString += wholeMapDict['id2char_dict'][str(c)]
    return int(tmpString)

def batch_predict(model, input_batch):
    with torch.no_grad():
        output = model(input_batch)
    return output

if __name__ == '__main__':
    try:
        set_seed(42)
        os.makedirs(ect_dir, exist_ok=True)
        logger.info(f'Model Path:{args.modelPath}')
        
        with open(os.path.join(sc_dir, wholeMapDictFile), 'r') as f:
            wholeMapDict = json.load(f)
        vocab_size = len(wholeMapDict['id2char_dict']) + 3  # 0, 1, padding
        
        batch_size = int(args.batch_size)
        model = load_model(args.modelPath, vocab_size, logger=logger)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        dataset = MyDataset(batchSize=batch_size, filePath=codedEdgeFile, shuffle=False)
        # dataloader = DataLoader(dataset, batch_size=int(args.batch_size), num_workers=10)
        dataloader = DataLoader(dataset, batch_size=batch_size) 

        all_predictions = []
        total_processed = 0
        total_samples = len(dataset)
        deliIndex = 0
        deliIndexList = dataset.deliIndexList
        relTable = defaultdict(list)
        vCodedArrayMap = dataset.getvCodedArrayMap()
        vIndexList = list(vCodedArrayMap.keys())
        vIndexIndex = 0
        vIndex = decode2vIndex(vCodedArrayMap[vIndexList[vIndexIndex]])

        start_time = time.time()
        for batch in tqdm(dataloader, desc="Processing batches"):
            if total_processed >= total_samples:
                break
            batch_size = min(batch.size(0), total_samples - total_processed)
            batch = batch[:batch_size]
        
            batch = batch.to(device)

            input_batch = batch[:, :-1]
            labels = batch[:, -1]

            predictions = batch_predict(model, input_batch)
            predicted_tokens = torch.argmax(predictions, dim=-1) 

            mask = (predicted_tokens != labels) 
            indices = torch.nonzero(mask).squeeze() 

            if indices.dim() == 0:
                indices = indices.unsqueeze(0)  
            for idx in indices:
                global_idx = total_processed + idx.item()
                if global_idx >= total_samples:
                    break
                while global_idx >= vIndexList[vIndexIndex]:
                    vIndexIndex += 1
                    vIndex = decode2vIndex(vCodedArrayMap[vIndexList[vIndexIndex]])
                correct_value = labels[idx].item()
                if vIndexIndex == 0:
                    relTable[vIndex].append((global_idx, correct_value))
                else:
                    relTable[vIndex].append((global_idx - vIndexList[vIndexIndex - 1], correct_value))
                # absTable[global_idx] = correct_value
            total_processed += batch_size
                

        end_time = time.time()
        logger.info(f'The time of generating Error Correlation Table : {(end_time - start_time)} seconds.')
        # logger.info(f'The size of generating Error Correlation Table : {get_total_size(relTable)} B.')

        correlationTableFileName = args.modelPath.split('.')[1].split('/')[-1]
        with open(os.path.join(ect_dir, correlationTableFileName + '.json'), 'w') as f:
            json.dump(relTable, f)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 