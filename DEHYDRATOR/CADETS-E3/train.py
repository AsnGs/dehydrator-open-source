import os
import json
import time
from tqdm import tqdm
import models
import logging
import argparse
import torch.nn as nn
from collections import deque

from models import *
from config import *

# Setting for logging
logger = logging.getLogger("trainEdgeModelLogger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_dir + trainEdgeModelLoggerFile)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

parser = argparse.ArgumentParser(description='Input')
# parser.add_argument('-—sequence_length', default=21, action='store', dest='sequenceLength')
parser.add_argument('--batch_size', default=4096, action='store', dest='batchsize')
parser.add_argument('--model_name', default='SingleLayerDecoderTF', action='store', dest='model_name')
parser.add_argument('--epoch', default=5, action='store', dest='epoch')
parser.add_argument('--dt', default=False, action='store', dest='dt')
parser.add_argument('--level', default=0, action='store', dest='level')
# parser.add_argument('--ff_iterations', default=2, action='store', dest='ff_iterations')

args = parser.parse_args()

if __name__ == '__main__':
    os.makedirs(model_dir, exist_ok=True)

    start_time = time.time()

    with open(os.path.join(sc_dir, wholeMapDictFile), 'r') as f:  
        wholeMapDict = json.load(f)
    vocab_size = len(wholeMapDict['id2char_dict'])+ 3  

    batchsize = int(args.batchsize)
    model_name = str(args.model_name)
    num_epochs = int(args.epoch)
    level = int(args.level)
    dt = bool(args.dt)

    free_gpu_id = get_free_gpu()
    device = "cuda:1"

    dataset = MyDataset(batchsize, codedEdgeFile, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=8, pin_memory=True)

    sequenceLength = dataset.sequenceLength
    logger.info(f'Model Name: {args.model_name}, Level :{args.level},  Sequence Length: {sequenceLength}, Epoch: {args.epoch}, Batch Size:{args.batchsize}, DT: {args.dt}')
    
    params = params_list[level]
    model = getattr(models, model_name)(vocab_size, sequenceLength - 1, **params)
    model = model.to(device)    

    criterion = nn.CrossEntropyLoss().to(device)   
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)  

    # early-stopping
    best_loss = float('inf')  
    patience = 1  
    loss_decThre = 0.01  
    num_epochs_no_improve = 0 

    if dt:
        queue = deque([0] * batchsize, maxlen=batchsize)

    for epoch in tqdm(range(num_epochs), desc="Training Epoches"):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Processing batches"):
            optimizer.zero_grad()
            batch = batch.to(device) 

            inputs = batch[:, :-1]  # (batch_size, sequenceLength-1)
            labels = batch[:, -1]   # (batch_size,)

            outputs = model(inputs)  # (batch_size, vocab_size)
            loss = criterion(outputs, labels)

            if not dt:
                loss.backward()
                optimizer.step()
            elif loss > (sum(queue) / len(queue)):  # dynamic training controller
                loss.backward()
                optimizer.step()
                queue.append(loss.item())
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
        
        # Early stopping check
        if avg_loss < (best_loss - loss_decThre):
            best_loss = avg_loss
            num_epochs_no_improve = 0
        else:
            num_epochs_no_improve += 1
        
        if num_epochs_no_improve >= patience or epoch == (num_epochs - 1):
            end_time = time.time()
            logger.info(f'The time of training Model {model_name} is {(end_time - start_time)} seconds.')
            modelSavePath = os.path.join(model_dir, "edge"+str([sequenceLength, batchsize, model_name, level])+".pth")
            torch.save(model.state_dict(), modelSavePath)
            logger.info(f"Model saved at epoch {epoch+1} with loss {avg_loss}")
            logger.info(f"Model {model_name}+Level{level}'s number of parameters is {count_parameters(model=model)}")
            break        