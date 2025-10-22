import sys
import torch
import models
import random
import pynvml
import psycopg2
import numpy as np

from config import *

def initialize_nvml():
    try:
        pynvml.nvmlInit()
        return True
    except pynvml.NVMLError as err:
        print(f"Failed to initialize NVML: {err}")
        return False

def get_free_gpu():
    if not initialize_nvml():
        raise RuntimeError("NVML initialization failed")
    device_count = pynvml.nvmlDeviceGetCount()
    free_memory = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory.append((i, mem_info.free))
    free_memory.sort(key=lambda x: x[1], reverse=True)
    pynvml.nvmlShutdown()  
    return free_memory[0][0]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_database_connection():
    if host is not None:
        connect = psycopg2.connect(database = database,
                                   host = host,
                                   user = user,
                                   password = password,
                                   port = port
                                  )
    else:
        connect = psycopg2.connect(database = database,
                                   user = user,
                                   password = password,
                                   port = port
                                  )
    cur = connect.cursor()
    return cur, connect

# string -> list
def codedData2id(codedDataList, char2id_dict, id2char_dict):
    finalCodedData = []
    for value in codedDataList:
        tmpString = str(value)
        for char in tmpString:
            if char not in char2id_dict:
                end = len(char2id_dict) +2
                char2id_dict[char] = end
                id2char_dict[end] = char
                finalCodedData.append(end)
            else:
                finalCodedData.append(char2id_dict[char])
        finalCodedData.append(0)
    finalCodedData.append(1)
    return finalCodedData

# list -> string
def List2strings(codedList, id2char_dict):
    strings = []
    for cl in codedList:
        tmpString = ''
        for num in cl:
            if num == 0: tmpString += ','
            elif num == 1: break
            else:
                tmpString += id2char_dict[str(num)]
        strings.append(tmpString[:-1])
    return strings

def parse_nodeOffset(node_str, delimiter='-'):
    nodeOffset = [int(v) - 3 for v in node_str.split(delimiter) if v]
    return range(nodeOffset[0], nodeOffset[1]+1)

def decode2EdgeList(tmpString, vIndex, usIndex, vertexMapDict):
    edgeList = []
    type_blocks = tmpString.split('c')
    for block in type_blocks:
        if not block:continue
        parts = block.split('b')
        type_index = int(parts[0])
        for time_node_info in parts[1:]:
            if not time_node_info:continue
            edge_parts = time_node_info.split('a')
            time_offset = int(edge_parts[0])
            if len(edge_parts) > 1:
                vertex_offsets = ungroup_nodeOffset(edge_parts[1])
                for vertex in vertex_offsets:
                    edgeList.append([str(usIndex[vertex]), vIndex, rel2id[type_index], str(time_offset + minsTime)])    
    for edge in edgeList:
        edge[0] = vertexMapDict['uuidIndex'][edge[0]]
        edge[1] = vertexMapDict['uuidIndex'][edge[1]]
    return edgeList


#  strings -> query response
def decodeString2Response(strings, minstime, mergedIndex2us, vertexMapDict):
    finalEdges = []
    for s in strings:
        try:
            tmpEdge = []
            vIndex, mergedIndex, startTime, endTime, content = s.split(',')
            usIndex = mergedIndex2us[int(mergedIndex)]
            # content = decodeContent(content)
            edgeList = decode2EdgeList(content, vIndex, usIndex, vertexMapDict)
            finalEdges.extend(edgeList)
        except:  
            continue
    return finalEdges


def get_total_size(obj, seen=None):
    """Recursively find the total memory size of an object and its contents."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_total_size(v, seen) for v in obj.values()])
        size += sum([get_total_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_total_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_total_size(i, seen) for i in obj])
    return size

def flip_dict(d):
    return {v: k for k, v in d.items()}

def splitDict(sourcDict, keys_to_move):
    targetDict = {}
    for key in keys_to_move:
        if key in sourcDict:
            targetDict[key] = sourcDict[key]
            del sourcDict[key]
    return sourcDict, targetDict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(model_path, vocab_size, logger):
    try:
        params = model_path.split('edge')[1].split('.')[0].strip('[]').split(', ')
        model_name = params[-2].strip("'")
        level = int(params[-1].strip("'"))
        model_args = {
            'vocab_size': vocab_size,  
            'max_length': int(params[0]) - 1  
        }
        model_params = params_list[level]
        model = getattr(models, model_name)(model_args['vocab_size'], model_args['max_length'], **model_params)
        # model = getattr(models, model_name)(**model_args)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model 
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def ungroup_nodeOffset(group_str):
    # if len(group_str) == 1 : return [int(group_str)]
    result = []
    ranges = group_str.split('-')
    if len(ranges) ==1 : return [int(ranges[0])]
    for i in range(0, len(ranges), 2):
        start = int(ranges[i])
        end = int(ranges[i+1])
        result.extend(range(start, end+1))
    return result


def group_nodeOffsetV2(arr, delimiter):
    if not arr:
        return ""

    compressed = []
    start = arr[0]
    end = arr[0]

    if len(arr) == 1:
        return f"{delimiter}{arr[0]}"

    for i in range(1, len(arr)):
        if arr[i] == end + 1:
            end = arr[i]
        else:
            if start == end:
                compressed.append(f"{delimiter}{str(start)}")
            else:
                compressed.append(f"{delimiter}{start}-{end}")
            start = arr[i]
            end = arr[i]

    if start == end:
        compressed.append(f"{delimiter}{str(start)}")
    else:
        compressed.append(f"{delimiter}{start}-{end}")

    return "".join(compressed)

def group_nodeOffset(n, delimiter):
    if not n: return []
    nums = [i for i in n]
    groups = ''
    start = nums[0]
    end = nums[0]
    for num in nums[1:]:
        if num == end + 1:end = num
        else:
            groups = groups + delimiter + str(start) + '-' + str(end)
            start = num
            end = num
    groups = groups + delimiter + str(start) + '-' + str(end)
    return groups