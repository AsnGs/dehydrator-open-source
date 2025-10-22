import os
import csv
import json
import pickle
import time
import logging
import argparse
import numpy as np
from collections import defaultdict

from config import *
from utils import *

# Setting for logging
logger = logging.getLogger("structCompressCode_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_dir + structCompressCodeLoggerFile)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('--vertex_csv_file', default='vertex.csv', action='store', dest='vertex_csv_file')
parser.add_argument('--edge_csv_file', default='edge.csv', action='store', dest='edge_csv_file')

args = parser.parse_args()

def group_nodeOffsetV2(arr, delimiter):
    if not arr:return ""
    compressed = []
    start = arr[0]
    end = arr[0]
    if len(arr) == 1:return f"{delimiter}{arr[0]}"
    for i in range(1, len(arr)):
        if arr[i] == end + 1:end = arr[i]
        else:
            if start == end:compressed.append(f"{delimiter}{str(start)}")
            else:compressed.append(f"{delimiter}{start}-{end}")
            start = arr[i]
            end = arr[i]
    if start == end:compressed.append(f"{delimiter}{str(start)}")
    else:compressed.append(f"{delimiter}{start}-{end}")
    return "".join(compressed)

# edgeList: [[edgeType, time]]  -> content:{} - > tmpString
def consAndCodeContentV2(edgeList):
    content = {} # content: {typeIndex:{timeOffset: vertexOffset]}}   
    for i in range(len(edgeList)):
        for j in range(len(edgeList[i])):
            offsetTime = int(edgeList[i][j][1]) - minsTime #  offset
            typeIndex = rel2id[edgeList[i][j][0]]
            if typeIndex not in content.keys(): content[typeIndex] = {}
            if offsetTime not in content[typeIndex].keys(): content[typeIndex][offsetTime] = []
            content[typeIndex][offsetTime].append(i)
    tmpString = ''  
    for typeindex in content.keys():
        tmpString += str(int(typeindex))
        for timeoffset in content[typeindex].keys():
            tmpString += 'b'
            tmpString += str(int(timeoffset))
            tmpString += group_nodeOffsetV2([vertexOffset for vertexOffset in content[typeindex][timeoffset]], delimiter='a')
        tmpString += 'c'  
    return tmpString

def mergeEdges(edgeList, mergedVetexIndex, v):
    # mergedEdge = [mergedVetexIndex, v]  # [merged] 
    mergedEdge = [v, mergedVetexIndex]
    times = [t - minsTime for t in sorted(list(set([int(e[1]) for edges in edgeList for e in edges])))]  # t - minsTime
    # startTime, endTime = min(times), max(times)
    if len(edgeList) == 1:
        mergedEdge.extend([times[0], times[0]])
    else:
        mergedEdge.extend([times[0], times[-1]])  

    mergedEdge.append(consAndCodeContentV2(edgeList))
    return mergedEdge

def getFieldValueIndex_MappingDict(jsonObj, mapDict):
    for i in range(len(jsonObj)):
        tempField = list(jsonObj.keys())[i]
        tempValue = jsonObj[tempField]
        if tempField == 'starttime' or tempField == 'endtime':
            continue
        else:
            if tempField not in mapDict.keys():
                mapDict[tempField] = {}
            if tempValue not in mapDict[tempField].keys():
                mapDict[tempField][tempValue] = len(mapDict[tempField].keys())  # {field: {value: index}}
    return mapDict

def createEdgeMap(csv_dir, edge_csv_file, wholeMapDict):
    vertexMap = defaultdict(set) # vertexMap : {v:us(U)}
    edgeMap = defaultdict(set) # edgeMap : {(U, v): e} 
    with open(os.path.join(csv_dir, edge_csv_file), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        vertex2Index = wholeMapDict['uuid']
        for edge in reader:  # edge: [u,v, type, time]
            edge = [vertex2Index[edge[0]], vertex2Index[edge[1]], edge[2], edge[3]]
            vertexMap[edge[1]].add(edge[0])
            edgeMap[(edge[0], edge[1])].add((edge[2], edge[3])) # list unhashable, 
    return vertexMap, edgeMap

def createCompressStructMap(vertexMap, edgeMap):
    newEdges = [] # [mergedUIndex, vIndex, startTime, endTime, {timeOffset:{type:[vertexOffset]}}]
    mergedIndex2us = []  # merged Vertex Index to u (one to many)  {i:[u1, u2, ...]}
    mergedVetexIndex = 0

    for v in vertexMap.keys():
        U = sorted(vertexMap[v])
        mergedIndex2us.append(U)
        tmpMergeEdges = [list(edgeMap[(u, v)]) for u in U]
        mergedEdge = mergeEdges(tmpMergeEdges, mergedVetexIndex, v)
        newEdges.append(mergedEdge) 
        mergedVetexIndex += 1 
    return newEdges, mergedIndex2us

def codeVertex(json_obj, wholeMapDict):
    codedDataList = []
    for key in json_obj.keys():
        tmpValue = json_obj[key]
        if key not in wholeMapDict.keys(): wholeMapDict[key] = {}
        if str(tmpValue) not in wholeMapDict[key].keys():
            codedDataList.append(len(wholeMapDict[key].keys()))
            wholeMapDict[key][str(tmpValue)] = len(wholeMapDict[key].keys())
        else:
            codedDataList.append(wholeMapDict[key][str(tmpValue)])

def codeVertexByEdge(edge, wholeMapDict):
    for uuid in edge[:2]:
        if 'uuid' not in wholeMapDict.keys(): wholeMapDict['uuid'] = {}
        if uuid not in wholeMapDict['uuid'].keys():
            wholeMapDict['uuid'][uuid] = len(wholeMapDict['uuid'].keys())

if __name__ == '__main__':
    os.makedirs(sc_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger.info("Start logging of structCompree&Code.")

    wholeMapDict= {}
    wholeMapDict['id2char_dict'] = {}
    wholeMapDict['char2id_dict'] = {}
    wholeMapDict['uuidIndex']    = {}

    vertex_csv_file = args.vertex_csv_file
    edge_csv_file   = args.edge_csv_file

    start_time = time.time()
    codedVertexData = []
    with open(os.path.join(csv_dir, edge_csv_file), 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for edge in reader:
            codeVertexByEdge(edge, wholeMapDict)
    # Create Compressed Struct Dict
    vertexMap, edgeMap = createEdgeMap(csv_dir=csv_dir, edge_csv_file=edge_csv_file, wholeMapDict=wholeMapDict)    
    newEdges, mergedIndex2us = createCompressStructMap(vertexMap, edgeMap)
    codedEdgeData = []
    for data in newEdges:
        codedEdgeData.append(codedData2id(data, wholeMapDict['char2id_dict'], wholeMapDict['id2char_dict']))
    codedEdgeArray = np.array([c for item in codedEdgeData for c in item], dtype=np.uint8)
    codedVertexArray = np.array(mergedIndex2us, dtype=object) 
    
    np.save(os.path.join(sc_dir, codedEdgeFile), codedEdgeArray) 
    np.save(os.path.join(sc_dir, codedVertexFile), codedVertexArray) 
    end_time = time.time()   
    logger.info(f'The time of struct compressing and coding is : {(end_time - start_time)} seconds')

    uuidIndex = flip_dict(wholeMapDict['uuid'])
    wholeMapDict['uuidIndex'] = uuidIndex     
    wholeMapDict, vertexMapDict = splitDict(wholeMapDict, ['uuid', 'uuidIndex'])

    with open(os.path.join(sc_dir, wholeMapDictFile), mode='w') as f:
        json.dump(wholeMapDict, f, indent=4)
    with open(os.path.join(sc_dir, vertexMapDictFile), mode='w') as f:
        json.dump(vertexMapDict, f, indent=4)
