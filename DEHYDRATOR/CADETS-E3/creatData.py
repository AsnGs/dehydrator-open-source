import os
import re
import csv
from tqdm import tqdm
import time
import logging

from config import *
from utils import *

# Setting for logging
logger = logging.getLogger("createData_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_dir + createDataLoggerFile)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def store_netflow(file_path, csv_file, filelist):
    # Parse data from logs
    netobj2hash = {} # {'UUID':[hash, nodeProperty], 'hash':UUID}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "NetFlowObject" in line:
                    try:
                        res = re.findall(
                            'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),',
                            line)[0]

                        nodeid = res[0] # UUID
                        srcaddr = res[2] # src ip
                        # srcport = res[3] # src port
                        dstaddr = res[4] # dst ip  
                        # dstport = res[5] # dst port

                        # nodeproperty = srcaddr + ":" + srcport + "->" + dstaddr + ":" + dstport
                        nodeproperty = srcaddr + "->" + dstaddr
                        netobj2hash[nodeid] = [nodeproperty]
                    except:
                        pass
    netobjuuidList = []
    csv_path = os.path.join(csv_dir, csv_file)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for hash_key, name in netobj2hash.items():
            writer.writerow([hash_key, name[0],'netflow'])
            netobjuuidList.append(hash_key)
    return netobjuuidList


def store_subject(file_path, csv_file, filelist):
    # Parse data from logs
    scusess_count = 0
    fail_count = 0
    # subject_objset = set()
    subject_obj2hash = {}  #
    for file in tqdm(filelist): 
        with open(file_path + file, "r") as f:
            for line in f:
                if "Event" in line:
                    subject_uuid = re.findall(
                        '"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}(.*?)"exec":"(.*?)"', line)
                    try:
                        subject_obj2hash[subject_uuid[0][0]] = [subject_uuid[0][-1]]
                        scusess_count += 1
                    except:
                        try:
                            subject_obj2hash[subject_uuid[0][0]] = "null"
                        except:
                            pass
                        fail_count += 1
    
    subjectuuidList = []
    csv_path = os.path.join(csv_dir, csv_file)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for hash_key, name in subject_obj2hash.items():
            writer.writerow([hash_key, name[0],'process'])
            subjectuuidList.append(hash_key)
    return subjectuuidList


def store_file(file_path, csv_file, filelist):
    file_node = set()
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "com.bbn.tc.schema.avro.cdm18.FileObject" in line:
                    Object_uuid = re.findall('FileObject":{"uuid":"(.*?)",', line)
                    try:
                        file_node.add(Object_uuid[0])
                    except:
                        print(line)

    file_obj2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line:
                    predicateObject_uuid = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}',
                                                      line)
                    if len(predicateObject_uuid) > 0:
                        if predicateObject_uuid[0] in file_node:
                            if '"predicateObjectPath":null,' not in line and '<unknown>' not in line:  # predicateObjectPath不能为空
                                path_name = re.findall('"predicateObjectPath":{"string":"(.*?)"', line)
                                file_obj2hash[predicateObject_uuid[0]] = path_name
    
    fileuudiList = []
    csv_path = os.path.join(csv_dir, csv_file)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for hash_key, name in file_obj2hash.items():
            writer.writerow([hash_key, name[0],'file'])
            fileuudiList.append(hash_key)
    return fileuudiList

def store_event(file_path, reverse, subject_uuid2hash, file_uuid2hash, net_uuid2hash, csv_file, filelist):
    minTime = sys.maxsize
    # valid_subjects = set(subject_uuid2hash)
    # valid_allnodes = set(subject_uuid2hash + file_uuid2hash + net_uuid2hash)
    # relMapKeys = set(relMap.keys())

    subject_uuid_pattern = re.compile('"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}')
    predicateObject_uuid_pattern = re.compile('"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}')
    type_pattern = re.compile('"type":"(.*?)"')
    timestamp_pattern = re.compile('"timestampNanos":(.*?),')

    datalist = []
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line:
                    relation_type_match = type_pattern.search(line)
                    if relation_type_match:
                        relation_type = relation_type_match.group(1)
                        if relation_type in include_edge_type:
                            relation_type = relMap[relation_type]  # sendto -> write
                            subject_uuid_match = subject_uuid_pattern.search(line)
                            predicateObject_uuid_match = predicateObject_uuid_pattern.search(line)
                            if subject_uuid_match and predicateObject_uuid_match:
                                subject_uuid = subject_uuid_match.group(1)
                                predicateObject_uuid = predicateObject_uuid_match.group(1)
                                # if subject_uuid in valid_subjects and predicateObject_uuid in valid_allnodes:
                                time_rec_match = timestamp_pattern.search(line)
                                if time_rec_match:
                                    time_rec = time_rec_match.group(1)
                                    if int(time_rec[:10]) < minTime: minTime = int(time_rec[:10])
                                    subjectId = subject_uuid
                                    objectId = predicateObject_uuid
                                    if relation_type in reverse:  # reverse
                                        datalist.append(
                                            [objectId, subjectId, relation_type, time_rec[:10]])  # timestamp 保存前 10 位到秒钟即可
                                    else :
                                        datalist.append(
                                            [subjectId, objectId, relation_type, time_rec[:10]])

    csv_path = os.path.join(csv_dir, csv_file)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for item in datalist:
            writer.writerow(item) 
    return minTime

if __name__ == "__main__":
    print("-----Generate CSV from Raw Logs -----")

    os.makedirs(csv_dir, exist_ok=True)
    with open(os.path.join(csv_dir, vertex_csv_file), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['uuid', 'name', 'type'])
    with open(os.path.join(csv_dir, edge_csv_file), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['src', 'dst', 'operation', 'timestamp'])

    print("-----Processing Vertex-----")
    start_time = time.time()
    netobj2hash = store_netflow(file_path=raw_dir, csv_file=vertex_csv_file, filelist=filelist)
    file_obj2hash = store_file(file_path=raw_dir, csv_file=vertex_csv_file, filelist=filelist)
    subject_obj2hash = store_subject(file_path=raw_dir, csv_file=vertex_csv_file, filelist=filelist)
    end_time = time.time()   
    logger.info(f'The time of extracting vertex info is : {(end_time - start_time)} seconds')
    print("-----Vertex CSV File Created-----\n")
    
    print("-----Processing Edges-----")
    start_time = time.time()
    minTime = store_event(
        file_path=raw_dir,
        reverse=edge_reversed,
        subject_uuid2hash=subject_obj2hash,
        file_uuid2hash=file_obj2hash,
        net_uuid2hash=netobj2hash,
        csv_file=edge_csv_file,
        filelist=filelist
    )
    end_time = time.time()   
    logger.info(f'The time of extracting edge info is : {(end_time - start_time)} seconds')
    with open(os.path.join('config.py'), mode='a+', newline='') as file:
        file.write('minsTime={}\n'.format(minTime))   
    print("-----Edge CSV File Created-----\n")



