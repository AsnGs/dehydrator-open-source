########################################################
#
#                   Artifacts path
#
########################################################

# The directory of the raw logs
raw_dir = 'XXX/Darpa/CADETS-E3/'  #! You need to enter the path to the extracted log .json file here.

# The directory to save all artifacts
artifact_dir = './artifact/'

csv_dir = artifact_dir + "csv/"
sc_dir = artifact_dir + "sc/"   # structure compressed
log_dir = artifact_dir + "log/"
model_dir = artifact_dir + "model/"
ect_dir = artifact_dir + "ect/"  # error correlation table


########################################################
#
#                   File Name
#
########################################################

# csv File
#  For convenience, you can simply uncomment and use the small vertex CSV and edge CSV files we provide.
vertex_csv_file = 'vertex.csv'
edge_csv_file = 'edge.csv'
# vertex_csv_file = 'vertexSmall.csv'
# edge_csv_file = 'edgeSmall.csv'

# npy File
codedVertexFile = 'vertex.npy'
codedEdgeFile = 'edge.npy'
# codedVertexFile = 'vertexSmall.npy'
# codedEdgeFile = 'edgeSmall.npy'

# Map Dict File
wholeMapDictFile = 'wholeMapDict.json'
vertexMapDictFile = 'vertexMapDict.json'
ectDictFile = 'ErrorCorrelationTable.json'

#  Logger File
createDataLoggerFile = 'createData.log'
structCompressCodeLoggerFile = 'structCompressCode.log'
trainVertexModelLoggerFile = 'trainVertexModel.log'
trainEdgeModelLoggerFile = 'trainEdgeModel.log'
generateErrorTableLoggerFile = 'generateErrorTable.log'
queryLoggerFile = 'query.log'
createSpecificGraphFile = 'createSpecificGraph.log'


########################################################
#
#               Postgres
#
########################################################

# Database name
database = 'cadet_e3_dataset_db'

# Only config this setting when you have the problem mentioned
# in the Troubleshooting section in settings/environment-settings.md.
# Otherwise, set it as None
host = '/var/run/postgresql'
# host = None

# Database user
user = 'postgres'

# The password to the database user
password = 'postgres'

# The port number for Postgres
port = '5432'

########################################################
#
#               Graph semantics
#
########################################################

# The directions of the following edge types need to be reversed
edge_reversed = [
    "EVENT_RECVFROM",
    "EVENT_READ"
]

include_edge_type=[
    'EVENT_READ',
    'EVENT_WRITE',
    'EVENT_EXECUTE',
    'EVENT_RECVFROM',
    'EVENT_SENDTO',
    'EVENT_SENDMSG',
    'EVENT_FORK',
    'EVENT_CLONE'
]

relMap = {
 'EVENT_WRITE': 'EVENT_WRITE',
 'EVENT_READ' : 'EVENT_READ',
 'EVENT_EXECUTE': 'EVENT_EXECUTE',
 'EVENT_SENDTO': 'EVENT_WRITE',
 'EVENT_SENDMSG': 'EVENT_WRITE',
 'EVENT_RECVFROM': 'EVENT_READ',
 'EVENT_FORK': 'EVENT_FORK',
 'EVENT_CLONE': 'EVENT_FORK',
}

rel2id = {
 1: 'EVENT_WRITE',
 'EVENT_WRITE': 1,
 2: 'EVENT_READ',
 'EVENT_READ': 2,
 3: 'EVENT_EXECUTE',
 'EVENT_EXECUTE': 3,
 4: 'EVENT_FORK',
 'EVENT_FORK': 4,
}


########################################################
#
#               Raw Data Json Files
#
########################################################

filelist = ['ta1-cadets-e3-official.json',
 'ta1-cadets-e3-official.json.1',
 'ta1-cadets-e3-official.json.2',
 'ta1-cadets-e3-official-1.json',
 'ta1-cadets-e3-official-1.json.1',
 'ta1-cadets-e3-official-1.json.2',
 'ta1-cadets-e3-official-1.json.3',
 'ta1-cadets-e3-official-1.json.4',
 'ta1-cadets-e3-official-2.json',
 'ta1-cadets-e3-official-2.json.1']

params_list = [
    {"d_model": 64, "nhead": 2, "dim_feedforward": 256},
    {"d_model": 128, "nhead": 4, "dim_feedforward": 512},
    {"d_model": 192, "nhead": 6, "dim_feedforward": 768},
    {"d_model": 256, "nhead": 8, "dim_feedforward": 1024},
    {"d_model": 384, "nhead": 12, "dim_feedforward": 1536},
    {"d_model": 512, "nhead": 16, "dim_feedforward": 2048},
    {"d_model": 32, "nhead": 1, "dim_feedforward": 128}   
]

########################################################
#
#               Parameters
#
########################################################
minsTime=1522706863
