#!/bin/bash  
python -u ../data/parse_vertex_ef_removeedge_pid.py -edge_file ../data/edges.npy -input_path ../raw_data/vertex.csv -input_path1 ../raw_data/edge.csv -output ../data/vertex.npy -param ../data/vertex.params.json
wait
python -u trainer_test_darpa.py -d ../data/vertex.npy -epoch 15 -batchsize 4096 -model_name LSTM_multi -name vertex.hdf5 -log_file ../data/logs_data/FC.log.csv -param ../data/vertex.params.json
wait
python transfer_edge_to_txt.py
wait
python only_lite_convert.py -model vertex.hdf5 -param ../data/vertex.params.json
wait
python -u check_prediction_time.py -gpu 0 -model vertex.hdf5 -model_name LSTM_multi -data ../data/vertex.npy -data_params ../data/vertex.params.json -model_path lite.h52048vertex.hdf5 -table_file table200m.params.json
wait
