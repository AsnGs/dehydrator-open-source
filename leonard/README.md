We fine-tuned Leonard's source code to properly ingest the data required by Dehydrator and have open-sourced the modified code along with the corresponding dataset. 

Specifically, we adapted the script (/dehydrator-open-source/leonard/CADETS-E3/data/parse_vertex_ef_removeedge_pid.py) to process the source data from CADETS-E3 and uploaded the original edge table (edge.csv) as well as the calibration table (table.params.json) generated during the experiments. 

Due to GitHub's file size limitations, we compressed both the original edge table (leonard/CADETS-E3/raw_data/edge.csv.zip) and the calibration table (leonard/CADETS-E3/src/table.params.json.gz), which originally occupied 1.29GB and 3.77GB, respectively. Reviewer #1 may decompress the edge table and follow Leonard's script workflow (https://github.com/dhl123/Leonard) to regenerate the calibration table.
