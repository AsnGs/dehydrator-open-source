Due to GitHub storage limitations, we provide only a small-scale CSV file dataset extracted from the original DARPA JSON data(`./CADETS-E3/artifact/csv/edgeSmall.csv/vertexSmall.csv`) (requires unzipping the .zip file).
***
For larger-scale experiments, you can download and unzip the data from the official DARPA website([DARPA3](https://drive.google.com/drive/folders/1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK)/[DARPA5](https://drive.google.com/drive/u/0/folders/1okt4AYElyBohW4XiOBqmsvjwXsnUjLVf)). Use our provided script (*createData.py*) to extract it into CSV files and store them in a suitable database.
***
*config.py* —— Stores filenames and paths. Some parts may need to be modified based on individual environment.

*createData.py* —— Extracts CSV files from the original JSON files and stores them in the artifact/csv folder

*generateCorrectionTable.py* —— Generates the error correction table and stores it in the artifact/ect folder

*models.py* —— Contains the data loader and various models

*query.py* —— Implements queries given a model, query node ID, and depth

*structCompressCode.py* — Filters field-level and structural-level redundancy

*train.py* — Trains the model

*utils.py* — Stores utility functions