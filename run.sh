#/usr/bin/env bash

echo "Predicting for spliting compound nouns"

INPUT_FILE="sample_x"

OUTPUT_FILE="output_file"

# In my case, I install the tensorflow with python3.5 version
# So I run the python3. 

# IF you wan to create result with output file named by you
# Type in OUTPUT FILE name like :
# python3 precidct.py 'input file' -o 'output file'
# python3 predict.py $INPUT_FILE -o $OUTPUT_FILE -n 1 

python3 predict.py $INPUT_FILE -o $OUTPUT_FILE -n 1 
