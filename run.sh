#/usr/bin/env bash


echo "Predicting for spliting compound nouns"


INPUT_FILE="test_x"

OUTPUT_FILE="output_file"

# In my case, I install the tensorflow with python3.5 version
# So I run the python3. 

# The runninig command below is with no output file 
# then, the result would be generated in DEFAULT OUTPUTFILE:
# 1. 
# 2. 

# IF you wan to create result with output file named by you
# Type in OUTPUT FILE name like :
# python3 precidct.py 'input file' -o 'output file'
#python3 predict.py $INPUT_FILE -o $OUTPUT_FILE -n 1\




#CUDA_VISIBLE_DEVICES=0 python3 predict.py $INPUT_FILE -o $OUTPUT_FILE -n 1 > log/predict_log 
