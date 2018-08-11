"""post-processing for split of compound noun of Korean language

This code is post-processing after evaluation of Neural networ(BiLSTM + CRF()) 

I made this code for https://sites.google.com/site/koreanlp2018/task-2

e.g. 
   <input>          <output>
   류보청      ->   류보 청
   류머티즘열  ->   류머티즘 열
"""
import sys
import os
import data_helper

TOKEN="\n"

B_TOKEN = "B"
I_TOKEN = "I"

DEBUGGING_MODE = False 

SAMPLE_RESULT_DIR = os.path.join(os.getcwd(), "sample_result")

def sample():
    if os.path.isdir(SAMPLE_RESULT_DIR):
        pass
    else: 
        os.mkdir(SAMPLE_RESULT_DIR)
        if DEBUGGING_MODE:
            print("\n {} was created".format(SAMPLE_RESULT_DIR))
   
# Input file 
PREDICTION_INPUT_FILE = data_helper.SAMPLE_TEST_X_FILE #os.path.join(os.getcwd(), "sample_test_x")
# output file 
PREDICTION_OUTPUT_FILE = os.path.join(SAMPLE_RESULT_DIR, "PREDICTIONS_TAGS")

# print answer!
TRUTH_FILE= os.path.join(SAMPLE_RESULT_DIR, "TRUTH_PREDICTED")

TRUTH_N_GRAM_FILE = os.path.join(SAMPLE_RESULT_DIR, "TRUTH_PREDICTED_WITH_N_GRAM")
 
N_GRAM_FILE = os.path.join(os.getcwd(), "data/five_gram_dict")

def write_file(path, data):
    """Write file 

    Args: 
        path(str):
        data(list):
   
    Return:
        None
    """
    with open(path, "w") as f:
        for line_idx, line_val in enumerate(data):
            f.write(line_val+"\n")

def read_file(path):
    """Read file 

    Args: 
       path(str): file name and loction to read 

    Return:
       temp_lines(list): the resulting list from input file line by line 
     
    """
    with open(path, "r") as f:
        temp_lines = [line.strip() for line in f.readlines() if line != TOKEN]
    
    if DEBUGGING_MODE:
       print("\n======== Read file named {} ============".format(path))
       print("lines: type-{}, len-{}\n{}".format(type(temp_lines), len(temp_lines), temp_lines[:10]))
    
    return temp_lines


def read_input_and_prediction(input_file, tags_predicted):
    """Read input and prediction value of deep learning

    Args:
        input_file(str): file name and location for predict.py
        tags_predicted(str): file name and location of predicted tags from predict.py

    Returns:
        input_lines(str): data list line by line for predict.py
        tag_lines(str): tag list resulted from predict.py
    """
    input_lines = read_file(input_file)
    tag_lines = read_file(tags_predicted)

    if DEBUGGING_MODE:
       assert len(input_lines) == len(tag_lines), "plz, check your number of lines in both input and output file"

    return input_lines, tag_lines


def read_five_gram_dict(path):
    """Read dictionary for post processing
 
    Arg:
        path(str): file name and location 
 
    Return:
        temp_linse: the result from reading the input file of five gram dictionary 
    """
    with open(path, "r") as f:
        temp_lines = [line.strip().split() for line in f.readlines() if line != TOKEN]

    if DEBUGGING_MODE:
       print("\n======== Read file named {} ============".format(path))
       print("lines: type-{}, len-{}\n{}".format(type(temp_lines), len(temp_lines), temp_lines[:10]))
   
    return temp_lines

def make_five_gram_dict(n_gram_dict): 
    """make dictrionary data structure for search

    Arg:
        n_gram_dict(list): the result from read_five_gram_dict function 

    Return:
        n_gram(dict): dictrionary data structure from read_five_gram_dict function
    """
    n_gram = dict()

    for line_idx, line_val in enumerate(n_gram_dict):
        if n_gram.get(line_val[0], None) == None:
            n_gram[line_val[0]] = [line_val[1]+" "+line_val[2], int(line_val[3])]

    if DEBUGGING_MODE:
        print("\n========== check your dict ===============")
        print("N_gram: type-{}, len-{}".format(type(n_gram), len(n_gram)))
        counting = 0
        for key, val in n_gram.items():
            if counting < 10:
                print("({}, {}), ".format(key, val), end=" ")
            counting +=1    
    return n_gram

def truth_check(input_data, tags_predicted_data):
    """spacing with input data and tags predicted 
    
    Args:
        input_data(list): data list for predict.py
        tags_predicted_data(list): tags list predicted by predict.py

    Return:
        nouns_spaced(list): The result of spacing with two arguments
    """
    nouns_spaced = list()
    for lines_idx, line_val in enumerate(input_data):
        temp_str  = ""
        for syll_idx, syll_val in enumerate(line_val):
            if syll_idx == 0:
                temp_str += syll_val
            else:
                if tags_predicted_data[lines_idx][syll_idx] == B_TOKEN:
                   temp_str += (" "+syll_val)
                else:
                   temp_str += syll_val

        nouns_spaced.append(temp_str)

    if DEBUGGING_MODE: 
        print("\n=============== checking spacing is correct ==============")
        print("ORIGINAL DATA: type-{}, len-{}\n{}".format(type(input_data), len(input_data), input_data[:10]))
        print("CHANGED DATA: type-{}, len-{}\n{}".format(type(nouns_spaced), len(nouns_spaced), nouns_spaced[:10]))
    
    return nouns_spaced

# more checking just string all not       
def n_gram_option(input_data, n_gram, option=1000):
    """final post processing 

    To-DO


    """
    for line_idx, line_val in enumerate(input_data):
        if not " " in line_val:
         

            if n_gram.get(line_val, None) != None:
                if n_gram[line_val][1] >= option:
                    if DEBUGGING_MODE:
                        print("\n=========== Change no spacing inot spacing ===============")
                        print("Original: idx{}: {}".format(line_idx, input_data[line_idx]))
                        print(n_gram[line_val])
                    #start_dix = input_data[line_idx].find(
                    input_data[line_idx] = n_gram[line_val][0]
                    if DEBUGGING_MODE:
                        print("After changing: idx{}: {}".format(line_idx, input_data[line_idx]))



if __name__ == "__main__":
     sample()
     inputs, tags = read_input_and_prediction(PREDICTION_INPUT_FILE, PREDICTION_OUTPUT_FILE)
     nouns_data_spaced = truth_check(inputs, tags)

     write_file(TRUTH_FILE, nouns_data_spaced)

     if False:
         lines = read_file(TRUTH_FILE)
         n_gram_data = read_five_gram_dict(N_GRAM_FILE)
         n_gram = make_five_gram_dict(n_gram_data)
         n_gram_option(lines, n_gram, option=1)
    
         write_file(TRUTH_N_GRAM_FILE, lines)


