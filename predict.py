"""Korean natural langauge processing with generating tag and raw for autospacing based word

This used data from separation compound nouns 2018 task

IF you want to know about it in detail, visit here, https://sites.google.com/site/koreanlp2018/task-1

Just if you have data set for compound nouns, I will deal with that data like the following :

for example,

    '자그마한 연못가에 오이 시렁 하나, 오이 넝쿨엔 헛된 꽃이 달리지 않는다.
    
Basic concept : BiLSTM + CRF()
"""
import os
import sys
import numpy as np
import tensorflow as tf
import data_helper
from tqdm import tqdm, trange
from time import sleep

import argparse
import post_processing

##################
# Version Check  #
##################

DEBUGGING_MODE = False

if DEBUGGING_MODE:
    print("The Version of system:\n{}".format(sys.version))
    print("Tensorflow Version: {}".format(tf.__version__))

######################
# DATA CONFIGURATION #
######################

data2idx = None
idx2data = None

idx2label = ["B", "I"]
label2idx = {"B":0, "I":1}

##################
# Data Location  #
##################

RESULT_DIR = os.path.join(os.getcwd(), "result")

def result_dir():
    if os.path.isdir(RESULT_DIR):
        pass
    else:
        os.mkdir(RESULT_DIR)
        if DEBUGGING_MODE:
            print("\n {} was created".format(RESULT_DIR))

PREDICTION_OUTPUT_FILE = os.path.join(RESULT_DIR, "PREDICTIONS_TAGS")

# First you have to have vocaburies
VOCABULARY_FILE = data_helper.VOCA_FILE #data_helper.SAMPLE_VOCA_FILE # None

TRUTH_FILE = os.path.join(RESULT_DIR, "PREDICTIONS_RESULT")

MODEL_FILE = os.path.join(os.getcwd(), "model")

if DEBUGGING_MODE:
    print("\n============= Your file's locations ==============")
    print("PREDICTION_OUTPUT_FILE: {}".format(PREDICTION_OUTPUT_FILE))
    print("VOCABULARY_FILE: {}".format(VOCABULARY_FILE))
    print("TRUTH_OUTPUT_FILE: {}".format(TRUTH_FILE))

#####################
#  DATA PREPARATION #
#####################

data2idx, idx2data = data_helper.read_voca_dict(VOCABULARY_FILE)

#####################
# Hyper parameter   #
#####################

BATCH_SIZE = 1  

VOCABULARY_SIZE = len(data2idx)
EMBEDDING_SIZE = 300 

HIDDEN_UNITS_SIZE = EMBEDDING_SIZE 

# Option For BiLSTM
ADD_OR_CONCAT = ["ADD", "CONCAT"]

BI_DIRECTION = BI_GRAM_AND_SYLLABLE = ADD_OR_CONCAT[1]

NUM_FEATURES = None

if BI_DIRECTION == BI_GRAM_AND_SYLLABLE and BI_DIRECTION == "ADD":
    NUM_FEATURES = EMBEDDING_SIZE
else:
    NUM_FEATURES = EMBEDDING_SIZE*4
    
NUM_TAGS = 2 # B or I

LEARNING_RATE = 0.001

BATCH_MAX_LENGTH = -1 # None

if DEBUGGING_MODE:
    print("\n============== Checking your hyperparameter =============")
    print("BATICH_SIZE: {}".format(BATCH_SIZE))
    print("VOCABULARY_SIZE: {}".format(VOCABULARY_SIZE))
    print("EMBEDDING_SIZE: {}".format(EMBEDDING_SIZE))
    print("BI_DIRECTION: {}".format(BI_DIRECTION))
    print("BI_GRAM_AND_SYLLABLE: {}".format(BI_GRAM_AND_SYLLABLE))
    print("NUM_FEATURES: {}".format(NUM_FEATURES))
    print("NUM_TAGS: {}".format(NUM_TAGS))
    print("LEARNING_RATE: {}".format(LEARNING_RATE))
    print("BATCH_MAX_LENGTH: {}".format(BATCH_MAX_LENGTH))
    print("MODEL_PATH:{}".format(MODEL_PATH))

######################
# Graph Input part   #
######################

max_length_per_a_batch = tf.placeholder(tf.int32,name="MAX_LENGTH_PER_A_BATCH")

syllable_char = tf.placeholder(tf.int32, (None, None), name="SYLLABLE_CHAR") # shape=(BATCH_SIZE, BATCH_MAX_LENGT)
bi_gram_word = tf.placeholder(tf.int32, (None, None), name="BI_GRAM_WORD") # shape=(BATCH_SIZE, BATCH_MAX_LENGTH)


max_syllable_sequences = tf.placeholder(tf.int32, (None), name="MAX_SYLLABLE_CHAR") # shaep = (BATCH_MAX_LENGTH)
max_bi_gram_sequences = tf.placeholder(tf.int32, (None), name="MAX_BI_GRAM") # shape = (BATCH_MAX_LENGTH)

labels = tf.placeholder(tf.int32, (None, None), name="LABEL") # shape =(BATCH_SIZE, BATCH_MAX_LENGTH)

# For embedding lookup table for bi gram word, 
# syllable char and another token like <PAD>, <UNK>, </WORD_END>
padding_matrix = tf.constant(0.0, shape=[1, EMBEDDING_SIZE], dtype=tf.float32, name="PADDING_MATRIX")
embedding_matrix = tf.get_variable(name="EMBEDDING_MATRIX", shape=[VOCABULARY_SIZE-1, EMBEDDING_SIZE], dtype=tf.float32)

concat_embedding_matrix = tf.concat([padding_matrix, embedding_matrix], 0)

syllable_char_embeddings = tf.nn.embedding_lookup(concat_embedding_matrix, syllable_char, name="SYLLABLE_CHAR_EMBEDDING")
bi_gram_word_embeddings = tf.nn.embedding_lookup(concat_embedding_matrix, bi_gram_word, name="BI_GRAM_EMBEDDING")


# for CRF 
weights =tf.get_variable("CRF_WEIGHTS", [NUM_FEATURES, NUM_TAGS])
bias = tf.get_variable("CRF_BIAS", [NUM_TAGS])


if DEBUGGING_MODE:
    print("\n=================== Checking Graph input ========================")
    print("\n++++++++++ placeholder +++++++++++")
    print("max_length_per_a_batch: {}".format(max_length_per_a_batch))
    print("syllable_char: {}".format(syllable_char))
    print("bi_gram_word: {}".format(bi_gram_word))
    print("max_syllable_sequences: {}".format(max_syllable_sequences))
    print("max_bi_gram_sequences: {}".format(max_bi_gram_sequences))
    print("labels: {}".format(labels))
    print("\n++++++++++ Embedding matrix ++++++++++")
    print("padding_matrix: {}".format(padding_matrix))
    print("embedding_matrix: {}".format(embedding_matrix))
    print("concat_embedding_matrix: {}".format(concat_embedding_matrix))
    print("syllable_char_embeddings: {}".format(syllable_char_embeddings))
    print("bi_gram_word_embeddings: {}".format(bi_gram_word_embeddings))
    print("weights: {}".format(weights))
    print("bias: {}".format(bias))
  
##########
# BiLSTM #
##########
syllable_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS_SIZE, name="SYLLABLE_FORWARD_LSTM")
syllable_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS_SIZE, name="SYLLABLE_BACKWARD_LSTM")

(output_syllable_fw, output_syllable_bw), output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=syllable_fw_cell,
                                                                                          cell_bw=syllable_bw_cell,
                                                                                          inputs=syllable_char_embeddings,
                                                                                          sequence_length=max_syllable_sequences,
                                                                                          dtype=tf.float32)

bi_gram_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS_SIZE, name="BI_GRAM_FORWARD_LSTM")
bi_gram_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS_SIZE, name="BI_GRAM_BACKWARD_LSTM")

(output_bi_gram_fw, output_bi_gram_bw), output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=bi_gram_fw_cell,
                                                                                          cell_bw=bi_gram_bw_cell,
                                                                                          inputs=bi_gram_word_embeddings,
                                                                                          sequence_length=max_bi_gram_sequences,
                                                                                          dtype=tf.float32)
if DEBUGGING_MODE:
    print("\n===================== BiLSTM Checking =======================")
    print("\n++++++++++++++++++ syllable BiLSTM ++++++++++++++++++++++++++")
    print("syllable_fw_cell: {}".format(syllable_fw_cell))
    print("syllable_bw_cell: {}".format(syllable_bw_cell))
    print("output_syllable_fw: {}".format(output_syllable_fw))
    print("output_syllable_bw: {}".format(output_syllable_bw))
    print("\n++++++++++++++++ bi gram BiLSTM +++++++++++++++++++++++++++++")
    print("bi_gram_fw_cell: {}".format(bi_gram_fw_cell))
    print("bi_gram_bw_cell: {}".format(bi_gram_bw_cell))
    print("output_bi_gram_fw: {}".format(output_bi_gram_fw))
    print("output_bi_gram_bw: {}".format(output_bi_gram_bw))

if BI_DIRECTION == "ADD":
    syllable = tf.add(output_syllable_fw, output_syllable_bw)
    bi_gram = tf.add(output_bi_gram_fw, output_bi_gram_bw)
elif BI_DIRECTION == "CONCAT":
    syllable = tf.concat([output_syllable_fw, output_syllable_bw], -1)
    bi_gram = tf.concat([output_bi_gram_fw, output_bi_gram_bw], -1)

if BI_GRAM_AND_SYLLABLE == "ADD":
    predictions = tf.add(syllable, bi_gram)
elif BI_GRAM_AND_SYLLABLE == "CONCAT":
    predictions = tf.concat([syllable, bi_gram], -1)

if DEBUGGING_MODE:
    print("\n============== Checking ADD or Concat ==================")
    print("\n++++++++++++++ First ++++++++++++++++++++++")
    print("syllable: {}".format(syllable))
    print("bi_gram: {}".format(bi_gram))
    print("prediction: {}".format(predictions))


# For CRF log_likelihood
matricized_predictions = tf.reshape(predictions, [-1, NUM_FEATURES])
matricized_unary_scores = tf.add(tf.matmul(matricized_predictions, weights), bias)
unary_scores = tf.reshape(matricized_unary_scores, [-1, max_length_per_a_batch, NUM_TAGS])

if DEBUGGING_MODE:
   print("\n==================== Before crf =============================")
   print("matricized_predictions: {}".format(matricized_predictions))
   print("matricized_unary_score: {}".format(matricized_unary_scores))
   print("unary_scores: {}".format(unary_scores))


log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores, labels, max_syllable_sequences)

# IF you want to prediction, ust this function.
viterbi_seqeunce, viterbi_score = tf.contrib.crf.crf_decode(unary_scores, transition_params, max_syllable_sequences)

if DEBUGGING_MODE:
   print("\n========================= viterbi and log_likelihood ===================")
   print("log_likelihood: {}".format(log_likelihood))
   print("transition_params: {}".format(transition_params))
   print("viterbi_sequence: {}".format(viterbi_seqeunce))
   print("vierbi_score: {}".format(viterbi_score))

# losses evaluation
losses = tf.reduce_mean(-log_likelihood)
tf.summary.scalar("LOSS", losses)

# unpdating 
global_step_var = tf.Variable(0, dtype=tf.int32, trainable=False, name="GLOBAL_STEP")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
training_op = optimizer.minimize(losses, global_step=global_step_var)

init_op = tf.global_variables_initializer()

merged_op = tf.summary.merge_all()

saver = tf.train.Saver()
if DEBUGGING_MODE: 
    ###################################
    #    Graph's Checking variable    #
    ###################################
    trainable_variable1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    trainable_variable2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    trainable_variable3 = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
    print("\n===== variable type =====")
    print("tf.GraphKeys.GLOBAL_VARIABLES: {}".format(trainable_variable1))
    print("tf.GraphKeys.TRAINABLE_VARIABLES: {}".format(trainable_variable2))
    print("tf.GraphKeys.LOCAL_VARIABLES: {}".format(trainable_variable3))                                                                                          

    print("\n\n===== all variables =====")
    for v in tf.global_variables():
        print(v.name)

def load_predinction_data(input_file, data_indexing, label_indexing, batch, randomly_selecting=True):
    """Load prediction data 

    This function would load data like 
     
    examples: 
            - 류보청
            - 류머티즘열

    Args:
        input_file(str): input file name and location
        data_indexing(dict): vocabuaries dictionary
        label_indexing(dict): labels dictionary
        batch(int): the size you want to make for a batch 
        randomly_selecting(bool): selecting shuffling data to create bathces for a epoch

    Returns:
        syllable_x(list): syllable list per each batches
        bi_gram_x(list): bi gram lise per each batches

    """

    if input_file == None:
        raise("Your input file is wrong!!!")

    total = data_helper.read_prediction_data(input_file, data_indexing, label_indexing)
   
    syllable_x, bi_gram_x = data_helper.gathering_batches_for_predictions(total, n_per_a_batch = batch, random=randomly_selecting, padding=True)

    if DEBUGGING_MODE:
        print("\n========================= Calling load_prediction_data function =================")
        print("syllable_x: {}\n{}".format(len(syllable_x), syllable_x[:10]))
        print("bi_gram_x: {}\n{}".format(len(bi_gram_x), bi_gram_x[:10]))
    
    return syllable_x, bi_gram_x

def _get_max_length(syllable_x, bi_gram_x):
    """Get maximum length 

    This function would return maximum size per a batch

    Args:
        syllable_x(list): syllable batches
        bi_gram_x(list): bi gram batches

    Returns: 
        syllable_max_length(list): each max length per each batches in syllabe_x
        bi_gram_max_length(list): each max length per each batches in bi_gram_x 

    """
    syllable_max_length =  data_helper.max_length_of_sequences(syllable_x)

    bi_gram_max_length = data_helper.max_length_of_sequences(bi_gram_x)

    if DEBUGGING_MODE:
        print("\n==================== Calling _get_max_length function ===========================")
        print("syllable_max_length: {}\n{} : {}".format(len(syllable_max_length), syllable_x, syllable_max_length[:10]))
        print("bi_gram_max_length: {}\n{} : {}".format(len(bi_gram_max_length), bi_gram_x, bi_gram_max_length[:10]))

    return syllable_max_length, bi_gram_max_length

def write_tag_results(path, data_list):
    """Write the tags predicted 

    Args:
        path(str): the location written with tags predicted
        data_list(list): tags predicted by My model

    Return:
        None
    """
    with open(path, "w") as f:
        for line_idx, line_val in enumerate(data_list):
            f.write(line_val+"\n")

def put_prediction_of_tags_together(collections, data_list, max_len):
    """gather tags predicted and each length per a data

    Args:
       collections(list): 
       data_list(list):
       max_len(list):
    
    Return: 
       None
    """
    for line_idx, line_val in enumerate(data_list):
        temp_label = ""
        for token_idx, token_val in enumerate(line_val):
            if token_idx < max_len[line_idx]:
                  temp_label += idx2label[token_val]
            else:
               break
        collections.append(temp_label)      
                    
# Prediction model
def predict(input_file, output_file):
    with tf.Session() as sess:
    
        ckpt = tf.train.get_checkpoint_state(MODEL_FILE)
        if ckpt and ckpt.model_checkpoint_path:
            if DEBUGGING_MODE:
                print("\n===== loading model to split compound nouns ===============")
                print("COMPOUND NOUNS Model: {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

        predictions_of_tags = list()
        for epoch_idx in range(1): 
            syllable_predictions_x, bi_gram_predictions_x = load_predinction_data(input_file, data2idx, label2idx, batch=BATCH_SIZE, randomly_selecting=False)
            syll_max_length_predictions, bi_gram_max_length_predictions = _get_max_length(syllable_predictions_x, bi_gram_predictions_x)

            for idx in tqdm(range(len(syllable_predictions_x))):
                BATCH_MAX_LENGTH = len(syllable_predictions_x[idx][0])

                tf_viterbi_sequence = sess.run(viterbi_seqeunce, feed_dict ={syllable_char : syllable_predictions_x[idx], bi_gram_word : bi_gram_predictions_x[idx], max_syllable_sequences : syll_max_length_predictions[idx], max_bi_gram_sequences : bi_gram_max_length_predictions[idx], labels : [[1]], max_length_per_a_batch : BATCH_MAX_LENGTH})   

              
                if DEBUGGING_MODE:
                    print("\n============= Your OUTPUT FILE ==============")
                    print("Check The file {}".format(output_file))
                tf_viterbi_sequence_list = tf_viterbi_sequence.tolist()
                correct_label_len = syll_max_length_predictions[idx]
                put_prediction_of_tags_together(predictions_of_tags, tf_viterbi_sequence_list, correct_label_len)

        write_tag_results(output_file, predictions_of_tags)
        if DEBUGGING_MODE: 
            print("\n========== Checking the file below for tag sequence ================")
            print("Check the file named {}".format(output_file))
        print("Prediction is done!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "split program for compound nouns")
    parser.add_argument("input", help="input file")
    parser.add_argument("-o","--output", help="output file", default=TRUTH_FILE)
    parser.add_argument("-n", "--number", type=int, help="Batch size, this must be less than your number of data, and memory size", default=BATCH_SIZE)
    args = parser.parse_args()


    if DEBUGGING_MODE:
       print("\n======= Checking the arguments ================")
       print("Checking arguments : {}".format(args))

    if args.input:
        result_dir()
        input_file = args.input
        print("\n\n============== Your input file is : =================")
        print("{}".format(input_file))
    else:
        try:
           raise()
        except:
           print("type in your input file like \n python3 predicte.py 'input_file_name'")

    if args.output:
        output_file = os.path.join(RESULT_DIR, args.output)
        print("\n================ Your output file is: ====================")
        print("{}".format(output_file))
    else:
        print("\n================ Your output file is: ====================")
        print("Since you didn't type in your output file name,") 
        print("your output file is by default {}".format(TRUTH_FILE))

    if args.number:
        BATCH_SIZE=args.number
        print("\n============= your batch size: {} ====================".format(BATCH_SIZE))
    else:
        print("\n============= your batch size: {} ====================".format(BATCH_SIZE))
        print("Since you didn't type in your batch size you want")
        print("System will work at the default size, 1")
        print("IF you want to configure batch size, type in 'n' option like  the following :")
        print("python3 predict input_file -o output_file -n BATCH_SIZE")

    # Prediction
    predict(input_file, PREDICTION_OUTPUT_FILE)   

    # OUTPUT
    inputs, tags = post_processing.read_input_and_prediction(input_file, PREDICTION_OUTPUT_FILE)
    nouns_data_spaced = post_processing.truth_check(inputs, tags)
    post_processing.write_file(output_file, nouns_data_spaced)



    
