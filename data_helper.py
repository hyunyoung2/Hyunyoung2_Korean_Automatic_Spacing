"""Data pre-processing helper for Tensorflow 

pre-processing code makes vocabularies dictionary and train data and test data for preparation of training on Tensorflow

Also before putting data on tf.placeholder, 

This code deals with make batch, syllable to bi gram sillable, indexing syllable into number
"""

import os
import numpy as np
from random import seed
from random import shuffle
from collections import Counter

TOKEN="\n"

END_TOKEN="</WORD_END>"
UNK_TOKEN="<UNK>"
PAD_TOKEN="<PAD>"


DIR = os.path.join(os.getcwd(),"data")

if os.path.isdir(DIR):
    pass
else:
    try:
        raise()
    except:
        print("You don't have directory name {}!".format(DIR))
 
TOTAL_VOCA = "vocabularies"

# Vocabularies 
VOCA_FILE = os.path.join(DIR, TOTAL_VOCA)

# For DATA
IDX2DATA = list()
DATA2IDX = dict()

# For LABEL
IDX2LABEL = ["B", "I"]
LABEL2IDX = {"B":0, "I":1}

# SAMPLE DATA
TEST_X = ["한편정부는16일'살충제계란'농가4곳이추가확인됐다고밝혔다.",
          "해답이뻔한데도절차적정당성이니환경영향평가니하며먼길을돌았다.",
          "헬스조선비타투어는10월14일~26일(11박13일)'발칸유럽힐링크루즈'를진행한다.",
          "혁신위는선언문에박근혜전대통령탄핵에대한입장도담지않았다.",
          "현정부출범직후인지난5월접수된총민원은26만건.",
          "현대건설측은'한국에돌아오는대로정밀진단을받을계획'이라고했다.",
          "호날두의라이벌리오넬메시(30·바르셀로나)도마찬가지혐의로재판에넘겨져유죄판결을받았다.",
          "화주스님은그엄마의집을찾아기어이아이를받아냈다.",
          "회의든식사든상석에앉은적도없다'며'최순실·정유라가누구인지도몰랐다'고말했다.",
          "효과성미흡이끊임없이지적돼온사회보험지원사업은오히려확대했다.",
          "휴가중인대통령은휴가가끝나면트럼프대통령과통화하겠다고한다.",
          ]

TEST_Y = ["BIBIIBIIBIIIBIIBIBIIBIBIIIIBIII",
          "BIIBIIIBIIBIIIIBIIIIIIBIBBIBIII",
          "BIIIBIIIIBIIBIIIIIIIIIIBIIIBIIBIBIBIIIIBIIII",
          "BIIIBIIIBIIBBIIBIIBIBIIBIBIII",
          "BBIBIBIIBIBIBIIBBIIBIIBI",
          "BIIIBIBIIIBIIIBIBIBIIBIBIIIIIBII",
          "BIIIBIIBIIBIIIIIIIIIIIIBIIIBIIBIIBIIBIBIIBIII",
          "BIBIIBBIIBIBIBIIBIIBIIII",
          "BIIBIIBIIBIBIBIIIBIIIIIIIIBIIIIBIIIIBIII",
          "BIIBIIBIIIBIIIBIIIBIBIIBIIBIIII",
          "BIBIBIIIBIIBIIBIIBIIIBIIIIIBII",
          ]

DEBUGGING_MODE = False 

def _error_check_line(data_list, label_list):
    """Check error which is the same lenght between two list

    Args:
        data_list(list): data list that I read named 'no_spacing_word'
        label_list(list): label list that I read named 'no_spacing_BI_tag'

    Return:
        None
    """
    assert len(data_list) == len(label_list), "your data is wrong because the length between data and lable is different"


def _error_check_char(data_list, label_list):
    """Check error if each line has the same length

    Args:
       data_list(list): data list that I read named 'no_spacing_word'
       label_list(list): label list that I read named 'no_spacing_BI_tag'

    Return:
       None
    """
    for idx, val in enumerate(data_list):
        if len(val) != len(label_list[idx]):
            print("ERROR{}: {} != {}".format(idx, val, label_list[idx]))


def _write_file(path, data):
    """write file for VOCABULARIES FILE named 'vocabularies'

    Args:
        path(str): Vocabularies file location you want to store 
        data(list): Vocabularies data list

    Return:
        None
    """
    with open(path, "w") as f: 
        for idx, val in enumerate(data):
            f.write(str(idx)+"\t"+val+"\n")
   
def read_file(path):
    """Read files like, in this case, data and label files

    Args:
       path(str): file location like data file named 'no_spacing_word' and label file namded 'no_spacing_BI_tag'
   
    Return:
       temp_list(list): the result read from file, line by line.
    """
    with open(path, "r") as f: 
        temp_list = [line.strip() for line in f.readlines() if line != TOKEN]

    if DEBUGGING_MODE:
        print("\n============== read file name {} ============".format(path))
        print("DATA: type-{}, len-{}\n{}".format(type(temp_list), len(temp_list), temp_list[0:10]))
  
    return temp_list

def read_data_and_label(data_file, label_file):
    """Read data and file together 

    Args: 
       data_file(str): the file name and location for data file named 'no_spacing_word'
       label_file(str): teh file name and location for label file named 'no_spacing_BI_tag'
   
    Returns:
       data_list(list): the result read from file named 'no_spacing_word' line by line
       lable_list(list): the result read from file named 'no_spacing_BI_tag' line by line 
    """
    data_list = read_file(data_file)
    label_list = read_file(label_file)
   
    if DEBUGGING_MODE:
        print("\n============== Comparing the lens between data file and label file ============")
        print("DATA: type-{}, len-{}\n{}".format(type(data_list), len(data_list), data_list[0:10]))
        print("LABEL: tyep-{}, len-{}\n{}".format(type(label_list), len(label_list), label_list[0:10]))
        _error_check_line(data_list, label_list)
        _error_check_char(data_list, label_list)

    return data_list, label_list

def maximum_length(data_list, name=None):
    """Search maximum length and data

    Args:
       data_list(list): data list to check what is maximum lenght and entry for _padding function 
                       - e.g. [[1,2,3,4], [1,2], [1]]
       name(str): For debugging, in order to check where do this function call?
    
    Returns:
       max_len(int): maximum length 
       max_len_data(list): maximum entry of data_list
    """
    # maximum data
    max_len_data = max(data_list, key=lambda x : len(x))
    max_len = len(max_len_data)
    if DEBUGGING_MODE:
        print("\n=========== Search maximum data ============")
        print("Maximu {}: len-{}\n{}".format(name, max_len, max_len_data))
 
    return  max_len, max_len_data


def _make_data_univoca(data_list):
    """Construct Univoca

    Arg:
        data_list(list): data list read from input file named 'no_spacing_word' line by line
   
    Return:
        uni_voca(list): the sorted uni_syllable of the whole words
    """
 
    syllable_counted = Counter("".join(data_list))
    uni_voca = sorted(syllable_counted.keys())  
   
    if DEBUGGING_MODE:
        print("\n============= univoca generation =========")
        print("UNI VOCA COUNTED: type-{}, len-{},\n{}".format(type(syllable_counted), len(syllable_counted), syllable_counted.most_common(10)))
        print("UNI VOCA DATA: type-{}, len-{}\n{}".format(type(uni_voca), len(uni_voca), uni_voca[0:10]))

    return uni_voca


def _separate_to_bi_gram(data_list):
    """split data from file named 'no_spacing_word' to bi_gram_syllable

    Arg:
      data_list(list): data list from file named 'no_spacing_word' line by line 

    Return:
      bi_gram_list(list): bi gram syllable including duplicate
    """
    # # of list
    bi_gram_list = list() 
    for line_idx, line_val in enumerate(data_list):
        # # of syllble 
        temp_bi = list()
        for syllable_idx, syllable_val in enumerate(line_val):
            # adding END_TOKEN
            if syllable_idx == len(line_val) - 1:
                temp_bi.append(syllable_val+END_TOKEN)
            else: 
                temp_bi.append(syllable_val+line_val[syllable_idx+1])

        bi_gram_list.append(temp_bi)

    if DEBUGGING_MODE: 
        print("\n========== bi gram generation ==============")
        print("BI GRAM: type-{}, len-{}\n{}".format(type(bi_gram_list), len(bi_gram_list), bi_gram_list[0:10]))
   
    return bi_gram_list

def _make_data_bivoca(data_list):
    """Make bi gram syllable dictionary 

    Arg: 
        data_list(list): data list from file named 'no_spacing_word' line by line 

    Return:
        bi_voca(list): bi vocabularies without duplicate
    """
    data = _separate_to_bi_gram(data_list)
    bi_gram_list = list()
     
    for idx, val in enumerate(data):
       bi_gram_list.extend(val)
    
    bi_gram_counted = Counter(bi_gram_list)
    bi_voca = sorted(bi_gram_counted.keys())

    if DEBUGGING_MODE:
        print("\n============= BI VOCA generation =========") 
        print("BI VOCA COUNTED: type-{}, len-{},\n{}".format(type(bi_gram_counted), len(bi_gram_counted), bi_gram_counted.most_common(10)))
        print("BI VOCA DATA: type-{}, len-{}\n{}".format(type(bi_voca), len(bi_voca), bi_voca[0:10]))

    return bi_voca


def make_data_voca(data_list, padding=True, unknown=True, end=True, bi=True):
    """Make the final vocabularies for the total dictionary

    This code makes a dictrionary including some tokens like <PAD>, <UNK>, and </WORD_END> 

    so the dictionary contains uni syllable vocabularies, bi syllable vocabularies and some tokens above.

    Args:
        data_list(list): data list from file named 'no_spacing_word' line by line 
        padding(bool): Check if I include <PAD> token in the dictionary
        unknown(bool): Check if I inlcude <UNK> token in the dictionary
        end(bool): Check if I include </WORK_END> token in the dictionary 
        bi(bool): Cehck if I inlcude bi syllable vocabularies in the dictionary 
    
    Return:
        temp_data2idx(dict): dictionary structure for indexing of voca
        temp_dix2data(list): voca list 
    """
    temp_idx2data = list()
    if padding: # <PAD>
        temp_idx2data.append(PAD_TOKEN)
    if unknown: # <UNK>
        temp_idx2data.append(UNK_TOKEN)
    if end: # </WORD_END>
        temp_idx2data.append(END_TOKEN)

    # make uni voca
    temp_idx2data += _make_data_univoca(data_list)

    if bi:
         # make bi voca
         temp_idx2data += _make_data_bivoca(data_list)

    temp_data2idx = dict()

    # dictinary {"token": index number} 
    for idx, val in enumerate(temp_idx2data):
        if temp_data2idx.get(val, None) == None:
            temp_data2idx[val] = idx
       
    if DEBUGGING_MODE:
        print("\n=========== make syllable dict list and index dict ======")
        check = True
        if len(temp_idx2data) == len(Counter(temp_idx2data)):
            check = True
        else:
            check = False
        print("duplicate check: {}".format(check))
        print("IDX2DATA: type-{},  len-{}\n{}".format(type(temp_idx2data), len(temp_idx2data), temp_idx2data[0:10]))
        print("DATA2IDX: type-{}, len-{}\n".format(type(temp_data2idx), len(temp_data2idx)))
        counting = 0 
        for key, val in temp_data2idx.items():
            if counting == 10:
                break
            print("({}: {})".format(key, val), end=", ")
            counting +=1
        print()

    return temp_data2idx, temp_idx2data


# From now on, for train and data set division
def _separate(data):
    """split zip of paris of (data, label) 

    Arg:
        data(list): zip of (data, label)
    
    Returns:
        x_data(list): syllable data
        y_data(list): label data
    """
    x_data = list()  #
    y_data = list()
    for line_idx, line_val in enumerate(data):
        x_data.append(line_val[0])
        y_data.append(line_val[1])

    if DEBUGGING_MODE:
        print("\n========== juse separation to x and y of data ==========")
        print("DATA: type-{}, len-{}\n{}".format(type(data), len(data), data[:10]))
        print("X_DATA: type-{}, len-{}\n{}".format(type(x_data), len(x_data), x_data[:10]))
        print("Y_DATA: type-{}, len-{}\n{}".format(type(y_data), len(y_data), y_data[:10]))

    return x_data, y_data   

def _write_train_and_test(path_x, path_y, data):
    """Write function data split

    Args:
        path_x(str): file name and location for data x
        path_y(str): file name and location for label y
        data(list): total data I want to write

    Return:
        None
    """
    x_data, y_data = _separate(data)

    with open(path_x, "w") as f: 
        for idx, val in enumerate(x_data):
            f.write(val+"\n")
    with open(path_y, "w") as f:
        for idx, val in enumerate(y_data):
            f.write(val+"\n")
    
def _get_test_len(data, percentage):
    """Get test pair's length 

    Args:
        data(list): total data set 
        percentage(float): percentage I want to make as test pairs

    Return:
        test_len(int): the total for test set 
    """
    total = len(data)
    test_len = np.ceil(total*percentage)
    
    if DEBUGGING_MODE:
        print("\n====== your test set length ==========")
        print("length: {}".format(test_len))

    return int(test_len)

# From now on, for train and test set
def _dividing_to_train_and_test(data, percentage):
    """Divide total data into two sets like train and test 

    Args:
        data(list): total data set 
        percentage(float): percentage for test pair

    Returns:
        data[:-data_test_len](list): training set 
        data[-data_test_len:](list): testing set
    """
    data_test_len = _get_test_len(data, percentage)
  
    return data[:-data_test_len], data[-data_test_len:]

def get_train_and_test_set(data, label, percentage=0.1, random=True):
    """Get train and test set

    Args: 
        data(list): data list from file named 'no_spacing_word'
        lable(list): label list from file named 'no_spacing_BI_tag'
        percentage(float): 

    Returns: 
        train(list): training set like (syllable, label)
        test(list): testing set like (syllable, label)
    """
    total_data = list(zip(data, label))

    if DEBUGGING_MODE:
        print("\n================ zip =======================")
        print("DATA: type-{}, len-{}\n{}".format(type(data), len(data), data[:10]))
        print("LABEL: type-{}, len-{}\n{}".format(type(label), len(label), label[:10]))
        print("TOTAL: type-{}, len-{}\n{}".format(type(total_data), len(total_data), total_data[:10]))

    if random:
       # For the same pattern random
       seed(1)
       shuffle(total_data)

    train, test = _dividing_to_train_and_test(total_data, percentage)

    if DEBUGGING_MODE:
       print("\n================ training set and test set =======================")
       print("TOTAL: type-{}, len-{}\n{}".format(type(total_data), len(total_data), total_data[:10]))
       print("TRAIN: type-{}, len-{}\n{}".format(type(train), len(train), train[:10]))
       print("TEST: type-{}, len-{}\n{}".format(type(test), len(test), test[:10]))

    return train, test


# From now on, Data preprocessing for Machine learning
def read_voca_dict(path):
    """Read vocabularies dictionary

    Arg:
        path(str): the file name and location 

    Return:
        data2idx: dictionary data structure for indexing number of word in idx2data list
        idx2data: data list sorted
    """
    with open(path, "r") as f:
        idx2data = [line.split()[1] for line in f.readlines() if line != TOKEN]

    data2idx = dict()

    for idx, val in enumerate(idx2data):
        if data2idx.get(val, None) == None:
            data2idx[val] = idx

    if DEBUGGING_MODE:
        print("\n=========== READ voca dictionary ======")
        check = True
        if len(idx2data) == len(Counter(idx2data)):
            check = True
        else:
            check = False
        print("duplicate check: {}".format(check))
        print("IDX2DATA: type-{},  len-{}\n{}".format(type(idx2data), len(idx2data), idx2data[0:10]))
        print("DATA2IDX: type-{}, len-{}\n".format(type(data2idx), len(data2idx)))
        counting = 0
        for key, val in data2idx.items():
            if counting == 10:
                break
            print("({}: {})".format(key, val), end=", ")
            counting +=1
        print()

    return data2idx, idx2data

 
def bi_gram_function(data_list):
    """Get bi gram pattern in syllables with no space
 
    Arg:
        data_list(list): data list from file named 'no_spacing_word'

    Return:
        bi_gram(list): bi gram syllable with END TOKEN 
                       like [['류머', '머티', '티즘', '즘열', '열</WORD_END>'], ...]
    """
    if DEBUGGING_MODE: 
       print("\n=============== Let's get bigram function =============")

    bi_gram = _separate_to_bi_gram(data_list)    

    return bi_gram


def mapping_data2idx(data, data2idx, name=None):
    """Mapping data into number 

    Args:
        data(list): data list like syllable data, bi gram data, and label data
                  like 1. sillable  ['류큐어', '르노도상', '류빛', '류사오치', '류머티즘열', '류푸']
                       2. bi gram syllable [['류큐', '큐어', '어</WORD_END>'], ['르노', '노도', '도상', '상</WORD_END>'], .......]
                       3. label ['BIB', 'BIIB', 'BB', 'BBII', 'BIIIB', 'BB']
        data2idx(dict): dictionary of vocabularies
        name(str): For debugging

    Return:
        dataset(list):
                   like [['류큐', '큐어', '어</WORD_END>'], ['르노', '노도', '도상', '상</WORD_END>'], .......] into 
                    [[27, 41, 35], [29, 21, 22, 34],  .......]
    """
    dataset = list()
    for line_idx, line_val in enumerate(data):
        data_time = list()
        for data_idx, data_val in enumerate(line_val):
            if data2idx.get(data_val, None) == None:
                data_time.append(data2idx[UNK_TOKEN])
            else:
                data_time.append(data2idx[data_val])

        dataset.append(data_time)

    if DEBUGGING_MODE:
        print("\n================ Mapping({}) data to index number of dictionary in mapping_data2idx function =======================".format(name))
        print("What is the UNK TOKEN: {}".format(UNK_TOKEN))
        print("DATA: type-{}, len-{}\n{}".format(type(data), len(data), data[:10]))
        print("dataset: type-{}, len-{}\n{}".format(type(dataset), len(dataset), dataset[:10]))
        print("DATA2IDX: type-{}, len-{}\n".format(type(data2idx), len(data2idx)))
        counting = 0
        for key, val in data2idx.items():
            if counting == 27:
                break
            print("({}: {})".format(key, val), end=", ")
            counting +=1
        print()

    return dataset

def read_x_and_y(path_x, path_y, data2idx=None, label2idx=None):
    """Read data and label files 

    Args:
        path_x: data file name and location 
        path_y: label file name and location 
        data2idx: dictionary for data voca
        label2idx: dictionary for label voca
    Returns:
        total(list):  zip of (uni_syllable, bi_gram_syllable, label)
    """
    data, label = read_data_and_label(path_x, path_y)
    bi_gram_data = bi_gram_function(data)


    if data2idx != None and label2idx != None:
        data = mapping_data2idx(data, data2idx, name="SYLLABLE_DATA")
        label = mapping_data2idx(label, label2idx, name="LABEL")
        bi_gram_data = mapping_data2idx(bi_gram_data, data2idx, name="BI_GRAM_DATA")

    total = list(zip(data, bi_gram_data, label))

    if DEBUGGING_MODE:
        print("\n================= Total after reading x and y of data and label in read_x_and_y function ==========")
        print("TOTAL: type-{}, len-{}\n{}".format(type(total), len(total), total[:10]))

    return total

def read_prediction_data(path_data, data2idx=None, label2idx=None):
    """Read prediction data 

    this function is for predict.py 
  
    Args: 
        path_data(str):  data file location for prediction
        data2idx(dict):  dictionary for data voca
        label2idx(dict): dictionary for label voca

    Return:
        total(list): zip of (uni_syllable, bi_gram_syllable, label)
    """
    data = read_file(path_data)
    bi_gram_data = bi_gram_function(data)
     
    if data2idx != None and label2idx != None:
        data = mapping_data2idx(data, data2idx, name="SYLLABLE_DATA_FOR_PREDICTIONS")
        bi_gram_data = mapping_data2idx(bi_gram_data, data2idx, name="BI_GRAM_DATA_FOR_PREDICTIONS")

    total = list(zip(data, bi_gram_data))

    if DEBUGGING_MODE:
        print("\n========================= Total after reading data for predictions ========")
        print("TOTAL: type-{}, len-{}\n{}".format(type(total), len(total), total[:10]))

    return total


def _get_length_of_a_batch(total_size, n_per_a_batch):
    """Get lenght for a batch 

    Args:
        total_size(int): the total size of data
        n_per_a_batch(int): the size for a batch

    Return:
        int(total_size/n_per_a_batch): the number of total batches
    """
    return int(total_size / n_per_a_batch)

# This function is need for fixing
def _generate_batch_with_idx(n_per_a_batch, total_data_len, random):
    """Generate batches based on index number, BUT this function need to be fixed

    Args:
        n_per_a_batch(int)
        total_data_len(int)
        random(bool)
    
    Return:
        batches(list):
    """
    total_data_idx = list(range(total_data_len))

    if random:
        shuffle(total_data_idx)

    n_batches = _get_length_of_a_batch(total_data_len, n_per_a_batch) 
    
    batches = list()

    for idx in range(n_batches):
        if idx == n_batches-1:
            batches.append(total_data_idx[n_per_a_batch*idx:])
        else:
            batches.append(total_data_idx[n_per_a_batch*idx:n_per_a_batch*(idx+1)])
    
    if DEBUGGING_MODE:
        print("\n=================== Batch Generation =================")
        print("TOTAL_SIZE_OF_DATA: len-{}, list_len-{}\n{}".format(total_data_len, len(total_data_idx), total_data_idx[:10]))
        print("SHUFFLE: {}".format(random))
        print("THE # of batches: {}".format(n_batches))
        print("EACH BATCHES: type-{}, len-{}, each_len-{}\n{}".format(type(batches), len(batches), len(batches[0]), batches[:10]))
        print("THE END OF BATCHE: type-{}, end_len-{},\n{}".format(type(batches), len(batches[-1]), batches[-1]))

    return batches


def _generate_batch_with_data(total_data,  n_per_a_batch, random):
    """Generate batches using data with zip 

    
    Args: 
        total_data(list): zip of (uni_syllable, bi_gram_syllable, label)
        n_per_a_batch(int): the size for a batch     
        random(bool): shuffling or not

    Return:
        batches(list): a list including each batch split on standard of input arguments
                      form like [([5, 18, 12], [27, 41, 35], [0, 1, 0]), ........]
    """
    if random:
       shuffle(total_data)

    batches = list()

    for idx in range(0, len(total_data), n_per_a_batch):
        batches.append(total_data[idx:idx+n_per_a_batch])

    if DEBUGGING_MODE:
        print("\n=================== Batch Generation with data in _generate_batch_with_data function =================")
        print("TOTAL_SIZE_OF_DATA: len-{}\n{}".format(len(total_data), total_data[0:10]))
        print("SHUFFLE: {}".format(random))
        print("EACH BATCHES: type-{}, len-{}, each_len-{}\n{}".format(type(batches), len(batches), len(batches[0]), batches[0:10]))
        print("THE END OF BATCHE: type-{}, end_len-{},\n{}".format(type(batches), len(batches[-1]), batches[-1]))

    return batches


def _padding_function(data_list, label_padding=False):
    """Padding function 

    Args:
        data_list(list): batches of each on uni_syllable, bi_gram syllable or label
        label_padding(bool): When I want to padding label into maximum length 
                             if true : -1
                             if false : 0
    Return:
        None
    """
    
    for batch_idx, batch_val in enumerate(data_list):
        max_len, max_len_data =  maximum_length(batch_val)
        if DEBUGGING_MODE and batch_idx < 10 and False:
            print("max len: {}, max_data: {}, {}".format(max_len, max_len_data, batch_val))
        for element_idx, element_val in enumerate(batch_val):
            if len(element_val) < max_len:
               if label_padding:
                   temp_padding = [-1] * (max_len-len(element_val))
               else:
                   temp_padding = [0] * (max_len-len(element_val))
               element_val.extend(temp_padding)
               if DEBUGGING_MODE and element_idx < 10 and False:
                   print("TEST: len-{}, {}".format(len(element_val), element_val))


def gathering_batches(zip_data, n_per_a_batch, random=True, padding=True):
    """Gather total data into bathces with padding and random shuffle

    This function is for training and test 

    Args:
        zip_data(list): total data set like (uni syllable, bi gram syllable, label)
        n_per_a_batch(int): the sizs for a batch 
        random(bool): shuffling or not 
        padding(bool): padding or not
    
    Returns:
        syllable_x(list): syllable list for each batch 
        bi_gram_x(list): bi gram list for each batch
        label(list): label list for each batch 
    """
    batches = _generate_batch_with_data(zip_data, n_per_a_batch, random)
    syllable_x = list()
    bi_gram_x = list()
    label = list()

    for batch_idx, batch_val in enumerate(batches):
        temp_syllable_x = list()
        temp_bi_gram_x = list()
        temp_label = list() 
        for real_idx, real_val in enumerate(batch_val):
            temp_syllable_x.append(real_val[0])
            temp_bi_gram_x.append(real_val[1])
            temp_label.append(real_val[2])
        syllable_x.append(temp_syllable_x)
        bi_gram_x.append(temp_bi_gram_x)
        label.append(temp_label)

    if padding: 
        _padding_function(syllable_x, label_padding=False)
        _padding_function(bi_gram_x, label_padding=False)
        _padding_function(label, label_padding=True)

    if DEBUGGING_MODE:
        print("\n================== Gathering Batches in gathering_batches =================")
        print("TOTAL_SIZE_OF_DATA: len-{}\n{}".format(len(batches), batches[0:10]))
        print("SHUFFLE: {}".format(random))
        print("SYLLABLE_X: type-{}, len-{}\n{}".format(type(syllable_x), len(syllable_x), syllable_x[0:10]))
        print("BI_GRAM_X: type-{}, len-{}\n{}".format(type(bi_gram_x), len(bi_gram_x), bi_gram_x[0:10]))
        print("LABEL: type-{}, len-{}\n{}".format(type(label), len(label), label[0:10]))

    return syllable_x, bi_gram_x, label 

def gathering_batches_for_predictions(zip_data, n_per_a_batch, random=False, padding=True):
    """Gather total data into batches with padding and random shuffle
    
    this function is for the version of predict.py to gather batches

    Args:
        zip_data(list): total data set like (uni syllable, bi gram syllable, label)
        n_per_a_batch(int): the sizs for a batch 
        random(bool): shuffling or not, Basically I have to configure this as False
        padding(bool): padding or not
    
    Returns:
        syllable_x(list): syllable list for each batch 
        bi_gram_x(list): bi gram list for each batch
    """
    batches = _generate_batch_with_data(zip_data, n_per_a_batch, random)
    syllable_x = list()
    bi_gram_x = list()

    for batch_idx, batch_val in enumerate(batches):
        temp_syllable_x = list()
        temp_bi_gram_x = list()
        for real_idx, real_val in enumerate(batch_val):
            temp_syllable_x.append(real_val[0])
            temp_bi_gram_x.append(real_val[1])
        syllable_x.append(temp_syllable_x)
        bi_gram_x.append(temp_bi_gram_x)

    if padding:
        _padding_function(syllable_x, label_padding=False)
        _padding_function(bi_gram_x, label_padding=False)

    if DEBUGGING_MODE:
        print("\n================== Gathering Batches in gathering_batches_for_predictions =================")
        print("TOTAL_SIZE_OF_DATA: len-{}\n{}".format(len(batches), batches[0:10]))
        print("SHUFFLE: {}".format(random))
        print("SYLLABLE_X: type-{}, len-{}\n{}".format(type(syllable_x), len(syllable_x), syllable_x[0:10]))
        print("BI_GRAM_X: type-{}, len-{}\n{}".format(type(bi_gram_x), len(bi_gram_x), bi_gram_x[0:10]))

    return syllable_x, bi_gram_x


def _maximum_length_for_LSTM(data_list):
    """Get length with no padding info in a list 

    Args:
       data_list(list): a list like [1, 2, 7, 0, 0]

    Return:
       list(sum_): length of a list like 3 in the case above
    """
    sign = np.sign(data_list)
    sum_ = np.sum(sign, axis = 1)

    return list(sum_)


def max_length_of_sequences(data_list):
    """Get a sequence maximum length with no padding 

    Args: 
        data_list(list): data list with padding like [[1,3,4,5,6], [1,2,0,0,0], .......] 

    Return:
        sequence(list): length list like [5, 2, .....]
    """
    sequence = list()

    for idx in (range(len(data_list))):
        sequence.append(_maximum_length_for_LSTM(data_list[idx]))

    if DEBUGGING_MODE:
        print("\n================ checking maximum Sequence in max_length_of_sequences =============")
        print("DATA: len-{}\n{}".format(len(data_list), data_list[:10]))
        print("SEQEUNCE: len-{}\n{}".format(len(sequence), sequence[:10]))
     
    return sequence 
