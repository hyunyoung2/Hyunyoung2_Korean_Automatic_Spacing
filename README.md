# Korean Automatic Spacing(한국어 자동 띄어 쓰기)
 
This Korean automatic spacing is made, in particular, on the following version.

```
OS: Ubuntu 16.04.1
Python version: 3.5.2
Tensorflow version: 1.8.0
```

If you encounter some error, check the version above once again. 

# Before running predic.py script, download model with downloader shell script

> ./downloader.sh

```shell
# nlp-gpu @ nlpgpu in ~/Hyunyoung2/competition/autospacing/concat/EPOCH_10/300/github_repository_for_competition/Hyunyoung2_Korean_Automatic_Spacing on git:master x [20:11:13] 
$ ./downloader.sh 
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  107M  100  107M    0     0  16.5M      0  0:00:06  0:00:06 --:--:-- 26.9M
model/
model/final-135000.index
model/final-135000.meta
model/checkpoint
model/final-135000.data-00000-of-00001
```

**After downloading the model, You would find out **model** directory, then you can run predict.py.**

But you don't have **curl**, intall it on Ubuntu

> sudo apt-get install curl

# How To Run this Korean automatic spacing program.

> python3 predict.py [-h] [-o OUTPUT] [-n NUMBER] input

1. -h: means help message about how to use this script to run 

2. -o  OUTPUT: output file you want to get on input file 

3. -n NUMBER: When you have big size file, you could test some batches together. 
           So "-n" means how much you want to test together maximumally on a test.

4. input: Input file

If you know abot how to run, see the help message below:

Also you can check on prompt if you type in **python3 predict.py -h**

```
usage: predict.py [-h] [-o OUTPUT] [-n NUMBER] input

split program for compound nouns

positional arguments:
  input                 input file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output file
  -n NUMBER, --number NUMBER
                        Batch size, this must be less than your number of
                        data, and memory size
```

When you run the python scipt, predict.py, you have to specify the input file, 

If you not, you get the following error:

```shell
$ python3 predict.py 
/home/nlp-gpu/.local/lib/python3.5/site-packages/tensorflow/python/ops/gradients_impl.py:100: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
usage: predict.py [-h] [-o OUTPUT] [-n NUMBER] input
predict.py: error: the following arguments are required: input
```
**Be careful**

Don't touch **data and model directory**

This file contain vocabulary dictionary and model file after traninig. 

### an example of running predict.py

In order to get result of Korean automatic spacing, If you type in as follows:

> python3 predict.py input_file -o output_file

````shell
# nlp-gpu @ nlpgpu in ~/Hyunyoung2/competition/autospacing/concat/EPOCH_10/300/github_repository_for_competition/Hyunyoung2_Korean_Automatic_Spacing on git:master x [17:46:00] 
$ python3 predict.py input_file -o output_file
/home/nlp-gpu/.local/lib/python3.5/site-packages/tensorflow/python/ops/gradients_impl.py:100: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "

============== Your input file is : =================
input_file

================ Your output file is: ====================
/home/nlp-gpu/Hyunyoung2/competition/autospacing/concat/EPOCH_10/300/github_repository_for_competition/Hyunyoung2_Korean_Automatic_Spacing/result/output_file

============= your batch size: 1 ====================
2018-08-16 17:46:16.346716: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
..........
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1500/1500 [02:37<00:00,  9.52it/s]
Prediction is done!!
````

But If you type in "-n" option, evaluation time would get shoter as follows: 

```
# nlp-gpu @ nlpgpu in ~/Hyunyoung2/competition/autospacing/concat/EPOCH_10/300/github_repository_for_competition/Hyunyoung2_Korean_Automatic_Spacing on git:master x [17:52:55] 
$ python3 predict.py input_file -o output_file -n 10
/home/nlp-gpu/.local/lib/python3.5/site-packages/tensorflow/python/ops/gradients_impl.py:100: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "

============== Your input file is : =================
input_file

================ Your output file is: ====================
/home/nlp-gpu/Hyunyoung2/competition/autospacing/concat/EPOCH_10/300/github_repository_for_competition/Hyunyoung2_Korean_Automatic_Spacing/result/output_file 
============= your batch size: 10 ====================
...........
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 150/150 [00:25<00:00,  5.88it/s]
Prediction is done!!
```

If you don't know how to run, go through **run.sh** script file. 

This would show how to run Korean automatic spacing. 

**sample_x** : is sample file to test whether predict.py script works or not,  When you execute **run.sh** 

# Result

There are two ways you can get result of running this Korean automatic spacing. 

But, in any case, all of the result will be created under **result** directory. 

First, If you specify output file, it would create itself under **result** directory.

```shell
./
|...
| result/
| | ourput_file
| | PREDICTIONS_TAGS # This tag value like "B" or "I"
| run.sh
.....
```

Second, If you don't specify output file. it would create the default output file named **PREDICTIONS_RESULT**

```shell
./
|...
| result/
| | PREDICTIONS_RESULT # -> this file is the defualt result file of Korean automatic spacing. 
| | PREDICTIONS_TAGS # This tag value like "B" or "I"
| run.sh
.....
```

