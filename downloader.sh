#/usr/bin/env bash 


# this shell script is downloader for model trained. 


MODEL="model.tar.gz"

wget "http://nlp.kookmin.ac.kr/hyunyoung2/autospacing/${MODEL}"

tar -xvzf ${MODEL}

rm ${MODEL}


