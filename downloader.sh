#/usr/bin/env bash 

# this shell script is downloader for model trained. 

echo "downloading........"

MODEL="model.tar.gz"

curl -O "http://nlp.kookmin.ac.kr/hyunyoung2/autospacing/${MODEL}"


echo "uncompressing......"

tar -xvzf ${MODEL}

rm ${MODEL}
