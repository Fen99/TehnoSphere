#!/usr/bin/env bash

export VW_BIN=/usr/bin/

PYTHON_PATH="/usr/bin/python3"

TRAIN="cat ../data/train.csv"
TEST="cat ../data/test.csv"
JAVA_BIN="java -Xmx4g -cp ../display-ad-java/target/*:. com.sigaphi.kaggle.displayad"

#echo "import data into redis ..."
$TRAIN | $JAVA_BIN.ToRedis
$TEST | $JAVA_BIN.ToRedis

#echo "making vw input files ..."
$TRAIN | $JAVA_BIN.FeaturesToVw | gzip > train.vw.gz
$TEST | $JAVA_BIN.FeaturesToVw | gzip > test.vw.gz

echo "training model ..."
$PYTHON_PATH ../scripts/vw_run.py poly_1 6 1

echo "making a submission file"
cat <(echo "Id,p1") <(paste -d"," <(tail -n +2 ../../data/sample_adv.txt | cut -f1 | cut -d"," -f1) prediction_test_Option.poly_1.txt) | $PYTHON_PATH ../scripts/submit.py
