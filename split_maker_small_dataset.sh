#!/bin/bash

#if [ "$#" != "4" ] ; then
#    echo "Arguments to the script: path to train split file, path to test split file, 
#          path to full dataset file where all rows are datapoints, dataset name"
#    exit 1
#fi

train_split_src=/home/siddhant.katyan/XML/GlasXC/data/Delicious/delicious_trSplit.txt
test_split_src=/home/siddhant.katyan/XML/GlasXC/data/Delicious/delicious_tstSplit.txt
full_dataset_src=/home/siddhant.katyan/XML/GlasXC/data/Delicious/Delicious_data.txt
dataset_name="Delicious"

echo $dataset_name

# training dataset
while read num rest;
    do
        sed -n "$num p" $full_dataset_src >> "$dataset_name""_train.txt";
    done < $train_split_src

# testing dataset
while read num rest;
    do
        sed -n "$num p" $full_dataset_src >> "$dataset_name""_test.txt";
    done < $test_split_src
