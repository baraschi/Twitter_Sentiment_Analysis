#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
if [ "$1" == "full" ]; then
  echo on the full data set
  cat Data/twitter-datasets/train_pos_full.txt Data/twitter-datasets/train_neg_full.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > Data/vocab.txt
else
  echo only on 10% of the data set
  cat Data/twitter-datasets/train_pos.txt Data/twitter-datasets/train_neg.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > Data/vocab.txt
fi
