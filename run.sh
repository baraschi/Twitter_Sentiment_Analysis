#!/bin/bash

./build_vocab.sh $1
./cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py
python3 glove_solution.py
