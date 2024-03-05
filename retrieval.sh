#!/bin/bash

source _setup.sh
nohup python -u exp/retrieval.py $1 $2 &> "logs/retrieval_$1_$2.out" &