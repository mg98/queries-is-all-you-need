#!/bin/bash

module load cuda12.3/toolkit/12.3
nohup python -u exp/retrieval.py "$1" "$2" &> "logs/retrieval_$1_$2.out" &