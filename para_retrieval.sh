#!/bin/bash

REPO_DIR=qiayn
N_DOCS=$1
n_queries=1

for _ in {1..2}; do
    # assuming 10 machines on node303-312
    for node in {303..312}; do
        echo "Starting experiments on node${node} with $N_DOCS docs and $n_queries queries"
        ssh "node${node}" "cd $REPO_DIR && ./retrieval.sh $N_DOCS $n_queries &"
        n_queries=$((n_queries + 1))
    done
done