#!/usr/bin/bash

set -eou pipefail

session_id="$1"
t1="$2"
n_procs=$3

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$n_procs

### Individual session run
recon-all \
    -time -parallel -threads $n_procs \
    -norandomness \
    -subjid "$session_id" \
    -i "$t1" \
    -all
