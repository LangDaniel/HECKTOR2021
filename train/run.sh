#!/bin/bash

source activate torch
nohup taskset -c 60-79 python main.py --gpu 1 $1 &
printf "$1: $!\n" >> log
conda deactivate
