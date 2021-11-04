#!/bin/bash

# unzip the training file
unzip -d ./../data/ ./../data/hecktor2021_train.zip
echo "unziped files"

source activate prep
# get coordinates of the tight bboxes surrounding the GTV
python get_tight_bbox.py
echo "tight bboxes generated"
# generate the patch hdf5 file
python get_patches.py
echo "patch file was created"
conda deactivate
