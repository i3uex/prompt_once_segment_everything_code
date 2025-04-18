#!/usr/bin/bash

# Create folder for the models checkpoints, if it doesn't exist

mkdir -p ./model_checkpoints
cd ./model_checkpoints || exit

# Download SAM models, from smallest to largest

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Download SAM 2 models, from smallest to largest

wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
