#!/bin/bash

# Create directory
mkdir -p ~/panns_data

# Copy file to the new directory
cp ./class_labels_indices.csv ~/panns_data/

# Download file from URL to the new directory
curl -L -o ~/panns_data/Cnn14_mAP=0.431.pth "https://huggingface.co/thelou1s/panns-inference/resolve/main/Cnn14_mAP%3D0.431.pth"
