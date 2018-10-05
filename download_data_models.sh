#!/bin/bash

PROJECT_DIR=$PWD
mkdir $PROJECT_DIR/data
cd $PROJECT_DIR/data
echo 'Downloading Datasets'
wget http://www.visionlab.cs.hku.hk/data/Face-Sketch-Wild/datasets.tgz
echo 'Downloading Pretrain Models'
wget http://www.visionlab.cs.hku.hk/data/Face-Sketch-Wild/models.tgz


