#!/bin/bash

PROJECT_DIR=$PWD
mkdir $PROJECT_DIR/data
mkdir $PROJECT_DIR/pretrain_model
echo 'Downloading Datasets'
wget http://www.visionlab.cs.hku.hk/data/Face-Sketch-Wild/datasets.tgz -P $PROJECT_DIR/data
echo 'Downloading Pretrain Models'
wget http://www.visionlab.cs.hku.hk/data/Face-Sketch-Wild/models.tgz -P $PROJECT_DIR/pretrain_model


