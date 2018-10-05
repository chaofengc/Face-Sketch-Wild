#!/bin/bash

PROJECT_DIR=$PWD
mkdir $PROJECT_DIR/data
cd $PROJECT_DIR/data
echo 'Downloading Precalculated Features'
wget http://www.visionlab.cs.hku.hk/data/Face-Sketch-Wild/features.tgz


