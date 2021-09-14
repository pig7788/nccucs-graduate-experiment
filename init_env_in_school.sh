#!/bin/bash

sed -i 's/albeli/ytlee/g' ./*.py
sed -i 's/albeli/ytlee/g' ./*.sh
sed -i 's/albeli/ytlee/g' ./config/*.*
mkdir -p /home/ytlee/workspace/NCCU/Experiment/models/t5_based/