#!/bin/bash -l
#PBS -l nodes=1:ppn=4,walltime=06:00:00

source torch
cd polarbear-reid

python plotting.py