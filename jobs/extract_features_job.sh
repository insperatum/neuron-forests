#!/bin/bash -l

#$ -S /bin/bash

#$ -l h_rt=0:5:0

#$ -l mem=2G

#$ -N NeuronForest-extract_features

#$ -wd /home/zcahg55/Scratch/output

cd $TMPDIR

module load python/enthought/7.2-2

python $HOME/python/extract_features.py $HOME/data/Helmstaedter_etal_Nature_2013_e2006_TrainingData_all.mat

cp -r $TMPDIR "$HOME/Scratch/job_files/$JOB_ID `date +"%d-%m-%y %T"`"
