#!/bin/bash -l

#$ -S /bin/bash

#$ -l h_rt=0:60:0

#$ -l mem=5G

#$ -N NeuronForest_comparison

#$ -wd /home/zcahg55/Scratch/output/Luke_Depth14

#$ -t 1-50

cd $TMPDIR

module load python/enthought/7.3-2_2013-10-04

python $HOME/python/lukeforest_comparison.py $HOME/data/Helmstaedter_etal_Nature_2013_e2006_TrainingData_all.mat $HOME/Scratch/features

cp -r $TMPDIR "$HOME/Scratch/job_files/Luke_Depth14/$JOB_ID.$SGE_TASK_ID (`date +"%d-%m-%y %T"`)"
