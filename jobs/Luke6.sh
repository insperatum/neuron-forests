#!/bin/bash -l

#$ -S /bin/bash

#$ -l h_rt=0:5:0

#$ -l mem=1G

#$ -N Luke6

#$ -wd /home/zcahg55/Scratch/output/

#$ -t 1-50

cd $TMPDIR
mkdir "$HOME/Scratch/job_files/Luke6"

module load python/enthought/7.3-2_2013-10-04

python $HOME/python/lukeforest_comparison.py $HOME/data/Helmstaedter_etal_Nature_2013_e2006_TrainingData_all.mat $HOME/Scratch/features \
"{'train_frac':0.1, 'offsets':[-2, 0, 2], 'depth':14, 'thresholds':1, 'max_features':'sqrt'}"

cp -r $TMPDIR "$HOME/Scratch/job_files/Luke6/$SGE_TASK_ID (`date +"%d-%m-%y %T"`)"
