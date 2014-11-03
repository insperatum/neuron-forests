#!/bin/bash -l

#$ -S /bin/bash

#$ -l h_rt=0:25:0

#$ -l mem=1G

#$ -N NF_Summarise

#$ -wd /home/zcahg55/Scratch/output/

#UNCOMMENT FOR ALL -t 1-9
export SGE_TASK_ID=10

cd $TMPDIR

module load python/enthought/7.3-2_2013-10-04
export frac=0.2
if [ $SGE_TASK_ID == 6 ]; then export frac=0.1; fi
if [ $SGE_TASK_ID == 7 ]; then export frac=0.4; fi
python ~/python/prediction_stats.py ~/data/Helmstaedter_etal_Nature_2013_e2006_TrainingData_all.mat ~/Scratch/job_files/Luke$SGE_TASK_ID "{'train_frac':$frac}"