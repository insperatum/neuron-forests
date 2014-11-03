#!/bin/bash -l

#$ -S /bin/bash

#$ -l h_rt=3:0:0

#$ -l mem=4G

#$ -l thr=12

# 5. Reserve one Matlab licence - this stops your job starting and failing when no
#    licences are available.
#$ -l matlab=1

#$ -ac exclusive

#$ -N Matlab_expt

#$ -wd /home/zcahg55/Scratch/output/

# 10. Your work *must* be done in $TMPDIR 
cd $TMPDIR

cp -r '/home/zcahg55/matlab/' .
cd matlab

# 12. Run Matlab job

module unload compilers/intel/11.1/072
module load compilers/gnu/4.6.3
module load matlab/full/r2013a/8.1
module list
echo ""
echo "Running matlab -nosplash -nodisplay < default_decision_forest2.mat ..."
echo ""
matlab -nosplash -nodesktop -nodisplay < default_decision_forest2.mat

mkdir $HOME/Scratch/job_files/matlab
cp -r $TMPDIR "$HOME/Scratch/job_files/matlab (`date +"%d-%m-%y %T"`)"