#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#--account
#SBATCH --job-name=fmriprep_ukbb_extract_%j.job
#SBATCH --output=/scratch/%u/.slurm/fmriprep_ukbb_extract_%j.out
#SBATCH --error=/scratch/%u/.slurm/fmriprep_ukbb_extract_%j.err
#--mail-user
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

mkdir $SLURM_TMPDIR/ukbb
scp -r $SCRATCH/datasets/ukbb/derivatives $SLURM_TMPDIR/ukbb/
scp -r $SCRATCH/atlases $SLURM_TMPDIR/
source /home/ltetrel/.virtualenvs/ts_extraction/bin/activate

python3 /home/ltetrel/ccna_ts_extraction/extract_timeseries_tar.py -i $SLURM_TMPDIR/ukbb/derivatives/fmriprep/fmriprep/ --atlas-path $SLURM_TMPDIR/atlases --dataset-name ukbb -o $SLURM_TMPDIR
exit 0
