"""
CCNA timeseries extraction and confound removal.

Save the output to a tar file.
"""
import tarfile
from pathlib import Path

import pandas as pd
import numpy as np

import nilearn
import nilearn.datasets
import nilearn.input_data

from bids import BIDSLayout
from nilearn.interfaces.fmriprep import load_confounds


CCNA_PATH = "/data/cisl/lussier/ccna/ccna_fmriprep/fmriprep_2020/fmriprep/"
BIDS_INFO = "/data/simexp/hwang/ccna_ts/.bids_info"
ATLAS_METADATA = {
    'difumo': {'type': "dynamic",
               'resolutions': [64],
               'label_idx': 1,
               'fetcher': "nilearn.datasets.fetch_atlas_difumo(dimension={resolution},resolution_mm=3)"},
}


def create_atlas_masker(atlas_name, nilearn_cache=""):
    """Create masker of all resolutions given metadata."""
    if atlas_name not in ATLAS_METADATA.keys():
        raise ValueError("{} not defined!".format(atlas_name))
    curr_atlas = ATLAS_METADATA[atlas_name]
    curr_atlas['name'] = atlas_name

    for resolution in curr_atlas['resolutions']:
        atlas = eval(curr_atlas['fetcher'].format(resolution=resolution))
        if curr_atlas['type'] == "static":
            masker = nilearn.input_data.NiftiLabelsMasker(
                atlas.maps, detrend=True)
        elif curr_atlas['type'] == "dynamic":
            masker = nilearn.input_data.NiftiMapsMasker(
                atlas.maps, detrend=True)
        if nilearn_cache:
            masker = masker.set_params(memory=nilearn_cache, memory_level=1)
        # fill atlas info
        curr_atlas[resolution] = {'masker': masker}
        if isinstance(atlas.labels[0], tuple) | isinstance(atlas.labels[0], list):
            if isinstance(atlas.labels[0][curr_atlas['label_idx']], bytes):
                curr_atlas[resolution]['labels'] = [
                    label[curr_atlas['label_idx']].decode() for label in atlas.labels]
            else:
                curr_atlas[resolution]['labels'] = [
                    label[curr_atlas['label_idx']] for label in atlas.labels]
        else:
            if isinstance(atlas.labels[0], bytes):
                curr_atlas[resolution]['labels'] = [
                    label.decode() for label in atlas.labels]
            else:
                curr_atlas[resolution]['labels'] = [
                    label for label in atlas.labels]

    return curr_atlas


def create_timeseries_root_dir(file_entitiles):
    """Create root directory for the timeseries file."""
    subject = f"sub-{file_entitiles['subject']}"
    session = f"ses-{file_entitiles['session']}" if file_entitiles.get(
        'session', False) is not None else None
    if session:
        timeseries_root_dir = output_dir / subject / session
    else:
        timeseries_root_dir = output_dir / subject
    timeseries_root_dir.mkdir(parents=True, exist_ok=True)

    return timeseries_root_dir

def bidsish_timeseries_file_name(file_entitiles, layout, atlas_name, resolution):
    """Create a BIDS-like file name to save extracted timeseries as tsv."""
    pattern = "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}][_rec-{reconstruction}][_run-{run}][_echo-{echo}]"
    base = layout.build_path(file_entitiles, pattern, validate=False)
    base += f"_atlas-{atlas_name}_nroi-{resolution}_timeseries.tsv"
    return base.split('/')[-1]


if __name__ == '__main__':
    layout = BIDSLayout(CCNA_PATH, config=['bids','derivatives'])
    # layout.save(BIDS_INFO)
    subject_list = layout.get(return_type='id', target='subject')
    output_root_dir = Path("/data/simexp/hwang/")

    for atlas_name in ATLAS_METADATA.keys():
        print("-- {} --".format(atlas_name))
        dataset_name = f"dataset-ccna2020_atlas-{atlas_name}"
        output_dir = output_root_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        atlas_masker = create_atlas_masker(atlas_name)
        for subject in subject_list:
            print(f"sub-{subject}")
            # Note for AB
            # if multiple run, use run 2
            # if multiple session, use ses 1
            fmri = layout.get(return_type='type',
                              subject=subject, session='1', task='rest',
                              space='MNI152NLin2009cAsym',
                              desc='preproc', suffix='bold', extension='nii.gz')

            confounds = layout.get(return_type='file',
                                   subject=subject, session='1', task='rest',
                                   desc='confounds', suffix='timeseries', extension='tsv')
            if len(fmri) == 2:  # soz I am lazy
                fmri = layout.get(return_type='type',
                                    subject=subject, session='1', task='rest',
                                    run='2',
                                    space='MNI152NLin2009cAsym',
                                    desc='preproc', suffix='bold', extension='nii.gz')

                confounds = layout.get(return_type='file',
                                        subject=subject, session='1', task='rest', run='2',
                                        desc='confounds', suffix='timeseries', extension='tsv')

            file_entitiles = fmri[0].entities
            timeseries_root_dir = create_timeseries_root_dir(
                file_entitiles)
            for resolution in atlas_masker['resolutions']:
                masker = atlas_masker[resolution]['masker']
                output_filename = bidsish_timeseries_file_name(
                    file_entitiles, layout, atlas_name, resolution)
                confounds, sample_mask = load_confounds(fmri[0].path,
                                                        strategy=['motion', 'high_pass', 'wm_csf', 'scrub', 'global_signal'],
                                                        motion='basic', wm_csf='basic', global_signal='basic',
                                                        scrub=5, fd_threshold=0.5, std_dvars_threshold=None,
                                                        demean=True)
                timeseries = masker.fit_transform(fmri[0].path, confounds=confounds, sample_mask=sample_mask)
                labels = atlas_masker[resolution]['labels']
                timeseries = pd.DataFrame(timeseries, columns=labels)
                timeseries.to_csv(timeseries_root_dir / output_filename, sep='\t', index=False)
    # tar the dataset
    with tarfile.open(output_root_dir / f"{dataset_name}.tar.gz", "w:gz") as tar:
        tar.add(output_dir, arcname=output_dir.name)
