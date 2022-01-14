"""
CCNA timeseries extraction and confound removal.

Save the output to a tar file.
"""
import tarfile
from pathlib import Path

import pandas as pd
import numpy as np

import nibabel as nb

import nilearn
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_atlas_difumo
from nilearn.regions import RegionExtractor
import nilearn.input_data

from bids import BIDSLayout
from nilearn.interfaces.fmriprep import load_confounds


CCNA_PATH = "/scratch/dlussier/ccna_fmriprep/1641486564/fmriprep/"
BIDS_INFO =  Path(__file__).parent / ".bids_info"
ATLAS_METADATA = {
    'difumo': {'type': "dynamic",
               'resolutions': [64],
               'label_idx': 1,
               'fetcher': "nilearn.datasets.fetch_atlas_difumo(dimension={resolution},resolution_mm=3)"},
}


def extract_regions_network(map_file):
    """Extract isolated regions from network map"""
    extractor = RegionExtractor(map_file, threshold=0.5,
                                thresholding_strategy='ratio_n_voxels',
                                extractor='local_regions',
                                standardize=True, detrend=True)
    extractor.fit()
    
    # create new labels indicating the original network/parcel
    labels = []
    for i in range(atlas_resolution):
        regions_indices_network = np.where(np.array(extractor.index_) == i)
        for index in regions_indices_network[0]:
            labels.append(f"region-{index + 1}_network-{i + 1}")
    return extractor, labels


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
    base += f"_atlas-{atlas_name}_network-{resolution}_timeseries.tsv"
    return base.split('/')[-1]


if __name__ == '__main__':
    layout = BIDSLayout(CCNA_PATH, config=['bids','derivatives'])
    # layout.save(BIDS_INFO)
    subject_list = layout.get(return_type='id', target='subject')
    output_root_dir = Path.home() / "scratch"

    for atlas_name in ATLAS_METADATA.keys():
        print("-- {} --".format(atlas_name))
        dataset_name = f"dataset-ccna2020_atlas-{atlas_name}"
        output_dir = output_root_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        atlas_resolution = 64
        atlas = fetch_atlas_difumo(dimension=atlas_resolution,resolution_mm=3)
        extractor, labels = extract_regions_network(atlas.maps)
        nb.save(extractor.regions_img_,
                output_dir / f"atlas-difumo_network-{atlas_resolution}_regions.nii.gz")

        for subject in subject_list:
            print(f"sub-{subject}")
            # Note from AB
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
            output_filename = bidsish_timeseries_file_name(
                file_entitiles, layout, atlas_name, atlas_resolution)
            confounds, sample_mask = load_confounds(fmri[0].path,
                                                    strategy=['motion', 'high_pass', 'wm_csf', 'scrub', 'global_signal'],
                                                    motion='basic', wm_csf='basic', global_signal='basic',
                                                    scrub=5, fd_threshold=0.5, std_dvars_threshold=None,
                                                    demean=True)
            timeseries = extractor.fit_transform(fmri[0].path, confounds=confounds, sample_mask=sample_mask)

            # Estimating connectomes
            corr_measure = ConnectivityMeasure(kind="correlation")
            connectome = corr_measure.fit_transform([timeseries])[0]

            # Save to file
            timeseries = pd.DataFrame(timeseries, columns=labels)
            timeseries.to_csv(timeseries_root_dir / output_filename, sep='\t', index=False)
            connectome = pd.DataFrame(connectome, columns=labels, index=labels)
            connectome.to_csv(timeseries_root_dir / output_filename.replace("timeseries", "connectome"), sep='\t')


    # tar the dataset
    with tarfile.open(output_root_dir / f"{dataset_name}.tar.gz", "w:gz") as tar:
        tar.add(output_dir, arcname=output_dir.name)
