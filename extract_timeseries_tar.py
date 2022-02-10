"""
CCNA timeseries extraction and confound removal.

Save the output to a tar file.
"""
import os
import tarfile
import pathlib
import pandas as pd
import numpy as np
import nibabel as nb
import templateflow
import templateflow.api
import templateflow.conf
import nilearn
import nilearn.connectome
import nilearn.datasets
import nilearn.regions
import nilearn.input_data
import nilearn.interfaces.fmriprep
import sklearn
import bids

USERNAME = os.environ.get("USER")
# DATA_PATH = "/scratch/" + USERNAME + "/datasets/ukbb/derivatives/fmriprep/fmriprep/"
DATA_PATH = "/home/ltetrel/Documents/data/ukbb/derivatives/fmriprep/fmriprep"
DATASET_NAME = "ukbb"
# ATLAS_PATH = os.path.join(os.path.sep, "scratch",
#                               USERNAME, "segmented_difumo_atlases")
# OUTPUT_ROOT_DIR = pathlib.Path.home() / "scratch"
OUTPUT_ROOT_DIR = pathlib.Path.home()
ATLAS_PATH = "/home/ltetrel/.cache/templateflow/tpl-MNI152NLin2009cAsym"
BIDS_INFO = pathlib.Path(__file__).parent / ".bids_info"
ATLAS_METADATA = {
    'segmented_difumo': {'type': "dynamic",
                         'dimensions': [64, 128, 256, 512, 1024],
                         'resolutions': [2, 3],
                         'fetcher': "segmented_difumo_fetcher(dimension={dimension}, resolution_mm={resolution}, atlas_path=ATLAS_PATH)"},
}


def segmented_difumo_fetcher(atlas_path, dimension=64, resolution_mm=3):

    templateflow.conf.TF_HOME = pathlib.Path(atlas_path)
    templateflow.conf.init_layout()

    img_path = str(templateflow.api.get("MNI152NLin2009cAsym", atlas="DiFuMo",
                                        desc=f"{dimension}dimensionsSegmented", resolution=f"0{resolution_mm}", extension=".nii.gz"))
    label_path = str(templateflow.api.get("MNI152NLin2009cAsym", atlas="DiFuMo",
                                          desc=f"{dimension}dimensionsSegmented", resolution=f"0{resolution_mm}", extension=".tsv"))
    labels = pd.read_csv(label_path, delimiter="\t")
    labels = (labels['Region'].astype(str) + ". " +
              labels['Difumo_names']).values.tolist()

    return sklearn.utils.Bunch(maps=img_path, labels=labels)


def create_atlas_masker(atlas_name, dimension, resolution, nilearn_cache=""):
    """Create masker of all dimensions and resolutions given metadata."""
    if atlas_name not in ATLAS_METADATA.keys():
        raise ValueError("{} not defined!".format(atlas_name))
    curr_atlas = ATLAS_METADATA[atlas_name]
    curr_atlas['name'] = atlas_name

    atlas = eval(curr_atlas['fetcher'].format(
        dimension=dimension, resolution=resolution))
    if curr_atlas['type'] == "static":
        masker = nilearn.input_data.NiftiLabelsMasker(
            atlas.maps, detrend=True)
    elif curr_atlas['type'] == "dynamic":
        masker = nilearn.input_data.NiftiMapsMasker(
            atlas.maps, detrend=True)
    if nilearn_cache:
        masker = masker.set_params(memory=nilearn_cache, memory_level=1)
    # fill atlas info
    curr_atlas[dimension] = {'masker': masker}
    if isinstance(atlas.labels[0], tuple) | isinstance(atlas.labels[0], list):
        if isinstance(atlas.labels[0][curr_atlas['label_idx']], bytes):
            labels = [label[curr_atlas['label_idx']].decode()
                      for label in atlas.labels]
        else:
            labels = [label[curr_atlas['label_idx']] for label in atlas.labels]
    else:
        if isinstance(atlas.labels[0], bytes):
            labels = [label.decode() for label in atlas.labels]
        else:
            labels = [label for label in atlas.labels]

    return masker, labels


def create_timeseries_root_dir(file_entitiles, output_dir):
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


def bidsish_timeseries_file_name(file_entitiles, layout, atlas_name, dimension):
    """Create a BIDS-like file name to save extracted timeseries as tsv."""
    pattern = "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}][_rec-{reconstruction}][_run-{run}][_echo-{echo}]"
    base = layout.build_path(file_entitiles, pattern, validate=False)
    base += f"_atlas-{atlas_name}_network-{dimension}_timeseries.tsv"
    return base.split('/')[-1]


if __name__ == '__main__':
    layout = bids.BIDSLayout(DATA_PATH, config=['bids', 'derivatives'])
    # layout.save(BIDS_INFO)
    subject_list = layout.get(return_type='id', target='subject')

    for atlas_name in ATLAS_METADATA.keys():
        print("-- {} --".format(atlas_name))
        dataset_title = f"dataset-{DATASET_NAME}_atlas-{atlas_name}"
        output_dir = OUTPUT_ROOT_DIR / dataset_title
        output_dir.mkdir(parents=True, exist_ok=True)
        for subject in subject_list:
            print(f"sub-{subject}")
            # TODO: loop through all
            # Note from AB
            # if multiple run, use run 2
            # if multiple session, use ses 1
            fmri = layout.get(return_type='type', subject=subject, space='MNI152NLin2009cAsym',
                              desc='preproc', suffix='bold', extension='nii.gz')
            for ii in range(len(fmri)):
                file_entitiles = fmri[ii].entities
                timeseries_root_dir = create_timeseries_root_dir(
                    file_entitiles, output_dir)
                for dimension in ATLAS_METADATA[atlas_name]['dimensions']:
                    for resolution in ATLAS_METADATA[atlas_name]['resolutions']:
                        masker, labels = create_atlas_masker(
                            atlas_name, dimension, resolution, nilearn_cache="")
                        output_filename = bidsish_timeseries_file_name(
                            file_entitiles, layout, atlas_name, dimension)
                        confounds, sample_mask = nilearn.interfaces.fmriprep.load_confounds(fmri[ii].path,
                                                                                            strategy=[
                                                                                                'motion', 'high_pass', 'wm_csf', 'scrub', 'global_signal'],
                                                                                            motion='basic', wm_csf='basic', global_signal='basic',
                                                                                            scrub=5, fd_threshold=0.5, std_dvars_threshold=None,
                                                                                            demean=True)
                        timeseries = masker.fit_transform(
                            fmri[ii].path, confounds=confounds, sample_mask=sample_mask)
                        # Estimating connectomes
                        corr_measure = nilearn.connectome.ConnectivityMeasure(
                            kind="correlation")
                        connectome = corr_measure.fit_transform([timeseries])[0]

                        # Save to file
                        timeseries = pd.DataFrame(timeseries, columns=labels)
                        timeseries.to_csv(timeseries_root_dir /
                                          output_filename, sep='\t', index=False)
                        connectome = pd.DataFrame(
                            connectome, columns=labels, index=labels)
                        connectome.to_csv(
                            timeseries_root_dir / output_filename.replace("timeseries", "connectome"), sep='\t')

    # tar the dataset
    with tarfile.open(OUTPUT_ROOT_DIR / f"{dataset_title}.tar.gz", "w:gz") as tar:
        tar.add(output_dir, arcname=output_dir.name)
