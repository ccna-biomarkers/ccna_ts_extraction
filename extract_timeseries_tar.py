"""
CCNA timeseries extraction and confound removal.

Save the output to a tar file.
"""
from distutils import extension
import os
import argparse
import tarfile
import pathlib
import pandas as pd
import templateflow.api
import templateflow.conf
import nilearn.connectome
import nilearn.datasets
import nilearn.input_data
import nilearn.interfaces.fmriprep
import sklearn.utils
import bids

ATLAS_METADATA = {
    # 'schaefer': {
    #     'source': "templateflow",
    #     'templates' : ['MNI152NLin2009cAsym', 'MNI152NLin6Asym'],
    #     'resolutions': ["01", "02"],
    #     'atlas': 'Schaefer2018',
    #     'description_pattern': "{dimension}Parcels{network}Networks",
    #     'dimensions': [100, 200, 300, 400, 500, 600, 800, 1000],
    #     'networks': [7, 17],
    #     'atlas_parameters': ['resolution', 'desc'],
    #     'label_parameters': ['desc'],
    #     },
    'schaefer7': {
        'source': "templateflow",
        'templates' : ['MNI152NLin2009cAsym', 'MNI152NLin6Asym'],
        'resolutions': [1, 2],
        'atlas': 'Schaefer2018',
        'description_pattern': "{dimension}Parcels7Networks",
        'dimensions': [100, 200, 300, 400, 500, 600, 800, 1000],
        'atlas_parameters': ['resolution', 'desc'],
        'label_parameters': ['desc'],
        },
    #TODO: fix upstream issues with difumo
    # 'difumo': {
    #     'source': "templateflow",
    #     'templates' : ['MNI152NLin2009cAsym', 'MNI152NLin6Asym'],
    #     'resolutions': [2, 3],
    #     'atlas': 'DiFuMo',
    #     'description_pattern': "{dimension}dimensions",
    #     'dimensions': [64, 128, 256, 512, 1024],
    #     'atlas_parameters': ['resolution', 'desc'],
    #     'label_parameters': ['resolution','desc'],
    #     },
    "segmented_difumo": {
        'source': "user_define",
        'templates' : ['MNI152NLin2009cAsym'],
        'resolutions': [2, 3],
        'atlas': 'DiFuMo',
        'description_pattern': "{dimension}dimensionsSegmented",
        'dimensions': [64, 128, 256, 512, 1024],
        'atlas_parameters': ['resolution', 'desc'],
        'label_parameters': ['resolution','desc'],
        },
    }

LOAD_CONFOUNDS_PARAMS = {
    'strategy': ['motion', 'high_pass', 'wm_csf', 'scrub', 'global_signal'],
    'motion': 'basic',
    'wm_csf': 'basic',
    'global_signal': 'basic',
    'scrub': 5,
    'fd_threshold': 0.5,
    'std_dvars_threshold': None,
    'demean': True
}

def update_templateflow_path(atlas_name, atlas_path):
    """Update local templateflow path, if needed."""

    atlas_source = ATLAS_METADATA[atlas_name]['source']
    
    # by default, it uses `~/.cache/templateflow/`
    if atlas_source == "templateflow":
        templateflow.conf.TF_HOME = os.path.join(os.getenv("HOME"), ".cache", "templateflow")
        templateflow.conf.init_layout()
    # otherwise use user defined atlas path
    elif atlas_source == "user_define":
        templateflow.conf.TF_HOME = pathlib.Path(atlas_path)
        templateflow.conf.init_layout()
    else:
        pass

#TODO: QC timeseries https://github.com/SIMEXP/mapsmasker_benchmark/blob/main/mapsmasker_benchmark/main.py
def fetch_atlas_path(atlas_name, template, resolution, description_keywords):
    """
    Generate a dictionary containing parameters for TemplateFlow quiery.

    Parameters
    ----------
    atlas_name : str
        Atlas name. Must be a key in ATLAS_METADATA.

    template : str
        TemplateFlow template name.

    resolution : int
        TemplateFlow template resolution.

    description_keywords : dict
        Keys and values to fill in description_pattern.
        For valid keys check relevant ATLAS_METADATA[atlas_name]['description_pattern'].

    Return
    ------
    sklearn.utils.Bunch
        Containing the following fields:

        maps : str
            Path to atlas map.

        labels : pandas.DataFrame
            The corresponding pandas dataframe of the atlas

        type : str
            'dseg' (for NiftiLabelsMasker) or 'probseg' (for NiftiMapsMasker)
    """

    cur_atlas_meta = ATLAS_METADATA[atlas_name].copy()

    img_parameters = generate_templateflow_parameters(cur_atlas_meta, "atlas", resolution, description_keywords)
    label_parameters = generate_templateflow_parameters(cur_atlas_meta, "label", resolution, description_keywords)
    print(img_parameters)
    print(label_parameters)
    img_path = templateflow.api.get(template, raise_empty=True, **img_parameters)
    img_path = str(img_path)
    label_path = templateflow.api.get(template, raise_empty=True, **label_parameters)
    labels = pd.read_csv(label_path, delimiter="\t")
    # labels = (labels['Region'].astype(str) + ". " +
    #           labels['Difumo_names']).values.tolist()
    atlas_type = img_path.split('_')[-1].split('.nii.gz')[0]
    
    return sklearn.utils.Bunch(maps=img_path, labels=labels, type=atlas_type)


def generate_templateflow_parameters(cur_atlas_meta, file_type, resolution, description_keywords):
    """
    Generate a dictionary containing parameters for TemplateFlow quiery.

    Parameters
    ----------
    cur_atlas_meta : dict
        The current TemplateFlow competable atlas metadata.

    file_type : str {'atlas', 'label'}
        Generate parameters to quiry atlas or label.

    resolution : int
        Templateflow template resolution.

    description_keywords : dict
        Keys and values to fill in description_pattern.
        For valid keys check relevant ATLAS_METADATA[atlas_name]['description_pattern'].

    Return
    ------
    dict
        A dictionary containing parameters to pass to a templateflow query.
    """
    description = cur_atlas_meta['description_pattern']
    description = description.format(**description_keywords)

    parameters_ = {key: None for key in cur_atlas_meta[f'{file_type}_parameters']}
    parameters_.update({'atlas': cur_atlas_meta['atlas'], 'extension': ".nii.gz"})
    if file_type == 'label':
        parameters_['extension'] = '.tsv'
    if parameters_.get('resolution', False) is None:
        parameters_['resolution'] = resolution
    if parameters_.get('desc', False) is None:
        parameters_['desc'] = description
    return parameters_


def create_atlas_masker(atlas_name, description_keywords, template='MNI152NLin2009cAsym', resolution=2, nilearn_cache=""):
    """Create masker given metadata.

    Parameters
    ----------
    atlas_name : str
        Atlas name. Must be a key in ATLAS_METADATA.

    description_keywords : dict
        Keys and values to fill in description_pattern.
        For valid keys check relevant ATLAS_METADATA[atlas_name]['description_pattern'].

    template : str
        TemplateFlow template name.

    resolution : str
        TemplateFlow template resolution.
    """
    atlas = fetch_atlas_path(atlas_name,
                             resolution=resolution,
                             template=template,
                             description_keywords=description_keywords)

    if atlas.type == 'dseg':
        masker = nilearn.input_data.NiftiLabelsMasker(atlas.maps, detrend=True)
    elif atlas.type == 'probseg':
        masker = nilearn.input_data.NiftiMapsMasker(atlas.maps, detrend=True)
    if nilearn_cache:
        masker = masker.set_params(memory=nilearn_cache, memory_level=1)
    labels = list(range(1, atlas.labels.shape[0] + 1))
    return masker, labels


def create_timeseries_root_dir(file_entitiles, output_dir):
    """Create root directory for the timeseries file."""
    subject = f"sub-{file_entitiles['subject']}"
    session = f"ses-{file_entitiles['session']}" if file_entitiles.get(
        'session', False) is not None else None
    if session:
        timeseries_root_dir = os.path.join(output_dir, subject, session)
    else:
        timeseries_root_dir = os.path.join(output_dir, subject)
    os.makedirs(timeseries_root_dir, exist_ok=True)

    return timeseries_root_dir

def download_atlases():
    """Download all atlases using ATLAS_METADATA."""

    for atlas_name in ATLAS_METADATA.keys():
        cur_atlas_meta = ATLAS_METADATA[atlas_name].copy()
        if cur_atlas_meta['source'] == "templateflow":
            for template in cur_atlas_meta['templates']:
                for resolution in cur_atlas_meta['resolutions']:
                    for dimension in cur_atlas_meta['dimensions']:
                        description_keywords = {"dimension": dimension}
                        print(f"-- {atlas_name}.{template}.{resolution}mm.{dimension}dim --")
                        atlas = fetch_atlas_path(atlas_name,
                                              resolution=resolution,
                                              template=template,
                                              description_keywords=description_keywords)

def bidsish_timeseries_file_name(file_entitiles, layout, atlas_name, dimension):
    """Create a BIDS-like file name to save extracted timeseries as tsv."""
    pattern = "sub-{subject}[_ses-{session}]_task-{task}[_acq-{acquisition}][_rec-{reconstruction}][_run-{run}][_echo-{echo}]"
    base = layout.build_path(file_entitiles, pattern, validate=False)
    base += f"_atlas-{atlas_name}_network-{dimension}_timeseries.tsv"

    return base.split(os.path.sep)[-1]

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="", epilog="""
    Documentation at https://github.com/ccna-biomarkers/ccna_ts_extraction/tree/masker_with_input_atlas
    """)

    parser.add_argument(
        "-i", "--input_dir", required=False, default=".", help="Input fmripre derivative directory in BIDS, inside \"fmriprep/fmriprep\" (default: \"./\")",
    )

    parser.add_argument(
        "--atlas-path", required=True, help="Input directory path to atlas",
    )

    parser.add_argument(
        "--dataset-name", required=True, help="Dataset name",
    )

    parser.add_argument(
        "--download-only", required=False, action='store_true', help="Download only the atlases, for HPC with firewalled nodes to download templateflow data in the login node.",
    )

    parser.add_argument(
        "-o", "--output-dir", required=False, default=".", help="Output directory (default: \"./\")",
    )

    return parser

if __name__ == '__main__':

    args = get_parser().parse_args()

    data_path = args.input_dir
    atlas_path = args.atlas_path
    dataset_name = args.dataset_name
    download_only = args.download_only
    output_root_dir = args.output_dir

    if download_only:
        print("Download only.")
        download_atlases()
    else:
        template = 'MNI152NLin2009cAsym'
        resolution = "02"

        layout = bids.BIDSLayout(data_path, config=['bids', 'derivatives'])
        subject_list = layout.get(return_type='id', target='subject')

        for atlas_name in ATLAS_METADATA.keys():
            print("-- {} --".format(atlas_name))
            dataset_title = f"dataset-{dataset_name}_atlas-{atlas_name}"
            output_dir = os.path.join(output_root_dir, dataset_title)
            os.makedirs(output_dir, exist_ok=True)
            update_templateflow_path(atlas_name, atlas_path)
            for subject in subject_list:
                print(f"sub-{subject}")
                # TODO: loop through all
                # Note from AB
                # if multiple run, use run 2
                # if multiple session, use ses 1
                fmri = layout.get(return_type='type', subject=subject, space=template,
                                  desc='preproc', suffix='bold', extension='nii.gz')
                brain_mask = layout.get(return_type='type', subject=subject, space=template,
                                        desc='brain', suffix='mask', extension='nii.gz')
                # TODO: check if brain_mask and fmri always come in pairs
                # according to doc, desc-preproc_bold and desc-brain_mask should come in pairs
                for ii in range(len(fmri)):
                    file_entitiles = fmri[ii].entities
                    timeseries_root_dir = create_timeseries_root_dir(
                        file_entitiles, output_dir)
                    for dimension in ATLAS_METADATA[atlas_name]['dimensions']:
                        print(f"\tatlas {atlas_name}\tdim{dimension}")
                        description_keywords = {"dimension": dimension}
                        masker, labels = create_atlas_masker(
                            atlas_name, description_keywords, template=template, resolution=resolution)
                        output_filename = bidsish_timeseries_file_name(
                            file_entitiles, layout, atlas_name, dimension)
                        confounds, sample_mask = nilearn.interfaces.fmriprep.load_confounds(fmri[ii].path,
                                                                                            **LOAD_CONFOUNDS_PARAMS)
                        masker.set_params(mask_img=brain_mask[ii].path)
                        timeseries = masker.fit_transform(
                            fmri[ii].path, confounds=confounds, sample_mask=sample_mask)
                        # Estimating connectomes
                        corr_measure = nilearn.connectome.ConnectivityMeasure(
                            kind="correlation")
                        connectome = corr_measure.fit_transform([timeseries])[0]
                        # Save to file
                        timeseries = pd.DataFrame(timeseries, columns=labels)
                        timeseries.to_csv(os.path.join(timeseries_root_dir, output_filename), sep='\t', index=False)
                        connectome = pd.DataFrame(
                            connectome, columns=labels, index=labels)
                        connectome.to_csv(
                            os.path.join(timeseries_root_dir, output_filename.replace("timeseries", "connectome")), sep='\t')

            # tar the dataset
            tar_path = os.path.join(output_root_dir, f"{dataset_title}.tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(output_dir, arcname=os.path.dirname(output_dir))


import pytest

def test_templateflow(tmpdir):

    template = 'MNI152NLin2009cAsym'
    resolution = 2
    templateflow.conf.TF_HOME = tmpdir
    templateflow.conf.update(local=True)
    # schaefer test
    atlas_name = 'schaefer7'
    dimension = 100
    print(f"\tatlas {atlas_name}\tdim{dimension}")
    description_keywords = {"dimension": dimension, "resolution": resolution}
    atlas = fetch_atlas_path(atlas_name,
                                resolution=resolution,
                                template=template,
                                description_keywords=description_keywords)
    assert atlas.maps.split('/')[-1] == 'tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-100Parcels7Networks_dseg.nii.gz'
    assert atlas.type == 'dseg'
    masker, labels = create_atlas_masker(atlas_name, description_keywords)
    assert type(masker) == nilearn.input_data.NiftiLabelsMasker
    assert len(labels) == 100
    # difumo test
    ATLAS_METADATA['difumo'] = {
        'source': "templateflow",
        'templates' : ['MNI152NLin2009cAsym', 'MNI152NLin6Asym'],
        'resolutions': [2, 3],
        'atlas': 'DiFuMo',
        'description_pattern': "{dimension}dimensions",
        'dimensions': [64, 128, 256, 512, 1024],
        'atlas_parameters': ['resolution', 'desc'],
        'label_parameters': ['resolution','desc'],
        },
    atlas_name = 'difumo'
    dimension = 64
    print(f"\tatlas {atlas_name}\tdim{dimension}")
    description_keywords = {"dimension": dimension, "resolution": resolution}
    atlas = fetch_atlas_path(atlas_name,
                                resolution=resolution,
                                template=template,
                                description_keywords=description_keywords)
    assert atlas.maps.split('/')[-1] == 'tpl-MNI152NLin6Asym_res-02_atlas-DiFuMo_desc-64dimensions_probseg.nii.gz'
    assert atlas.type == 'probseg'
    masker, labels = create_atlas_masker(atlas_name, description_keywords)
    assert type(masker) == nilearn.input_data.NiftiLabelsMasker
    assert len(labels) == 64

