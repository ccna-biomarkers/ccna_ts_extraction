"""
CCNA timeseries extraction and confound removal.

Save the output to a tar file.
"""
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
    'difumo': {'type': "dynamic",
               'dimensions': [64, 128, 256, 512, 1024],
               'resolutions': [2, 3],
               'label_idx': 1,
               'fetcher': "nilearn.datasets.fetch_atlas_difumo(dimension={dimension}, resolution_mm={resolution}, data_dir=\"{atlas_path}\")"},
    'segmented_difumo': {'type': "dynamic",
                         'dimensions': [64, 128, 256, 512, 1024],
                         'resolutions': [2, 3],
                         'fetcher': "segmented_difumo_fetcher(dimension={dimension}, resolution_mm={resolution}, atlas_path=\"{atlas_path}\")"},
}

#TODO: mask data using subject mask
#TODO: wait you PR before making a standalone tool (templateflow downloading, confound parameters loading)
#TODO: QC timeseries https://github.com/SIMEXP/mapsmasker_benchmark/blob/main/mapsmasker_benchmark/main.py 
def segmented_difumo_fetcher(atlas_path, dimension=64, resolution_mm=3):

    templateflow.conf.TF_HOME = pathlib.Path(atlas_path)
    templateflow.conf.init_layout()
    templateflow.conf.update(local=True)

    img_path = str(templateflow.api.get("MNI152NLin2009cAsym", atlas="DiFuMo",
                                        desc=f"{dimension}dimensionsSegmented", resolution=f"0{resolution_mm}", extension=".nii.gz"))
    label_path = str(templateflow.api.get("MNI152NLin2009cAsym", atlas="DiFuMo",
                                          desc=f"{dimension}dimensionsSegmented", resolution=f"0{resolution_mm}", extension=".tsv"))
    labels = pd.read_csv(label_path, delimiter="\t")
    labels = (labels['Region'].astype(str) + ". " +
              labels['Difumo_names']).values.tolist()

    return sklearn.utils.Bunch(maps=img_path, labels=labels)


def create_atlas_masker(atlas_name, atlas_path, dimension, resolution, nilearn_cache=""):
    """Create masker of all dimensions and resolutions given metadata."""
    if atlas_name not in ATLAS_METADATA.keys():
        raise ValueError("{} not defined!".format(atlas_name))
    curr_atlas = ATLAS_METADATA[atlas_name]
    curr_atlas['name'] = atlas_name

    atlas = eval(curr_atlas['fetcher'].format(
        atlas_path=atlas_path, dimension=dimension, resolution=resolution))
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
        timeseries_root_dir = os.path.join(output_dir, subject, session)
    else:
        timeseries_root_dir = os.path.join(output_dir, subject)
    os.makedirs(timeseries_root_dir, exist_ok=True)

    return timeseries_root_dir


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
        "-o", "--output-dir", required=False, default=".", help="Output directory (default: \"./\")",
    )

    return parser

if __name__ == '__main__':

    args = get_parser().parse_args()

    data_path = args.input_dir
    atlas_path = args.atlas_path
    dataset_name = args.dataset_name
    output_root_dir = args.output_dir

    layout = bids.BIDSLayout(data_path, config=['bids', 'derivatives'])
    subject_list = layout.get(return_type='id', target='subject')

    for atlas_name in ATLAS_METADATA.keys():
        print("-- {} --".format(atlas_name))
        dataset_title = f"dataset-{dataset_name}_atlas-{atlas_name}"
        output_dir = os.path.join(output_root_dir, dataset_title)
        os.makedirs(output_dir, exist_ok=True)
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
                        print(f"\t dim{dimension} - res0{resolution}")
                        masker, labels = create_atlas_masker(
                            atlas_name, atlas_path, dimension, resolution)
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
                        timeseries.to_csv(os.path.join(timeseries_root_dir, output_filename), sep='\t', index=False)
                        connectome = pd.DataFrame(
                            connectome, columns=labels, index=labels)
                        connectome.to_csv(
                            os.path.join(timeseries_root_dir, output_filename.replace("timeseries", "connectome")), sep='\t')

        # tar the dataset
        tar_path = os.path.join(output_root_dir, f"{dataset_title}.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(output_dir, arcname=os.path.dirname(output_dir))
