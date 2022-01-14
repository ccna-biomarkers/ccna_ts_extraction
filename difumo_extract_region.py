"""Demo for extracting regions from probability atlas with networks."""

import numpy as np
from nilearn import plotting

from nilearn.datasets import fetch_atlas_difumo
from nilearn.image import index_img
from nilearn.plotting import find_xyz_cut_coords
from nilearn.regions import RegionExtractor


atlas = fetch_atlas_difumo()


extractor = RegionExtractor(atlas.maps, threshold=0.5,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True)
extractor.fit()
regions_extracted_img = extractor.regions_img_

regions_index = extractor.index_
n_regions_extracted = regions_extracted_img.shape[-1]
plotting.plot_prob_atlas(regions_extracted_img,
                         view_type='contours',
                         title=f"{n_regions_extracted} from difumo 64")

i = 63  # let's see how one network was broken down
regions_indices_network = np.where(np.array(extractor.index_) == i)
for index in regions_indices_network[0]:
    cur_img = index_img(extractor.regions_img_, index)
    coords = find_xyz_cut_coords(cur_img)
    plotting.plot_stat_map(cur_img, display_mode='z', cut_coords=coords[2:3],
                            title=f"Blob {index} from network {i + 1}", colorbar=False)

plotting.show()
