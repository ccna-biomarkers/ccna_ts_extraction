# ccna time series extraction

Script for CCNA time series extraction and functional connectivity metric calculation.

This script uses the latest development version of `nilearn` (date: 13/01/2021)

## Procedure

Difumo 64 dimesnion atlas was selected to extract time seires and connectome construction. 
Since the parcellation could be networks, prior to time series extraction, the networks were broken down into isolated parcels with `nilearn.regions.RegionExtractor`.
The region extraction resulted in 101 disconnected regions for time series extraction. 
All time series were detrended and standardized to z-score.
Each region was labeled with the new label and the original parcel it belonged to. 
The new labels were saved as a nifiti file for references.
Functional connectivity was calculated through Pearson's correlation with nilearn function `nilearn.connectome.ConnectivityMeasure`.
