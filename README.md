# rc-csp-mdd

This repository contains FMRI data and scripts for training and testing best performing models from paper ***"Reservoir Computing Combined with Common Spatial Patterns for Representation and Classification of fMRI-based Data of Patients with Major Depressive Disorder"***.

## Folders
- [`filtered_data`](filtered_data): fMRI BOLD signals used for training and testing.
- [`groups`](groups): contains a file defining anatomical segregation of input signals.

## Scripts
- [`Approach_I`](Approach_I.ipynb): Classical CSP-based approach.
- [`Approach_II`](Approach_II.ipynb): RC combined with CSP.
- [`Approach_III`](Approach_III.ipynb): Combined approach involving CSP feature extraction and further use of the RC with CSP.

[rcnet.py](rcnet.py) contains reservoir computing layer implementation.